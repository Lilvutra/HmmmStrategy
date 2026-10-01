#!/usr/bin/env python3
"""Download data from a SQL database into local parquet (step 2: data_collection).

Connection details come from environment variables (injected as plutus
``secrets`` at run time) — nothing is hard-coded. Results are written as parquet
so the downstream processing / backtest steps can read them with polars, matching
the project1 convention.

Env vars (manifest secret names, with common aliases):
    HOST      / DB_HOST
    PORT      / DB_PORT      (default 5432)
    DATABASE  / DB_NAME
    USER_DB   / DB_USER
    PASSWORD  / DB_PASSWORD
    DB_SCHEME                (default "postgresql"; e.g. postgresql | mysql | clickhouse)

Usage:
    python scripts/download_data.py --query-file sql/query.sql -o data/raw/dataset.parquet
    python scripts/download_data.py --query "SELECT * FROM my_table" -o data/raw/out.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl


def _clean_sql(text: str) -> str:
    """Strip line comments + a trailing semicolon.

    connectorx sends the text as a single statement and rejects a trailing ';'
    (and the comments after it). We drop ``-- ...`` line comments and the final
    ';' so a normal, well-commented .sql file works as-is. (Queries here contain
    no string literals with ``--``, so naive comment stripping is safe.)
    """
    lines = []
    for line in text.splitlines():
        idx = line.find("--")
        if idx != -1:
            line = line[:idx]
        lines.append(line)
    return "\n".join(lines).strip().rstrip(";").strip()


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def build_uri() -> str:
    """Build a SQLAlchemy-style connection URI from env vars."""
    scheme = _env("DB_SCHEME", default="postgresql")
    host = _env("HOST", "DB_HOST")
    port = _env("PORT", "DB_PORT", default="5432")
    database = _env("DATABASE", "DB_NAME")
    user = _env("USER_DB", "DB_USER")
    password = _env("PASSWORD", "DB_PASSWORD")

    missing = [
        label
        for label, val in (
            ("HOST", host),
            ("DATABASE", database),
            ("USER_DB", user),
            ("PASSWORD", password),
        )
        if not val
    ]
    if missing:
        sys.exit(
            "[download] missing DB connection env var(s): "
            + ", ".join(missing)
            + "\n  Set them (or their DB_* aliases) before running — see this file's docstring."
        )
    return f"{scheme}://{user}:{password}@{host}:{port}/{database}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SQL query results to parquet.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query", help="Inline SQL query string.")
    source.add_argument("--query-file", type=Path, help="Path to a .sql file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output parquet path.")
    args = parser.parse_args()

    query = _clean_sql(args.query if args.query else args.query_file.read_text())
    uri = build_uri()

    print(f"[download] running query -> {args.output}", flush=True)
    # read_database_uri needs a backend engine: `pip install connectorx` (default)
    # or `adbc-driver-*`. See requirements.txt.
    frame = pl.read_database_uri(query=query, uri=uri)
    print(f"[download] fetched {frame.height} rows x {frame.width} cols", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"[download] wrote {args.output} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
