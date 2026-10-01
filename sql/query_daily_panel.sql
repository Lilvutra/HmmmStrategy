-- project4 DAILY split-adjusted close + volume panel for a broad VN stock universe.
--
-- Why adjusted prices: pairs trading measures the SPREAD between two names. A cash
-- dividend or a split changes the raw `quote.close` level without changing anything
-- economic, which would open a fake spread. `quote.adjclose` is back-adjusted for
-- those corporate actions, so the spread is a genuine relative-value signal.
--
-- The server-side pre-screen is deliberately COARSE: it only removes names that
-- cannot plausibly form a tradeable pair (too short a history, near-zero liquidity),
-- so the download stays a few million daily rows. The real screening happens in
-- scripts/pairs_screen.py.
--
-- Universe: common stocks on HSX / HNX / UPCOM (excludes warrants, funds, indices,
-- futures and the TWOTC third market). History from 2015 so both the formation and
-- the out-of-sample windows have enough length.
--
-- Units: price is THOUSANDS of VND, so quantity * price is thousands-VND turnover.
-- The 300000 threshold is ~300m VND of average daily turnover.
WITH stocks AS (
    SELECT tickersymbol
    FROM quote.ticker
    WHERE instrumenttype = 'stock'
      AND exchangeid IN ('HSX', 'HNX', 'UPCOM')
),
activity AS (
    -- Coarse liquidity / history pre-screen, computed on the daily close and volume.
    SELECT dv.tickersymbol,
           count(*)                                             AS n_days,
           avg(dv.quantity::double precision * c.price)         AS avg_turnover_kvnd
    FROM quote.dailyvolume dv
    JOIN stocks s ON s.tickersymbol = dv.tickersymbol
    JOIN quote.close c
      ON c.tickersymbol = dv.tickersymbol
     AND c.datetime     = dv.datetime
    WHERE dv.datetime >= date '2015-01-01'
      AND dv.quantity > 0
      AND c.price     > 0
    GROUP BY dv.tickersymbol
    HAVING count(*) >= 1000                                    -- ~4 years of sessions
       AND avg(dv.quantity::double precision * c.price) >= 300000
)
SELECT
    ac.datetime                  AS date,
    ac.tickersymbol              AS ticker,
    ac.price::double precision   AS adj_close,
    dv.quantity                  AS volume
FROM quote.adjclose ac
JOIN activity a ON a.tickersymbol = ac.tickersymbol
JOIN stocks   s ON s.tickersymbol = ac.tickersymbol
LEFT JOIN quote.dailyvolume dv
       ON dv.tickersymbol = ac.tickersymbol
      AND dv.datetime     = ac.datetime
WHERE ac.datetime >= date '2015-01-01'
  AND ac.price > 0
ORDER BY ac.tickersymbol, ac.datetime;
