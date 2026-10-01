-- project4 headline fundamentals for the fundamental sanity check.
--
-- Pairs trading is a statistical claim ("this spread mean-reverts"); the economics
-- are what make the claim durable. Two names with the same demand pool or the same
-- input costs are a candidate pair; two names whose earnings have stopped tracking
-- are a pair about to break.
--
-- This pulls quarterly headline income-statement line items:
--   code 20 = Net Revenue
--   code 70 = Net Profit After Tax
-- See financial.item for the full dictionary. quarter = 0 is the ANNUAL figure;
-- 1..4 are the quarterlies (industry convention in this feed: q4 is a quarter).
--
-- Used by scripts/pairs_screen.py --fundamentals to attach a profit-linkage
-- diagnostic to every pair. Optional: the price screen runs without it.
SELECT
    i.tickersymbol              AS ticker,
    i.year,
    i.quarter,
    i.code,
    i.value::double precision   AS value
FROM financial.info i
JOIN quote.ticker t ON t.tickersymbol = i.tickersymbol
WHERE t.instrumenttype = 'stock'
  AND i.code IN ('20', '70')
  AND i.year >= 2010
ORDER BY i.tickersymbol, i.year, i.quarter, i.code;
