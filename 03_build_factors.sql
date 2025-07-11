CREATE OR REPLACE TABLE factors AS
SELECT
    ts,
    symbol,
    close,
    (close / LAG(close, 5) OVER w) - 1 AS mom_5,
    (AVG(close) OVER w20)             AS sma20
FROM bars_1m
WINDOW
  w    AS (PARTITION BY symbol ORDER BY ts),
  w20  AS (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 19 PRECEDING AND CURRENT ROW);
