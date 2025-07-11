import duckdb, pathlib as p, glob, os
from risklens.config import DUCKDB_PATH

db = duckdb.connect(DUCKDB_PATH, config={"external_file_cache_size": "8GB"})

TICKS_ROOT = "data/ticks"
BARS_ROOT  = "data/bars"

db.execute("""
CREATE OR REPLACE VIEW ticks AS
SELECT * FROM read_parquet($ticks)
""", {"ticks": f"{TICKS_ROOT}/symbol=*/date=*/part-*.parquet"})

db.execute("""
CREATE OR REPLACE VIEW bars_1m AS
SELECT * FROM read_parquet($bars)
""", {"bars": f"{BARS_ROOT}/freq=1m/symbol=*/date=*/part-*.parquet"})
