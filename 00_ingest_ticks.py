from risklens.schema import TICK_COLS
import polars as pl
from pathlib import Path
import datetime as dt

def ingest(csv_path: str, symbol: str) -> None:
    df = (
        pl.read_csv(csv_path, has_header=False, new_columns=list(TICK_COLS))
        .with_columns(pl.col("ts").str.strptime(pl.Datetime, fmt="%Y-%m-%dT%H:%M:%S.%fZ"))
        .cast(TICK_COLS)
    )
    df = df.with_columns(pl.col("ts").dt.truncate("1d").alias("date"))
    for date, sub in df.partition_by("date", as_dict=False):
        out = Path("data") / "ticks" / f"symbol={symbol}" / f"date={date}"
        out.mkdir(parents=True, exist_ok=True)
        sub.drop("date").write_parquet(out / "part-0.parquet", compression="zstd")
