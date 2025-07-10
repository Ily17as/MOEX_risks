import polars as pl, pathlib as p

def ingest(csv_path: str, symbol: str) -> None:
    df = (
        pl.read_csv(
            csv_path,
            has_header=False,
            new_columns=list(TICK_COLS),
            try_parse_dates=True,
        )
        .with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
        .cast(TICK_COLS)
        .sort("ts")
    )
    for date, sub in df.partition_by("ts", as_dict=False, stable=True):
        out = (
            p.Path("data")
            / "ticks"
            / f"symbol={symbol}"
            / f"date={date.strftime('%Y-%m-%d')}"
        )
        out.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(out / "part-0.parquet", compression="zstd")