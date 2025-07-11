import polars as pl, sys, pathlib as p

symbol, date, freq = sys.argv[1:]  # SBER 2025-07-09 1m
src = f"data/ticks/symbol={symbol}/date={date}/*.parquet"
out_dir = p.Path("../data") / "bars" / f"freq={freq}" / f"symbol={symbol}" / f"date={date}"
out_dir.mkdir(parents=True, exist_ok=True)

bars = (
    pl.scan_parquet(src)
      .group_by_dynamic("ts", every=freq, closed="left")
      .agg(
          pl.col("price").first().alias("open"),
          pl.col("price").max().alias("high"),
          pl.col("price").min().alias("low"),
          pl.col("price").last().alias("close"),
          pl.col("qty").sum().alias("volume"),
      )
      .sort("ts")
      .collect(streaming=True)
)
bars.write_parquet(out_dir / "part-0.parquet", compression="zstd")