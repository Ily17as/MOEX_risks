import sys, logging
from pathlib import Path
import polars as pl
from risklens.schema import TICK_COLS  # убедись, что schema.py возвращает str → pl.DataType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --- 1. Приводим строковые типы к реальным Polars DType -------------------------
_STR2DTYPE = {
    "datetime[μs, UTC]": pl.Datetime("us", "UTC"),
    "float64":           pl.Float64,
    "uint64":            pl.UInt64,
    "uint32":            pl.UInt32,
    "uint8":             pl.UInt8,
    # добавь сюда остальные, если появятся
}

TYPES = {col: _STR2DTYPE.get(dt, dt) for col, dt in TICK_COLS.items()}

# --- 2. Ингест ------------------------------------------------------------------
def ingest(csv_path: str, symbol: str) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logging.error("CSV %s not found", csv_path)
        sys.exit(1)

    logging.info("Reading %s", csv_path)
    try:
        df = (
            pl.read_csv(csv_path, has_header=False, new_columns=list(TYPES))
            .with_columns(
                # ISO-8601 + μs → «%.f» в новой версии Polars
                pl.col("ts").str.strptime(
                    pl.Datetime(time_unit="us", time_zone="UTC"),
                    format="%Y-%m-%dT%H:%M:%S%.fZ",
                    strict=False,
                )
            )
            .with_columns(
                # остальные колонки к нужным типам
                [pl.col(c).cast(t) for c, t in TYPES.items() if c != "ts"]
            )
        )
    except Exception as exc:
        logging.exception("Failed to parse CSV: %s", exc)
        sys.exit(1)

    if df.height == 0:
        logging.error("CSV parsed but empty — abort")
        sys.exit(1)

    logging.info("Rows read: %d", df.height)

    # дата-партиция
    df = df.with_columns(pl.col("ts").dt.truncate("1d").alias("date"))

    root = Path("../data") / "ticks" / f"symbol={symbol}"
    total = 0
    for date, sub in df.partition_by("date", as_dict=False):
        out_dir = root / f"date={date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part-0.parquet"
        sub.drop("date").write_parquet(out_file, compression="zstd")
        logging.info("Written %s (%d rows)", out_file, sub.height)
        total += sub.height

    logging.info("DONE — %d ticks written for %s", total, symbol)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 00_ingest_ticks.py <csv_path> <symbol>")
        sys.exit(1)
    ingest(sys.argv[1], sys.argv[2])