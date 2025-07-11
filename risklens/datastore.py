import duckdb, pathlib as p
from .config import DUCKDB_PATH

class DataStore:
    def __init__(self, db_path=DUCKDB_PATH):
        self.con = duckdb.connect(db_path,
                                  config={"external_file_cache_size": "8GB"})

    def ticks(self, symbol: str, date: str):
        return self.con.execute(
            "SELECT * FROM ticks WHERE symbol = ? AND date = ?", [symbol, date]
        ).df()

    def bars(self, symbol: str, start: str, end: str):
        return self.con.execute("""
            FROM bars_1m
            WHERE symbol = ?
              AND ts BETWEEN ? AND ?
            ORDER BY ts
        """, [symbol, start, end]).df()

    def factor(self, symbol: str, col: str, start: str, end: str):
        return self.con.execute(f"""
            SELECT ts, {col}
            FROM factors WHERE symbol = ?
              AND ts BETWEEN ? AND ?
        """, [symbol, start, end]).df()

