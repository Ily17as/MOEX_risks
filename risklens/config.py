from pathlib import Path

# Path to DuckDB database containing market data
DUCKDB_PATH = Path("risk.db")

# Path to SQLite database storing VaR/CVaR results
SQLITE_DB_PATH = Path("risklens.db")
