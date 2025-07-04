# prototype.py
import sys                     # ← добавили
import asyncio
import datetime as dt
import sqlite3
from contextlib import asynccontextmanager
from typing import List

import httpx
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException

# ─────────────────────────────── Конфигурация ──────────────────────────────── #
TICKERS: List[str] = ["SBER"]
VAR_LEVEL: float = 0.99
JOB_HOUR, JOB_MINUTE = 9, 5
DB_PATH = "risklens.db"

# ────────────────────────────── База данных ───────────────────────────────── #
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute(
    """CREATE TABLE IF NOT EXISTS results (
           ts      TEXT,
           ticker  TEXT,
           var     REAL,
           cvar    REAL,
           PRIMARY KEY (ts, ticker)
       )"""
)
db.commit()

# ────────────────────────────── Логика расчёта ────────────────────────────── #
MOEX_BASE = "https://iss.moex.com/iss"


async def fetch_candles(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    url = (
        f"{MOEX_BASE}/engines/stock/markets/shares/"
        f"securities/{ticker}/candles.json?"
        f"from={start}&till={end}&interval=24&iss.meta=off"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()          # ← теперь 4xx/5xx не скрываются
        data = resp.json()
    return pd.DataFrame(data["candles"]["data"], columns=data["candles"]["columns"])


def calc_var_cvar(df: pd.DataFrame, level: float = VAR_LEVEL) -> tuple[float, float]:
    returns = df["close"].pct_change().dropna()
    var_pct = np.quantile(returns, 1 - level)
    cvar_pct = returns[returns <= var_pct].mean()
    last_price = df["close"].iloc[-1]
    return float(var_pct * last_price), float(cvar_pct * last_price)


async def daily_job() -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=365)
    for ticker in TICKERS:
        try:
            candles = await fetch_candles(ticker, start, today)
            if candles.empty:
                print(f"[WARN] {ticker}: 0 rows (board/market не совпали?)")
                continue
            var_val, cvar_val = calc_var_cvar(candles)
            db.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                (today.isoformat(), ticker, var_val, cvar_val),
            )
            db.commit()
        except Exception as exc:
            # минимальный лог
            print(f"[ERROR] {ticker}: {exc}")


# ─────────────────────────── Lifespan FastAPI ─────────────────────────────── #
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Windows: корректная политика event-loop
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    scheduler.add_job(daily_job, "cron", hour=JOB_HOUR, minute=JOB_MINUTE)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)
    db.close()


app = FastAPI(
    title="RiskLens Local",
    description="VaR/CVaR по тикерам с данными MOEX ISS API",
    lifespan=lifespan,
)

# ────────────────────────────── HTTP-эндпоинты ────────────────────────────── #
@app.get("/var/{ticker}")
def read_var(ticker: str):
    row = db.execute(
        "SELECT ts, var, cvar FROM results WHERE ticker=? ORDER BY ts DESC LIMIT 1",
        (ticker.upper(),),
    ).fetchone()
    if not row:
        raise HTTPException(404, "no data yet")
    ts, var_val, cvar_val = row
    return {
        "ticker": ticker.upper(),
        "timestamp": ts,
        "VaR": var_val,
        "CVaR": cvar_val,
        "confidence": VAR_LEVEL,
    }


@app.post("/run-now")
async def run_now():
    await daily_job()
    return {"status": "ok"}


# ───────────────────────────── Точка входа ─────────────────────────────────── #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("prototype:app", host="127.0.0.1", port=8000)
