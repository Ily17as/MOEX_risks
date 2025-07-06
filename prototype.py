# prototype.py — RiskLens Extended with ad‑hoc portfolio API
# Полная поддержка: акции, облигации, фьючерсы (опционы — пока заглушка)
# Методы: исторический и параметрический VaR/CVaR + портфельный расчёт

import sys
import asyncio
import datetime as dt
import sqlite3
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple, Literal

import httpx
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ─────────────────────────────── КОНФИГ ░░░░░░░░░░░░░░░░░░░░ #
DB_PATH = "risklens.db"
DEFAULT_CONFIDENCE = 0.99           # по умолчанию 99‑процентный уровень доверия
LOOKBACK_DAYS = 365                 # горизонт ретроспективы
JOB_HOUR, JOB_MINUTE = 9, 5         # расписание фонового джоба (МСК)

# Портфель по умолчанию — пример для фонового джоба
SCHEDULED_PORTFOLIO: List[Dict] = [
    {"ticker": "SBER",          "type": "stock",  "position": 1_000_000},  # ₽
    {"ticker": "SU26240RMFS6",  "type": "bond",   "position":   500_000},  # ОФЗ‑240
    {"ticker": "SiU5",          "type": "future", "position":         2},  # 2 контракта
]

# ─────────────────────────────── ЛОГ ░░░░░░░░░░░░░░░░░░░░░░░ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("risklens")

# ─────────────────────────────── БАЗА ░░░░░░░░░░░░░░░░░░░░░ #

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute(
    """CREATE TABLE IF NOT EXISTS results (
         ts          TEXT,
         ticker      TEXT,
         type        TEXT,
         method      TEXT,
         position    REAL,
         var         REAL,
         cvar        REAL,
         confidence  REAL,
         PRIMARY KEY (ts, ticker, method)
     )"""
)
db.execute(
    """CREATE TABLE IF NOT EXISTS portfolio (
         ts          TEXT,
         method      TEXT,
         var         REAL,
         cvar        REAL,
         confidence  REAL,
         PRIMARY KEY (ts, method)
     )"""
)

# ──────────────────────────────── Pydantic ░░░░░░░░░░░░░░░░ #
class Position(BaseModel):
    ticker: str = Field(..., description="Тикер инструмента, как на MOEX")
    type:   Literal["stock", "bond", "future", "option"]
    position: float = Field(..., description="Размер позиции (₽ для spot/bond, лоты/контракты для сроч.)")

class PortfolioRequest(BaseModel):
    portfolio: List[Position]
    method: Literal["historical", "parametric"] = "historical"
    confidence: float = Field(DEFAULT_CONFIDENCE, gt=0, lt=1)

# ──────────────────────── MOEX ISS ENDPOINTS ░░░░░░░░░░░░░░░ #
BASE_ISS = "https://iss.moex.com/iss"

def endpoint_for(sec_type: str, ticker: str) -> str:
    if sec_type == "stock":
        return f"{BASE_ISS}/engines/stock/markets/shares/securities/{ticker}/candles.json"
    if sec_type == "bond":
        return f"{BASE_ISS}/engines/stock/markets/bonds/securities/{ticker}/candles.json"
    if sec_type == "future":
        return f"{BASE_ISS}/engines/futures/markets/forts/securities/{ticker}/candles.json"
    if sec_type == "option":
        raise NotImplementedError("MOEX ISS не отдает свечи для опционов")
    raise ValueError(sec_type)


async def fetch_candles(ticker: str, sec_type: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    url = f"{endpoint_for(sec_type, ticker)}?from={start}&till={end}&interval=24&iss.meta=off"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    df = pd.DataFrame(data["candles"]["data"], columns=data["candles"]["columns"])
    if df.empty:
        return df
    date_col = "begin" if "begin" in df.columns else "date"
    df["ts"] = pd.to_datetime(df[date_col]).dt.date
    return df[["ts", "close"]].rename(columns={"close": "price"})

# ──────────────────────── VaR / CVaR HELPERS ░░░░░░░░░░░░░░░ #
try:
    from scipy.stats import norm
except ImportError:
    norm = None
    log.warning("SciPy отсутствует — параметрический VaR через эмпирический z‑score")


def _historical_var_cvar(series: pd.Series, level: float) -> Tuple[float, float]:
    var_q = np.quantile(series, 1 - level)
    cvar_q = series[series <= var_q].mean()
    return float(var_q), float(cvar_q)


def _parametric_var_cvar(series: pd.Series, level: float) -> Tuple[float, float]:
    mu, sigma = series.mean(), series.std(ddof=1)
    if norm:
        z = norm.ppf(level)
        pdf = norm.pdf(z)
    else:
        z = abs(np.quantile(np.random.normal(0, 1, 200_000), 1 - level))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    var_q = mu - z * sigma
    cvar_q = mu - sigma * pdf / (1 - level)
    return float(var_q), float(cvar_q)


def _pct_to_value(pct_loss: float, position: float) -> float:
    """Конвертируем %-убыток в абсолютный (знак «+» = потери)."""
    return -pct_loss * position

# ──────────────────────── CORE CALCULATION ░░░░░░░░░░░░░░░░ #

async def calc_instrument(inst: Position, level: float, method: str) -> Tuple[float, float, pd.Series]:
    today = dt.date.today()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    df = await fetch_candles(inst.ticker, inst.type, start, today)

    if df.empty or len(df) < 50:
        raise RuntimeError(f"{inst.ticker}: недостаточно данных")

    returns = df["price"].pct_change().dropna()
    if method == "historical":
        var_pct, cvar_pct = _historical_var_cvar(returns, level)
    else:
        var_pct, cvar_pct = _parametric_var_cvar(returns, level)

    abs_var = _pct_to_value(var_pct, inst.position)
    abs_cvar = _pct_to_value(cvar_pct, inst.position)
    return abs_var, abs_cvar, returns


async def calc_portfolio_once(req: PortfolioRequest) -> Dict:
    level = req.confidence
    method = req.method

    per_instr = []
    returns_map: Dict[str, pd.Series] = {}
    weights: List[float] = []

    for inst in req.portfolio:
        v, cv, series = await calc_instrument(inst, level, method)
        per_instr.append({"ticker": inst.ticker, "type": inst.type, "position": inst.position, "VaR": v, "CVaR": cv})
        returns_map[inst.ticker] = series
        weights.append(inst.position)

    # портфельная агрегация
    combined = pd.concat(returns_map.values(), axis=1, keys=returns_map.keys()).dropna()
    if combined.empty:
        raise RuntimeError("Не удалось выровнять даты для портфельного расчёта")

    w = np.array(weights) / np.sum(weights)
    port_ret = (combined * w).sum(axis=1)

    if method == "historical":
        var_pct, cvar_pct = _historical_var_cvar(port_ret, level)
    else:
        var_pct, cvar_pct = _parametric_var_cvar(port_ret, level)

    port_val = np.sum(weights)
    port_var = _pct_to_value(var_pct, port_val)
    port_cvar = _pct_to_value(cvar_pct, port_val)

    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "confidence": level,
        "method": method,
        "portfolio_value": port_val,
        "VaR": port_var,
        "CVaR": port_cvar,
        "details": per_instr,
    }

# ──────────────────────── SCHEDULED DAILY JOB ░░░░░░░░░░░░░░ #

async def daily_job() -> None:
    today = dt.date.today()
    log.info("⏳ daily_job started")

    returns_map = {}
    weights = []

    for meta in SCHEDULED_PORTFOLIO:
        try:
            inst = Position(**meta)
            for m in ["historical", "parametric"]:
                var, cvar, series = await calc_instrument(inst, DEFAULT_CONFIDENCE, m)
                db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)",
                    (
                        today.isoformat(),
                        inst.ticker,
                        inst.type,
                        m,
                        inst.position,
                        var,
                        cvar,
                        DEFAULT_CONFIDENCE,
                    ),
                )
                returns_map.setdefault(m, {})[inst.ticker] = series
            weights.append(inst.position)
        except Exception as exc:
            log.error(exc)

    db.commit()

    # портфельная запись
    for m in ["historical", "parametric"]:
        combined = pd.concat(returns_map.get(m, {}).values(), axis=1).dropna()
        if combined.empty:
            continue
        w = np.array(weights) / np.sum(weights)
        port_ret = (combined * w).sum(axis=1)
        if m == "historical":
            var_pct, cvar_pct = _historical_var_cvar(port_ret, DEFAULT_CONFIDENCE)
        else:
            var_pct, cvar_pct = _parametric_var_cvar(port_ret, DEFAULT_CONFIDENCE)
        port_val = np.sum(weights)
        db.execute(
            "INSERT OR REPLACE INTO portfolio VALUES (?,?,?,?,?)",
            (
                today.isoformat(),
                m,
                _pct_to_value(var_pct, port_val),
                _pct_to_value(cvar_pct, port_val),
                DEFAULT_CONFIDENCE,
            ),
        )
    db.commit()
    log.info("✅ daily_job finished")

# ─────────────────────────────── FastAPI ░░░░░░░░░░░░░░░░░░ #

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    scheduler.add_job(daily_job, "cron", hour=JOB_HOUR, minute=JOB_MINUTE)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)
    db.close()

app = FastAPI(
    title="RiskLens Extended",
    description="VaR/CVaR для пользовательского портфеля инструментов MOEX",
    lifespan=lifespan,
)

# ─────────────────────────────── ENDPOINTS ░░░░░░░░░░░░░░░░ #

@app.post("/calc")
async def calc_endpoint(req: PortfolioRequest):
    """Ad‑hoc расчёт VaR/CVaR по переданному портфелю без сохранения в БД."""
    try:
        result = await calc_portfolio_once(req)
        return result
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.get("/var/{ticker}")
def read_var(
    ticker: str,
    method: Literal["historical", "parametric"] = Query("historical"),
):
    row = db.execute(
        "SELECT ts, var, cvar, confidence FROM results WHERE ticker=? AND method=? ORDER BY ts DESC LIMIT 1",
        (ticker.upper(), method),
    ).fetchone()
    if not row:
        raise HTTPException(404, "no data yet")
    ts, var_val, cvar_val, conf = row
    return {
        "ticker": ticker.upper(),
        "method": method,
        "timestamp": ts,
        "VaR": var_val,
        "CVaR": cvar_val,
        "confidence": conf,
    }


@app.get("/portfolio")
def read_portfolio(method: Literal["historical", "parametric"] = Query("historical")):
    row = db.execute(
        "SELECT ts, var, cvar, confidence FROM portfolio WHERE method=? ORDER BY ts DESC LIMIT 1",
        (method,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "no data yet")
    ts, var_val, cvar_val, conf = row
    return {
        "portfolio": "scheduled_default",
        "method": method,
        "timestamp": ts,
        "VaR": var_val,
        "CVaR": cvar_val,
        "confidence": conf,
    }


@app.post("/run-now")
async def run_now():
    await daily_job()
    return {"status": "ok"}

# ──────────────────────────── CLI‑запуск ░░░░░░░░░░░░░░░░░░ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prototype:app", host="127.0.0.1", port=8000, reload=False)
