import sys
import asyncio
import datetime as dt
import sqlite3
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple, Literal, Optional

import httpx
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from greece import delta as opt_delta, gamma as opt_gamma, vega as opt_vega, theta as opt_theta, rho as opt_rho

DB_PATH = "risklens.db"
DEFAULT_CONFIDENCE = 0.99
LOOKBACK_DAYS = 365
JOB_HOUR, JOB_MINUTE = 9, 5
RISK_FREE_RATE = 0.08

SCHEDULED_PORTFOLIO: List[Dict] = [
    {"ticker": "SBER",         "type": "stock",  "position": 1_000_000},
    {"ticker": "SU26240RMFS6", "type": "bond",   "position":   500_000},
    {"ticker": "SiU5",         "type": "future", "position":         2},
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("risklens")

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute(
    """
    CREATE TABLE IF NOT EXISTS results (
        ts          TEXT,
        ticker      TEXT,
        type        TEXT,
        method      TEXT,
        position    REAL,
        var         REAL,
        cvar        REAL,
        confidence  REAL,
        mean_return REAL,
        volatility  REAL,
        sharpe      REAL,
        delta       REAL,
        gamma       REAL,
        vega        REAL,
        theta       REAL,
        rho         REAL,
        PRIMARY KEY(ts, ticker, method)
    )"""
)
db.execute(
    """
    CREATE TABLE IF NOT EXISTS portfolio (
        ts          TEXT,
        method      TEXT,
        var         REAL,
        cvar        REAL,
        confidence  REAL,
        PRIMARY KEY(ts, method)
    )"""
)
db.commit()

class Position(BaseModel):
    ticker: str
    type:   Literal["stock", "bond", "future", "option"]
    position: float
    strike: Optional[float] = None
    expiry: Optional[dt.date] = None
    implied_vol: Optional[float] = None
    underlying: Optional[str] = None
    option_type: Optional[Literal["call", "put"]] = None

class PortfolioRequest(BaseModel):
    portfolio: List[Position]
    method: Literal["historical", "parametric"] = "historical"
    confidence: float = Field(DEFAULT_CONFIDENCE, gt=0, lt=1)

BASE_ISS = "https://iss.moex.com/iss"

def endpoint_for(sec_type: str, ticker: str) -> str:
    if sec_type == "stock":
        return f"{BASE_ISS}/engines/stock/markets/shares/securities/{ticker}/candles.json"
    if sec_type == "bond":
        return f"{BASE_ISS}/engines/stock/markets/bonds/securities/{ticker}/candles.json"
    if sec_type == "future":
        return f"{BASE_ISS}/engines/futures/markets/forts/securities/{ticker}/candles.json"
    if sec_type == "option":
        raise NotImplementedError
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

try:
    from scipy.stats import norm
except ImportError:
    norm = None

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
    return -pct_loss * position

async def calc_instrument(inst: Position, level: float, method: str) -> Tuple:
    today = dt.date.today()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    if inst.type == "option":
        if not all([inst.strike, inst.expiry, inst.implied_vol, inst.underlying, inst.option_type]):
            raise RuntimeError(f"Incomplete option data for {inst.ticker}")
        df_u = await fetch_candles(inst.underlying, "stock", start, today)
        if df_u.empty:
            raise RuntimeError(f"No data for underlying {inst.underlying}")
        S = df_u["price"].iloc[-1]
        K = inst.strike
        T = (inst.expiry - today).days / 252
        r = RISK_FREE_RATE
        sigma = inst.implied_vol
        d = opt_delta(S, K, T, r, sigma, inst.option_type)
        g = opt_gamma(S, K, T, r, sigma)
        v = opt_vega(S, K, T, r, sigma)
        t = opt_theta(S, K, T, r, sigma, inst.option_type)
        p = opt_rho(S, K, T, r, sigma, inst.option_type)
        return d * inst.position, g * inst.position, v * inst.position, t * inst.position, p * inst.position, pd.Series()
    df = await fetch_candles(inst.ticker, inst.type, start, today)
    if df.empty or len(df) < 50:
        raise RuntimeError(f"{inst.ticker}: insufficient data")
    returns = df["price"].pct_change().dropna()
    mean_ret = float(returns.mean())
    vol = float(returns.std(ddof=1))
    rf_daily = RISK_FREE_RATE / 252
    sharpe = (mean_ret - rf_daily) / vol if vol else 0.0
    if method == "historical":
        var_pct, cvar_pct = _historical_var_cvar(returns, level)
    else:
        var_pct, cvar_pct = _parametric_var_cvar(returns, level)
    abs_var = _pct_to_value(var_pct, inst.position)
    abs_cvar = _pct_to_value(cvar_pct, inst.position)
    return abs_var, abs_cvar, mean_ret, vol, sharpe, returns

async def calc_portfolio_once(req: PortfolioRequest) -> Dict:
    per_instr = []
    returns_map: Dict[str, pd.Series] = {}
    weights: List[float] = []
    for inst in req.portfolio:
        if inst.type == "option":
            d, g, v, t, p, _ = await calc_instrument(inst, req.confidence, req.method)
            per_instr.append({
                "ticker": inst.ticker,
                "type": inst.type,
                "position": inst.position,
                "delta": d,
                "gamma": g,
                "vega": v,
                "theta": t,
                "rho": p
            })
        else:
            v, cv, m, vol, sr, series = await calc_instrument(inst, req.confidence, req.method)
            per_instr.append({
                "ticker": inst.ticker,
                "type": inst.type,
                "position": inst.position,
                "VaR": v,
                "CVaR": cv,
                "mean_return": m,
                "volatility": vol,
                "sharpe_ratio": sr
            })
            returns_map[inst.ticker] = series
            weights.append(inst.position)
    combined = pd.concat(returns_map.values(), axis=1, keys=returns_map.keys()).dropna() if returns_map else pd.DataFrame()
    cov_matrix = combined.cov().to_dict() if not combined.empty else {}
    corr_matrix = combined.corr().to_dict() if not combined.empty else {}
    port_var = port_cvar = 0.0
    if not combined.empty:
        w = np.array(weights) / np.sum(weights)
        port_ret = (combined * w).sum(axis=1)
        if req.method == "historical":
            vp, cp = _historical_var_cvar(port_ret, req.confidence)
        else:
            vp, cp = _parametric_var_cvar(port_ret, req.confidence)
        total = np.sum(weights)
        port_var = _pct_to_value(vp, total)
        port_cvar = _pct_to_value(cp, total)
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "confidence": req.confidence,
        "method": req.method,
        "portfolio_value": sum(weights),
        "VaR": port_var,
        "CVaR": port_cvar,
        "meta": {"cov_matrix": cov_matrix, "corr_matrix": corr_matrix},
        "details": per_instr
    }

async def daily_job() -> None:
    today = dt.date.today()
    log.info("daily_job started")
    returns_map: Dict[str, pd.Series] = {}
    weights: List[float] = []
    for meta in SCHEDULED_PORTFOLIO:
        inst = Position(**meta)
        if inst.type != "option":
            for m in ["historical", "parametric"]:
                v, c, m_r, vol, sr, series = await calc_instrument(inst, DEFAULT_CONFIDENCE, m)
                db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        today.isoformat(), inst.ticker, inst.type, m, inst.position,
                        v, c, DEFAULT_CONFIDENCE, m_r, vol, sr, 0.0, 0.0, 0.0, 0.0
                    )
                )
                returns_map.setdefault(m, {})[inst.ticker] = series
            weights.append(inst.position)
    db.commit()
    for m in ["historical", "parametric"]:
        combined = pd.concat(returns_map.get(m, {}, {}).values(), axis=1).dropna() if returns_map.get(m) else pd.DataFrame()
        if combined.empty:
            continue
        w = np.array(weights) / np.sum(weights)
        port_ret = (combined * w).sum(axis=1)
        if m == "historical":
            vp, cp = _historical_var_cvar(port_ret, DEFAULT_CONFIDENCE)
        else:
            vp, cp = _parametric_var_cvar(port_ret, DEFAULT_CONFIDENCE)
        total = np.sum(weights)
        db.execute(
            "INSERT OR REPLACE INTO portfolio VALUES (?,?,?,?,?)",
            (
                today.isoformat(), m,
                _pct_to_value(vp, total),
                _pct_to_value(cp, total),
                DEFAULT_CONFIDENCE
            )
        )
    db.commit()
    log.info("daily_job finished")

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

app = FastAPI(lifespan=lifespan)

@app.post("/calc")
async def calc_endpoint(req: PortfolioRequest):
    try:
        return await calc_portfolio_once(req)
    except Exception as exc:
        raise HTTPException(400, str(exc))

@app.get("/var/{ticker}")
def read_var(
    ticker: str,
    method: Literal["historical", "parametric"] = Query("historical")
):
    row = db.execute(
        "SELECT ts, var, cvar, confidence FROM results WHERE ticker=? AND method=? ORDER BY ts DESC LIMIT 1",
        (ticker.upper(), method),
    ).fetchone()
    if not row:
        raise HTTPException(404, "no data yet")
    ts, var_val, cvar_val, conf = row
    return {"ticker": ticker.upper(), "method": method, "timestamp": ts, "VaR": var_val, "CVaR": cvar_val, "confidence": conf}

@app.get("/portfolio")
def read_portfolio(method: Literal["historical", "parametric"] = Query("historical")):
    row = db.execute(
        "SELECT ts, var, cvar, confidence FROM portfolio WHERE method=? ORDER BY ts DESC LIMIT 1",
        (method,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "no data yet")
    ts, var_val, cvar_val, conf = row
    return {"portfolio": "scheduled_default", "method": method, "timestamp": ts, "VaR": var_val, "CVaR": cvar_val, "confidence": conf}

@app.post("/run-now")
async def run_now():
    await daily_job()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prototype:app", host="127.0.0.1", port=8000)
