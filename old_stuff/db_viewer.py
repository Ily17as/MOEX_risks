import sqlite3
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import altair as alt   # для графика

DB_PATH = Path("../risklens.db")            # поменяйте, если БД в другом месте

# ──────────────────────── helpers ───────────────────────── #
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Файл {DB_PATH} не найден")
        st.stop()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM results", conn,
                     parse_dates={"ts": {"format": "%Y-%m-%d"}})
    conn.close()
    df.rename(columns={"ts": "date"}, inplace=True)
    return df


# ──────────────────────── UI ─────────────────────────────── #
st.set_page_config("RiskLens DB Viewer", "📊", layout="wide")
st.title("📊 RiskLens — просмотр базы данных")

df = load_data()
if df.empty:
    st.info("Таблица results пока пустая — сделайте хотя бы один расчёт.")
    st.stop()

with st.sidebar:
    st.header("Фильтры")
    tickers = sorted(df["ticker"].unique())
    chosen = st.multiselect("Тикер(ы)", tickers, default=tickers[:1])

    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    start, end = st.date_input(
        "Диапазон дат", (min_d, max_d),
        min_value=min_d, max_value=max_d, format="DD.MM.YYYY"
    )

# применяем фильтры
start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

mask = df["ticker"].isin(chosen) & df["date"].between(start_ts, end_ts)
view = df.loc[mask].sort_values(["ticker", "date"])

st.subheader("Таблица результатов")
st.dataframe(view.style.format({"var": "{:,.2f}", "cvar": "{:,.2f}"}),
             use_container_width=True, height=400)

if not view.empty:
    st.subheader("График VaR / CVaR")
    chart = alt.Chart(view).transform_fold(
        ["var", "cvar"], as_=["Metric", "Value"]
    ).mark_line().encode(
        x="date:T", y="Value:Q", color="Metric:N",
        tooltip=["date:T", "Metric:N", "Value:Q"]
    ).properties(height=350, width="container")
    st.altair_chart(chart, use_container_width=True)

st.caption(f"Записей в выборке: {len(view):,}")
