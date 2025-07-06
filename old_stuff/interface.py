# streamlit_app.py
import requests
import streamlit as st
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"        # если меняли порт/хост — поправьте здесь

st.set_page_config(page_title="RiskLens — VaR/CVaR", page_icon="📈", layout="centered")
st.title("📈  RiskLens — локальный VaR / CVaR")

# ─────────────────────────── Ввод тикера ────────────────────────────
ticker = st.text_input("Тикер (латиницей)", value="SBER").upper().strip()
if not ticker:
    st.stop()

# ─────────────────────────── Кнопки действий ────────────────────────
col_run, col_get = st.columns(2)

with col_run:
    if st.button("🚀 Запустить расчёт сейчас"):
        try:
            r = requests.post(f"{API_BASE}/run-now", timeout=60)
            if r.ok:
                st.success("Запущено")
            else:
                st.error(f"HTTP {r.status_code} | body: {r.text!r}")
        except requests.RequestException as exc:
            st.error(f"Сетевое исключение: {exc}")

with col_get:
    if st.button("🔄 Получить последний VaR/CVaR"):
        placeholder = st.empty()          # для анимации спиннера
        with placeholder.container():
            st.spinner("Запрашиваем данные…")
        try:
            r = requests.get(f"{API_BASE}/var/{ticker}", timeout=10)
            placeholder.empty()

            if r.ok:
                data = r.json()
                st.subheader(f"Результаты на {data['timestamp']}")

                st.metric("VaR",  f"{data['VaR']  :,.2f}")
                st.metric("CVaR", f"{data['CVaR'] :,.2f}")
                st.caption(f"Уровень доверия: {data['confidence']*100:.0f}%")
            else:
                placeholder.empty()
                st.warning(f"⛔ Сервер вернул HTTP {r.status_code}: {r.text or 'No body'}")

        except requests.RequestException as exc:
            placeholder.empty()
            st.error(f"Сетевое исключение: {exc}")

# ─────────────────────────── Footer ────────────────────────────
st.divider()
st.caption(
    f"⏰ Последнее обновление страницы: "
    f"{datetime.now().strftime('%H:%M:%S')}"
)
