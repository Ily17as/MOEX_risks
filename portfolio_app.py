import json
import requests
import streamlit as st

API = "http://localhost:8000/calc"

st.set_page_config("RiskLens — Portfolio VaR", "📊")
st.title("📊 RiskLens — портфельный VaR/CVaR")

# ─────────────────────── UI: начальный портфель ─────────────────────── #
st.subheader("1) Определите портфель")

# Используем session_state для хранения портфеля между перезапусками
if "portfolio_rows" not in st.session_state:
    st.session_state.portfolio_rows = [
        {"ticker": "SBER", "type": "stock", "position": 1_000_000},
        {"ticker": "SiU5", "type": "future", "position": 2},
    ]

# Кнопка для добавления строки
if st.button("➕ Добавить строку"):
    st.session_state.portfolio_rows.append({"ticker": "", "type": "stock", "position": 0.0})

# Отображаем редактор с поддержкой динамического числа строк
rows = st.data_editor(
    st.session_state.portfolio_rows,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
)

# Обновляем session_state после редактирования
st.session_state.portfolio_rows = rows

# ─────────────────────── UI: параметры расчёта ─────────────────────── #
st.subheader("2) Параметры расчёта")

method = st.selectbox("Метод расчёта", ["historical", "parametric"])
confidence = st.slider("Уровень доверия", 0.90, 0.995, 0.99)

# ─────────────────────── UI: запуск ─────────────────────── #
if st.button("🚀 Рассчитать VaR/CVaR"):
    payload = {"portfolio": rows, "method": method, "confidence": confidence}

    try:
        r = requests.post(API, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        st.subheader("📉 Результаты портфеля")
        st.metric("VaR",  f"{data['VaR']:,.2f}")
        st.metric("CVaR", f"{data['CVaR']:,.2f}")
        st.caption(f"Метод: {data['method']} • Уровень доверия: {data['confidence']*100:.1f}%")

        st.subheader("📋 Детализация по инструментам")
        st.dataframe(data["details"], use_container_width=True)

    except requests.RequestException as exc:
        st.error(f"Ошибка запроса: {exc}")
