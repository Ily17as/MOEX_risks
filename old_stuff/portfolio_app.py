import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, date

# --- Конфигурация страницы ---
st.set_page_config(
    page_title="RiskLens — Портфельный анализ VaR/CVaR и греков",
    page_icon="📊",
    layout="wide"
)

# --- Заголовок ---
st.title("📊 RiskLens — Портфельный анализ VaR/CVaR и греков")

# --- Sidebar: настройки расчёта ---
st.sidebar.header("Настройки расчёта")
METHOD = st.sidebar.selectbox("Метод расчёта", ["historical", "parametric"])
CONFIDENCE = st.sidebar.slider("Уровень доверия", 0.90, 0.995, 0.99, format="%.3f")

# --- Инициализация состояния портфеля ---
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# --- Форма для ввода портфеля ---
st.subheader("1) Определите портфель")
with st.form("portfolio_form"):
    # Кнопка добавления новой строки
    if st.form_submit_button("➕ Добавить инструмент", use_container_width=True, on_click=lambda: st.session_state.portfolio.append({
        "ticker": "",
        "type": "stock",
        "position": 0.0,
        "underlying": "",
        "strike": 0.0,
        "expiry": date.today(),
        "implied_vol": 0.0,
        "option_type": "call"
    })):
        pass

    # Отображаем каждую запись
    for idx, inst in enumerate(st.session_state.portfolio):
        st.markdown(f"**Инструмент #{idx+1}**")
        cols = st.columns([1.5, 1, 1, 1, 1, 1, 1])
        inst["ticker"] = cols[0].text_input("Ticker", inst["ticker"], key=f"ticker_{idx}")
        inst["type"] = cols[1].selectbox("Тип", ["stock", "future", "option"],
                                         index=["stock", "future", "option"].index(inst["type"]),
                                         key=f"type_{idx}")
        inst["position"] = cols[2].number_input("Позиция", value=inst["position"], key=f"position_{idx}")

        # Дополнительные поля для опционов
        if inst["type"] == "option":
            inst["underlying"] = cols[3].text_input("Базовый актив", inst["underlying"], key=f"under_{idx}")
            inst["strike"] = cols[4].number_input("Страйк", value=inst["strike"], key=f"strike_{idx}")
            inst["expiry"] = cols[5].date_input("Экспирация", value=inst["expiry"], key=f"expiry_{idx}")
            inst["implied_vol"] = cols[6].number_input("Импл. волатильность", value=inst["implied_vol"],
                                                       format="%.2f", key=f"vol_{idx}")
            inst.setdefault("option_type", "call")
            inst["option_type"] = cols[6].selectbox("call/put", ["call", "put"],
                                                    index=["call", "put"].index(inst["option_type"]),
                                                    key=f"otype_{idx}")

    # Кнопка расчёта
    calc = st.form_submit_button("🚀 Рассчитать VaR/CVaR и греки")


# --- Вызов бэкенда и вывод ---
if calc:
    API_URL = "http://localhost:8000/calc"
    payload = {
        "portfolio": [],
        "method": METHOD,
        "confidence": CONFIDENCE
    }

    # Подготовка данных
    for inst in st.session_state.portfolio:
        row = inst.copy()
        # Приведение даты к ISO-формату
        if isinstance(row.get("expiry"), date):
            row["expiry"] = row["expiry"].isoformat()
        payload["portfolio"].append(row)

    # Запрос к API
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        # --- Основные метрики ---
        st.subheader("📉 Результаты портфеля")
        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio Value", f"{data.get('portfolio_value', 0):,.2f}")
        m2.metric("VaR",             f"{data.get('VaR', 0):,.2f}")
        m3.metric("CVaR",            f"{data.get('CVaR', 0):,.2f}")
        st.caption(f"Метод: {data['method']} • Уровень доверия: {data['confidence']*100:.1f}%")

        # --- Матрицы риска ---
        cov = data.get("meta", {}).get("cov_matrix")
        corr = data.get("meta", {}).get("corr_matrix")
        if cov and corr:
            st.subheader("🔢 Матрицы риска")
            c1, c2 = st.columns(2)
            c1.dataframe(pd.DataFrame(cov),  use_container_width=True)
            c2.dataframe(pd.DataFrame(corr), use_container_width=True)

        # --- Детали по инструментам и скачивание ---
        st.subheader("📋 Детализация по инструментам")
        df_details = pd.DataFrame(data.get("details", []))
        st.dataframe(df_details, use_container_width=True)
        csv = df_details.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Скачать детали CSV", csv, "details.csv", "text/csv")
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
        except requests.HTTPError:
            st.error(f"Status: {response.status_code}")
            st.text(response.text)
            st.json(payload)
            raise
    except requests.RequestException as e:
        st.error(f"Ошибка запроса к API: {e}")
    except ValueError:
        st.error("Получен некорректный ответ от сервера.")
