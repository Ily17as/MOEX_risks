# 📊 RiskLens — портфельный анализ VaR / CVaR для MOEX

Проект для оценки финансовых рисков (Value at Risk и Conditional VaR) на основе исторических данных MOEX. Поддерживаются акции, облигации и фьючерсы. Реализован интерактивный веб-интерфейс на Streamlit и API-сервер на FastAPI.

---

## 🚀 Возможности

- Расчёт **исторического** и **параметрического** VaR / CVaR
- Поддержка **портфельного анализа**
- Работа с инструментами: `stock`, `bond`, `future`
- Веб-интерфейс для ручного ввода портфеля
- REST API для интеграции / автоматизации
- Встроенный ежедневный пересчёт (по cron)

---

## 🏗️ Структура проекта

| Файл / модуль       | Назначение                                 |
|---------------------|--------------------------------------------|
| `risklens/api.py`           | Основной API-сервер и расчётная логика     |
| `risklens/streamlit_app.py` | Интерфейс Streamlit для ввода портфеля     |
| `risklens/config.py`| Пути к базам данных                        |

---

## 📦 Установка

```bash
git clone https://github.com/yourname/risklens
cd risklens
python -m venv .venv
source .venv/bin/activate     # для Linux/MacOS
# или
.venv\\Scripts\\activate     # для Windows
pip install -e .
```

## ▶️ Запуск

### API сервер

```bash
risklens-api
```

### Веб-интерфейс

```bash
risklens-app
```
