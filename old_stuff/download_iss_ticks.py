#!/usr/bin/env python
# download_iss_ticks.py — ISS-тики в CSV (ts,price,qty,side,trade_id)

import sys
import pathlib as p
import requests
import pandas as pd

ISS = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/"
    "boards/TQBR/securities/{symbol}/trades.json?date={date}&start={start}"
)

def fetch(sym: str, date: str, out_csv: str) -> None:
    rows, start, first_cols = [], 0, None

    while True:
        url = ISS.format(symbol=sym, date=date, start=start)
        data = requests.get(url, timeout=10).json()
        chunk = data["trades"]["data"]
        if not chunk:
            break

        if first_cols is None:                # запоминаем имена колонок с первой страницы
            first_cols = data["trades"]["columns"]
            print("ISS columns:", first_cols)

        rows.extend(chunk)
        start += len(chunk)

    if not rows:
        raise RuntimeError("ISS вернул 0 сделок — проверь символ/дату")

    cols = first_cols                      # колонки одинаковы на всех страницах

    # --- ищем столбец-идентификатор ------------------------------------------------
    id_col = next(
        (c for c in cols if "TRADE" in c and ("ID" in c or "NUM" in c or "NO" in c)),
        None,
    )
    if id_col is None:
        raise KeyError("Не найден столбец идентификатора сделки (*TRADE*ID/NUM/NO*)")

    rename = {
        "TRADETIME": "ts",
        "PRICE": "price",
        "QUANTITY": "qty",
        "BUYSELL": "side",
        id_col: "trade_id",
    }

    df = (
        pd.DataFrame(rows, columns=cols)[list(rename)]
        .rename(columns=rename)
        .assign(
            side=lambda d: d["side"].map({"B": 1, "S": 2}).fillna(0).astype("uint8"),
            ts=lambda d: pd.to_datetime(d["ts"], utc=True)
                         .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        )
    )

    dest = p.Path(out_csv)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, header=False, index=False)
    print(f"[DONE] {len(df):,} ticks → {dest}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python download_iss_ticks.py <SYMBOL> <YYYY-MM-DD> <OUT_CSV>")
    fetch(*sys.argv[1:])
