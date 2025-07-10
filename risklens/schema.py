TICK_COLS = {
    "ts":        "datetime[μs, UTC]",  # микросекунды, чтобы не терять внутримиллисекундную сортировку
    "price":     "float64",
    "qty":       "uint32",
    "side":      "uint8",              # 1=buy, 2=sell, 0=undef
    "trade_id":  "uint64",
}
BAR_COLS = {
    "ts":     "datetime[ms, UTC]",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "uint64",
}