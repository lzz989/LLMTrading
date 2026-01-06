from __future__ import annotations


def resample_to_weekly(df, *, week_rule: str = "W-FRI"):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r requirements.txt") from exc

    if df is None or df.empty:
        raise ValueError("df 为空，别闹。")

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    have_ohlc = all(c in df2.columns for c in ["open", "high", "low", "close"])
    if not have_ohlc:
        if "close" not in df2.columns:
            raise ValueError("缺少 close 列，没法转周K。")
        df2["open"] = df2["close"]
        df2["high"] = df2["close"]
        df2["low"] = df2["close"]

    if "volume" not in df2.columns:
        df2["volume"] = 0.0

    df2 = df2.set_index("date")
    # 关键：周K 的日期用“该周最后一个交易日”，别用 resample 的周五标签去误导人（数据没更新也会看起来像更新了）。
    df2["_last_date"] = df2.index
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "_last_date": "last",
    }
    if "amount" in df2.columns:
        agg["amount"] = "sum"
    out = df2.resample(week_rule).agg(agg)
    out = out.dropna(subset=["close"]).reset_index()
    if "_last_date" in out.columns:
        out["date"] = out["_last_date"]
        out = out.drop(columns=["_last_date"])
    return out
