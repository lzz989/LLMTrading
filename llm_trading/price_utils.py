from __future__ import annotations

import math
from typing import Any


def calc_pct_chg(prev_close: float | None, close: float | None) -> float | None:
    if prev_close is None or close is None:
        return None
    try:
        c0 = float(prev_close)
        c1 = float(close)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if c0 <= 0 or (not math.isfinite(c0)) or (not math.isfinite(c1)):
        return None
    return float((c1 / c0) - 1.0)


def extract_close_pair(df: Any) -> tuple[float | None, float | None, str | None]:
    """Return (prev_close, close_last, as_of_date) from a price DataFrame."""
    if df is None or getattr(df, "empty", True):
        return None, None, None
    try:
        import pandas as pd

        close_ser = pd.to_numeric(df["close"], errors="coerce").astype(float)
        if close_ser is None or close_ser.empty:
            return None, None, None
        close_last = float(close_ser.iloc[-1])
        prev_close = float(close_ser.iloc[-2]) if len(close_ser) >= 2 else None
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        return None, None, None

    as_of = None
    try:
        last_dt = df["date"].iloc[-1]
        as_of = str(last_dt.date()) if hasattr(last_dt, "date") else str(last_dt)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        as_of = None

    return prev_close, close_last, as_of


def select_price_df(df_raw: Any, df_fallback: Any, *, asset: str) -> tuple[Any, str, str | None]:
    """
    Pick price series for display (prefer raw). Return (df, price_basis, warning).
    """
    if df_raw is not None and (not getattr(df_raw, "empty", True)):
        return df_raw, "raw", None

    a = str(asset or "").strip().lower()
    if a in {"stock", "etf"}:
        return df_fallback, "qfq", "raw_missing_fallback_qfq"
    return df_fallback, "raw", "raw_missing_fallback_raw"
