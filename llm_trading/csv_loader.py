from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class CsvSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnGuess:
    date: str
    open: str | None
    high: str | None
    low: str | None
    close: str
    volume: str | None


def _first_match(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {c.strip(): c for c in columns}
    lower_map = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in normalized:
            return normalized[cand]
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def guess_columns(columns: Iterable[str]) -> ColumnGuess:
    date_col = _first_match(
        columns,
        ["date", "datetime", "time", "timestamp", "日期", "时间", "Date", "Datetime", "Time", "Timestamp"],
    )
    open_col = _first_match(columns, ["open", "Open", "开盘", "开盘价"])
    high_col = _first_match(columns, ["high", "High", "最高", "最高价"])
    low_col = _first_match(columns, ["low", "Low", "最低", "最低价"])
    close_col = _first_match(columns, ["close", "Close", "收盘", "收盘价"])
    volume_col = _first_match(columns, ["volume", "Volume", "成交量", "vol", "Vol"])

    if not date_col:
        raise CsvSchemaError("找不到日期列：请用 --date-col 指定。")
    if not close_col:
        raise CsvSchemaError("找不到收盘列：请用 --close-col 指定。")

    return ColumnGuess(date=date_col, open=open_col, high=high_col, low=low_col, close=close_col, volume=volume_col)


def load_ohlcv_csv(
    csv_path: str | Path,
    *,
    date_col: str | None = None,
    open_col: str | None = None,
    high_col: str | None = None,
    low_col: str | None = None,
    close_col: str | None = None,
    volume_col: str | None = None,
    encoding: str | None = None,
):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r requirements.txt") from exc

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 不存在：{path}")

    df = pd.read_csv(path, encoding=encoding)
    if df.empty:
        raise CsvSchemaError("CSV 是空的，别闹。")

    guessed = guess_columns(df.columns)
    date_col = date_col or guessed.date
    open_col = open_col or guessed.open
    high_col = high_col or guessed.high
    low_col = low_col or guessed.low
    close_col = close_col or guessed.close
    volume_col = volume_col or guessed.volume

    keep_cols = [date_col, close_col]
    for c in [open_col, high_col, low_col]:
        if c and c in df.columns and c not in keep_cols:
            keep_cols.append(c)
    if volume_col and volume_col in df.columns:
        keep_cols.append(volume_col)

    df = df[keep_cols].copy()
    df.rename(
        columns={
            date_col: "date",
            (open_col or ""): "open",
            (high_col or ""): "high",
            (low_col or ""): "low",
            close_col: "close",
            (volume_col or ""): "volume",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    for c in ["open", "high", "low"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df
