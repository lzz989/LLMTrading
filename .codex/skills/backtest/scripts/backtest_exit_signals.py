#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势出场信号回测（MVP，可复现）

设计目标：
- 先把“出场规则对比”跑通：MACD / Bollinger / 固定止盈
- 严禁未来函数：t 日信号 -> t+1 开盘成交（保守执行）
- 单标的全仓、非重叠交易：避免组合/并发干扰，让对比更干净

注意：
- “强趋势主升浪”是入场过滤，不是圣杯；你要改规则请改参数，别魔改代码。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd


ExitMode = Literal["macd_cross_down", "bollinger_mid_break", "take_profit_fixed"]


@dataclass(frozen=True, slots=True)
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    ret: float
    hold_days: int


def _sf(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in dfx.columns:
            if col in {"open", "high", "low"} and "close" in dfx.columns:
                dfx[col] = dfx["close"]
            else:
                dfx[col] = 0.0
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    dfx = dfx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return dfx


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    close = pd.to_numeric(dfx["close"], errors="coerce").astype(float)
    high = pd.to_numeric(dfx["high"], errors="coerce").astype(float)
    low = pd.to_numeric(dfx["low"], errors="coerce").astype(float)

    dfx["ma50"] = close.rolling(window=50, min_periods=50).mean()
    dfx["ma200"] = close.rolling(window=200, min_periods=200).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    dfx["macd"] = macd
    dfx["macd_signal"] = sig

    # Bollinger (20,2)
    mid = close.rolling(window=20, min_periods=20).mean()
    std = close.rolling(window=20, min_periods=20).std()
    dfx["bb_mid"] = mid
    dfx["bb_upper"] = mid + 2.0 * std
    dfx["bb_lower"] = mid - 2.0 * std

    # ADX(14)（MVP 实现：用 ta 库更稳；这里手写一个简化版，够用）
    # 参考：Wilder's ADX
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    n = 14
    atr = tr.rolling(window=n, min_periods=n).mean()
    plus_di = 100.0 * (plus_dm.rolling(window=n, min_periods=n).mean() / atr.replace({0.0: float("nan")}))
    minus_di = 100.0 * (minus_dm.rolling(window=n, min_periods=n).mean() / atr.replace({0.0: float("nan")}))
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace({0.0: float("nan")})).fillna(0.0)
    dfx["adx14"] = dx.rolling(window=n, min_periods=n).mean()

    # Donchian 20d high (exclude today to avoid lookahead)
    dfx["donchian20_high"] = high.shift(1).rolling(window=20, min_periods=20).max()
    return dfx


def _entry_signal_row(row) -> bool:
    # 强趋势主升浪（可复现定义）
    try:
        if row["close"] <= row["donchian20_high"]:
            return False
        if row["ma50"] <= row["ma200"]:
            return False
        if row["adx14"] <= 20:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _exit_signal_row(row, prev_row, *, mode: ExitMode, take_profit_ret: float) -> bool:
    try:
        if mode == "macd_cross_down":
            # macd 从上到下穿 signal
            return bool((prev_row["macd"] >= prev_row["macd_signal"]) and (row["macd"] < row["macd_signal"]))
        if mode == "bollinger_mid_break":
            return bool(row["close"] < row["bb_mid"])
        if mode == "take_profit_fixed":
            # 固定止盈在外层用价格判断，这里不产生技术信号
            return False
    except Exception:  # noqa: BLE001
        return False
    return False


def _simulate(
    df: pd.DataFrame,
    *,
    exit_mode: ExitMode,
    take_profit_ret: float,
    max_hold_days: int,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[list[Trade], pd.DataFrame]:
    """
    回测：t 日产生信号，t+1 open 成交。非重叠交易（flat 才能进）。
    """
    fee = max(0.0, float(fee_bps)) / 10000.0
    slip = max(0.0, float(slippage_bps)) / 10000.0

    in_pos = False
    entry_idx = None
    entry_price = None
    trades: list[Trade] = []

    equity = []
    eq = 1.0

    # 我们用“开盘成交”，所以信号日 t 的判断发生在 close 后，成交在 t+1 open。
    for i in range(1, len(df) - 1):  # -1 因为 i+1 需要 open
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        prev_row = df.iloc[i - 1]

        date_s = str(row["date"].date())
        next_date_s = str(next_row["date"].date())

        # 记录 equity（用收盘标记；简化）
        equity.append({"date": date_s, "equity": eq, "in_pos": bool(in_pos)})

        if not in_pos:
            if _entry_signal_row(row):
                px = _sf(next_row.get("open")) or _sf(next_row.get("close")) or 0.0
                if px <= 0:
                    continue
                # 买入成本：滑点+手续费
                entry_price = float(px) * (1.0 + slip + fee)
                entry_idx = i + 1
                in_pos = True
            continue

        # in position
        assert entry_idx is not None and entry_price is not None
        hold_days = (i + 1) - int(entry_idx)
        if hold_days < 1:
            hold_days = 1

        exit_now = False
        # 固定止盈：用 next open 触发（保守：只要 close 已经达到阈值，下一天开盘出）
        if exit_mode == "take_profit_fixed":
            if float(row["close"]) >= float(entry_price) * (1.0 + float(take_profit_ret)):
                exit_now = True

        # 技术出场：在信号日 t 判断，t+1 open 成交
        if not exit_now:
            exit_now = _exit_signal_row(row, prev_row, mode=exit_mode, take_profit_ret=take_profit_ret)

        # 兜底：最长持有期
        if (not exit_now) and int(max_hold_days) > 0 and int(hold_days) >= int(max_hold_days):
            exit_now = True

        if exit_now:
            px = _sf(next_row.get("open")) or _sf(next_row.get("close")) or 0.0
            if px <= 0:
                px = float(row["close"])
                next_date_s = date_s
            # 卖出成本：滑点+手续费
            exit_price = float(px) * (1.0 - slip - fee)
            r = float(exit_price / float(entry_price) - 1.0)
            trades.append(
                Trade(
                    entry_date=str(df.iloc[int(entry_idx)]["date"].date()),
                    entry_price=float(entry_price),
                    exit_date=str(next_date_s),
                    exit_price=float(exit_price),
                    ret=float(r),
                    hold_days=int(hold_days),
                )
            )
            # 更新 equity（离散复利）
            eq *= 1.0 + float(r)
            in_pos = False
            entry_idx = None
            entry_price = None

    if len(df) >= 1:
        equity.append({"date": str(df.iloc[-1]["date"].date()), "equity": eq, "in_pos": bool(in_pos)})
    eq_df = pd.DataFrame(equity)
    return trades, eq_df


def _max_drawdown(eq: pd.Series) -> float:
    if eq is None or eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def _cagr(eq0: float, eq1: float, days: int) -> float | None:
    if days <= 0 or eq0 <= 0 or eq1 <= 0:
        return None
    years = float(days) / 252.0
    if years <= 0:
        return None
    return float((eq1 / eq0) ** (1.0 / years) - 1.0)


def _summarize(trades: list[Trade], eq_df: pd.DataFrame) -> dict:
    if eq_df is None or eq_df.empty:
        return {"trades": 0}
    eq = pd.to_numeric(eq_df["equity"], errors="coerce").astype(float)
    mdd = _max_drawdown(eq)
    start = str(eq_df["date"].iloc[0])
    end = str(eq_df["date"].iloc[-1])
    days = int(len(eq_df))
    cagr = _cagr(float(eq.iloc[0]), float(eq.iloc[-1]), days)

    rets = [float(t.ret) for t in trades]
    holds = [int(t.hold_days) for t in trades]
    wins = int(sum(1 for r in rets if r > 0))
    return {
        "start": start,
        "end": end,
        "days": days,
        "equity_end": float(eq.iloc[-1]),
        "cagr": cagr,
        "max_drawdown": float(mdd),
        "trades": int(len(trades)),
        "win_rate": (float(wins) / float(len(trades))) if trades else None,
        "avg_ret": (sum(rets) / len(rets)) if rets else None,
        "med_ret": float(pd.Series(rets).median()) if rets else None,
        "avg_hold_days": (sum(holds) / len(holds)) if holds else None,
        "p50_hold_days": float(pd.Series(holds).quantile(0.50)) if holds else None,
        "p90_hold_days": float(pd.Series(holds).quantile(0.90)) if holds else None,
    }


def _load_daily(asset: str, symbol: str, *, source: str, cache_ttl_hours: float) -> pd.DataFrame:
    from llm_trading.akshare_source import FetchParams
    from llm_trading.data_cache import fetch_daily_cached

    cache_dir = Path("data") / "cache" / str(asset)
    df = fetch_daily_cached(
        FetchParams(asset=str(asset), symbol=str(symbol), source=str(source)),
        cache_dir=cache_dir,
        ttl_hours=float(cache_ttl_hours),
    )
    return _ensure_ohlcv(df)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="etf", choices=["etf", "stock", "index", "crypto"])
    p.add_argument("--symbols", required=True, help="逗号分隔，如 sh518880,sh159937")
    p.add_argument("--source", default="akshare", help="akshare/tushare/auto（依赖 llm_trading FetchParams）")
    p.add_argument("--start", default="", help="YYYY-MM-DD")
    p.add_argument("--end", default="", help="YYYY-MM-DD")
    p.add_argument("--cache-ttl-hours", type=float, default=24.0)

    p.add_argument("--fee-bps", type=float, default=10.0, help="单边手续费（bps）")
    p.add_argument("--slippage-bps", type=float, default=5.0, help="单边滑点（bps）")

    p.add_argument("--take-profit-ret", type=float, default=0.20)
    p.add_argument("--max-hold-days", type=int, default=252)

    p.add_argument("--out", required=True, help="输出 markdown 路径")
    args = p.parse_args()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("--symbols 不能为空")

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# 趋势出场信号回测（MVP）\n")
    lines.append(f"- generated_at: {datetime.now().isoformat()}\n")
    lines.append(f"- asset: {args.asset}\n")
    lines.append(f"- symbols: {', '.join(symbols)}\n")
    lines.append(f"- period: {args.start or 'auto'} ~ {args.end or 'auto'}\n")
    lines.append(f"- exec: signal@t close -> trade@t+1 open (conservative)\n")
    lines.append(f"- cost: fee_bps={float(args.fee_bps)} slippage_bps={float(args.slippage_bps)} (per side)\n")
    lines.append("")

    modes: list[ExitMode] = ["macd_cross_down", "bollinger_mid_break", "take_profit_fixed"]

    for sym in symbols:
        df = _load_daily(str(args.asset), str(sym), source=str(args.source), cache_ttl_hours=float(args.cache_ttl_hours))
        if df is None or getattr(df, "empty", True):
            lines.append(f"## {sym}\n\n- error: no_data\n")
            continue

        if args.start:
            try:
                d0 = pd.to_datetime(str(args.start)).date()
                df = df[df["date"].dt.date >= d0].reset_index(drop=True)
            except Exception:  # noqa: BLE001
                pass
        if args.end:
            try:
                d1 = pd.to_datetime(str(args.end)).date()
                df = df[df["date"].dt.date <= d1].reset_index(drop=True)
            except Exception:  # noqa: BLE001
                pass

        df = _add_indicators(df)
        df = df.dropna(subset=["ma200", "donchian20_high", "bb_mid", "macd", "macd_signal", "adx14"]).reset_index(drop=True)
        if len(df) < 260:
            lines.append(f"## {sym}\n\n- warning: short_history rows={len(df)}\n")

        rows = []
        for m in modes:
            trades, eq = _simulate(
                df,
                exit_mode=m,
                take_profit_ret=float(args.take_profit_ret),
                max_hold_days=int(args.max_hold_days),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            s = _summarize(trades, eq)
            rows.append((m, s, trades))

        lines.append(f"## {sym}\n")
        lines.append("")
        lines.append("| 出场规则 | CAGR | MaxDD | Trades | WinRate | AvgRet | MedRet | AvgHoldDays | P90HoldDays |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m, s, _tr in rows:
            def fmt(x, *, pct: bool = False):
                if x is None:
                    return ""
                try:
                    v = float(x)
                except Exception:  # noqa: BLE001
                    return ""
                if pct:
                    return f"{v*100:.2f}%"
                return f"{v:.4f}"

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    m,
                    fmt(s.get("cagr"), pct=True),
                    fmt(s.get("max_drawdown"), pct=True),
                    str(s.get("trades") or 0),
                    fmt(s.get("win_rate"), pct=True),
                    fmt(s.get("avg_ret"), pct=True),
                    fmt(s.get("med_ret"), pct=True),
                    fmt(s.get("avg_hold_days"), pct=False),
                    fmt(s.get("p90_hold_days"), pct=False),
                )
            )
        lines.append("")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(str(out_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

