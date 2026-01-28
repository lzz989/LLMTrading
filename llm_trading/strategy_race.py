from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class WeeklyStrategyDef:
    key: str
    name: str
    description: str
    compute_desired_at_close: Callable[[Any, Any, dict[str, Any]], Any]


def _ensure_weekly_ohlc(dfw):
    df2 = dfw
    if "open" not in df2.columns:
        df2 = df2.copy()
        df2["open"] = df2.get("close")
    if "high" not in df2.columns:
        df2 = df2.copy()
        df2["high"] = df2.get("close")
    if "low" not in df2.columns:
        df2 = df2.copy()
        df2["low"] = df2.get("close")
    return df2


def _to_bool_series(df, s):
    import pandas as pd

    if s is None:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)
    if len(s) != len(df):
        raise ValueError("信号长度不匹配")
    return s.fillna(False).astype(bool)


def _desired_buy_hold(df_weekly, _df_daily, _cfg: dict[str, Any]):
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)
    return pd.Series([True] * len(df_weekly), index=df_weekly.index, dtype=bool)


def _desired_ma_timing(df_weekly, _df_daily, cfg: dict[str, Any]):
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    w = int(cfg.get("ma_weeks") or 40)
    w = max(2, min(w, 400))
    close = pd.to_numeric(df_weekly["close"], errors="coerce").astype(float)
    ma = close.rolling(window=w, min_periods=w).mean()
    return (close > ma).fillna(False).astype(bool)


def _desired_tsmom(df_weekly, _df_daily, cfg: dict[str, Any]):
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    lb = int(cfg.get("lookback_weeks") or 52)
    lb = max(4, min(lb, 520))
    close = pd.to_numeric(df_weekly["close"], errors="coerce").astype(float)
    mom = (close / close.shift(lb).replace({0.0: float("nan")})) - 1.0
    return (mom > 0.0).fillna(False).astype(bool)


def _desired_turtle(df_weekly, _df_daily, cfg: dict[str, Any]):
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    entry_w = int(cfg.get("entry_weeks") or 20)
    exit_w = int(cfg.get("exit_weeks") or 10)
    entry_w = max(4, min(entry_w, 200))
    exit_w = max(2, min(exit_w, entry_w))

    high = pd.to_numeric(df_weekly.get("high", df_weekly["close"]), errors="coerce").astype(float)
    low = pd.to_numeric(df_weekly.get("low", df_weekly["close"]), errors="coerce").astype(float)
    close = pd.to_numeric(df_weekly["close"], errors="coerce").astype(float)

    upper = high.rolling(entry_w, min_periods=entry_w).max().shift(1)
    lower = low.rolling(exit_w, min_periods=exit_w).min().shift(1)

    desired: list[bool] = []
    in_pos = False
    for i in range(int(len(df_weekly))):
        c = float(close.iloc[i]) if i < len(close) else float("nan")
        u = float(upper.iloc[i]) if i < len(upper) else float("nan")
        l = float(lower.iloc[i]) if i < len(lower) else float("nan")
        if not in_pos:
            if math.isfinite(c) and math.isfinite(u) and c > u:
                in_pos = True
        else:
            if math.isfinite(c) and math.isfinite(l) and c < l:
                in_pos = False
        desired.append(bool(in_pos))

    return pd.Series(desired, index=df_weekly.index, dtype=bool)


def _desired_boll_mr(df_weekly, _df_daily, cfg: dict[str, Any]):
    import pandas as pd

    from .indicators import add_bollinger_bands

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    window = int(cfg.get("window") or 20)
    k = float(cfg.get("k") or 2.0)
    max_hold_weeks = int(cfg.get("max_hold_weeks") or 8)
    window = max(5, min(window, 200))
    max_hold_weeks = max(1, min(max_hold_weeks, 52))

    df = add_bollinger_bands(df_weekly, window=window, k=k)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    mid = pd.to_numeric(df["boll_mid"], errors="coerce").astype(float)
    lower = pd.to_numeric(df["boll_lower"], errors="coerce").astype(float)

    desired: list[bool] = []
    in_pos = False
    entry_i = -1
    for i in range(int(len(df))):
        c = float(close.iloc[i])
        m = float(mid.iloc[i])
        lo = float(lower.iloc[i])

        if not in_pos:
            if math.isfinite(c) and math.isfinite(lo) and c < lo:
                in_pos = True
                entry_i = int(i)
        else:
            hold = int(i - entry_i) if entry_i >= 0 else 0
            # 到点不涨就滚：均值回归不该拿着当长线
            if hold >= int(max_hold_weeks):
                in_pos = False
                entry_i = -1
            elif math.isfinite(c) and math.isfinite(m) and c > m:
                in_pos = False
                entry_i = -1

        desired.append(bool(in_pos))

    return pd.Series(desired, index=df.index, dtype=bool)


def _desired_bbb_weekly(df_weekly, df_daily, cfg: dict[str, Any]):
    """
    BBB（周线版）：用现有 BBB 入场信号 + 周线 hard exit（连续2周跌破MA50）。
    注：为统一赛马口径，这里不启用日线 soft exit（否则频率变成日线，比较不公平）。
    """
    import pandas as pd

    from .bbb import BBBParams, compute_bbb_entry_signal
    from .indicators import add_moving_averages

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    entry_ma = int(cfg.get("entry_ma") or 50)
    dist_ma50_max = float(cfg.get("dist_ma50_max") or 0.12)
    max_above_20w = float(cfg.get("max_above_20w") or 0.05)
    min_bars = int(cfg.get("min_weekly_bars_total") or 60)
    req_w_bull = bool(cfg.get("require_weekly_macd_bullish", True))
    req_w_above0 = bool(cfg.get("require_weekly_macd_above_zero", True))
    req_d_bull = bool(cfg.get("require_daily_macd_bullish", True))

    params = BBBParams(
        entry_ma=max(2, int(entry_ma)),
        dist_ma50_max=max(0.0, float(dist_ma50_max)),
        max_above_20w=max(0.0, float(max_above_20w)),
        min_weekly_bars_total=max(10, int(min_bars)),
        require_weekly_macd_bullish=bool(req_w_bull),
        require_weekly_macd_above_zero=bool(req_w_above0),
        require_daily_macd_bullish=bool(req_d_bull),
    )

    entry_sig = compute_bbb_entry_signal(df_weekly, df_daily, params=params)
    entry_sig = _to_bool_series(df_weekly, entry_sig).to_numpy(dtype=bool)

    dfw = df_weekly.copy()
    if "ma50" not in dfw.columns:
        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
    close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    ma50 = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
    hard = ((close < ma50) & (close.shift(1) < ma50.shift(1))).fillna(False).astype(bool).to_numpy(dtype=bool)

    desired: list[bool] = []
    in_pos = False
    for i in range(int(len(dfw))):
        if not in_pos:
            if bool(entry_sig[i]):
                in_pos = True
        else:
            if bool(hard[i]):
                in_pos = False
        desired.append(bool(in_pos))
    return pd.Series(desired, index=dfw.index, dtype=bool)


def list_default_weekly_strategies() -> list[WeeklyStrategyDef]:
    """
    经典策略“最小集合”：别整一百个指标把自己搞疯。
    """
    return [
        WeeklyStrategyDef(
            key="buyhold",
            name="Buy&Hold（基线）",
            description="一直持有（用来当对照组，不然赛马没参照物）",
            compute_desired_at_close=_desired_buy_hold,
        ),
        WeeklyStrategyDef(
            key="ma_timing",
            name="MA择时（10月均线）",
            description="周收盘>MA40 则持有，否则空仓（经典 TAA 风格，主打降回撤）",
            compute_desired_at_close=_desired_ma_timing,
        ),
        WeeklyStrategyDef(
            key="tsmom",
            name="TSMOM（时间序列动量）",
            description="过去 52W 动量>0 则持有，否则空仓（趋势/危机期更友好）",
            compute_desired_at_close=_desired_tsmom,
        ),
        WeeklyStrategyDef(
            key="turtle",
            name="海龟（唐奇安突破）",
            description="20W 突破入场，10W 跌破出场（经典趋势跟随）",
            compute_desired_at_close=_desired_turtle,
        ),
        WeeklyStrategyDef(
            key="boll_mr",
            name="BOLL均值回归（超跌反抽）",
            description="收盘跌破下轨入场，站回中轨/超时退出（更偏震荡市）",
            compute_desired_at_close=_desired_boll_mr,
        ),
        WeeklyStrategyDef(
            key="bbb",
            name="BBB（周线定方向+回踩）",
            description="现有 BBB 入场 + 周线 hard exit（统一周线口径）",
            compute_desired_at_close=_desired_bbb_weekly,
        ),
    ]


def _max_drawdown(equity: list[float]) -> float | None:
    if not equity:
        return None
    peak = -1e18
    worst = 0.0
    for x in equity:
        v = float(x)
        if not math.isfinite(v) or v <= 0:
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < worst:
                worst = dd
    return float(worst)


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = 0.0
    n = 0
    for x in xs:
        v = float(x)
        if not math.isfinite(v):
            continue
        s += v
        n += 1
    return float(s / n) if n > 0 else None


def _safe_std(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = _safe_mean(xs)
    if m is None:
        return None
    s2 = 0.0
    n = 0
    for x in xs:
        v = float(x)
        if not math.isfinite(v):
            continue
        s2 += (v - m) ** 2
        n += 1
    if n <= 1:
        return 0.0
    return float(math.sqrt(s2 / float(n - 1)))


def _prod_1p(xs: list[float]) -> float:
    acc = 1.0
    for x in xs:
        v = float(x)
        if not math.isfinite(v):
            continue
        acc *= 1.0 + v
    return float(acc - 1.0)


def _breakdown_by_label(*, period_returns: list[float], labels: list[str]) -> dict[str, Any]:
    """
    按“period 的 label”分段统计（无未来函数 label 已由调用方对齐好）。
    """
    out: dict[str, Any] = {}
    if not period_returns or not labels:
        return out
    n = min(len(period_returns), len(labels))

    buckets: dict[str, list[float]] = {}
    for i in range(n):
        lb = str(labels[i] or "unknown")
        buckets.setdefault(lb, []).append(float(period_returns[i]))

    for lb in ["bull", "bear", "neutral", "unknown"]:
        xs = buckets.get(lb) or []
        if not xs:
            out[lb] = {
                "periods": 0,
                "compounded_return": 0.0,
                "ann_return": None,
                "avg_return": 0.0,
                "win_rate": 0.0,
            }
            continue
        wins = sum(1 for x in xs if float(x) > 0)
        comp = float(_prod_1p(xs))
        ann = None
        try:
            if len(xs) > 0 and (1.0 + comp) > 0:
                ann = float((1.0 + comp) ** (52.0 / float(len(xs))) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            ann = None
        out[lb] = {
            "periods": int(len(xs)),
            "compounded_return": comp,
            "ann_return": ann,
            "avg_return": float(_safe_mean(xs) or 0.0),
            "win_rate": float(wins / len(xs)) if len(xs) > 0 else 0.0,
        }
    return out


def simulate_weekly_inout(
    df_weekly,
    *,
    desired_at_close,
    buy_cost: float,
    sell_cost: float,
) -> dict[str, Any]:
    """
    周线 in/out 回测（交易在“下周开盘”执行）。

    约定：
    - desired_at_close[i]：第 i 根周K 收盘后“是否希望持仓”
    - 实际持仓发生在 i+1 周开盘（避免未来函数）
    """
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        raise ValueError("df_weekly 为空")

    dfw = _ensure_weekly_ohlc(df_weekly).copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
    if len(dfw) < 10:
        raise ValueError("周线数据太短")

    desired = _to_bool_series(dfw, desired_at_close).to_numpy(dtype=bool)

    open_px = pd.to_numeric(dfw["open"], errors="coerce").astype(float).to_numpy()
    low_px = pd.to_numeric(dfw.get("low", dfw["close"]), errors="coerce").astype(float).to_numpy()
    high_px = pd.to_numeric(dfw.get("high", dfw["close"]), errors="coerce").astype(float).to_numpy()
    dates = dfw["date"]

    n = int(len(dfw))
    # pos_open[i]：第 i 周开盘是否持仓（由上周收盘的 desired 决定）
    pos_open = [False] * n
    for i in range(1, n):
        pos_open[i] = bool(desired[i - 1])

    buy_c = max(0.0, float(buy_cost))
    sell_c = max(0.0, float(sell_cost))

    equity: list[float] = [1.0] * n
    in_pos = False
    shares = 0.0
    entry_i = -1
    entry_equity = 0.0
    entry_px = 0.0
    trades: list[dict[str, Any]] = []

    maes: list[float] = []
    mfes: list[float] = []

    for i in range(0, n - 1):
        want = bool(pos_open[i])
        px = float(open_px[i]) if i < len(open_px) else float("nan")
        if not math.isfinite(px) or px <= 0:
            equity[i + 1] = equity[i]
            continue

        if (not in_pos) and want:
            # enter at open i
            shares = float(equity[i]) / (px * (1.0 + buy_c))
            in_pos = True
            entry_i = int(i)
            entry_equity = float(equity[i]) / (1.0 + buy_c)
            entry_px = float(px)
            equity[i] = float(entry_equity)
        elif in_pos and (not want):
            # exit at open i
            exit_equity = float(equity[i]) * (1.0 - sell_c)
            gross_ret = (float(px) / float(entry_px)) - 1.0 if entry_px > 0 else 0.0
            net_ret = (float(exit_equity) / float(entry_equity) - 1.0) if entry_equity > 0 else 0.0
            hold_weeks = int(i - entry_i)

            # MAE/MFE：用持仓期间周线 low/high 估一下（粗糙，但比没有强）
            span_low = low_px[entry_i:i] if entry_i >= 0 and i > entry_i else []
            span_high = high_px[entry_i:i] if entry_i >= 0 and i > entry_i else []
            mae = None
            mfe = None
            try:
                if len(span_low) > 0 and entry_px > 0:
                    mae = float(float(min(span_low)) / float(entry_px) - 1.0)
                    maes.append(mae)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                mae = None
            try:
                if len(span_high) > 0 and entry_px > 0:
                    mfe = float(float(max(span_high)) / float(entry_px) - 1.0)
                    mfes.append(mfe)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                mfe = None

            trades.append(
                {
                    "entry_week_end": str(dates.iloc[entry_i].date()) if entry_i >= 0 else None,
                    "exit_week_end": str(dates.iloc[i].date()),
                    "entry_price_open": float(entry_px),
                    "exit_price_open": float(px),
                    "hold_weeks": int(hold_weeks),
                    "net_return": float(net_ret),
                    "gross_return": float(gross_ret),
                    "mae": mae,
                    "mfe": mfe,
                }
            )

            equity[i] = float(exit_equity)
            in_pos = False
            shares = 0.0
            entry_i = -1
            entry_equity = 0.0
            entry_px = 0.0

        # next open mark-to-market
        next_px = float(open_px[i + 1])
        if not math.isfinite(next_px) or next_px <= 0:
            equity[i + 1] = equity[i]
        else:
            equity[i + 1] = float(shares * next_px) if in_pos else float(equity[i])

    # 期末强制平仓（用最后一根周K开盘价）
    if in_pos and n >= 1:
        last_i = n - 1
        last_px = float(open_px[last_i])
        if math.isfinite(last_px) and last_px > 0 and entry_px > 0 and entry_equity > 0:
            exit_equity = float(equity[last_i]) * (1.0 - sell_c)
            gross_ret = (float(last_px) / float(entry_px)) - 1.0
            net_ret = (float(exit_equity) / float(entry_equity) - 1.0)
            hold_weeks = int(last_i - entry_i)
            trades.append(
                {
                    "entry_week_end": str(dates.iloc[entry_i].date()) if entry_i >= 0 else None,
                    "exit_week_end": str(dates.iloc[last_i].date()),
                    "entry_price_open": float(entry_px),
                    "exit_price_open": float(last_px),
                    "hold_weeks": int(hold_weeks),
                    "net_return": float(net_ret),
                    "gross_return": float(gross_ret),
                    "mae": None,
                    "mfe": None,
                    "forced_exit": True,
                }
            )
            equity[last_i] = float(exit_equity)

    # period returns（按周开盘到下周开盘）
    period_returns: list[float] = []
    for i in range(0, n - 1):
        a = float(equity[i])
        b = float(equity[i + 1])
        if a > 0 and math.isfinite(a) and math.isfinite(b):
            period_returns.append(float(b / a - 1.0))
        else:
            period_returns.append(0.0)

    # trades stats
    tr_rets = [float(t.get("net_return") or 0.0) for t in trades if isinstance(t, dict)]
    tr_wins = int(sum(1 for x in tr_rets if x > 0))

    dd = _max_drawdown(equity)
    years = None
    try:
        days = float((dates.iloc[-1] - dates.iloc[0]).days)
        years = max(0.0, days / 365.25)
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        years = None
    total_ret = float(equity[-1] / equity[0] - 1.0) if equity and equity[0] > 0 else 0.0
    cagr = None
    if years and years > 0 and equity[0] > 0:
        try:
            cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cagr = None

    vol = _safe_std(period_returns)
    mean_r = _safe_mean(period_returns)
    sharpe = None
    if vol is not None and vol > 0 and mean_r is not None:
        sharpe = float((mean_r / vol) * math.sqrt(52.0))

    stats = {
        "total_return": float(total_ret),
        "cagr": cagr,
        "max_drawdown": dd,
        "periods": int(len(period_returns)),
        "period_avg_return": float(mean_r or 0.0),
        "period_vol_ann": float((vol or 0.0) * math.sqrt(52.0)),
        "period_sharpe": sharpe,
        "trades": int(len(tr_rets)),
        "trade_win_rate": float(tr_wins / len(tr_rets)) if tr_rets else 0.0,
        "trade_avg_return": float(_safe_mean(tr_rets) or 0.0),
        "avg_mae": float(_safe_mean(maes) or 0.0) if maes else None,
        "worst_mae": float(min(maes)) if maes else None,
        "avg_mfe": float(_safe_mean(mfes) or 0.0) if mfes else None,
        "best_mfe": float(max(mfes)) if mfes else None,
    }

    return {
        "stats": stats,
        "sample": {
            "last_week_end": str(dates.iloc[-1].date()),
            "equity_last": float(equity[-1]),
            "equity_curve_tail": equity[-260:],
            "period_returns_tail": period_returns[-260:],
            "trades_tail": trades[-200:],
        },
        "trades": trades,  # 调试用；写文件前建议截断
        "equity_curve": equity,  # 调试用；写文件前建议截断
        "period_returns": period_returns,
    }


def align_regime_labels_to_weekly_opens(*, df_regime_weekly, asset_week_end_dates) -> list[str]:
    """
    把指数 regime(周线) 对齐到“资产周线每周开盘的可用标签”：
    - 对 asset 的周K（date=周最后交易日），用“严格小于该周 date 的最后一个指数周标签”
    - 这样本周交易不使用本周收盘才能知道的标签（避免未来函数）
    """
    import numpy as np
    import pandas as pd

    if df_regime_weekly is None or getattr(df_regime_weekly, "empty", True):
        return ["unknown"] * int(len(asset_week_end_dates))

    w = df_regime_weekly.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date", "label"]).sort_values("date").reset_index(drop=True)
    if w.empty:
        return ["unknown"] * int(len(asset_week_end_dates))

    w_dates = w["date"].to_numpy(dtype="datetime64[ns]")
    w_labels = w["label"].astype(str).to_numpy()

    a_dates = pd.to_datetime(asset_week_end_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(w_dates, a_dates, side="left") - 1
    out = np.where(pos >= 0, w_labels[pos], "unknown")
    return [str(x) for x in list(out)]


def race_weekly_strategies(
    *,
    df_weekly,
    df_daily,
    df_regime_weekly,
    strategies: list[WeeklyStrategyDef],
    buy_cost: float,
    sell_cost: float,
    strategy_cfg: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    多策略赛马（周线口径），并按 regime(bull/bear/neutral) 分段输出。
    """
    import pandas as pd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        raise ValueError("df_weekly 为空")

    dfw = _ensure_weekly_ohlc(df_weekly).copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)
    if dfw.empty:
        raise ValueError("周线无有效数据")

    labels_at_open = align_regime_labels_to_weekly_opens(df_regime_weekly=df_regime_weekly, asset_week_end_dates=dfw["date"])

    out: dict[str, Any] = {
        "as_of": str(dfw["date"].max().date()),
        "weeks": int(len(dfw)),
        "regime_labels_tail": labels_at_open[-20:],
        "strategies": [],
    }

    cfg2 = strategy_cfg or {}
    for spec in strategies:
        c = dict(cfg2.get(spec.key) or {})
        desired = spec.compute_desired_at_close(dfw, df_daily, c)
        sim = simulate_weekly_inout(dfw, desired_at_close=desired, buy_cost=float(buy_cost), sell_cost=float(sell_cost))

        pr = [float(x) for x in list(sim.get("period_returns") or [])]
        lb = labels_at_open[: len(pr)]
        by_regime = _breakdown_by_label(period_returns=pr, labels=lb)

        out["strategies"].append(
            {
                "key": spec.key,
                "name": spec.name,
                "description": spec.description,
                "cfg": c,
                "stats": sim.get("stats"),
                "by_regime": by_regime,
                "sample": sim.get("sample"),
            }
        )

    return out
