from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForwardReturnStats:
    horizon_weeks: int
    trades: int
    wins: int
    win_rate: float
    avg_return: float
    median_return: float
    avg_mae: float | None
    worst_mae: float | None
    avg_mfe: float | None
    best_mfe: float | None


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return float(xs2[mid])
    return float((xs2[mid - 1] + xs2[mid]) / 2.0)


def forward_holding_backtest(
    df_weekly,
    *,
    entry_signal,
    horizon_weeks: int,
    buy_cost: float,
    sell_cost: float,
    non_overlapping: bool = True,
) -> tuple[ForwardReturnStats, dict[str, Any]]:
    """
    前向持有回测（适合“买入信号参考”的胜率统计）：
    - 信号在第 i 根周K 收盘产生
    - 买入在第 i+1 根周K 开盘
    - 持有 horizon_weeks 周
    - 卖出在第 i+1+horizon_weeks 根周K 开盘

    注意：这是“信号->未来收益分布”统计，不等于完整策略回测，更不构成投资建议。
    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 numpy？先跑：pip install -r \"requirements.txt\"") from exc

    if df_weekly is None or getattr(df_weekly, "empty", True):
        raise ValueError("df_weekly 为空，别闹。")
    if horizon_weeks <= 0:
        raise ValueError("horizon_weeks 必须 > 0")

    need = {"date", "open", "high", "low", "close"}
    if not need.issubset(set(df_weekly.columns)):
        raise ValueError(f"df_weekly 缺列：需要 {sorted(need)}，实际 {sorted(df_weekly.columns)}")

    n = int(len(df_weekly))
    if n < horizon_weeks + 3:
        stats = ForwardReturnStats(
            horizon_weeks=horizon_weeks,
            trades=0,
            wins=0,
            win_rate=0.0,
            avg_return=0.0,
            median_return=0.0,
            avg_mae=None,
            worst_mae=None,
            avg_mfe=None,
            best_mfe=None,
        )
        return stats, {"returns": [], "mae": [], "mfe": []}

    open_px = df_weekly["open"].astype(float).to_numpy()
    high_px = df_weekly["high"].astype(float).to_numpy()
    low_px = df_weekly["low"].astype(float).to_numpy()
    dates = df_weekly["date"]

    sig = np.asarray(entry_signal, dtype=bool)
    if sig.shape[0] != n:
        raise ValueError("entry_signal 长度与 df_weekly 不一致")

    rets: list[float] = []
    maes: list[float] = []
    mfes: list[float] = []
    wins = 0

    i = 0
    # i 是“信号发生在第 i 根收盘”
    last_i = n - horizon_weeks - 2  # 需要 i+1+horizon < n
    while i <= last_i:
        if not bool(sig[i]):
            i += 1
            continue

        entry_idx = i + 1
        exit_idx = entry_idx + horizon_weeks
        if exit_idx >= n:
            break

        entry = float(open_px[entry_idx])
        exit_ = float(open_px[exit_idx])
        if entry <= 0 or exit_ <= 0:
            i += 1
            continue

        net_ret = (exit_ * (1.0 - float(sell_cost))) / (entry * (1.0 + float(buy_cost))) - 1.0
        rets.append(float(net_ret))
        if net_ret > 0:
            wins += 1

        # 磨损/有利波动：用持仓期间的 low/high 估个 MAE/MFE（到 exit 开盘为止，不包含 exit 那周）
        span_low = low_px[entry_idx:exit_idx]
        span_high = high_px[entry_idx:exit_idx]
        if span_low.size > 0 and span_high.size > 0:
            min_low = float(np.nanmin(span_low))
            max_high = float(np.nanmax(span_high))
            if min_low > 0:
                maes.append(min_low / entry - 1.0)
            if max_high > 0:
                mfes.append(max_high / entry - 1.0)

        if non_overlapping:
            i = exit_idx
        else:
            i += 1

    trades = len(rets)
    if trades == 0:
        stats = ForwardReturnStats(
            horizon_weeks=horizon_weeks,
            trades=0,
            wins=0,
            win_rate=0.0,
            avg_return=0.0,
            median_return=0.0,
            avg_mae=None,
            worst_mae=None,
            avg_mfe=None,
            best_mfe=None,
        )
        return stats, {"returns": [], "mae": [], "mfe": []}

    avg_ret = float(sum(rets) / trades)
    med_ret = float(_median(rets))
    win_rate = float(wins / trades) if trades > 0 else 0.0

    avg_mae = float(sum(maes) / len(maes)) if maes else None
    worst_mae = float(min(maes)) if maes else None
    avg_mfe = float(sum(mfes) / len(mfes)) if mfes else None
    best_mfe = float(max(mfes)) if mfes else None

    stats = ForwardReturnStats(
        horizon_weeks=horizon_weeks,
        trades=trades,
        wins=int(wins),
        win_rate=win_rate,
        avg_return=avg_ret,
        median_return=med_ret,
        avg_mae=avg_mae,
        worst_mae=worst_mae,
        avg_mfe=avg_mfe,
        best_mfe=best_mfe,
    )

    # 这里只回一些分布摘要，别一股脑塞几十万条 trade 明细进 JSON（你机器先炸）
    sample = {
        "returns": rets[-200:],
        "mae": maes[-200:],
        "mfe": mfes[-200:],
        "last_date": str(dates.iloc[-1].date()) if hasattr(dates.iloc[-1], "date") else str(dates.iloc[-1]),
    }
    return stats, sample

