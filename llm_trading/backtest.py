from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .utils_stats import median

@dataclass(frozen=True)
class ForwardReturnStats:
    horizon_weeks: int
    trades: int
    wins: int
    win_rate: float
    avg_return: float
    median_return: float
    avg_log_return: float | None
    implied_ann: float | None
    gross_wins: int
    gross_win_rate: float
    gross_avg_return: float
    gross_median_return: float
    gross_avg_log_return: float | None
    gross_implied_ann: float | None
    avg_mae: float | None
    worst_mae: float | None
    avg_mfe: float | None
    best_mfe: float | None


def shrunk_win_rate(*, wins: int, trades: int, prior_mean: float = 0.5, prior_strength: float = 20.0) -> float:
    """
    Beta-Binomial 收缩胜率（用于防小样本吹牛逼）。
    - prior_mean: 先验胜率均值
    - prior_strength: 先验“等效样本数”（越大越保守）
    """
    n = int(trades)
    w = int(wins)
    if n <= 0:
        return 0.0
    pm = float(prior_mean)
    ps = float(prior_strength)
    if ps <= 0:
        return float(w / n)
    if pm <= 0:
        pm = 0.0
    if pm >= 1:
        pm = 1.0
    a = pm * ps
    b = (1.0 - pm) * ps
    return float((w + a) / (n + a + b))


def _implied_ann_from_avg_log(*, avg_log_return: float | None, horizon_weeks: int) -> float | None:
    if avg_log_return is None:
        return None
    try:
        h = int(horizon_weeks)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if h <= 0:
        return None
    try:
        x = float(avg_log_return) * (52.0 / float(h))
        if not math.isfinite(x):
            return None
        return float(math.exp(x) - 1.0)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None


def _sample_weight_from_trades(*, trades: int, prior_strength: float = 20.0) -> float:
    """
    小样本权重（0~1）：trades 越少，越别让它“吹牛逼”。
    - 形式：n / (n + prior_strength)
    - prior_strength 越大越保守
    """
    n = int(trades or 0)
    if n <= 0:
        return 0.0
    ps = float(prior_strength or 0.0)
    if ps <= 0:
        return 1.0
    return float(n / (float(n) + float(ps)))


def score_forward_stats(stats: ForwardReturnStats | None, *, mode: str = "win_rate") -> float:
    """
    给 ForwardReturnStats 打一个“排序用分数”（经验分）。
    - mode=win_rate：胜率优先（默认，适合“胜率优先+别被磨损搞死”）
    - mode=annualized：年化优先（适合“小资金+磨损大+最大年化”）
    """
    if stats is None or int(getattr(stats, "trades", 0) or 0) <= 0:
        return 0.0

    m = str(mode or "win_rate").strip().lower()

    mae_penalty = 0.0
    if stats.avg_mae is not None:
        mae_penalty = abs(float(stats.avg_mae)) * 100.0

    # 用“收缩胜率”替代原始胜率：trades 少的别瞎排前面
    wr = shrunk_win_rate(wins=int(stats.wins), trades=int(stats.trades), prior_mean=0.5, prior_strength=20.0)
    # 小样本权重：只对“收益类项”做缩放，避免 annualized 因 trades=1 直接霸榜
    sw = _sample_weight_from_trades(trades=int(stats.trades), prior_strength=20.0)

    if m in {"annualized", "ann", "cagr"}:
        ann = stats.implied_ann
        if ann is None:
            # fallback：用 avg_return 近似（不如 avg_log 稳，但至少能跑）
            try:
                h = int(stats.horizon_weeks)
                if h > 0 and float(stats.avg_return) > -0.99:
                    ann = float((1.0 + float(stats.avg_return)) ** (52.0 / float(h)) - 1.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                ann = None
        ann2 = float(ann) if ann is not None else 0.0
        # 年化封顶：ETF/指数这种东西，动不动几百%年化基本都是“小样本幻觉”
        # - 目的：防止 score_mode=annualized 被 trades=1 的极端值直接打穿
        try:
            if not math.isfinite(ann2):
                ann2 = 0.0
            ann2 = max(-0.99, min(float(ann2), 2.0))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            ann2 = 0.0
        # 年化优先：收益>胜率（但保留一点胜率/样本奖励 + 回撤惩罚）
        return (
            float(ann2) * 120.0 * float(sw)
            + float(wr) * 20.0
            + float(stats.avg_return) * 10.0 * float(sw)
            - mae_penalty * 0.8
            + math.log(float(stats.trades) + 1.0) * 2.0
        )

    # 默认：胜率优先（更稳健）
    return float(wr) * 100.0 + float(stats.avg_return) * 80.0 * float(sw) - mae_penalty * 0.6 + math.log(float(stats.trades) + 1.0) * 3.0


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
            avg_log_return=None,
            implied_ann=None,
            gross_wins=0,
            gross_win_rate=0.0,
            gross_avg_return=0.0,
            gross_median_return=0.0,
            gross_avg_log_return=None,
            gross_implied_ann=None,
            avg_mae=None,
            worst_mae=None,
            avg_mfe=None,
            best_mfe=None,
        )
        return stats, {"returns": [], "gross_returns": [], "mae": [], "mfe": []}

    open_px = df_weekly["open"].astype(float).to_numpy()
    high_px = df_weekly["high"].astype(float).to_numpy()
    low_px = df_weekly["low"].astype(float).to_numpy()
    dates = df_weekly["date"]

    sig = np.asarray(entry_signal, dtype=bool)
    if sig.shape[0] != n:
        raise ValueError("entry_signal 长度与 df_weekly 不一致")

    rets: list[float] = []
    rets_gross: list[float] = []
    maes: list[float] = []
    mfes: list[float] = []
    wins = 0
    wins_gross = 0
    sum_log = 0.0
    sum_log_gross = 0.0

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

        gross_ret = (exit_ / entry) - 1.0
        net_ret = (exit_ * (1.0 - float(sell_cost))) / (entry * (1.0 + float(buy_cost))) - 1.0
        rets.append(float(net_ret))
        rets_gross.append(float(gross_ret))
        if net_ret > 0:
            wins += 1
        if gross_ret > 0:
            wins_gross += 1
        try:
            sum_log += float(math.log1p(float(net_ret)))
        except ValueError:
            # 极端情况（-100%）别炸
            sum_log += float("-inf")
        try:
            sum_log_gross += float(math.log1p(float(gross_ret)))
        except ValueError:
            sum_log_gross += float("-inf")

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
            avg_log_return=None,
            implied_ann=None,
            gross_wins=0,
            gross_win_rate=0.0,
            gross_avg_return=0.0,
            gross_median_return=0.0,
            gross_avg_log_return=None,
            gross_implied_ann=None,
            avg_mae=None,
            worst_mae=None,
            avg_mfe=None,
            best_mfe=None,
        )
        return stats, {"returns": [], "gross_returns": [], "mae": [], "mfe": []}

    avg_ret = float(sum(rets) / trades)
    avg_ret_g = float(sum(rets_gross) / trades) if rets_gross else 0.0
    med_ret = float(median(rets))
    med_ret_g = float(median(rets_gross)) if rets_gross else 0.0
    win_rate = float(wins / trades) if trades > 0 else 0.0
    win_rate_g = float(wins_gross / trades) if trades > 0 else 0.0
    avg_log_ret = None
    try:
        avg_log_ret = float(sum_log / float(trades)) if trades > 0 and math.isfinite(float(sum_log)) else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        avg_log_ret = None
    implied_ann = _implied_ann_from_avg_log(avg_log_return=avg_log_ret, horizon_weeks=int(horizon_weeks))

    avg_log_ret_g = None
    try:
        avg_log_ret_g = float(sum_log_gross / float(trades)) if trades > 0 and math.isfinite(float(sum_log_gross)) else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        avg_log_ret_g = None
    implied_ann_g = _implied_ann_from_avg_log(avg_log_return=avg_log_ret_g, horizon_weeks=int(horizon_weeks))

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
        avg_log_return=avg_log_ret,
        implied_ann=implied_ann,
        gross_wins=int(wins_gross),
        gross_win_rate=win_rate_g,
        gross_avg_return=avg_ret_g,
        gross_median_return=med_ret_g,
        gross_avg_log_return=avg_log_ret_g,
        gross_implied_ann=implied_ann_g,
        avg_mae=avg_mae,
        worst_mae=worst_mae,
        avg_mfe=avg_mfe,
        best_mfe=best_mfe,
    )

    # 这里只回一些分布摘要，别一股脑塞几十万条 trade 明细进 JSON（你机器先炸）
    sample = {
        "returns": rets[-200:],
        "gross_returns": rets_gross[-200:],
        "mae": maes[-200:],
        "mfe": mfes[-200:],
        "last_date": str(dates.iloc[-1].date()) if hasattr(dates.iloc[-1], "date") else str(dates.iloc[-1]),
    }
    return stats, sample
