from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .backtest import ForwardReturnStats, forward_holding_backtest, score_forward_stats, shrunk_win_rate
from .bbb import BBBParams, compute_bbb_entry_signal


ScoreMode = Literal["win_rate", "annualized"]


@dataclass(frozen=True, slots=True)
class BBBVariant:
    key: str
    params: BBBParams


def bbb_params_from_mode(
    *,
    mode: str,
    entry_ma: int,
    dist_ma50_max: float,
    max_above_20w: float,
    min_weekly_bars_total: int,
) -> BBBParams:
    """
    统一 BBB mode -> BBBParams 的映射口径，避免 CLI/评估各写各的然后悄悄打架。
    """
    m = str(mode or "strict").strip().lower()
    if m not in {"auto", "strict", "pullback", "early"}:
        m = "strict"

    # strict: 周MACD金叉且>0 + 日MACD为多 + 位置靠均线（最保守）
    # pullback: 周MACD>0（允许周线回踩造成的“周MACD未转多”）+ 日MACD为多（更贴近右侧定方向+回踩挑位置）
    # early: 周MACD金叉（允许发生在0轴下）+ 日MACD为多（更早，但更容易吃回撤）
    req_w_bull = True
    req_w_above0 = True
    req_d_bull = True
    if m == "pullback":
        req_w_bull = False
        req_w_above0 = True
        req_d_bull = True
    elif m == "early":
        req_w_bull = True
        req_w_above0 = False
        req_d_bull = True

    return BBBParams(
        entry_ma=max(2, int(entry_ma)),
        dist_ma50_max=max(0.0, float(dist_ma50_max)),
        max_above_20w=max(0.0, float(max_above_20w)),
        min_weekly_bars_total=max(10, int(min_weekly_bars_total)),
        require_weekly_macd_bullish=bool(req_w_bull),
        require_weekly_macd_above_zero=bool(req_w_above0),
        require_daily_macd_bullish=bool(req_d_bull),
    )


def build_oat_variants(base: BBBParams, *, include_modes: bool = True) -> list[BBBVariant]:
    """
    one-at-a-time 参数扰动（±20%）做稳健性快检：
    - 不做笛卡尔积（防爆炸），只做“单参数扰动”
    - 再额外加几个模式变体（strict/pullback/early）
    """
    variants: list[BBBVariant] = [BBBVariant(key="base", params=base)]

    def add_unique(key: str, p: BBBParams):
        k2 = str(key).strip()
        if not k2:
            return
        for v in variants:
            if v.key == k2:
                return
        variants.append(BBBVariant(key=k2, params=p))

    # entry_ma：只做整数扰动
    em = int(getattr(base, "entry_ma", 50) or 50)
    for mult, tag in [(0.8, "-20%"), (1.2, "+20%")]:
        v = max(2, int(round(float(em) * float(mult))))
        if v != em:
            add_unique(
                f"entry_ma{tag}={v}",
                BBBParams(
                    entry_ma=v,
                    dist_ma50_max=float(base.dist_ma50_max),
                    max_above_20w=float(base.max_above_20w),
                    min_weekly_bars_total=int(base.min_weekly_bars_total),
                    require_weekly_macd_bullish=bool(base.require_weekly_macd_bullish),
                    require_weekly_macd_above_zero=bool(base.require_weekly_macd_above_zero),
                    require_daily_macd_bullish=bool(base.require_daily_macd_bullish),
                ),
            )

    # dist/max_above：按比例扰动
    for field, name in [("dist_ma50_max", "dist_ma_max"), ("max_above_20w", "max_above_20w")]:
        base_v = float(getattr(base, field))
        for mult, tag in [(0.8, "-20%"), (1.2, "+20%")]:
            v = max(0.0, base_v * float(mult))
            if abs(v - base_v) < 1e-12:
                continue
            kw = dict(
                entry_ma=int(base.entry_ma),
                dist_ma50_max=float(base.dist_ma50_max),
                max_above_20w=float(base.max_above_20w),
                min_weekly_bars_total=int(base.min_weekly_bars_total),
                require_weekly_macd_bullish=bool(base.require_weekly_macd_bullish),
                require_weekly_macd_above_zero=bool(base.require_weekly_macd_above_zero),
                require_daily_macd_bullish=bool(base.require_daily_macd_bullish),
            )
            kw[field] = float(v)
            add_unique(f"{name}{tag}={v:.4f}", BBBParams(**kw))

    if include_modes:
        # 模式扰动：只改 require flags，不动其它参数
        modes = [
            ("strict", True, True),
            ("pullback", False, True),
            ("early", True, False),
        ]
        for mode_name, req_bull, req_above0 in modes:
            if bool(base.require_weekly_macd_bullish) == bool(req_bull) and bool(base.require_weekly_macd_above_zero) == bool(req_above0):
                continue
            add_unique(
                f"mode={mode_name}",
                BBBParams(
                    entry_ma=int(base.entry_ma),
                    dist_ma50_max=float(base.dist_ma50_max),
                    max_above_20w=float(base.max_above_20w),
                    min_weekly_bars_total=int(base.min_weekly_bars_total),
                    require_weekly_macd_bullish=bool(req_bull),
                    require_weekly_macd_above_zero=bool(req_above0),
                    require_daily_macd_bullish=bool(base.require_daily_macd_bullish),
                ),
            )

    return variants


def _mask_signals_for_window(
    sig_all: Any,
    *,
    start_idx: int,
    end_idx: int,
    horizon_weeks: int,
) -> list[bool]:
    n = int(len(sig_all))
    s = [bool(x) for x in sig_all]
    a = max(0, int(start_idx))
    b = min(n, int(end_idx))
    h = max(1, int(horizon_weeks))

    # outside window -> False
    for i in range(0, a):
        s[i] = False
    for i in range(b, n):
        s[i] = False

    # 防未来函数：保证 trade exit 在 window 内（i+1+h < end_idx）
    last_i = b - h - 2
    if last_i < 0:
        for i in range(n):
            s[i] = False
        return s
    for i in range(last_i + 1, b):
        s[i] = False
    return s


def forward_stats_for_window(
    df_weekly,
    *,
    sig_all: Any,
    start_idx: int,
    end_idx: int,
    horizon_weeks: int,
    buy_cost: float,
    sell_cost: float,
    non_overlapping: bool,
) -> ForwardReturnStats:
    sig = _mask_signals_for_window(sig_all, start_idx=int(start_idx), end_idx=int(end_idx), horizon_weeks=int(horizon_weeks))
    st, _ = forward_holding_backtest(
        df_weekly,
        entry_signal=sig,
        horizon_weeks=int(horizon_weeks),
        buy_cost=float(buy_cost),
        sell_cost=float(sell_cost),
        non_overlapping=bool(non_overlapping),
    )
    return st


def summarize_stats(stats: ForwardReturnStats) -> dict[str, Any]:
    trades = int(getattr(stats, "trades", 0) or 0)
    wins = int(getattr(stats, "wins", 0) or 0)
    out = {
        "horizon_weeks": int(getattr(stats, "horizon_weeks", 0) or 0),
        "trades": trades,
        "net": {
            "wins": wins,
            "win_rate": float(getattr(stats, "win_rate", 0.0) or 0.0),
            "win_rate_shrunk": float(shrunk_win_rate(wins=wins, trades=trades)),
            "avg_return": float(getattr(stats, "avg_return", 0.0) or 0.0),
            "median_return": float(getattr(stats, "median_return", 0.0) or 0.0),
            "implied_ann": getattr(stats, "implied_ann", None),
        },
        "gross": {
            "wins": int(getattr(stats, "gross_wins", 0) or 0),
            "win_rate": float(getattr(stats, "gross_win_rate", 0.0) or 0.0),
            "implied_ann": getattr(stats, "gross_implied_ann", None),
        },
        "risk": {
            "avg_mae": getattr(stats, "avg_mae", None),
            "worst_mae": getattr(stats, "worst_mae", None),
            "avg_mfe": getattr(stats, "avg_mfe", None),
            "best_mfe": getattr(stats, "best_mfe", None),
        },
    }
    return out


def walk_forward_select_and_eval(
    df_weekly,
    df_daily,
    *,
    variants: list[BBBVariant],
    horizon_weeks: int,
    score_mode: ScoreMode,
    buy_cost: float,
    sell_cost: float,
    non_overlapping: bool,
    train_weeks: int,
    test_weeks: int,
    step_weeks: int,
) -> dict[str, Any]:
    """
    walk-forward（滚动训练→验证）：
    - 在训练窗内选出“最优参数”
    - 用该参数在后面的验证窗评估 OOS 表现

    注意：这里用 forward_holding_backtest 做 OOS（避免出场策略的 trade 日期过滤复杂度）。
    """
    n = int(len(df_weekly)) if df_weekly is not None else 0
    if n <= 0:
        raise ValueError("df_weekly 为空")

    h = max(1, int(horizon_weeks))
    train = max(h + 10, int(train_weeks))
    test = max(h + 10, int(test_weeks))
    step = max(1, int(step_weeks))

    # 预计算每个 variant 的 entry signal（避免 fold 内重复算）
    sig_map: dict[str, Any] = {}
    for v in variants:
        sig_map[v.key] = compute_bbb_entry_signal(df_weekly, df_daily, params=v.params)

    folds: list[dict[str, Any]] = []

    start = 0
    while True:
        train_start = int(start)
        train_end = train_start + int(train)
        test_end = train_end + int(test)
        if test_end > n:
            break

        # 每个 fold：训练选参
        best = None
        best_score = -9e18
        best_train_stats = None
        for v in variants:
            st_train = forward_stats_for_window(
                df_weekly,
                sig_all=sig_map[v.key],
                start_idx=train_start,
                end_idx=train_end,
                horizon_weeks=h,
                buy_cost=float(buy_cost),
                sell_cost=float(sell_cost),
                non_overlapping=bool(non_overlapping),
            )
            sc = float(score_forward_stats(st_train, mode=str(score_mode)))
            if sc > best_score:
                best_score = sc
                best = v
                best_train_stats = st_train

        if best is None:
            break

        # OOS：用选出来的参数在 test window 跑
        st_test = forward_stats_for_window(
            df_weekly,
            sig_all=sig_map[best.key],
            start_idx=train_end,
            end_idx=test_end,
            horizon_weeks=h,
            buy_cost=float(buy_cost),
            sell_cost=float(sell_cost),
            non_overlapping=bool(non_overlapping),
        )

        st_test_base = None
        base_key = variants[0].key if variants else "base"
        if base_key in sig_map:
            st_test_base = forward_stats_for_window(
                df_weekly,
                sig_all=sig_map[base_key],
                start_idx=train_end,
                end_idx=test_end,
                horizon_weeks=h,
                buy_cost=float(buy_cost),
                sell_cost=float(sell_cost),
                non_overlapping=bool(non_overlapping),
            )

        folds.append(
            {
                "train": {"start_idx": train_start, "end_idx": train_end, "best_key": best.key, "best_score": float(best_score), "stats": summarize_stats(best_train_stats)},
                "test": {"start_idx": train_end, "end_idx": test_end, "best_key": best.key, "stats": summarize_stats(st_test), "base_stats": summarize_stats(st_test_base) if st_test_base is not None else None},
            }
        )

        start += step

    # 汇总 OOS
    oos_trades = 0
    oos_wins = 0
    oos_sum_ret = 0.0
    base_trades = 0
    base_wins = 0
    base_sum_ret = 0.0

    for f in folds:
        t = ((f.get("test") or {}).get("stats") or {}).get("trades") or 0
        oos_trades += int(t)
        oos_wins += int((((f.get("test") or {}).get("stats") or {}).get("net") or {}).get("wins") or 0)
        avg_ret = float((((f.get("test") or {}).get("stats") or {}).get("net") or {}).get("avg_return") or 0.0)
        oos_sum_ret += float(avg_ret) * float(t)

        b = (f.get("test") or {}).get("base_stats")
        if isinstance(b, dict):
            bt = int(b.get("trades") or 0)
            base_trades += bt
            base_wins += int(((b.get("net") or {}).get("wins") or 0))
            base_sum_ret += float(((b.get("net") or {}).get("avg_return") or 0.0) * float(bt))

    oos_wr = float(oos_wins / oos_trades) if oos_trades > 0 else 0.0
    base_wr = float(base_wins / base_trades) if base_trades > 0 else 0.0
    oos_avg_ret = float(oos_sum_ret / oos_trades) if oos_trades > 0 else 0.0
    base_avg_ret = float(base_sum_ret / base_trades) if base_trades > 0 else 0.0

    # 稳定性粗评分：OOS trades 少就别装逼
    stability = 0.0
    if oos_trades > 0:
        stability = float(shrunk_win_rate(wins=oos_wins, trades=oos_trades, prior_mean=0.5, prior_strength=30.0)) * 100.0
        stability += float(oos_avg_ret) * 80.0
    return {
        "horizon_weeks": int(h),
        "train_weeks": int(train),
        "test_weeks": int(test),
        "step_weeks": int(step),
        "folds": folds,
        "oos_summary": {
            "trades": int(oos_trades),
            "wins": int(oos_wins),
            "win_rate": float(oos_wr),
            "win_rate_shrunk": float(shrunk_win_rate(wins=oos_wins, trades=oos_trades)),
            "avg_return": float(oos_avg_ret),
        },
        "base_oos_summary": {
            "trades": int(base_trades),
            "wins": int(base_wins),
            "win_rate": float(base_wr),
            "win_rate_shrunk": float(shrunk_win_rate(wins=base_wins, trades=base_trades)) if base_trades > 0 else 0.0,
            "avg_return": float(base_avg_ret),
        },
        "stability_score": float(stability),
    }
