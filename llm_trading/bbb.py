from __future__ import annotations

from dataclasses import dataclass

from .utils_stats import median

@dataclass(frozen=True, slots=True)
class BBBParams:
    # “位置优先”：离 entry_ma 太远就别上车（避免套山顶）
    # - entry_ma=50：更“稳健”，但会更晚、更容易出现“本轮无候选”
    # - entry_ma=20：更贴近“做数周波段”的位置参考
    entry_ma: int = 50
    dist_ma50_max: float = 0.12

    # 不追高：允许略高于 20W 上轨的空间（避免刚突破一点就被判“追高”）
    max_above_20w: float = 0.05

    # 周线样本最低要求：不足就别扯什么 MA50/MACD 了
    min_weekly_bars_total: int = 60

    require_weekly_macd_bullish: bool = True
    require_weekly_macd_above_zero: bool = True
    require_daily_macd_bullish: bool = True


@dataclass(frozen=True, slots=True)
class BBBExitParams:
    """
    BBB 出场风控参数（研究用途）。

    设计目标（KISS 但能抗“股灾”级别的下砸）：
    - 周线：hard（MA50 确认）负责“趋势死了就走”
    - 周线：trail（MA20 锚）负责“别把利润全吐回去”
    - 日线：soft（MACD死叉确认+破MA20）负责“更早的风控提醒”
    - 额外兜底：
      - stop_loss：最大亏损止损（可选，默认关）
      - profit_stop：盈利后回撤保护（默认开）
      - panic：大跌/深回撤快速认怂（默认开）
    """

    weekly_trail_ma: int = 20
    enable_weekly_trail: bool = True

    stop_loss_ret: float = 0.0  # 0=关闭；例如 0.08 表示亏 8% 触发（按收盘价）

    profit_stop_enabled: bool = True
    profit_stop_min_profit_ret: float = 0.20
    profit_stop_dd_pct: float = 0.12

    panic_exit_enabled: bool = True
    panic_vol_mult: float = 3.0  # 日收益 <= -max(vol20*mult, min_drop) 触发
    panic_min_drop: float = 0.04
    panic_drawdown_252d: float = 0.25  # 1年回撤 >= 25% 触发（dd<=-0.25）


def align_daily_macd_to_weekly(df_daily, df_weekly):
    """
    把日线 MACD 状态对齐到周线（以周K的 date=该周最后一个交易日为基准）。
    返回一个新的 df_weekly（追加列）：
    - daily_macd
    - daily_macd_signal
    - daily_ok (bool): daily_macd > daily_macd_signal
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    from .indicators import add_macd

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return df_weekly

    dfw = df_weekly.copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df_daily is None or getattr(df_daily, "empty", True):
        dfw["daily_macd"] = None
        dfw["daily_macd_signal"] = None
        dfw["daily_ok"] = False
        return dfw

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if "macd" not in dfd.columns or "macd_signal" not in dfd.columns:
        dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    dfd2 = dfd[["date", "macd", "macd_signal"]].rename(columns={"macd": "daily_macd", "macd_signal": "daily_macd_signal"})
    aligned = pd.merge_asof(dfw[["date"]], dfd2, on="date", direction="backward")
    dfw["daily_macd"] = aligned["daily_macd"]
    dfw["daily_macd_signal"] = aligned["daily_macd_signal"]

    dm = pd.to_numeric(dfw["daily_macd"], errors="coerce")
    ds = pd.to_numeric(dfw["daily_macd_signal"], errors="coerce")
    dfw["daily_ok"] = (dm > ds).fillna(False)
    return dfw


def compute_bbb_entry_signal(df_weekly, df_daily=None, *, params: BBBParams | None = None):
    """
    BBB 入场信号（周线定方向 + 位置 + 日线MACD择时）。

    说明：
    - 信号在周K收盘产生（对应 forward_holding_backtest 的定义）
    - 日线择时使用“该周最后一个交易日”的 MACD 状态对齐到周K
    """
    params2 = params or BBBParams()

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    from .indicators import add_donchian_channels, add_macd, add_moving_averages

    if df_weekly is None or getattr(df_weekly, "empty", True):
        return pd.Series([], dtype=bool)

    dfw = df_weekly.copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if int(len(dfw)) < int(params2.min_weekly_bars_total):
        return pd.Series([False] * len(dfw), index=dfw.index, dtype=bool)

    if "ma50" not in dfw.columns:
        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)

    # 20W 上轨：用 20 周唐奇安上轨（shift=1，避免未来函数）
    if "donchian_entry_upper" not in dfw.columns or "donchian_entry_lower" not in dfw.columns:
        dfw = add_donchian_channels(
            dfw,
            window=20,
            upper_col="donchian_entry_upper",
            lower_col="donchian_entry_lower",
            shift=1,
        )

    if "macd" not in dfw.columns or "macd_signal" not in dfw.columns:
        dfw = add_macd(dfw, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    dfw = align_daily_macd_to_weekly(df_daily, dfw)

    close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    # entry_ma：用于“位置优先”的入场参考线（默认 MA50；可调到 MA20 更贴近数周波段）
    entry_ma = max(2, int(getattr(params2, "entry_ma", 50) or 50))
    entry_col = f"ma{entry_ma}"
    if entry_col not in dfw.columns:
        dfw = dfw.copy()
        dfw[entry_col] = dfw["close"].rolling(window=entry_ma, min_periods=entry_ma).mean()
    ma_entry = pd.to_numeric(dfw[entry_col], errors="coerce").astype(float)
    upper = pd.to_numeric(dfw["donchian_entry_upper"], errors="coerce").astype(float)
    macd = pd.to_numeric(dfw["macd"], errors="coerce").astype(float)
    macd_sig = pd.to_numeric(dfw["macd_signal"], errors="coerce").astype(float)

    # 位置：不许离 entry_ma 太远（避免追高 / 套山顶）
    dist = (close - ma_entry).abs() / ma_entry.replace({0.0: float("nan")})
    ok_dist = (dist <= float(params2.dist_ma50_max)).fillna(False)

    # 不追高：不许明显高于 20W 上轨
    ok_not_chasing = (close <= upper * (1.0 + float(params2.max_above_20w))).fillna(False)

    ok_weekly_macd = pd.Series([True] * len(dfw), index=dfw.index, dtype=bool)
    if params2.require_weekly_macd_bullish:
        ok_weekly_macd = ok_weekly_macd & (macd > macd_sig).fillna(False)
    if params2.require_weekly_macd_above_zero:
        ok_weekly_macd = ok_weekly_macd & (macd > 0.0).fillna(False)

    ok_daily = pd.Series([True] * len(dfw), index=dfw.index, dtype=bool)
    if params2.require_daily_macd_bullish:
        ok_daily = ok_daily & dfw.get("daily_ok", False).fillna(False)

    sig = (ok_dist & ok_not_chasing & ok_weekly_macd & ok_daily).fillna(False)
    return sig.astype(bool)


@dataclass(frozen=True, slots=True)
class BBBExitBacktestStats:
    trades: int
    wins: int
    win_rate: float
    win_rate_shrunk: float
    avg_return: float
    median_return: float
    gross_wins: int
    gross_win_rate: float
    gross_win_rate_shrunk: float
    gross_avg_return: float
    gross_median_return: float
    avg_hold_days: float
    median_hold_days: float
    avg_mae: float | None
    worst_mae: float | None
    avg_mfe: float | None
    best_mfe: float | None
    exits_soft: int
    exits_hard: int
    exits_trail: int
    exits_stop_loss: int
    exits_profit_stop: int
    exits_panic: int


def _ensure_ohlc(df):
    df2 = df
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


def bbb_exit_backtest(
    df_weekly,
    df_daily,
    *,
    params: BBBParams | None = None,
    exit_params: BBBExitParams | None = None,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    min_hold_days: int = 5,
    cooldown_days: int = 0,
    include_samples: bool = False,
) -> tuple[BBBExitBacktestStats, dict]:
    """
    BBB 出场规则的“闭环回测”（多周期）：
    - 入场：周线 BBB 信号在周收盘触发 -> 下一交易日开盘买入
    - 出场（取最先触发者）：
      - panic：大跌/深回撤兜底（更快认怂） -> 下一交易日开盘卖出
      - stop_loss：最大亏损止损（可选，按收盘触发） -> 下一交易日开盘卖出
      - hard：周线 close<MA50 连续2周确认 -> 下一交易日开盘卖出
      - trail：周线 close<MA(trail_ma) -> 下一交易日开盘卖出
      - profit_stop：盈利后回撤保护（max_close_since_entry*(1-dd)） -> 下一交易日开盘卖出
      - soft：日线 MACD 死叉连续2天 且 close<MA20 -> 下一交易日开盘卖出

    注意：这是研究工具统计，不构成投资建议。
    """
    params2 = params or BBBParams()
    ep = exit_params or BBBExitParams()

    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .backtest import shrunk_win_rate
    from .indicators import add_macd, add_moving_averages

    if df_weekly is None or getattr(df_weekly, "empty", True):
        raise ValueError("df_weekly 为空")
    if df_daily is None or getattr(df_daily, "empty", True):
        raise ValueError("df_daily 为空")

    min_hold_days2 = max(0, int(min_hold_days))
    cooldown_days2 = max(0, int(cooldown_days))

    # exit 参数裁剪（别让用户传个负数把回测搞炸）
    trail_ma = max(2, int(getattr(ep, "weekly_trail_ma", 20) or 20))
    enable_trail = bool(getattr(ep, "enable_weekly_trail", True))

    stop_loss_ret = float(getattr(ep, "stop_loss_ret", 0.0) or 0.0)
    stop_loss_ret = max(0.0, min(stop_loss_ret, 0.80))

    profit_stop_enabled = bool(getattr(ep, "profit_stop_enabled", True))
    profit_min_ret = float(getattr(ep, "profit_stop_min_profit_ret", 0.20) or 0.20)
    profit_min_ret = max(0.0, min(profit_min_ret, 5.0))
    profit_dd = float(getattr(ep, "profit_stop_dd_pct", 0.12) or 0.12)
    profit_dd = max(0.0, min(profit_dd, 0.50))

    panic_enabled = bool(getattr(ep, "panic_exit_enabled", True))
    panic_vol_mult = float(getattr(ep, "panic_vol_mult", 3.0) or 3.0)
    panic_vol_mult = max(0.0, min(panic_vol_mult, 20.0))
    panic_min_drop = float(getattr(ep, "panic_min_drop", 0.04) or 0.04)
    panic_min_drop = max(0.0, min(panic_min_drop, 0.50))
    panic_dd = float(getattr(ep, "panic_drawdown_252d", 0.25) or 0.25)
    panic_dd = max(0.0, min(panic_dd, 0.80))

    dfw = df_weekly.copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dfw = _ensure_ohlc(dfw)

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dfd = _ensure_ohlc(dfd)

    if "ma50" not in dfw.columns:
        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
    if "macd" not in dfw.columns or "macd_signal" not in dfw.columns:
        dfw = add_macd(dfw, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    # entry: 周线 BBB 信号
    entry_sig_w = compute_bbb_entry_signal(dfw, dfd, params=params2).astype(bool)
    if len(entry_sig_w) != len(dfw):
        raise ValueError("entry_sig 长度不匹配")

    # hard exit: 周线连续两周跌破 MA50
    close_w = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    ma50_w = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
    hard_w = (close_w < ma50_w) & (close_w.shift(1) < ma50_w.shift(1))
    hard_w = hard_w.fillna(False).astype(bool)

    # trail exit: 周线跌破“锚线”（默认 MA20，更早保护利润；也更容易被震荡抖出去）
    trail_w = pd.Series([False] * len(dfw), index=dfw.index, dtype=bool)
    if bool(enable_trail):
        try:
            ma_anchor = close_w.rolling(window=int(trail_ma), min_periods=int(trail_ma)).mean()
            trail_w = ((close_w < ma_anchor) & ma_anchor.notna()).fillna(False).astype(bool)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            trail_w = pd.Series([False] * len(dfw), index=dfw.index, dtype=bool)

    # daily soft exit: 2日死叉确认 + 跌破 MA20
    if "ma20" not in dfd.columns:
        dfd = dfd.copy()
        dfd["ma20"] = dfd["close"].astype(float).rolling(window=20, min_periods=20).mean()
    if "macd" not in dfd.columns or "macd_signal" not in dfd.columns:
        dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    close_d = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
    ma20_d = pd.to_numeric(dfd["ma20"], errors="coerce").astype(float)
    macd_d = pd.to_numeric(dfd["macd"], errors="coerce").astype(float)
    sig_d = pd.to_numeric(dfd["macd_signal"], errors="coerce").astype(float)

    bearish = (macd_d < sig_d)
    bearish2 = bearish & bearish.shift(1, fill_value=False)
    below_ma20 = (close_d < ma20_d)
    soft_d = (bearish2 & below_ma20).fillna(False).astype(bool)

    # panic exit：大跌/深回撤兜底（更像 07/15 那种“砍人行情”）
    panic_d = pd.Series([False] * len(dfd), index=dfd.index, dtype=bool)
    if bool(panic_enabled):
        try:
            prev_close = close_d.shift(1).replace({0.0: float("nan")})
            r1 = (close_d / prev_close) - 1.0
            vol20 = r1.rolling(window=20, min_periods=20).std()
            roll_max = close_d.rolling(window=252, min_periods=20).max().replace({0.0: float("nan")})
            dd252 = (close_d / roll_max) - 1.0
            thresh = pd.concat([vol20 * float(panic_vol_mult), pd.Series([float(panic_min_drop)] * len(dfd), index=dfd.index)], axis=1).max(axis=1)
            panic_d = ((r1 <= -thresh) | (dd252 <= -float(panic_dd))).fillna(False).astype(bool)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            panic_d = pd.Series([False] * len(dfd), index=dfd.index, dtype=bool)

    # 对齐：把 weekly 的信号日期映射到 daily 索引（周K date=该周最后交易日）
    dt_d = dfd["date"].to_numpy(dtype="datetime64[ns]")
    dt_w = dfw["date"].to_numpy(dtype="datetime64[ns]")
    end_pos = np.searchsorted(dt_d, dt_w, side="right") - 1

    n_d = int(len(dfd))
    entry_flag = np.zeros(n_d, dtype=bool)
    hard_flag = np.zeros(n_d, dtype=bool)
    trail_flag = np.zeros(n_d, dtype=bool)

    for i in range(int(len(dfw))):
        p = int(end_pos[i])
        if p < 0 or p >= n_d:
            continue
        if bool(hard_w.iloc[i]):
            hard_flag[p] = True
        if bool(enable_trail) and bool(trail_w.iloc[i]):
            trail_flag[p] = True
        if bool(entry_sig_w.iloc[i]):
            e = p + 1
            if 0 <= e < n_d:
                entry_flag[e] = True

    open_px = pd.to_numeric(dfd["open"], errors="coerce").astype(float).to_numpy()
    high_px = pd.to_numeric(dfd["high"], errors="coerce").astype(float).to_numpy()
    low_px = pd.to_numeric(dfd["low"], errors="coerce").astype(float).to_numpy()
    close_px = close_d.to_numpy(dtype=float)
    soft_flag = soft_d.to_numpy(dtype=bool)
    panic_flag = panic_d.to_numpy(dtype=bool)

    rets: list[float] = []
    rets_gross: list[float] = []
    holds: list[float] = []
    maes: list[float] = []
    mfes: list[float] = []
    wins = 0
    wins_gross = 0
    exits_soft = 0
    exits_hard = 0
    exits_trail = 0
    exits_stop_loss = 0
    exits_profit_stop = 0
    exits_panic = 0

    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    next_allowed = 0
    max_close_since_entry = 0.0
    profit_stop_active = False

    # 按日扫描（exit 在 t+1 开盘执行）
    for t in range(0, n_d - 1):
        if (not in_pos) and entry_flag[t] and t >= next_allowed:
            px = float(open_px[t])
            if px > 0:
                in_pos = True
                entry_idx = int(t)
                entry_price = px
                max_close_since_entry = float(px)
                profit_stop_active = False

        if not in_pos:
            continue

        # 更新“进场以来最高收盘”（用于回撤止盈）
        try:
            c = float(close_px[t])
            if c > 0:
                if c > float(max_close_since_entry):
                    max_close_since_entry = float(c)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            pass

        # 回撤止盈“启动条件”：只要曾经达到过目标浮盈，就一直保持启用（别反复开关）
        if (not profit_stop_active) and bool(profit_stop_enabled) and entry_price > 0 and max_close_since_entry >= entry_price * (1.0 + float(profit_min_ret)):
            profit_stop_active = True

        # 优先级：panic > stop_loss > hard > trail > profit_stop > soft
        exit_reason = None
        if bool(panic_enabled) and bool(panic_flag[t]):
            exit_reason = "panic"
        elif stop_loss_ret > 0 and entry_price > 0:
            try:
                if float(close_px[t]) > 0 and float(close_px[t]) <= entry_price * (1.0 - float(stop_loss_ret)):
                    exit_reason = "stop_loss"
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pass
        if (exit_reason is None) and bool(hard_flag[t]):
            exit_reason = "hard"
        if (exit_reason is None) and bool(enable_trail) and bool(trail_flag[t]):
            exit_reason = "trail"
        if (exit_reason is None) and profit_stop_active and bool(profit_stop_enabled) and max_close_since_entry > 0:
            try:
                profit_stop = float(max_close_since_entry) * (1.0 - float(profit_dd))
                if float(close_px[t]) > 0 and float(close_px[t]) <= float(profit_stop):
                    exit_reason = "profit_stop"
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pass
        if (exit_reason is None) and bool(soft_flag[t]) and (t - entry_idx + 1) >= min_hold_days2:
            exit_reason = "soft"

        if not exit_reason:
            continue

        exit_idx = int(t + 1)
        exit_price = float(open_px[exit_idx])
        if exit_price <= 0 or entry_price <= 0:
            in_pos = False
            next_allowed = exit_idx + cooldown_days2
            continue

        gross_ret = (exit_price / entry_price) - 1.0
        net_ret = (exit_price * (1.0 - float(sell_cost))) / (entry_price * (1.0 + float(buy_cost))) - 1.0
        rets.append(float(net_ret))
        rets_gross.append(float(gross_ret))
        if net_ret > 0:
            wins += 1
        if gross_ret > 0:
            wins_gross += 1

        hold_days = exit_idx - entry_idx
        holds.append(float(hold_days))

        span_low = low_px[entry_idx:exit_idx]
        span_high = high_px[entry_idx:exit_idx]
        if span_low.size > 0:
            try:
                min_low = float(np.nanmin(span_low))
                if min_low > 0:
                    maes.append(min_low / entry_price - 1.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pass
        if span_high.size > 0:
            try:
                max_high = float(np.nanmax(span_high))
                if max_high > 0:
                    mfes.append(max_high / entry_price - 1.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pass

        if exit_reason == "soft":
            exits_soft += 1
        elif exit_reason == "hard":
            exits_hard += 1
        elif exit_reason == "trail":
            exits_trail += 1
        elif exit_reason == "stop_loss":
            exits_stop_loss += 1
        elif exit_reason == "profit_stop":
            exits_profit_stop += 1
        else:
            exits_panic += 1

        in_pos = False
        next_allowed = exit_idx + cooldown_days2

    trades = len(rets)
    if trades <= 0:
        stats = BBBExitBacktestStats(
            trades=0,
            wins=0,
            win_rate=0.0,
            win_rate_shrunk=0.0,
            avg_return=0.0,
            median_return=0.0,
            gross_wins=0,
            gross_win_rate=0.0,
            gross_win_rate_shrunk=0.0,
            gross_avg_return=0.0,
            gross_median_return=0.0,
            avg_hold_days=0.0,
            median_hold_days=0.0,
            avg_mae=None,
            worst_mae=None,
            avg_mfe=None,
            best_mfe=None,
            exits_soft=int(exits_soft),
            exits_hard=int(exits_hard),
            exits_trail=int(exits_trail),
            exits_stop_loss=int(exits_stop_loss),
            exits_profit_stop=int(exits_profit_stop),
            exits_panic=int(exits_panic),
        )
        return stats, {
            "returns": [],
            "hold_days": [],
            "last_date": str(dfd["date"].iloc[-1]),
            "exits": {
                "soft": int(exits_soft),
                "hard": int(exits_hard),
                "trail": int(exits_trail),
                "stop_loss": int(exits_stop_loss),
                "profit_stop": int(exits_profit_stop),
                "panic": int(exits_panic),
            },
        }

    avg_ret = float(sum(rets) / trades)
    avg_ret_g = float(sum(rets_gross) / trades) if rets_gross else 0.0
    med_ret = float(median(rets))
    med_ret_g = float(median(rets_gross)) if rets_gross else 0.0
    win_rate = float(wins / trades) if trades > 0 else 0.0
    win_rate_s = float(shrunk_win_rate(wins=wins, trades=trades))
    win_rate_g = float(wins_gross / trades) if trades > 0 else 0.0
    win_rate_gs = float(shrunk_win_rate(wins=wins_gross, trades=trades))

    avg_hold = float(sum(holds) / len(holds)) if holds else 0.0
    med_hold = float(median(holds)) if holds else 0.0

    avg_mae = float(sum(maes) / len(maes)) if maes else None
    worst_mae = float(min(maes)) if maes else None
    avg_mfe = float(sum(mfes) / len(mfes)) if mfes else None
    best_mfe = float(max(mfes)) if mfes else None

    stats = BBBExitBacktestStats(
        trades=int(trades),
        wins=int(wins),
        win_rate=float(win_rate),
        win_rate_shrunk=float(win_rate_s),
        avg_return=float(avg_ret),
        median_return=float(med_ret),
        gross_wins=int(wins_gross),
        gross_win_rate=float(win_rate_g),
        gross_win_rate_shrunk=float(win_rate_gs),
        gross_avg_return=float(avg_ret_g),
        gross_median_return=float(med_ret_g),
        avg_hold_days=float(avg_hold),
        median_hold_days=float(med_hold),
        avg_mae=avg_mae,
        worst_mae=worst_mae,
        avg_mfe=avg_mfe,
        best_mfe=best_mfe,
        exits_soft=int(exits_soft),
        exits_hard=int(exits_hard),
        exits_trail=int(exits_trail),
        exits_stop_loss=int(exits_stop_loss),
        exits_profit_stop=int(exits_profit_stop),
        exits_panic=int(exits_panic),
    )

    sample: dict = {
        "returns": rets[-200:],
        "gross_returns": rets_gross[-200:],
        "hold_days": holds[-200:],
        "last_date": str(dfd["date"].iloc[-1].date()) if hasattr(dfd["date"].iloc[-1], "date") else str(dfd["date"].iloc[-1]),
        "exits": {
            "soft": int(exits_soft),
            "hard": int(exits_hard),
            "trail": int(exits_trail),
            "stop_loss": int(exits_stop_loss),
            "profit_stop": int(exits_profit_stop),
            "panic": int(exits_panic),
        },
        "min_hold_days": int(min_hold_days2),
        "cooldown_days": int(cooldown_days2),
        "exit_params": {
            "weekly_trail_ma": int(trail_ma),
            "enable_weekly_trail": bool(enable_trail),
            "stop_loss_ret": float(stop_loss_ret),
            "profit_stop_enabled": bool(profit_stop_enabled),
            "profit_stop_min_profit_ret": float(profit_min_ret),
            "profit_stop_dd_pct": float(profit_dd),
            "panic_exit_enabled": bool(panic_enabled),
            "panic_vol_mult": float(panic_vol_mult),
            "panic_min_drop": float(panic_min_drop),
            "panic_drawdown_252d": float(panic_dd),
        },
    }
    if include_samples:
        sample["mae"] = maes[-200:]
        sample["mfe"] = mfes[-200:]

    return stats, sample
