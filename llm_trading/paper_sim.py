from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import math
from typing import Any, Literal

from .costs import TradeCost, bps_to_rate, calc_shares_for_capital, cash_buy, cash_sell, estimate_slippage_bps, trade_cost_from_params
from .utils_stats import median
from .utils_time import parse_date_any_opt

PaperStrategy = Literal["bbb_etf", "bbb_stock", "rot_stock_weekly"]
PaperMode = Literal["single", "portfolio"]

def _date_str(x) -> str:
    if hasattr(x, "date"):
        try:
            return str(x.date())
        except (AttributeError):  # noqa: BLE001
            pass
    if hasattr(x, "strftime"):
        try:
            return x.strftime("%Y-%m-%d")
        except (AttributeError):  # noqa: BLE001
            pass
    return str(x)


def _floor_to_lot(shares: int, lot: int) -> int:
    lot2 = max(1, int(lot))
    n = int(shares)
    if n <= 0:
        return 0
    return (n // lot2) * lot2


def simulate_bbb_paper(
    df_daily,
    *,
    symbol: str,
    asset: Literal["etf", "stock"] = "etf",
    start_date: str | None = None,
    end_date: str | None = None,
    capital_yuan: float = 3000.0,
    roundtrip_cost_yuan: float = 10.0,
    min_fee_yuan: float = 0.0,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    lot_size: int = 100,
    max_trades: int = 300,
    bbb_entry_gap_max: float = 0.015,
    bbb_entry_ma: int = 20,
    bbb_dist_ma_max: float = 0.12,
    bbb_max_above_20w: float = 0.05,
    bbb_min_weeks: int = 60,
    bbb_require_weekly_macd_bullish: bool = True,
    bbb_require_weekly_macd_above_zero: bool = True,
    bbb_require_daily_macd_bullish: bool = True,
    bbb_min_hold_days: int = 5,
    bbb_cooldown_days: int = 0,
) -> dict[str, Any]:
    """
    BBB（ETF/股票）模拟盘（按信号自动生成“虚拟成交”）。

    约定（与 bbb_exit_backtest 一致）：
    - 周线信号在周收盘产生 -> 下一交易日开盘买入
    - soft/hard 触发 -> 下一交易日开盘卖出
    """
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .bbb import BBBParams, compute_bbb_entry_signal
    from .indicators import add_macd, add_moving_averages
    from .resample import resample_to_weekly

    asset2 = str(asset or "etf").strip().lower()
    if asset2 not in {"etf", "stock"}:
        asset2 = "etf"

    if df_daily is None or getattr(df_daily, "empty", True):
        return {"ok": False, "error": "无数据", "symbol": symbol, "asset": asset2}

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if dfd.empty or len(dfd) < 60:
        return {"ok": False, "error": "K线太少", "symbol": symbol, "asset": asset2}

    # end_date 过滤（start_date 不截断，避免指标不足；只用于禁止入场）
    end_dt = parse_date_any_opt(end_date)
    if end_dt is not None:
        dfd = dfd[dfd["date"] <= end_dt].reset_index(drop=True)
    if dfd.empty or len(dfd) < 60:
        return {"ok": False, "error": "时间区间过滤后无数据", "symbol": symbol, "asset": asset2}

    # 保证 OHLC
    if "open" not in dfd.columns:
        dfd["open"] = dfd["close"]
    if "high" not in dfd.columns:
        dfd["high"] = dfd["close"]
    if "low" not in dfd.columns:
        dfd["low"] = dfd["close"]

    # start_idx：从该日及之后才允许入场（模拟盘=从 start_date 开始空仓）
    start_dt = parse_date_any_opt(start_date)
    start_idx = 0
    if start_dt is not None:
        try:
            m = dfd["date"] >= start_dt
            start_idx = int(np.argmax(m.to_numpy(dtype=bool))) if bool(m.any()) else int(len(dfd))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            start_idx = 0

    # 周线
    dfw = resample_to_weekly(dfd)
    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if dfw.empty or len(dfw) < 30:
        return {"ok": False, "error": "周线数据太少", "symbol": symbol, "asset": asset2}

    # BBB 参数
    params = BBBParams(
        entry_ma=max(2, int(bbb_entry_ma)),
        dist_ma50_max=float(bbb_dist_ma_max),
        max_above_20w=float(bbb_max_above_20w),
        min_weekly_bars_total=int(bbb_min_weeks),
        require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
        require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
        require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
    )

    # weekly MA/MACD（hard 用到 MA50；BBB entry 也会用到）
    if "ma50" not in dfw.columns:
        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
    if "macd" not in dfw.columns or "macd_signal" not in dfw.columns:
        dfw = add_macd(dfw, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    # entry: 周线 BBB 信号
    entry_sig_w = compute_bbb_entry_signal(dfw, dfd, params=params).astype(bool)
    if len(entry_sig_w) != len(dfw):
        return {"ok": False, "error": "entry_sig 长度不匹配", "symbol": symbol, "asset": asset2}

    # hard exit: 周线连续两周跌破 MA50（在该周最后交易日标记）
    close_w = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    ma50_w = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
    hard_w = ((close_w < ma50_w) & (close_w.shift(1) < ma50_w.shift(1))).fillna(False).astype(bool)

    # daily soft exit: 2日死叉确认 + 跌破 MA20（在该日收盘标记，次日开盘卖）
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

    # 对齐：weekly date（周末）-> daily index（周最后交易日）
    dt_d = dfd["date"].to_numpy(dtype="datetime64[ns]")
    dt_w = dfw["date"].to_numpy(dtype="datetime64[ns]")
    end_pos = np.searchsorted(dt_d, dt_w, side="right") - 1

    n_d = int(len(dfd))
    entry_flag = np.zeros(n_d, dtype=bool)
    hard_flag = np.zeros(n_d, dtype=bool)
    for i in range(int(len(dfw))):
        p = int(end_pos[i])
        if p < 0 or p >= n_d:
            continue
        if bool(hard_w.iloc[i]):
            hard_flag[p] = True
        if bool(entry_sig_w.iloc[i]):
            e = p + 1
            if 0 <= e < n_d:
                entry_flag[e] = True

    # 交易成本（固定磨损：默认均分到买/卖）
    fixed_half = max(0.0, float(roundtrip_cost_yuan or 0.0)) / 2.0
    fee_min = max(0.0, float(min_fee_yuan or 0.0))
    cost = TradeCost(
        buy_cost=float(buy_cost or 0.0),
        sell_cost=float(sell_cost or 0.0),
        buy_fee_yuan=fixed_half,
        sell_fee_yuan=fixed_half,
        buy_fee_min_yuan=fee_min,
        sell_fee_min_yuan=fee_min,
    )

    open_px = pd.to_numeric(dfd["open"], errors="coerce").astype(float).to_numpy()
    soft_flag = soft_d.to_numpy(dtype=bool)
    dates = dfd["date"]

    min_hold_days2 = max(0, int(bbb_min_hold_days))
    cooldown_days2 = max(0, int(bbb_cooldown_days))
    lot = max(1, int(lot_size))
    gap_max = float(bbb_entry_gap_max or 0.0)
    gap_max = max(0.0, min(gap_max, 0.50))

    trades: list[dict[str, Any]] = []
    warnings: list[str] = []
    in_pos = False
    entry_idx = -1
    entry_price = 0.0
    entry_shares = 0
    entry_cash = 0.0
    entry_fee = 0.0
    next_allowed = int(start_idx)

    for t in range(0, n_d - 1):
        if (not in_pos) and entry_flag[t] and t >= next_allowed and t >= int(start_idx):
            px = float(open_px[t])
            if px > 0:
                # 防“开盘一脚踩山顶”：如果次日开盘相对前收跳空过大，跳过这次入场（不引入未来函数）。
                if gap_max > 0 and t - 1 >= 0:
                    prev_close = float(close_d.iloc[t - 1]) if close_d is not None else 0.0
                    if prev_close > 0 and px > prev_close * (1.0 + gap_max):
                        if len(warnings) < 5:
                            warnings.append(f"开盘跳空>{gap_max:.2%}（open={px:.3f} > prev_close={prev_close:.3f}），入场跳过")
                        continue

                sh = calc_shares_for_capital(capital_yuan=float(capital_yuan), price=float(px), cost=cost, lot_size=lot)
                if sh <= 0:
                    warnings.append("资金太小（扣完最低佣金/磨损后买不到一手），信号被跳过")
                else:
                    cash_in, fee = cash_buy(shares=sh, price=float(px), cost=cost)
                    in_pos = True
                    entry_idx = int(t)
                    entry_price = float(px)
                    entry_shares = int(sh)
                    entry_cash = float(cash_in)
                    entry_fee = float(fee)

        if not in_pos:
            continue

        exit_reason = None
        if hard_flag[t]:
            exit_reason = "hard"
        elif soft_flag[t] and (t - entry_idx + 1) >= min_hold_days2:
            exit_reason = "soft"
        if not exit_reason:
            continue

        exit_idx = int(t + 1)
        exit_price = float(open_px[exit_idx])
        if exit_price <= 0 or entry_price <= 0 or entry_shares <= 0:
            in_pos = False
            next_allowed = exit_idx + cooldown_days2
            continue

        cash_out, fee_out = cash_sell(shares=int(entry_shares), price=float(exit_price), cost=cost)
        pnl = float(cash_out - entry_cash)
        ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
        trades.append(
            {
                "asset": asset2,
                "symbol": str(symbol),
                "entry_date": _date_str(dates.iloc[entry_idx]),
                "exit_date": _date_str(dates.iloc[exit_idx]),
                "entry_price": float(entry_price),
                "entry_price_type": "open",
                "exit_price": float(exit_price),
                "exit_price_type": "open",
                "shares": int(entry_shares),
                "buy_fee_yuan": float(entry_fee),
                "sell_fee_yuan": float(fee_out),
                "entry_cash": float(entry_cash),
                "exit_cash": float(cash_out),
                "pnl_net": float(pnl),
                "pnl_net_pct": float(ret),
                "hold_days": int(exit_idx - entry_idx),
                "reason": str(exit_reason),
            }
        )

        in_pos = False
        entry_idx = -1
        entry_price = 0.0
        entry_shares = 0
        entry_cash = 0.0
        entry_fee = 0.0
        next_allowed = exit_idx + cooldown_days2

    last = dfd.iloc[-1]
    as_of = _date_str(last.get("date"))
    last_close = float(last.get("close")) if last.get("close") is not None else None

    open_pos = None
    if in_pos and entry_idx >= 0 and entry_shares > 0 and last_close is not None and last_close > 0:
        mv = float(entry_shares) * float(last_close)
        cash_out, fee_out = cash_sell(shares=int(entry_shares), price=float(last_close), cost=cost)
        pnl = float(cash_out - entry_cash)
        ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
        open_pos = {
            "asset": asset2,
            "symbol": str(symbol),
            "as_of": as_of,
            "entry_date": _date_str(dates.iloc[entry_idx]),
            "entry_price": float(entry_price),
            "entry_price_type": "open",
            "shares": int(entry_shares),
            "entry_cash": float(entry_cash),
            "last_close": float(last_close),
            "market_value": float(mv),
            "exit_cash_if_sell_now": float(cash_out),
            "sell_fee_yuan_if_sell_now": float(fee_out),
            "pnl_net_if_sell_now": float(pnl),
            "pnl_net_pct_if_sell_now": float(ret),
        }

    rets = [float(x.get("pnl_net_pct") or 0.0) for x in trades]
    wins = int(sum(1 for x in rets if x > 0))
    stats = {
        "trades": int(len(trades)),
        "wins": int(wins),
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "avg_return": float(sum(rets) / len(rets)) if trades else 0.0,
        "median_return": float(median(rets)) if trades else 0.0,
    }

    mt = int(max_trades or 0)
    trades_out = trades if mt <= 0 else trades[-mt:]
    return {
        "ok": True,
        "asset": asset2,
        "symbol": str(symbol),
        "as_of": as_of,
        "last_close": last_close,
        "stats": stats,
        "trades": trades_out,
        "open_position": open_pos,
        "warnings": warnings[:5],
    }


def simulate_shortline_paper(
    df_daily,
    *,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    capital_yuan: float = 3000.0,
    roundtrip_cost_yuan: float = 10.0,
    min_fee_yuan: float = 0.0,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    lot_size: int = 100,
    max_trades: int = 300,
    target_ret: float = 0.05,
    min_hold_days: int = 1,
    max_hold_days: int = 3,
    stop_loss_ret: float = 0.0,
) -> dict[str, Any]:
    """（已移除）原 shortline/周内短线模块对应的模拟盘入口。"""
    return {
        "ok": False,
        "error": "shortline/周内短线模块已从仓库精简移除",
        "symbol": str(symbol),
        "asset": "stock",
    }


def _max_drawdown(equity: list[float]) -> float | None:
    if not equity:
        return None
    peak = -1e18
    worst = 0.0
    for x in equity:
        v = float(x)
        if not math.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < worst:
                worst = dd
    return float(worst)


def _build_daily_maps(
    df_daily,
    *,
    start_date: str | None,
    end_date: str | None,
    need_atr_pct: bool = False,
    need_factor7: bool = False,
) -> dict[str, Any]:
    """
    把日线 DataFrame 变成“按日期字符串索引”的快查表：
    - open_by_date/close_by_date
    - ma20_prev_by_date：上一交易日收盘时可得的 MA20（用于入场候选排序，避免未来函数）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    if df_daily is None or getattr(df_daily, "empty", True):
        return {"ok": False, "error": "无数据"}

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return {"ok": False, "error": "无有效K线"}

    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]

    start_dt = parse_date_any_opt(start_date)
    end_dt = parse_date_any_opt(end_date)
    if start_dt is not None:
        df = df[df["date"] >= start_dt].reset_index(drop=True)
    if end_dt is not None:
        df = df[df["date"] <= end_dt].reset_index(drop=True)
    if df.empty:
        return {"ok": False, "error": "时间区间过滤后无数据"}

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    ma20_prev = close.rolling(window=20, min_periods=20).mean().shift(1)
    ma60_prev = None
    if bool(need_factor7):
        # 进攻轮动/因子排序会用到“长趋势过滤”（上一交易日可得，避免未来函数）
        ma60_prev = close.rolling(window=60, min_periods=60).mean().shift(1)

    # 流动性参考：近20日均成交额（上一交易日可得，避免未来函数）
    amount = None
    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    elif "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        amount = close * vol

    amount_avg20_prev = None
    if amount is not None:
        amount_avg20_prev = amount.rolling(window=20, min_periods=20).mean().shift(1)

    # ATR%（上一交易日可得，避免未来函数；只在需要时计算，别tm平白拖慢）
    atr14_pct_prev = None
    if bool(need_atr_pct) or bool(need_factor7):
        try:
            from .indicators import add_atr

            df_atr = add_atr(df, period=14, out_col="atr14")
            atr14 = pd.to_numeric(df_atr.get("atr14"), errors="coerce").astype(float)
            close2 = pd.to_numeric(df_atr.get("close"), errors="coerce").astype(float)
            atr_pct = (atr14 / close2).replace([math.inf, -math.inf], float("nan"))
            atr14_pct_prev = atr_pct.shift(1)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            atr14_pct_prev = None

    # 7因子（上一交易日可得，避免未来函数；只在需要时计算）
    mom63_prev = None
    mom126_prev = None
    vol20_prev = None
    dd252_prev = None
    adx14_prev = None
    boll_bw_rel_prev = None
    amount_ratio_prev = None
    volume_ratio_prev = None
    if bool(need_factor7):
        try:
            # 动量（63/126交易日≈12/26周）
            mom63_prev = ((close / close.shift(63).replace({0.0: float("nan")})) - 1.0).shift(1)
            mom126_prev = ((close / close.shift(126).replace({0.0: float("nan")})) - 1.0).shift(1)

            # 20D波动率（收益std）
            r1 = (close / close.shift(1).replace({0.0: float("nan")})) - 1.0
            vol20_prev = r1.rolling(window=20, min_periods=20).std().shift(1)

            # 252D回撤（close/rolling_max-1）
            roll_max = close.rolling(window=252, min_periods=20).max()
            dd252_prev = ((close / roll_max.replace({0.0: float("nan")})) - 1.0).shift(1)

            # ADX14（趋势强度）
            try:
                from .indicators import add_adx

                df_adx = add_adx(df, period=14, adx_col="adx14", di_plus_col="di_plus14", di_minus_col="di_minus14")
                adx_s = pd.to_numeric(df_adx.get("adx14"), errors="coerce").astype(float)
                adx14_prev = adx_s.shift(1)
            except (AttributeError):  # noqa: BLE001
                adx14_prev = None

            # BOLL 带宽相对自身252D中位数（bandwidth_rel 越小越“挤”）
            try:
                from .indicators import add_bollinger_bands

                df_bb = add_bollinger_bands(df, window=20, k=2.0, bandwidth_col="boll_bw")
                bw = pd.to_numeric(df_bb.get("boll_bw"), errors="coerce").astype(float)
                bw_med = bw.rolling(window=252, min_periods=60).median()
                bw_rel = (bw / bw_med.replace({0.0: float("nan")})).replace([math.inf, -math.inf], float("nan"))
                boll_bw_rel_prev = bw_rel.shift(1)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                boll_bw_rel_prev = None

            # 量能比：上一交易日成交额/近20日均成交额（上一交易日可得）
            if amount is not None and amount_avg20_prev is not None:
                amount_ratio_prev = (amount.shift(1) / amount_avg20_prev.replace({0.0: float("nan")})).replace([math.inf, -math.inf], float("nan"))
            if "volume" in df.columns:
                vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
                vol_avg20_prev = vol.rolling(window=20, min_periods=20).mean().shift(1)
                volume_ratio_prev = (vol.shift(1) / vol_avg20_prev.replace({0.0: float("nan")})).replace([math.inf, -math.inf], float("nan"))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            mom63_prev = None
            mom126_prev = None
            vol20_prev = None
            dd252_prev = None
            adx14_prev = None
            boll_bw_rel_prev = None
            amount_ratio_prev = None
            volume_ratio_prev = None

    dates = [_date_str(x) for x in df["date"]]
    open_px = pd.to_numeric(df["open"], errors="coerce").astype(float).to_list()
    high_px = pd.to_numeric(df["high"], errors="coerce").astype(float).to_list()
    low_px = pd.to_numeric(df["low"], errors="coerce").astype(float).to_list()
    close_px = close.to_list()
    ma20_prev_px = pd.to_numeric(ma20_prev, errors="coerce").astype(float).to_list()
    ma60_prev_px = pd.to_numeric(ma60_prev, errors="coerce").astype(float).to_list() if ma60_prev is not None else None
    amount_avg20_prev_px = (
        pd.to_numeric(amount_avg20_prev, errors="coerce").astype(float).to_list() if amount_avg20_prev is not None else None
    )
    atr14_pct_prev_px = pd.to_numeric(atr14_pct_prev, errors="coerce").astype(float).to_list() if atr14_pct_prev is not None else None
    mom63_prev_px = pd.to_numeric(mom63_prev, errors="coerce").astype(float).to_list() if mom63_prev is not None else None
    mom126_prev_px = pd.to_numeric(mom126_prev, errors="coerce").astype(float).to_list() if mom126_prev is not None else None
    vol20_prev_px = pd.to_numeric(vol20_prev, errors="coerce").astype(float).to_list() if vol20_prev is not None else None
    dd252_prev_px = pd.to_numeric(dd252_prev, errors="coerce").astype(float).to_list() if dd252_prev is not None else None
    adx14_prev_px = pd.to_numeric(adx14_prev, errors="coerce").astype(float).to_list() if adx14_prev is not None else None
    boll_bw_rel_prev_px = (
        pd.to_numeric(boll_bw_rel_prev, errors="coerce").astype(float).to_list() if boll_bw_rel_prev is not None else None
    )
    amount_ratio_prev_px = (
        pd.to_numeric(amount_ratio_prev, errors="coerce").astype(float).to_list() if amount_ratio_prev is not None else None
    )
    volume_ratio_prev_px = (
        pd.to_numeric(volume_ratio_prev, errors="coerce").astype(float).to_list() if volume_ratio_prev is not None else None
    )

    vol_px = None
    if "volume" in df.columns:
        vol_px = pd.to_numeric(df["volume"], errors="coerce").astype(float).to_list()
    amt_px = amount.to_list() if amount is not None else None

    open_by_date: dict[str, float] = {}
    close_by_date: dict[str, float] = {}
    high_by_date: dict[str, float] = {}
    low_by_date: dict[str, float] = {}
    volume_by_date: dict[str, float] = {}
    amount_by_date: dict[str, float] = {}
    ma20_prev_by_date: dict[str, float] = {}
    ma60_prev_by_date: dict[str, float] = {}
    amount_avg20_prev_by_date: dict[str, float] = {}
    atr14_pct_prev_by_date: dict[str, float] = {}
    mom63_prev_by_date: dict[str, float] = {}
    mom126_prev_by_date: dict[str, float] = {}
    vol20_prev_by_date: dict[str, float] = {}
    dd252_prev_by_date: dict[str, float] = {}
    adx14_prev_by_date: dict[str, float] = {}
    boll_bw_rel_prev_by_date: dict[str, float] = {}
    amount_ratio_prev_by_date: dict[str, float] = {}
    volume_ratio_prev_by_date: dict[str, float] = {}
    for i, d in enumerate(dates):
        op = float(open_px[i]) if i < len(open_px) else float("nan")
        hp = float(high_px[i]) if i < len(high_px) else float("nan")
        lp = float(low_px[i]) if i < len(low_px) else float("nan")
        cp = float(close_px[i]) if i < len(close_px) else float("nan")
        mp = float(ma20_prev_px[i]) if i < len(ma20_prev_px) else float("nan")
        mp60 = float(ma60_prev_px[i]) if (ma60_prev_px is not None and i < len(ma60_prev_px)) else float("nan")
        ap = float(amount_avg20_prev_px[i]) if (amount_avg20_prev_px is not None and i < len(amount_avg20_prev_px)) else float("nan")
        atrp = float(atr14_pct_prev_px[i]) if (atr14_pct_prev_px is not None and i < len(atr14_pct_prev_px)) else float("nan")
        m63 = float(mom63_prev_px[i]) if (mom63_prev_px is not None and i < len(mom63_prev_px)) else float("nan")
        m126 = float(mom126_prev_px[i]) if (mom126_prev_px is not None and i < len(mom126_prev_px)) else float("nan")
        v20 = float(vol20_prev_px[i]) if (vol20_prev_px is not None and i < len(vol20_prev_px)) else float("nan")
        ddp = float(dd252_prev_px[i]) if (dd252_prev_px is not None and i < len(dd252_prev_px)) else float("nan")
        adx = float(adx14_prev_px[i]) if (adx14_prev_px is not None and i < len(adx14_prev_px)) else float("nan")
        bwrel = float(boll_bw_rel_prev_px[i]) if (boll_bw_rel_prev_px is not None and i < len(boll_bw_rel_prev_px)) else float("nan")
        ar = float(amount_ratio_prev_px[i]) if (amount_ratio_prev_px is not None and i < len(amount_ratio_prev_px)) else float("nan")
        vr = float(volume_ratio_prev_px[i]) if (volume_ratio_prev_px is not None and i < len(volume_ratio_prev_px)) else float("nan")
        vp = float(vol_px[i]) if (vol_px is not None and i < len(vol_px)) else float("nan")
        amt = float(amt_px[i]) if (amt_px is not None and i < len(amt_px)) else float("nan")
        if math.isfinite(op) and op > 0:
            open_by_date[str(d)] = float(op)
        if math.isfinite(hp) and hp > 0:
            high_by_date[str(d)] = float(hp)
        if math.isfinite(lp) and lp > 0:
            low_by_date[str(d)] = float(lp)
        if math.isfinite(cp) and cp > 0:
            close_by_date[str(d)] = float(cp)
        if math.isfinite(mp) and mp > 0:
            ma20_prev_by_date[str(d)] = float(mp)
        if math.isfinite(mp60) and mp60 > 0:
            ma60_prev_by_date[str(d)] = float(mp60)
        if math.isfinite(ap) and ap > 0:
            amount_avg20_prev_by_date[str(d)] = float(ap)
        if math.isfinite(atrp) and atrp > 0:
            atr14_pct_prev_by_date[str(d)] = float(atrp)
        if math.isfinite(m63):
            mom63_prev_by_date[str(d)] = float(m63)
        if math.isfinite(m126):
            mom126_prev_by_date[str(d)] = float(m126)
        if math.isfinite(v20) and v20 >= 0:
            vol20_prev_by_date[str(d)] = float(v20)
        if math.isfinite(ddp):
            dd252_prev_by_date[str(d)] = float(ddp)
        if math.isfinite(adx) and adx >= 0:
            adx14_prev_by_date[str(d)] = float(adx)
        if math.isfinite(bwrel) and bwrel > 0:
            boll_bw_rel_prev_by_date[str(d)] = float(bwrel)
        if math.isfinite(ar) and ar > 0:
            amount_ratio_prev_by_date[str(d)] = float(ar)
        if math.isfinite(vr) and vr > 0:
            volume_ratio_prev_by_date[str(d)] = float(vr)
        if math.isfinite(vp) and vp >= 0:
            volume_by_date[str(d)] = float(vp)
        if math.isfinite(amt) and amt >= 0:
            amount_by_date[str(d)] = float(amt)

    return {
        "ok": True,
        "dates": dates,
        "open_by_date": open_by_date,
        "close_by_date": close_by_date,
        "high_by_date": high_by_date,
        "low_by_date": low_by_date,
        "volume_by_date": volume_by_date,
        "amount_by_date": amount_by_date,
        "ma20_prev_by_date": ma20_prev_by_date,
        "ma60_prev_by_date": ma60_prev_by_date,
        "amount_avg20_prev_by_date": amount_avg20_prev_by_date,
        "atr14_pct_prev_by_date": atr14_pct_prev_by_date,
        "mom63_prev_by_date": mom63_prev_by_date,
        "mom126_prev_by_date": mom126_prev_by_date,
        "vol20_prev_by_date": vol20_prev_by_date,
        "dd252_prev_by_date": dd252_prev_by_date,
        "adx14_prev_by_date": adx14_prev_by_date,
        "boll_bw_rel_prev_by_date": boll_bw_rel_prev_by_date,
        "amount_ratio_prev_by_date": amount_ratio_prev_by_date,
        "volume_ratio_prev_by_date": volume_ratio_prev_by_date,
        "last_date": dates[-1],
    }


def _compute_regime_label_by_date(*, df_index_daily, calendar_dates: list[str]) -> dict[str, str]:
    """
    用指数周线标签（date=周末）对齐到“日线交易日开盘可用”的标签：
    - 日 d 用 “严格小于 d 的最后一个周末标签”，避免未来函数
    """
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .market_regime import compute_market_regime_weekly_series

    if df_index_daily is None or getattr(df_index_daily, "empty", True) or not calendar_dates:
        return {}

    dfw = compute_market_regime_weekly_series(index_symbol="index", df_daily=df_index_daily)
    if dfw is None or getattr(dfw, "empty", True) or "date" not in dfw.columns or "label" not in dfw.columns:
        return {}

    w = dfw.copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date", "label"]).sort_values("date").reset_index(drop=True)
    if w.empty:
        return {}

    w_dates = w["date"].to_numpy(dtype="datetime64[ns]")
    w_labels = w["label"].astype(str).to_numpy()

    d_dates = pd.to_datetime(calendar_dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(w_dates, d_dates, side="left") - 1
    labels = np.where(pos >= 0, w_labels[pos], "unknown")
    out: dict[str, str] = {}
    for i, d in enumerate(calendar_dates):
        out[str(d)] = str(labels[i])
    return out


def _build_mom_63d_by_date(df_index_daily) -> dict[str, float]:
    """
    指数日线 63 日动量（close/close.shift(63)-1），用于把 bull 再细分成 hot/slow（疯牛/慢牛）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    if df_index_daily is None or getattr(df_index_daily, "empty", True):
        return {}

    df = df_index_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return {}

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    mom = (close / close.shift(63).replace({0.0: float("nan")})) - 1.0
    out: dict[str, float] = {}
    for i in range(len(df)):
        d = _date_str(df.iloc[i].get("date"))
        try:
            v = float(mom.iloc[i])
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            continue
        if d and math.isfinite(v):
            out[str(d)] = float(v)
    return out


def _prepare_bbb_exec_dates(
    df_daily,
    *,
    bbb_entry_ma: int,
    bbb_dist_ma_max: float,
    bbb_max_above_20w: float,
    bbb_min_weeks: int,
    bbb_require_weekly_macd_bullish: bool,
    bbb_require_weekly_macd_above_zero: bool,
    bbb_require_daily_macd_bullish: bool,
    weekly_anchor_ma: int = 20,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    为 BBB（ETF/股票）准备“次日开盘执行”的信号日集合：
    - entry_exec_dates：周线入场信号 -> 次日开盘
    - hard_exec_dates：周线 MA50 连续2周确认跌破 -> 次日开盘
    - trail_exec_dates：周线 close 跌破周线锚（默认 MA20）-> 次日开盘
    - soft_exec_dates：日线 MACD 死叉2日确认 且 close<MA20 -> 次日开盘
    """
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .bbb import BBBParams, compute_bbb_entry_signal
    from .indicators import add_macd, add_moving_averages
    from .resample import resample_to_weekly

    if df_daily is None or getattr(df_daily, "empty", True):
        return {"ok": False, "error": "无数据"}

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if dfd.empty or len(dfd) < 60:
        return {"ok": False, "error": "K线太少"}

    end_dt = parse_date_any_opt(end_date)
    if end_dt is not None:
        dfd = dfd[dfd["date"] <= end_dt].reset_index(drop=True)
    if dfd.empty or len(dfd) < 60:
        return {"ok": False, "error": "时间区间过滤后无数据"}

    # 保证 OHLC
    if "open" not in dfd.columns:
        dfd["open"] = dfd["close"]
    if "high" not in dfd.columns:
        dfd["high"] = dfd["close"]
    if "low" not in dfd.columns:
        dfd["low"] = dfd["close"]
    dates = [_date_str(x) for x in dfd["date"]]

    # soft（日线）
    if "ma20" not in dfd.columns:
        close_d = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
        dfd = dfd.copy()
        dfd["ma20"] = close_d.rolling(window=20, min_periods=20).mean()
    if "macd" not in dfd.columns or "macd_signal" not in dfd.columns:
        dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    close_d = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
    ma20_d = pd.to_numeric(dfd["ma20"], errors="coerce").astype(float)
    macd_d = pd.to_numeric(dfd["macd"], errors="coerce").astype(float)
    sig_d = pd.to_numeric(dfd["macd_signal"], errors="coerce").astype(float)
    bearish = (macd_d < sig_d)
    bearish2 = bearish & bearish.shift(1, fill_value=False)
    soft_d = (bearish2 & (close_d < ma20_d)).fillna(False).astype(bool)

    soft_exec: set[str] = set()
    for t in range(0, int(len(dfd)) - 1):
        if bool(soft_d.iloc[t]):
            soft_exec.add(str(dates[t + 1]))

    # 周线：entry/hard/trail
    dfw = resample_to_weekly(dfd)
    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if dfw.empty or len(dfw) < 30:
        return {"ok": False, "error": "周线数据太少"}

    params = BBBParams(
        entry_ma=max(2, int(bbb_entry_ma)),
        dist_ma50_max=float(bbb_dist_ma_max),
        max_above_20w=float(bbb_max_above_20w),
        min_weekly_bars_total=int(bbb_min_weeks),
        require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
        require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
        require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
    )

    entry_w = compute_bbb_entry_signal(dfw, dfd, params=params).astype(bool)
    if int(len(entry_w)) != int(len(dfw)):
        return {"ok": False, "error": "entry_sig 长度不匹配"}

    # 周线 MA（hard 用 MA50；trail 用周线锚 MA20）
    if "ma50" not in dfw.columns:
        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
    w_close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    w_ma50 = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
    hard_w = ((w_close < w_ma50) & (w_close.shift(1) < w_ma50.shift(1))).fillna(False).astype(bool)

    anchor = max(2, int(weekly_anchor_ma))
    w_ma_anchor = w_close.rolling(window=anchor, min_periods=anchor).mean()
    trail_w = ((w_close < w_ma_anchor) & w_ma_anchor.notna()).fillna(False).astype(bool)

    dt_d = dfd["date"].to_numpy(dtype="datetime64[ns]")
    dt_w = dfw["date"].to_numpy(dtype="datetime64[ns]")
    end_pos = np.searchsorted(dt_d, dt_w, side="right") - 1
    n_d = int(len(dfd))

    entry_exec: set[str] = set()
    hard_exec: set[str] = set()
    trail_exec: set[str] = set()
    for i in range(int(len(dfw))):
        p = int(end_pos[i])
        if p < 0 or p >= n_d:
            continue
        e = int(p + 1)
        if not (0 <= e < n_d):
            continue
        d_exec = str(dates[e])
        if bool(entry_w.iloc[i]):
            entry_exec.add(d_exec)
        if bool(hard_w.iloc[i]):
            hard_exec.add(d_exec)
        if bool(trail_w.iloc[i]):
            trail_exec.add(d_exec)

    return {
        "ok": True,
        "dates": dates,
        "entry_exec_dates": entry_exec,
        "hard_exec_dates": hard_exec,
        "trail_exec_dates": trail_exec,
        "soft_exec_dates": soft_exec,
    }


def simulate_portfolio_paper(
    dfs_by_symbol: dict[str, Any],
    *,
    strategy: PaperStrategy,
    start_date: str | None = None,
    end_date: str | None = None,
    capital_yuan: float = 100000.0,
    roundtrip_cost_yuan: float = 10.0,
    min_fee_yuan: float = 0.0,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    slippage_mode: str = "none",
    slippage_bps: float = 0.0,
    slippage_ref_amount_yuan: float = 1e8,
    slippage_bps_min: float = 0.0,
    slippage_bps_max: float = 30.0,
    slippage_unknown_bps: float = 10.0,
    slippage_vol_mult: float = 0.0,
    lot_size: int = 100,
    max_positions: int = 0,
    max_exposure_pct: float = 0.0,
    vol_target: float = 0.0,
    vol_lookback_days: int = 20,
    max_turnover_pct: float = 0.0,
    max_corr: float = 0.0,
    max_per_theme: int = 0,
    limit_up_pct: float = 0.0,
    limit_down_pct: float = 0.0,
    halt_vol_zero: bool = True,
    df_regime_index_daily=None,
    df_rs_index_daily=None,
    core_holdings: dict[str, float] | None = None,
    core_min_pct: float = 0.0,
    min_trade_notional_yuan: float = 0.0,
    portfolio_dd_stop: float = 0.0,
    portfolio_dd_cooldown_days: int = 0,
    portfolio_dd_restart_ma_days: int = 0,
    rot_rebalance_weeks: int = 1,
    rot_hold_n: int = 6,
    rot_buffer_n: int = 2,
    rot_rank_mode: str = "factor7",
    rot_gap_max: float = 0.015,
    rot_split_exec_days: int = 1,
    rot_pullback_ma20_dist_max: float = 0.0,
    rot_trend_ma: int = 0,
    bbb_entry_gap_max: float = 0.015,
    bbb_entry_rank_mode: str = "ma20_dist",
    bbb_factor7_weights: str = "",
    bbb_entry_ma: int = 20,
    bbb_dist_ma_max: float = 0.12,
    bbb_max_above_20w: float = 0.05,
    bbb_min_weeks: int = 60,
    bbb_require_weekly_macd_bullish: bool = True,
    bbb_require_weekly_macd_above_zero: bool = True,
    bbb_require_daily_macd_bullish: bool = True,
    bbb_min_hold_days: int = 5,
    bbb_cooldown_days: int = 0,
    target_ret: float = 0.05,
    min_hold_days: int = 1,
    max_hold_days: int = 3,
    stop_loss_ret: float = 0.0,
) -> dict[str, Any]:
    """
    组合模拟盘（账户级）：共享一笔资金，不是“每个标的各跑各的”。

    约定：
    - BBB：入/出场都按“次日开盘”（exit_exec=open）
    - rot_stock_weekly：信号=上周收盘；执行=本周首个交易日开盘（可选分两天执行）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .bbb import BBBParams, compute_bbb_entry_signal
    from .indicators import add_macd, add_moving_averages
    from .positioning import risk_profile_for_regime
    from .quality_gate import StockQualityGate
    from .resample import resample_to_weekly
    from .take_profit import TakeProfitConfig, calc_tp1_sell_shares, classify_bull_phase

    if not dfs_by_symbol:
        return {"ok": False, "error": "watchlist 为空"}

    strat = str(strategy or "bbb_etf").strip()
    if strat not in {"bbb_etf", "bbb_stock", "rot_stock_weekly"}:
        return {"ok": False, "error": f"未知 strategy：{strategy}"}
    is_bbb = strat in {"bbb_etf", "bbb_stock"}
    is_rot = strat == "rot_stock_weekly"

    # BBB 候选排序模式（不影响硬规则，只影响“同一天多个信号抢额度/现金”的优先级）
    rank_mode = str(bbb_entry_rank_mode or "ma20_dist").strip().lower() or "ma20_dist"
    if rank_mode not in {"ma20_dist", "factor7"}:
        rank_mode = "ma20_dist"

    rot_rank_mode2 = str(rot_rank_mode or "factor7").strip().lower() or "factor7"
    if rot_rank_mode2 not in {"factor7", "mom63", "mom126"}:
        rot_rank_mode2 = "factor7"

    # rot_stock_weekly 的 rank_mode 也需要 mom63/mom126 等“派生字段”，统一走 need_factor7 开关计算（省得再加一堆 flag）。
    need_factor7 = bool((is_bbb and rank_mode == "factor7") or is_rot)

    # core holdings（用于“吃beta”：没信号也尽量别空仓）
    core_w: dict[str, float] = {}
    if is_bbb and isinstance(core_holdings, dict) and core_holdings:
        for k, v in core_holdings.items():
            sym = str(k or "").strip()
            if not sym:
                continue
            try:
                w = float(v)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(w)) or w <= 0:
                continue
            core_w[sym] = float(w)
        sw = float(sum(core_w.values()))
        if sw > 0:
            core_w = {k: float(v) / float(sw) for k, v in core_w.items()}
        else:
            core_w = {}

    core_min_pct2 = float(core_min_pct or 0.0)
    if (not math.isfinite(core_min_pct2)) or core_min_pct2 <= 0:
        core_min_pct2 = 0.0
    core_min_pct2 = float(max(0.0, min(core_min_pct2, 1.0)))

    min_trade_notional2 = float(min_trade_notional_yuan or 0.0)
    if (not math.isfinite(min_trade_notional2)) or min_trade_notional2 <= 0:
        min_trade_notional2 = 0.0
    min_trade_notional2 = float(max(0.0, min_trade_notional2))

    default_w7 = {"rs": 0.35, "trend": 0.15, "vol": 0.15, "drawdown": 0.15, "liquidity": 0.10, "boll": 0.05, "volume": 0.05}
    w7 = dict(default_w7)
    try:
        txt = str(bbb_factor7_weights or "").strip()
        if txt:
            for part in txt.split(","):
                p = str(part or "").strip()
                if not p or "=" not in p:
                    continue
                k, v = p.split("=", 1)
                key = str(k or "").strip().lower()
                if key not in w7:
                    continue
                try:
                    w7[key] = float(v)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
        s2 = float(sum(max(0.0, float(x)) for x in w7.values()))
        if s2 > 0:
            w7 = {k: float(max(0.0, float(v))) / s2 for k, v in w7.items()}
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        w7 = dict(default_w7)

    # 成本/滑点（统一口径；scan/backtest/paper-sim/run 用同一套参数）
    cost = trade_cost_from_params(
        roundtrip_cost_yuan=float(roundtrip_cost_yuan or 0.0),
        min_fee_yuan=float(min_fee_yuan or 0.0),
        buy_cost=float(buy_cost or 0.0),
        sell_cost=float(sell_cost or 0.0),
    )
    slip_mode = str(slippage_mode or "none").strip().lower() or "none"
    slip_bps = float(slippage_bps or 0.0)
    slip_ref_amt = float(slippage_ref_amount_yuan or 1e8)
    slip_bps_min = float(slippage_bps_min or 0.0)
    slip_bps_max = float(slippage_bps_max or 30.0)
    slip_unknown_bps = float(slippage_unknown_bps or 10.0)
    slip_vm = float(slippage_vol_mult or 0.0)

    # 真实交易约束（默认不开启“涨跌停”假设，避免我瞎拍脑袋；你要用再显式传 pct）
    lim_up = max(0.0, float(limit_up_pct or 0.0))
    lim_dn = max(0.0, float(limit_down_pct or 0.0))
    halt_zero = bool(halt_vol_zero)
    lot = max(1, int(lot_size or 100))
    gap_max = float(bbb_entry_gap_max or 0.0)
    gap_max = max(0.0, min(gap_max, 0.50))

    # 先把每个标的的日线做成 map（给净值曲线/候选排序用）
    maps_by_symbol: dict[str, dict[str, Any]] = {}
    map_errors: list[dict[str, Any]] = []
    for sym, df in dfs_by_symbol.items():
        # BBB 需要“看更长的历史”算周线/指标（start_date 只限制入场，不截断历史）。
        sd_for_maps = None if (is_bbb or is_rot) else start_date
        m = _build_daily_maps(df, start_date=sd_for_maps, end_date=end_date, need_atr_pct=bool(slip_vm > 0), need_factor7=bool(need_factor7))
        if not bool(m.get("ok", False)):
            map_errors.append({"symbol": str(sym), "error": str(m.get("error") or "build_maps_failed")})
            continue
        maps_by_symbol[str(sym)] = m

    if not maps_by_symbol:
        return {"ok": False, "error": "所有标的都没有有效K线", "errors": map_errors[:50]}

    # calendar：优先用大盘指数（更统一）；否则用所有标的 union
    idx_map = None
    rs_idx_map = None
    calendar: list[str] = []
    if df_regime_index_daily is not None and (not getattr(df_regime_index_daily, "empty", True)):
        # BBB 也别截断指数历史：不然 63D动量/波动/回撤这些前面全是 NaN，排序/风控等于瞎。
        sd_idx_maps = None if (is_bbb or is_rot) else start_date
        idx_map2 = _build_daily_maps(
            df_regime_index_daily,
            start_date=sd_idx_maps,
            end_date=end_date,
            need_atr_pct=bool(slip_vm > 0),
            need_factor7=bool(need_factor7),
        )
        if bool(idx_map2.get("ok", False)):
            idx_map = idx_map2
            calendar = list(idx_map2.get("dates") or [])

    # RS 基准（仅给 BBB factor7 用）：允许和 regime-index 不同，别把“风险分段”和“相对强弱尺子”硬绑死。
    if need_factor7 and df_rs_index_daily is not None and (not getattr(df_rs_index_daily, "empty", True)):
        try:
            sd_rs_maps = None if (is_bbb or is_rot) else start_date
            rs_map2 = _build_daily_maps(
                df_rs_index_daily,
                start_date=sd_rs_maps,
                end_date=end_date,
                need_atr_pct=bool(slip_vm > 0),
                need_factor7=True,
            )
            if bool(rs_map2.get("ok", False)):
                rs_idx_map = rs_map2
        except (AttributeError):  # noqa: BLE001
            rs_idx_map = None
    if not calendar:
        s: set[str] = set()
        for m in maps_by_symbol.values():
            for d in (m.get("dates") or []):
                s.add(str(d))
        calendar = sorted(s)

    # 统一把 start_date/end_date 规范成 YYYY-MM-DD 再过滤 calendar（避免传 20200101 这种把 2019-xx-xx 误杀掉）
    sd = parse_date_any_opt(start_date)
    ed = parse_date_any_opt(end_date)
    if sd is not None:
        sd_s = str(sd.date())
        calendar = [d for d in calendar if str(d) >= sd_s]
    if ed is not None:
        ed_s = str(ed.date())
        calendar = [d for d in calendar if str(d) <= ed_s]
    if not calendar:
        return {"ok": False, "error": "无可用交易日（日历为空）", "errors": map_errors[:50]}

    # 牛熊标签（可选）
    regime_by_date = _compute_regime_label_by_date(df_index_daily=df_regime_index_daily, calendar_dates=calendar) if df_regime_index_daily is not None else {}

    # --------- 组合构建约束（KISS 版本）---------
    vol_tgt = max(0.0, float(vol_target or 0.0))
    vol_lb = max(5, min(int(vol_lookback_days or 20), 252))
    max_turn = max(0.0, float(max_turnover_pct or 0.0))
    max_turn = min(max_turn, 5.0)
    max_corr2 = float(max_corr or 0.0)
    if max_corr2 < 0:
        max_corr2 = 0.0
    if max_corr2 > 0.999:
        max_corr2 = 0.999
    max_theme = max(0, int(max_per_theme or 0))

    # 硬过滤：不碰杂毛/妖股（避免未来函数：只用 exec 日可得的价格 + prev_20d 均成交额）
    stock_gate = StockQualityGate() if strat in {"bbb_stock", "rot_stock_weekly"} else None
    stock_gate_params = (
        {
            "exclude_bj": bool(stock_gate.exclude_bj),
            "min_price": float(stock_gate.min_price),
            "min_amount_avg20_yuan": float(stock_gate.min_amount_avg20_yuan),
        }
        if stock_gate is not None
        else None
    )

    def _stock_quality_ok(sym: str, *, exec_date: str, exec_price: float) -> tuple[bool, str]:
        if stock_gate is None:
            return True, ""

        s2 = str(sym or "").strip().lower()
        if stock_gate.exclude_bj and s2.startswith("bj"):
            return False, "exclude_bj"

        px = float(exec_price or 0.0)
        if float(stock_gate.min_price) > 0 and (px <= 0 or px + 1e-12 < float(stock_gate.min_price)):
            return False, "min_price"

        if float(stock_gate.min_amount_avg20_yuan) > 0:
            m = maps_by_symbol.get(str(sym)) or {}
            liq = (m.get("amount_avg20_prev_by_date") or {}).get(str(exec_date))
            try:
                liq2 = float(liq) if liq is not None else 0.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                liq2 = 0.0
            if liq2 <= 0 or liq2 + 1e-6 < float(stock_gate.min_amount_avg20_yuan):
                return False, "min_amount_avg20"

        return True, ""

    name_by_symbol: dict[str, str] = {}
    infer_theme = None
    if max_theme > 0:
        try:
            from .symbol_names import load_universe_name_map
            from .portfolio import infer_theme as _infer_theme

            infer_theme = _infer_theme
            asset_for_names = "etf" if strat == "bbb_etf" else "stock"
            name_by_symbol = {str(k): str(v) for k, v in load_universe_name_map(asset_for_names, ttl_hours=24.0).items()}
        except (AttributeError):  # noqa: BLE001
            name_by_symbol = {}
            infer_theme = None

    corr_abs_tail = None
    if max_corr2 > 0:
        try:
            from .portfolio import corr_abs_tail as _corr_abs_tail

            corr_abs_tail = _corr_abs_tail
        except (AttributeError):  # noqa: BLE001
            corr_abs_tail = None

    idx_close_by_date = (idx_map.get("close_by_date") if isinstance(idx_map, dict) else None) if vol_tgt > 0 else None
    vol_cache: dict[int, float | None] = {}

    def _index_realized_vol_ann(end_idx: int) -> float | None:
        # end_idx：用到 calendar[end_idx] 的 close（通常传 i-1，避免未来函数）
        if vol_tgt <= 0 or idx_close_by_date is None:
            return None
        ei = int(end_idx)
        if ei in vol_cache:
            return vol_cache[ei]
        if ei <= 0 or ei >= len(calendar):
            vol_cache[ei] = None
            return None

        # 取最近 vol_lb 日收益（不足就退化）
        rets: list[float] = []
        start = max(1, ei - int(vol_lb) + 1)
        for j in range(start, ei + 1):
            d0 = str(calendar[j - 1])
            d1 = str(calendar[j])
            c0 = idx_close_by_date.get(d0) if isinstance(idx_close_by_date, dict) else None
            c1 = idx_close_by_date.get(d1) if isinstance(idx_close_by_date, dict) else None
            try:
                c0f = float(c0) if c0 is not None else 0.0
                c1f = float(c1) if c1 is not None else 0.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if c0f <= 0 or c1f <= 0:
                continue
            rets.append(float(c1f / c0f - 1.0))
        if len(rets) < 5:
            vol_cache[ei] = None
            return None
        m = sum(rets) / float(len(rets))
        var = sum((x - m) ** 2 for x in rets) / float(max(1, len(rets) - 1))
        std = math.sqrt(max(0.0, float(var)))
        v = float(std) * float(math.sqrt(252.0))
        vol_cache[ei] = (float(v) if math.isfinite(v) and v > 0 else None)
        return vol_cache[ei]

    def _daily_returns_tail(sym: str, *, end_idx: int, window_days: int = 60) -> list[float]:
        if end_idx <= 0:
            return []
        m = maps_by_symbol.get(sym) or {}
        cb = (m.get("close_by_date") or {}) if isinstance(m, dict) else {}
        if not isinstance(cb, dict):
            return []
        ei = min(int(end_idx), len(calendar) - 1)
        start = max(1, ei - int(window_days) + 1)
        out: list[float] = []
        for j in range(start, ei + 1):
            d0 = str(calendar[j - 1])
            d1 = str(calendar[j])
            c0 = cb.get(d0)
            c1 = cb.get(d1)
            try:
                c0f = float(c0) if c0 is not None else 0.0
                c1f = float(c1) if c1 is not None else 0.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if c0f <= 0 or c1f <= 0:
                continue
            out.append(float(c1f / c0f - 1.0))
        return out

    def _name_theme(sym: str) -> tuple[str | None, str | None]:
        if infer_theme is None:
            return None, None
        name = name_by_symbol.get(str(sym).strip().lower())
        try:
            theme = infer_theme(str(name or ""))
        except (AttributeError):  # noqa: BLE001
            theme = None
        return (str(name) if name else None), (str(theme) if theme else None)

    # 组合级最大回撤熔断（通用；研究用途）
    dd_stop = float(portfolio_dd_stop or 0.0)
    if (not math.isfinite(dd_stop)) or dd_stop <= 0:
        dd_stop = 0.0
    dd_stop = float(max(0.0, min(dd_stop, 0.99)))

    dd_cool = int(portfolio_dd_cooldown_days or 0)
    dd_cool = max(0, min(dd_cool, 3650))

    dd_restart_ma_days = int(portfolio_dd_restart_ma_days or 0)
    dd_restart_ma_days = max(0, min(dd_restart_ma_days, 20))

    idx_close_by_date_restart = (idx_map.get("close_by_date") if isinstance(idx_map, dict) else None) if isinstance(idx_map, dict) else None
    idx_ma20_prev_by_date_restart = (idx_map.get("ma20_prev_by_date") if isinstance(idx_map, dict) else None) if isinstance(idx_map, dict) else None

    def _index_above_ma20_consecutive(*, end_idx: int, days: int) -> bool:
        """
        判断“最近 days 个收盘（到 prev_d 为止）是否都 > MA20”。
        - 在开盘执行：只能用到 prev_d 的收盘，所以这里比较 close[day] vs ma20_prev[next_day]。
        """
        n = int(days or 0)
        if n <= 0:
            return True
        if not isinstance(idx_close_by_date_restart, dict) or not isinstance(idx_ma20_prev_by_date_restart, dict):
            return False
        ei = int(end_idx)
        if ei <= n:
            return False
        # 检查区间：j in [ei-n, ei-1]
        for j in range(int(ei - n), int(ei)):
            if j < 0 or j + 1 >= len(calendar):
                return False
            d_close = str(calendar[j])
            d_next = str(calendar[j + 1])
            c = idx_close_by_date_restart.get(d_close)
            ma = idx_ma20_prev_by_date_restart.get(d_next)
            try:
                cf = float(c) if c is not None else 0.0
                maf = float(ma) if ma is not None else 0.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return False
            if cf <= 0 or maf <= 0:
                return False
            if not (cf > maf):
                return False
        return True

    # rot_stock_weekly（组合）：周频“进攻轮动”骨架（信号=上周收盘，执行=本周首个交易日开盘）。
    # - 目标：解决“周一调仓踩雷”的可控性：支持 gap 过滤 + 2日分批执行。
    # - 严格避免未来函数：只用 exec 日(prev_close)可得的数据做排序/决策。
    if is_rot:
        import bisect

        reb_weeks = int(rot_rebalance_weeks or 1)
        reb_weeks = max(1, min(reb_weeks, 52))

        hold_n = int(rot_hold_n or 0)
        if hold_n <= 0:
            hold_n = int(max_positions or 0)
        hold_n = max(1, min(hold_n, 50))

        buffer_n = int(rot_buffer_n or 0)
        buffer_n = max(0, min(buffer_n, 50))

        gap_rot = float(rot_gap_max or 0.0)
        if (not math.isfinite(gap_rot)) or gap_rot < 0:
            gap_rot = 0.0
        gap_rot = float(max(0.0, min(gap_rot, 0.50)))

        split_days = int(rot_split_exec_days or 1)
        split_days = 1 if split_days <= 1 else 2

        pullback_ma20_dist_max = float(rot_pullback_ma20_dist_max or 0.0)
        if (not math.isfinite(pullback_ma20_dist_max)) or pullback_ma20_dist_max < 0:
            pullback_ma20_dist_max = 0.0
        pullback_ma20_dist_max = float(max(0.0, min(pullback_ma20_dist_max, 0.50)))

        trend_ma = int(rot_trend_ma or 0)
        trend_ma = max(0, min(trend_ma, 400))
        # 当前只预计算了 MA60_prev（为了不把 maps 计算搞得太重）；别传一些花里胡哨的值来坑我。
        if trend_ma not in {0, 60}:
            trend_ma = 60

        # ---- 成本/约束：按“执行日”动态估 slippage（避免未来函数） ----
        slip_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def _slippage_for(sym: str, d: str) -> dict[str, Any]:
            key = (str(sym), str(d))
            if key in slip_cache:
                return slip_cache[key]
            m = maps_by_symbol.get(sym) or {}
            amt = (m.get("amount_avg20_prev_by_date") or {}).get(d)
            atrp = (m.get("atr14_pct_prev_by_date") or {}).get(d) if float(slip_vm) > 0 else None
            slip_bps2 = estimate_slippage_bps(
                mode=str(slip_mode or "none"),
                amount_avg20_yuan=(float(amt) if amt is not None else None),
                atr_pct=(float(atrp) if atrp is not None else None),
                bps=float(slip_bps),
                ref_amount_yuan=float(slip_ref_amt),
                min_bps=float(slip_bps_min),
                max_bps=float(slip_bps_max),
                unknown_bps=float(slip_unknown_bps),
                vol_mult=float(slip_vm),
            )
            out = {
                "slippage_mode": str(slip_mode or "none"),
                "slippage_bps": float(slip_bps2),
                "slippage_rate": float(bps_to_rate(float(slip_bps2))),
                "amount_avg20_prev_yuan": (float(amt) if amt is not None else None),
                "atr14_pct_prev": (float(atrp) if atrp is not None else None),
            }
            slip_cache[key] = out
            return out

        def _cost_for_trade(sym: str, d: str) -> tuple[TradeCost, dict[str, Any]]:
            slip = _slippage_for(sym, d)
            r = float(slip.get("slippage_rate") or 0.0)
            c = TradeCost(
                buy_cost=float(cost.buy_cost) + float(r),
                sell_cost=float(cost.sell_cost) + float(r),
                buy_fee_yuan=float(cost.buy_fee_yuan),
                sell_fee_yuan=float(cost.sell_fee_yuan),
                buy_fee_min_yuan=float(cost.buy_fee_min_yuan),
                sell_fee_min_yuan=float(cost.sell_fee_min_yuan),
            )
            return c, slip

        def _tradeability_flags(sym: str, d: str, prev_d: str | None) -> dict[str, Any]:
            """
            用日线 OHLCV 做一个“能不能在开盘成交”的粗估（研究用途）：
            - halt：volume/amount=0（通常就是停牌/无成交）
            - locked_limit_up/down：一字板（high==low）且开盘涨跌幅达到阈值
            """
            m = maps_by_symbol.get(sym) or {}
            op = (m.get("open_by_date") or {}).get(d)
            hp = (m.get("high_by_date") or {}).get(d)
            lp = (m.get("low_by_date") or {}).get(d)
            cp_prev = (m.get("close_by_date") or {}).get(prev_d) if prev_d else None
            vol = (m.get("volume_by_date") or {}).get(d)
            amt = (m.get("amount_by_date") or {}).get(d)

            halted = False
            if halt_zero:
                try:
                    halted = bool((vol is not None and float(vol) == 0.0) or (amt is not None and float(amt) == 0.0))
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    halted = False

            pct_open = None
            try:
                if op is not None and cp_prev is not None and float(cp_prev) > 0:
                    pct_open = float(op) / float(cp_prev) - 1.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pct_open = None

            one_word = False
            try:
                if op is not None and hp is not None and lp is not None:
                    op2 = float(op)
                    hp2 = float(hp)
                    lp2 = float(lp)
                    tol = max(1e-9, abs(hp2) * 1e-6)
                    one_word = bool(abs(hp2 - lp2) <= tol and abs(op2 - hp2) <= tol and abs(op2 - lp2) <= tol)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                one_word = False

            locked_up = bool(float(lim_up) > 0 and one_word and (pct_open is not None) and float(pct_open) >= float(lim_up) - 1e-6)
            locked_dn = bool(float(lim_dn) > 0 and one_word and (pct_open is not None) and float(pct_open) <= -float(lim_dn) + 1e-6)
            return {
                "halted": bool(halted),
                "one_word": bool(one_word),
                "pct_open": (float(pct_open) if pct_open is not None else None),
                "locked_limit_up": bool(locked_up),
                "locked_limit_down": bool(locked_dn),
            }

        def _week_key(d: str) -> tuple[int, int] | None:
            dt = parse_date_any_opt(d)
            if dt is None:
                return None
            iso = dt.date().isocalendar()
            return int(getattr(iso, "year", iso[0])), int(getattr(iso, "week", iso[1]))

        def _rank_universe(d_exec: str) -> list[dict[str, Any]]:
            """
            返回：[{symbol, score, rank}] 按 score 由高到低。
            约定：用 maps_by_symbol 的 *_prev_by_date[d_exec]（上一交易日可得）避免未来函数。
            """

            idx_m = rs_idx_map if isinstance(rs_idx_map, dict) else (idx_map if isinstance(idx_map, dict) else {})

            def _get(m: dict, key: str) -> float | None:
                dct = m.get(key) if isinstance(m.get(key), dict) else None
                v = (dct or {}).get(d_exec) if isinstance(dct, dict) else None
                try:
                    x = None if v is None else float(v)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    return None
                return float(x) if (x is not None and math.isfinite(float(x))) else None

            rows: list[dict[str, Any]] = []

            # 非 factor7：直接用动量排序（更快）
            if rot_rank_mode2 in {"mom63", "mom126"}:
                key = "mom63_prev_by_date" if rot_rank_mode2 == "mom63" else "mom126_prev_by_date"
                for sym in maps_by_symbol:
                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(d_exec)
                    try:
                        op2 = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        op2 = 0.0
                    if op2 <= 0 or (not math.isfinite(op2)):
                        continue
                    v = _get(m, key)
                    if v is None:
                        continue
                    rows.append({"symbol": str(sym), "score": float(v)})
                rows.sort(key=lambda r: (-float(r.get("score") or 0.0), str(r.get("symbol") or "")))
                for i, r in enumerate(rows):
                    r["rank"] = int(i)
                return rows

            # factor7：复用 BBB 的 7 因子加权口径（但应用在“全体候选”）
            rs63_vals: list[float] = []
            rs126_vals: list[float] = []
            adx_vals: list[float] = []
            vol20_vals: list[float] = []
            atrp_vals: list[float] = []
            dd_vals: list[float] = []
            liq_vals: list[float] = []
            boll_vals: list[float] = []
            ar_vals: list[float] = []
            vr_vals: list[float] = []

            idx_m63 = _get(idx_m, "mom63_prev_by_date")
            idx_m126 = _get(idx_m, "mom126_prev_by_date")

            for sym in maps_by_symbol:
                m = maps_by_symbol.get(sym) or {}
                op = (m.get("open_by_date") or {}).get(d_exec)
                try:
                    op2 = float(op) if op is not None else 0.0
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    op2 = 0.0
                if op2 <= 0 or (not math.isfinite(op2)):
                    continue

                m63 = _get(m, "mom63_prev_by_date")
                m126 = _get(m, "mom126_prev_by_date")
                adx = _get(m, "adx14_prev_by_date")
                v20 = _get(m, "vol20_prev_by_date")
                atrp = _get(m, "atr14_pct_prev_by_date")
                dd = _get(m, "dd252_prev_by_date")
                liq = _get(m, "amount_avg20_prev_by_date")
                boll = _get(m, "boll_bw_rel_prev_by_date")
                ar = _get(m, "amount_ratio_prev_by_date")
                vr = _get(m, "volume_ratio_prev_by_date")

                rs63 = (float(m63) - float(idx_m63)) if (m63 is not None and idx_m63 is not None) else None
                rs126 = (float(m126) - float(idx_m126)) if (m126 is not None and idx_m126 is not None) else None

                if rs63 is not None:
                    rs63_vals.append(float(rs63))
                if rs126 is not None:
                    rs126_vals.append(float(rs126))
                if adx is not None:
                    adx_vals.append(float(adx))
                if v20 is not None:
                    vol20_vals.append(float(v20))
                if atrp is not None:
                    atrp_vals.append(float(atrp))
                if dd is not None:
                    dd_vals.append(float(dd))
                if liq is not None:
                    liq_vals.append(float(liq))
                if boll is not None:
                    boll_vals.append(float(boll))
                if ar is not None:
                    ar_vals.append(float(ar))
                if vr is not None:
                    vr_vals.append(float(vr))

                # ma20_dist：作为 tie-break（越贴近MA20越“安全入场”，更像左侧/回踩）
                ma20 = _get(m, "ma20_prev_by_date")
                ma20_dist = 1e9
                if ma20 is not None and float(ma20) > 0:
                    ma20_dist = float(abs(float(op2) / float(ma20) - 1.0))

                rows.append(
                    {
                        "symbol": str(sym),
                        "ma20_dist": float(ma20_dist),
                        "rs63": rs63,
                        "rs126": rs126,
                        "adx": adx,
                        "vol20": v20,
                        "atrp": atrp,
                        "dd252": dd,
                        "liq": liq,
                        "boll": boll,
                        "ar": ar,
                        "vr": vr,
                    }
                )

            rs63_vals.sort()
            rs126_vals.sort()
            adx_vals.sort()
            vol20_vals.sort()
            atrp_vals.sort()
            dd_vals.sort()
            liq_vals.sort()
            boll_vals.sort()
            ar_vals.sort()
            vr_vals.sort()

            def _norm_rank(sorted_vals: list[float], v: float | None, *, higher_better: bool) -> float:
                if not sorted_vals or v is None:
                    return 0.0
                try:
                    x = float(v)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    return 0.0
                if not math.isfinite(x):
                    return 0.0
                n = len(sorted_vals)
                if n <= 1:
                    base = 0.0
                else:
                    lo = bisect.bisect_left(sorted_vals, x)
                    hi = bisect.bisect_right(sorted_vals, x)
                    pos = (float(lo) + float(hi - 1)) / 2.0
                    pct = float(pos) / float(n - 1)
                    base = (pct - 0.5) * 2.0
                return float(base if higher_better else -base)

            for r in rows:
                rs_parts: list[float] = []
                if r.get("rs63") is not None:
                    rs_parts.append(_norm_rank(rs63_vals, r.get("rs63"), higher_better=True))
                if r.get("rs126") is not None:
                    rs_parts.append(_norm_rank(rs126_vals, r.get("rs126"), higher_better=True))
                rs_sc = float(sum(rs_parts) / len(rs_parts)) if rs_parts else 0.0

                trend_sc = _norm_rank(adx_vals, r.get("adx"), higher_better=True)

                vol_parts: list[float] = []
                if r.get("vol20") is not None:
                    vol_parts.append(_norm_rank(vol20_vals, r.get("vol20"), higher_better=False))
                if r.get("atrp") is not None:
                    vol_parts.append(_norm_rank(atrp_vals, r.get("atrp"), higher_better=False))
                vol_sc = float(sum(vol_parts) / len(vol_parts)) if vol_parts else 0.0

                dd_sc = _norm_rank(dd_vals, r.get("dd252"), higher_better=True)
                liq_sc = _norm_rank(liq_vals, r.get("liq"), higher_better=True)
                boll_sc = _norm_rank(boll_vals, r.get("boll"), higher_better=False)

                vc_parts: list[float] = []
                if r.get("ar") is not None:
                    vc_parts.append(_norm_rank(ar_vals, r.get("ar"), higher_better=True))
                if r.get("vr") is not None:
                    vc_parts.append(_norm_rank(vr_vals, r.get("vr"), higher_better=True))
                vc_sc = float(sum(vc_parts) / len(vc_parts)) if vc_parts else 0.0

                score7 = (
                    float(w7.get("rs", 0.0)) * float(rs_sc)
                    + float(w7.get("trend", 0.0)) * float(trend_sc)
                    + float(w7.get("vol", 0.0)) * float(vol_sc)
                    + float(w7.get("drawdown", 0.0)) * float(dd_sc)
                    + float(w7.get("liquidity", 0.0)) * float(liq_sc)
                    + float(w7.get("boll", 0.0)) * float(boll_sc)
                    + float(w7.get("volume", 0.0)) * float(vc_sc)
                )
                r["score"] = float(score7)

            rows.sort(key=lambda r: (-float(r.get("score") or 0.0), float(r.get("ma20_dist") or 1e9), str(r.get("symbol") or "")))
            for i, r in enumerate(rows):
                r["rank"] = int(i)
            return rows

        # --------- simulate ----------
        init_cash = float(capital_yuan or 0.0)
        if (not math.isfinite(init_cash)) or init_cash <= 0:
            return {"ok": False, "error": f"capital_yuan 非法：{capital_yuan}"}

        cash = float(init_cash)
        positions: dict[str, dict[str, Any]] = {}
        pending_buys: dict[str, dict[str, Any]] = {}
        want_set: set[str] = set()
        want_list: list[str] = []

        equity_dates: list[str] = []
        equity_vals: list[float] = []
        equity_cash: list[float] = []
        equity_pos: list[int] = []
        trades: list[dict[str, Any]] = []

        skipped: dict[str, int] = {
            "halt": 0,
            "limit": 0,
            "gap_skip": 0,
            "quality_gate": 0,
            "pullback_filter": 0,
            "trend_filter": 0,
            "restart_blocked": 0,
            "exit_blocked": 0,
            "bad_price": 0,
            "no_cash": 0,
            "min_trade_notional": 0,
            "max_positions": 0,
            "max_exposure": 0,
            "max_turnover": 0,
            "max_corr": 0,
            "theme_limit": 0,
        }

        peak_equity = float(init_cash)
        meltdown_mode = False
        meltdown_trigger_date = None
        cooldown_until_idx = 0
        recovery_mode = False
        restart_ma_days = int(dd_restart_ma_days)

        last_week = None
        week_no = -1

        # 记账：买入侧换手（用于 max_turnover_pct）
        turnover_buy_yuan_by_day: dict[str, float] = {}

        for i, d2 in enumerate(calendar):
            prev_d = str(calendar[i - 1]) if i > 0 else None

            wk = _week_key(str(d2))
            is_week_start = False
            if wk != last_week:
                week_no += 1
                last_week = wk
                is_week_start = True

            can_rebalance_now = bool(
                (not meltdown_mode)
                and bool(is_week_start)
                and bool(prev_d)
                and (int(week_no) % int(reb_weeks) == 0)
                and int(i) >= int(cooldown_until_idx)
            )
            if can_rebalance_now and recovery_mode:
                if restart_ma_days > 0 and (not _index_above_ma20_consecutive(end_idx=int(i), days=int(restart_ma_days))):
                    skipped["restart_blocked"] += 1
                    can_rebalance_now = False
                else:
                    # 通过“重启闸门”：允许恢复正常交易
                    recovery_mode = False

            # 1) 先处理“熔断模式”：只卖不买
            if meltdown_mode:
                for sym, pos in list(positions.items()):
                    flags = _tradeability_flags(sym, str(d2), prev_d)
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        continue
                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(str(d2))
                    try:
                        px = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px <= 0 or sh <= 0:
                        continue

                    cost2, slip2 = _cost_for_trade(sym, str(d2))
                    cash_out, fee_out = cash_sell(shares=int(sh), price=float(px), cost=cost2)
                    cash += float(cash_out)
                    entry_cash = float(pos.get("entry_cash") or 0.0)
                    pnl = float(cash_out - entry_cash)
                    ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
                    trades.append(
                        {
                            "asset": "stock",
                            "symbol": sym,
                            "entry_date": str(pos.get("entry_date")),
                            "exit_date": str(d2),
                            "entry_price": float(pos.get("entry_price") or 0.0),
                            "exit_price": float(px),
                            "exit_price_type": "open",
                            "shares": int(sh),
                            "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(entry_cash),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(pos.get("hold_days") or 0),
                            "reason": "dd_stop",
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )
                    del positions[sym]

                pending_buys = {}
                want_set = set()
                want_list = []

            # 2) 周频再平衡：更新 want_set，并把“该卖的”标记出来
            if can_rebalance_now:
                ranked = _rank_universe(str(d2))
                ranks = {str(r.get("symbol") or ""): int(r.get("rank") or 0) for r in ranked}

                keep = []
                for sym in positions:
                    rk = ranks.get(sym)
                    if rk is None:
                        continue
                    if int(rk) <= int(hold_n + buffer_n):
                        keep.append(sym)

                # keep 过多就按 rank 留最强的
                keep.sort(key=lambda s: int(ranks.get(s, 1_000_000)))
                keep = keep[: int(hold_n)]

                want: list[str] = list(keep)
                for r in ranked:
                    if len(want) >= int(hold_n):
                        break
                    sym = str(r.get("symbol") or "")
                    if not sym or sym in want:
                        continue
                    want.append(sym)

                want_list = list(want)
                want_set = set(want_list)

                # 取消不再想买的 pending
                for sym in list(pending_buys.keys()):
                    if sym not in want_set:
                        del pending_buys[sym]

                # 标记要卖的
                for sym, pos in positions.items():
                    if sym in want_set:
                        continue
                    pos["pending_exit_reason"] = "rebalance"

            # 3) 执行 pending exits（包含 rebalance/dd_stop）
            if not meltdown_mode:
                for sym, pos in list(positions.items()):
                    reason = str(pos.get("pending_exit_reason") or "").strip()
                    if not reason:
                        continue
                    flags = _tradeability_flags(sym, str(d2), prev_d)
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        continue
                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(str(d2))
                    try:
                        px = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px <= 0 or sh <= 0:
                        continue

                    cost2, slip2 = _cost_for_trade(sym, str(d2))
                    cash_out, fee_out = cash_sell(shares=int(sh), price=float(px), cost=cost2)
                    cash += float(cash_out)
                    entry_cash = float(pos.get("entry_cash") or 0.0)
                    pnl = float(cash_out - entry_cash)
                    ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
                    trades.append(
                        {
                            "asset": "stock",
                            "symbol": sym,
                            "entry_date": str(pos.get("entry_date")),
                            "exit_date": str(d2),
                            "entry_price": float(pos.get("entry_price") or 0.0),
                            "exit_price": float(px),
                            "exit_price_type": "open",
                            "shares": int(sh),
                            "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(entry_cash),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(pos.get("hold_days") or 0),
                            "reason": str(reason),
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )
                    del positions[sym]

            # 4) 周频再平衡：生成 buys（entry/分批），落到 pending_buys
            if can_rebalance_now:
                # 动态 max_positions/max_exposure：允许用 regime_profile 做“只降不升”的控仓
                label = str(regime_by_date.get(str(d2)) or "unknown")
                rp = risk_profile_for_regime(label)
                mp = int(hold_n)
                me = float(max_exposure_pct or 0.0) if float(max_exposure_pct or 0.0) > 0 else float(rp.max_exposure_pct)
                me = float(max(0.0, min(me, 1.0)))

                # equity at open（执行价）
                eq_open = float(cash)
                for sym, pos in positions.items():
                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(str(d2))
                    try:
                        px = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px > 0 and sh > 0:
                        eq_open += float(px) * float(sh)

                max_invest = float(eq_open) * float(me)
                invested_mv = 0.0
                for sym, pos in positions.items():
                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(str(d2))
                    try:
                        px = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px > 0 and sh > 0:
                        invested_mv += float(px) * float(sh)

                # 买入预算：等权给“缺的仓位”
                cur_cnt = int(len(positions))
                need_cnt = int(max(0, int(mp) - int(cur_cnt)))
                if need_cnt > 0:
                    per_budget = float(max_invest / float(mp)) if mp > 0 else float(max_invest)

                    buy_turn_cap = None
                    if float(max_turn) > 0:
                        buy_turn_cap = float(eq_open) * float(max_turn)
                    buy_turn_used = float(turnover_buy_yuan_by_day.get(str(d2), 0.0))

                    # 依次补齐 want_set 里缺的
                    for sym in want_list:
                        if sym in positions:
                            continue
                        if sym in pending_buys:
                            continue

                        if int(len(positions) + len(pending_buys)) >= int(mp):
                            break

                        if invested_mv >= max_invest and max_invest > 0:
                            skipped["max_exposure"] += 1
                            break

                        m = maps_by_symbol.get(sym) or {}
                        op = (m.get("open_by_date") or {}).get(str(d2))
                        try:
                            px = float(op) if op is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            px = 0.0
                        if px <= 0 or (not math.isfinite(px)):
                            skipped["bad_price"] += 1
                            continue

                        flags = _tradeability_flags(sym, str(d2), prev_d)
                        if bool(flags.get("halted")):
                            skipped["halt"] += 1
                            continue
                        if bool(flags.get("locked_limit_up")):
                            skipped["limit"] += 1
                            continue

                        # gap 过滤：open > prev_close*(1+gap) 跳过（周一冲高最常见）
                        if gap_rot > 0 and prev_d:
                            prev_close = (m.get("close_by_date") or {}).get(prev_d)
                            try:
                                pc = float(prev_close) if prev_close is not None else 0.0
                            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                                pc = 0.0
                            if pc > 0 and px > pc * (1.0 + float(gap_rot)):
                                skipped["gap_skip"] += 1
                                continue

                        # 同主题限仓（可选）
                        name, theme = _name_theme(sym)
                        if max_theme > 0 and theme:
                            theme_cnt = 0
                            for psym, ppos in positions.items():
                                th0 = ppos.get("theme")
                                if not th0:
                                    _, th0 = _name_theme(psym)
                                    if th0:
                                        ppos["theme"] = str(th0)
                                if str(th0 or "").strip() == str(theme):
                                    theme_cnt += 1
                            if int(theme_cnt) >= int(max_theme):
                                skipped["theme_limit"] += 1
                                continue

                        # 相关性过滤（可选）
                        if max_corr2 > 0 and corr_abs_tail is not None and i > 0 and positions:
                            ra = _daily_returns_tail(sym, end_idx=int(i) - 1, window_days=60)
                            too_corr = False
                            if ra:
                                for psym in positions:
                                    rb = _daily_returns_tail(str(psym), end_idx=int(i) - 1, window_days=60)
                                    if not rb:
                                        continue
                                    c = corr_abs_tail(ra, rb, min_overlap=20)
                                    if c is None:
                                        continue
                                    if float(c) >= float(max_corr2):
                                        too_corr = True
                                        break
                            if too_corr:
                                skipped["max_corr"] += 1
                                continue

                        budget = min(float(cash), float(per_budget))
                        if buy_turn_cap is not None:
                            remain_turn = float(buy_turn_cap) - float(buy_turn_used)
                            if remain_turn <= 0:
                                skipped["max_turnover"] += 1
                                break
                            budget = min(float(budget), float(remain_turn))

                        if budget <= max(0.0, float(cost.buy_fee_yuan) + float(cost.buy_fee_min_yuan) + 1e-6):
                            skipped["no_cash"] += 1
                            continue

                        # 2日分批：首日只下 50%（剩下的留到次日；如果首日没成交，则次日全量）
                        budget1 = float(budget)
                        if int(split_days) >= 2:
                            budget1 = float(budget) * 0.5

                        pending_buys[sym] = {
                            "asset": "stock",
                            "symbol": sym,
                            "name": (str(name) if name else None),
                            "theme": (str(theme) if theme else None),
                            "created_at": str(d2),
                            "budget_total": float(budget),
                            "budget_remaining": float(budget),
                            "budget_today": float(budget1),
                            "reason": "entry",
                        }

            # 5) 执行 pending buys（每天都尝试；周一/周二分批由 budget_today 控制）
            if not meltdown_mode:
                for sym, od in list(pending_buys.items()):
                    if sym in positions:
                        del pending_buys[sym]
                        continue
                    if want_set and sym not in want_set:
                        del pending_buys[sym]
                        continue

                    budget_today = float(od.get("budget_today") or 0.0)
                    budget_remaining = float(od.get("budget_remaining") or 0.0)
                    if budget_today <= 0 or budget_remaining <= 0:
                        del pending_buys[sym]
                        continue

                    m = maps_by_symbol.get(sym) or {}
                    op = (m.get("open_by_date") or {}).get(str(d2))
                    try:
                        px = float(op) if op is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px = 0.0
                    if px <= 0 or (not math.isfinite(px)):
                        skipped["bad_price"] += 1
                        continue

                    ok_q, _why_q = _stock_quality_ok(str(sym), exec_date=str(d2), exec_price=float(px))
                    if not ok_q:
                        skipped["quality_gate"] += 1
                        continue

                    flags = _tradeability_flags(sym, str(d2), prev_d)
                    if bool(flags.get("halted")):
                        skipped["halt"] += 1
                        continue
                    if bool(flags.get("locked_limit_up")):
                        skipped["limit"] += 1
                        continue

                    if gap_rot > 0 and prev_d:
                        prev_close = (m.get("close_by_date") or {}).get(prev_d)
                        try:
                            pc = float(prev_close) if prev_close is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            pc = 0.0
                        if pc > 0 and px > pc * (1.0 + float(gap_rot)):
                            skipped["gap_skip"] += 1
                            continue

                    # A：趋势里买回撤（pullback in uptrend）
                    if trend_ma > 0 and prev_d:
                        prev_close = (m.get("close_by_date") or {}).get(prev_d)
                        ma60 = (m.get("ma60_prev_by_date") or {}).get(str(d2))
                        try:
                            pc2 = float(prev_close) if prev_close is not None else 0.0
                            ma60_2 = float(ma60) if ma60 is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            pc2 = 0.0
                            ma60_2 = 0.0
                        if pc2 <= 0 or ma60_2 <= 0 or (not (pc2 > ma60_2)):
                            skipped["trend_filter"] += 1
                            continue

                    if pullback_ma20_dist_max > 0:
                        ma20 = (m.get("ma20_prev_by_date") or {}).get(str(d2))
                        try:
                            ma20_2 = float(ma20) if ma20 is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            ma20_2 = 0.0
                        if ma20_2 <= 0:
                            skipped["pullback_filter"] += 1
                            continue
                        dist = float(abs(float(px) / float(ma20_2) - 1.0))
                        if (not math.isfinite(dist)) or dist > float(pullback_ma20_dist_max) + 1e-12:
                            skipped["pullback_filter"] += 1
                            continue

                    budget = min(float(cash), float(budget_today), float(budget_remaining))
                    if float(max_turn) > 0:
                        eq_ref = float(equity_vals[-1]) if equity_vals else float(init_cash)
                        cap = float(eq_ref) * float(max_turn)
                        used = float(turnover_buy_yuan_by_day.get(str(d2), 0.0))
                        remain = float(cap) - float(used)
                        if remain <= 0:
                            skipped["max_turnover"] += 1
                            break
                        budget = min(float(budget), float(remain))
                    if budget <= max(0.0, float(cost.buy_fee_yuan) + float(cost.buy_fee_min_yuan) + 1e-6):
                        skipped["no_cash"] += 1
                        continue

                    cost2, slip2 = _cost_for_trade(sym, str(d2))
                    sh = calc_shares_for_capital(capital_yuan=float(budget), price=float(px), cost=cost2, lot_size=lot)
                    if sh <= 0:
                        skipped["min_trade_notional"] += 1
                        continue

                    if float(min_trade_notional2) > 0:
                        notional = float(sh) * float(px)
                        if notional + 1e-6 < float(min_trade_notional2):
                            skipped["min_trade_notional"] += 1
                            continue

                    cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost2)
                    if cash_in > cash + 1e-6:
                        skipped["no_cash"] += 1
                        continue

                    cash -= float(cash_in)
                    turnover_buy_yuan_by_day[str(d2)] = float(turnover_buy_yuan_by_day.get(str(d2), 0.0)) + float(cash_in)
                    positions[sym] = {
                        "asset": "stock",
                        "symbol": sym,
                        "name": od.get("name"),
                        "theme": od.get("theme"),
                        "entry_date": str(d2),
                        "entry_price": float(px),
                        "entry_price_type": "open",
                        "shares": int(sh),
                        "entry_cash": float(cash_in),
                        "buy_fee_yuan": float(fee_in),
                        "hold_days": 0,
                    }
                    trades.append(
                        {
                            "asset": "stock",
                            "symbol": sym,
                            "entry_date": str(d2),
                            "exit_date": None,
                            "entry_price": float(px),
                            "exit_price": None,
                            "exit_price_type": None,
                            "shares": int(sh),
                            "buy_fee_yuan": float(fee_in),
                            "sell_fee_yuan": None,
                            "entry_cash": float(cash_in),
                            "exit_cash": None,
                            "pnl_net": None,
                            "pnl_net_pct": None,
                            "hold_days": 0,
                            "reason": str(od.get("reason") or "entry"),
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )

                    # 更新剩余预算：按 cash_in 扣（包含费用，口径一致）
                    od["budget_remaining"] = float(budget_remaining - cash_in)
                    if float(od.get("budget_remaining") or 0.0) <= max(0.0, float(min_trade_notional2) * 0.5):
                        del pending_buys[sym]
                    else:
                        # 首日成交后：次日把 budget_today 切到“剩余预算”（2日分批）
                        od["budget_today"] = float(od.get("budget_remaining") or 0.0) if int(split_days) >= 2 else float(0.0)
                        pending_buys[sym] = od

            # 6) 收盘记账：equity curve + DD 熔断判定（触发后次日开盘卖）
            eq_close = float(cash)
            pos_cnt = 0
            for sym, pos in positions.items():
                m = maps_by_symbol.get(sym) or {}
                cl = (m.get("close_by_date") or {}).get(str(d2))
                try:
                    px = float(cl) if cl is not None else 0.0
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    px = 0.0
                sh = int(pos.get("shares") or 0)
                if px > 0 and sh > 0:
                    eq_close += float(px) * float(sh)
                    pos["last_close"] = float(px)
                    pos_cnt += 1
                # 持有天数（用于 report 展示；不参与策略逻辑）
                pos["hold_days"] = int(pos.get("hold_days") or 0) + 1

            equity_dates.append(str(d2))
            equity_vals.append(float(eq_close))
            equity_cash.append(float(cash))
            equity_pos.append(int(pos_cnt))

            if float(eq_close) > float(peak_equity):
                peak_equity = float(eq_close)

            if (not meltdown_mode) and dd_stop > 0 and float(peak_equity) > 0:
                dd_now = float(eq_close) / float(peak_equity) - 1.0
                if float(dd_now) <= -float(dd_stop) - 1e-12:
                    meltdown_mode = True
                    meltdown_trigger_date = str(d2)
                    cooldown_until_idx = int(i + 1 + dd_cool)
                    for sym, pos in positions.items():
                        pos["pending_exit_reason"] = "dd_stop"
                    pending_buys = {}
                    want_set = set()
                    want_list = []

            # 熔断后：清空且冷却结束 -> 恢复正常交易
            if meltdown_mode and (not positions) and int(i) >= int(cooldown_until_idx):
                meltdown_mode = False
                recovery_mode = bool(int(restart_ma_days) > 0)

        # ---- 输出口径对齐 BBB ----
        as_of = str(equity_dates[-1]) if equity_dates else None
        equity_last = float(equity_vals[-1]) if equity_vals else float(init_cash)

        dd = _max_drawdown(equity_vals)

        # trades：把“entry-only”记录剔除（只统计已闭合的交易，用于 PF/Payoff）
        trades_closed = [t for t in trades if isinstance(t, dict) and t.get("exit_date")]

        # open positions（按 as_of 收盘估值）
        open_positions: list[dict[str, Any]] = []
        for sym, pos in positions.items():
            sh = int(pos.get("shares") or 0)
            if sh <= 0:
                continue
            m = maps_by_symbol.get(sym) or {}
            last_close = (m.get("close_by_date") or {}).get(as_of) if as_of else None
            try:
                last_close2 = float(last_close) if last_close is not None else float(pos.get("last_close") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                last_close2 = float(pos.get("last_close") or 0.0)
            if last_close2 <= 0:
                continue
            mv = float(last_close2) * float(sh)
            cash_out, fee_out = cash_sell(shares=int(sh), price=float(last_close2), cost=cost)
            entry_cash = float(pos.get("entry_cash") or 0.0)
            pnl = float(cash_out - entry_cash)
            ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
            open_positions.append(
                {
                    "asset": "stock",
                    "symbol": sym,
                    "name": (pos.get("name") if pos.get("name") else None),
                    "theme": (pos.get("theme") if pos.get("theme") else None),
                    "as_of": as_of,
                    "entry_date": str(pos.get("entry_date")),
                    "entry_price": float(pos.get("entry_price") or 0.0),
                    "entry_price_type": str(pos.get("entry_price_type") or "open"),
                    "shares": int(sh),
                    "entry_cash": float(entry_cash),
                    "last_close": float(last_close2),
                    "market_value": float(mv),
                    "exit_cash_if_sell_now": float(cash_out),
                    "sell_fee_yuan_if_sell_now": float(fee_out),
                    "pnl_net_if_sell_now": float(pnl),
                    "pnl_net_pct_if_sell_now": float(ret),
                    "regime_at_entry": str(pos.get("regime_at_entry") or "unknown"),
                }
            )

        open_positions.sort(key=lambda x: (str(x.get("symbol") or "")))

        # PF/Payoff（只用已闭合交易；另给一个“含未平仓强平”的口径）
        gp = 0.0
        gl = 0.0
        wn = 0
        ln = 0
        for t in trades_closed:
            try:
                p = float(t.get("pnl_net") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(p)) or p == 0:
                continue
            if p > 0:
                gp += float(p)
                wn += 1
            else:
                gl += float(p)
                ln += 1
        avg_win_yuan = (float(gp) / float(wn)) if wn > 0 else None
        avg_loss_yuan = (float(abs(gl)) / float(ln)) if ln > 0 else None
        profit_factor = (float(gp) / float(abs(gl))) if gl < 0 else None
        payoff = (float(avg_win_yuan) / float(avg_loss_yuan)) if (avg_win_yuan is not None and avg_loss_yuan is not None and float(avg_loss_yuan) > 0) else None

        years = None
        try:
            if equity_dates:
                d0 = parse_date_any_opt(equity_dates[0])
                d1 = parse_date_any_opt(equity_dates[-1])
                if d0 is not None and d1 is not None:
                    years = max(0.0, float((d1 - d0).days) / 365.25)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            years = None

        total_ret = float(equity_last / init_cash - 1.0) if init_cash > 0 else 0.0
        cagr = None
        try:
            if years and float(years) > 0 and init_cash > 0 and equity_last > 0:
                cagr = float((equity_last / init_cash) ** (1.0 / float(years)) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cagr = None

        equity_liquidated = float(cash) + float(sum(float(p.get("exit_cash_if_sell_now") or 0.0) for p in open_positions))
        total_ret_liquidated = float(equity_liquidated / init_cash - 1.0) if init_cash > 0 else 0.0
        cagr_liquidated = None
        try:
            if years and float(years) > 0 and init_cash > 0 and equity_liquidated > 0:
                cagr_liquidated = float((equity_liquidated / init_cash) ** (1.0 / float(years)) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cagr_liquidated = None

        # 含未平仓的 PF/Payoff
        gp2 = float(gp)
        gl2 = float(gl)
        wn2 = int(wn)
        ln2 = int(ln)
        for p0 in open_positions:
            try:
                p = float(p0.get("pnl_net_if_sell_now") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(p)) or p == 0:
                continue
            if p > 0:
                gp2 += float(p)
                wn2 += 1
            else:
                gl2 += float(p)
                ln2 += 1
        avg_win_yuan2 = (float(gp2) / float(wn2)) if wn2 > 0 else None
        avg_loss_yuan2 = (float(abs(gl2)) / float(ln2)) if ln2 > 0 else None
        profit_factor2 = (float(gp2) / float(abs(gl2))) if gl2 < 0 else None
        payoff2 = (float(avg_win_yuan2) / float(avg_loss_yuan2)) if (avg_win_yuan2 is not None and avg_loss_yuan2 is not None and float(avg_loss_yuan2) > 0) else None

        tail_n = 260
        eq_tail = [
            {
                "date": equity_dates[i],
                "equity": float(equity_vals[i]),
                "cash": float(equity_cash[i]),
                "label": str(regime_by_date.get(equity_dates[i]) or "unknown"),
                "positions": int(equity_pos[i]) if i < len(equity_pos) else None,
            }
            for i in range(max(0, len(equity_dates) - tail_n), len(equity_dates))
        ]

        return {
            "ok": True,
            "mode": "portfolio",
            "strategy": str(strat),
            "as_of": as_of,
            "summary": {
                "capital_yuan": float(init_cash),
                "equity_last": float(equity_last),
                "cash_last": float(cash),
                "equity_liquidated": float(equity_liquidated),
                "period_years": float(years) if years is not None else None,
                "total_return": float(total_ret),
                "total_return_liquidated": float(total_ret_liquidated),
                "cagr": cagr,
                "cagr_liquidated": cagr_liquidated,
                "max_drawdown": dd,
                "trades": int(len(trades_closed)),
                "wins": int(wn),
                "win_rate": float(wn / len(trades_closed)) if trades_closed else 0.0,
                "pnl_gross_profit_yuan": float(gp),
                "pnl_gross_loss_yuan": float(abs(gl)),
                "profit_factor": float(profit_factor) if profit_factor is not None else None,
                "avg_win_yuan": float(avg_win_yuan) if avg_win_yuan is not None else None,
                "avg_loss_yuan": float(avg_loss_yuan) if avg_loss_yuan is not None else None,
                "payoff": float(payoff) if payoff is not None else None,
                "pnl_gross_profit_yuan_incl_open": float(gp2),
                "pnl_gross_loss_yuan_incl_open": float(abs(gl2)),
                "profit_factor_incl_open": float(profit_factor2) if profit_factor2 is not None else None,
                "avg_win_yuan_incl_open": float(avg_win_yuan2) if avg_win_yuan2 is not None else None,
                "avg_loss_yuan_incl_open": float(avg_loss_yuan2) if avg_loss_yuan2 is not None else None,
                "payoff_incl_open": float(payoff2) if payoff2 is not None else None,
                "open_positions": int(len(open_positions)),
                "last_regime": str(regime_by_date.get(as_of) or "unknown") if as_of else "unknown",
                "skipped": skipped,
                "stock_quality_gate": stock_gate_params,
                "symbols": int(len(maps_by_symbol)),
                "orders_next_open": 0,
                "turnover": None,
                "capacity": None,
                "regime_stats": None,
                "rot": {
                    "rebalance_weeks": int(reb_weeks),
                    "hold_n": int(hold_n),
                    "buffer_n": int(buffer_n),
                    "rank_mode": str(rot_rank_mode2),
                    "gap_max": float(gap_rot),
                    "split_exec_days": int(split_days),
                    "portfolio_dd_stop": float(dd_stop),
                    "portfolio_dd_cooldown_days": int(dd_cool),
                    "portfolio_dd_restart_ma_days": int(restart_ma_days),
                    "meltdown_trigger_date": meltdown_trigger_date,
                },
            },
            "orders_next_open": [],
            "equity_curve_tail": eq_tail,
            "positions": open_positions,
            "trades": trades_closed[-800:],
            "errors": map_errors[:50],
        }

    # BBB（组合）：用“实盘撮合思路”跑（支持分批止盈/让利润跑），别再拿单标的回测结果硬拼凑组合了。
    if is_bbb:
        tp_cfg = TakeProfitConfig()

        # 牛熊标签（可选）
        regime_by_date = _compute_regime_label_by_date(df_index_daily=df_regime_index_daily, calendar_dates=calendar) if df_regime_index_daily is not None else {}
        mom63_by_date = _build_mom_63d_by_date(df_regime_index_daily) if df_regime_index_daily is not None else {}

        # ---- 成本/约束：按“执行日”动态估 slippage（避免未来函数） ----
        slip_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def _slippage_for(sym: str, d: str) -> dict[str, Any]:
            key = (str(sym), str(d))
            if key in slip_cache:
                return slip_cache[key]
            m = maps_by_symbol.get(sym) or {}
            amt = (m.get("amount_avg20_prev_by_date") or {}).get(d)
            atrp = (m.get("atr14_pct_prev_by_date") or {}).get(d) if float(slip_vm) > 0 else None
            slip_bps2 = estimate_slippage_bps(
                mode=str(slip_mode or "none"),
                amount_avg20_yuan=(float(amt) if amt is not None else None),
                atr_pct=(float(atrp) if atrp is not None else None),
                bps=float(slip_bps),
                ref_amount_yuan=float(slip_ref_amt),
                min_bps=float(slip_bps_min),
                max_bps=float(slip_bps_max),
                unknown_bps=float(slip_unknown_bps),
                vol_mult=float(slip_vm),
            )
            out = {
                "slippage_mode": str(slip_mode or "none"),
                "slippage_bps": float(slip_bps2),
                "slippage_rate": float(bps_to_rate(float(slip_bps2))),
                "amount_avg20_prev_yuan": (float(amt) if amt is not None else None),
                "atr14_pct_prev": (float(atrp) if atrp is not None else None),
            }
            slip_cache[key] = out
            return out

        def _cost_for_trade(sym: str, d: str) -> tuple[TradeCost, dict[str, Any]]:
            slip = _slippage_for(sym, d)
            r = float(slip.get("slippage_rate") or 0.0)
            c = TradeCost(
                buy_cost=float(cost.buy_cost) + float(r),
                sell_cost=float(cost.sell_cost) + float(r),
                buy_fee_yuan=float(cost.buy_fee_yuan),
                sell_fee_yuan=float(cost.sell_fee_yuan),
                buy_fee_min_yuan=float(cost.buy_fee_min_yuan),
                sell_fee_min_yuan=float(cost.sell_fee_min_yuan),
            )
            return c, slip

        def _tradeability_flags(sym: str, d: str, prev_d: str | None) -> dict[str, Any]:
            """
            用日线 OHLCV 做一个“能不能在开盘成交”的粗估（研究用途）：
            - halt：volume/amount=0（通常就是停牌/无成交）
            - locked_limit_up/down：一字板（high==low）且开盘涨跌幅达到阈值
            """
            m = maps_by_symbol.get(sym) or {}
            op = (m.get("open_by_date") or {}).get(d)
            hp = (m.get("high_by_date") or {}).get(d)
            lp = (m.get("low_by_date") or {}).get(d)
            cp_prev = (m.get("close_by_date") or {}).get(prev_d) if prev_d else None
            vol = (m.get("volume_by_date") or {}).get(d)
            amt = (m.get("amount_by_date") or {}).get(d)

            halted = False
            if halt_zero:
                try:
                    halted = bool((vol is not None and float(vol) == 0.0) or (amt is not None and float(amt) == 0.0))
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    halted = False

            pct_open = None
            try:
                if op is not None and cp_prev is not None and float(cp_prev) > 0:
                    pct_open = float(op) / float(cp_prev) - 1.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pct_open = None

            one_word = False
            try:
                if op is not None and hp is not None and lp is not None:
                    op2 = float(op)
                    hp2 = float(hp)
                    lp2 = float(lp)
                    tol = max(1e-9, abs(hp2) * 1e-6)
                    one_word = bool(abs(hp2 - lp2) <= tol and abs(op2 - hp2) <= tol and abs(op2 - lp2) <= tol)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                one_word = False

            locked_up = bool(float(lim_up) > 0 and one_word and (pct_open is not None) and float(pct_open) >= float(lim_up) - 1e-6)
            locked_dn = bool(float(lim_dn) > 0 and one_word and (pct_open is not None) and float(pct_open) <= -float(lim_dn) + 1e-6)
            return {
                "halted": bool(halted),
                "one_word": bool(one_word),
                "pct_open": (float(pct_open) if pct_open is not None else None),
                "locked_limit_up": bool(locked_up),
                "locked_limit_down": bool(locked_dn),
            }

        # 每个标的的“执行日集合”
        exec_by_symbol: dict[str, dict[str, Any]] = {}
        exec_errors: list[dict[str, Any]] = []
        for sym, df in dfs_by_symbol.items():
            sym2 = str(sym)
            if sym2 not in maps_by_symbol:
                continue
            if core_w and sym2 in core_w:
                # core 只用于填仓位，不跟 BBB 信号走（否则又回到“空仓拖累”）
                continue
            st = _prepare_bbb_exec_dates(
                df,
                bbb_entry_ma=int(bbb_entry_ma or 20),
                bbb_dist_ma_max=float(bbb_dist_ma_max or 0.12),
                bbb_max_above_20w=float(bbb_max_above_20w or 0.05),
                bbb_min_weeks=int(bbb_min_weeks or 60),
                bbb_require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
                bbb_require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
                bbb_require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
                weekly_anchor_ma=20,
                end_date=end_date,
            )
            if not bool(st.get("ok", False)):
                exec_errors.append({"symbol": sym2, "error": str(st.get("error") or "prepare_exec_failed")})
                continue
            exec_by_symbol[sym2] = st

        if (not exec_by_symbol) and (not core_w):
            return {"ok": False, "error": "所有标的都没有有效 BBB 信号集", "errors": (map_errors + exec_errors)[:50]}

        # entry_by_date：避免每天扫全表
        entry_by_date: dict[str, list[str]] = {}
        for sym, st in exec_by_symbol.items():
            for d in (st.get("entry_exec_dates") or set()):
                entry_by_date.setdefault(str(d), []).append(str(sym))

        # 组合撮合（账户级）
        cash = max(0.0, float(capital_yuan or 0.0))
        init_cash = float(cash)
        positions: dict[str, dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []
        equity_dates: list[str] = []
        equity_vals: list[float] = []
        equity_cash: list[float] = []
        equity_pos: list[int] = []
        skipped: dict[str, int] = {
            "max_positions": 0,
            "max_exposure": 0,
            "no_cash": 0,
            "min_trade_notional": 0,
            "bad_price": 0,
            "gap_skip": 0,
            "halt": 0,
            "limit": 0,
            "exit_blocked": 0,
            "turnover_buy": 0,
            "max_corr": 0,
            "theme_limit": 0,
            "restart_blocked": 0,
            "quality_gate": 0,
        }

        # cooldown：按“日历索引”计数
        cooldown_until_idx: dict[str, int] = {}
        lot = max(1, int(lot_size or 100))
        min_hold_days2 = max(0, int(bbb_min_hold_days or 0))
        cooldown_days2 = max(0, int(bbb_cooldown_days or 0))

        buy_turnover_total_yuan = 0.0

        last_label = "unknown"
        peak_equity = float(init_cash)
        meltdown_mode = False
        meltdown_trigger_date = None
        cooldown_until_idx_portfolio = 0
        recovery_mode = False
        for i, d in enumerate(calendar):
            d2 = str(d)
            if not d2:
                continue

            # 环境：在“开盘执行”口径，用上一交易日已知的 mom_63d（避免未来函数）
            label = str(regime_by_date.get(d2) or "unknown")
            last_label = label
            prev_d = str(calendar[i - 1]) if i > 0 else None
            mom_prev = mom63_by_date.get(prev_d) if prev_d else None
            bull_phase = classify_bull_phase(label=label, mom_63d=mom_prev, cfg=tp_cfg)
            rp = risk_profile_for_regime(label)
            if recovery_mode:
                if dd_restart_ma_days > 0 and _index_above_ma20_consecutive(end_idx=int(i), days=int(dd_restart_ma_days)):
                    recovery_mode = False
                else:
                    skipped["restart_blocked"] += 1

            mp = int(max_positions or 0) if int(max_positions or 0) > 0 else int(rp.max_positions)
            mp = max(1, min(mp, 50))
            me = float(max_exposure_pct or 0.0) if float(max_exposure_pct or 0.0) > 0 else float(rp.max_exposure_pct)
            # vol targeting：只降不升（用指数历史波动率缩放 max_exposure_pct；避免未来函数用 i-1）
            if vol_tgt > 0 and i > 0:
                v = _index_realized_vol_ann(int(i) - 1)
                if v is not None and float(v) > 0:
                    me = float(me) * float(min(1.0, float(vol_tgt) / float(v)))
            me = max(0.0, min(float(me), 1.0))
            max_invest = float(init_cash) * float(me)
            buy_turnover_budget_today = None
            buy_turnover_used_today = 0.0
            if max_turn > 0:
                eq_prev = float(equity_vals[-1]) if equity_vals else float(init_cash)
                if eq_prev > 0:
                    buy_turnover_budget_today = float(eq_prev) * float(max_turn)

            # --- portfolio dd stop：熔断模式只卖不买（研究用途） ---
            if meltdown_mode:
                for sym, pos in list(positions.items()):
                    flags = _tradeability_flags(sym, d2, prev_d)
                    # 卖出：停牌 / 一字跌停 -> 无法成交，延迟到下一个交易日再试
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        pos["pending_exit_reason"] = "dd_stop"
                        pos.setdefault("pending_exit_since", d2)
                        continue

                    m = maps_by_symbol.get(sym) or {}
                    px = (m.get("open_by_date") or {}).get(d2)
                    try:
                        px2 = float(px) if px is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px2 = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px2 <= 0 or sh <= 0:
                        continue

                    cost2, slip2 = _cost_for_trade(sym, d2)
                    cash_out, fee_out = cash_sell(shares=int(sh), price=float(px2), cost=cost2)
                    cash += float(cash_out)
                    entry_cash = float(pos.get("entry_cash") or 0.0)
                    pnl = float(cash_out - entry_cash)
                    ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
                    trades.append(
                        {
                            "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": sym,
                            "entry_date": str(pos.get("entry_date")),
                            "exit_date": d2,
                            "entry_price": float(pos.get("entry_price") or 0.0),
                            "exit_price": float(px2),
                            "exit_price_type": "open",
                            "shares": int(sh),
                            "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(entry_cash),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(pos.get("hold_days") or 0),
                            "reason": "dd_stop",
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )
                    del positions[sym]
                    cooldown_until_idx[sym] = int(i + 1 + cooldown_days2)

                # 熔断状态：不做后续买入/加仓（含 core fill），直接收盘记账
            # --- open exits（先卖再买）---
            for sym, pos in list(positions.items()):
                if meltdown_mode:
                    continue
                if bool(pos.get("core", False)):
                    continue
                st = exec_by_symbol.get(sym)
                if st is None:
                    continue

                # full exits：hard/trail 永远优先；soft 仅在非 bull 环境（别把牛市里的震荡当末日）
                pending = str(pos.get("pending_exit_reason") or "").strip()
                reason = pending or None
                if not reason:
                    if d2 in (st.get("hard_exec_dates") or set()):
                        reason = "hard"
                    elif d2 in (st.get("trail_exec_dates") or set()):
                        reason = "trail"
                    elif bull_phase is None and d2 in (st.get("soft_exec_dates") or set()) and int(pos.get("hold_days") or 0) >= int(min_hold_days2):
                        reason = "soft"
                if reason:
                    flags = _tradeability_flags(sym, d2, prev_d)
                    # 卖出：停牌 / 一字跌停 -> 无法成交，延迟到下一个交易日再试
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        pos["pending_exit_reason"] = str(reason)
                        pos.setdefault("pending_exit_since", d2)
                        continue

                    m = maps_by_symbol.get(sym) or {}
                    px = (m.get("open_by_date") or {}).get(d2)
                    try:
                        px2 = float(px) if px is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px2 = 0.0
                    sh = int(pos.get("shares") or 0)
                    if px2 <= 0 or sh <= 0:
                        continue
                    cost2, slip2 = _cost_for_trade(sym, d2)
                    cash_out, fee_out = cash_sell(shares=int(sh), price=float(px2), cost=cost2)
                    cash += float(cash_out)
                    entry_cash = float(pos.get("entry_cash") or 0.0)
                    pnl = float(cash_out - entry_cash)
                    ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
                    trades.append(
                        {
                            "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": sym,
                            "entry_date": str(pos.get("entry_date")),
                            "exit_date": d2,
                            "entry_price": float(pos.get("entry_price") or 0.0),
                            "exit_price": float(px2),
                            "exit_price_type": "open",
                            "shares": int(sh),
                            "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(entry_cash),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(pos.get("hold_days") or 0),
                            "reason": str(reason),
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )
                    del positions[sym]
                    cooldown_until_idx[sym] = int(i + 1 + cooldown_days2)

            # --- partial take profit（慢牛：只卖一次 1/3）---
            if (not meltdown_mode) and bull_phase == "slow" and prev_d:
                for sym, pos in list(positions.items()):
                    if bool(pos.get("core", False)):
                        continue
                    if bool(pos.get("tp1_done", False)):
                        continue
                    # 如果当天要 full exit，就别分批了（先活命）
                    st = exec_by_symbol.get(sym) or {}
                    if d2 in (st.get("hard_exec_dates") or set()) or d2 in (st.get("trail_exec_dates") or set()):
                        continue
                    if bull_phase is None and d2 in (st.get("soft_exec_dates") or set()) and int(pos.get("hold_days") or 0) >= int(min_hold_days2):
                        continue

                    m = maps_by_symbol.get(sym) or {}
                    close_prev = (m.get("close_by_date") or {}).get(prev_d)
                    try:
                        close_prev2 = float(close_prev) if close_prev is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        close_prev2 = 0.0
                    entry_px = float(pos.get("entry_price") or 0.0)
                    if close_prev2 <= 0 or entry_px <= 0:
                        continue
                    if close_prev2 < entry_px * (1.0 + float(tp_cfg.slow_bull_tp1_trigger_ret)):
                        continue

                    sell_sh = calc_tp1_sell_shares(shares=int(pos.get("shares") or 0), lot_size=lot, cfg=tp_cfg)
                    if sell_sh <= 0:
                        continue

                    px = (m.get("open_by_date") or {}).get(d2)
                    try:
                        px2 = float(px) if px is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px2 = 0.0
                    if px2 <= 0:
                        continue

                    if float(min_trade_notional2) > 0:
                        notional = float(sell_sh) * float(px2)
                        if notional + 1e-6 < float(min_trade_notional2):
                            # 分批止盈是“可选动作”，别为了 5 元起步手续费卖成碎渣渣
                            continue

                    flags = _tradeability_flags(sym, d2, prev_d)
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        continue

                    entry_cash_total = float(pos.get("entry_cash") or 0.0)
                    sh_total = int(pos.get("shares") or 0)
                    if sh_total <= 0 or entry_cash_total <= 0:
                        continue

                    cost2, slip2 = _cost_for_trade(sym, d2)
                    cash_out, fee_out = cash_sell(shares=int(sell_sh), price=float(px2), cost=cost2)
                    cash += float(cash_out)

                    alloc = float(entry_cash_total) * float(sell_sh) / float(sh_total)
                    pnl = float(cash_out - alloc)
                    ret = float(pnl / alloc) if alloc > 0 else 0.0
                    trades.append(
                        {
                            "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": sym,
                            "entry_date": str(pos.get("entry_date")),
                            "exit_date": d2,
                            "entry_price": float(pos.get("entry_price") or 0.0),
                            "exit_price": float(px2),
                            "exit_price_type": "open",
                            "shares": int(sell_sh),
                            "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0) * float(sell_sh) / float(sh_total),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(alloc),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(pos.get("hold_days") or 0),
                            "reason": "tp1",
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )

                    # 更新仓位：按成本摊薄
                    pos["tp1_done"] = True
                    pos["entry_cash"] = float(entry_cash_total - alloc)
                    pos["shares"] = int(sh_total - int(sell_sh))

            # --- open entries ---
            todays_syms = [] if (meltdown_mode or recovery_mode) else list(entry_by_date.get(d2) or [])
            if todays_syms:
                # baseline：只用上一交易日可得的数据（避免未来函数）
                def _ma20_dist(sym: str) -> float:
                    m = maps_by_symbol.get(sym) or {}
                    ma20 = (m.get("ma20_prev_by_date") or {}).get(d2)
                    op = (m.get("open_by_date") or {}).get(d2)
                    try:
                        ma = float(ma20) if ma20 is not None else None
                        px = float(op) if op is not None else None
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        ma = None
                        px = None
                    if ma is None or px is None or ma <= 0 or px <= 0:
                        return 1e9
                    return float(abs(px / ma - 1.0))

                if (not need_factor7) or rank_mode == "ma20_dist" or len(todays_syms) <= 1:
                    # 旧口径：越靠近 MA20(上一日) 越优先；其次按 symbol 稳定排序
                    todays_syms.sort(key=lambda s: (_ma20_dist(str(s)), str(s)))
                else:
                    # 7因子加权排序：同一天多个信号抢额度/现金时，用更稳的组合优先级
                    idx_m = rs_idx_map if isinstance(rs_idx_map, dict) else (idx_map if isinstance(idx_map, dict) else {})

                    def _get(m: dict, key: str) -> float | None:
                        dct = m.get(key) if isinstance(m.get(key), dict) else None
                        v = (dct or {}).get(d2) if isinstance(dct, dict) else None
                        try:
                            x = None if v is None else float(v)
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            return None
                        return float(x) if (x is not None and math.isfinite(float(x))) else None

                    # raw 值
                    cand_rows: list[dict[str, Any]] = []
                    rs63_vals: list[float] = []
                    rs126_vals: list[float] = []
                    adx_vals: list[float] = []
                    vol20_vals: list[float] = []
                    atrp_vals: list[float] = []
                    dd_vals: list[float] = []
                    liq_vals: list[float] = []
                    boll_vals: list[float] = []
                    ar_vals: list[float] = []
                    vr_vals: list[float] = []

                    idx_m63 = _get(idx_m, "mom63_prev_by_date")
                    idx_m126 = _get(idx_m, "mom126_prev_by_date")

                    for sym in todays_syms:
                        sym2 = str(sym)
                        m = maps_by_symbol.get(sym2) or {}

                        m63 = _get(m, "mom63_prev_by_date")
                        m126 = _get(m, "mom126_prev_by_date")
                        rs63 = (float(m63) - float(idx_m63)) if (m63 is not None and idx_m63 is not None) else m63
                        rs126 = (float(m126) - float(idx_m126)) if (m126 is not None and idx_m126 is not None) else m126

                        adx = _get(m, "adx14_prev_by_date")
                        v20 = _get(m, "vol20_prev_by_date")
                        atrp = _get(m, "atr14_pct_prev_by_date")
                        dd = _get(m, "dd252_prev_by_date")
                        liq0 = _get(m, "amount_avg20_prev_by_date")
                        liq = float(math.log1p(float(liq0))) if (liq0 is not None and float(liq0) > 0) else None
                        boll = _get(m, "boll_bw_rel_prev_by_date")
                        ar = _get(m, "amount_ratio_prev_by_date")
                        vr = _get(m, "volume_ratio_prev_by_date")

                        if rs63 is not None:
                            rs63_vals.append(float(rs63))
                        if rs126 is not None:
                            rs126_vals.append(float(rs126))
                        if adx is not None:
                            adx_vals.append(float(adx))
                        if v20 is not None:
                            vol20_vals.append(float(v20))
                        if atrp is not None:
                            atrp_vals.append(float(atrp))
                        if dd is not None:
                            dd_vals.append(float(dd))
                        if liq is not None:
                            liq_vals.append(float(liq))
                        if boll is not None:
                            boll_vals.append(float(boll))
                        if ar is not None:
                            ar_vals.append(float(ar))
                        if vr is not None:
                            vr_vals.append(float(vr))

                        cand_rows.append(
                            {
                                "symbol": sym2,
                                "ma20_dist": float(_ma20_dist(sym2)),
                                "rs63": rs63,
                                "rs126": rs126,
                                "adx": adx,
                                "vol20": v20,
                                "atrp": atrp,
                                "dd252": dd,
                                "liq": liq,
                                "boll": boll,
                                "ar": ar,
                                "vr": vr,
                            }
                        )

                    rs63_vals.sort()
                    rs126_vals.sort()
                    adx_vals.sort()
                    vol20_vals.sort()
                    atrp_vals.sort()
                    dd_vals.sort()
                    liq_vals.sort()
                    boll_vals.sort()
                    ar_vals.sort()
                    vr_vals.sort()

                    import bisect

                    def _norm_rank(sorted_vals: list[float], v: float | None, *, higher_better: bool) -> float:
                        if not sorted_vals or v is None:
                            return 0.0
                        try:
                            x = float(v)
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            return 0.0
                        if not math.isfinite(x):
                            return 0.0
                        n = len(sorted_vals)
                        if n <= 1:
                            base = 0.0
                        else:
                            lo = bisect.bisect_left(sorted_vals, x)
                            hi = bisect.bisect_right(sorted_vals, x)
                            pos = (float(lo) + float(hi - 1)) / 2.0
                            pct = float(pos) / float(n - 1)
                            base = (pct - 0.5) * 2.0
                        return float(base if higher_better else -base)

                    for r in cand_rows:
                        rs_parts: list[float] = []
                        if r.get("rs63") is not None:
                            rs_parts.append(_norm_rank(rs63_vals, r.get("rs63"), higher_better=True))
                        if r.get("rs126") is not None:
                            rs_parts.append(_norm_rank(rs126_vals, r.get("rs126"), higher_better=True))
                        rs_sc = float(sum(rs_parts) / len(rs_parts)) if rs_parts else 0.0

                        trend_sc = _norm_rank(adx_vals, r.get("adx"), higher_better=True)

                        vol_parts: list[float] = []
                        if r.get("vol20") is not None:
                            vol_parts.append(_norm_rank(vol20_vals, r.get("vol20"), higher_better=False))
                        if r.get("atrp") is not None:
                            vol_parts.append(_norm_rank(atrp_vals, r.get("atrp"), higher_better=False))
                        vol_sc = float(sum(vol_parts) / len(vol_parts)) if vol_parts else 0.0

                        dd_sc = _norm_rank(dd_vals, r.get("dd252"), higher_better=True)
                        liq_sc = _norm_rank(liq_vals, r.get("liq"), higher_better=True)
                        boll_sc = _norm_rank(boll_vals, r.get("boll"), higher_better=False)

                        vc_parts: list[float] = []
                        if r.get("ar") is not None:
                            vc_parts.append(_norm_rank(ar_vals, r.get("ar"), higher_better=True))
                        if r.get("vr") is not None:
                            vc_parts.append(_norm_rank(vr_vals, r.get("vr"), higher_better=True))
                        vc_sc = float(sum(vc_parts) / len(vc_parts)) if vc_parts else 0.0

                        score7 = (
                            float(w7.get("rs", 0.0)) * float(rs_sc)
                            + float(w7.get("trend", 0.0)) * float(trend_sc)
                            + float(w7.get("vol", 0.0)) * float(vol_sc)
                            + float(w7.get("drawdown", 0.0)) * float(dd_sc)
                            + float(w7.get("liquidity", 0.0)) * float(liq_sc)
                            + float(w7.get("boll", 0.0)) * float(boll_sc)
                            + float(w7.get("volume", 0.0)) * float(vc_sc)
                        )
                        r["score7"] = float(score7)

                    cand_rows.sort(key=lambda r: (-float(r.get("score7") or 0.0), float(r.get("ma20_dist") or 1e9), str(r.get("symbol") or "")))
                    todays_syms = [str(r.get("symbol") or "") for r in cand_rows if str(r.get("symbol") or "")]

            invested = float(sum(float(p.get("entry_cash") or 0.0) for p in positions.values()))
            per_budget = float(max_invest / float(mp)) if mp > 0 else float(max_invest)
            theme_counts: dict[str, int] = {}
            if max_theme > 0 and infer_theme is not None:
                for psym, pos in positions.items():
                    if bool(pos.get("core", False)):
                        continue
                    th0 = pos.get("theme")
                    if not th0:
                        _, th0 = _name_theme(psym)
                        if th0:
                            pos["theme"] = str(th0)
                    th = str(th0 or "").strip()
                    if th:
                        theme_counts[th] = int(theme_counts.get(th, 0)) + 1

            pos_cnt = int(sum(1 for p in positions.values() if not bool(p.get("core", False))))

            def _equity_and_core_mv_at_open() -> tuple[float, float]:
                mv_all = 0.0
                mv_core = 0.0
                for psym, ppos in positions.items():
                    sh = int(ppos.get("shares") or 0)
                    if sh <= 0:
                        continue
                    m0 = maps_by_symbol.get(str(psym)) or {}
                    op0 = (m0.get("open_by_date") or {}).get(d2)
                    try:
                        px0 = float(op0) if op0 is not None else None
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px0 = None
                    if px0 is None or (not math.isfinite(float(px0))) or float(px0) <= 0:
                        try:
                            px0 = float(ppos.get("last_close") or ppos.get("entry_price") or 0.0)
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            px0 = 0.0
                    if px0 is None or (not math.isfinite(float(px0))) or float(px0) <= 0:
                        continue
                    mv0 = float(sh) * float(px0)
                    mv_all += float(mv0)
                    if bool(ppos.get("core", False)):
                        mv_core += float(mv0)
                eq = float(cash) + float(mv_all)
                return float(eq), float(mv_core)

            def _sell_core_to_free_alloc(*, target_alloc_yuan: float, max_sell_mv_yuan: float | None = None) -> tuple[float, float]:
                """
                为了给 BBB 新入场腾预算/现金：卖出 core（按成本 alloc 口径）。
                返回：(本次从 core 里“释放”的 entry_cash, 本次卖出的市值(不含费))。
                """
                nonlocal cash
                tgt = float(target_alloc_yuan or 0.0)
                if tgt <= 0:
                    return 0.0, 0.0

                mv_budget = None
                if max_sell_mv_yuan is not None:
                    mv_budget = max(0.0, float(max_sell_mv_yuan))
                    if mv_budget <= 0:
                        return 0.0, 0.0

                # 先挑“市值更大”的 core 卖，减少碎单
                cores: list[tuple[float, str, float]] = []
                for csym, cpos in positions.items():
                    if not bool(cpos.get("core", False)):
                        continue
                    sh_total = int(cpos.get("shares") or 0)
                    if sh_total <= 0:
                        continue
                    m0 = maps_by_symbol.get(csym) or {}
                    op0 = (m0.get("open_by_date") or {}).get(d2)
                    try:
                        px0 = float(op0) if op0 is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        px0 = 0.0
                    if px0 <= 0 or (not math.isfinite(px0)):
                        continue
                    mv0 = float(sh_total) * float(px0)
                    if mv0 <= 0:
                        continue
                    cores.append((mv0, str(csym), float(px0)))

                cores.sort(key=lambda x: (-float(x[0]), str(x[1])))
                freed = 0.0
                sold_mv = 0.0
                for _, csym, px0 in cores:
                    if freed >= tgt - 1e-6:
                        break
                    if mv_budget is not None and sold_mv >= mv_budget - 1e-6:
                        break
                    cpos = positions.get(csym)
                    if not isinstance(cpos, dict):
                        continue
                    sh_total = int(cpos.get("shares") or 0)
                    if sh_total <= 0:
                        continue

                    flags = _tradeability_flags(csym, d2, prev_d)
                    if bool(flags.get("halted")) or bool(flags.get("locked_limit_down")):
                        skipped["exit_blocked"] += 1
                        continue

                    entry_cash_total = float(cpos.get("entry_cash") or 0.0)
                    if entry_cash_total <= 0:
                        continue
                    alloc_per_share = float(entry_cash_total) / float(sh_total) if sh_total > 0 else 0.0
                    if alloc_per_share <= 0:
                        continue

                    need_alloc = max(0.0, float(tgt - freed))
                    target_sh = int(math.ceil(float(need_alloc) / float(alloc_per_share)))
                    sell_sh = int(math.ceil(float(target_sh) / float(lot))) * int(lot)
                    sell_sh = min(int(sell_sh), int(sh_total))
                    sell_sh = int((int(sell_sh) // int(lot)) * int(lot))
                    if sell_sh <= 0:
                        continue

                    if mv_budget is not None:
                        rem_mv = float(mv_budget - sold_mv)
                        max_sh_by_mv = (int(int(rem_mv / float(px0)) // int(lot)) * int(lot)) if (rem_mv > 0 and float(px0) > 0) else 0
                        if max_sh_by_mv <= 0:
                            continue
                        sell_sh = int(min(int(sell_sh), int(max_sh_by_mv)))
                        sell_sh = int((int(sell_sh) // int(lot)) * int(lot))
                        if sell_sh <= 0:
                            continue

                    alloc = float(entry_cash_total) * float(sell_sh) / float(sh_total)
                    cost2, slip2 = _cost_for_trade(csym, d2)
                    cash_out, fee_out = cash_sell(shares=int(sell_sh), price=float(px0), cost=cost2)
                    cash += float(cash_out)

                    pnl = float(cash_out - alloc)
                    ret = float(pnl / alloc) if alloc > 0 else 0.0
                    trades.append(
                        {
                            "asset": str(cpos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": str(csym),
                            "entry_date": str(cpos.get("entry_date")),
                            "exit_date": d2,
                            "entry_price": float(cpos.get("entry_price") or 0.0),
                            "exit_price": float(px0),
                            "exit_price_type": "open",
                            "shares": int(sell_sh),
                            "buy_fee_yuan": float(cpos.get("buy_fee_yuan") or 0.0) * float(sell_sh) / float(sh_total),
                            "sell_fee_yuan": float(fee_out),
                            "entry_cash": float(alloc),
                            "exit_cash": float(cash_out),
                            "pnl_net": float(pnl),
                            "pnl_net_pct": float(ret),
                            "hold_days": int(cpos.get("hold_days") or 0),
                            "reason": "core_sell",
                            "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                            "mode": "portfolio",
                        }
                    )

                    # 更新 core 仓位：按成本摊薄
                    remain_sh = int(sh_total - int(sell_sh))
                    if remain_sh <= 0:
                        del positions[csym]
                    else:
                        cpos["shares"] = int(remain_sh)
                        cpos["entry_cash"] = float(entry_cash_total - alloc)
                        cpos["buy_fee_yuan"] = float(cpos.get("buy_fee_yuan") or 0.0) * float(remain_sh) / float(sh_total)

                    freed += float(alloc)
                    sold_mv += float(sell_sh) * float(px0)
                return float(freed), float(sold_mv)

            for sym in todays_syms:
                sym2 = str(sym)
                if sym2 in positions:
                    continue
                if int(i) < int(cooldown_until_idx.get(sym2, 0)):
                    continue
                if int(pos_cnt) >= int(mp):
                    skipped["max_positions"] += 1
                    continue
                if invested >= max_invest and (not any(bool(p.get("core", False)) for p in positions.values())):
                    skipped["max_exposure"] += 1
                    continue

                m = maps_by_symbol.get(sym2) or {}
                op = (m.get("open_by_date") or {}).get(d2)
                try:
                    px = float(op) if op is not None else 0.0
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    px = 0.0
                if px <= 0 or (not math.isfinite(px)):
                    skipped["bad_price"] += 1
                    continue

                ok_q, _why_q = _stock_quality_ok(sym2, exec_date=d2, exec_price=float(px))
                if not ok_q:
                    skipped["quality_gate"] += 1
                    continue

                flags = _tradeability_flags(sym2, d2, prev_d)
                # 买入：停牌 / 一字涨停 -> 当天没法成交，直接跳过（不排队，别YY）
                if bool(flags.get("halted")):
                    skipped["halt"] += 1
                    continue
                if bool(flags.get("locked_limit_up")):
                    skipped["limit"] += 1
                    continue

                # 防“开盘一脚踩山顶”：open > prev_close*(1+gap) 则跳过（不引入未来函数）
                if gap_max > 0 and prev_d:
                    prev_close = (m.get("close_by_date") or {}).get(prev_d)
                    try:
                        pc = float(prev_close) if prev_close is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        pc = 0.0
                    if pc > 0 and px > pc * (1.0 + float(gap_max)):
                        skipped["gap_skip"] += 1
                        continue

                # 同主题限仓（可选）
                name, theme = _name_theme(sym2)
                if max_theme > 0 and theme:
                    if int(theme_counts.get(str(theme), 0)) >= int(max_theme):
                        skipped["theme_limit"] += 1
                        continue

                # 相关性过滤（可选，避免买一堆“看似分散实际同涨同跌”的东西；避免未来函数用 i-1）
                if max_corr2 > 0 and corr_abs_tail is not None and i > 0 and positions:
                    ra = _daily_returns_tail(sym2, end_idx=int(i) - 1, window_days=60)
                    too_corr = False
                    if ra:
                        for psym, ppos in positions.items():
                            if bool(ppos.get("core", False)):
                                continue
                            rb = _daily_returns_tail(str(psym), end_idx=int(i) - 1, window_days=60)
                            if not rb:
                                continue
                            c = corr_abs_tail(ra, rb, min_overlap=20)
                            if c is None:
                                continue
                            if float(c) >= float(max_corr2):
                                too_corr = True
                                break
                    if too_corr:
                        skipped["max_corr"] += 1
                        continue

                cost2, slip2 = _cost_for_trade(sym2, d2)
                remain_budget = max(0.0, float(max_invest - invested))
                budget_cap = min(float(cash), float(remain_budget))

                min_budget = max(0.0, float(cost2.buy_fee_yuan) + float(cost2.buy_fee_min_yuan) + 1e-6)

                # 最小下单金额：默认只影响买入侧（避免 5 元起步手续费把收益磨没）
                min_shares_by_notional = 0
                min_cash_in_by_notional = None
                if float(min_trade_notional2) > 0 and float(px) > 0:
                    try:
                        min_shares_by_notional = int(math.ceil(float(min_trade_notional2) / float(px) / float(lot))) * int(lot)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        min_shares_by_notional = 0
                    if min_shares_by_notional > 0:
                        cash_need, _ = cash_buy(shares=int(min_shares_by_notional), price=float(px), cost=cost2)
                        if cash_need is not None:
                            try:
                                cash_need2 = float(cash_need)
                            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                                cash_need2 = 0.0
                            if cash_need2 > 0 and math.isfinite(cash_need2):
                                min_cash_in_by_notional = float(cash_need2)

                # 预算：默认按 per_budget；但如果你设了 min_trade_notional，那就允许“宁可少持仓，也别买成碎渣渣”
                budget_limit = float(per_budget)
                if min_cash_in_by_notional is not None:
                    budget_limit = float(max(float(budget_limit), float(min_cash_in_by_notional)))

                req_budget = float(min_budget)
                if min_cash_in_by_notional is not None:
                    req_budget = float(max(float(req_budget), float(min_cash_in_by_notional)))

                # core 存在：必要时卖一点 core，把“预算/现金”挪出来（但不能卖穿 core_min_pct2）
                if core_w and budget_cap + 1e-6 < req_budget:
                    for _ in range(3):
                        remain_budget = max(0.0, float(max_invest - invested))
                        budget_cap = min(float(cash), float(remain_budget))
                        if budget_cap + 1e-6 >= req_budget:
                            break

                        max_sell_mv_yuan = None
                        if float(core_min_pct2) > 0:
                            eq_open, core_mv_open = _equity_and_core_mv_at_open()
                            req_core_mv = float(core_min_pct2) * float(eq_open)
                            max_sell_mv_yuan = max(0.0, float(core_mv_open - req_core_mv))
                            if max_sell_mv_yuan <= 0:
                                break

                        need = max(0.0, float(req_budget - budget_cap))
                        freed, _sold_mv = _sell_core_to_free_alloc(target_alloc_yuan=float(need), max_sell_mv_yuan=max_sell_mv_yuan)
                        if freed <= 0:
                            break
                        invested = max(0.0, float(invested) - float(freed))

                remain_budget = max(0.0, float(max_invest - invested))
                budget_cap = min(float(cash), float(remain_budget))
                budget = min(float(budget_cap), float(budget_limit))
                if budget + 1e-6 < req_budget:
                    if min_cash_in_by_notional is not None and budget + 1e-6 < float(min_cash_in_by_notional):
                        skipped["min_trade_notional"] += 1
                    else:
                        skipped["no_cash"] += 1
                    continue

                sh = calc_shares_for_capital(capital_yuan=float(budget), price=float(px), cost=cost2, lot_size=lot)
                if min_shares_by_notional > 0 and min_cash_in_by_notional is not None and budget + 1e-6 >= float(min_cash_in_by_notional):
                    sh = int(max(int(sh), int(min_shares_by_notional)))
                if sh <= 0:
                    skipped["no_cash"] += 1
                    continue

                # 换手约束（KISS：只限制 buy 侧；按“上一日权益”给预算）
                if buy_turnover_budget_today is not None:
                    rem = float(buy_turnover_budget_today) - float(buy_turnover_used_today)
                    max_by_turn = (int(rem / float(px)) // int(lot)) * int(lot) if rem > 0 else 0
                    if max_by_turn <= 0:
                        skipped["turnover_buy"] += 1
                        break
                    sh = int(min(int(sh), int(max_by_turn)))
                    if sh <= 0:
                        skipped["turnover_buy"] += 1
                        break

                if float(min_trade_notional2) > 0:
                    notional = float(sh) * float(px)
                    if notional + 1e-6 < float(min_trade_notional2):
                        skipped["min_trade_notional"] += 1
                        continue

                cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost2)
                if cash_in > cash + 1e-6:
                    skipped["no_cash"] += 1
                    continue

                cash -= float(cash_in)
                invested += float(cash_in)
                pos_cnt += 1
                if buy_turnover_budget_today is not None:
                    buy_turnover_used_today += float(sh) * float(px)
                    buy_turnover_total_yuan += float(sh) * float(px)
                positions[sym2] = {
                    "asset": ("etf" if strat == "bbb_etf" else "stock"),
                    "symbol": sym2,
                    "name": name,
                    "theme": theme,
                    "entry_date": d2,
                    "entry_price": float(px),
                    "entry_price_type": "open",
                    "shares": int(sh),
                    "entry_cash": float(cash_in),
                    "buy_fee_yuan": float(fee_in),
                    "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                    "reason": "entry",
                    "exit_exec": "open",
                    "last_close": float(px),
                    "hold_days": 0,
                    "regime_at_entry": label,
                    "tp1_done": False,
                }

            # --- core fill（吃beta：没信号也尽量别空仓）---
            if core_w and (not recovery_mode) and (not meltdown_mode):
                remain_budget0 = max(0.0, float(max_invest - invested))
                desired_total0 = min(float(cash), float(remain_budget0))
                if desired_total0 > 0 and (float(min_trade_notional2) <= 0 or desired_total0 + 1e-6 >= float(min_trade_notional2)):
                    core_items0 = sorted(core_w.items(), key=lambda x: str(x[0]))
                    tradable: list[tuple[str, float, float]] = []  # (sym, w, px_open)
                    for csym, w in core_items0:
                        m = maps_by_symbol.get(str(csym)) or {}
                        op = (m.get("open_by_date") or {}).get(d2)
                        try:
                            px = float(op) if op is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            px = 0.0
                        if px <= 0 or (not math.isfinite(px)):
                            continue
                        ok_q, _why_q = _stock_quality_ok(str(csym), exec_date=d2, exec_price=float(px))
                        if not ok_q:
                            skipped["quality_gate"] += 1
                            continue
                        flags = _tradeability_flags(str(csym), d2, prev_d)
                        if bool(flags.get("halted")) or bool(flags.get("locked_limit_up")):
                            continue
                        try:
                            w2 = float(w)
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            continue
                        if (not math.isfinite(w2)) or w2 <= 0:
                            continue
                        tradable.append((str(csym), float(w2), float(px)))

                    sw = float(sum(w for _, w, _ in tradable))
                    if tradable and sw > 0:
                        tradable = [(sym, float(w) / float(sw), px) for sym, w, px in tradable]

                    if tradable:
                        # 目标：尽量往权重靠，但优先避免把现金切成小碎单（小资金+5元起步手续费=纯磨损）
                        cur_mv: dict[str, float] = {}
                        total_cur = 0.0
                        for csym, _, px in tradable:
                            sh0 = 0
                            try:
                                sh0 = int((positions.get(str(csym)) or {}).get("shares") or 0)
                            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                                sh0 = 0
                            mv0 = float(sh0) * float(px) if sh0 > 0 else 0.0
                            cur_mv[str(csym)] = float(mv0)
                            total_cur += float(mv0)

                        remaining_cash = float(desired_total0)
                        total_target = float(total_cur) + float(remaining_cash)

                        gaps: list[tuple[float, str, float]] = []  # (gap_mv, sym, px)
                        for csym, w, px in tradable:
                            target_mv = float(total_target) * float(w)
                            gap = float(target_mv) - float(cur_mv.get(str(csym), 0.0))
                            gaps.append((float(gap), str(csym), float(px)))

                        gaps.sort(key=lambda x: (-float(x[0]), str(x[1])))
                        for idx, (gap, csym, px) in enumerate(gaps):
                            remain_budget = max(0.0, float(max_invest - invested))
                            if remain_budget <= 0:
                                break
                            if cash <= 0:
                                break
                            if remaining_cash <= 0:
                                break

                            # 最后一只兜底吃掉尾巴现金
                            if idx == len(gaps) - 1:
                                alloc = float(remaining_cash)
                            else:
                                alloc = float(min(float(remaining_cash), max(0.0, float(gap))))
                            alloc = max(0.0, min(float(alloc), float(cash), float(remain_budget)))
                            if alloc <= 0:
                                continue
                            if float(min_trade_notional2) > 0 and alloc + 1e-6 < float(min_trade_notional2):
                                continue

                            cost2, slip2 = _cost_for_trade(str(csym), d2)
                            min_budget = max(0.0, float(cost2.buy_fee_yuan) + float(cost2.buy_fee_min_yuan) + 1e-6)
                            if alloc <= min_budget:
                                continue

                            sh = calc_shares_for_capital(capital_yuan=float(alloc), price=float(px), cost=cost2, lot_size=lot)
                            if sh <= 0:
                                continue

                            if float(min_trade_notional2) > 0:
                                notional = float(sh) * float(px)
                                if notional + 1e-6 < float(min_trade_notional2):
                                    continue

                            cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost2)
                            if cash_in > cash + 1e-6:
                                continue

                            cash -= float(cash_in)
                            invested += float(cash_in)
                            remaining_cash = max(0.0, float(remaining_cash - cash_in))

                            name, theme = _name_theme(str(csym))
                            if str(csym) in positions:
                                cpos = positions.get(str(csym)) or {}
                                if isinstance(cpos, dict):
                                    cpos["shares"] = int(cpos.get("shares") or 0) + int(sh)
                                    cpos["entry_cash"] = float(cpos.get("entry_cash") or 0.0) + float(cash_in)
                                    cpos["buy_fee_yuan"] = float(cpos.get("buy_fee_yuan") or 0.0) + float(fee_in)
                                    cpos["last_close"] = float(px)
                                    cpos["core"] = True
                                    cpos["tp1_done"] = True
                            else:
                                positions[str(csym)] = {
                                    "asset": ("etf" if strat == "bbb_etf" else "stock"),
                                    "symbol": str(csym),
                                    "name": name,
                                    "theme": theme,
                                    "entry_date": d2,
                                    "entry_price": float(px),
                                    "entry_price_type": "open",
                                    "shares": int(sh),
                                    "entry_cash": float(cash_in),
                                    "buy_fee_yuan": float(fee_in),
                                    "slippage_bps": slip2.get("slippage_bps") if isinstance(slip2, dict) else None,
                                    "reason": "core",
                                    "exit_exec": "open",
                                    "last_close": float(px),
                                    "hold_days": 0,
                                    "regime_at_entry": label,
                                    "tp1_done": True,
                                    "core": True,
                                }

            # --- mark-to-market at close ---
            mv = 0.0
            for sym, pos in positions.items():
                m = maps_by_symbol.get(sym) or {}
                cb = (m.get("close_by_date") or {})
                last_close = float(pos.get("last_close") or pos.get("entry_price") or 0.0)
                px = cb.get(d2)
                if px is not None:
                    try:
                        pxf = float(px)
                        if math.isfinite(pxf) and pxf > 0:
                            last_close = float(pxf)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        pass
                pos["last_close"] = float(last_close)
                pos["hold_days"] = int(pos.get("hold_days") or 0) + 1
                sh = int(pos.get("shares") or 0)
                if sh > 0 and last_close > 0:
                    mv += float(sh) * float(last_close)

            eq = float(cash + mv)
            equity_dates.append(d2)
            equity_vals.append(eq)
            equity_cash.append(float(cash))
            equity_pos.append(int(len(positions)))

            if float(eq) > float(peak_equity):
                peak_equity = float(eq)

            if (not meltdown_mode) and dd_stop > 0 and float(peak_equity) > 0:
                dd_now = float(eq) / float(peak_equity) - 1.0
                if float(dd_now) <= -float(dd_stop) - 1e-12:
                    meltdown_mode = True
                    meltdown_trigger_date = str(d2)
                    cooldown_until_idx_portfolio = int(i + 1 + dd_cool)
                    for pos in positions.values():
                        if isinstance(pos, dict):
                            pos["pending_exit_reason"] = "dd_stop"

            # 熔断后：清空且冷却结束 -> 恢复正常交易（可选重启闸门）
            if meltdown_mode and (not positions) and int(i) >= int(cooldown_until_idx_portfolio):
                meltdown_mode = False
                recovery_mode = bool(int(dd_restart_ma_days) > 0)

        equity_last = float(equity_vals[-1]) if equity_vals else float(cash)
        total_ret = float(equity_last / init_cash - 1.0) if init_cash > 0 else 0.0

        years = None
        cagr = None
        try:
            if equity_dates:
                d0 = parse_date_any_opt(equity_dates[0])
                d1 = parse_date_any_opt(equity_dates[-1])
                if d0 is not None and d1 is not None:
                    years = max(0.0, float((d1 - d0).days) / 365.25)
                    if years > 0 and init_cash > 0 and equity_last > 0:
                        cagr = float((equity_last / init_cash) ** (1.0 / years) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cagr = None

        dd = _max_drawdown(equity_vals)

        wins = int(sum(1 for t in trades if float(t.get("pnl_net") or 0.0) > 0))
        tr_n = int(len(trades))

        # ------ 组合级指标：换手 / 分段统计 / 容量粗估（研究用途） ------
        turnover: dict[str, Any] = {}
        try:
            buy_notional = 0.0
            sell_notional = 0.0
            for t in trades:
                try:
                    sh = int(t.get("shares") or 0)
                    ep = float(t.get("entry_price") or 0.0)
                    xp = float(t.get("exit_price") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
                if sh <= 0:
                    continue
                if ep > 0:
                    buy_notional += float(sh) * float(ep)
                if xp > 0:
                    sell_notional += float(sh) * float(xp)

            turnover_notional = float(buy_notional + sell_notional)
            turnover_pct = float(turnover_notional / init_cash) if init_cash > 0 else None
            turnover_annualized = float(turnover_pct / years) if (turnover_pct is not None and years and years > 0) else None
            turnover = {
                "buy_notional_yuan": float(buy_notional),
                "sell_notional_yuan": float(sell_notional),
                "turnover_notional_yuan": float(turnover_notional),
                "turnover_pct_of_capital": float(turnover_pct) if turnover_pct is not None else None,
                "turnover_annualized": float(turnover_annualized) if turnover_annualized is not None else None,
                "note": "turnover=累计成交额(买+卖)/初始资金；用于估算换手与磨损压力（研究用途）",
            }
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            turnover = {}

        regime_stats: dict[str, Any] = {}
        try:
            import statistics

            by_label: dict[str, list[float]] = {}
            for j in range(1, len(equity_dates)):
                prev = float(equity_vals[j - 1]) if j - 1 < len(equity_vals) else 0.0
                cur = float(equity_vals[j]) if j < len(equity_vals) else 0.0
                if prev <= 0:
                    continue
                r = float(cur / prev - 1.0)
                lbl = str(regime_by_date.get(str(equity_dates[j])) or "unknown")
                by_label.setdefault(lbl, []).append(r)

            by_label_stats: dict[str, Any] = {}
            for lbl, rs in by_label.items():
                if not rs:
                    continue
                prod = 1.0
                for x in rs:
                    prod *= (1.0 + float(x))
                total = float(prod - 1.0)
                avg = float(statistics.fmean(rs)) if rs else 0.0
                vol = float(statistics.pstdev(rs)) if len(rs) >= 2 else 0.0
                by_label_stats[str(lbl)] = {
                    "days": int(len(rs)),
                    "total_return": float(total),
                    "avg_daily_return": float(avg),
                    "vol_daily": float(vol),
                    "best_day": float(max(rs)) if rs else None,
                    "worst_day": float(min(rs)) if rs else None,
                }

            # 连续分段（同 label 连在一起算一个 regime segment）
            segs: list[dict[str, Any]] = []
            if equity_dates and equity_vals:
                cur_lbl = str(regime_by_date.get(str(equity_dates[0])) or "unknown")
                s0 = 0
                for k in range(1, len(equity_dates)):
                    lbl = str(regime_by_date.get(str(equity_dates[k])) or "unknown")
                    if lbl != cur_lbl:
                        s1 = k - 1
                        e0 = float(equity_vals[s0]) if s0 < len(equity_vals) else 0.0
                        e1 = float(equity_vals[s1]) if s1 < len(equity_vals) else 0.0
                        seg_ret = float(e1 / e0 - 1.0) if e0 > 0 else None
                        seg_dd = _max_drawdown([float(x) for x in equity_vals[s0 : s1 + 1]])
                        segs.append(
                            {
                                "label": str(cur_lbl),
                                "start_date": str(equity_dates[s0]),
                                "end_date": str(equity_dates[s1]),
                                "days": int(s1 - s0 + 1),
                                "total_return": float(seg_ret) if seg_ret is not None else None,
                                "max_drawdown": float(seg_dd) if seg_dd is not None else None,
                            }
                        )
                        cur_lbl = lbl
                        s0 = k
                # last
                s1 = len(equity_dates) - 1
                e0 = float(equity_vals[s0]) if s0 < len(equity_vals) else 0.0
                e1 = float(equity_vals[s1]) if s1 < len(equity_vals) else 0.0
                seg_ret = float(e1 / e0 - 1.0) if e0 > 0 else None
                seg_dd = _max_drawdown([float(x) for x in equity_vals[s0 : s1 + 1]])
                segs.append(
                    {
                        "label": str(cur_lbl),
                        "start_date": str(equity_dates[s0]),
                        "end_date": str(equity_dates[s1]),
                        "days": int(s1 - s0 + 1),
                        "total_return": float(seg_ret) if seg_ret is not None else None,
                        "max_drawdown": float(seg_dd) if seg_dd is not None else None,
                    }
                )

            best_seg = None
            worst_seg = None
            for s in segs:
                rr = s.get("total_return")
                dd2 = s.get("max_drawdown")
                if isinstance(rr, (int, float)) and (best_seg is None or float(rr) > float(best_seg.get("total_return") or -1e18)):
                    best_seg = s
                if isinstance(dd2, (int, float)) and (worst_seg is None or float(dd2) < float(worst_seg.get("max_drawdown") or 1e18)):
                    worst_seg = s

            regime_stats = {
                "by_label": by_label_stats,
                "segments_count": int(len(segs)),
                "best_segment": best_seg,
                "worst_segment": worst_seg,
                "segments_tail": segs[-12:] if len(segs) > 12 else segs,
                "note": "by_label=按日收益分组（乘法可交换，分解口径）；segments=连续同label分段（研究用途）",
            }
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            regime_stats = {}

        capacity: dict[str, Any] = {}
        try:
            parts: list[float] = []
            worst_legs: list[dict[str, Any]] = []
            legs_total = 0
            legs_with_liq = 0

            def _add_leg(*, sym: str, side: str, date: str, price: float, shares: int) -> None:
                nonlocal legs_total, legs_with_liq
                legs_total += 1
                if shares <= 0 or price <= 0:
                    return
                notional = float(shares) * float(price)
                m = maps_by_symbol.get(sym) or {}
                liq = (m.get("amount_avg20_prev_by_date") or {}) if isinstance(m.get("amount_avg20_prev_by_date"), dict) else {}
                amt20 = liq.get(date)
                try:
                    a = float(amt20) if amt20 is not None else None
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    a = None
                if a is None or (not math.isfinite(a)) or a <= 0:
                    return
                legs_with_liq += 1
                p = float(notional / a)
                if not math.isfinite(p) or p < 0:
                    return
                parts.append(float(p))
                worst_legs.append(
                    {
                        "symbol": str(sym),
                        "side": str(side),
                        "date": str(date),
                        "notional_yuan": float(notional),
                        "amount_avg20_prev_yuan": float(a),
                        "participation": float(p),
                    }
                )

            for t in trades:
                sym = str(t.get("symbol") or "").strip()
                if not sym:
                    continue
                try:
                    sh = int(t.get("shares") or 0)
                    ep = float(t.get("entry_price") or 0.0)
                    xp = float(t.get("exit_price") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
                ed = str(t.get("entry_date") or "")
                xd = str(t.get("exit_date") or "")
                if ed:
                    _add_leg(sym=sym, side="buy", date=ed, price=float(ep), shares=int(sh))
                if xd:
                    _add_leg(sym=sym, side="sell", date=xd, price=float(xp), shares=int(sh))

            parts2 = sorted([float(x) for x in parts if math.isfinite(float(x)) and float(x) >= 0.0])

            def _q(xs: list[float], q: float) -> float | None:
                if not xs:
                    return None
                qq = max(0.0, min(1.0, float(q)))
                n = len(xs)
                if n == 1:
                    return float(xs[0])
                pos = qq * float(n - 1)
                lo = int(math.floor(pos))
                hi = int(math.ceil(pos))
                if lo == hi:
                    return float(xs[lo])
                w = float(pos - lo)
                return float(xs[lo] * (1.0 - w) + xs[hi] * w)

            worst_legs.sort(key=lambda x: float(x.get("participation") or 0.0), reverse=True)
            worst_legs = worst_legs[:10]

            avg_p = sum(parts2) / len(parts2) if parts2 else None
            capacity = {
                "legs_total": int(legs_total),
                "legs_with_liquidity": int(legs_with_liq),
                "avg_participation": float(avg_p) if avg_p is not None else None,
                "p50_participation": float(_q(parts2, 0.50)) if parts2 else None,
                "p95_participation": float(_q(parts2, 0.95)) if parts2 else None,
                "max_participation": float(max(parts2)) if parts2 else None,
                "worst_legs": worst_legs,
                "note": "participation=单次成交额/近20日均成交额(上一交易日可得)，用于粗估容量/冲击成本（研究用途）",
            }
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            capacity = {}

        # open positions 输出
        open_positions: list[dict[str, Any]] = []
        as_of = str(equity_dates[-1]) if equity_dates else None
        for sym, pos in positions.items():
            sh = int(pos.get("shares") or 0)
            last_close = float(pos.get("last_close") or 0.0)
            mv = float(sh) * float(last_close) if sh > 0 and last_close > 0 else 0.0
            cash_out, fee_out = cash_sell(shares=int(sh), price=float(last_close), cost=cost) if (sh > 0 and last_close > 0) else (0.0, 0.0)
            entry_cash = float(pos.get("entry_cash") or 0.0)
            pnl = float(cash_out - entry_cash)
            ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
            open_positions.append(
                {
                    "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                    "symbol": sym,
                    "name": (pos.get("name") if pos.get("name") else None),
                    "theme": (pos.get("theme") if pos.get("theme") else None),
                    "as_of": as_of,
                    "entry_date": str(pos.get("entry_date")),
                    "entry_price": float(pos.get("entry_price") or 0.0),
                    "entry_price_type": str(pos.get("entry_price_type") or "open"),
                    "shares": int(sh),
                    "entry_cash": float(entry_cash),
                    "last_close": float(last_close),
                    "market_value": float(mv),
                    "exit_cash_if_sell_now": float(cash_out),
                    "sell_fee_yuan_if_sell_now": float(fee_out),
                    "pnl_net_if_sell_now": float(pnl),
                    "pnl_net_pct_if_sell_now": float(ret),
                    "regime_at_entry": str(pos.get("regime_at_entry") or "unknown"),
                    "tp1_done": bool(pos.get("tp1_done", False)),
                    "core": bool(pos.get("core", False)),
                }
            )

        open_positions.sort(key=lambda x: (str(x.get("asset") or ""), str(x.get("symbol") or "")))

        # next open orders（用于“收盘后刷新 -> 次日开盘手动执行”）
        orders_next_open: list[dict[str, Any]] = []
        try:
            if as_of:
                label2 = str(regime_by_date.get(as_of) or last_label or "unknown")
                mom_now = mom63_by_date.get(as_of)
                bull_phase2 = classify_bull_phase(label=label2, mom_63d=mom_now, cfg=tp_cfg)
                rp2 = risk_profile_for_regime(label2)
                mp2 = int(max_positions or 0) if int(max_positions or 0) > 0 else int(rp2.max_positions)
                mp2 = max(1, min(mp2, 50))
                me2 = float(max_exposure_pct or 0.0) if float(max_exposure_pct or 0.0) > 0 else float(rp2.max_exposure_pct)
                me2 = max(0.0, min(me2, 1.0))
                max_invest2 = float(init_cash) * float(me2)

                # 1) 应该卖（次日开盘）：hard/trail/soft/tp1
                sell_syms: set[str] = set()
                sell_cash_est = 0.0
                for sym, pos in list(positions.items()):
                    if bool(pos.get("core", False)):
                        continue
                    m = maps_by_symbol.get(sym) or {}
                    close_today = (m.get("close_by_date") or {}).get(as_of)
                    try:
                        close_today2 = float(close_today) if close_today is not None else 0.0
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        close_today2 = 0.0
                    sh = int(pos.get("shares") or 0)
                    if close_today2 <= 0 or sh <= 0:
                        continue

                    df = dfs_by_symbol.get(sym)
                    if df is None or getattr(df, "empty", True):
                        continue

                    # 信号：只看 as_of（收盘后）
                    dfd = df.copy()
                    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
                    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if dfd.empty or _date_str(dfd.iloc[-1].get("date")) != as_of:
                        continue

                    # 周线 hard/trail（要求 as_of 是周末）
                    dfw = resample_to_weekly(dfd)
                    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

                    hard_today = False
                    trail_today = False
                    if dfw is not None and (not getattr(dfw, "empty", True)) and len(dfw) >= 2:
                        w_close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
                        if "ma50" not in dfw.columns:
                            dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
                        w_ma50 = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
                        hard_w = ((w_close < w_ma50) & (w_close.shift(1) < w_ma50.shift(1))).fillna(False)
                        hard_today = bool(hard_w.iloc[-1]) and (_date_str(dfw.iloc[-1].get("date")) == as_of)

                        w_ma20 = w_close.rolling(window=20, min_periods=20).mean()
                        trail_w = ((w_close < w_ma20) & w_ma20.notna()).fillna(False)
                        trail_today = bool(trail_w.iloc[-1]) and (_date_str(dfw.iloc[-1].get("date")) == as_of)

                    # 日线 soft（非 bull 才启用）
                    soft_today = False
                    if bull_phase2 is None and int(pos.get("hold_days") or 0) >= int(min_hold_days2):
                        if "ma20" not in dfd.columns:
                            dfd = dfd.copy()
                            dfd["ma20"] = dfd["close"].astype(float).rolling(window=20, min_periods=20).mean()
                        if "macd" not in dfd.columns or "macd_signal" not in dfd.columns:
                            dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
                        close_d = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
                        ma20_d = pd.to_numeric(dfd["ma20"], errors="coerce").astype(float)
                        macd_d = pd.to_numeric(dfd["macd"], errors="coerce").astype(float)
                        sig_d = pd.to_numeric(dfd["macd_signal"], errors="coerce").astype(float)
                        bearish = (macd_d < sig_d).fillna(False)
                        bearish2 = (bearish & bearish.shift(1, fill_value=False)).fillna(False)
                        soft_today = bool((bearish2 & (close_d < ma20_d)).fillna(False).iloc[-1])

                    reason = "hard" if hard_today else ("trail" if trail_today else ("soft" if soft_today else None))

                    # 慢牛分批（只卖一次）：优先级低于 full exit
                    if (reason is None) and (bull_phase2 == "slow") and (not bool(pos.get("tp1_done", False))):
                        entry_px = float(pos.get("entry_price") or 0.0)
                        if entry_px > 0 and close_today2 >= entry_px * (1.0 + float(tp_cfg.slow_bull_tp1_trigger_ret)):
                            sell_sh = calc_tp1_sell_shares(shares=int(sh), lot_size=lot, cfg=tp_cfg)
                            if sell_sh > 0:
                                if float(min_trade_notional2) > 0:
                                    notional = float(sell_sh) * float(close_today2)
                                    if notional + 1e-6 < float(min_trade_notional2):
                                        sell_sh = 0
                                if sell_sh <= 0:
                                    continue
                                orders_next_open.append(
                                    {
                                        "side": "sell",
                                        "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                                        "symbol": sym,
                                        "name": None,
                                        "shares": int(sell_sh),
                                        "signal_date": as_of,
                                        "exec": "next_open",
                                        "price_ref": float(close_today2),
                                        "price_ref_type": "close",
                                        "order_type": "market",
                                        "est_cash": None,
                                        "est_fee_yuan": None,
                                        "reason": "tp1",
                                    }
                                )

                    if not reason:
                        continue

                    sell_syms.add(sym)
                    cash_out, fee_out = cash_sell(shares=int(sh), price=float(close_today2), cost=cost)
                    sell_cash_est += float(cash_out)
                    orders_next_open.append(
                        {
                            "side": "sell",
                            "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": sym,
                            "name": None,
                            "shares": int(sh),
                            "signal_date": as_of,
                            "exec": "next_open",
                            "price_ref": float(close_today2),
                            "price_ref_type": "close",
                            "order_type": "market",
                            "est_cash": float(cash_out),
                            "est_fee_yuan": float(fee_out),
                            "reason": str(reason),
                        }
                    )

                # 2) 应该买（次日开盘）：只在“本周周收盘刚发 entry”时列出来（与 BBB 定义一致）
                cash_avail = float(cash + sell_cash_est)

                # core：如果现金不够，允许用 core 卖出“给 entry 腾位置/资金”（只做估算与下单草案）
                core_pool: list[dict[str, Any]] = []
                if core_w:
                    for sym, pos in positions.items():
                        if not bool(pos.get("core", False)):
                            continue
                        m = maps_by_symbol.get(sym) or {}
                        close_today = (m.get("close_by_date") or {}).get(as_of)
                        try:
                            close_today2 = float(close_today) if close_today is not None else 0.0
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            close_today2 = 0.0
                        sh = int(pos.get("shares") or 0)
                        if close_today2 <= 0 or sh <= 0:
                            continue
                        core_pool.append({"symbol": str(sym), "shares": int(sh), "price_ref": float(close_today2), "mv": float(sh) * float(close_today2)})
                    core_pool.sort(key=lambda x: (-float(x.get("mv") or 0.0), str(x.get("symbol") or "")))

                # BBB 持仓上限/仓位预算：只算“非core”的持仓；core 作为填仓位的底仓/流动性池
                invested_remaining = float(sum(float(p.get("entry_cash") or 0.0) for s, p in positions.items() if (s not in sell_syms) and (not bool(p.get("core", False)))))
                invested_remaining = max(0.0, invested_remaining)
                pos_cnt = int(sum(1 for s, p in positions.items() if (s not in sell_syms) and (not bool(p.get("core", False)))))

                buy_candidates: list[dict[str, Any]] = []
                params = BBBParams(
                    entry_ma=max(2, int(bbb_entry_ma or 20)),
                    dist_ma50_max=float(bbb_dist_ma_max or 0.12),
                    max_above_20w=float(bbb_max_above_20w or 0.05),
                    min_weekly_bars_total=int(bbb_min_weeks or 60),
                    require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
                    require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
                    require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
                )

                for sym, df in dfs_by_symbol.items():
                    if core_w and sym in core_w:
                        continue
                    if sym in positions and sym not in sell_syms:
                        continue
                    if sym in sell_syms:
                        continue

                    dfd = df.copy()
                    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
                    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if dfd.empty:
                        continue
                    if _date_str(dfd.iloc[-1].get("date")) != as_of:
                        continue
                    last_close = float(dfd.iloc[-1].get("close") or 0.0)
                    if last_close <= 0:
                        continue

                    dfw = resample_to_weekly(dfd)
                    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if dfw is None or getattr(dfw, "empty", True):
                        continue

                    sig_w = compute_bbb_entry_signal(dfw, dfd, params=params)
                    if sig_w is None or getattr(sig_w, "empty", True) or int(len(sig_w)) != int(len(dfw)):
                        continue
                    if _date_str(dfw.iloc[-1].get("date")) != as_of:
                        continue
                    if not bool(sig_w.iloc[-1]):
                        continue

                    # 分数：越靠近 entry_ma 越优先（避免“太右侧”）
                    entry_ma = max(2, int(bbb_entry_ma or 20))
                    ma_entry = None
                    try:
                        close_w = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
                        ma_entry = float(close_w.rolling(window=entry_ma, min_periods=entry_ma).mean().iloc[-1])
                    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                        ma_entry = None
                    score = 1e9
                    if ma_entry is not None and ma_entry > 0:
                        score = float(abs(last_close / ma_entry - 1.0))

                    buy_candidates.append({"symbol": sym, "asset": ("etf" if strat == "bbb_etf" else "stock"), "price_ref": float(last_close), "score": float(score)})

                buy_candidates.sort(key=lambda x: (float(x.get("score") or 0.0), str(x.get("symbol") or "")))

                def _sell_core_for_cash(*, need_cash_yuan: float) -> None:
                    """
                    下单草案：为了让 entry 可执行，必要时先卖 core（估算用 close 价，执行=次日开盘手动）。
                    """
                    nonlocal cash_avail
                    need = float(need_cash_yuan or 0.0)
                    if need <= 0:
                        return
                    if not core_pool:
                        return

                    for row in core_pool:
                        if need <= 0:
                            break
                        sym = str(row.get("symbol") or "")
                        sh_avail = int(row.get("shares") or 0)
                        px = float(row.get("price_ref") or 0.0)
                        if not sym or sh_avail <= 0 or px <= 0:
                            continue
                        # 估算需要卖多少份：按 close 近似（因为我们是收盘后生成 orders）
                        want = int(math.ceil(float(need) / float(px)))
                        sell_sh = int(math.ceil(float(want) / float(lot))) * int(lot)
                        sell_sh = min(int(sell_sh), int(sh_avail))
                        sell_sh = int((int(sell_sh) // int(lot)) * int(lot))
                        if sell_sh <= 0:
                            continue
                        cash_out, fee_out = cash_sell(shares=int(sell_sh), price=float(px), cost=cost)
                        if cash_out <= 0:
                            continue
                        row["shares"] = int(sh_avail - int(sell_sh))
                        cash_avail += float(cash_out)
                        need -= float(cash_out)
                        orders_next_open.append(
                            {
                                "side": "sell",
                                "asset": ("etf" if strat == "bbb_etf" else "stock"),
                                "symbol": sym,
                                "name": None,
                                "shares": int(sell_sh),
                                "signal_date": as_of,
                                "exec": "next_open",
                                "price_ref": float(px),
                                "price_ref_type": "close",
                                "order_type": "market",
                                "est_cash": float(cash_out),
                                "est_fee_yuan": float(fee_out),
                                "reason": "core_rotate",
                            }
                        )

                per_budget = float(max_invest2 / float(mp2)) if mp2 > 0 else float(max_invest2)
                for c in buy_candidates:
                    if pos_cnt >= mp2:
                        break
                    if invested_remaining >= max_invest2:
                        break

                    px = float(c.get("price_ref") or 0.0)
                    if px <= 0 or (not math.isfinite(px)):
                        continue

                    remain_budget = max(0.0, float(max_invest2 - invested_remaining))
                    budget = min(float(cash_avail), float(per_budget), float(remain_budget))
                    if budget <= max(0.0, float(cost.buy_fee_yuan) + float(cost.buy_fee_min_yuan) + 1e-6):
                        continue

                    sh = calc_shares_for_capital(capital_yuan=float(budget), price=float(px), cost=cost, lot_size=lot)
                    if sh <= 0:
                        continue
                    if float(min_trade_notional2) > 0:
                        notional = float(sh) * float(px)
                        if notional + 1e-6 < float(min_trade_notional2):
                            continue

                    cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost)
                    if cash_in > cash_avail + 1e-6:
                        _sell_core_for_cash(need_cash_yuan=float(cash_in - cash_avail))
                    if cash_in > cash_avail + 1e-6:
                        continue

                    cash_avail -= float(cash_in)
                    invested_remaining += float(cash_in)
                    pos_cnt += 1
                    max_open = float(px) * (1.0 + float(gap_max)) if float(gap_max) > 0 else None
                    orders_next_open.append(
                        {
                            "side": "buy",
                            "asset": str(c.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                            "symbol": str(c.get("symbol") or ""),
                            "name": None,
                            "shares": int(sh),
                            "signal_date": as_of,
                            "exec": "next_open",
                            "price_ref": float(px),
                            "price_ref_type": "close",
                            "gap_max": float(gap_max),
                            "max_open_price": float(max_open) if (max_open is not None) else None,
                            "order_type": "limit",
                            "limit_price": float(max_open) if (max_open is not None) else None,
                            "est_cash": float(cash_in),
                            "est_fee_yuan": float(fee_in),
                            "reason": "entry",
                        }
                    )
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            orders_next_open = []

        # 交易统计：Profit Factor / Payoff（盈亏比），用 pnl_net 口径
        gp = 0.0  # gross profit
        gl = 0.0  # gross loss（负数累加）
        wn = 0
        ln = 0
        for t in trades:
            try:
                p = float(t.get("pnl_net") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(p)) or p == 0:
                continue
            if p > 0:
                gp += float(p)
                wn += 1
            else:
                gl += float(p)
                ln += 1
        avg_win_yuan = (float(gp) / float(wn)) if wn > 0 else None
        avg_loss_yuan = (float(abs(gl)) / float(ln)) if ln > 0 else None
        profit_factor = (float(gp) / float(abs(gl))) if gl < 0 else None
        payoff = (float(avg_win_yuan) / float(avg_loss_yuan)) if (avg_win_yuan is not None and avg_loss_yuan is not None and float(avg_loss_yuan) > 0) else None

        # 期末“强平口径”：把未平仓按 as_of 收盘价卖掉（含卖出费），用于避免 PF 被“长持未平仓收益”扭曲
        equity_liquidated = float(cash) + float(sum(float(p.get("exit_cash_if_sell_now") or 0.0) for p in open_positions))
        total_ret_liquidated = float(equity_liquidated / init_cash - 1.0) if init_cash > 0 else 0.0
        cagr_liquidated = None
        try:
            if years and float(years) > 0 and init_cash > 0 and equity_liquidated > 0:
                cagr_liquidated = float((equity_liquidated / init_cash) ** (1.0 / float(years)) - 1.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cagr_liquidated = None

        gp2 = 0.0
        gl2 = 0.0
        wn2 = 0
        ln2 = 0
        for t in trades:
            try:
                p = float(t.get("pnl_net") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(p)) or p == 0:
                continue
            if p > 0:
                gp2 += float(p)
                wn2 += 1
            else:
                gl2 += float(p)
                ln2 += 1
        for p0 in open_positions:
            try:
                p = float(p0.get("pnl_net_if_sell_now") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
            if (not math.isfinite(p)) or p == 0:
                continue
            if p > 0:
                gp2 += float(p)
                wn2 += 1
            else:
                gl2 += float(p)
                ln2 += 1
        avg_win_yuan2 = (float(gp2) / float(wn2)) if wn2 > 0 else None
        avg_loss_yuan2 = (float(abs(gl2)) / float(ln2)) if ln2 > 0 else None
        profit_factor2 = (float(gp2) / float(abs(gl2))) if gl2 < 0 else None
        payoff2 = (float(avg_win_yuan2) / float(avg_loss_yuan2)) if (avg_win_yuan2 is not None and avg_loss_yuan2 is not None and float(avg_loss_yuan2) > 0) else None

        tail_n = 260
        eq_tail = [
            {
                "date": equity_dates[i],
                "equity": float(equity_vals[i]),
                "cash": float(equity_cash[i]),
                "label": str(regime_by_date.get(equity_dates[i]) or "unknown"),
                "positions": int(equity_pos[i]) if i < len(equity_pos) else None,
            }
            for i in range(max(0, len(equity_dates) - tail_n), len(equity_dates))
        ]

        return {
            "ok": True,
            "mode": "portfolio",
            "strategy": str(strat),
            "as_of": as_of,
            "summary": {
                "capital_yuan": float(init_cash),
                "equity_last": float(equity_last),
                "cash_last": float(cash),
                "equity_liquidated": float(equity_liquidated),
                "period_years": float(years) if years is not None else None,
                "total_return": float(total_ret),
                "total_return_liquidated": float(total_ret_liquidated),
                "cagr": cagr,
                "cagr_liquidated": cagr_liquidated,
                "max_drawdown": dd,
                "portfolio_dd_stop": float(dd_stop),
                "portfolio_dd_cooldown_days": int(dd_cool),
                "portfolio_dd_restart_ma_days": int(dd_restart_ma_days),
                "meltdown_trigger_date": meltdown_trigger_date,
                "stock_quality_gate": stock_gate_params,
                "trades": tr_n,
                "wins": int(wins),
                "win_rate": float(wins / tr_n) if tr_n > 0 else 0.0,
                "pnl_gross_profit_yuan": float(gp),
                "pnl_gross_loss_yuan": float(abs(gl)),
                "profit_factor": float(profit_factor) if profit_factor is not None else None,
                "avg_win_yuan": float(avg_win_yuan) if avg_win_yuan is not None else None,
                "avg_loss_yuan": float(avg_loss_yuan) if avg_loss_yuan is not None else None,
                "payoff": float(payoff) if payoff is not None else None,
                "pnl_gross_profit_yuan_incl_open": float(gp2),
                "pnl_gross_loss_yuan_incl_open": float(abs(gl2)),
                "profit_factor_incl_open": float(profit_factor2) if profit_factor2 is not None else None,
                "avg_win_yuan_incl_open": float(avg_win_yuan2) if avg_win_yuan2 is not None else None,
                "avg_loss_yuan_incl_open": float(avg_loss_yuan2) if avg_loss_yuan2 is not None else None,
                "payoff_incl_open": float(payoff2) if payoff2 is not None else None,
                "open_positions": int(len(open_positions)),
                "last_regime": str(last_label),
                "last_bull_phase": classify_bull_phase(label=str(last_label), mom_63d=mom63_by_date.get(as_of), cfg=tp_cfg) if as_of else None,
                "skipped": skipped,
                "symbols": int(len(exec_by_symbol)),
                "orders_next_open": int(len(orders_next_open)),
                "turnover": turnover,
                "capacity": capacity,
                "regime_stats": regime_stats,
            },
            "orders_next_open": orders_next_open,
            "equity_curve_tail": eq_tail,
            "positions": open_positions,
            "trades": trades[-800:],
            "errors": (map_errors + exec_errors)[:50],
        }

    # ------- 组合级模拟盘（legacy 执行引擎） -------
    # 提前把“候选成交”（entry/exit date+price）抽出来
    candidates: list[dict[str, Any]] = []
    sim_errors: list[dict[str, Any]] = []

    # 抽候选时用“大资金”避免被“资金太小买不到一手”误伤
    probe_cap = max(10000.0, float(capital_yuan or 0.0), 1e6)

    for sym, df in dfs_by_symbol.items():
        sym2 = str(sym)
        if sym2 not in maps_by_symbol:
            continue

        if is_bbb:
            out = simulate_bbb_paper(
                df,
                symbol=sym2,
                asset=("etf" if strat == "bbb_etf" else "stock"),
                start_date=start_date,
                end_date=end_date,
                capital_yuan=float(probe_cap),
                roundtrip_cost_yuan=0.0,
                buy_cost=0.0,
                sell_cost=0.0,
                lot_size=lot,
                max_trades=0,
                bbb_entry_gap_max=float(gap_max),
                bbb_entry_ma=int(bbb_entry_ma or 20),
                bbb_dist_ma_max=float(bbb_dist_ma_max or 0.12),
                bbb_max_above_20w=float(bbb_max_above_20w or 0.05),
                bbb_min_weeks=int(bbb_min_weeks or 60),
                bbb_require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
                bbb_require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
                bbb_require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
                bbb_min_hold_days=int(bbb_min_hold_days or 5),
                bbb_cooldown_days=int(bbb_cooldown_days or 0),
            )
        else:
            out = {"ok": False, "error": f"strategy_not_supported: {strat}"}

        if not isinstance(out, dict) or not bool(out.get("ok", False)):
            sim_errors.append({"symbol": sym2, "error": str((out or {}).get("error") or "simulate_failed")})
            continue

        for t in (out.get("trades") or []):
            if not isinstance(t, dict):
                continue
            ed = str(t.get("entry_date") or "").strip()
            if not ed:
                continue
            cand = {
                "asset": str(t.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                "symbol": sym2,
                "entry_date": ed,
                "entry_price": float(t.get("entry_price") or 0.0),
                "entry_price_type": str(t.get("entry_price_type") or "open"),
                "exit_date": (str(t.get("exit_date") or "").strip() or None),
                "exit_price": (float(t.get("exit_price") or 0.0) if t.get("exit_price") is not None else None),
                "exit_price_type": str(t.get("exit_price_type") or ("open" if is_bbb else "close")),
                "reason": str(t.get("reason") or ""),
                "exit_exec": ("open" if is_bbb else "close"),
            }
            candidates.append(cand)

        op = out.get("open_position")
        if isinstance(op, dict) and str(op.get("entry_date") or "").strip():
            candidates.append(
                {
                    "asset": str(op.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                    "symbol": sym2,
                    "entry_date": str(op.get("entry_date")),
                    "entry_price": float(op.get("entry_price") or 0.0),
                    "entry_price_type": str(op.get("entry_price_type") or "open"),
                    "exit_date": None,
                    "exit_price": None,
                    "exit_price_type": "open",
                    "reason": "open",
                    "exit_exec": ("open" if is_bbb else "close"),
                }
            )

    if not candidates:
        return {"ok": True, "strategy": strat, "summary": {"trades": 0, "wins": 0, "win_rate": 0.0}, "positions": [], "trades": [], "errors": (map_errors + sim_errors)[:50]}

    # 分组：按 entry_date
    entries_by_date: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        d = str(c.get("entry_date") or "")
        if not d:
            continue
        entries_by_date.setdefault(d, []).append(c)

    # BBB 候选排序：越贴近 MA20(上一日) 越靠前；否则稳定按 symbol
    def bbb_score(c: dict[str, Any]) -> float:
        sym = str(c.get("symbol") or "")
        d = str(c.get("entry_date") or "")
        px = float(c.get("entry_price") or 0.0)
        m = maps_by_symbol.get(sym) or {}
        ma20 = (m.get("ma20_prev_by_date") or {}).get(d)
        if ma20 is None:
            return 1e9
        try:
            ma = float(ma20)
            if ma <= 0 or px <= 0:
                return 1e9
            return float(abs(px / ma - 1.0))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 1e9

    score_fn: Callable[[dict[str, Any]], float] = (bbb_score if is_bbb else (lambda _c: 0.0))

    # 组合撮合
    cash = max(0.0, float(capital_yuan or 0.0))
    init_cash = float(cash)
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    equity_dates: list[str] = []
    equity_vals: list[float] = []
    equity_cash: list[float] = []
    equity_pos: list[int] = []
    skipped: dict[str, int] = {
        "max_positions": 0,
        "max_exposure": 0,
        "no_cash": 0,
        "bad_price": 0,
        "turnover_buy": 0,
        "max_corr": 0,
        "theme_limit": 0,
    }

    buy_turnover_total_yuan = 0.0
    last_label = "unknown"

    for i, d in enumerate(calendar):
        d2 = str(d)
        if not d2:
            continue

        label = str(regime_by_date.get(d2) or "unknown")
        last_label = label
        rp = risk_profile_for_regime(label)
        mp = int(max_positions or 0)
        if mp <= 0:
            mp = int(rp.max_positions)
        mp = max(1, min(mp, 50))
        me = float(max_exposure_pct or 0.0)
        if me <= 0:
            me = float(rp.max_exposure_pct)
        # vol targeting：只降不升；避免未来函数用 i-1
        if vol_tgt > 0 and i > 0:
            v = _index_realized_vol_ann(int(i) - 1)
            if v is not None and float(v) > 0:
                me = float(me) * float(min(1.0, float(vol_tgt) / float(v)))
        me = max(0.0, min(me, 1.0))
        max_invest = float(init_cash) * float(me)
        buy_turnover_budget_today = None
        buy_turnover_used_today = 0.0
        if max_turn > 0:
            eq_prev = float(equity_vals[-1]) if equity_vals else float(init_cash)
            if eq_prev > 0:
                buy_turnover_budget_today = float(eq_prev) * float(max_turn)

        # --- open exits（先卖再买）---
        for sym, pos in list(positions.items()):
            if str(pos.get("exit_exec") or "") != "open":
                continue
            if str(pos.get("exit_date") or "") != d2:
                continue
            px = float(pos.get("exit_price") or 0.0)
            sh = int(pos.get("shares") or 0)
            if px <= 0 or sh <= 0:
                continue
            cash_out, fee_out = cash_sell(shares=int(sh), price=float(px), cost=cost)
            cash += float(cash_out)
            entry_cash = float(pos.get("entry_cash") or 0.0)
            pnl = float(cash_out - entry_cash)
            ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
            trades.append(
                {
                    "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                    "symbol": sym,
                    "entry_date": str(pos.get("entry_date")),
                    "exit_date": d2,
                    "entry_price": float(pos.get("entry_price") or 0.0),
                    "exit_price": float(px),
                    "shares": int(sh),
                    "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                    "sell_fee_yuan": float(fee_out),
                    "entry_cash": float(entry_cash),
                    "exit_cash": float(cash_out),
                    "pnl_net": float(pnl),
                    "pnl_net_pct": float(ret),
                    "hold_days": int(pos.get("hold_days") or 0),
                    "reason": str(pos.get("reason") or ""),
                    "mode": "portfolio",
                }
            )
            del positions[sym]

        # --- open entries ---
        todays = list(entries_by_date.get(d2) or [])
        if todays:
            todays.sort(key=lambda c: (score_fn(c), str(c.get("symbol") or "")))

        # 暴力但清晰：用“成本法”控制总曝险（别装逼搞一堆未来函数）
        invested = float(sum(float(p.get("entry_cash") or 0.0) for p in positions.values()))
        theme_counts: dict[str, int] = {}
        if max_theme > 0 and infer_theme is not None:
            for psym, pos in positions.items():
                th0 = pos.get("theme")
                if not th0:
                    _, th0 = _name_theme(psym)
                    if th0:
                        pos["theme"] = str(th0)
                th = str(th0 or "").strip()
                if th:
                    theme_counts[th] = int(theme_counts.get(th, 0)) + 1

        for c in todays:
            sym = str(c.get("symbol") or "")
            if not sym:
                continue
            if sym in positions:
                continue
            if len(positions) >= mp:
                skipped["max_positions"] += 1
                continue
            if invested >= max_invest:
                skipped["max_exposure"] += 1
                continue

            px = float(c.get("entry_price") or 0.0)
            if px <= 0 or not math.isfinite(px):
                skipped["bad_price"] += 1
                continue

            # 同主题限仓（可选）
            name, theme = _name_theme(sym)
            if max_theme > 0 and theme:
                if int(theme_counts.get(str(theme), 0)) >= int(max_theme):
                    skipped["theme_limit"] += 1
                    continue

            # 相关性过滤（可选；避免未来函数用 i-1）
            if max_corr2 > 0 and corr_abs_tail is not None and i > 0 and positions:
                ra = _daily_returns_tail(sym, end_idx=int(i) - 1, window_days=60)
                too_corr = False
                if ra:
                    for psym in positions:
                        rb = _daily_returns_tail(str(psym), end_idx=int(i) - 1, window_days=60)
                        if not rb:
                            continue
                        c0 = corr_abs_tail(ra, rb, min_overlap=20)
                        if c0 is None:
                            continue
                        if float(c0) >= float(max_corr2):
                            too_corr = True
                            break
                if too_corr:
                    skipped["max_corr"] += 1
                    continue

            # 单笔目标资金：总仓位/最大持仓数；再叠加“剩余额度”
            per_budget = max_invest / float(mp) if mp > 0 else max_invest
            remain_budget = max(0.0, max_invest - invested)
            budget = min(float(cash), float(per_budget), float(remain_budget))
            if budget <= max(0.0, float(cost.buy_fee_yuan) + 1e-6):
                skipped["no_cash"] += 1
                continue

            sh = calc_shares_for_capital(capital_yuan=float(budget), price=float(px), cost=cost, lot_size=lot)
            if sh <= 0:
                skipped["no_cash"] += 1
                continue

            # 换手约束（KISS：只限制 buy 侧）
            if buy_turnover_budget_today is not None:
                rem = float(buy_turnover_budget_today) - float(buy_turnover_used_today)
                max_by_turn = (int(rem / float(px)) // int(lot)) * int(lot) if rem > 0 else 0
                if max_by_turn <= 0:
                    skipped["turnover_buy"] += 1
                    break
                sh = int(min(int(sh), int(max_by_turn)))
                if sh <= 0:
                    skipped["turnover_buy"] += 1
                    break

            cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost)
            if cash_in > cash + 1e-6:
                skipped["no_cash"] += 1
                continue

            cash -= float(cash_in)
            invested += float(cash_in)
            if buy_turnover_budget_today is not None:
                buy_turnover_used_today += float(sh) * float(px)
                buy_turnover_total_yuan += float(sh) * float(px)
            if max_theme > 0 and theme:
                theme_counts[str(theme)] = int(theme_counts.get(str(theme), 0)) + 1

            # 计划退出（如果有）
            exit_date = c.get("exit_date")
            exit_price = c.get("exit_price")
            exit_exec = str(c.get("exit_exec") or "open")
            positions[sym] = {
                "asset": str(c.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                "symbol": sym,
                "name": name,
                "theme": theme,
                "entry_date": d2,
                "entry_price": float(px),
                "entry_price_type": str(c.get("entry_price_type") or "open"),
                "shares": int(sh),
                "entry_cash": float(cash_in),
                "buy_fee_yuan": float(fee_in),
                "reason": str(c.get("reason") or ""),
                "exit_date": (str(exit_date) if exit_date else None),
                "exit_price": (float(exit_price) if exit_price is not None else None),
                "exit_price_type": str(c.get("exit_price_type") or ("open" if is_bbb else "close")),
                "exit_exec": exit_exec,
                "last_close": float(px),
                "hold_days": 0,
                "regime_at_entry": label,
            }

        # --- close exits（短线：止盈/止损/收盘价）---
        for sym, pos in list(positions.items()):
            if str(pos.get("exit_exec") or "") != "close":
                continue
            if str(pos.get("exit_date") or "") != d2:
                continue
            px = float(pos.get("exit_price") or 0.0)
            sh = int(pos.get("shares") or 0)
            if px <= 0 or sh <= 0:
                continue
            cash_out, fee_out = cash_sell(shares=int(sh), price=float(px), cost=cost)
            cash += float(cash_out)
            entry_cash = float(pos.get("entry_cash") or 0.0)
            pnl = float(cash_out - entry_cash)
            ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
            trades.append(
                {
                    "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                    "symbol": sym,
                    "entry_date": str(pos.get("entry_date")),
                    "exit_date": d2,
                    "entry_price": float(pos.get("entry_price") or 0.0),
                    "exit_price": float(px),
                    "exit_price_type": str(pos.get("exit_price_type") or "close"),
                    "shares": int(sh),
                    "buy_fee_yuan": float(pos.get("buy_fee_yuan") or 0.0),
                    "sell_fee_yuan": float(fee_out),
                    "entry_cash": float(entry_cash),
                    "exit_cash": float(cash_out),
                    "pnl_net": float(pnl),
                    "pnl_net_pct": float(ret),
                    "hold_days": int(pos.get("hold_days") or 0),
                    "reason": str(pos.get("reason") or ""),
                    "mode": "portfolio",
                }
            )
            del positions[sym]

        # --- mark-to-market at close ---
        mv = 0.0
        for sym, pos in positions.items():
            m = maps_by_symbol.get(sym) or {}
            cb = (m.get("close_by_date") or {})
            last_close = float(pos.get("last_close") or pos.get("entry_price") or 0.0)
            px = cb.get(d2)
            if px is not None:
                try:
                    pxf = float(px)
                    if math.isfinite(pxf) and pxf > 0:
                        last_close = float(pxf)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    pass
            pos["last_close"] = float(last_close)
            pos["hold_days"] = int(pos.get("hold_days") or 0) + 1
            sh = int(pos.get("shares") or 0)
            if sh > 0 and last_close > 0:
                mv += float(sh) * float(last_close)

        eq = float(cash + mv)
        equity_dates.append(d2)
        equity_vals.append(eq)
        equity_cash.append(float(cash))
        equity_pos.append(int(len(positions)))

    equity_last = float(equity_vals[-1]) if equity_vals else float(cash)
    total_ret = float(equity_last / init_cash - 1.0) if init_cash > 0 else 0.0

    cagr = None
    try:
        if equity_dates:
            d0 = parse_date_any_opt(equity_dates[0])
            d1 = parse_date_any_opt(equity_dates[-1])
            if d0 is not None and d1 is not None:
                years = max(0.0, float((d1 - d0).days) / 365.25)
                if years > 0 and init_cash > 0 and equity_last > 0:
                    cagr = float((equity_last / init_cash) ** (1.0 / years) - 1.0)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        cagr = None

    dd = _max_drawdown(equity_vals)

    wins = int(sum(1 for t in trades if float(t.get("pnl_net") or 0.0) > 0))
    tr_n = int(len(trades))

    # open positions 输出
    open_positions: list[dict[str, Any]] = []
    for sym, pos in positions.items():
        sh = int(pos.get("shares") or 0)
        last_close = float(pos.get("last_close") or 0.0)
        mv = float(sh) * float(last_close) if sh > 0 and last_close > 0 else 0.0
        cash_out, fee_out = cash_sell(shares=int(sh), price=float(last_close), cost=cost) if (sh > 0 and last_close > 0) else (0.0, 0.0)
        entry_cash = float(pos.get("entry_cash") or 0.0)
        pnl = float(cash_out - entry_cash)
        ret = float(pnl / entry_cash) if entry_cash > 0 else 0.0
        open_positions.append(
            {
                "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                "symbol": sym,
                "name": (pos.get("name") if pos.get("name") else None),
                "theme": (pos.get("theme") if pos.get("theme") else None),
                "as_of": (equity_dates[-1] if equity_dates else None),
                "entry_date": str(pos.get("entry_date")),
                "entry_price": float(pos.get("entry_price") or 0.0),
                "entry_price_type": str(pos.get("entry_price_type") or "open"),
                "shares": int(sh),
                "entry_cash": float(entry_cash),
                "last_close": float(last_close),
                "market_value": float(mv),
                "exit_cash_if_sell_now": float(cash_out),
                "sell_fee_yuan_if_sell_now": float(fee_out),
                "pnl_net_if_sell_now": float(pnl),
                "pnl_net_pct_if_sell_now": float(ret),
                "regime_at_entry": str(pos.get("regime_at_entry") or "unknown"),
            }
        )

    open_positions.sort(key=lambda x: (str(x.get("asset") or ""), str(x.get("symbol") or "")))

    # --- next open orders（用于“收盘后刷新 -> 次日开盘手动执行”）---
    orders_next_open: list[dict[str, Any]] = []
    try:
        as_of = str(equity_dates[-1]) if equity_dates else None
        if as_of:
            label = str(regime_by_date.get(as_of) or last_label or "unknown")
            rp = risk_profile_for_regime(label)
            mp = int(max_positions or 0) if int(max_positions or 0) > 0 else int(rp.max_positions)
            mp = max(1, min(mp, 50))
            me = float(max_exposure_pct or 0.0) if float(max_exposure_pct or 0.0) > 0 else float(rp.max_exposure_pct)
            me = max(0.0, min(me, 1.0))

            max_invest = float(init_cash) * float(me)

            # 1) 先算“应该卖”（次日开盘）：只对当前持仓判断
            sell_syms: set[str] = set()
            sell_cash_est = 0.0
            for sym, pos in list(positions.items()):
                if str(pos.get("exit_exec") or "") != "open":
                    continue

                df = dfs_by_symbol.get(sym)
                if df is None or getattr(df, "empty", True):
                    continue

                # 只看最后一天（as_of）的信号，避免未来函数
                dfd = df.copy()
                dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
                dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                if dfd.empty:
                    continue
                last_dt = _date_str(dfd.iloc[-1].get("date"))
                if last_dt != as_of:
                    continue

                # BBB exit 规则：hard(周线) / soft(日线)
                # hard：周线 close<MA50 连续2周确认（且当前是周末K）
                dfw = resample_to_weekly(dfd)
                dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                hard_today = False
                if dfw is not None and (not getattr(dfw, "empty", True)) and len(dfw) >= 2:
                    if "ma50" not in dfw.columns:
                        dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)
                    close_w = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
                    ma50_w = pd.to_numeric(dfw["ma50"], errors="coerce").astype(float)
                    hard_w = ((close_w < ma50_w) & (close_w.shift(1) < ma50_w.shift(1))).fillna(False)
                    hard_today = bool(hard_w.iloc[-1]) and (_date_str(dfw.iloc[-1].get("date")) == as_of)

                # soft：日线 MACD 死叉 2日确认 + close<MA20（且满足最小持仓天数）
                hold_days = int(pos.get("hold_days") or 0)
                soft_today = False
                if hold_days >= int(bbb_min_hold_days or 0):
                    if "ma20" not in dfd.columns:
                        dfd = dfd.copy()
                        dfd["ma20"] = dfd["close"].astype(float).rolling(window=20, min_periods=20).mean()
                    if "macd" not in dfd.columns or "macd_signal" not in dfd.columns:
                        dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
                    close_d = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
                    ma20_d = pd.to_numeric(dfd["ma20"], errors="coerce").astype(float)
                    macd_d = pd.to_numeric(dfd["macd"], errors="coerce").astype(float)
                    sig_d = pd.to_numeric(dfd["macd_signal"], errors="coerce").astype(float)
                    bearish = (macd_d < sig_d).fillna(False)
                    bearish2 = (bearish & bearish.shift(1, fill_value=False)).fillna(False)
                    soft_today = bool((bearish2 & (close_d < ma20_d)).fillna(False).iloc[-1])

                reason = "hard" if hard_today else ("soft" if soft_today else None)
                if not reason:
                    continue

                sh = int(pos.get("shares") or 0)
                last_close = float(pos.get("last_close") or 0.0)
                if sh <= 0 or last_close <= 0:
                    continue

                cash_out, fee_out = cash_sell(shares=int(sh), price=float(last_close), cost=cost)
                sell_cash_est += float(cash_out)
                sell_syms.add(sym)
                orders_next_open.append(
                    {
                        "side": "sell",
                        "asset": str(pos.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                        "symbol": sym,
                        "shares": int(sh),
                        "signal_date": as_of,
                        "exec": "next_open",
                        "price_ref": float(last_close),
                        "price_ref_type": "close",
                        "order_type": "market",
                        "limit_price": None,
                        "est_cash": float(cash_out),
                        "est_fee_yuan": float(fee_out),
                        "reason": str(reason),
                    }
                )

            # 2) 再算“应该买”（次日开盘）：只对未持仓标的判断，并按仓位约束挑选
            invested_remaining = float(sum(float(p.get("entry_cash") or 0.0) for s, p in positions.items() if s not in sell_syms))
            cash_avail = float(cash + sell_cash_est)
            pos_cnt = int(sum(1 for s in positions.keys() if s not in sell_syms))

            buy_candidates: list[dict[str, Any]] = []
            if is_bbb:
                params = BBBParams(
                    entry_ma=max(2, int(bbb_entry_ma)),
                    dist_ma50_max=float(bbb_dist_ma_max),
                    max_above_20w=float(bbb_max_above_20w),
                    min_weekly_bars_total=int(bbb_min_weeks),
                    require_weekly_macd_bullish=bool(bbb_require_weekly_macd_bullish),
                    require_weekly_macd_above_zero=bool(bbb_require_weekly_macd_above_zero),
                    require_daily_macd_bullish=bool(bbb_require_daily_macd_bullish),
                )

                for sym, df in dfs_by_symbol.items():
                    if sym in positions and sym not in sell_syms:
                        continue
                    if sym in sell_syms:
                        continue

                    dfd = df.copy()
                    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
                    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if dfd.empty:
                        continue
                    if _date_str(dfd.iloc[-1].get("date")) != as_of:
                        continue
                    last_close = float(dfd.iloc[-1].get("close") or 0.0)
                    if last_close <= 0:
                        continue

                    dfw = resample_to_weekly(dfd)
                    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if dfw is None or getattr(dfw, "empty", True):
                        continue

                    sig_w = compute_bbb_entry_signal(dfw, dfd, params=params)
                    if sig_w is None or getattr(sig_w, "empty", True) or int(len(sig_w)) != int(len(dfw)):
                        continue
                    if _date_str(dfw.iloc[-1].get("date")) != as_of:
                        continue
                    if not bool(sig_w.iloc[-1]):
                        continue

                    # 分数：越靠近 entry_ma 越优先（避免“太右侧”）
                    entry_ma = max(2, int(bbb_entry_ma))
                    ma_entry = None
                    try:
                        close_w = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
                        ma_entry = float(close_w.rolling(window=entry_ma, min_periods=entry_ma).mean().iloc[-1])
                    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                        ma_entry = None
                    score = 1e9
                    if ma_entry is not None and ma_entry > 0:
                        score = float(abs(last_close / ma_entry - 1.0))

                    buy_candidates.append({"symbol": sym, "asset": ("etf" if strat == "bbb_etf" else "stock"), "price_ref": float(last_close), "score": float(score)})

                buy_candidates.sort(key=lambda x: (float(x.get("score") or 0.0), str(x.get("symbol") or "")))

            per_budget = float(max_invest / float(mp)) if mp > 0 else float(max_invest)
            for c in buy_candidates:
                if pos_cnt >= mp:
                    break
                if invested_remaining >= max_invest:
                    break

                px = float(c.get("price_ref") or 0.0)
                if px <= 0 or (not math.isfinite(px)):
                    continue

                remain_budget = max(0.0, float(max_invest - invested_remaining))
                budget = min(float(cash_avail), float(per_budget), float(remain_budget))
                if budget <= max(0.0, float(cost.buy_fee_yuan) + float(cost.buy_fee_min_yuan) + 1e-6):
                    continue

                sh = calc_shares_for_capital(capital_yuan=float(budget), price=float(px), cost=cost, lot_size=lot)
                if sh <= 0:
                    continue
                if float(min_trade_notional2) > 0:
                    notional = float(sh) * float(px)
                    if notional + 1e-6 < float(min_trade_notional2):
                        continue

                cash_in, fee_in = cash_buy(shares=int(sh), price=float(px), cost=cost)
                if cash_in > cash_avail + 1e-6:
                    continue

                cash_avail -= float(cash_in)
                invested_remaining += float(cash_in)
                pos_cnt += 1
                max_open = float(px) * (1.0 + float(gap_max)) if float(gap_max) > 0 else None
                orders_next_open.append(
                    {
                        "side": "buy",
                        "asset": str(c.get("asset") or ("etf" if strat == "bbb_etf" else "stock")),
                        "symbol": str(c.get("symbol") or ""),
                        "shares": int(sh),
                        "signal_date": as_of,
                        "exec": "next_open",
                        "price_ref": float(px),
                        "price_ref_type": "close",
                        "gap_max": float(gap_max) if is_bbb else None,
                        "max_open_price": float(max_open) if (is_bbb and max_open is not None) else None,
                        "order_type": ("limit" if is_bbb else "market"),
                        "limit_price": float(max_open) if (is_bbb and max_open is not None) else None,
                        "est_cash": float(cash_in),
                        "est_fee_yuan": float(fee_in),
                        "reason": "entry",
                    }
                )
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        orders_next_open = []

    # 交易统计：Profit Factor / Payoff（盈亏比），用 pnl_net 口径
    gp = 0.0  # gross profit
    gl = 0.0  # gross loss（负数累加）
    wn = 0
    ln = 0
    for t in trades:
        try:
            p = float(t.get("pnl_net") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        if (not math.isfinite(p)) or p == 0:
            continue
        if p > 0:
            gp += float(p)
            wn += 1
        else:
            gl += float(p)
            ln += 1
    avg_win_yuan = (float(gp) / float(wn)) if wn > 0 else None
    avg_loss_yuan = (float(abs(gl)) / float(ln)) if ln > 0 else None
    profit_factor = (float(gp) / float(abs(gl))) if gl < 0 else None
    payoff = (float(avg_win_yuan) / float(avg_loss_yuan)) if (avg_win_yuan is not None and avg_loss_yuan is not None and float(avg_loss_yuan) > 0) else None

    years2 = None
    try:
        if equity_dates:
            d0 = parse_date_any_opt(equity_dates[0])
            d1 = parse_date_any_opt(equity_dates[-1])
            if d0 is not None and d1 is not None:
                years2 = max(0.0, float((d1 - d0).days) / 365.25)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        years2 = None

    equity_liquidated = float(cash) + float(sum(float(p.get("exit_cash_if_sell_now") or 0.0) for p in open_positions))
    total_ret_liquidated = float(equity_liquidated / init_cash - 1.0) if init_cash > 0 else 0.0
    cagr_liquidated = None
    try:
        if years2 and float(years2) > 0 and init_cash > 0 and equity_liquidated > 0:
            cagr_liquidated = float((equity_liquidated / init_cash) ** (1.0 / float(years2)) - 1.0)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        cagr_liquidated = None

    gp2 = 0.0
    gl2 = 0.0
    wn2 = 0
    ln2 = 0
    for t in trades:
        try:
            p = float(t.get("pnl_net") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        if (not math.isfinite(p)) or p == 0:
            continue
        if p > 0:
            gp2 += float(p)
            wn2 += 1
        else:
            gl2 += float(p)
            ln2 += 1
    for p0 in open_positions:
        try:
            p = float(p0.get("pnl_net_if_sell_now") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        if (not math.isfinite(p)) or p == 0:
            continue
        if p > 0:
            gp2 += float(p)
            wn2 += 1
        else:
            gl2 += float(p)
            ln2 += 1
    avg_win_yuan2 = (float(gp2) / float(wn2)) if wn2 > 0 else None
    avg_loss_yuan2 = (float(abs(gl2)) / float(ln2)) if ln2 > 0 else None
    profit_factor2 = (float(gp2) / float(abs(gl2))) if gl2 < 0 else None
    payoff2 = (float(avg_win_yuan2) / float(avg_loss_yuan2)) if (avg_win_yuan2 is not None and avg_loss_yuan2 is not None and float(avg_loss_yuan2) > 0) else None

    tail_n = 260
    eq_tail = [
        {
            "date": equity_dates[i],
            "equity": float(equity_vals[i]),
            "cash": float(equity_cash[i]),
            "label": str(regime_by_date.get(equity_dates[i]) or "unknown"),
            "positions": int(equity_pos[i]) if i < len(equity_pos) else None,
        }
        for i in range(max(0, len(equity_dates) - tail_n), len(equity_dates))
    ]

    return {
        "ok": True,
        "mode": "portfolio",
        "strategy": strat,
        "as_of": (equity_dates[-1] if equity_dates else None),
        "summary": {
            "capital_yuan": float(init_cash),
            "equity_last": float(equity_last),
            "cash_last": float(cash),
            "equity_liquidated": float(equity_liquidated),
            "total_return": float(total_ret),
            "total_return_liquidated": float(total_ret_liquidated),
            "cagr": cagr,
            "cagr_liquidated": cagr_liquidated,
            "max_drawdown": dd,
            "trades": tr_n,
            "wins": int(wins),
            "win_rate": float(wins / tr_n) if tr_n > 0 else 0.0,
            "pnl_gross_profit_yuan": float(gp),
            "pnl_gross_loss_yuan": float(abs(gl)),
            "profit_factor": float(profit_factor) if profit_factor is not None else None,
            "avg_win_yuan": float(avg_win_yuan) if avg_win_yuan is not None else None,
            "avg_loss_yuan": float(avg_loss_yuan) if avg_loss_yuan is not None else None,
            "payoff": float(payoff) if payoff is not None else None,
            "pnl_gross_profit_yuan_incl_open": float(gp2),
            "pnl_gross_loss_yuan_incl_open": float(abs(gl2)),
            "profit_factor_incl_open": float(profit_factor2) if profit_factor2 is not None else None,
            "avg_win_yuan_incl_open": float(avg_win_yuan2) if avg_win_yuan2 is not None else None,
            "avg_loss_yuan_incl_open": float(avg_loss_yuan2) if avg_loss_yuan2 is not None else None,
            "payoff_incl_open": float(payoff2) if payoff2 is not None else None,
            "open_positions": int(len(open_positions)),
            "last_regime": str(last_label),
            "skipped": skipped,
            "symbols": int(len(maps_by_symbol)),
            "orders_next_open": int(len(orders_next_open)),
        },
        "orders_next_open": orders_next_open,
        "equity_curve_tail": eq_tail,
        "positions": open_positions,
        "trades": trades[-500:],
        "errors": (map_errors + sim_errors)[:50],
    }
