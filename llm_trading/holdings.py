from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


def analyze_holdings(
    *,
    holdings: list[dict[str, Any]],
    regime_index: str = "sh000300",
    regime_canary_downgrade: bool = True,
    sell_cost_yuan: float = 5.0,
    small_position_threshold_yuan: float = 2000.0,
    tol_pct_small: float = 0.10,
    tol_pct_default: float = 0.05,
    cache_ttl_hours: float = 6.0,
    stock_adjust: str = "qfq",
    profit_stop_enabled: bool = True,
    profit_stop_lookback_days: int | None = None,
    profit_stop_min_profit_ret: float | None = None,
    profit_stop_dd_pct_hot_bull: float | None = None,
    profit_stop_dd_pct_slow_bull: float | None = None,
) -> dict[str, Any]:
    """
    持仓分析（ETF/股票；研究用途）：
    - 止损触发：只看收盘价（close_only）
    - 止损强弱：跟随牛熊（bull/neutral 用周线参考；bear 用日线 MA20 更紧）
      - regime_index=auto：按标的自动选“基准指数/自身”（更贴近真实波动）
      - regime_index=sh000300：固定用该指数判定牛熊（旧行为）
      - regime_index=off：关闭牛熊跟随（全部按 unknown 处理）
    - 最大亏损兜底：仓位<=2000 允许 10%，否则 5%（并把卖出磨损摊到每股）
    - 止盈（离场参考）：盈利状态下，如果出现“日线 MACD 2日死叉确认 + 跌破日线 MA20”，提示止盈

    返回 JSON 可序列化 dict（已做 NaN/Inf 清洗）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .akshare_source import FetchParams, resolve_symbol
    from .data_cache import fetch_daily_cached
    from .indicators import add_adx, add_atr, add_bollinger_bands, add_donchian_channels, add_macd, add_rsi
    from .json_utils import sanitize_for_json
    from .market_regime import compute_market_regime, compute_market_regime_payload, market_regime_to_dict
    from .positioning import risk_profile_for_regime
    from .resample import resample_to_weekly
    from .symbol_names import get_symbol_name
    from .take_profit import TakeProfitConfig, calc_tp1_sell_shares, classify_bull_phase

    idx_raw = str(regime_index or "sh000300").strip() or "sh000300"
    idx_norm = idx_raw.strip().lower()
    sell_cost_yuan2 = float(sell_cost_yuan or 0.0)
    sell_cost_yuan2 = max(0.0, min(sell_cost_yuan2, 1000.0))

    small_th = float(small_position_threshold_yuan or 0.0)
    small_th = max(0.0, small_th)
    tol_small = max(0.0, min(float(tol_pct_small or 0.0), 0.50))
    tol_def = max(0.0, min(float(tol_pct_default or 0.0), 0.50))

    tp_cfg = TakeProfitConfig()
    # 允许从上层调用覆盖回撤止盈参数（不想搞配置文件/数据库；KISS）。
    try:
        tp_cfg = replace(tp_cfg, profit_stop_enabled=bool(profit_stop_enabled))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass
    try:
        if profit_stop_lookback_days is not None:
            lb = int(profit_stop_lookback_days)
            lb = max(20, min(lb, 2520))
            tp_cfg = replace(tp_cfg, profit_stop_lookback_days=int(lb))
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        pass
    try:
        if profit_stop_min_profit_ret is not None:
            mr = float(profit_stop_min_profit_ret)
            mr = max(0.0, min(mr, 5.0))
            tp_cfg = replace(tp_cfg, profit_stop_min_profit_ret=float(mr))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass
    try:
        if profit_stop_dd_pct_hot_bull is not None:
            ddh = float(profit_stop_dd_pct_hot_bull)
            ddh = max(0.0, min(ddh, 0.50))
            tp_cfg = replace(tp_cfg, profit_stop_dd_pct_hot_bull=float(ddh))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass
    try:
        if profit_stop_dd_pct_slow_bull is not None:
            dds = float(profit_stop_dd_pct_slow_bull)
            dds = max(0.0, min(dds, 0.50))
            tp_cfg = replace(tp_cfg, profit_stop_dd_pct_slow_bull=float(dds))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass

    regime_mode: str
    meta_index: str | None
    if idx_norm in {"", "off", "none", "0"}:
        regime_mode = "off"
        meta_index = None
    elif idx_norm in {"auto", "smart"}:
        regime_mode = "auto"
        meta_index = "sh000300"
    else:
        regime_mode = "fixed"
        meta_index = idx_raw

    # 页面级的大盘牛熊（用于 meta 展示；持仓级别的止盈止损逻辑会根据 regime_mode 决定走 fixed/auto/off）
    regime_dict = None
    regime_error = None
    regime_index_eff = None
    if meta_index is not None:
        try:
            regime_dict, regime_error, regime_index_eff = compute_market_regime_payload(
                str(meta_index),
                cache_dir=Path("data") / "cache" / "index",
                ttl_hours=float(cache_ttl_hours),
                ensemble_mode="risk_first",  # 用户确认：多指数合并风险优先
                canary_downgrade=bool(regime_canary_downgrade),
            )
        except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
            regime_dict = None
            regime_error = str(exc)
            regime_index_eff = str(meta_index)

    regime_label = str((regime_dict or {}).get("label") or "unknown")
    bull_phase = classify_bull_phase(label=regime_label, mom_63d=(regime_dict or {}).get("mom_63d"), cfg=tp_cfg)
    if isinstance(regime_dict, dict):
        regime_dict = dict(regime_dict)
        regime_dict["bull_phase"] = bull_phase

    # fixed/off 会复用；auto 按标的计算
    profile_fixed = risk_profile_for_regime(regime_label)

    proxy_index_cache: dict[str, dict[str, Any]] = {}
    proxy_index_error_cache: dict[str, str] = {}

    # BBB 7因子面板里的 RS 基准（解释用）：默认 300+500 等权（更中性）。
    # 注意：这不是“买卖按钮”，只是给你一个更像样的解释面板。
    rs_index_symbol = None
    rs_index_weekly = None
    try:
        from .index_composite import fetch_index_daily_spec

        df_rs, rs_eff = fetch_index_daily_spec(
            "sh000300+sh000905",
            cache_dir=Path("data") / "cache" / "index",
            ttl_hours=float(cache_ttl_hours),
        )
        rs_index_symbol = str(rs_eff or "sh000300+sh000905")
        rs_index_weekly = resample_to_weekly(df_rs) if df_rs is not None and (not getattr(df_rs, "empty", True)) else None
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        rs_index_symbol = None
        rs_index_weekly = None

    def _compute_regime_payload(symbol: str, df_daily) -> dict[str, Any]:
        rr = compute_market_regime(index_symbol=str(symbol), df_daily=df_daily, ma_fast=50, ma_slow=200, ma_fast_slope_weeks=4)
        d = market_regime_to_dict(rr)
        bp = classify_bull_phase(label=str(d.get("label") or "unknown"), mom_63d=d.get("mom_63d"), cfg=tp_cfg)
        d = dict(d)
        d["bull_phase"] = bp
        return d

    def _get_index_regime(index_symbol: str) -> tuple[dict[str, Any] | None, str | None]:
        key = str(index_symbol or "").strip().lower()
        if not key:
            return None, "index_symbol 为空"
        if key in proxy_index_cache:
            return dict(proxy_index_cache[key]), None
        if key in proxy_index_error_cache:
            return None, proxy_index_error_cache[key]
        try:
            df_idx = fetch_daily_cached(
                FetchParams(asset="index", symbol=key, source="auto"),
                cache_dir=Path("data") / "cache" / "index",
                ttl_hours=float(cache_ttl_hours),
            )
            d = _compute_regime_payload(key, df_idx)
            proxy_index_cache[key] = dict(d)
            return dict(d), None
        except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
            err = str(exc)
            proxy_index_error_cache[key] = err
            return None, err

    def _infer_regime_proxy(*, asset: str, symbol: str, name: str | None) -> dict[str, str]:
        """
        给持仓推一个更合适的“牛熊基准”：
        - 股票：按板块指数（上证/深证/创业板/科创50/北证50）
        - ETF：优先按名称识别宽基（沪深300/中证500/中证1000/上证50/创业板/科创50），否则用自身
        """
        a = str(asset or "").strip().lower()
        sym = str(symbol or "").strip().lower()
        nm = str(name or "").strip().replace(" ", "")

        if a == "stock":
            code = sym[2:] if sym.startswith(("sh", "sz", "bj")) else sym
            if sym.startswith("sz") and code.startswith("3"):
                return {"mode": "index", "symbol": "sz399006", "name": "创业板指"}
            if sym.startswith("sh") and code.startswith("688"):
                return {"mode": "index", "symbol": "sh000688", "name": "科创50"}
            if sym.startswith("sh"):
                return {"mode": "index", "symbol": "sh000001", "name": "上证指数"}
            if sym.startswith("sz"):
                return {"mode": "index", "symbol": "sz399001", "name": "深证成指"}
            if sym.startswith("bj"):
                return {"mode": "index", "symbol": "bj899050", "name": "北证50"}
            # 兜底：回到全市场口径
            return {"mode": "index", "symbol": "sh000300", "name": "沪深300"}

        # ETF：宽基/主流指数能识别就用“指数”；否则按自身走势判定牛熊
        rules: list[tuple[str, str, str]] = [
            ("沪深300", "sh000300", "沪深300"),
            ("上证50", "sh000016", "上证50"),
            ("中证500", "sh000905", "中证500"),
            ("中证1000", "sh000852", "中证1000"),
            ("科创50", "sh000688", "科创50"),
            ("创业板", "sz399006", "创业板指"),
            ("深证成指", "sz399001", "深证成指"),
            ("上证指数", "sh000001", "上证指数"),
        ]
        for kw, idx_sym, idx_name in rules:
            if kw and kw in nm:
                return {"mode": "index", "symbol": idx_sym, "name": idx_name}

        return {"mode": "self", "symbol": sym, "name": (str(name).strip() if name else sym)}

    def fnum(x):
        try:
            v = None if x is None else float(x)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return None
        try:
            import math

            return None if (v is None or not math.isfinite(v)) else float(v)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return float(v) if v is not None else None

    def _ret(series, n: int) -> float | None:
        try:
            n2 = int(n)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return None
        if n2 <= 0:
            return None
        try:
            if series is None or len(series) <= n2:
                return None
            a = float(series.iloc[-1])
            b = float(series.iloc[-(n2 + 1)])
            if b == 0:
                return None
            return a / b - 1.0
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            return None

    def _slope(series, n: int) -> float | None:
        """
        简易“趋势线斜率”：
        - 取最近 n 个点做线性回归，返回“每个bar的相对斜率”（约等于每bar涨跌幅）
        - 这不是画K线的主观趋势线，但够用来判断“是上行/走平/下行”
        """
        try:
            n2 = int(n)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return None
        if n2 < 5:
            return None
        try:
            import numpy as np

            s = pd.to_numeric(series, errors="coerce").astype(float).dropna()
            if len(s) < n2:
                return None
            y = s.iloc[-n2:].to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            # y = a*x + b
            a, b = np.polyfit(x, y, 1)
            y0 = float(b)
            # 用截距做尺度，避免价格量级影响；y0<=0 则退化用最后价
            scale = y0 if y0 > 0 else float(y[-1])
            if scale <= 0:
                return None
            return float(a) / float(scale)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return None

    if not isinstance(holdings, list) or not holdings:
        raise ValueError("holdings 不能为空")

    items: list[dict[str, Any]] = []
    for h in holdings:
        asset = str((h or {}).get("asset") or "etf").strip().lower()
        if asset not in {"etf", "stock"}:
            items.append({"asset": asset, "symbol": str((h or {}).get("symbol") or ""), "ok": False, "error": "asset 只支持 etf/stock"})
            continue

        entry_style = str((h or {}).get("entry_style") or "").strip().lower()
        # left=左侧试仓：不把 MA 锚线当“硬止损按钮”，只当观察/确认线（避免被震荡抖飞）。
        left_mode = entry_style in {"left", "left_side", "leftside", "try", "probe"}

        sym_raw = str((h or {}).get("symbol") or "").strip()
        if not sym_raw:
            items.append({"asset": asset, "symbol": "", "ok": False, "error": "symbol 为空"})
            continue

        try:
            shares = int((h or {}).get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            shares = 0
        try:
            cost = float((h or {}).get("cost") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cost = 0.0

        if shares <= 0 or cost <= 0:
            items.append({"asset": asset, "symbol": sym_raw, "ok": False, "error": "shares/cost 非法"})
            continue

        sym = resolve_symbol(asset, sym_raw)
        name = get_symbol_name(asset, sym)
        adjust = "qfq" if asset == "etf" else str(stock_adjust or "qfq").strip() or "qfq"
        if adjust not in {"", "qfq", "hfq"}:
            adjust = "qfq"

        cache_dir = Path("data") / "cache" / asset
        # 持仓风控/复盘口径：优先 TuShare（更贴近常见券商/行情软件收盘价），失败再回退 AkShare。
        df = fetch_daily_cached(
            FetchParams(asset=asset, symbol=sym, adjust=adjust, source="auto"),
            cache_dir=cache_dir,
            ttl_hours=float(cache_ttl_hours),
        )
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            items.append({"asset": asset, "symbol": sym, "symbol_input": sym_raw, "ok": False, "error": "无K线"})
            continue

        # 指标/风控计算依赖 high/low/volume/amount，缺了就补（别让源站脏数据把流程炸了）。
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        if "amount" not in df.columns:
            try:
                df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
            except (AttributeError):  # noqa: BLE001
                df["amount"] = None

        last = df.iloc[-1]
        asof = last.get("date")
        asof_str = asof.strftime("%Y-%m-%d") if hasattr(asof, "strftime") else str(asof)
        close = fnum(last.get("close"))

        close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
        ma20_s = close_s.rolling(20, min_periods=20).mean()
        daily_ma20 = fnum(ma20_s.iloc[-1]) if len(ma20_s) else None
        ma60_s = close_s.rolling(60, min_periods=60).mean()
        daily_ma60 = fnum(ma60_s.iloc[-1]) if len(ma60_s) else None
        ma120_s = close_s.rolling(120, min_periods=120).mean()
        daily_ma120 = fnum(ma120_s.iloc[-1]) if len(ma120_s) else None

        # 日线动量/趋势：RSI/ADX/MACD
        df_mom_d = add_rsi(df, period=14, out_col="rsi14")
        df_mom_d = add_adx(df_mom_d, period=14, adx_col="adx14", di_plus_col="di_plus14", di_minus_col="di_minus14")

        # 日线 MACD（用于软离场/止盈提示）
        df_macd = add_macd(df, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
        macd_s = pd.to_numeric(df_macd.get("macd"), errors="coerce").astype(float)
        macd_sig_s = pd.to_numeric(df_macd.get("macd_signal"), errors="coerce").astype(float)
        bearish = (macd_s < macd_sig_s).fillna(False)
        bearish2 = (bearish & bearish.shift(1, fill_value=False)).fillna(False)
        soft_exit = bool(bearish2.iloc[-1]) if len(bearish2) else False

        dfw = resample_to_weekly(df)
        dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        w_close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
        w_ma20 = w_close.rolling(20, min_periods=20).mean()
        w_ma50 = w_close.rolling(50, min_periods=50).mean()
        w_ma200 = w_close.rolling(200, min_periods=200).mean()
        weekly_ma20 = fnum(w_ma20.iloc[-1]) if len(w_ma20) else None
        weekly_ma50 = fnum(w_ma50.iloc[-1]) if len(w_ma50) else None
        weekly_ma200 = fnum(w_ma200.iloc[-1]) if len(w_ma200) else None

        # 周线支撑/阻力：20周唐奇安上下轨（shift=1，避免把“本周没收盘”的高低点当成既成事实）
        dfw_sr = add_donchian_channels(
            dfw,
            window=20,
            upper_col="donchian_upper_20w",
            lower_col="donchian_lower_20w",
            shift=1,
        )
        sr_last = dfw_sr.iloc[-1] if not dfw_sr.empty else None
        support_20w = fnum(sr_last.get("donchian_lower_20w")) if sr_last is not None else None
        resistance_20w = fnum(sr_last.get("donchian_upper_20w")) if sr_last is not None else None

        # 周线动量：RSI/MACD/ADX
        dfw_mom = add_rsi(dfw, period=14, out_col="rsi14")
        dfw_mom = add_macd(dfw_mom, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
        dfw_mom = add_adx(dfw_mom, period=14, adx_col="adx14", di_plus_col="di_plus14", di_minus_col="di_minus14")
        w_last2 = dfw_mom.iloc[-1] if not dfw_mom.empty else None
        weekly_rsi14 = fnum(w_last2.get("rsi14")) if w_last2 is not None else None
        weekly_adx14 = fnum(w_last2.get("adx14")) if w_last2 is not None else None
        weekly_macd = fnum(w_last2.get("macd")) if w_last2 is not None else None
        weekly_macd_sig = fnum(w_last2.get("macd_signal")) if w_last2 is not None else None
        if weekly_macd is not None and weekly_macd_sig is not None:
            weekly_macd_state = "bullish" if weekly_macd > weekly_macd_sig else ("bearish" if weekly_macd < weekly_macd_sig else "neutral")
        else:
            weekly_macd_state = "unknown"

        # 日线动量快照（最后一根）
        d_last2 = df_mom_d.iloc[-1] if not df_mom_d.empty else None
        daily_rsi14 = fnum(d_last2.get("rsi14")) if d_last2 is not None else None
        daily_adx14 = fnum(d_last2.get("adx14")) if d_last2 is not None else None
        d_last_macd = df_macd.iloc[-1] if not df_macd.empty else None
        daily_macd2 = fnum(d_last_macd.get("macd")) if d_last_macd is not None else None
        daily_macd_sig2 = fnum(d_last_macd.get("macd_signal")) if d_last_macd is not None else None
        if daily_macd2 is not None and daily_macd_sig2 is not None:
            daily_macd_state2 = "bullish" if daily_macd2 > daily_macd_sig2 else ("bearish" if daily_macd2 < daily_macd_sig2 else "neutral")
        else:
            daily_macd_state2 = "unknown"

        # 趋势线（数值化）：近 N 日/周斜率 + 动量收益
        daily_ret_5d = _ret(close_s, 5)
        daily_ret_20d = _ret(close_s, 20)
        daily_ret_60d = _ret(close_s, 60)
        weekly_ret_4w = _ret(w_close, 4)
        weekly_ret_12w = _ret(w_close, 12)
        weekly_ret_26w = _ret(w_close, 26)

        daily_slope_20d = _slope(close_s, 20)
        daily_slope_60d = _slope(close_s, 60)
        weekly_slope_12w = _slope(w_close, 12)

        # === BBB 7因子解释面板（持仓版本；只做解释/排序，不当买卖按钮）===
        factor_panel_7: dict[str, Any] = {"ok": False, "as_of": asof_str}
        try:
            # 日线：波动/ATR%/回撤/BOLL/量能比
            r1 = (close_s / close_s.shift(1).replace({0.0: float("nan")})) - 1.0
            vol_20d_s = r1.rolling(window=20, min_periods=20).std()
            vol_20d = fnum(vol_20d_s.iloc[-1]) if len(vol_20d_s) else None

            atr14_pct = None
            try:
                df_atr = add_atr(df, period=14, out_col="atr14")
                atr14 = fnum(df_atr.iloc[-1].get("atr14")) if not df_atr.empty else None
                if atr14 is not None and close is not None and float(close) > 0:
                    atr14_pct = float(atr14) / float(close)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                atr14_pct = None

            dd_252d = None
            from_low_252d = None
            try:
                roll_max = close_s.rolling(window=252, min_periods=20).max()
                roll_min = close_s.rolling(window=252, min_periods=20).min()
                dd = (close_s / roll_max.replace({0.0: float("nan")})) - 1.0
                up_from_low = (close_s / roll_min.replace({0.0: float("nan")})) - 1.0
                dd_252d = fnum(dd.iloc[-1]) if len(dd) else None
                from_low_252d = fnum(up_from_low.iloc[-1]) if len(up_from_low) else None
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                dd_252d = None
                from_low_252d = None

            boll_bw = None
            boll_bw_rel = None
            boll_squeeze = None
            try:
                df_bb = add_bollinger_bands(df, window=20, k=2.0, bandwidth_col="boll_bw")
                bw_s = pd.to_numeric(df_bb.get("boll_bw"), errors="coerce").astype(float)
                bw_last = fnum(bw_s.iloc[-1]) if len(bw_s) else None
                boll_bw = bw_last
                bw_med = bw_s.rolling(window=252, min_periods=60).median()
                bw_med_last = fnum(bw_med.iloc[-1]) if len(bw_med) else None
                if bw_last is not None and bw_med_last is not None and float(bw_med_last) > 0:
                    boll_bw_rel = float(bw_last) / float(bw_med_last)
                    boll_squeeze = bool(float(boll_bw_rel) <= 0.80)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                boll_bw = None
                boll_bw_rel = None
                boll_squeeze = None

            amount_s = pd.to_numeric(df.get("amount"), errors="coerce").astype(float)
            volume_s = pd.to_numeric(df.get("volume"), errors="coerce").astype(float)
            amount_last = fnum(amount_s.iloc[-1]) if len(amount_s) else None
            volume_last = fnum(volume_s.iloc[-1]) if len(volume_s) else None
            amount_avg20 = fnum(float(amount_s.tail(20).mean())) if len(amount_s) else None
            volume_avg20 = fnum(float(volume_s.tail(20).mean())) if len(volume_s) else None
            amount_ratio = None
            volume_ratio = None
            try:
                if amount_last is not None and amount_avg20 is not None and float(amount_avg20) > 0:
                    amount_ratio = float(amount_last) / float(amount_avg20)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                amount_ratio = None
            try:
                if volume_last is not None and volume_avg20 is not None and float(volume_avg20) > 0:
                    volume_ratio = float(volume_last) / float(volume_avg20)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                volume_ratio = None

            # 周线：动量 + RS（相对强弱）
            mom_12w_s = (w_close / w_close.shift(12).replace({0.0: float("nan")})) - 1.0
            mom_26w_s = (w_close / w_close.shift(26).replace({0.0: float("nan")})) - 1.0
            mom_12w = fnum(mom_12w_s.iloc[-1]) if len(mom_12w_s) else None
            mom_26w = fnum(mom_26w_s.iloc[-1]) if len(mom_26w_s) else None

            rs_12w = None
            rs_26w = None
            try:
                if rs_index_weekly is not None and (not getattr(rs_index_weekly, "empty", True)):
                    wi = rs_index_weekly[["date", "close"]].copy()
                    wi["date"] = pd.to_datetime(wi["date"], errors="coerce")
                    wi = wi.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if not wi.empty:
                        aligned = pd.merge_asof(
                            dfw[["date"]],
                            wi[["date", "close"]].rename(columns={"close": "idx_close"}),
                            on="date",
                            direction="backward",
                        )
                        idx_close = pd.to_numeric(aligned["idx_close"], errors="coerce").astype(float)
                        idx_mom_12w = (idx_close / idx_close.shift(12).replace({0.0: float("nan")})) - 1.0
                        idx_mom_26w = (idx_close / idx_close.shift(26).replace({0.0: float("nan")})) - 1.0
                        rs_12w_s = mom_12w_s - idx_mom_12w
                        rs_26w_s = mom_26w_s - idx_mom_26w
                        rs_12w = fnum(rs_12w_s.iloc[-1]) if len(rs_12w_s) else None
                        rs_26w = fnum(rs_26w_s.iloc[-1]) if len(rs_26w_s) else None
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                rs_12w = None
                rs_26w = None

            factor_panel_7 = {
                "ok": True,
                "as_of": asof_str,
                "rs": {"index": (str(rs_index_symbol) if rs_index_symbol else None), "rs_12w": rs_12w, "rs_26w": rs_26w},
                "mom": {"mom_12w": mom_12w, "mom_26w": mom_26w},
                "trend": {"adx14": daily_adx14},
                "vol": {"vol_20d": vol_20d, "atr14_pct": atr14_pct},
                "drawdown": {"dd_252d": dd_252d, "from_low_252d": from_low_252d},
                "liquidity": {"amount_avg20": amount_avg20, "amount_ratio": amount_ratio, "volume_ratio": volume_ratio},
                "boll": {"bandwidth": boll_bw, "bandwidth_rel": boll_bw_rel, "squeeze": (bool(boll_squeeze) if boll_squeeze is not None else None)},
                "note": "holdings-user面板口径=最后一根日线收盘可得；仅用于解释/排序，别拿它当买卖按钮。",
            }
        except Exception:  # noqa: BLE001
            factor_panel_7 = {"ok": False, "as_of": asof_str}

        # === 牛熊口径（决定止损模式/止盈风格）===
        h_regime_label = regime_label
        h_bull_phase = bull_phase
        h_profile = profile_fixed
        h_regime_proxy = None
        h_regime_error = None

        if regime_mode == "off":
            h_regime_label = "unknown"
            h_bull_phase = None
            h_profile = risk_profile_for_regime(h_regime_label)
            h_regime_proxy = {"mode": "off", "symbol": "off", "name": "off"}
        elif regime_mode == "fixed":
            idx2 = str(regime_index_eff or meta_index or idx_raw).strip().lower()
            if idx2:
                h_regime_proxy = {"mode": "index", "symbol": idx2, "name": idx2}
        elif regime_mode == "auto":
            proxy = _infer_regime_proxy(asset=asset, symbol=sym, name=name)
            h_regime_proxy = dict(proxy)
            pr = None
            err = None
            if proxy.get("mode") == "index":
                pr, err = _get_index_regime(str(proxy.get("symbol") or ""))
            else:
                try:
                    pr = _compute_regime_payload(sym, df)
                except (AttributeError) as exc:  # noqa: BLE001
                    pr = None
                    err = str(exc)

            if pr is None:
                # 自动标的失败：退回页面级口径，至少别把止盈止损算挂了
                pr = regime_dict
                if err:
                    h_regime_error = err
            else:
                h_regime_error = err

            h_regime_label = str((pr or {}).get("label") or "unknown")
            h_bull_phase = (pr or {}).get("bull_phase") or classify_bull_phase(label=h_regime_label, mom_63d=(pr or {}).get("mom_63d"), cfg=tp_cfg)
            h_profile = risk_profile_for_regime(h_regime_label)

        # 最大亏损阈值（含卖出磨损）：仓位<=2000 允许 10%，否则 5%
        cost_total = float(shares) * float(cost)
        tol_pct = tol_small if cost_total <= float(small_th) else tol_def
        loss_stop = float(cost) * (1.0 - float(tol_pct)) + (sell_cost_yuan2 / float(shares))

        # 硬止损：按牛熊选择（收盘价触发）
        hard_stop = None
        hard_ref = None
        if h_profile.stop_mode == "daily_ma20":
            hard_stop = daily_ma20
            hard_ref = "日线MA20"
            if hard_stop is None:
                hard_stop = weekly_ma20 or weekly_ma50
                hard_ref = "周线MA20/MA50(兜底)"
        else:
            hard_stop = weekly_ma20 or weekly_ma50
            hard_ref = "周线MA20" if weekly_ma20 is not None else ("周线MA50" if weekly_ma50 is not None else None)
            if hard_stop is None:
                hard_stop = daily_ma20
                hard_ref = "日线MA20(兜底)"

        effective_stop = None
        pnl_gross = None
        pnl_net = None
        pnl_pct = None
        if close is not None and close > 0:
            mv = float(shares) * float(close)
            pnl_gross = mv - cost_total
            pnl_net = (mv - sell_cost_yuan2) - cost_total
            pnl_pct = pnl_net / cost_total if cost_total > 0 else None

        # 回撤止盈（盈利保护）：浮盈足够后，启用“近 N 日最高收盘 * (1-dd)”的保护线，避免一刀砍掉大半利润。
        profit_stop = None
        profit_ref = None
        profit_lookback_days = None
        profit_dd_pct = None
        if bool(tp_cfg.profit_stop_enabled) and pnl_pct is not None and float(pnl_pct) >= float(tp_cfg.profit_stop_min_profit_ret):
            dd = float(tp_cfg.profit_stop_dd_pct_hot_bull if h_bull_phase == "hot" else tp_cfg.profit_stop_dd_pct_slow_bull)
            dd = max(0.0, min(dd, 0.50))

            lb0 = int(tp_cfg.profit_stop_lookback_days)
            lb0 = max(5, lb0)
            lb0 = min(lb0, int(len(close_s) or 0))
            if lb0 > 0:
                mp0 = min(20, int(lb0))
                roll_max = close_s.rolling(window=int(lb0), min_periods=int(mp0)).max()
                max_close = fnum(roll_max.iloc[-1]) if len(roll_max) else None
            else:
                max_close = None

            if max_close is not None and float(max_close) > 0:
                profit_stop = float(max_close) * (1.0 - float(dd))
                profit_dd_pct = float(dd)
                profit_lookback_days = int(lb0)
                profit_ref = f"近{int(lb0)}日高点回撤{int(round(float(dd) * 100.0))}%"

        # 左侧试仓：MA 锚线只用于“观察/确认”，不参与硬止损线合成；硬止损只看试错线/盈利保护线。
        stops_for_effective = [loss_stop, profit_stop] if left_mode else [hard_stop, loss_stop, profit_stop]
        for v in stops_for_effective:
            if v is None:
                continue
            effective_stop = v if effective_stop is None else max(float(effective_stop), float(v))

        # 让 UI 一眼看懂：当前“有效止损线”到底是哪条
        effective_ref = None
        if effective_stop is not None:
            eps = 1e-9
            try:
                if hard_stop is not None and abs(float(effective_stop) - float(hard_stop)) <= eps:
                    effective_ref = hard_ref
                elif profit_stop is not None and abs(float(effective_stop) - float(profit_stop)) <= eps:
                    effective_ref = profit_ref
                else:
                    effective_ref = "最大亏损兜底"
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                effective_ref = None

        # 信号：小白只看动作（继续/注意/止损）
        status = "hold"
        tp1_slow_bull = False
        tp1_sell_shares = 0
        if h_bull_phase == "slow" and pnl_pct is not None and float(pnl_pct) >= float(tp_cfg.slow_bull_tp1_trigger_ret):
            # 注意：持仓分析是“无状态”的；这里的建议含义是“如果你还没分批止盈过”
            tp1_slow_bull = True
            tp1_sell_shares = calc_tp1_sell_shares(shares=int(shares), lot_size=100, cfg=tp_cfg)

        # soft_exit（日线MACD死叉x2+破MA20）在 bull 里容易把人抖飞：让利润跑就别用它当止盈按钮
        take_profit_soft = bool(
            h_bull_phase is None
            and (pnl_net is not None and float(pnl_net) > 0)
            and soft_exit
            and (close is not None and daily_ma20 is not None and close <= float(daily_ma20))
            and h_profile.stop_mode != "daily_ma20"
        )

        tp_plan = None
        tp_sell_shares_out = None
        tp_ref = None
        tp_stop = None
        if close is not None and effective_stop is not None and close <= float(effective_stop):
            status = "take_profit" if (pnl_net is not None and float(pnl_net) > 0) else "stop"
            if status == "take_profit":
                # 盈利状态触发“止损线”，本质是移动止盈/保护利润
                # 优先标明是“锚线”（周/日 MA）或“回撤线”，方便你用人话理解：跌破锚线/回撤线 -> 离场
                try:
                    if effective_ref and str(effective_ref).startswith(("周线", "日线")) and hard_stop is not None:
                        tp_ref = hard_ref
                        tp_stop = float(hard_stop)
                        tp_plan = "weekly_anchor_break" if str(hard_ref).startswith("周线") else "anchor_break"
                    elif profit_stop is not None and effective_ref == profit_ref:
                        tp_ref = profit_ref
                        tp_stop = float(profit_stop)
                        tp_plan = "profit_stop"
                    else:
                        tp_plan = "profit_stop"
                except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                    tp_plan = "profit_stop"
        elif tp1_slow_bull and tp1_sell_shares > 0:
            status = "take_profit"
            tp_plan = "slow_bull_tp1"
            tp_sell_shares_out = int(tp1_sell_shares)
        elif take_profit_soft:
            status = "take_profit"
            tp_plan = "soft_exit"
        elif left_mode and close is not None and hard_stop is not None and close <= float(hard_stop):
            # 左侧：跌破锚线不砍，只标记“未确认/需要盯”。
            status = "watch"
        elif close is not None and daily_ma20 is not None and close <= float(daily_ma20):
            status = "watch" if h_profile.stop_mode != "daily_ma20" else "stop"

        items.append(
            {
                "asset": asset,
                "symbol": sym,
                "symbol_input": sym_raw,
                "name": name,
                "ok": True,
                "asof": asof_str,
                "close": close,
                # 可解释因子面板（7因子）：给人看用，别拿它当圣杯。
                "factor_panel": factor_panel_7,
                "factor_panel_7": factor_panel_7,
                "shares": shares,
                "cost": cost,
                "entry_style": ("left" if left_mode else (entry_style if entry_style else None)),
                "market_value": float(shares) * float(close) if close is not None else None,
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
                "pnl_net_pct": pnl_pct,
                "regime": {
                    "mode": regime_mode,
                    "proxy": (h_regime_proxy or {}).get("symbol"),
                    "proxy_name": (h_regime_proxy or {}).get("name"),
                    "label": h_regime_label,
                    "bull_phase": h_bull_phase,
                    "error": h_regime_error,
                },
                "stops": {
                    "hard_stop": hard_stop,
                    "hard_ref": hard_ref,
                    "hard_enforced": bool(not left_mode),
                    "loss_stop": float(loss_stop),
                    "loss_tol_pct": float(tol_pct),
                    "profit_enabled": bool(tp_cfg.profit_stop_enabled),
                    "profit_stop": profit_stop,
                    "profit_ref": profit_ref,
                    "profit_lookback_days": profit_lookback_days,
                    "profit_dd_pct": profit_dd_pct,
                    "profit_min_profit_ret": float(tp_cfg.profit_stop_min_profit_ret),
                    "effective_stop": effective_stop,
                    "effective_ref": effective_ref,
                },
                "levels": {
                    "daily_ma20": daily_ma20,
                    "daily_ma60": daily_ma60,
                    "daily_ma120": daily_ma120,
                    "weekly_ma20": weekly_ma20,
                    "weekly_ma50": weekly_ma50,
                    "weekly_ma200": weekly_ma200,
                    "support_20w": support_20w,
                    "resistance_20w": resistance_20w,
                },
                "trend": {
                    "daily_ret_5d": daily_ret_5d,
                    "daily_ret_20d": daily_ret_20d,
                    "daily_ret_60d": daily_ret_60d,
                    "weekly_ret_4w": weekly_ret_4w,
                    "weekly_ret_12w": weekly_ret_12w,
                    "weekly_ret_26w": weekly_ret_26w,
                    "daily_slope_20d": daily_slope_20d,
                    "daily_slope_60d": daily_slope_60d,
                    "weekly_slope_12w": weekly_slope_12w,
                },
                "momentum": {
                    "daily": {
                        "rsi14": daily_rsi14,
                        "adx14": daily_adx14,
                        "macd": daily_macd2,
                        "macd_signal": daily_macd_sig2,
                        "macd_state": daily_macd_state2,
                    },
                    "weekly": {
                        "rsi14": weekly_rsi14,
                        "adx14": weekly_adx14,
                        "macd": weekly_macd,
                        "macd_signal": weekly_macd_sig,
                        "macd_state": weekly_macd_state,
                    },
                },
                "signals": {
                    "soft_exit_daily_macd_ma20": bool(soft_exit and (close is not None and daily_ma20 is not None and close <= float(daily_ma20))),
                    "tp1_slow_bull": bool(tp1_slow_bull and tp1_sell_shares > 0),
                },
                "take_profit": {
                    "plan": tp_plan,
                    "sell_shares": (int(tp_sell_shares_out) if tp_sell_shares_out else None),
                    "trigger_ret": float(tp_cfg.slow_bull_tp1_trigger_ret),
                    "sell_ratio": float(tp_cfg.slow_bull_tp1_sell_ratio),
                    "bull_phase": h_bull_phase,
                    "ref": tp_ref,
                    "stop": tp_stop,
                },
                "status": status,
            }
        )

    # as_of：尽量用持仓里最后一根日线日期（字符串字典序可比：YYYY-MM-DD）。
    as_of = None
    try:
        for it in items:
            if not isinstance(it, dict):
                continue
            d = str(it.get("asof") or "").strip()
            if not d:
                continue
            if as_of is None or d > as_of:
                as_of = d
    except (AttributeError):  # noqa: BLE001
        as_of = None

    return sanitize_for_json(
        {
            "generated_at": datetime.now().isoformat(),
            "as_of": as_of,
            "regime_index": idx_raw,
            "market_regime": regime_dict,
            "market_regime_error": regime_error,
            "market_regime_index": regime_index_eff,
            "market_regime_mode": regime_mode,
            "stop_trigger": "close_only",
            "stop_follow_regime": bool(regime_mode != "off"),
            "holdings": items,
        }
    )


def analyze_etf_holdings(
    *,
    holdings: list[dict[str, Any]],
    regime_index: str = "sh000300",
    regime_canary_downgrade: bool = True,
    sell_cost_yuan: float = 5.0,
    small_position_threshold_yuan: float = 2000.0,
    tol_pct_small: float = 0.10,
    tol_pct_default: float = 0.05,
) -> dict[str, Any]:
    """
    兼容旧接口：只分析 ETF 持仓（研究用途）。
    """
    holdings2 = []
    for h in holdings:
        if isinstance(h, dict):
            holdings2.append({**h, "asset": "etf"})
    return analyze_holdings(
        holdings=holdings2,
        regime_index=regime_index,
        regime_canary_downgrade=bool(regime_canary_downgrade),
        sell_cost_yuan=sell_cost_yuan,
        small_position_threshold_yuan=small_position_threshold_yuan,
        tol_pct_small=tol_pct_small,
        tol_pct_default=tol_pct_default,
    )
