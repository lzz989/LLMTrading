# -*- coding: utf-8 -*-
"""
因子研究最小闭环（Phase1 / P0）

目标：
- 给定固定 universe + as_of，多次运行结果一致（可复现）
- 输出 IC/IR（1/5/10/20）+ 衰减 + 样本量/剔除比例 + 成本敏感性（最小佣金 5 元 + 滑点）
- 输出结构化 JSON/CSV，方便 DuckDB SQL 查询

注意（别自欺欺人）：
- IC 这里用 Spearman（rank IC），更稳健
- 执行口径默认 T+1：t 日因子 -> 从 t+1 开盘买入 -> t+1+h 日收盘退出
- 本模块是研究工具，不构成投资建议
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

import math


try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("缺依赖：请先安装 requirements.txt（pandas/numpy）") from exc

from ..akshare_source import FetchParams
from ..data_cache import fetch_daily_cached
from ..indicators import (
    add_adx,
    add_atr,
    add_bollinger_bands,
    add_ichimoku,
    add_macd,
    add_moving_averages,
    add_rsi,
)
from ..pipeline import write_json
from ..resample import resample_to_weekly
from ..tradeability import TradeabilityConfig, tradeability_flags
from ..utils_time import parse_date_any_opt


ScanAsset = Literal["etf", "stock", "index"]
ScanFreq = Literal["daily", "weekly"]
CANONICAL_HORIZONS: tuple[int, ...] = (1, 5, 10, 20)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _clip01(x: Any) -> float:
    v = _safe_float(x)
    if v is None:
        return 0.0
    return float(max(0.0, min(1.0, v)))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # 数值稳定一点
    if isinstance(x, np.ndarray):
        xx = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-xx))
    xx2 = float(max(-60.0, min(60.0, float(x))))
    return float(1.0 / (1.0 + math.exp(-xx2)))


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    研究链路只认这几列：date/open/high/low/close/volume/(amount 可选)。
    缺了就补（ETF/指数有时只有 close）。
    """
    df2 = df.copy()
    if "date" not in df2.columns:
        raise ValueError("缺少 date 列")
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "close" not in df2.columns:
        raise ValueError("缺少 close 列")

    for col in ("open", "high", "low"):
        if col not in df2.columns:
            df2[col] = df2["close"]
    if "volume" not in df2.columns:
        df2["volume"] = 0.0

    for c in ("open", "high", "low", "close", "volume"):
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    if "amount" not in df2.columns:
        # amount 缺失：用 close*volume 粗估（研究用途）
        df2["amount"] = df2["close"] * df2["volume"]
    else:
        df2["amount"] = pd.to_numeric(df2["amount"], errors="coerce")

    return df2


def compute_forward_returns(
    df: pd.DataFrame,
    *,
    horizons: list[int],
    t_plus_one: bool = True,
) -> pd.DataFrame:
    """
    forward return（研究用途）：
    - T+1：entry=下一交易日 open；exit=entry+h 日后的 close（h=1/5/10/20）
    """
    dfx = _ensure_ohlcv(df)
    close = pd.to_numeric(dfx["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(dfx["open"], errors="coerce").astype(float)

    out = pd.DataFrame({"date": dfx["date"]})

    for h in horizons:
        hh = int(h)
        if hh <= 0:
            continue
        if t_plus_one:
            entry = open_.shift(-1)
            exit_ = close.shift(-(1 + hh))
        else:
            entry = close
            exit_ = close.shift(-hh)

        r = (exit_ / entry) - 1.0
        out[f"fwd_ret_{hh}"] = r.replace([np.inf, -np.inf], np.nan)
    return out


def compute_tradeability_mask(
    df: pd.DataFrame,
    *,
    cfg: TradeabilityConfig,
) -> pd.DataFrame:
    """
    给每个 date 打标：下一交易日是否“理论上能在开盘成交”（研究用途粗估）。
    """
    dfx = _ensure_ohlcv(df)
    out = pd.DataFrame({"date": dfx["date"]})

    # 用“t+1 的日线”判断是否能买入（开盘一字/停牌等）
    op = pd.to_numeric(dfx["open"], errors="coerce").astype(float).shift(-1)
    hp = pd.to_numeric(dfx["high"], errors="coerce").astype(float).shift(-1)
    lp = pd.to_numeric(dfx["low"], errors="coerce").astype(float).shift(-1)
    prev_close = pd.to_numeric(dfx["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(dfx["volume"], errors="coerce").astype(float).shift(-1)
    amt = pd.to_numeric(dfx["amount"], errors="coerce").astype(float).shift(-1)

    halted = []
    locked_up = []
    locked_dn = []
    one_word = []
    for i in range(len(dfx)):
        f = tradeability_flags(
            open_price=_safe_float(op.iloc[i]),
            high_price=_safe_float(hp.iloc[i]),
            low_price=_safe_float(lp.iloc[i]),
            prev_close=_safe_float(prev_close.iloc[i]),
            volume=_safe_float(vol.iloc[i]),
            amount=_safe_float(amt.iloc[i]),
            cfg=cfg,
        )
        halted.append(bool(f.get("halted")))
        locked_up.append(bool(f.get("locked_limit_up")))
        locked_dn.append(bool(f.get("locked_limit_down")))
        one_word.append(bool(f.get("one_word")))

    out["halted_t1"] = halted
    out["one_word_t1"] = one_word
    out["locked_limit_up_t1"] = locked_up
    out["locked_limit_down_t1"] = locked_dn
    out["tradeable_t1"] = (~out["halted_t1"]) & (~out["locked_limit_up_t1"]) & (~out["locked_limit_down_t1"])
    return out


def _dyn_price_step(close: float) -> float:
    c = float(close)
    if c < 1.0:
        return 0.05
    if c < 2.0:
        return 0.1
    if c < 10.0:
        return 0.5
    return 1.0


def compute_factor_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 17 个“技术因子 score” + 5 个 game_theory proxy 的 score（研究用途）。
    输出列都是可 SQL 的扁平字段（factor_<name>）。
    """
    dfx = _ensure_ohlcv(df)

    # 常用指标一次性补齐
    dfx = add_moving_averages(dfx, ma_fast=20, ma_slow=50)
    dfx = add_moving_averages(dfx, ma_fast=60, ma_slow=200)
    dfx = add_macd(dfx, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
    dfx = add_rsi(dfx, period=14, out_col="rsi14")
    dfx = add_adx(dfx, period=14, adx_col="adx14", di_plus_col="di_plus14", di_minus_col="di_minus14")
    dfx = add_atr(dfx, period=14, out_col="atr14")
    dfx = add_bollinger_bands(dfx, window=20, k=2.0, mid_col="boll_mid", upper_col="boll_upper", lower_col="boll_lower", bandwidth_col="boll_bandwidth")
    dfx = add_ichimoku(dfx, tenkan=9, kijun=26, span_b=52, displacement=26, prefix="ichimoku_")

    close = pd.to_numeric(dfx["close"], errors="coerce").astype(float)
    open_ = pd.to_numeric(dfx["open"], errors="coerce").astype(float)
    high = pd.to_numeric(dfx["high"], errors="coerce").astype(float)
    low = pd.to_numeric(dfx["low"], errors="coerce").astype(float)
    vol = pd.to_numeric(dfx["volume"], errors="coerce").fillna(0.0).astype(float)
    amt = pd.to_numeric(dfx["amount"], errors="coerce").fillna(0.0).astype(float)

    out = pd.DataFrame({"date": dfx["date"]})

    # --- trend ---
    # ma_cross
    ma20 = pd.to_numeric(dfx.get("ma20"), errors="coerce").astype(float)
    ma50 = pd.to_numeric(dfx.get("ma50"), errors="coerce").astype(float)
    dist_pct = (ma20 - ma50) / ma50.replace({0.0: np.nan})
    out["factor_ma_cross"] = pd.Series(_sigmoid((dist_pct * 20.0).to_numpy(dtype=float))).astype(float)

    # macd
    macd = pd.to_numeric(dfx.get("macd"), errors="coerce").astype(float)
    macd_sig = pd.to_numeric(dfx.get("macd_signal"), errors="coerce").astype(float)
    hist = pd.to_numeric(dfx.get("macd_hist"), errors="coerce").astype(float)
    prev_hist = hist.shift(1)
    hist_expanding = (hist.abs() > prev_hist.abs())
    hist_positive = hist > 0
    score_macd = 0.5 + np.where(macd > macd_sig, 0.2, -0.2) + np.where(macd > 0, 0.15, -0.15)
    score_macd = score_macd + np.where(hist_positive & hist_expanding, 0.15, np.where((~hist_positive) & hist_expanding, -0.15, 0.0))
    out["factor_macd"] = np.clip(score_macd, 0.0, 1.0)

    # adx
    adx = pd.to_numeric(dfx.get("adx14"), errors="coerce").astype(float)
    di_p = pd.to_numeric(dfx.get("di_plus14"), errors="coerce").astype(float)
    di_m = pd.to_numeric(dfx.get("di_minus14"), errors="coerce").astype(float)
    strength = np.clip(adx / 50.0, 0.0, 1.0)
    strong = adx > 25.0
    score_adx = np.where(strong & (di_p > di_m), 0.5 + 0.5 * strength, np.where(strong & (di_m > di_p), 0.5 - 0.5 * strength, 0.5))
    out["factor_adx"] = np.clip(score_adx, 0.0, 1.0)

    # ichimoku
    tenkan = pd.to_numeric(dfx.get("ichimoku_tenkan"), errors="coerce").astype(float)
    kijun = pd.to_numeric(dfx.get("ichimoku_kijun"), errors="coerce").astype(float)
    span_a = pd.to_numeric(dfx.get("ichimoku_span_a"), errors="coerce").astype(float)
    span_b = pd.to_numeric(dfx.get("ichimoku_span_b"), errors="coerce").astype(float)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bot = np.minimum(span_a, span_b)
    score_ichi = np.full(len(dfx), 0.5, dtype=float)
    score_ichi = score_ichi + np.where(close > cloud_top, 0.2, np.where(close < cloud_bot, -0.2, 0.0))
    score_ichi = score_ichi + np.where(tenkan > kijun, 0.15, -0.15)
    score_ichi = score_ichi + np.where(span_a > span_b, 0.1, -0.1)
    score_ichi = score_ichi + np.where(close > kijun, 0.05, -0.05)
    out["factor_ichimoku"] = np.clip(score_ichi, 0.0, 1.0)

    # --- momentum ---
    # rsi
    rsi = pd.to_numeric(dfx.get("rsi14"), errors="coerce").astype(float)
    oversold = 30.0
    overbought = 70.0
    score_rsi = np.where(
        rsi <= oversold,
        0.7 + 0.3 * (oversold - rsi) / oversold,
        np.where(
            rsi >= overbought,
            0.3 - 0.3 * (rsi - overbought) / (100.0 - overbought),
            0.3 + 0.4 * (rsi - oversold) / (overbought - oversold),
        ),
    )
    out["factor_rsi"] = np.clip(score_rsi, 0.0, 1.0)

    # roc
    roc_period = 12
    roc = (close - close.shift(roc_period)) / close.shift(roc_period) * 100.0
    out["factor_roc"] = pd.Series(_sigmoid((roc / 5.0).to_numpy(dtype=float))).astype(float)

    # momentum
    mom_s = (close - close.shift(5)) / close.shift(5)
    mom_m = (close - close.shift(20)) / close.shift(20)
    mom_l = (close - close.shift(60)) / close.shift(60)
    weighted_mom = mom_s * 0.2 + mom_m * 0.3 + mom_l * 0.5
    out["factor_momentum"] = pd.Series(_sigmoid((weighted_mom * 10.0).to_numpy(dtype=float))).astype(float)

    # --- volume ---
    # volume_ratio
    vol_ma = vol.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol / vol_ma.replace({0.0: np.nan})
    price_chg = close.pct_change()
    score_vr = np.full(len(dfx), 0.5, dtype=float)
    high_vol = vol_ratio >= 2.0
    low_vol = vol_ratio <= 0.5
    score_vr = np.where(high_vol & (price_chg > 0.01), np.minimum(1.0, 0.6 + vol_ratio * 0.1), score_vr)
    score_vr = np.where(high_vol & (price_chg < -0.01), np.maximum(0.0, 0.4 - vol_ratio * 0.1), score_vr)
    score_vr = np.where(low_vol & (price_chg > 0.0), 0.55, score_vr)
    score_vr = np.where(low_vol & (price_chg < 0.0), 0.45, score_vr)
    out["factor_volume_ratio"] = np.clip(score_vr, 0.0, 1.0)

    # obv
    # 方向：close 上涨=+vol，下跌=-vol，不变=0；再 cumsum
    sign = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
    obv_flow = sign * vol.to_numpy(dtype=float)
    if len(obv_flow) > 0:
        obv_flow[0] = vol.iloc[0] if len(vol) > 0 else 0.0
    obv = pd.Series(obv_flow, index=dfx.index).cumsum()
    obv_ma = obv.rolling(window=20, min_periods=20).mean()
    obv_trend = obv > obv_ma
    price_trend = close > close.shift(20)
    price_new_high = close >= close.rolling(window=20, min_periods=20).max()
    obv_new_high = obv >= obv.rolling(window=20, min_periods=20).max()
    bearish_div = price_new_high & (~obv_new_high)
    bullish_div = (~price_new_high) & obv_new_high
    score_obv = 0.5 + np.where(obv_trend, 0.2, -0.2)
    score_obv = np.where(bullish_div, score_obv + 0.15, np.where(bearish_div, score_obv - 0.15, score_obv))
    out["factor_obv"] = np.clip(score_obv, 0.0, 1.0)

    # mfi
    typical = (high + low + close) / 3.0
    raw_mf = typical * vol
    tp_diff = typical.diff()
    pos_flow = raw_mf.where(tp_diff > 0.0, 0.0)
    neg_flow = raw_mf.where(tp_diff < 0.0, 0.0).abs()
    pos_sum = pos_flow.rolling(window=14, min_periods=14).sum()
    neg_sum = neg_flow.rolling(window=14, min_periods=14).sum()
    mr = pos_sum / (neg_sum + 1e-10)
    mfi = 100.0 - (100.0 / (1.0 + mr))
    mfi_overbought = 80.0
    mfi_oversold = 20.0
    score_mfi = np.where(
        mfi <= mfi_oversold,
        0.7 + 0.3 * (mfi_oversold - mfi) / mfi_oversold,
        np.where(
            mfi >= mfi_overbought,
            0.3 - 0.3 * (mfi - mfi_overbought) / (100.0 - mfi_overbought),
            0.3 + 0.4 * (mfi - mfi_oversold) / (mfi_overbought - mfi_oversold),
        ),
    )
    out["factor_mfi"] = np.clip(score_mfi, 0.0, 1.0)

    # --- volatility ---
    atr = pd.to_numeric(dfx.get("atr14"), errors="coerce").astype(float)
    atr_pct = atr / close.replace({0.0: np.nan})
    # 低波动=高分：用滚动 z-score 近似 percentile（避免未来函数）
    atr_mu = atr_pct.rolling(window=252, min_periods=60).mean()
    atr_sd = atr_pct.rolling(window=252, min_periods=60).std(ddof=0).replace({0.0: np.nan})
    atr_z = (atr_pct - atr_mu) / atr_sd
    out["factor_atr"] = pd.Series(_sigmoid((-atr_z).to_numpy(dtype=float))).astype(float)

    # bollinger
    upper = pd.to_numeric(dfx.get("boll_upper"), errors="coerce").astype(float)
    lower = pd.to_numeric(dfx.get("boll_lower"), errors="coerce").astype(float)
    middle = pd.to_numeric(dfx.get("boll_mid"), errors="coerce").astype(float)
    bw = (upper - lower)
    percent_b = (close - lower) / bw.replace({0.0: np.nan})
    score_boll = np.where(
        percent_b <= 0.0,
        0.8,
        np.where(
            percent_b >= 1.0,
            0.2,
            np.where(
                percent_b < 0.5,
                0.5 + 0.3 * (0.5 - percent_b),
                0.5 - 0.3 * (percent_b - 0.5),
            ),
        ),
    )
    out["factor_bollinger"] = np.clip(score_boll, 0.0, 1.0)

    # --- pattern ---
    # zt_type（循环做，别装逼写 rolling.apply）
    zt_thresh = 0.095
    lookback = 10
    zt_score = np.full(len(dfx), 0.5, dtype=float)
    last_zt_idx: int | None = None
    last_type_score: float = 0.5
    for i in range(1, len(dfx)):
        ret = _safe_float((close.iloc[i] / close.iloc[i - 1] - 1.0) if _safe_float(close.iloc[i - 1]) else None)
        if ret is not None and ret >= zt_thresh:
            # 计算“涨停类型”分（用当天K线特征）
            prev_close = float(close.iloc[i - 1]) if _safe_float(close.iloc[i - 1]) else float(close.iloc[i])
            zt_open = float(open_.iloc[i])
            zt_high = float(high.iloc[i])
            zt_low = float(low.iloc[i])
            zt_close = float(close.iloc[i])
            price_range = zt_high - zt_low
            body = abs(zt_close - zt_open)
            if price_range < prev_close * 0.005:
                last_type_score = 0.6
            elif (zt_low == zt_open) and (body < price_range * 0.3):
                last_type_score = 0.7
            elif price_range > prev_close * 0.05:
                last_type_score = 0.3
            else:
                last_type_score = 0.8
            last_zt_idx = i

        if last_zt_idx is None:
            zt_score[i] = 0.5
            continue

        days_since = i - last_zt_idx
        if days_since <= 0 or days_since > lookback:
            zt_score[i] = 0.5
            continue

        if days_since == 1:
            day_score = 0.7
        elif days_since == 2:
            day_score = 0.85
        elif days_since == 3:
            day_score = 0.7
        elif days_since <= 5:
            day_score = 0.5
        else:
            day_score = 0.3
        zt_score[i] = float(0.5 * last_type_score + 0.5 * day_score)
    out["factor_zt_type"] = np.clip(zt_score, 0.0, 1.0)

    # pullback（循环做）
    ma_pb = close.rolling(window=10, min_periods=10).mean()
    pb_score = np.full(len(dfx), 0.5, dtype=float)
    for i in range(len(dfx)):
        if i < 20 or i < 10:
            pb_score[i] = 0.5
            continue
        win_hi = high.iloc[i - 19 : i + 1].to_numpy(dtype=float)
        if len(win_hi) != 20:
            pb_score[i] = 0.5
            continue
        j = int(np.argmax(win_hi))
        recent_high = float(win_hi[j])
        days_from_high = int(19 - j)
        cur_close = float(close.iloc[i])
        if recent_high <= 0:
            pb_score[i] = 0.5
            continue
        pullback_pct = (recent_high - cur_close) / recent_high
        px_20d_ago = float(close.iloc[i - 20])
        prior_gain = (recent_high - px_20d_ago) / px_20d_ago if px_20d_ago > 0 else 0.0
        cur_ma = float(ma_pb.iloc[i]) if _safe_float(ma_pb.iloc[i]) else float("nan")
        dist_to_ma = (cur_close - cur_ma) / cur_ma if (math.isfinite(cur_ma) and cur_ma > 0) else float("nan")

        avg_vol_20 = float(vol.iloc[i - 20 : i - 5].mean())
        recent_vol = float(vol.iloc[i - 4 : i + 1].mean())
        vol_shrink = bool(avg_vol_20 > 0 and recent_vol < avg_vol_20 * 0.7)

        score = 0.5
        if prior_gain >= 0.08:
            score += 0.15
        else:
            score -= 0.1

        if pullback_pct <= 0.06:
            if pullback_pct >= 0.02:
                score += 0.2
            else:
                score += 0.1
        else:
            score -= 0.15

        if math.isfinite(dist_to_ma):
            if -0.02 <= dist_to_ma <= 0.02:
                score += 0.15
            elif dist_to_ma < -0.02:
                score -= 0.1

        if vol_shrink:
            score += 0.1

        pb_score[i] = float(max(0.0, min(1.0, score)))
    out["factor_pullback"] = np.clip(pb_score, 0.0, 1.0)

    # candle_pattern（3根K线）
    candle_score = np.full(len(dfx), 0.5, dtype=float)
    for i in range(2, len(dfx)):
        c_open, c_high, c_low, c_close = float(open_.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        p_open, p_close = float(open_.iloc[i - 1]), float(close.iloc[i - 1])
        p2_open, p2_close = float(open_.iloc[i - 2]), float(close.iloc[i - 2])

        c_body = abs(c_close - c_open)
        c_range = c_high - c_low
        c_body_ratio = c_body / c_range if c_range > 0 else 0.0
        c_bull = c_close > c_open
        p_bull = p_close > p_open
        p2_bull = p2_close > p2_open

        score = 0.5

        lower_shadow = min(c_open, c_close) - c_low
        upper_shadow = c_high - max(c_open, c_close)

        if c_range > 0 and lower_shadow > c_body * 2 and upper_shadow < c_body * 0.5:
            score += 0.15
        if c_range > 0 and upper_shadow > c_body * 2 and lower_shadow < c_body * 0.5:
            score += 0.1
        if c_body_ratio < 0.1:
            score += 0.05
        # engulfing
        if (not p_bull) and c_bull and (c_close > p_open) and (c_open < p_close):
            score += 0.2
        if p_bull and (not c_bull) and (c_close < p_open) and (c_open > p_close):
            score -= 0.2
        # morning/evening star（简化）
        p_body = abs(p_close - p_open)
        p2_range = float(high.iloc[i - 2] - low.iloc[i - 2])
        p_body_small = bool(p2_range > 0 and p_body < p2_range * 0.3)
        if (not p2_bull) and p_body_small and c_bull:
            if c_close > (p2_open + p2_close) / 2:
                score += 0.2
        if p2_bull and p_body_small and (not c_bull):
            if c_close < (p2_open + p2_close) / 2:
                score -= 0.2

        candle_score[i] = float(max(0.0, min(1.0, score)))
    out["factor_candle_pattern"] = np.clip(candle_score, 0.0, 1.0)

    # --- market ---
    # regime（用 MA50/MA200 + MACD + drawdown_252）
    ma200 = pd.to_numeric(dfx.get("ma200"), errors="coerce").astype(float)
    ma_bull = ma50 > ma200
    macd_bull = macd > macd_sig
    macd_above0 = macd > 0.0
    roll_max_252 = close.rolling(window=252, min_periods=60).max()
    dd_252 = (close - roll_max_252) / roll_max_252.replace({0.0: np.nan})
    deep_dd = dd_252 <= -0.25

    bull_cnt = (ma_bull.astype(int) + macd_bull.astype(int) + macd_above0.astype(int)).astype(int)
    bear_cnt = ((~ma_bull).astype(int) + (~macd_bull).astype(int) + (~macd_above0).astype(int) + deep_dd.astype(int)).astype(int)

    score_regime = np.where(
        deep_dd,
        0.2,
        np.where(
            bull_cnt >= 3,
            0.8,
            np.where(
                bear_cnt >= 3,
                0.2,
                np.where(
                    bull_cnt >= 2,
                    0.6,
                    0.4,
                ),
            ),
        ),
    )
    out["factor_regime"] = np.clip(score_regime, 0.0, 1.0)

    # breadth
    vol_ma20 = vol.rolling(window=20, min_periods=20).mean()
    vol_ratio2 = vol / vol_ma20.replace({0.0: np.nan})
    price_ma20 = close.rolling(window=20, min_periods=20).mean()
    price_strength = (close - price_ma20) / price_ma20.replace({0.0: np.nan})
    ret = close.pct_change()
    up_days = (ret.gt(0.0)).rolling(window=20, min_periods=20).sum().astype(float)
    dn_days = (ret.lt(0.0)).rolling(window=20, min_periods=20).sum().astype(float)
    up_dn = up_days / (dn_days + 1.0)

    score_breadth = np.full(len(dfx), 0.5, dtype=float)
    score_breadth = np.where((vol_ratio2 > 1.2) & (price_strength > 0), score_breadth + 0.2, score_breadth)
    score_breadth = np.where((vol_ratio2 < 0.8) & (price_strength < 0), score_breadth + 0.1, score_breadth)
    score_breadth = np.where((vol_ratio2 > 1.2) & (price_strength < -0.02), score_breadth - 0.15, score_breadth)
    score_breadth = np.where((vol_ratio2 < 0.8) & (price_strength > 0.02), score_breadth - 0.05, score_breadth)
    score_breadth = np.where(up_dn > 1.5, score_breadth + 0.1, score_breadth)
    score_breadth = np.where(up_dn < 0.7, score_breadth - 0.1, score_breadth)
    out["factor_breadth"] = np.clip(score_breadth, 0.0, 1.0)

    # --- game_theory proxies（尽量保持与 factors/game_theory.py 同口径，但要可批量） ---
    # liquidity_trap（bull/bear trap）
    lb = 20
    sweep_pct = 0.006
    swing_high = high.rolling(window=lb, min_periods=lb).max().shift(1)
    swing_low = low.rolling(window=lb, min_periods=lb).min().shift(1)
    bull_trap = (high > swing_high * (1.0 + sweep_pct)) & (close < swing_high)
    bear_trap = (low < swing_low * (1.0 - sweep_pct)) & (close > swing_low)
    strength_lt = np.where(bull_trap, (high - swing_high) / swing_high.replace({0.0: np.nan}), np.where(bear_trap, (swing_low - low) / swing_low.replace({0.0: np.nan}), 0.0))
    base_lt = np.where((bull_trap | bear_trap), np.clip(strength_lt / (sweep_pct * 2.0), 0.0, 1.0), 0.0)
    vol_mean_prev = vol.rolling(window=20, min_periods=20).mean().shift(1)
    vol_ratio3 = vol / vol_mean_prev.replace({0.0: np.nan})
    vol_boost = np.where((bull_trap | bear_trap) & (vol_ratio3 > 1.0), np.minimum(0.2, (vol_ratio3 - 1.0) * 0.1), 0.0)
    out["factor_liquidity_trap"] = np.clip(base_lt * 0.9 + vol_boost, 0.0, 1.0)

    # stop_cluster（离关键位越近，风险/触发越强；这里把它当“强度分”）
    score_full = 0.02
    ma60 = pd.to_numeric(dfx.get("ma60"), errors="coerce").astype(float)
    # 整数/半整数位（动态步长）
    steps = np.array([_dyn_price_step(float(x)) if _safe_float(x) else 1.0 for x in close.to_list()], dtype=float)
    int_level = np.round(close.to_numpy(dtype=float) / steps) * steps

    levels = np.column_stack(
        [
            swing_high.to_numpy(dtype=float),
            swing_low.to_numpy(dtype=float),
            ma20.to_numpy(dtype=float),
            ma60.to_numpy(dtype=float),
            ma200.to_numpy(dtype=float),
            int_level.astype(float),
        ]
    )
    close_np = close.to_numpy(dtype=float)
    dist_pct = np.full_like(close_np, np.inf, dtype=float)
    for j in range(levels.shape[1]):
        lvl = levels[:, j]
        ok = np.isfinite(lvl) & (lvl > 0) & np.isfinite(close_np)
        cur = np.full_like(close_np, np.inf, dtype=float)
        cur[ok] = np.abs(close_np[ok] - lvl[ok]) / lvl[ok]
        dist_pct = np.minimum(dist_pct, cur)
    dist_pct = np.where(np.isfinite(dist_pct), dist_pct, np.nan)
    out["factor_stop_cluster"] = np.clip(1.0 - (dist_pct / float(score_full)), 0.0, 1.0)

    # capitulation / fomo（ATR 倍数 + 放量 + RSI）
    atr14 = atr
    prev_close = close.shift(1)
    down_move = (prev_close - close).clip(lower=0.0)
    up_move = (close - prev_close).clip(lower=0.0)
    move_atr_dn = down_move / atr14.replace({0.0: np.nan})
    move_atr_up = up_move / atr14.replace({0.0: np.nan})
    mv_full = 2.0
    move_score_dn = np.clip((move_atr_dn - 1.0) / (mv_full - 1.0), 0.0, 1.0)
    move_score_up = np.clip((move_atr_up - 1.0) / (mv_full - 1.0), 0.0, 1.0)
    vol_score2 = np.clip((vol_ratio3 - 1.0) / 1.5, 0.0, 1.0)
    rsi_score_dn = np.clip((35.0 - rsi) / 15.0, 0.0, 1.0)
    rsi_score_up = np.clip((rsi - 65.0) / 15.0, 0.0, 1.0)
    out["factor_capitulation"] = np.clip(move_score_dn * 0.5 + vol_score2 * 0.3 + rsi_score_dn * 0.2, 0.0, 1.0)
    out["factor_fomo"] = np.clip(move_score_up * 0.5 + vol_score2 * 0.3 + rsi_score_up * 0.2, 0.0, 1.0)

    # wyckoff_phase_proxy（用 rolling 近似）
    lb_w = 60
    hi_w = high.rolling(window=lb_w, min_periods=lb_w).max()
    lo_w = low.rolling(window=lb_w, min_periods=lb_w).min()
    range_width_pct = (hi_w - lo_w) / close.replace({0.0: np.nan})
    range_score = np.clip(1.0 - (range_width_pct / 0.30), 0.0, 1.0)
    atr_med = atr14.rolling(window=lb_w, min_periods=lb_w).median()
    vol_contract = np.where((atr_med > 0) & np.isfinite(atr_med), np.clip(1.0 - (atr14 / atr_med), 0.0, 1.0), 0.0)
    # obv_dir: 简化为 obv vs obv_ma
    obv_dir_bull = np.where(obv > obv_ma, 1.0, np.where(obv < obv_ma, 0.0, 0.5))
    obv_dir_bear = np.where(obv < obv_ma, 1.0, np.where(obv > obv_ma, 0.0, 0.5))
    accumulation_like = np.clip(range_score * 0.4 + vol_contract * 0.4 + obv_dir_bull * 0.2, 0.0, 1.0)
    distribution_like = np.clip(range_score * 0.4 + vol_contract * 0.4 + obv_dir_bear * 0.2, 0.0, 1.0)
    out["factor_wyckoff_phase_proxy"] = np.clip(0.5 + 0.5 * (accumulation_like - distribution_like), 0.0, 1.0)

    return out


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """
    Spearman rank correlation（不依赖 scipy）。
    返回 None 表示样本不足/方差为 0。
    """
    if x.size != y.size or x.size < 3:
        return None

    # 去 NaN
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    xx = x[ok]
    yy = y[ok]

    # 全常数直接 None
    if float(np.nanstd(xx)) <= 1e-12 or float(np.nanstd(yy)) <= 1e-12:
        return None

    rx = pd.Series(xx).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(yy).rank(method="average").to_numpy(dtype=float)

    # Pearson(rx, ry)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.sqrt(np.sum(rx * rx)) * np.sqrt(np.sum(ry * ry)))
    if denom <= 1e-18:
        return None
    return float(np.sum(rx * ry) / denom)


def _run_tushare_macro_factor_research(
    *,
    as_of: date,
    freq: ScanFreq,
    start_date: date | None,
    context_index_symbol: str,
    price_cache_dir: Path,
    tushare_cache_dir: Path,
    cache_ttl_hours: float,
    horizons: list[int],
    rt_cost_rate: float,
    walk_forward: bool,
    train_window: int,
    test_window: int,
    step_window: int,
    top_quantile: float,
) -> dict[str, Any]:
    """
    TuShare “宏观温度计”研究（时间序列）：
    - ERP proxy series（基于 index pe_ttm & shibor）
    - HSGT north/south series

    口径：t 日宏观因子 -> 从 t+1 开盘执行 -> t+1+h 日收盘退出（与主研究一致）
    """
    out: dict[str, Any] = {
        "schema": "llm_trading.factor_research_macro.v1",
        "ok": False,
        "as_of": str(as_of),
        "ref_date": str(as_of),
        "asset": "index",
        "freq": str(freq),
        "context_index_symbol": str(context_index_symbol),
        "t_plus_one": True,
        "horizons": [int(x) for x in sorted({int(h) for h in (horizons or []) if int(h) > 0} | set(CANONICAL_HORIZONS))],
        "start_date": str(start_date) if start_date is not None else None,
        "source": {"name": "tushare", "price_source": "auto"},
        "cost": {"roundtrip_cost_rate": float(rt_cost_rate)},
        "components": {},
        "factors": [],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # 1) 指数价格（用于 forward returns）
    fp = FetchParams(
        asset="index",  # type: ignore[arg-type]
        symbol=str(context_index_symbol),
        start_date=start_date.strftime("%Y%m%d") if start_date else None,
        end_date=as_of.strftime("%Y%m%d"),
        adjust=None,
        source="auto",
    )
    try:
        idx_raw = fetch_daily_cached(fp, cache_dir=price_cache_dir, ttl_hours=float(cache_ttl_hours))
    except Exception as exc:  # noqa: BLE001
        out["components"]["index_price"] = {"ok": False, "error": str(exc)}
        return out
    if idx_raw is None or getattr(idx_raw, "empty", True):
        out["components"]["index_price"] = {"ok": False, "error": "指数K线为空"}
        return out

    idx = _ensure_ohlcv(idx_raw)
    idx = idx[idx["date"].dt.date <= as_of].reset_index(drop=True)
    if str(freq) == "weekly":
        idx = resample_to_weekly(idx)
        idx = _ensure_ohlcv(idx)
        idx = idx[idx["date"].dt.date <= as_of].reset_index(drop=True)
    if idx is None or getattr(idx, "empty", True) or len(idx) < 80:
        out["components"]["index_price"] = {"ok": False, "error": "指数K线样本不足"}
        return out
    out["components"]["index_price"] = {"ok": True, "rows": int(len(idx)), "last_date": str(pd.to_datetime(idx["date"].iloc[-1]).date())}

    fwd = compute_forward_returns(idx, horizons=out["horizons"], t_plus_one=True)
    panel = fwd.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    panel["date_d"] = panel["date"].dt.date

    # 2) 宏观序列（TuShare）
    try:
        from ..tushare_factors import compute_erp_proxy_series_tushare, compute_hsgt_flow_series_tushare

        erp_pack = compute_erp_proxy_series_tushare(
            as_of=as_of,
            index_symbol_prefixed=str(context_index_symbol),
            cache_dir=tushare_cache_dir / "erp_series",
            ttl_hours=float(cache_ttl_hours),
        )
        out["components"]["erp"] = {"ok": bool(erp_pack.get("ok")), "error": erp_pack.get("error")}
        if bool(erp_pack.get("ok")) and (erp_pack.get("rows") is not None):
            erp_df = pd.DataFrame(list(erp_pack.get("rows") or []))
            if (not erp_df.empty) and "date" in erp_df.columns:
                erp_df["date_d"] = pd.to_datetime(erp_df["date"], errors="coerce").dt.date
                erp_df["erp"] = pd.to_numeric(erp_df.get("erp"), errors="coerce")
                erp_df["erp_z"] = pd.to_numeric(erp_df.get("z"), errors="coerce")
                erp_df["erp_score01"] = pd.to_numeric(erp_df.get("score01"), errors="coerce")
                erp_df = erp_df[["date_d", "erp", "erp_z", "erp_score01"]]
                panel = panel.merge(erp_df, on="date_d", how="left")
    except (TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
        out["components"]["erp"] = {"ok": False, "error": str(exc)}

    try:
        from ..tushare_factors import compute_hsgt_flow_series_tushare

        hsgt_pack = compute_hsgt_flow_series_tushare(
            as_of=as_of,
            cache_dir=tushare_cache_dir / "hsgt_series",
            ttl_hours=float(cache_ttl_hours),
        )
        out["components"]["hsgt"] = {"ok": bool(hsgt_pack.get("ok")), "error": hsgt_pack.get("error")}
        if bool(hsgt_pack.get("ok")) and (hsgt_pack.get("rows") is not None):
            hsgt_df = pd.DataFrame(list(hsgt_pack.get("rows") or []))
            if (not hsgt_df.empty) and "date" in hsgt_df.columns:
                hsgt_df["date_d"] = pd.to_datetime(hsgt_df["date"], errors="coerce").dt.date
                hsgt_df["north_score01"] = pd.to_numeric(hsgt_df.get("north_score01"), errors="coerce")
                hsgt_df["south_score01"] = pd.to_numeric(hsgt_df.get("south_score01"), errors="coerce")
                hsgt_df = hsgt_df[["date_d", "north_score01", "south_score01"]]
                panel = panel.merge(hsgt_df, on="date_d", how="left")
    except (TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
        out["components"]["hsgt"] = {"ok": False, "error": str(exc)}

    # 3) 统计：时间序列 Spearman corr（因子 vs future returns）
    macro_factors = {
        "erp": "erp",
        "erp_score01": "erp_score01",
        "north_score01": "north_score01",
        "south_score01": "south_score01",
    }

    top_q = float(top_quantile or 0.8)
    if not (0.5 < top_q < 1.0):
        top_q = 0.8

    wf_enabled = bool(walk_forward)
    wf_train = max(10, int(train_window or 252))
    wf_test = max(5, int(test_window or 63))
    wf_step = max(1, int(step_window or wf_test))
    wf_min_days_train = max(10, int(wf_train * 0.3))
    wf_min_days_test = max(5, int(wf_test * 0.3))

    # 固定 schema（DuckDB view 依赖这些 key）
    factors_summary: dict[str, dict[str, Any]] = {}
    for fac in macro_factors.keys():
        row: dict[str, Any] = {"factor": fac}
        for hh0 in CANONICAL_HORIZONS:
            row[f"ic_{hh0}"] = None
            row[f"ir_{hh0}"] = None
            row[f"ic_samples_{hh0}"] = 0
            row[f"avg_cross_n_{hh0}"] = None
            row[f"ic_train_{hh0}"] = None
            row[f"ic_test_{hh0}"] = None
            row[f"top20_gross_mean_{hh0}"] = None
            row[f"top20_net_mean_{hh0}"] = None
            row[f"wf_windows_{hh0}"] = 0
            row[f"wf_ic_train_mean_{hh0}"] = None
            row[f"wf_ic_test_mean_{hh0}"] = None
            row[f"wf_ic_test_median_{hh0}"] = None
            row[f"wf_ic_test_pos_ratio_{hh0}"] = None
            row[f"wf_top20_net_mean_{hh0}"] = None
            row[f"wf_top20_net_pos_ratio_{hh0}"] = None
        factors_summary[fac] = row

    for fac, col in macro_factors.items():
        for hh0 in CANONICAL_HORIZONS:
            ycol = f"fwd_ret_{int(hh0)}"
            if ycol not in panel.columns or col not in panel.columns:
                continue

            sub = panel[["date_d", col, ycol]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")
            sub = sub.dropna(subset=["date_d", col, ycol]).sort_values("date_d").reset_index(drop=True)
            if sub.empty or len(sub) < 30:
                continue

            x = sub[col].to_numpy(dtype=float)
            y = sub[ycol].to_numpy(dtype=float)
            ic = _spearman_corr(x, y)
            if ic is None:
                continue

            # top quantile（按日期序列分位）
            top_g = None
            top_n = None
            try:
                thr = float(np.nanquantile(x, top_q))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                thr = float("nan")
            if math.isfinite(thr):
                sel = x >= thr
                if int(sel.sum()) >= 5:
                    try:
                        gross = float(np.nanmean(y[sel]))
                        net = float(gross - float(rt_cost_rate))
                        top_g = gross if math.isfinite(gross) else None
                        top_n = net if math.isfinite(net) else None
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        top_g = None
                        top_n = None

            # time split（同主研究：70/30）
            dates = sub["date_d"].to_list()
            split_idx = int(len(dates) * 0.7)
            train_end = dates[split_idx - 1] if split_idx >= 1 and split_idx < len(dates) else None
            test_start = dates[split_idx] if split_idx >= 1 and split_idx < len(dates) else None
            ic_tr = None
            ic_te = None
            if train_end is not None and test_start is not None:
                try:
                    tr = sub[sub["date_d"] <= train_end]
                    te = sub[sub["date_d"] >= test_start]
                    ic_tr = _spearman_corr(tr[col].to_numpy(dtype=float), tr[ycol].to_numpy(dtype=float)) if len(tr) >= 10 else None
                    ic_te = _spearman_corr(te[col].to_numpy(dtype=float), te[ycol].to_numpy(dtype=float)) if len(te) >= 10 else None
                except (AttributeError):  # noqa: BLE001
                    ic_tr = None
                    ic_te = None

            # walk-forward（窗口内直接算 corr）
            wf_tr_vals: list[float] = []
            wf_te_vals: list[float] = []
            wf_te_net: list[float] = []
            wf_te_net_pos = 0
            if wf_enabled and len(dates) >= (wf_train + wf_test + 1):
                i = 0
                while i + wf_train + wf_test <= len(dates):
                    tr_s = dates[i]
                    tr_e = dates[i + wf_train - 1]
                    te_s = dates[i + wf_train]
                    te_e = dates[i + wf_train + wf_test - 1]
                    i += wf_step

                    tr = sub[(sub["date_d"] >= tr_s) & (sub["date_d"] <= tr_e)]
                    te = sub[(sub["date_d"] >= te_s) & (sub["date_d"] <= te_e)]
                    if int(len(tr)) < int(wf_min_days_train) or int(len(te)) < int(wf_min_days_test):
                        continue
                    tr_ic = _spearman_corr(tr[col].to_numpy(dtype=float), tr[ycol].to_numpy(dtype=float))
                    te_ic = _spearman_corr(te[col].to_numpy(dtype=float), te[ycol].to_numpy(dtype=float))
                    if tr_ic is None or te_ic is None:
                        continue
                    wf_tr_vals.append(float(tr_ic))
                    wf_te_vals.append(float(te_ic))

                    # 同窗口的 top quantile mean(net)
                    try:
                        xv = te[col].to_numpy(dtype=float)
                        yv = te[ycol].to_numpy(dtype=float)
                        thr2 = float(np.nanquantile(xv, top_q))
                        if math.isfinite(thr2):
                            sel2 = xv >= thr2
                            if int(sel2.sum()) >= 3:
                                gross2 = float(np.nanmean(yv[sel2]))
                                net2 = float(gross2 - float(rt_cost_rate))
                                wf_te_net.append(net2)
                                if net2 > 0:
                                    wf_te_net_pos += 1
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        pass

            row = factors_summary[fac]
            row[f"ic_{hh0}"] = float(ic)
            row[f"ic_samples_{hh0}"] = int(len(sub))
            row[f"top20_gross_mean_{hh0}"] = top_g
            row[f"top20_net_mean_{hh0}"] = top_n
            row[f"ic_train_{hh0}"] = float(ic_tr) if ic_tr is not None else None
            row[f"ic_test_{hh0}"] = float(ic_te) if ic_te is not None else None

            if wf_te_vals:
                row[f"wf_windows_{hh0}"] = int(len(wf_te_vals))
                row[f"wf_ic_train_mean_{hh0}"] = float(np.mean(wf_tr_vals)) if wf_tr_vals else None
                row[f"wf_ic_test_mean_{hh0}"] = float(np.mean(wf_te_vals))
                row[f"wf_ic_test_median_{hh0}"] = float(np.median(wf_te_vals))
                row[f"wf_ic_test_pos_ratio_{hh0}"] = float(sum(1 for v in wf_te_vals if v > 0) / max(1, len(wf_te_vals)))
                if wf_te_net:
                    row[f"wf_top20_net_mean_{hh0}"] = float(np.mean(wf_te_net))
                    row[f"wf_top20_net_pos_ratio_{hh0}"] = float(wf_te_net_pos / max(1, len(wf_te_net)))

    out["factors"] = list(factors_summary.values())
    out["ok"] = any((_safe_float(f.get("ic_5")) is not None) or (_safe_float(f.get("ic_1")) is not None) for f in out["factors"])
    out["notes"] = "宏观因子按时间序列研究：Spearman(c_t, fwd_ret_{h})；walk-forward 在每个窗口内直接算 corr。"
    return out


@dataclass(frozen=True)
class FactorResearchParams:
    asset: ScanAsset
    freq: ScanFreq
    universe: list[str]
    start_date: date | None
    as_of: date | None
    horizons: list[int]
    # tradeability / cost
    limit_up_pct: float
    limit_down_pct: float
    min_fee_yuan: float
    slippage_bps_each_side: float
    notional_yuan: float
    # research config
    walk_forward: bool = True
    train_window: int = 252
    test_window: int = 63
    step_window: int = 63
    min_cross_n: int = 30
    top_quantile: float = 0.8
    # tushare factors pack（按需开启；没 token/接口挂了要能降级）
    include_tushare_micro: bool = False
    include_tushare_macro: bool = False
    context_index_symbol: str = "sh000300"
    max_tushare_symbols: int = 80


def run_factor_research(
    *,
    params: FactorResearchParams,
    cache_dir: Path,
    cache_ttl_hours: float,
    out_dir: Path,
    source: str = "auto",
) -> dict[str, Any]:
    """
    跑一遍因子研究并落盘：
    - factor_research_summary.json
    - factor_research_ic.csv（日期粒度）
    - factor_research_whitelist.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # TuShare 因子包（可选）：microstructure 是横截面因子；ERP/HSGT 更像“宏观温度计”（单序列）
    micro_enabled = bool(getattr(params, "include_tushare_micro", False)) and str(params.asset) == "stock"
    macro_enabled = bool(getattr(params, "include_tushare_macro", False))
    ctx_index_symbol = str(getattr(params, "context_index_symbol", "sh000300") or "sh000300").strip().lower() or "sh000300"
    max_ts_symbols = int(getattr(params, "max_tushare_symbols", 80) or 0)
    if max_ts_symbols < 0:
        max_ts_symbols = 0

    # microstructure 因子列（score01 推荐；ratio/z 作为补充）
    micro_factor_cols = [
        "factor_microstructure_score01",
        "factor_microstructure_z",
        "factor_microstructure_net_big_ratio",
        "factor_microstructure_net_total_ratio",
    ]
    micro_attempted = 0
    micro_ok = 0
    micro_skipped = 0
    micro_errors: list[dict[str, Any]] = []

    # 先拉数据 & 计算面板（symbol 级别）
    frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []

    for sym in params.universe:
        fp = FetchParams(
            asset=params.asset,
            symbol=str(sym),
            start_date=params.start_date.strftime("%Y%m%d") if params.start_date else None,
            end_date=params.as_of.strftime("%Y%m%d") if params.as_of else None,
            adjust="qfq" if params.asset in {"etf", "stock"} else None,
            source=str(source),
        )
        df_raw = fetch_daily_cached(fp, cache_dir=cache_dir, ttl_hours=float(cache_ttl_hours))
        if df_raw is None or getattr(df_raw, "empty", True):
            continue

        # 日线原始（microstructure 需要日成交额；weekly 时也需要对齐到“周末最后交易日”）
        dfx_daily = _ensure_ohlcv(df_raw)
        if params.as_of is not None:
            dfx_daily = dfx_daily[dfx_daily["date"].dt.date <= params.as_of].reset_index(drop=True)

        dfx = dfx_daily
        if params.freq == "weekly":
            dfx = resample_to_weekly(dfx_daily)
            dfx = _ensure_ohlcv(dfx)

        if dfx is None or getattr(dfx, "empty", True) or len(dfx) < 80:
            continue

        last_date = dfx["date"].iloc[-1]
        meta_rows.append({"symbol": str(sym), "rows": int(len(dfx)), "last_date": str(pd.to_datetime(last_date).date())})

        panel = compute_factor_panel(dfx)
        fwd = compute_forward_returns(dfx, horizons=params.horizons, t_plus_one=True)
        trad = compute_tradeability_mask(
            dfx,
            cfg=TradeabilityConfig(limit_up_pct=float(params.limit_up_pct), limit_down_pct=float(params.limit_down_pct), halt_vol_zero=True),
        )

        merged = panel.merge(fwd, on="date", how="left").merge(trad, on="date", how="left")

        # TuShare microstructure（横截面因子）：只对 stock 有意义；失败要能降级
        if micro_enabled:
            micro_df = None
            if (max_ts_symbols <= 0) or (micro_attempted < max_ts_symbols):
                micro_attempted += 1
                try:
                    from ..akshare_source import resolve_symbol
                    from ..tushare_factors import compute_stock_microstructure_series_tushare

                    sym_prefixed = resolve_symbol("stock", str(sym))
                    # 日成交额映射（元）：key=YYYY-MM-DD
                    amt_map: dict[str, float] = {}
                    try:
                        for _, row in dfx_daily[["date", "amount"]].dropna().iterrows():
                            ds = None
                            try:
                                ds = str(pd.to_datetime(row.get("date")).date())
                            except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                                ds = None
                            v = _safe_float(row.get("amount"))
                            if ds and v is not None:
                                amt_map[ds] = float(v)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        amt_map = {}

                    # as_of：优先用用户指定；否则用该 symbol 最后一根K线日期（再由全局 as_of_final 统一裁剪）
                    as_of_micro = params.as_of or pd.to_datetime(last_date).date()
                    ts_cache_root = cache_dir.parent / "tushare" / "micro_series"
                    pack = compute_stock_microstructure_series_tushare(
                        as_of=as_of_micro,
                        symbol_prefixed=str(sym_prefixed),
                        daily_amount_by_date=amt_map,
                        cache_dir=ts_cache_root,
                        ttl_hours=float(cache_ttl_hours),
                    )
                    if bool(pack.get("ok")) and (pack.get("rows") is not None):
                        micro_df = pd.DataFrame(list(pack.get("rows") or []))
                    else:
                        micro_df = None
                        micro_errors.append({"symbol": str(sym), "error": str(pack.get("error") or "microstructure 空/失败")})
                except (AttributeError) as exc:  # noqa: BLE001
                    micro_df = None
                    micro_errors.append({"symbol": str(sym), "error": str(exc)})
            else:
                micro_skipped += 1

            if micro_df is not None and (not micro_df.empty) and "date" in micro_df.columns:
                try:
                    micro_df["date"] = pd.to_datetime(micro_df["date"], errors="coerce")
                    micro_df = micro_df.rename(
                        columns={
                            "score01": "factor_microstructure_score01",
                            "z": "factor_microstructure_z",
                            "net_big_ratio": "factor_microstructure_net_big_ratio",
                            "net_total_ratio": "factor_microstructure_net_total_ratio",
                        }
                    )
                    keep = ["date"] + [c for c in micro_factor_cols if c in micro_df.columns]
                    micro_df = micro_df[keep]
                except (AttributeError):  # noqa: BLE001
                    micro_df = None

            if micro_df is not None and (not micro_df.empty):
                merged = merged.merge(micro_df, on="date", how="left")
                micro_ok += 1

            # 就算这只票拉 microstructure 失败，也要把列补上（不然 concat 后列不稳定）
            for c in micro_factor_cols:
                if c not in merged.columns:
                    merged[c] = np.nan

        merged["symbol"] = str(sym)
        frames.append(merged)

    if not frames:
        raise RuntimeError("universe 全部抓数失败/为空：没有可研究样本")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).reset_index(drop=True)

    # as_of：默认取各 symbol 共同的最小 last_date，确保“同一天横截面”不缺数据
    as_of_final = params.as_of
    if as_of_final is None:
        try:
            as_of_final = min(pd.to_datetime(x["last_date"]).date() for x in meta_rows if x.get("last_date"))
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_final = None
    if as_of_final is not None:
        data = data[data["date"].dt.date <= as_of_final].reset_index(drop=True)

    # 过滤：至少要能交易（t+1 不停牌/不一字）
    data["tradeable_t1"] = data.get("tradeable_t1", True).fillna(False).astype(bool)

    factor_cols = [c for c in data.columns if c.startswith("factor_")]
    horizons_eff = sorted({int(x) for x in (params.horizons or []) if int(x) > 0} | set(CANONICAL_HORIZONS))
    ret_cols = [f"fwd_ret_{h}" for h in horizons_eff if f"fwd_ret_{h}" in data.columns]

    df_cs = data[data["tradeable_t1"]].copy()

    # 样本外：时间切分 + walk-forward（滚动窗口）
    all_dates = sorted({d for d in df_cs["date"].dt.date.dropna().to_list()})
    split_ratio = 0.7
    split_idx = int(len(all_dates) * split_ratio)
    train_end = all_dates[split_idx - 1] if split_idx >= 1 and split_idx < len(all_dates) else None
    test_start = all_dates[split_idx] if split_idx >= 1 and split_idx < len(all_dates) else None

    # 成本：固定最小佣金 + 滑点（bps）
    rt_fixed = 2.0 * max(0.0, float(params.min_fee_yuan))
    rt_slip = 2.0 * max(0.0, float(params.slippage_bps_each_side)) / 10000.0
    rt_cost_rate = float(rt_fixed / max(1.0, float(params.notional_yuan)) + rt_slip)

    # 统计剔除比例（tradeability）
    total_rows = int(len(data))
    tradeable_rows = int(data["tradeable_t1"].sum())
    blocked_rows = int(total_rows - tradeable_rows)
    blocked_breakdown = {
        "halted_t1": int(data.get("halted_t1", False).fillna(False).sum()),
        "locked_limit_up_t1": int(data.get("locked_limit_up_t1", False).fillna(False).sum()),
        "locked_limit_down_t1": int(data.get("locked_limit_down_t1", False).fillna(False).sum()),
        "one_word_t1": int(data.get("one_word_t1", False).fillna(False).sum()),
    }

    # walk-forward windows（按日期序列划窗；train/test 都是“日粒度横截面 IC 的均值”）
    wf_enabled = bool(getattr(params, "walk_forward", True))
    wf_train = max(10, int(getattr(params, "train_window", 252) or 252))
    wf_test = max(5, int(getattr(params, "test_window", 63) or 63))
    wf_step = max(1, int(getattr(params, "step_window", wf_test) or wf_test))
    wf_windows: list[dict[str, Any]] = []
    if wf_enabled and len(all_dates) >= (wf_train + wf_test + 1):
        i = 0
        while i + wf_train + wf_test <= len(all_dates):
            tr_start = all_dates[i]
            tr_end = all_dates[i + wf_train - 1]
            te_start = all_dates[i + wf_train]
            te_end = all_dates[i + wf_train + wf_test - 1]
            wf_windows.append(
                {
                    "train_start": str(tr_start),
                    "train_end": str(tr_end),
                    "test_start": str(te_start),
                    "test_end": str(te_end),
                }
            )
            i += wf_step

    # IC timeseries（长表），方便 SQL（顺带写 top20 gross/net）
    ic_rows: list[dict[str, Any]] = []
    min_cross_n = max(5, int(getattr(params, "min_cross_n", 30) or 30))
    top_q = float(getattr(params, "top_quantile", 0.8) or 0.8)
    if not (0.5 < top_q < 1.0):
        top_q = 0.8

    # 分组只做一次，避免在 factor/horizon 循环里反复 groupby
    grp = df_cs.groupby(df_cs["date"].dt.date, sort=True)
    for fcol in factor_cols:
        fac_name = str(fcol.removeprefix("factor_"))
        for rcol in ret_cols:
            h = int(rcol.removeprefix("fwd_ret_"))
            for d, g in grp:
                x = pd.to_numeric(g[fcol], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(g[rcol], errors="coerce").to_numpy(dtype=float)
                ic = _spearman_corr(x, y)
                if ic is None:
                    continue
                ok = np.isfinite(x) & np.isfinite(y)
                n = int(ok.sum())

                top_g = None
                top_n = None
                if n >= int(min_cross_n):
                    x_ok = x[ok]
                    y_ok = y[ok]
                    try:
                        thr = float(np.nanquantile(x_ok, top_q))
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        thr = float("nan")
                    if math.isfinite(thr):
                        sel = x_ok >= thr
                        if int(sel.sum()) >= 3:
                            gross = float(np.nanmean(y_ok[sel]))
                            net = float(gross - rt_cost_rate)
                            if math.isfinite(gross):
                                top_g = float(gross)
                                top_n = float(net)

                ic_rows.append(
                    {
                        "date": str(d),
                        "factor": fac_name,
                        "horizon": int(h),
                        "ic": float(ic),
                        "n_obs": int(n),
                        "top20_gross": top_g,
                        "top20_net": top_n,
                    }
                )

    ic_df = pd.DataFrame(ic_rows)
    if not ic_df.empty:
        ic_df["date"] = pd.to_datetime(ic_df["date"], errors="coerce").dt.date
        ic_df["horizon"] = pd.to_numeric(ic_df["horizon"], errors="coerce")
        ic_df["ic"] = pd.to_numeric(ic_df["ic"], errors="coerce")
        ic_df["n_obs"] = pd.to_numeric(ic_df["n_obs"], errors="coerce")
        ic_df["top20_gross"] = pd.to_numeric(ic_df["top20_gross"], errors="coerce")
        ic_df["top20_net"] = pd.to_numeric(ic_df["top20_net"], errors="coerce")
        # 稳定字段（便于审计/SQL；多标的研究没有单一 symbol）
        ic_df["asset"] = str(params.asset)
        ic_df["freq"] = str(params.freq)
        ic_df["as_of"] = str(as_of_final) if as_of_final is not None else None
        ic_df["ref_date"] = str(as_of_final) if as_of_final is not None else None
        ic_df["source"] = "factor_research"

    # 落盘（先写 IC CSV，确保失败也有残骸可查）
    ic_path = out_dir / "factor_research_ic.csv"
    ic_df.to_csv(ic_path, index=False, encoding="utf-8")

    # 汇总：按 factor/horizon 出 IC/IR + time-split + walk-forward
    summary: dict[str, Any] = {}
    wf_min_days_train = max(10, int(wf_train * 0.3))
    wf_min_days_test = max(5, int(wf_test * 0.3))

    for fcol in factor_cols:
        fac = str(fcol.removeprefix("factor_"))
        fsum: dict[str, Any] = {"factor": fac}

        # 固定 schema（DuckDB view 依赖这些 key）
        for hh0 in CANONICAL_HORIZONS:
            fsum[f"ic_{hh0}"] = None
            fsum[f"ir_{hh0}"] = None
            fsum[f"ic_samples_{hh0}"] = 0
            fsum[f"avg_cross_n_{hh0}"] = None
            fsum[f"ic_train_{hh0}"] = None
            fsum[f"ic_test_{hh0}"] = None
            fsum[f"top20_gross_mean_{hh0}"] = None
            fsum[f"top20_net_mean_{hh0}"] = None
            fsum[f"wf_windows_{hh0}"] = 0
            fsum[f"wf_ic_train_mean_{hh0}"] = None
            fsum[f"wf_ic_test_mean_{hh0}"] = None
            fsum[f"wf_ic_test_median_{hh0}"] = None
            fsum[f"wf_ic_test_pos_ratio_{hh0}"] = None
            fsum[f"wf_top20_net_mean_{hh0}"] = None
            fsum[f"wf_top20_net_pos_ratio_{hh0}"] = None

        sub_f = ic_df[ic_df["factor"] == fac] if (not ic_df.empty) else ic_df
        for hh0 in CANONICAL_HORIZONS:
            sub = sub_f[sub_f["horizon"] == int(hh0)] if (not sub_f.empty) else sub_f
            if sub is None or getattr(sub, "empty", True):
                continue

            ics = pd.to_numeric(sub["ic"], errors="coerce").dropna().astype(float)
            if ics.empty:
                continue

            ic_mean = float(ics.mean())
            ic_std = float(ics.std(ddof=0)) if len(ics) >= 2 else None
            ir = (float(ic_mean) / float(ic_std)) if (ic_std is not None and ic_std > 1e-12) else None

            fsum[f"ic_{hh0}"] = ic_mean
            fsum[f"ir_{hh0}"] = ir
            fsum[f"ic_samples_{hh0}"] = int(len(ics))
            try:
                fsum[f"avg_cross_n_{hh0}"] = float(pd.to_numeric(sub["n_obs"], errors="coerce").dropna().astype(float).mean())
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                fsum[f"avg_cross_n_{hh0}"] = None

            try:
                fsum[f"top20_gross_mean_{hh0}"] = float(pd.to_numeric(sub["top20_gross"], errors="coerce").dropna().astype(float).mean())
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                fsum[f"top20_gross_mean_{hh0}"] = None
            try:
                fsum[f"top20_net_mean_{hh0}"] = float(pd.to_numeric(sub["top20_net"], errors="coerce").dropna().astype(float).mean())
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                fsum[f"top20_net_mean_{hh0}"] = None

            if train_end is not None and test_start is not None:
                try:
                    sub_tr = sub[sub["date"] <= train_end]
                    sub_te = sub[sub["date"] >= test_start]
                    fsum[f"ic_train_{hh0}"] = float(pd.to_numeric(sub_tr["ic"], errors="coerce").dropna().astype(float).mean()) if (not sub_tr.empty) else None
                    fsum[f"ic_test_{hh0}"] = float(pd.to_numeric(sub_te["ic"], errors="coerce").dropna().astype(float).mean()) if (not sub_te.empty) else None
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    pass

            # walk-forward：对每个 window，算 train/test 的 mean(ic)；再聚合
            if wf_windows:
                tr_means: list[float] = []
                te_means: list[float] = []
                te_net_means: list[float] = []
                te_net_pos: int = 0

                for w in wf_windows:
                    try:
                        tr_s = pd.to_datetime(w["train_start"]).date()
                        tr_e = pd.to_datetime(w["train_end"]).date()
                        te_s = pd.to_datetime(w["test_start"]).date()
                        te_e = pd.to_datetime(w["test_end"]).date()
                    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                        continue

                    sub_tr = sub[(sub["date"] >= tr_s) & (sub["date"] <= tr_e)]
                    sub_te = sub[(sub["date"] >= te_s) & (sub["date"] <= te_e)]
                    if int(len(sub_tr)) < int(wf_min_days_train) or int(len(sub_te)) < int(wf_min_days_test):
                        continue

                    tr_ic = pd.to_numeric(sub_tr["ic"], errors="coerce").dropna().astype(float)
                    te_ic = pd.to_numeric(sub_te["ic"], errors="coerce").dropna().astype(float)
                    if tr_ic.empty or te_ic.empty:
                        continue

                    tr_means.append(float(tr_ic.mean()))
                    te_means.append(float(te_ic.mean()))

                    te_net = pd.to_numeric(sub_te["top20_net"], errors="coerce").dropna().astype(float)
                    if not te_net.empty:
                        v = float(te_net.mean())
                        te_net_means.append(v)
                        if v > 0:
                            te_net_pos += 1

                if te_means:
                    fsum[f"wf_windows_{hh0}"] = int(len(te_means))
                    fsum[f"wf_ic_train_mean_{hh0}"] = float(np.mean(tr_means)) if tr_means else None
                    fsum[f"wf_ic_test_mean_{hh0}"] = float(np.mean(te_means))
                    fsum[f"wf_ic_test_median_{hh0}"] = float(np.median(te_means))
                    fsum[f"wf_ic_test_pos_ratio_{hh0}"] = float(sum(1 for v in te_means if v > 0) / max(1, len(te_means)))
                    fsum[f"wf_top20_net_mean_{hh0}"] = float(np.mean(te_net_means)) if te_net_means else None
                    fsum[f"wf_top20_net_pos_ratio_{hh0}"] = float(te_net_pos / max(1, len(te_net_means))) if te_net_means else None

        summary[fac] = fsum

    # 白/黑名单：优先用 walk-forward 的 OOS（否则退化到全样本）
    whitelist: list[str] = []
    blacklist: list[str] = []
    for fac, row in summary.items():
        wf_n = int(row.get("wf_windows_5") or 0)
        ic_oos = _safe_float(row.get("wf_ic_test_mean_5")) if wf_n >= 2 else None
        net_oos = _safe_float(row.get("wf_top20_net_mean_5")) if wf_n >= 2 else None
        if ic_oos is None:
            ic_oos = _safe_float(row.get("ic_test_5")) or _safe_float(row.get("ic_5"))
        if net_oos is None:
            net_oos = _safe_float(row.get("top20_net_mean_5"))

        if ic_oos is None:
            continue
        if ic_oos >= 0.02 and (net_oos is None or net_oos >= 0.0):
            whitelist.append(str(fac))
        if ic_oos <= -0.02 and (net_oos is None or net_oos <= 0.0):
            blacklist.append(str(fac))

    # TuShare 因子包：把“可用字段”纳入研究范围；宏观温度计单独落一份报告（避免横截面 IC 无意义）
    tushare_info: dict[str, Any] = {
        "enabled": {"microstructure": bool(micro_enabled), "macro": bool(macro_enabled)},
        "context_index_symbol": str(ctx_index_symbol),
    }
    if micro_enabled:
        tushare_info["microstructure"] = {
            "max_symbols": int(max_ts_symbols),
            "attempted": int(micro_attempted),
            "ok_symbols": int(micro_ok),
            "skipped_symbols": int(micro_skipped),
            "error_samples": micro_errors[:10],
            "note": "microstructure 需要 TuShare moneyflow；可能受积分/限流影响。",
        }

    if macro_enabled:
        if as_of_final is None:
            tushare_info["macro"] = {"ok": False, "error": "as_of 缺失：宏观温度计跳过"}
        else:
            try:
                macro_obj = _run_tushare_macro_factor_research(
                    as_of=as_of_final,
                    freq=params.freq,
                    start_date=params.start_date,
                    context_index_symbol=str(ctx_index_symbol),
                    price_cache_dir=cache_dir.parent / "index",
                    tushare_cache_dir=cache_dir.parent / "tushare",
                    cache_ttl_hours=float(cache_ttl_hours),
                    horizons=horizons_eff,
                    rt_cost_rate=float(rt_cost_rate),
                    walk_forward=bool(wf_enabled),
                    train_window=int(wf_train),
                    test_window=int(wf_test),
                    step_window=int(wf_step),
                    top_quantile=float(top_q),
                )
                write_json(out_dir / "factor_research_macro.json", macro_obj)
                tushare_info["macro"] = {
                    "ok": bool(macro_obj.get("ok")),
                    "file": "factor_research_macro.json",
                    "components": macro_obj.get("components"),
                    "note": macro_obj.get("notes"),
                }
            except Exception as exc:  # noqa: BLE001
                tushare_info["macro"] = {"ok": False, "error": str(exc)}

    summary_obj = {
        "schema": "llm_trading.factor_research.v1",
        "asset": str(params.asset),
        "freq": str(params.freq),
        "as_of": str(as_of_final) if as_of_final is not None else None,
        "ref_date": str(as_of_final) if as_of_final is not None else None,
        "start_date": str(params.start_date) if params.start_date is not None else None,
        "horizons": [int(x) for x in horizons_eff],
        "horizons_requested": [int(x) for x in (params.horizons or [])],
        "t_plus_one": True,
        "universe_size": int(len(params.universe)),
        "symbols_used": int(len(meta_rows)),
        "oos_split": {
            "mode": "time_split",
            "train_ratio": float(split_ratio),
            "train_end": str(train_end) if train_end is not None else None,
            "test_start": str(test_start) if test_start is not None else None,
            "unique_dates": int(len(all_dates)),
        },
        "walk_forward": {
            "enabled": bool(wf_enabled),
            "train_window": int(wf_train),
            "test_window": int(wf_test),
            "step_window": int(wf_step),
            "windows": wf_windows,
            "window_count": int(len(wf_windows)),
            "min_days_train": int(wf_min_days_train),
            "min_days_test": int(wf_min_days_test),
            "notes": "walk-forward 基于“每日横截面 IC”的窗口均值（train/test）。",
        },
        "tradeability": {
            "total_rows": int(total_rows),
            "tradeable_rows": int(tradeable_rows),
            "blocked_rows": int(blocked_rows),
            "blocked_breakdown": blocked_breakdown,
            "cfg": {"limit_up_pct": float(params.limit_up_pct), "limit_down_pct": float(params.limit_down_pct)},
        },
        "cost": {
            "min_fee_yuan_each_side": float(params.min_fee_yuan),
            "slippage_bps_each_side": float(params.slippage_bps_each_side),
            "notional_yuan": float(params.notional_yuan),
            "roundtrip_cost_rate": float(rt_cost_rate),
        },
        "tushare": tushare_info,
        "factors": list(summary.values()),
        "source": {"name": "factor_research", "price_source": str(source)},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(out_dir / "factor_research_summary.json", summary_obj)
    write_json(
        out_dir / "factor_research_whitelist.json",
        {
            "schema": "llm_trading.factor_research_whitelist.v1",
            "as_of": str(as_of_final) if as_of_final is not None else None,
            "ref_date": str(as_of_final) if as_of_final is not None else None,
            "asset": str(params.asset),
            "freq": str(params.freq),
            "source": {"name": "factor_research"},
            "whitelist": whitelist,
            "blacklist": blacklist,
            "notes": "阈值是研究用途的默认值（ic_5>=0.02 & ir_5>=0.3）；别迷信，先看报告再定策略。",
        },
    )

    write_json(
        out_dir / "factor_research_symbols.json",
        {
            "schema": "llm_trading.factor_research_symbols.v1",
            "as_of": str(as_of_final) if as_of_final is not None else None,
            "ref_date": str(as_of_final) if as_of_final is not None else None,
            "asset": str(params.asset),
            "freq": str(params.freq),
            "source": {"name": "factor_research"},
            "symbols": meta_rows,
        },
    )

    return summary_obj
