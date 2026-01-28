from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RegimeLabel = Literal["bull", "bear", "neutral", "unknown"]


@dataclass(frozen=True, slots=True)
class MarketRegime:
    symbol: str
    label: RegimeLabel
    last_date: str
    close: float
    ma_fast: float | None
    ma_slow: float | None
    ma_fast_slope_weeks: int
    ma_fast_slope: float | None
    macd: float | None
    macd_signal: float | None
    macd_state: str | None
    ret_1d: float | None
    vol_20d: float | None
    drawdown_252d: float | None
    mom_63d: float | None
    mom_126d: float | None
    mom_252d: float | None
    mom_risk_off: bool
    panic: bool
    scores: dict[str, int]
    rules: dict[str, dict[str, bool]]


def compute_market_regime(
    *,
    index_symbol: str,
    df_daily,
    ma_fast: int = 50,
    ma_slow: int = 200,
    ma_fast_slope_weeks: int = 4,
    confirm_days: int = 3,
) -> MarketRegime:
    """
    大盘牛熊/风险偏好粗判（研究用途）。

    核心思想：别搞花里胡哨的“玄学牛熊”，用朴素但更稳的“日线趋势 + 风险急剧恶化兜底”：
    - 日线 MA50 / MA200（而不是周线 MA50/MA200 这种离谱滞后）
    - MA50 相对 MA200 + MA50 斜率
    - MACD 方向/0轴
    - drawdown/波动冲击（panic）做兜底：不预测黑天鹅，只求更快识别“别交易了”

    输出 label：
    - bull：趋势共振更偏多
    - bear：趋势共振更偏空
    - neutral：中性/震荡（别强行站队）
    """
    try:
        import math
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    from .indicators import add_macd, add_moving_averages

    if df_daily is None or getattr(df_daily, "empty", True):
        raise ValueError("df_daily 为空")

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("df_daily 无有效K线")

    ma_fast2 = max(2, int(ma_fast))
    ma_slow2 = max(ma_fast2 + 1, int(ma_slow))
    slope_weeks = max(1, int(ma_fast_slope_weeks))
    slope_days = int(slope_weeks) * 5  # 近似交易日
    confirm = max(1, int(confirm_days))

    min_need = max(ma_slow2, ma_fast2 + slope_days) + 5
    if len(df) < int(min_need):
        last_dt = df.iloc[-1].get("date")
        last_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)
        close = float(df.iloc[-1].get("close") or 0.0)
        return MarketRegime(
            symbol=index_symbol,
            label="unknown",
            last_date=last_str,
            close=close,
            ma_fast=None,
            ma_slow=None,
            ma_fast_slope_weeks=int(slope_weeks),
            ma_fast_slope=None,
            macd=None,
            macd_signal=None,
            macd_state=None,
            ret_1d=None,
            vol_20d=None,
            drawdown_252d=None,
            mom_63d=None,
            mom_126d=None,
            mom_252d=None,
            mom_risk_off=False,
            panic=False,
            scores={"bull": 0, "bear": 0},
            rules={"bull": {}, "bear": {}},
        )

    df = add_moving_averages(df, ma_fast=int(ma_fast2), ma_slow=int(ma_slow2))
    df = add_macd(df, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    last = df.iloc[-1]
    last_dt = last.get("date")
    last_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)

    def f(v) -> float | None:
        try:
            x = None if v is None else float(v)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return None
        return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x

    close = f(last.get("close")) or 0.0
    ma_f = f(last.get(f"ma{int(ma_fast2)}"))
    ma_s = f(last.get(f"ma{int(ma_slow2)}"))

    macd = f(last.get("macd"))
    macd_sig = f(last.get("macd_signal"))
    macd_state = None
    if macd is not None and macd_sig is not None:
        macd_state = "bullish" if macd > macd_sig else ("bearish" if macd < macd_sig else "neutral")

    prev_ma_f = None
    try:
        prev = df.iloc[-1 - slope_days].get(f"ma{int(ma_fast2)}")
        prev_ma_f = f(prev)
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        prev_ma_f = None

    ma_slope = None
    if ma_f is not None and prev_ma_f is not None and prev_ma_f > 0:
        ma_slope = float(ma_f / prev_ma_f - 1.0)

    # 日收益/波动/回撤（兜底用：不预测黑天鹅，只求更快识别“别交易了”）
    ret_1d = None
    vol_20d = None
    dd_252 = None
    mom_63d = None
    mom_126d = None
    mom_252d = None
    mom_risk_off = False
    panic = False
    label_series = None
    try:
        close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
        prev_close = close_s.shift(1)
        r1 = (close_s / prev_close.replace({0.0: float("nan")})) - 1.0
        vol20 = r1.rolling(window=20, min_periods=20).std()

        roll_max = close_s.rolling(window=252, min_periods=20).max()
        dd = (close_s / roll_max.replace({0.0: float("nan")})) - 1.0

        ret_1d = f(r1.iloc[-1])
        vol_20d = f(vol20.iloc[-1])
        dd_252 = f(dd.iloc[-1])

        # 绝对动量（PAA/DAA 常用 “动量是否为正”当 risk-on/off 的开关）
        mom_63d = f(((close_s / close_s.shift(63).replace({0.0: float("nan")})) - 1.0).iloc[-1])
        mom_126d = f(((close_s / close_s.shift(126).replace({0.0: float("nan")})) - 1.0).iloc[-1])
        mom_252d = f(((close_s / close_s.shift(252).replace({0.0: float("nan")})) - 1.0).iloc[-1])
        # 保守的 risk-off：中期+长期都为负，才当“风险偏好明显转弱”
        mom_risk_off = bool(
            (mom_126d is not None and float(mom_126d) < 0.0) and (mom_252d is not None and float(mom_252d) < 0.0)
        )

        # panic：大幅下跌 + 波动冲击（越保守越不容易误报，但更晚）
        thresh_s = pd.concat([vol20 * 3.0, pd.Series([0.04] * len(df), index=df.index)], axis=1).max(axis=1)
        panic_s = ((r1 <= -thresh_s) | (dd <= -0.25)).fillna(False)
        panic = bool(panic_s.iloc[-1])

        # 生成“日线 raw label 序列”，用于 confirm_days 去抖（不未来函数）
        ma_f_s = pd.to_numeric(df[f"ma{int(ma_fast2)}"], errors="coerce").astype(float)
        ma_s_s = pd.to_numeric(df[f"ma{int(ma_slow2)}"], errors="coerce").astype(float)
        macd_s = pd.to_numeric(df["macd"], errors="coerce").astype(float)
        macd_sig_s = pd.to_numeric(df["macd_signal"], errors="coerce").astype(float)
        ma_slope_s = (ma_f_s / ma_f_s.shift(slope_days)) - 1.0

        bull_rules_s = {
            "close_above_ma_slow": (close_s > ma_s_s),
            "close_above_ma_fast": (close_s > ma_f_s),
            "ma_fast_above_ma_slow": (ma_f_s > ma_s_s),
            "ma_fast_rising": (ma_slope_s > 0),
            "macd_above_signal": (macd_s > macd_sig_s),
            "macd_above_zero": (macd_s > 0),
        }
        bear_rules_s = {
            "close_below_ma_slow": (close_s < ma_s_s),
            "close_below_ma_fast": (close_s < ma_f_s),
            "ma_fast_below_ma_slow": (ma_f_s < ma_s_s),
            "ma_fast_falling": (ma_slope_s < 0),
            "macd_below_signal": (macd_s < macd_sig_s),
            "macd_below_zero": (macd_s < 0),
        }

        bull_score_s = sum((x.fillna(False).astype(int)) for x in bull_rules_s.values())
        bear_score_s = sum((x.fillna(False).astype(int)) for x in bear_rules_s.values())

        label_series = pd.Series(["neutral"] * len(df), index=df.index, dtype="object")
        label_series = label_series.mask(
            (bull_score_s >= 4)
            & (bull_score_s > bear_score_s)
            & bull_rules_s["close_above_ma_slow"].fillna(False)
            & bull_rules_s["ma_fast_above_ma_slow"].fillna(False),
            "bull",
        )
        label_series = label_series.mask(
            (bear_score_s >= 4)
            & (bear_score_s > bull_score_s)
            & bear_rules_s["close_below_ma_slow"].fillna(False)
            & bear_rules_s["ma_fast_below_ma_slow"].fillna(False),
            "bear",
        )

        # deep drawdown 兜底（>=20% 回撤 + 跌破趋势线）也按 bear
        label_series = label_series.mask(panic_s, "bear")
        label_series = label_series.mask(((dd <= -0.20) & ((close_s < ma_s_s) | (close_s < ma_f_s))).fillna(False), "bear")

        # 样本不足就 unknown（避免 MA 未成熟导致乱判）
        label_series = label_series.mask(df.index < (min_need - 1), "unknown")
    except (AttributeError):  # noqa: BLE001
        ret_1d = None
        vol_20d = None
        dd_252 = None
        mom_63d = None
        mom_126d = None
        mom_252d = None
        mom_risk_off = False
        panic = False
        label_series = None

    bull_rules = {
        "close_above_ma_slow": bool(ma_s is not None and close > ma_s),
        "close_above_ma_fast": bool(ma_f is not None and close > ma_f),
        "ma_fast_above_ma_slow": bool(ma_f is not None and ma_s is not None and ma_f > ma_s),
        "ma_fast_rising": bool(ma_slope is not None and ma_slope > 0),
        "macd_above_signal": bool(macd is not None and macd_sig is not None and macd > macd_sig),
        "macd_above_zero": bool(macd is not None and macd > 0),
    }
    bear_rules = {
        "close_below_ma_slow": bool(ma_s is not None and close < ma_s),
        "close_below_ma_fast": bool(ma_f is not None and close < ma_f),
        "ma_fast_below_ma_slow": bool(ma_f is not None and ma_s is not None and ma_f < ma_s),
        "ma_fast_falling": bool(ma_slope is not None and ma_slope < 0),
        "macd_below_signal": bool(macd is not None and macd_sig is not None and macd < macd_sig),
        "macd_below_zero": bool(macd is not None and macd < 0),
    }

    bull_score = int(sum(1 for v in bull_rules.values() if bool(v)))
    bear_score = int(sum(1 for v in bear_rules.values() if bool(v)))

    label_raw: RegimeLabel = "neutral"
    # 经验阈值：>=4 认为有明显倾向；否则一律当震荡/中性
    # 同时加“门槛”：bull 至少得站上 MA200 且 MA50>MA200；bear 至少得跌破 MA200 且 MA50<MA200
    if bull_score >= 4 and bull_score > bear_score and bull_rules["close_above_ma_slow"] and bull_rules["ma_fast_above_ma_slow"]:
        label_raw = "bull"
    elif bear_score >= 4 and bear_score > bull_score and bear_rules["close_below_ma_slow"] and bear_rules["ma_fast_below_ma_slow"]:
        label_raw = "bear"
    else:
        label_raw = "neutral"

    # 兜底：panic/深回撤直接按 bear（不预测黑天鹅，只求更早识别“别交易了”）
    try:
        if bool(panic):
            label_raw = "bear"
        elif dd_252 is not None and float(dd_252) <= -0.20:
            # 深回撤（>=20%）且趋势不在强势区，直接判 bear
            if (ma_s is not None and close < float(ma_s)) or (ma_f is not None and close < float(ma_f)):
                label_raw = "bear"
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass

    # confirm_days：去抖。小资金最怕“震荡反复横跳”，宁愿慢半拍也别天天磨损。
    label: RegimeLabel = label_raw
    if confirm > 1 and label_series is not None and hasattr(label_series, "tail"):
        try:
            recent = [str(x) for x in list(label_series.tail(confirm)) if str(x) in {"bull", "bear", "neutral"}]
            if recent:
                # 近期出现过 panic 就直接 bear（别装死）
                if "bear" in recent and bool(panic):
                    label = "bear"
                else:
                    need = (confirm + 1) // 2
                    bull_n = int(sum(1 for x in recent if x == "bull"))
                    bear_n = int(sum(1 for x in recent if x == "bear"))
                    if bull_n >= need and bull_n > bear_n:
                        label = "bull"
                    elif bear_n >= need and bear_n > bull_n:
                        label = "bear"
                    else:
                        label = "neutral"
        except Exception:  # noqa: BLE001
            label = label_raw

    return MarketRegime(
        symbol=index_symbol,
        label=label,
        last_date=last_str,
        close=float(close),
        ma_fast=ma_f,
        ma_slow=ma_s,
        ma_fast_slope_weeks=int(slope_weeks),
        ma_fast_slope=ma_slope,
        macd=macd,
        macd_signal=macd_sig,
        macd_state=macd_state,
        ret_1d=ret_1d,
        vol_20d=vol_20d,
        drawdown_252d=dd_252,
        mom_63d=mom_63d,
        mom_126d=mom_126d,
        mom_252d=mom_252d,
        mom_risk_off=bool(mom_risk_off),
        panic=bool(panic),
        scores={"bull": bull_score, "bear": bear_score},
        rules={"bull": bull_rules, "bear": bear_rules},
    )


def compute_market_regime_weekly_series(
    *,
    index_symbol: str,
    df_daily,
    ma_fast: int = 10,
    ma_slow: int = 40,
    ma_fast_slope_weeks: int = 4,
):
    """
    计算“每周一条”的牛熊标签序列（给回测/分段统计用）。

    注意：
    - 输出是周K口径（date=该周最后一个交易日）
    - 仅做趋势/风险偏好粗判（研究用途），别拿它当圣经

    这玩意儿追求“稳健”：
    - 周线用 MA10/MA40 近似日线 MA50/MA200（别再用周线 MA50/MA200 那种 4 年级别滞后）
    - 加 drawdown 兜底：深回撤时直接判 bear（不预测黑天鹅，只求更快识别风险）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑 pip install -r \"requirements.txt\"") from exc

    from .indicators import add_macd, add_moving_averages
    from .resample import resample_to_weekly

    if df_daily is None or getattr(df_daily, "empty", True):
        raise ValueError("df_daily 为空")

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("df_daily 无有效K线")

    dfw = resample_to_weekly(df)
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if dfw.empty:
        raise ValueError("转周K后无数据")

    ma_fast2 = max(2, int(ma_fast))
    ma_slow2 = max(ma_fast2 + 1, int(ma_slow))
    slope_weeks = max(1, int(ma_fast_slope_weeks))

    dfw = add_moving_averages(dfw, ma_fast=ma_fast2, ma_slow=ma_slow2)
    dfw = add_macd(dfw, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
    ma_f = pd.to_numeric(dfw[f"ma{ma_fast2}"], errors="coerce").astype(float)
    ma_s = pd.to_numeric(dfw[f"ma{ma_slow2}"], errors="coerce").astype(float)
    macd = pd.to_numeric(dfw["macd"], errors="coerce").astype(float)
    macd_sig = pd.to_numeric(dfw["macd_signal"], errors="coerce").astype(float)

    ma_slope = (ma_f / ma_f.shift(slope_weeks)) - 1.0
    ret_1w = (close / close.shift(1).replace({0.0: float("nan")})) - 1.0
    vol_13w = ret_1w.rolling(window=13, min_periods=13).std()
    dd_52w = (close / close.rolling(window=52, min_periods=20).max().replace({0.0: float("nan")})) - 1.0

    bull_rules = {
        "close_above_ma_slow": (close > ma_s),
        "close_above_ma_fast": (close > ma_f),
        "ma_fast_above_ma_slow": (ma_f > ma_s),
        "ma_fast_rising": (ma_slope > 0),
        "macd_above_signal": (macd > macd_sig),
        "macd_above_zero": (macd > 0),
    }
    bear_rules = {
        "close_below_ma_slow": (close < ma_s),
        "close_below_ma_fast": (close < ma_f),
        "ma_fast_below_ma_slow": (ma_f < ma_s),
        "ma_fast_falling": (ma_slope < 0),
        "macd_below_signal": (macd < macd_sig),
        "macd_below_zero": (macd < 0),
    }

    bull_score = sum((x.fillna(False).astype(int)) for x in bull_rules.values())
    bear_score = sum((x.fillna(False).astype(int)) for x in bear_rules.values())

    label = pd.Series(["neutral"] * len(dfw), index=dfw.index, dtype="object")
    label = label.mask(
        (bull_score >= 4)
        & (bull_score > bear_score)
        & bull_rules["close_above_ma_slow"].fillna(False)
        & bull_rules["ma_fast_above_ma_slow"].fillna(False),
        "bull",
    )
    label = label.mask(
        (bear_score >= 4)
        & (bear_score > bull_score)
        & bear_rules["close_below_ma_slow"].fillna(False)
        & bear_rules["ma_fast_below_ma_slow"].fillna(False),
        "bear",
    )

    # 兜底：panic/深回撤按 bear（更“保命”，不追求预测）
    try:
        panic_w = (ret_1w <= -pd.concat([vol_13w * 3.0, pd.Series([0.06] * len(dfw), index=dfw.index)], axis=1).max(axis=1)).fillna(False)
        panic_w = panic_w | (dd_52w <= -0.25).fillna(False)
    except (AttributeError):  # noqa: BLE001
        panic_w = pd.Series([False] * len(dfw), index=dfw.index, dtype=bool)
    label = label.mask(panic_w, "bear")
    label = label.mask(((dd_52w <= -0.20) & ((close < ma_s) | (close < ma_f))).fillna(False), "bear")

    # 样本不足就标 unknown（别在缺 MA 的情况下瞎判牛熊）
    min_need = max(ma_slow2, ma_fast2 + slope_weeks) + 5
    label = label.mask(dfw.index < (min_need - 1), "unknown")

    out = dfw[["date", "close"]].copy()
    out["symbol"] = str(index_symbol)
    out["ma_fast"] = ma_f
    out["ma_slow"] = ma_s
    out["ma_fast_slope_weeks"] = int(slope_weeks)
    out["ma_fast_slope"] = ma_slope
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["ret_1w"] = ret_1w
    out["vol_13w"] = vol_13w
    out["drawdown_52w"] = dd_52w
    out["panic"] = (label == "bear") & panic_w.fillna(False)
    out["bull_score"] = bull_score
    out["bear_score"] = bear_score
    out["label"] = label
    return out.reset_index(drop=True)


def market_regime_to_dict(r: MarketRegime) -> dict[str, Any]:
    return {
        "symbol": r.symbol,
        "label": r.label,
        "last_date": r.last_date,
        "close": r.close,
        "ma_fast": r.ma_fast,
        "ma_slow": r.ma_slow,
        "ma_fast_slope_weeks": r.ma_fast_slope_weeks,
        "ma_fast_slope": r.ma_fast_slope,
        "macd": r.macd,
        "macd_signal": r.macd_signal,
        "macd_state": r.macd_state,
        "ret_1d": r.ret_1d,
        "vol_20d": r.vol_20d,
        "drawdown_252d": r.drawdown_252d,
        "mom_63d": r.mom_63d,
        "mom_126d": r.mom_126d,
        "mom_252d": r.mom_252d,
        "mom_risk_off": bool(r.mom_risk_off),
        "panic": bool(r.panic),
        "scores": r.scores,
        "rules": r.rules,
    }


def parse_regime_index_list(regime_index: str) -> list[str]:
    """
    解析 --regime-index：
    - off/none/0 => []
    - 单指数 => [idx]
    - 逗号分隔多指数 => [idx1, idx2, ...]（去重但保序）
    - 兼容 '+' 分隔：sh000300+sh000905（很多人会把它当“组合”，但这里语义是“多指数投票/合并”）
    - 兼容 “主指数;canary指数” 语法：例如 "sh000300,sz399006;sh000852"
      - 这里会把 ';' 当成分隔符并扁平化成一个列表（主+canary），便于复用老逻辑。
    """
    raw = str(regime_index or "").strip()
    if raw.lower() in {"", "off", "none", "0"}:
        return []
    # ';' 用来区分“主指数”和“canary指数”，但对“只要拿到一串 index 列表”的场景没必要关心分组，直接扁平化。
    raw2 = raw.replace(";", ",").replace("+", ",")
    parts = [p.strip() for p in raw2.split(",") if p.strip()]
    out: list[str] = []
    for it in parts:
        s = str(it).strip().lower()
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


def parse_regime_index_spec(regime_index: str) -> tuple[list[str], list[str]]:
    """
    解析 --regime-index 的“主指数 + canary指数”分组。

    语法：
    - "off/none/0/''" => ([], [])
    - "sh000300" => (["sh000300"], [])
    - "sh000300,sz399006" / "sh000300+sz399006" => (["sh000300","sz399006"], ["sz399006"])  # 兼容旧口径：第二个指数也当 canary
    - "sh000300,sz399006;sh000852" => (["sh000300","sz399006"], ["sh000852"])  # 显式指定 canary，只做降级，不参与主合并
    """
    raw = str(regime_index or "").strip()
    if raw.lower() in {"", "off", "none", "0"}:
        return [], []

    if ";" in raw:
        left, right = raw.split(";", 1)
        primary = parse_regime_index_list(left)
        canary0 = parse_regime_index_list(right)
        # canary 去重（并避免跟 primary 重复）
        canary: list[str] = []
        for x in canary0:
            if x in primary:
                continue
            if x not in canary:
                canary.append(x)
        return primary, canary

    idxs = parse_regime_index_list(raw)
    primary = idxs
    canary = idxs[1:] if len(idxs) > 1 else []
    return primary, canary


def compute_market_regime_payload(
    regime_index: str,
    *,
    cache_dir,
    ttl_hours: float = 6.0,
    ensemble_mode: str = "risk_first",
    canary_downgrade: bool = True,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """
    计算大盘牛熊/风险偏好（研究用途；给 scan-* / holdings-user / run 复用）。

    返回：
    - regime_dict: 可 JSON 化的 dict（或 None）
    - regime_error: 错误字符串（或 None）
    - regime_index_eff: 实际使用的指数代码；若关闭则为 None

    多指数合并（用户确认：风险优先）：
    - panic 任一触发 => bear
    - 否则任一 bear => bear
    - 否则任一 neutral/unknown => neutral
    - 全 bull 才 bull
    - canary：
      - 兼容旧口径：逗号后面的指数当 canary
      - 显式口径：用 ';' 分隔主指数和 canary，例如 "sh000300,sz399006;sh000852"
      - canary 只用 mom_risk_off=True 做降级（bull->neutral，neutral->bear），不做升级
    """
    idx_raw = str(regime_index or "").strip()
    primary_idxs, canary_idxs = parse_regime_index_spec(idx_raw)
    idxs_all = list(primary_idxs) + [x for x in canary_idxs if x not in set(primary_idxs)]
    if not idxs_all:
        return None, None, None
    if not primary_idxs:
        # 兜底：别让用户一不小心写成 ";sh000852" 就直接没 regime 了
        primary_idxs = list(idxs_all)
        canary_idxs = list(idxs_all[1:]) if len(idxs_all) > 1 else []

    # 依赖都放函数里：避免 import 链绕来绕去把 CLI 启动变慢
    from .akshare_source import FetchParams
    from .data_cache import fetch_daily_cached

    errors: list[str] = []
    parts: list[dict[str, Any]] = []
    for idx in idxs_all:
        try:
            df_idx = fetch_daily_cached(
                FetchParams(asset="index", symbol=str(idx)),
                cache_dir=cache_dir,
                ttl_hours=float(ttl_hours),
            )
            r = compute_market_regime(index_symbol=str(idx), df_daily=df_idx, ma_fast=50, ma_slow=200, ma_fast_slope_weeks=4)
            parts.append(market_regime_to_dict(r))
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            errors.append(f"{idx}: {exc}")

    if not parts:
        return None, "；".join(errors) if errors else "market_regime: all indices failed", idx_raw

    if len(parts) == 1:
        return dict(parts[0]), None, primary_idxs[0]

    mode = str(ensemble_mode or "").strip().lower() or "risk_first"
    if mode not in {"risk_first", "vote"}:
        mode = "risk_first"

    def _label(x: dict[str, Any]) -> str:
        return str(x.get("label") or "unknown").strip().lower() or "unknown"

    # 分组：主指数用于“主合并”，canary 只做降级（用户明确要求时）
    by_sym: dict[str, dict[str, Any]] = {}
    for p in parts:
        s = str(p.get("symbol") or "").strip().lower()
        if s:
            by_sym[s] = p

    primary_parts = [by_sym.get(s) for s in primary_idxs if by_sym.get(s) is not None]
    primary_parts = [p for p in primary_parts if p is not None]
    canary_parts = [by_sym.get(s) for s in canary_idxs if by_sym.get(s) is not None]
    canary_parts = [p for p in canary_parts if p is not None]
    if not primary_parts:
        # 主指数全挂了：退化成“谁活着就用谁”，别直接崩
        primary_parts = list(parts)
        canary_parts = list(parts[1:]) if len(parts) > 1 else []

    labels = [_label(p) for p in primary_parts]
    panic_any = any(bool(p.get("panic")) for p in primary_parts)

    counts = {
        "bull": int(sum(1 for x in labels if x == "bull")),
        "bear": int(sum(1 for x in labels if x == "bear")),
        "neutral": int(sum(1 for x in labels if x == "neutral")),
        "unknown": int(sum(1 for x in labels if x not in {"bull", "bear", "neutral"})),
    }

    combined_label = "neutral"
    rule_primary = None
    rule_final = None

    if mode == "vote":
        # 兼容旧口径：多数票；平票按 neutral；panic 优先
        if panic_any:
            combined_label = "bear"
            rule_primary = "panic_override"
        else:
            best = max(("bull", "bear", "neutral"), key=lambda k: counts.get(k, 0))
            top = counts.get(best, 0)
            tied = [k for k in ("bull", "bear", "neutral") if counts.get(k, 0) == top]
            if len(tied) == 1 and top > 0:
                combined_label = best
                rule_primary = "majority_vote"
            else:
                combined_label = "neutral"
                rule_primary = "tie_neutral"
        rule_final = rule_primary
    else:
        # 风险优先：取“最保守”的结论（unknown 视为 neutral）
        if panic_any:
            combined_label = "bear"
            rule_primary = "panic_override"
        else:
            lvl_map = {"bull": 2, "neutral": 1, "bear": 0, "unknown": 1}
            lvls = [lvl_map.get(x, 1) for x in labels]
            worst = min(lvls) if lvls else 1
            if worst <= 0:
                combined_label = "bear"
                rule_primary = "risk_first_any_bear"
            elif worst == 1:
                combined_label = "neutral"
                rule_primary = "risk_first_any_neutral" if any(x == "neutral" for x in labels) else "risk_first_unknown_to_neutral"
            else:
                combined_label = "bull"
                rule_primary = "risk_first_all_bull"
        rule_final = rule_primary

    # canary（只降不升）：逗号后面的指数当“风险温度计”
    canary_risk_off = any(bool(p.get("mom_risk_off")) for p in canary_parts)
    rule_canary = None
    if bool(canary_downgrade) and (not panic_any) and bool(canary_risk_off):
        if combined_label == "bull":
            combined_label = "neutral"
            rule_canary = "canary_risk_off_downgrade"
        elif combined_label == "neutral":
            combined_label = "bear"
            rule_canary = "canary_risk_off_to_bear"
    if rule_canary:
        rule_final = "{}+{}".format(str(rule_primary or ""), str(rule_canary))

    # 聚合字段：只为“少拍脑袋更保守”
    def _min_float(key: str, *, ps: list[dict[str, Any]]) -> float | None:
        vals: list[float] = []
        for p in ps:
            v = p.get(key)
            try:
                x = None if v is None else float(v)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                x = None
            if x is None:
                continue
            try:
                import math

                if not math.isfinite(x):
                    continue
            except (AttributeError):  # noqa: BLE001
                pass
            vals.append(float(x))
        return min(vals) if vals else None

    def _max_float(key: str, *, ps: list[dict[str, Any]]) -> float | None:
        vals: list[float] = []
        for p in ps:
            v = p.get(key)
            try:
                x = None if v is None else float(v)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                x = None
            if x is None:
                continue
            try:
                import math

                if not math.isfinite(x):
                    continue
            except (AttributeError):  # noqa: BLE001
                pass
            vals.append(float(x))
        return max(vals) if vals else None

    # as_of：给你把时间差说明白（别被“看起来更新了”误导）
    last_dates = [str(p.get("last_date") or "").strip() for p in parts if str(p.get("last_date") or "").strip()]
    as_of_min = min(last_dates) if last_dates else None
    as_of_max = max(last_dates) if last_dates else None

    base = dict(primary_parts[0]) if primary_parts else (dict(parts[0]) if parts else {})
    base["symbol"] = idx_raw
    base["label"] = str(combined_label)
    base["panic"] = bool(panic_any)
    base["mom_risk_off"] = bool(any(bool(p.get("mom_risk_off")) for p in parts))

    # 风险优先：动量取 min；波动取 max；回撤取 min（更负）
    base["mom_63d"] = _min_float("mom_63d", ps=primary_parts) if mode == "risk_first" else base.get("mom_63d")
    base["mom_126d"] = _min_float("mom_126d", ps=primary_parts) if mode == "risk_first" else base.get("mom_126d")
    base["mom_252d"] = _min_float("mom_252d", ps=primary_parts) if mode == "risk_first" else base.get("mom_252d")
    base["vol_20d"] = _max_float("vol_20d", ps=primary_parts) if mode == "risk_first" else base.get("vol_20d")
    base["drawdown_252d"] = _min_float("drawdown_252d", ps=primary_parts) if mode == "risk_first" else base.get("drawdown_252d")

    base["ensemble"] = {
        "mode": mode,
        "rule": rule_final,
        "rule_primary": rule_primary,
        "counts": counts,
        "components": parts,
        "primary_indices": list(primary_idxs),
        "as_of_min": as_of_min,
        "as_of_max": as_of_max,
        "canary": {
            "enabled": bool(canary_downgrade),
            "risk_off": bool(canary_risk_off),
            "indices": list(canary_idxs),
            "note": "canary=逗号后的指数（兼容旧口径），或 ';' 后的指数（显式口径）；mom_risk_off=True(126d&252d动量都为负) => risk_off；只做降级不做升级",
        },
        "errors": errors,
    }

    # A) 两融杠杆温度计（只降不升）：过热/去杠杆时，把 label 往保守方向挪一档。
    # - 这属于“环境风险加权”，不是趋势预测；数据源挂了必须可降级。
    try:
        from datetime import date as _date
        from datetime import datetime as _datetime
        from pathlib import Path

        from .margin_risk import compute_market_margin_risk

        as_of_m = None
        try:
            s = str(as_of_max or as_of_min or "").strip()
            as_of_m = _datetime.strptime(s, "%Y-%m-%d").date() if s else None
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_m = None
        if as_of_m is None:
            as_of_m = _date.today()

        cache_dir2 = Path(cache_dir) if cache_dir is not None else (Path("data") / "cache" / "index")
        margin = compute_market_margin_risk(
            as_of=as_of_m,
            cache_dir=cache_dir2.parent / "margin",
            ttl_hours=float(ttl_hours),
            lookback_days=180,
        )
        base["market_margin"] = margin

        # downgrade label (risk-first)
        label_before = str(base.get("label") or "neutral").strip().lower() or "neutral"
        base["label_technical"] = label_before

        margin_rule = None
        try:
            if isinstance(margin, dict) and bool(margin.get("ok")):
                overheat = bool(margin.get("overheat"))
                deleveraging = bool(margin.get("deleveraging"))
                if overheat or deleveraging:
                    label_after = label_before
                    if label_after == "bull":
                        label_after = "neutral"
                        margin_rule = "margin_downgrade_bull_to_neutral"
                    elif label_after == "neutral":
                        label_after = "bear"
                        margin_rule = "margin_downgrade_neutral_to_bear"

                    base["label"] = label_after

                    # 规则链：别让“为什么变保守了”变成玄学
                    try:
                        rule0 = str(base.get("ensemble", {}).get("rule") or "").strip()
                        rule1 = f"{rule0}+{margin_rule}" if rule0 else str(margin_rule)
                        if isinstance(base.get("ensemble"), dict):
                            base["ensemble"]["rule"] = rule1
                    except (AttributeError):  # noqa: BLE001
                        pass

                # 透出关键字段（给 cash_signal/report 做解释）
                if isinstance(base.get("ensemble"), dict):
                    base["ensemble"]["margin"] = {
                        "enabled": True,
                        "as_of": str(as_of_m),
                        "ok": True,
                        "score01": margin.get("score01"),
                        "overheat": bool(overheat),
                        "deleveraging": bool(deleveraging),
                        "ref_date": margin.get("ref_date"),
                        "rule": margin_rule,
                    }
            else:
                if isinstance(base.get("ensemble"), dict):
                    base["ensemble"]["margin"] = {
                        "enabled": True,
                        "as_of": str(as_of_m),
                        "ok": False,
                        "error": (margin.get("error") if isinstance(margin, dict) else "unknown"),
                    }
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        # margin_risk 是可选“加权项”，失败就跳过，别影响主流程。
        pass
    return base, None, idx_raw
