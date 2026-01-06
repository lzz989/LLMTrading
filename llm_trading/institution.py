from __future__ import annotations

from datetime import datetime
from typing import Any


class InstitutionError(RuntimeError):
    pass


def _to_float(x: Any, *, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:  # noqa: BLE001
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _linear_slope(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for x, y in zip(xs, values, strict=True):
        dx = float(x) - x_mean
        dy = float(y) - y_mean
        num += dx * dy
        den += dx * dx
    if den == 0.0:
        return None
    return num / den


def _compute_obv(close: list[float], volume: list[float]) -> list[float]:
    if not close:
        return []
    obv = [0.0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + float(volume[i]))
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - float(volume[i]))
        else:
            obv.append(obv[-1])
    return obv


def _parse_prefixed_stock_symbol(symbol: str) -> tuple[str, str] | None:
    s = (symbol or "").strip().lower()
    if not s:
        return None
    market = ""
    code = s
    if s.startswith(("sh", "sz", "bj")):
        market = s[:2]
        code = s[2:]
    if market not in {"sh", "sz", "bj"}:
        return None
    if len(code) != 6 or (not code.isdigit()):
        return None
    return market, code


def _try_fetch_stock_fund_flow(symbol_prefixed: str) -> dict[str, Any] | None:
    parsed = _parse_prefixed_stock_symbol(symbol_prefixed)
    if not parsed:
        return None
    market, code = parsed
    try:
        import akshare as ak
    except ModuleNotFoundError:
        return None

    try:
        df = ak.stock_individual_fund_flow(stock=code, market=market)
    except Exception:  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True):
        return None

    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    df2 = df.copy()
    if "日期" in df2.columns:
        df2["日期"] = pd.to_datetime(df2["日期"], errors="coerce")
        df2 = df2.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    if df2.empty:
        return None

    def col(name: str) -> str | None:
        return name if name in df2.columns else None

    c_date = col("日期")
    c_main_amt = col("主力净流入-净额")
    c_main_pct = col("主力净流入-净占比")
    c_big_amt = col("大单净流入-净额")
    c_super_amt = col("超大单净流入-净额")
    c_close = col("收盘价")

    if not c_date or not c_main_amt:
        return None

    def sum_tail(c: str | None, n: int) -> float | None:
        if not c:
            return None
        try:
            return float(df2[c].tail(n).astype(float).sum())
        except Exception:  # noqa: BLE001
            return None

    def mean_tail(c: str | None, n: int) -> float | None:
        if not c:
            return None
        try:
            return float(df2[c].tail(n).astype(float).mean())
        except Exception:  # noqa: BLE001
            return None

    last_dt = df2.iloc[-1][c_date]
    last_date = last_dt.strftime("%Y-%m-%d") if isinstance(last_dt, datetime) else str(last_dt)

    tail_n = min(20, int(len(df2)))
    tail = df2.tail(tail_n)
    tail_rows: list[dict[str, Any]] = []
    for _, row in tail.iterrows():
        dt = row.get(c_date)
        tail_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else str(dt),
                "close": _to_float(row.get(c_close)),
                "main_net": _to_float(row.get(c_main_amt)),
                "main_pct": _to_float(row.get(c_main_pct)),
                "super_net": _to_float(row.get(c_super_amt)),
                "big_net": _to_float(row.get(c_big_amt)),
            }
        )

    return {
        "source": "eastmoney",
        "last_date": last_date,
        "main_net_5d": sum_tail(c_main_amt, 5),
        "main_net_20d": sum_tail(c_main_amt, 20),
        "main_pct_avg_5d": mean_tail(c_main_pct, 5),
        "main_pct_avg_20d": mean_tail(c_main_pct, 20),
        "super_net_5d": sum_tail(c_super_amt, 5),
        "big_net_5d": sum_tail(c_big_amt, 5),
        "tail_20d": tail_rows,
    }


def compute_institution_report(
    df,
    *,
    asset: str | None,
    symbol_prefixed: str | None,
    freq: str,
    vsa_vol_window: int = 20,
    vsa_spread_window: int = 20,
) -> dict[str, Any]:
    """
    “机构探测器”（研究用途）：
    - 不是真能看见机构账户；这里只是对 量价行为 + 资金流(可选) 做可复现的启发式评分。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise InstitutionError("没装 pandas？先装依赖：pip install -r \"requirements.txt\"") from exc

    df2 = df.copy()
    if "date" not in df2.columns or "close" not in df2.columns:
        return {"method": "institution", "summary": {"state": "unknown", "score": 0, "evidence": ["缺 date/close 列"]}}
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df2.empty:
        return {"method": "institution", "summary": {"state": "unknown", "score": 0, "evidence": ["无有效数据"]}}

    if "volume" not in df2.columns:
        df2["volume"] = 0.0
    if "open" not in df2.columns:
        df2["open"] = df2["close"]
    if "high" not in df2.columns:
        df2["high"] = df2["close"]
    if "low" not in df2.columns:
        df2["low"] = df2["close"]

    # A/D（如果上游没算，这里补一下）
    if "ad_line" not in df2.columns:
        try:
            from .indicators import add_accumulation_distribution_line
        except Exception:  # noqa: BLE001
            add_accumulation_distribution_line = None
        if add_accumulation_distribution_line is not None:
            try:
                df2 = add_accumulation_distribution_line(df2)
            except Exception:  # noqa: BLE001
                pass

    close_s = df2["close"].astype(float)
    vol_s = df2["volume"].fillna(0.0).astype(float)

    obv = _compute_obv(close_s.tolist(), vol_s.tolist())
    df2["obv"] = obv

    last = df2.iloc[-1]
    last_dt = last.get("date")
    last_date = last_dt.strftime("%Y-%m-%d") if isinstance(last_dt, datetime) else str(last_dt)

    ad_delta_20 = None
    obv_delta_20 = None
    obv_slope_20 = None
    if len(df2) >= 21:
        try:
            if "ad_line" in df2.columns:
                ad_delta_20 = _to_float(float(df2.iloc[-1]["ad_line"]) - float(df2.iloc[-21]["ad_line"]))
        except Exception:  # noqa: BLE001
            ad_delta_20 = None
        try:
            obv_delta_20 = _to_float(float(df2.iloc[-1]["obv"]) - float(df2.iloc[-21]["obv"]))
        except Exception:  # noqa: BLE001
            obv_delta_20 = None

    try:
        tail_obv = df2["obv"].tail(20).astype(float).tolist()
        obv_slope_20 = _linear_slope(tail_obv)
    except Exception:  # noqa: BLE001
        obv_slope_20 = None

    # VSA 偏置（用最后一根量价特征做一个粗糙方向锚）
    vsa_bias = 0.0
    vsa_summary: dict[str, Any] | None = None
    try:
        from .vsa import compute_vsa_report

        _df_feat, vsa = compute_vsa_report(
            df2,
            vol_window=int(vsa_vol_window),
            spread_window=int(vsa_spread_window),
            lookback_events=60,
        )
        vsa_summary = (vsa or {}).get("summary") or {}
        vol_level = str(vsa_summary.get("vol_level") or "")
        spread_level = str(vsa_summary.get("spread_level") or "")
        close_pos = _to_float(vsa_summary.get("close_pos"), default=None)

        op = _to_float(last.get("open"))
        cl = _to_float(last.get("close"))
        is_up = bool(op is not None and cl is not None and cl >= op)

        if vol_level in {"high", "very_high"} and spread_level == "wide" and (not is_up) and (close_pos is not None and close_pos >= 0.6):
            vsa_bias += 1.0
        if vol_level == "low" and spread_level == "narrow" and (not is_up) and (close_pos is not None and close_pos >= 0.3):
            vsa_bias += 0.6
        if vol_level == "very_high" and spread_level == "wide" and is_up and (close_pos is not None and close_pos >= 0.65):
            vsa_bias -= 0.8
        if vol_level in {"high", "very_high"} and spread_level == "wide" and (close_pos is not None and close_pos <= 0.25):
            vsa_bias -= 1.0
    except Exception:  # noqa: BLE001
        vsa_summary = None

    fund_flow = None
    if (asset or "").lower() == "stock" and symbol_prefixed:
        fund_flow = _try_fetch_stock_fund_flow(symbol_prefixed)

    evidence: list[str] = []
    score = 50.0

    if ad_delta_20 is not None:
        if ad_delta_20 > 0:
            score += 10.0
            evidence.append("A/D 20期上行")
        elif ad_delta_20 < 0:
            score -= 10.0
            evidence.append("A/D 20期下行")

    if obv_delta_20 is not None:
        if obv_delta_20 > 0:
            score += 8.0
            evidence.append("OBV 20期上行")
        elif obv_delta_20 < 0:
            score -= 8.0
            evidence.append("OBV 20期下行")

    if obv_slope_20 is not None:
        if obv_slope_20 > 0:
            score += 4.0
            evidence.append("OBV 近20期斜率>0")
        elif obv_slope_20 < 0:
            score -= 4.0
            evidence.append("OBV 近20期斜率<0")

    if vsa_summary is not None:
        if vsa_bias > 0.2:
            score += 6.0
            evidence.append("VSA 偏吸筹")
        elif vsa_bias < -0.2:
            score -= 6.0
            evidence.append("VSA 偏派发")

    if fund_flow is not None:
        main_5 = _to_float(fund_flow.get("main_net_5d"))
        main_20 = _to_float(fund_flow.get("main_net_20d"))
        pct_5 = _to_float(fund_flow.get("main_pct_avg_5d"))

        if main_5 is not None:
            if main_5 > 0:
                score += 10.0
                evidence.append("主力净流入(5D)>0")
            elif main_5 < 0:
                score -= 10.0
                evidence.append("主力净流入(5D)<0")

        if main_20 is not None:
            if main_20 > 0:
                score += 6.0
                evidence.append("主力净流入(20D)>0")
            elif main_20 < 0:
                score -= 6.0
                evidence.append("主力净流入(20D)<0")

        if pct_5 is not None:
            if pct_5 > 0:
                score += 4.0
                evidence.append("主力净占比(5D)>0")
            elif pct_5 < 0:
                score -= 4.0
                evidence.append("主力净占比(5D)<0")

    score = _clamp(score, 0.0, 100.0)
    if score >= 65.0:
        state = "accumulation"
    elif score <= 35.0:
        state = "distribution"
    else:
        state = "neutral"

    confidence = abs(score - 50.0) / 50.0
    confidence = _clamp(confidence, 0.0, 1.0)

    return {
        "method": "institution",
        "generated_at": datetime.now().isoformat(),
        "asset": asset,
        "symbol": symbol_prefixed,
        "freq": freq,
        "params": {"vsa_vol_window": int(vsa_vol_window), "vsa_spread_window": int(vsa_spread_window)},
        "summary": {
            "state": state,
            "score": round(float(score), 1),
            "confidence": round(float(confidence), 3),
            "evidence": evidence,
            "last_date": last_date,
        },
        "price_volume": {
            "ad_delta_20": ad_delta_20,
            "obv_delta_20": obv_delta_20,
            "obv_slope_20": obv_slope_20,
            "vsa_bias": vsa_bias,
            "vsa_summary": vsa_summary,
        },
        "fund_flow": fund_flow,
        "disclaimer": "研究用途：这是量价+公开资金流的启发式推断，不代表真实机构账户行为，更不构成投资建议。",
    }

