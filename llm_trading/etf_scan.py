from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .akshare_source import DataSourceError, FetchParams, fetch_daily
from .chanlun import ChanlunError, compute_chanlun_structure
from .dow import DowError, compute_dow_structure
from .indicators import (
    add_accumulation_distribution_line,
    add_adx,
    add_atr,
    add_donchian_channels,
    add_ichimoku,
    add_macd,
    add_moving_averages,
    add_rsi,
)
from .resample import resample_to_weekly
from .vsa import compute_vsa_report


@dataclass(frozen=True)
class EtfUniverseItem:
    symbol: str
    name: str
    amount: float | None = None
    volume: float | None = None
    last_price: float | None = None
    pct_chg: float | None = None
    fund_type: str | None = None


def _require_akshare():
    try:
        import akshare  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "没装 akshare？先跑：pip install -r \"requirements.txt\""
        ) from exc


def _is_standard_etf_code(code: str) -> bool:
    c = str(code).strip()
    if len(c) != 6 or (not c.isdigit()):
        return False
    # 深市 ETF 常见 15xxxx/159xxx；16xxxx 通常是 LOF
    if c.startswith("15"):
        return True
    # 沪市 ETF 常见 51/52/53/56/58/59；501/502 多是 LOF/场内基金
    if c.startswith("5") and c[1] in {"1", "2", "3", "6", "8", "9"}:
        return True
    return False


def _code_to_prefixed_symbol(code: str) -> str:
    c = str(code).strip().zfill(6)
    if c.startswith("5"):
        return f"sh{c}"
    return f"sz{c}"


def _is_etf_symbol(sym: str) -> bool:
    s = str(sym).strip().lower()
    if s.startswith(("sh", "sz")):
        s = s[2:]
    return _is_standard_etf_code(s)


def load_etf_universe(*, include_all_funds: bool = False) -> list[EtfUniverseItem]:
    """
    使用 Sina 的 ETF/场内基金列表（包含部分 LOF/封基之类，别太较真，先能跑起来）。
    """
    _require_akshare()
    import akshare as ak

    df = ak.fund_etf_fund_daily_em()
    if df is None or getattr(df, "empty", True):
        return []

    def num(x):
        try:
            return None if x is None else float(x)
        except Exception:  # noqa: BLE001
            return None

    items: list[EtfUniverseItem] = []
    for _, row in df.iterrows():
        code = str(row.get("基金代码", "")).strip()
        name = str(row.get("基金简称", "")).strip()
        ftype = str(row.get("类型", "")).strip() or None
        if not code or not name:
            continue
        if (not include_all_funds) and (not _is_standard_etf_code(code)):
            continue

        # 默认只扫股票/海外股票，固收那玩意儿波段没意思（除非你就爱磨）
        if (not include_all_funds) and ftype and ("股票" not in ftype):
            continue

        sym = _code_to_prefixed_symbol(code)
        items.append(
            EtfUniverseItem(
                symbol=sym,
                name=name,
                amount=None,
                volume=None,
                last_price=num(row.get("市价")),
                pct_chg=num(str(row.get("增长率", "")).replace("%", "").strip()),
                fund_type=ftype,
            )
        )

    # 默认按代码排序（EM 列表本身没成交额；后续在结果里给你算成交量/估算成交额）
    items.sort(key=lambda x: x.symbol)
    return items


def _ensure_ohlc(df):
    df_local = df
    if "high" not in df_local.columns or "low" not in df_local.columns:
        df_local = df_local.copy()
        df_local["open"] = df_local.get("open", df_local["close"])
        df_local["high"] = df_local["close"]
        df_local["low"] = df_local["close"]
    if "open" not in df_local.columns:
        df_local = df_local.copy()
        df_local["open"] = df_local["close"]
    if "volume" not in df_local.columns:
        df_local = df_local.copy()
        df_local["volume"] = 0.0
    return df_local


def _compute_ichimoku_state(df_local):
    last = df_local.iloc[-1]

    def f(key: str):
        v = last.get(key)
        try:
            x = None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None
        # Ichimoku 的前移列末尾经常是 NaN，别把 NaN 当数字用，JS 也解析不了（会变成非法 JSON）。
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except Exception:  # noqa: BLE001
            return x

    close = f("close")
    tenkan = f("ichimoku_tenkan")
    kijun = f("ichimoku_kijun")
    # 用 raw（当前时点）算状态；span_a/span_b（shift 后）是画图用的
    span_a = f("ichimoku_span_a_raw") if "ichimoku_span_a_raw" in df_local.columns else f("ichimoku_span_a")
    span_b = f("ichimoku_span_b_raw") if "ichimoku_span_b_raw" in df_local.columns else f("ichimoku_span_b")

    cloud_top = None
    cloud_bottom = None
    position = "unknown"
    if span_a is not None and span_b is not None:
        cloud_top = float(max(span_a, span_b))
        cloud_bottom = float(min(span_a, span_b))
        if close is not None:
            if close > cloud_top:
                position = "above"
            elif close < cloud_bottom:
                position = "below"
            else:
                position = "inside"

    tk_cross = "none"
    if len(df_local) >= 2:
        prev = df_local.iloc[-2]
        try:
            prev_diff = float(prev["ichimoku_tenkan"]) - float(prev["ichimoku_kijun"])
            cur_diff = float(last["ichimoku_tenkan"]) - float(last["ichimoku_kijun"])
            if prev_diff <= 0 < cur_diff:
                tk_cross = "bullish"
            elif prev_diff >= 0 > cur_diff:
                tk_cross = "bearish"
        except Exception:  # noqa: BLE001
            tk_cross = "unknown"

    return {
        "close": close,
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": span_a,
        "span_b": span_b,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "position": position,
        "tk_cross": tk_cross,
    }


def _compute_momentum_state(df_local):
    last = df_local.iloc[-1]

    def f(key: str):
        v = last.get(key)
        try:
            x = None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except Exception:  # noqa: BLE001
            return x

    rsi = f("rsi")
    macd = f("macd")
    macd_sig = f("macd_signal")
    adx = f("adx")
    di_p = f("di_plus")
    di_m = f("di_minus")

    rsi_state = "unknown"
    if rsi is not None:
        if rsi >= 70:
            rsi_state = "overbought"
        elif rsi <= 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

    macd_state = "unknown"
    if macd is not None and macd_sig is not None:
        macd_state = "bullish" if macd > macd_sig else ("bearish" if macd < macd_sig else "neutral")

    trend_strength = "unknown"
    if adx is not None:
        trend_strength = "strong" if adx >= 25 else ("weak" if adx <= 20 else "medium")

    direction = "unknown"
    if di_p is not None and di_m is not None:
        direction = "up" if di_p > di_m else ("down" if di_p < di_m else "neutral")

    return {
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_sig,
        "adx": adx,
        "di_plus": di_p,
        "di_minus": di_m,
        "state": {
            "rsi": rsi_state,
            "macd": macd_state,
            "trend_strength": trend_strength,
            "direction": direction,
        },
    }


def _score_trend(*, close: float | None, ma200: float | None, ich_pos: str, tk_cross: str, turtle_breakout: bool, macd_state: str, adx: float | None, dow_trend: str, chan_pos: str, chan_last_dir: str | None, vsa_close_pos: float | None, vsa_spread_level: str) -> int:
    score = 0
    if ich_pos == "above":
        score += 20
    elif ich_pos == "inside":
        score += 10
    elif ich_pos == "below":
        score -= 10

    if tk_cross == "bullish":
        score += 8
    elif tk_cross == "bearish":
        score -= 8

    if turtle_breakout:
        score += 25

    if close is not None and ma200 is not None and close > ma200:
        score += 10

    if macd_state == "bullish":
        score += 12
    elif macd_state == "bearish":
        score -= 12

    if adx is not None:
        if adx >= 25:
            score += 12
        elif adx >= 20:
            score += 8
        elif adx <= 15:
            score -= 5

    if dow_trend == "up":
        score += 10
    elif dow_trend == "down":
        score -= 10

    if chan_pos == "above":
        score += 6
    elif chan_pos == "below":
        score -= 6

    if chan_last_dir == "up":
        score += 4
    elif chan_last_dir == "down":
        score -= 4

    # VSA：大波动收在下部，趋势追进去容易吃闷棍
    if vsa_spread_level == "wide" and vsa_close_pos is not None and vsa_close_pos <= 0.35:
        score -= 8

    return int(score)


def _score_swing(*, close: float | None, support: float | None, resistance: float | None, macd_state: str, rsi: float | None, ich_pos: str, dow_trend: str, bos: str, vsa_vol_level: str, vsa_close_pos: float | None) -> int:
    score = 0
    if close is None or support is None:
        return 0

    if close < support:
        score -= 25
    else:
        dist = (close - support) / max(close, 1e-9)
        if dist <= 0.03:
            score += 25
        elif dist <= 0.06:
            score += 18
        elif dist <= 0.10:
            score += 10

    if resistance is not None and close is not None:
        near_top = (resistance - close) / max(resistance, 1e-9)
        if 0 <= near_top <= 0.03:
            score -= 8

    if macd_state == "bullish":
        score += 10
    elif macd_state == "bearish":
        score -= 6

    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 8
        elif rsi < 35:
            score += 2  # 可能是“快到底”，也可能是“还没跌完”，别加太多
        elif rsi > 70:
            score -= 6

    if ich_pos in {"above", "inside"}:
        score += 5
    elif ich_pos == "below":
        score -= 5

    if dow_trend == "down":
        score -= 8

    if bos and bos != "none":
        # 结构破坏，别硬扛
        score -= 20

    if vsa_vol_level in {"low", "normal"} and vsa_close_pos is not None and vsa_close_pos >= 0.6:
        score += 10

    return int(score)


def analyze_etf_symbol(
    item: EtfUniverseItem,
    *,
    freq: str = "weekly",
    window: int = 400,
    ichimoku_params: dict[str, int] | None = None,
    turtle_entry: int = 20,
    turtle_exit: int = 10,
    turtle_atr: int = 20,
) -> dict[str, Any]:
    ichimoku_params = ichimoku_params or {"tenkan": 9, "kijun": 26, "span_b": 52, "displacement": 26}

    try:
        df = fetch_daily(FetchParams(asset="etf", symbol=item.symbol))
    except DataSourceError as exc:
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}

    # 日线 MACD（给“周线主导 + 日线择时”用）
    last_daily_date_str = None
    daily_bars = int(len(df))
    daily_macd_state = "unknown"
    daily_macd = None
    daily_macd_signal = None
    daily_amount_last = None
    daily_amount_avg20 = None
    daily_volume_last = None
    daily_volume_avg20 = None
    try:
        df_d = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        daily_bars = int(len(df_d))
        df_d = add_macd(df_d, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
        if not df_d.empty:
            last_d = df_d.iloc[-1]
            last_d_date = last_d.get("date")
            last_daily_date_str = (
                last_d_date.strftime("%Y-%m-%d") if isinstance(last_d_date, datetime) else (str(last_d_date) if last_d_date is not None else None)
            )
            try:
                daily_volume_last = float(last_d.get("volume")) if last_d.get("volume") is not None else None
            except Exception:  # noqa: BLE001
                daily_volume_last = None
            if "volume" in df_d.columns and daily_bars > 0:
                try:
                    daily_volume_avg20 = float(df_d["volume"].tail(20).astype(float).mean())
                except Exception:  # noqa: BLE001
                    daily_volume_avg20 = None

            if "amount" in df_d.columns:
                try:
                    daily_amount_last = float(last_d.get("amount")) if last_d.get("amount") is not None else None
                except Exception:  # noqa: BLE001
                    daily_amount_last = None
                if daily_bars > 0:
                    try:
                        daily_amount_avg20 = float(df_d["amount"].tail(20).astype(float).mean())
                    except Exception:  # noqa: BLE001
                        daily_amount_avg20 = None

            daily_macd = float(last_d.get("macd"))
            daily_macd_signal = float(last_d.get("macd_signal"))
            if daily_macd > daily_macd_signal:
                daily_macd_state = "bullish"
            elif daily_macd < daily_macd_signal:
                daily_macd_state = "bearish"
            else:
                daily_macd_state = "neutral"
    except Exception:  # noqa: BLE001
        daily_macd_state = "unknown"

    if freq == "weekly":
        df = resample_to_weekly(df)

    weekly_bars_total = int(len(df))

    if window and len(df) > int(window):
        df = df.tail(int(window)).reset_index(drop=True)

    weekly_bars_used = int(len(df))

    df = add_moving_averages(df, ma_fast=50, ma_slow=200)
    df = add_accumulation_distribution_line(df)
    df = _ensure_ohlc(df)

    # Ichimoku
    df = add_ichimoku(
        df,
        tenkan=int(ichimoku_params["tenkan"]),
        kijun=int(ichimoku_params["kijun"]),
        span_b=int(ichimoku_params["span_b"]),
        displacement=int(ichimoku_params["displacement"]),
    )
    ich = _compute_ichimoku_state(df)

    # Turtle
    df = add_donchian_channels(df, window=int(turtle_entry), upper_col="donchian_entry_upper", lower_col="donchian_entry_lower", shift=1)
    df = add_donchian_channels(df, window=int(turtle_exit), upper_col="donchian_exit_upper", lower_col="donchian_exit_lower", shift=1)
    df = add_atr(df, period=int(turtle_atr), out_col="atr")

    last = df.iloc[-1]

    def f(key: str):
        v = last.get(key)
        try:
            x = None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except Exception:  # noqa: BLE001
            return x

    close = f("close")
    ma50 = f("ma50")
    ma200 = f("ma200")
    ad_line = f("ad_line")
    ad_delta_20 = None
    if ad_line is not None and "ad_line" in df.columns and len(df) >= 21:
        try:
            prev = df.iloc[-21].get("ad_line")
            prev_f = None if prev is None else float(prev)
        except Exception:  # noqa: BLE001
            prev_f = None
        if prev_f is not None:
            try:
                ad_delta_20 = float(ad_line) - float(prev_f)
            except Exception:  # noqa: BLE001
                ad_delta_20 = None
    entry_u = f("donchian_entry_upper")
    entry_l = f("donchian_entry_lower")
    atr = f("atr")
    vol_last = f("volume")
    try:
        vol_avg20 = float(df["volume"].tail(20).astype(float).mean()) if "volume" in df.columns and len(df) >= 1 else None
    except Exception:  # noqa: BLE001
        vol_avg20 = None

    amount_last = f("amount") if "amount" in df.columns else None
    if amount_last is None and close is not None and vol_last is not None:
        try:
            amount_last = float(close) * float(vol_last)
        except Exception:  # noqa: BLE001
            amount_last = None
    try:
        amount_avg20 = float(df["amount"].tail(20).astype(float).mean()) if "amount" in df.columns and len(df) >= 1 else None
    except Exception:  # noqa: BLE001
        amount_avg20 = None

    turnover_est_last = None
    if close is not None and vol_last is not None:
        try:
            turnover_est_last = float(close) * float(vol_last)
        except Exception:  # noqa: BLE001
            turnover_est_last = None

    turtle_breakout = bool(close is not None and entry_u is not None and close > entry_u)
    turtle_breakdown = bool(close is not None and entry_l is not None and close < entry_l)

    # Momentum
    df_m = add_rsi(df, period=14, out_col="rsi")
    df_m = add_macd(df_m, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
    df_m = add_adx(df_m, period=14, adx_col="adx", di_plus_col="di_plus", di_minus_col="di_minus")
    mom = _compute_momentum_state(df_m)

    # Dow
    try:
        dow = compute_dow_structure(df, lookback=2, min_gap=2)
    except DowError:
        dow = {"summary": {"trend": "unknown", "bos": "none"}}

    # Chan
    try:
        chan = compute_chanlun_structure(df, min_gap=4)
        chan_sum = chan.get("summary", {}) or {}
        chan_pos = str(chan_sum.get("position_vs_last_center", "none"))
        chan_last_dir = chan_sum.get("last_stroke_direction")
    except ChanlunError:
        chan_pos = "none"
        chan_last_dir = None

    # VSA（只要最后一根特征，事件不扫，省时间）
    try:
        _, vsa = compute_vsa_report(df, vol_window=20, spread_window=20, lookback_events=0)
        vsa_sum = vsa.get("summary", {}) or {}
        vsa_vol_level = str(vsa_sum.get("vol_level", "unknown"))
        vsa_spread_level = str(vsa_sum.get("spread_level", "unknown"))
        vsa_close_pos = vsa_sum.get("close_pos")
        try:
            vsa_close_pos = None if vsa_close_pos is None else float(vsa_close_pos)
        except Exception:  # noqa: BLE001
            vsa_close_pos = None
    except Exception:  # noqa: BLE001
        vsa_vol_level = "unknown"
        vsa_spread_level = "unknown"
        vsa_close_pos = None

    dow_sum = (dow.get("summary") or {}) if isinstance(dow, dict) else {}
    dow_trend = str(dow_sum.get("trend", "unknown"))
    dow_bos = str(dow_sum.get("bos", "none"))

    mom_state = (mom.get("state") or {}) if isinstance(mom, dict) else {}
    macd_state = str(mom_state.get("macd", "unknown"))
    rsi = mom.get("rsi")
    try:
        rsi = None if rsi is None else float(rsi)
    except Exception:  # noqa: BLE001
        rsi = None
    macd = mom.get("macd")
    try:
        macd = None if macd is None else float(macd)
    except Exception:  # noqa: BLE001
        macd = None
    macd_sig = mom.get("macd_signal")
    try:
        macd_sig = None if macd_sig is None else float(macd_sig)
    except Exception:  # noqa: BLE001
        macd_sig = None
    adx = mom.get("adx")
    try:
        adx = None if adx is None else float(adx)
    except Exception:  # noqa: BLE001
        adx = None

    score_trend = _score_trend(
        close=close,
        ma200=ma200,
        ich_pos=str(ich.get("position", "unknown")),
        tk_cross=str(ich.get("tk_cross", "none")),
        turtle_breakout=turtle_breakout,
        macd_state=macd_state,
        adx=adx,
        dow_trend=dow_trend,
        chan_pos=chan_pos,
        chan_last_dir=chan_last_dir,
        vsa_close_pos=vsa_close_pos,
        vsa_spread_level=vsa_spread_level,
    )
    score_swing = _score_swing(
        close=close,
        support=entry_l,
        resistance=entry_u,
        macd_state=macd_state,
        rsi=rsi,
        ich_pos=str(ich.get("position", "unknown")),
        dow_trend=dow_trend,
        bos=dow_bos,
        vsa_vol_level=vsa_vol_level,
        vsa_close_pos=vsa_close_pos,
    )

    last_date = last.get("date")
    last_date_str = last_date.strftime("%Y-%m-%d") if isinstance(last_date, datetime) else str(last_date)

    return {
        "symbol": item.symbol,
        "name": item.name,
        "fund_type": item.fund_type,
        "last_date": last_date_str,
        "last_daily_date": last_daily_date_str,
        "close": close,
        "amount": amount_last,
        "pct_chg": item.pct_chg,
        "bars": {"daily": daily_bars, "weekly_total": weekly_bars_total, "weekly_used": weekly_bars_used},
        "liquidity": {
            "volume_last": vol_last,
            "volume_avg20": vol_avg20,
            "amount_last": amount_last,
            "amount_avg20": amount_avg20,
            "turnover_est_last": turnover_est_last,
            "daily_volume_last": daily_volume_last,
            "daily_volume_avg20": daily_volume_avg20,
            "daily_amount_last": daily_amount_last,
            "daily_amount_avg20": daily_amount_avg20,
        },
        "levels": {"support_20w": entry_l, "resistance_20w": entry_u, "atr": atr, "ma50": ma50, "ma200": ma200},
        "wyckoff": {"ad_delta_20": ad_delta_20},
        "ichimoku": {
            "position": ich.get("position"),
            "tk_cross": ich.get("tk_cross"),
            "cloud_top": ich.get("cloud_top"),
            "cloud_bottom": ich.get("cloud_bottom"),
            "tenkan": ich.get("tenkan"),
            "kijun": ich.get("kijun"),
            "span_a": ich.get("span_a"),
            "span_b": ich.get("span_b"),
        },
        "momentum": {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_sig,
            "macd_state": macd_state,
            "adx": adx,
            "direction": mom_state.get("direction"),
            "trend_strength": mom_state.get("trend_strength"),
        },
        "turtle": {"breakout": turtle_breakout, "breakdown": turtle_breakdown},
        "dow": {"trend": dow_trend, "bos": dow_bos},
        "chan": {"position_vs_last_center": chan_pos, "last_stroke_direction": chan_last_dir},
        "vsa": {"vol_level": vsa_vol_level, "spread_level": vsa_spread_level, "close_pos": vsa_close_pos},
        "daily": {"macd": daily_macd, "macd_signal": daily_macd_signal, "macd_state": daily_macd_state},
        "scores": {"trend": score_trend, "swing": score_swing},
    }
