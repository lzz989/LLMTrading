from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .akshare_source import DataSourceError, FetchParams, fetch_daily
from .chanlun import ChanlunError, compute_chanlun_structure
from .data_cache import fetch_daily_cached
from .dow import DowError, compute_dow_structure
from .indicators import (
    add_accumulation_distribution_line,
    add_adx,
    add_atr,
    add_bollinger_bands,
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
    # 优先在线抓列表；但网络/源站抽风时要能降级（否则 scan-etf 直接崩）。
    df = None
    try:
        _require_akshare()
        import akshare as ak

        df = ak.fund_etf_fund_daily_em()
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        df = None

    # 离线兜底：读本地缓存（data/cache/universe/etf_names.json）
    if df is None or getattr(df, "empty", True):
        try:
            import json

            cache_p = Path("data") / "cache" / "universe" / "etf_names.json"
            obj = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else None
            items_map = obj.get("items") if isinstance(obj, dict) else None
            if not isinstance(items_map, dict) or (not items_map):
                return []

            def _maybe_keep(name0: str, code0: str) -> bool:
                if include_all_funds:
                    return True
                # 离线时没有 fund_type，只能做保守的“股票ETF优先”过滤。
                # 规则：排除明显的固收/货币/商品类，避免 BBB 被这些噪音塞满。
                n = str(name0 or "")
                c = str(code0 or "").strip()
                if c.startswith("511"):  # 常见货币/债券ETF
                    return False
                bad_tokens = ("债", "货币", "现金", "短债", "中债", "国债", "信用", "利率", "同业存单", "黄金", "原油", "商品")
                return not any(t in n for t in bad_tokens)

            items: list[EtfUniverseItem] = []
            for sym, name in items_map.items():
                s = str(sym or "").strip().lower()
                if not s:
                    continue
                code = s[2:] if s.startswith(("sh", "sz")) else s
                if (not include_all_funds) and (not _is_standard_etf_code(code)):
                    continue
                if not _maybe_keep(str(name), code):
                    continue
                items.append(EtfUniverseItem(symbol=s if s.startswith(("sh", "sz")) else _code_to_prefixed_symbol(code), name=str(name)))

            items.sort(key=lambda x: x.symbol)
            return items
        except (OSError, TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
            return []

    def num(x):
        try:
            return None if x is None else float(x)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
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
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            return None
        # Ichimoku 的前移列末尾经常是 NaN，别把 NaN 当数字用，JS 也解析不了（会变成非法 JSON）。
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except (AttributeError):  # noqa: BLE001
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
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
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
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            return None
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except (AttributeError):  # noqa: BLE001
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
    bbb_params=None,
    bbb_horizons: list[int] | None = None,
    bbb_rank_horizon: int = 8,
    bbb_score_mode: str = "win_rate",
    bbb_buy_cost: float = 0.0,
    bbb_sell_cost: float = 0.0,
    bbb_slippage_mode: str = "none",
    bbb_slippage_bps: float = 0.0,
    bbb_slippage_ref_amount_yuan: float = 1e8,
    bbb_slippage_bps_min: float = 0.0,
    bbb_slippage_bps_max: float = 30.0,
    bbb_slippage_unknown_bps: float = 10.0,
    bbb_slippage_vol_mult: float = 0.0,
    bbb_non_overlapping: bool = True,
    bbb_exit_min_hold_days: int = 5,
    bbb_exit_cooldown_days: int = 0,
    bbb_exit_trail_ma: int = 20,
    bbb_exit_enable_trail: bool = True,
    bbb_exit_stop_loss_ret: float = 0.0,
    bbb_exit_profit_stop_enabled: bool = True,
    bbb_exit_profit_stop_min_profit_ret: float = 0.20,
    bbb_exit_profit_stop_dd_pct: float = 0.12,
    bbb_exit_panic_enabled: bool = True,
    bbb_exit_panic_vol_mult: float = 3.0,
    bbb_exit_panic_min_drop: float = 0.04,
    bbb_exit_panic_drawdown_252d: float = 0.25,
    include_bbb_samples: bool = False,
    cache_dir: Path | None = None,
    cache_ttl_hours: float = 24.0,
    analysis_cache: bool = True,
    analysis_cache_dir: Path | None = None,
    rs_index_symbol: str | None = None,
    rs_index_weekly=None,
) -> dict[str, Any]:
    ichimoku_params = ichimoku_params or {"tenkan": 9, "kijun": 26, "span_b": 52, "displacement": 26}

    try:
        df = fetch_daily_cached(
            FetchParams(asset="etf", symbol=item.symbol),
            cache_dir=cache_dir or (Path("data") / "cache" / "etf"),
            ttl_hours=float(cache_ttl_hours),
        )
    except DataSourceError as exc:
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}

    # 派生结果缓存（加速重复扫描）：key = symbol + last_date + params_hash
    last_daily_date_str0 = None
    try:
        if df is not None and (not getattr(df, "empty", True)) and "date" in df.columns:
            dt0 = df["date"].iloc[-1]
            last_daily_date_str0 = dt0.strftime("%Y-%m-%d") if isinstance(dt0, datetime) else str(dt0)
    except (KeyError, IndexError, AttributeError):  # noqa: BLE001
        last_daily_date_str0 = None

    cache_hit = False
    if bool(analysis_cache) and analysis_cache_dir is not None and last_daily_date_str0:
        try:
            from .analysis_cache import ANALYSIS_CACHE_VERSION, cache_path, compute_params_hash, read_cached_json
            from . import __version__ as _ver

            params_hash = compute_params_hash(
                {
                    "v": int(ANALYSIS_CACHE_VERSION),
                    "pkg": str(_ver),
                    "symbol": str(item.symbol),
                    "freq": str(freq),
                    "window": int(window),
                    "ichimoku_params": dict(ichimoku_params or {}),
                    "turtle_entry": int(turtle_entry),
                    "turtle_exit": int(turtle_exit),
                    "turtle_atr": int(turtle_atr),
                    "bbb": {
                        "params": bbb_params,
                        "horizons": list(bbb_horizons or []),
                        "rank_horizon": int(bbb_rank_horizon),
                        "score_mode": str(bbb_score_mode),
                        "buy_cost": float(bbb_buy_cost),
                        "sell_cost": float(bbb_sell_cost),
                        "slippage_mode": str(bbb_slippage_mode),
                        "slippage_bps": float(bbb_slippage_bps),
                        "slippage_ref_amount_yuan": float(bbb_slippage_ref_amount_yuan),
                        "slippage_bps_min": float(bbb_slippage_bps_min),
                        "slippage_bps_max": float(bbb_slippage_bps_max),
                        "slippage_unknown_bps": float(bbb_slippage_unknown_bps),
                        "slippage_vol_mult": float(bbb_slippage_vol_mult),
                        "non_overlapping": bool(bbb_non_overlapping),
                        "exit_min_hold_days": int(bbb_exit_min_hold_days),
                        "exit_cooldown_days": int(bbb_exit_cooldown_days),
                        "exit_trail_ma": int(bbb_exit_trail_ma),
                        "exit_enable_trail": bool(bbb_exit_enable_trail),
                        "exit_stop_loss_ret": float(bbb_exit_stop_loss_ret),
                        "exit_profit_stop_enabled": bool(bbb_exit_profit_stop_enabled),
                        "exit_profit_stop_min_profit_ret": float(bbb_exit_profit_stop_min_profit_ret),
                        "exit_profit_stop_dd_pct": float(bbb_exit_profit_stop_dd_pct),
                        "exit_panic_enabled": bool(bbb_exit_panic_enabled),
                        "exit_panic_vol_mult": float(bbb_exit_panic_vol_mult),
                        "exit_panic_min_drop": float(bbb_exit_panic_min_drop),
                        "exit_panic_drawdown_252d": float(bbb_exit_panic_drawdown_252d),
                        "include_samples": bool(include_bbb_samples),
                    },
                }
            )
            p = cache_path(cache_dir=Path(analysis_cache_dir), symbol=item.symbol, last_date=str(last_daily_date_str0), params_hash=str(params_hash))
            cached = read_cached_json(p)
            if cached is not None and cached.get("symbol") == item.symbol and cached.get("last_daily_date") == last_daily_date_str0:
                # 别让缓存把“实时列表字段”冻住（不影响策略核心）
                cached["name"] = item.name
                cached["pct_chg"] = item.pct_chg
                cached["_analysis_cache"] = {"hit": True, "path": str(p)}
                return cached
        except (AttributeError):  # noqa: BLE001
            pass

    # 日线 MACD（给“周线主导 + 日线择时”用）
    last_daily_date_str = None
    daily_bars = int(len(df))
    daily_macd_state = "unknown"
    daily_macd = None
    daily_macd_signal = None
    daily_close_last = None
    daily_ma20_last = None
    daily_macd_bearish_2d = False
    daily_close_below_ma20_2d = False
    daily_amount_last = None
    daily_amount_avg20 = None
    daily_volume_last = None
    daily_volume_avg20 = None
    try:
        df_d = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        daily_bars = int(len(df_d))
        # 日线 MA20：用来做“软风控”的位置参考（min_periods=20，别硬算假的均线）
        df_d["ma20"] = df_d["close"].astype(float).rolling(window=20, min_periods=20).mean()
        df_d = add_macd(df_d, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")
        if not df_d.empty:
            last_d = df_d.iloc[-1]
            last_d_date = last_d.get("date")
            last_daily_date_str = (
                last_d_date.strftime("%Y-%m-%d") if isinstance(last_d_date, datetime) else (str(last_d_date) if last_d_date is not None else None)
            )
            try:
                daily_close_last = float(last_d.get("close")) if last_d.get("close") is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                daily_close_last = None
            try:
                v = last_d.get("ma20")
                daily_ma20_last = None if v is None else float(v)
                if daily_ma20_last is not None:
                    import math

                    if math.isnan(daily_ma20_last) or math.isinf(daily_ma20_last):
                        daily_ma20_last = None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                daily_ma20_last = None
            try:
                daily_volume_last = float(last_d.get("volume")) if last_d.get("volume") is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                daily_volume_last = None
            if "volume" in df_d.columns and daily_bars > 0:
                try:
                    daily_volume_avg20 = float(df_d["volume"].tail(20).astype(float).mean())
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    daily_volume_avg20 = None

            if "amount" in df_d.columns:
                try:
                    daily_amount_last = float(last_d.get("amount")) if last_d.get("amount") is not None else None
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    daily_amount_last = None
                if daily_bars > 0:
                    try:
                        daily_amount_avg20 = float(df_d["amount"].tail(20).astype(float).mean())
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        daily_amount_avg20 = None

            daily_macd = float(last_d.get("macd"))
            daily_macd_signal = float(last_d.get("macd_signal"))
            if daily_macd > daily_macd_signal:
                daily_macd_state = "bullish"
            elif daily_macd < daily_macd_signal:
                daily_macd_state = "bearish"
            else:
                daily_macd_state = "neutral"

            # 2日确认：避免一天假死叉/假跌破就把人磨得稀碎
            if len(df_d) >= 2:
                prev_d = df_d.iloc[-2]
                try:
                    prev_macd = float(prev_d.get("macd"))
                    prev_sig = float(prev_d.get("macd_signal"))
                    daily_macd_bearish_2d = bool(prev_macd < prev_sig and daily_macd < daily_macd_signal)
                except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                    daily_macd_bearish_2d = False

                try:
                    prev_close = float(prev_d.get("close")) if prev_d.get("close") is not None else None
                    prev_ma20 = float(prev_d.get("ma20")) if prev_d.get("ma20") is not None else None
                    cur_close = daily_close_last
                    cur_ma20 = daily_ma20_last
                    daily_close_below_ma20_2d = bool(
                        prev_close is not None and prev_ma20 is not None and cur_close is not None and cur_ma20 is not None and prev_close < prev_ma20 and cur_close < cur_ma20
                    )
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    daily_close_below_ma20_2d = False
    except Exception:  # noqa: BLE001
        daily_macd_state = "unknown"

    # -------- 7因子面板（ETF 版本；研究用途）--------
    # 注意：这里只算“当前时点（最后一根日线收盘可得）”的值；paper-sim 回测里会用“上一交易日可得”的版本避免未来函数。
    factor_panel_7: dict[str, Any] = {"ok": False, "as_of": last_daily_date_str}
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None  # type: ignore[assignment]

    df_d2 = None
    try:
        if "df_d" in locals() and df_d is not None and (not getattr(df_d, "empty", True)):
            df_d2 = df_d.copy()
    except (AttributeError):  # noqa: BLE001
        df_d2 = None

    # 日线派生：波动/ATR%/回撤/BOLL/量能比
    vol_20d = None
    atr14_pct = None
    dd_252d = None
    from_low_252d = None
    boll_bw = None
    boll_bw_rel = None
    boll_squeeze = None
    amount_ratio = None
    volume_ratio = None
    try:
        if df_d2 is not None and (not getattr(df_d2, "empty", True)):
            # 统一排序 + 数字化
            if pd is not None:
                df_d2["date"] = pd.to_datetime(df_d2["date"], errors="coerce")
                df_d2 = df_d2.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            close_s = df_d2["close"].astype(float)

            # 20D 波动率（收益标准差）
            r1 = (close_s / close_s.shift(1).replace({0.0: float("nan")})) - 1.0
            v20 = r1.rolling(window=20, min_periods=20).std()
            vol_20d = float(v20.iloc[-1]) if len(v20) > 0 else None

            # ATR%（日线 ATR14 / close）
            try:
                df_atr = add_atr(df_d2, period=14, out_col="atr14")
                atr14 = df_atr.get("atr14")
                atr14 = None if atr14 is None else float(atr14.iloc[-1])
                close_last = float(close_s.iloc[-1]) if len(close_s) > 0 else None
                if atr14 is not None and close_last is not None and close_last > 0:
                    atr14_pct = float(atr14) / float(close_last)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                atr14_pct = None

            # 252D 回撤 / 距52周低点（位置）
            roll_max = close_s.rolling(window=252, min_periods=20).max()
            roll_min = close_s.rolling(window=252, min_periods=20).min()
            dd = (close_s / roll_max.replace({0.0: float("nan")})) - 1.0
            up_from_low = (close_s / roll_min.replace({0.0: float("nan")})) - 1.0
            dd_252d = float(dd.iloc[-1]) if len(dd) > 0 else None
            from_low_252d = float(up_from_low.iloc[-1]) if len(up_from_low) > 0 else None

            # BOLL 带宽 / squeeze（用带宽相对自身252D中位数，避免未来函数/避免算分位太慢）
            try:
                df_bb = add_bollinger_bands(df_d2, window=20, k=2.0, bandwidth_col="boll_bw")
                bw = df_bb.get("boll_bw")
                if bw is not None:
                    bw_s = pd.to_numeric(bw, errors="coerce").astype(float) if pd is not None else bw.astype(float)
                    bw_last = float(bw_s.iloc[-1]) if len(bw_s) > 0 else None
                    boll_bw = bw_last
                    bw_med = bw_s.rolling(window=252, min_periods=60).median()
                    bw_med_last = float(bw_med.iloc[-1]) if len(bw_med) > 0 else None
                    if bw_last is not None and bw_med_last is not None and bw_med_last > 0:
                        boll_bw_rel = float(bw_last) / float(bw_med_last)
                        boll_squeeze = bool(boll_bw_rel <= 0.80)  # 经验阈值：越小越“挤”
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                boll_bw = None
                boll_bw_rel = None
                boll_squeeze = None

            # 量能确认：相对20日均（最后一根日线收盘可得）
            try:
                if daily_amount_last is not None and daily_amount_avg20 is not None and float(daily_amount_avg20) > 0:
                    amount_ratio = float(daily_amount_last) / float(daily_amount_avg20)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                amount_ratio = None
            try:
                if daily_volume_last is not None and daily_volume_avg20 is not None and float(daily_volume_avg20) > 0:
                    volume_ratio = float(daily_volume_last) / float(daily_volume_avg20)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                volume_ratio = None
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        pass

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
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            return None
        try:
            import math

            return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
        except (AttributeError):  # noqa: BLE001
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
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            prev_f = None
        if prev_f is not None:
            try:
                ad_delta_20 = float(ad_line) - float(prev_f)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                ad_delta_20 = None
    entry_u = f("donchian_entry_upper")
    entry_l = f("donchian_entry_lower")
    atr = f("atr")
    vol_last = f("volume")
    try:
        vol_avg20 = float(df["volume"].tail(20).astype(float).mean()) if "volume" in df.columns and len(df) >= 1 else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        vol_avg20 = None

    amount_last = f("amount") if "amount" in df.columns else None
    if amount_last is None and close is not None and vol_last is not None:
        try:
            amount_last = float(close) * float(vol_last)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            amount_last = None
    try:
        amount_avg20 = float(df["amount"].tail(20).astype(float).mean()) if "amount" in df.columns and len(df) >= 1 else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        amount_avg20 = None

    turnover_est_last = None
    if close is not None and vol_last is not None:
        try:
            turnover_est_last = float(close) * float(vol_last)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
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
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            vsa_close_pos = None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
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
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        rsi = None
    macd = mom.get("macd")
    try:
        macd = None if macd is None else float(macd)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        macd = None
    macd_sig = mom.get("macd_signal")
    try:
        macd_sig = None if macd_sig is None else float(macd_sig)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        macd_sig = None
    adx = mom.get("adx")
    try:
        adx = None if adx is None else float(adx)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        adx = None

    # 周线动量/相对强弱（12W/26W）
    mom_12w = None
    mom_26w = None
    rs_12w = None
    rs_26w = None
    try:
        if pd is not None:
            w = df[["date", "close"]].copy()
            w["date"] = pd.to_datetime(w["date"], errors="coerce")
            w = w.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if not w.empty:
                close_w = pd.to_numeric(w["close"], errors="coerce").astype(float)
                mom_12w_s = (close_w / close_w.shift(12).replace({0.0: float("nan")})) - 1.0
                mom_26w_s = (close_w / close_w.shift(26).replace({0.0: float("nan")})) - 1.0
                mom_12w = float(mom_12w_s.iloc[-1]) if len(mom_12w_s) > 0 else None
                mom_26w = float(mom_26w_s.iloc[-1]) if len(mom_26w_s) > 0 else None

                if rs_index_weekly is not None and (not getattr(rs_index_weekly, "empty", True)):
                    wi = rs_index_weekly[["date", "close"]].copy()
                    wi["date"] = pd.to_datetime(wi["date"], errors="coerce")
                    wi = wi.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    if not wi.empty:
                        close_i = pd.to_numeric(wi["close"], errors="coerce").astype(float)
                        # 对齐：用 <= 标的周末的最近一根指数周K
                        aligned = pd.merge_asof(
                            w[["date"]],
                            wi[["date", "close"]].rename(columns={"close": "idx_close"}),
                            on="date",
                            direction="backward",
                        )
                        idx_close = pd.to_numeric(aligned["idx_close"], errors="coerce").astype(float)
                        idx_mom_12w = (idx_close / idx_close.shift(12).replace({0.0: float("nan")})) - 1.0
                        idx_mom_26w = (idx_close / idx_close.shift(26).replace({0.0: float("nan")})) - 1.0
                        rs_12w_s = (mom_12w_s - idx_mom_12w)
                        rs_26w_s = (mom_26w_s - idx_mom_26w)
                        rs_12w = float(rs_12w_s.iloc[-1]) if len(rs_12w_s) > 0 else None
                        rs_26w = float(rs_26w_s.iloc[-1]) if len(rs_26w_s) > 0 else None
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        mom_12w = None
        mom_26w = None
        rs_12w = None
        rs_26w = None

    # 组装 7 因子面板（当前时点）
    try:
        import math

        def _fnum(x):
            try:
                v = None if x is None else float(x)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                return None
            return None if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else float(v)

        factor_panel_7 = {
            "ok": True,
            "as_of": last_daily_date_str or last_date_str,
            "rs": {"index": (str(rs_index_symbol) if rs_index_symbol else None), "rs_12w": _fnum(rs_12w), "rs_26w": _fnum(rs_26w)},
            "mom": {"mom_12w": _fnum(mom_12w), "mom_26w": _fnum(mom_26w)},
            "trend": {"adx14": _fnum(adx)},
            "vol": {"vol_20d": _fnum(vol_20d), "atr14_pct": _fnum(atr14_pct)},
            "drawdown": {"dd_252d": _fnum(dd_252d), "from_low_252d": _fnum(from_low_252d)},
            "liquidity": {
                "amount_avg20": _fnum(daily_amount_avg20 if daily_amount_avg20 is not None else amount_avg20),
                "amount_ratio": _fnum(amount_ratio),
                "volume_ratio": _fnum(volume_ratio),
            },
            "boll": {"bandwidth": _fnum(boll_bw), "bandwidth_rel": _fnum(boll_bw_rel), "squeeze": (bool(boll_squeeze) if boll_squeeze is not None else None)},
            "note": "scan-etf面板口径=最后一根日线收盘可得；paper-sim回测口径会用上一交易日可得的版本避免未来函数。",
        }
    except Exception:  # noqa: BLE001
        factor_panel_7 = {"ok": False, "as_of": last_daily_date_str or last_date_str}

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

    # 退出框架（给“已经持仓的人”用）：周线管生死，日线管风控（但别被噪声磨死）
    weekly_below_ma50 = False
    weekly_below_ma50_confirm2 = False
    if close is not None and ma50 is not None and close > 0 and ma50 > 0:
        weekly_below_ma50 = bool(close < ma50)
        if len(df) >= 2 and "ma50" in df.columns:
            try:
                prev = df.iloc[-2]
                prev_close = prev.get("close")
                prev_ma50 = prev.get("ma50")
                prev_close_f = None if prev_close is None else float(prev_close)
                prev_ma50_f = None if prev_ma50 is None else float(prev_ma50)
                weekly_below_ma50_confirm2 = bool(
                    weekly_below_ma50 and prev_close_f is not None and prev_ma50_f is not None and prev_ma50_f > 0 and prev_close_f < prev_ma50_f
                )
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                weekly_below_ma50_confirm2 = False

    daily_below_ma20 = bool(daily_close_last is not None and daily_ma20_last is not None and daily_close_last < daily_ma20_last)
    daily_bearish = bool(daily_macd is not None and daily_macd_signal is not None and daily_macd < daily_macd_signal)

    exit_suggestion = "hold"
    if weekly_below_ma50_confirm2:
        exit_suggestion = "exit"
    elif daily_macd_bearish_2d and daily_below_ma20:
        exit_suggestion = "reduce"

    exit_info = {
        "suggestion": exit_suggestion,
        "weekly": {
            "below_ma50": bool(weekly_below_ma50),
            "below_ma50_confirm2": bool(weekly_below_ma50_confirm2),
        },
        "daily": {
            "close": daily_close_last,
            "ma20": daily_ma20_last,
            "below_ma20": bool(daily_below_ma20),
            "below_ma20_confirm2": bool(daily_close_below_ma20_2d),
            "bearish": bool(daily_bearish),
            "bearish_confirm2": bool(daily_macd_bearish_2d),
        },
    }

    # BBB：周线定方向 + 位置 + 日线MACD择时（可选：胜率优先 / 年化优先）
    bbb = {"ok": False, "fails": [], "why": "", "score": None, "rank_horizon": None}
    bbb_forward: dict[str, Any] = {}
    bbb_best: dict[str, Any] | None = None
    bbb_exit_bt: dict[str, Any] | None = None
    bbb_cost: dict[str, Any] | None = None
    bbb_entry_ma: int | None = None
    bbb_ma_entry: float | None = None
    bbb_dist_entry: float | None = None
    try:
        from .bbb import BBBExitParams, BBBParams, bbb_exit_backtest, compute_bbb_entry_signal
        from .backtest import forward_holding_backtest, score_forward_stats, shrunk_win_rate
        from .costs import bps_to_rate, estimate_slippage_bps

        params = bbb_params if isinstance(bbb_params, BBBParams) else BBBParams()
        bbb_score_mode2 = str(bbb_score_mode or "win_rate").strip().lower()
        # 用户不传就用默认；传了就去重/排序
        horizons2 = bbb_horizons or [4, 8, 12]
        horizons2 = sorted({int(x) for x in horizons2 if int(x) > 0})
        if not horizons2:
            horizons2 = [8]

        # 先用“当前时点”给出明确的 ok/fails（别让人看一堆指标还不知道到底能不能买）
        fails: list[str] = []
        if weekly_bars_total < int(params.min_weekly_bars_total):
            fails.append("周K不足")

        entry_ma_n = max(2, int(getattr(params, "entry_ma", 50) or 50))
        bbb_entry_ma = int(entry_ma_n)
        ma_entry_val = None
        try:
            col = f"ma{entry_ma_n}"
            if col not in df.columns:
                # 别用 min_periods=1 去“硬算”均线，新ETF会冒出假的 MA，误导你以为有效。
                df = df.copy()
                df[col] = df["close"].rolling(window=int(entry_ma_n), min_periods=int(entry_ma_n)).mean()
            v = df.iloc[-1].get(col)
            ma_entry_val = None if v is None else float(v)
            import math

            if ma_entry_val is not None and (math.isnan(ma_entry_val) or math.isinf(ma_entry_val)):
                ma_entry_val = None
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            ma_entry_val = None
        bbb_ma_entry = ma_entry_val

        if close is None or ma_entry_val is None or close <= 0 or ma_entry_val <= 0:
            fails.append(f"缺close/MA{entry_ma_n}")
        else:
            dist = abs(float(close) - float(ma_entry_val)) / float(ma_entry_val)
            bbb_dist_entry = float(dist)
            if dist > float(params.dist_ma50_max):
                fails.append(f"离MA{entry_ma_n}太远")

        if entry_u is None or entry_u <= 0:
            fails.append("缺20W上轨")
        else:
            if close is not None and close > float(entry_u) * (1.0 + float(params.max_above_20w)):
                fails.append("追高(高于20W上轨)")

        if params.require_weekly_macd_bullish and macd_state != "bullish":
            fails.append("周MACD未转多")
        if params.require_weekly_macd_above_zero and (macd is None or float(macd) <= 0):
            fails.append("周MACD<=0")
        if params.require_daily_macd_bullish and daily_macd_state != "bullish":
            fails.append("日MACD未转多")

        ok_now = len(fails) == 0

        why = ""
        if ok_now:
            dist_pct = ""
            room_txt = ""
            try:
                base = float(ma_entry_val) if ma_entry_val is not None else float("nan")
                dist_pct = f"{((float(close) - base) / base) * 100:.1f}%"
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                dist_pct = ""
            try:
                delta = (float(close) - float(entry_u)) / float(entry_u)
                if delta >= 0:
                    room_txt = f"高于20W上轨{delta * 100:.1f}%（<=允许{float(params.max_above_20w) * 100:.0f}%）"
                else:
                    room_txt = f"离20W上轨{(-delta) * 100:.1f}%"
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                room_txt = ""

            why = f"通过：周MACD多且>0 / 日MACD多 / 位置靠MA{entry_ma_n}"
            if dist_pct:
                why += f"({dist_pct})"
            why += " / 未追高"
            if room_txt:
                why += f"({room_txt})"

        bbb = {"ok": bool(ok_now), "fails": fails, "why": why, "score": None, "rank_horizon": None, "score_mode": bbb_score_mode2}

        # 只有 ok 才做胜率/磨损统计（否则扫全市场太慢）
        if ok_now:
            # 成本口径（net）：固定磨损换算成比例成本 + 可选滑点/冲击（按成交额/波动估算）
            atr_pct = None
            try:
                if atr is not None and close is not None and float(close) > 0 and float(atr) >= 0:
                    atr_pct = float(atr) / float(close)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                atr_pct = None

            amt_for_slip = daily_amount_avg20
            if amt_for_slip is None:
                amt_for_slip = amount_avg20

            slip_bps = estimate_slippage_bps(
                mode=str(bbb_slippage_mode or "none"),
                amount_avg20_yuan=amt_for_slip,
                atr_pct=atr_pct,
                bps=float(bbb_slippage_bps or 0.0),
                ref_amount_yuan=float(bbb_slippage_ref_amount_yuan or 1e8),
                min_bps=float(bbb_slippage_bps_min or 0.0),
                max_bps=float(bbb_slippage_bps_max or 30.0),
                unknown_bps=float(bbb_slippage_unknown_bps or 10.0),
                vol_mult=float(bbb_slippage_vol_mult or 0.0),
            )
            slip_rate = bps_to_rate(float(slip_bps))
            buy_cost_eff = float(bbb_buy_cost) + float(slip_rate)
            sell_cost_eff = float(bbb_sell_cost) + float(slip_rate)
            bbb_cost = {
                "buy_cost_base": float(bbb_buy_cost),
                "sell_cost_base": float(bbb_sell_cost),
                "slippage": {
                    "mode": str(bbb_slippage_mode or "none"),
                    "bps": float(slip_bps),
                    "ref_amount_yuan": float(bbb_slippage_ref_amount_yuan or 1e8),
                    "bps_min": float(bbb_slippage_bps_min or 0.0),
                    "bps_max": float(bbb_slippage_bps_max or 30.0),
                    "unknown_bps": float(bbb_slippage_unknown_bps or 10.0),
                    "vol_mult": float(bbb_slippage_vol_mult or 0.0),
                    "atr_pct": atr_pct,
                    "amount_avg20_yuan": amt_for_slip,
                },
                "buy_cost": float(buy_cost_eff),
                "sell_cost": float(sell_cost_eff),
                "roundtrip_cost": float(buy_cost_eff + sell_cost_eff),
            }

            sig = compute_bbb_entry_signal(df, df_d if "df_d" in locals() else None, params=params)
            stats_rank: dict[int, Any] = {}
            for h in horizons2:
                try:
                    st, sample = forward_holding_backtest(
                        df,
                        entry_signal=sig,
                        horizon_weeks=int(h),
                        buy_cost=float(buy_cost_eff),
                        sell_cost=float(sell_cost_eff),
                        non_overlapping=bool(bbb_non_overlapping),
                    )
                except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
                    bbb_forward[f"{h}w"] = {"error": str(exc)}
                    continue

                bbb_forward[f"{h}w"] = {
                    "horizon_weeks": st.horizon_weeks,
                    "trades": st.trades,
                    "net_wins": st.wins,
                    "net_win_rate": st.win_rate,
                    "net_win_rate_shrunk": shrunk_win_rate(wins=int(st.wins), trades=int(st.trades)),
                    "net_avg_return": st.avg_return,
                    "net_median_return": st.median_return,
                    "net_implied_ann": st.implied_ann,
                    "gross_wins": st.gross_wins,
                    "gross_win_rate": st.gross_win_rate,
                    "gross_win_rate_shrunk": shrunk_win_rate(wins=int(st.gross_wins), trades=int(st.trades)),
                    "gross_avg_return": st.gross_avg_return,
                    "gross_median_return": st.gross_median_return,
                    "gross_implied_ann": st.gross_implied_ann,
                    "avg_mae": st.avg_mae,
                    "worst_mae": st.worst_mae,
                    "avg_mfe": st.avg_mfe,
                    "best_mfe": st.best_mfe,
                }
                # 兼容旧字段：默认按 net 口径（别让老输出/旧前端直接炸）
                bbb_forward[f"{h}w"]["wins"] = bbb_forward[f"{h}w"]["net_wins"]
                bbb_forward[f"{h}w"]["win_rate"] = bbb_forward[f"{h}w"]["net_win_rate"]
                bbb_forward[f"{h}w"]["win_rate_shrunk"] = bbb_forward[f"{h}w"]["net_win_rate_shrunk"]
                bbb_forward[f"{h}w"]["avg_return"] = bbb_forward[f"{h}w"]["net_avg_return"]
                bbb_forward[f"{h}w"]["median_return"] = bbb_forward[f"{h}w"]["net_median_return"]
                bbb_forward[f"{h}w"]["implied_ann"] = bbb_forward[f"{h}w"]["net_implied_ann"]
                if include_bbb_samples:
                    bbb_forward[f"{h}w"]["sample"] = sample
                stats_rank[h] = st

            # rank_horizon：决定 bbb.score 的口径；best：给“建议持有多久”一个数据参考
            rank_h = int(bbb_rank_horizon)
            if rank_h not in horizons2:
                rank_h = 8 if 8 in horizons2 else horizons2[-1]
            st_rank = stats_rank.get(rank_h)
            bbb_score = score_forward_stats(st_rank, mode=bbb_score_mode2)
            bbb["score"] = float(bbb_score)
            bbb["rank_horizon"] = int(rank_h)

            best_h = None
            best_score = -9e18
            for h, st in stats_rank.items():
                sc = score_forward_stats(st, mode=bbb_score_mode2)
                if sc > best_score:
                    best_score = sc
                    best_h = h
            if best_h is not None and best_h in stats_rank:
                st_best = stats_rank[best_h]
                bbb_best = {
                    "horizon_weeks": int(best_h),
                    "score": float(best_score),
                    "score_mode": bbb_score_mode2,
                    "trades": int(st_best.trades),
                    "net_win_rate": float(st_best.win_rate),
                    "net_win_rate_shrunk": float(shrunk_win_rate(wins=int(st_best.wins), trades=int(st_best.trades))),
                    "net_avg_return": float(st_best.avg_return),
                    "net_median_return": float(st_best.median_return),
                    "net_implied_ann": st_best.implied_ann,
                    "gross_win_rate": float(st_best.gross_win_rate),
                    "gross_win_rate_shrunk": float(shrunk_win_rate(wins=int(st_best.gross_wins), trades=int(st_best.trades))),
                    "gross_avg_return": float(st_best.gross_avg_return),
                    "gross_median_return": float(st_best.gross_median_return),
                    "gross_implied_ann": st_best.gross_implied_ann,
                    "avg_mae": st_best.avg_mae,
                    "worst_mae": st_best.worst_mae,
                    "avg_mfe": st_best.avg_mfe,
                    "best_mfe": st_best.best_mfe,
                }
                # 兼容旧字段：默认按 net 口径
                bbb_best["win_rate"] = bbb_best["net_win_rate"]
                bbb_best["win_rate_shrunk"] = bbb_best["net_win_rate_shrunk"]
                bbb_best["avg_return"] = bbb_best["net_avg_return"]
                bbb_best["median_return"] = bbb_best["net_median_return"]
                bbb_best["implied_ann"] = bbb_best["net_implied_ann"]

            # 出场闭环回测：给“波段大概拿多久/更像怎么卖”的量化参考
            try:
                exit_params = BBBExitParams(
                    weekly_trail_ma=int(bbb_exit_trail_ma or 20),
                    enable_weekly_trail=bool(bbb_exit_enable_trail),
                    stop_loss_ret=float(bbb_exit_stop_loss_ret or 0.0),
                    profit_stop_enabled=bool(bbb_exit_profit_stop_enabled),
                    profit_stop_min_profit_ret=float(bbb_exit_profit_stop_min_profit_ret or 0.0),
                    profit_stop_dd_pct=float(bbb_exit_profit_stop_dd_pct or 0.0),
                    panic_exit_enabled=bool(bbb_exit_panic_enabled),
                    panic_vol_mult=float(bbb_exit_panic_vol_mult or 0.0),
                    panic_min_drop=float(bbb_exit_panic_min_drop or 0.0),
                    panic_drawdown_252d=float(bbb_exit_panic_drawdown_252d or 0.0),
                )
                bt_stats, bt_sample = bbb_exit_backtest(
                    df,
                    df_d if "df_d" in locals() else None,
                    params=params,
                    exit_params=exit_params,
                    buy_cost=float(buy_cost_eff),
                    sell_cost=float(sell_cost_eff),
                    min_hold_days=int(bbb_exit_min_hold_days),
                    cooldown_days=int(bbb_exit_cooldown_days),
                    include_samples=bool(include_bbb_samples),
                )
                bbb_exit_bt = {
                    "trades": bt_stats.trades,
                    "net_wins": bt_stats.wins,
                    "net_win_rate": bt_stats.win_rate,
                    "net_win_rate_shrunk": bt_stats.win_rate_shrunk,
                    "net_avg_return": bt_stats.avg_return,
                    "net_median_return": bt_stats.median_return,
                    "gross_wins": bt_stats.gross_wins,
                    "gross_win_rate": bt_stats.gross_win_rate,
                    "gross_win_rate_shrunk": bt_stats.gross_win_rate_shrunk,
                    "gross_avg_return": bt_stats.gross_avg_return,
                    "gross_median_return": bt_stats.gross_median_return,
                    "avg_hold_days": bt_stats.avg_hold_days,
                    "median_hold_days": bt_stats.median_hold_days,
                    "avg_mae": bt_stats.avg_mae,
                    "worst_mae": bt_stats.worst_mae,
                    "avg_mfe": bt_stats.avg_mfe,
                    "best_mfe": bt_stats.best_mfe,
                    "exits_soft": bt_stats.exits_soft,
                    "exits_hard": bt_stats.exits_hard,
                    "exits_trail": bt_stats.exits_trail,
                    "exits_stop_loss": bt_stats.exits_stop_loss,
                    "exits_profit_stop": bt_stats.exits_profit_stop,
                    "exits_panic": bt_stats.exits_panic,
                    "sample": bt_sample,
                }
                # 兼容旧字段：默认按 net 口径
                bbb_exit_bt["wins"] = bbb_exit_bt["net_wins"]
                bbb_exit_bt["win_rate"] = bbb_exit_bt["net_win_rate"]
                bbb_exit_bt["win_rate_shrunk"] = bbb_exit_bt["net_win_rate_shrunk"]
                bbb_exit_bt["avg_return"] = bbb_exit_bt["net_avg_return"]
                bbb_exit_bt["median_return"] = bbb_exit_bt["net_median_return"]
            except Exception:  # noqa: BLE001
                bbb_exit_bt = None
    except Exception:  # noqa: BLE001
        # BBB 是增强项：别因为它把全扫描搞挂了
        pass

    # Phase2：OpportunityScore（给 scan-* 做统一过滤/排序用；不影响 BBB ok/fails 口径）
    opp_score = None
    opp_bucket = None
    opp_verdict = None
    trap_risk = None
    try:
        # as_of：尽量用最后一根“日线”日期（更贴近真实交易日），否则用周线 last_date
        as_of_s = last_daily_date_str or last_date_str
        try:
            as_of_d = datetime.strptime(str(as_of_s), "%Y-%m-%d").date() if as_of_s else datetime.now().date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_d = datetime.now().date()

        # trap_risk：用 game_theory 的 liquidity_trap.score（0~1，越高越危险）
        try:
            from .factors.game_theory import LiquidityTrapFactor

            r_trap = LiquidityTrapFactor().compute(df)
            try:
                trap_risk = None if r_trap.score is None else float(r_trap.score)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                trap_risk = None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            trap_risk = None

        # expected_holding_days：优先用 BBB best horizon_weeks（粗略*5）
        expected_holding_days = 10
        try:
            if isinstance(bbb_best, dict):
                hw = bbb_best.get("horizon_weeks")
                if hw is not None:
                    expected_holding_days = max(1, int(float(hw) * 5))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            expected_holding_days = 10

        # key_level：默认 ma50（缺就 close）
        kl_name = "ma50"
        kl_value = ma50
        if kl_value is None:
            kl_name = "close"
            kl_value = close

        from .opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        opp = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol=str(item.symbol),
                asset="etf",
                as_of=as_of_d,
                ref_date=as_of_d,
                min_score=0.70,
                t_plus_one=True,
                trap_risk=trap_risk,
                fund_flow=None,
                expected_holding_days=int(expected_holding_days),
            ),
            key_level_name=str(kl_name),
            key_level_value=(None if kl_value is None else float(kl_value)),
        )
        if isinstance(opp, dict):
            try:
                opp_score = None if opp.get("total_score") is None else float(opp.get("total_score"))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                opp_score = None
            opp_bucket = str(opp.get("bucket") or "").strip() or None
            opp_verdict = str(opp.get("verdict") or "").strip() or None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        opp_score = None
        opp_bucket = None
        opp_verdict = None
        trap_risk = None

    out = {
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
        "levels": {
            "support_20w": entry_l,
            "resistance_20w": entry_u,
            "atr": atr,
            "ma50": ma50,
            "ma200": ma200,
            "bbb_entry_ma": bbb_entry_ma,
            "bbb_ma_entry": bbb_ma_entry,
            "bbb_dist_entry": bbb_dist_entry,
        },
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
        "exit": exit_info,
        "bbb": bbb,
        "bbb_cost": bbb_cost,
        "bbb_forward": bbb_forward,
        "bbb_best": bbb_best,
        "bbb_exit_bt": bbb_exit_bt,
        # Alias: external modules/users often just look for "factor_panel"
        "factor_panel": factor_panel_7,
        "factor_panel_7": factor_panel_7,
        # Phase2 scoring (0~1; higher is better)
        "opp_score": opp_score,
        "opp_bucket": opp_bucket,
        "opp_verdict": opp_verdict,
        "trap_risk": trap_risk,
    }

    # 写派生缓存：失败就算了，别影响主流程
    if bool(analysis_cache) and analysis_cache_dir is not None and last_daily_date_str0:
        try:
            from .analysis_cache import ANALYSIS_CACHE_VERSION, cache_path, compute_params_hash, write_cached_json
            from . import __version__ as _ver

            params_hash = compute_params_hash(
                {
                    "v": int(ANALYSIS_CACHE_VERSION),
                    "pkg": str(_ver),
                    "symbol": str(item.symbol),
                    "freq": str(freq),
                    "window": int(window),
                    "ichimoku_params": dict(ichimoku_params or {}),
                    "turtle_entry": int(turtle_entry),
                    "turtle_exit": int(turtle_exit),
                    "turtle_atr": int(turtle_atr),
                    "bbb": {
                        "params": bbb_params,
                        "horizons": list(bbb_horizons or []),
                        "rank_horizon": int(bbb_rank_horizon),
                        "score_mode": str(bbb_score_mode),
                        "buy_cost": float(bbb_buy_cost),
                        "sell_cost": float(bbb_sell_cost),
                        "slippage_mode": str(bbb_slippage_mode),
                        "slippage_bps": float(bbb_slippage_bps),
                        "slippage_ref_amount_yuan": float(bbb_slippage_ref_amount_yuan),
                        "slippage_bps_min": float(bbb_slippage_bps_min),
                        "slippage_bps_max": float(bbb_slippage_bps_max),
                        "slippage_unknown_bps": float(bbb_slippage_unknown_bps),
                        "slippage_vol_mult": float(bbb_slippage_vol_mult),
                        "non_overlapping": bool(bbb_non_overlapping),
                        "exit_min_hold_days": int(bbb_exit_min_hold_days),
                        "exit_cooldown_days": int(bbb_exit_cooldown_days),
                        "include_samples": bool(include_bbb_samples),
                    },
                }
            )
            p = cache_path(cache_dir=Path(analysis_cache_dir), symbol=item.symbol, last_date=str(last_daily_date_str0), params_hash=str(params_hash))
            out2 = dict(out)
            out2["_analysis_cache"] = {"hit": False, "path": str(p), "v": int(ANALYSIS_CACHE_VERSION), "pkg": str(_ver)}
            write_cached_json(p, out2)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    return out
