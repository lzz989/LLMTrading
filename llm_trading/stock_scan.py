from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .akshare_source import DataSourceError, FetchParams, fetch_daily
from .backtest import forward_holding_backtest
from .indicators import add_donchian_channels, add_macd, add_moving_averages
from .resample import resample_to_weekly


ScanFreq = Literal["daily", "weekly"]
DailyFilter = Literal["none", "ma20", "macd"]


@dataclass(frozen=True)
class StockUniverseItem:
    symbol: str  # sh/sz/bj 前缀
    name: str


def _require_akshare():
    try:
        import akshare  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 akshare？先跑：pip install -r \"requirements.txt\"") from exc


def _code_to_prefixed_symbol(code: str) -> str | None:
    c = str(code).strip().zfill(6)
    if len(c) != 6 or (not c.isdigit()):
        return None
    if c.startswith("6"):
        return f"sh{c}"
    if c.startswith(("0", "3")):
        return f"sz{c}"
    # 北交所（AkShare stock_info_a_code_name 里通常是 9xxxxx）
    if c.startswith("9"):
        return f"bj{c}"
    return None


def load_stock_universe(*, include_st: bool = False, include_bj: bool = True) -> list[StockUniverseItem]:
    """
    全A股票列表（含北交所）。
    """
    _require_akshare()
    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df is None or getattr(df, "empty", True):
        return []

    items: list[StockUniverseItem] = []
    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip().replace(" ", "")
        if not code or not name:
            continue
        if (not include_st) and ("ST" in name.upper()):
            continue

        sym = _code_to_prefixed_symbol(code)
        if not sym:
            continue
        if (not include_bj) and sym.startswith("bj"):
            continue

        items.append(StockUniverseItem(symbol=sym, name=name))

    def sort_key(x: StockUniverseItem):
        sym = x.symbol.lower()
        market_rank = 9
        code = 0
        if sym.startswith("sh"):
            market_rank = 0
            code = int(sym[2:] or "0")
        elif sym.startswith("sz"):
            market_rank = 1
            code = int(sym[2:] or "0")
        elif sym.startswith("bj"):
            market_rank = 2
            code = int(sym[2:] or "0")
        return (market_rank, code, sym)

    # 别按字符串排序：'bj' 在 'sh/sz' 前面，limit 小的时候全是北交所，看着像“抽风”。
    items.sort(key=sort_key)
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


def _fetch_daily_cached(params: FetchParams, *, cache_dir: Path, ttl_hours: float) -> Any:
    """
    简单缓存：同一天里反复扫全A，不要每次都把源站薅秃。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    adjust = params.adjust if params.adjust is not None else "qfq"
    key = f"{params.asset}_{params.symbol}_{adjust}.csv".replace("/", "_").replace("\\", "_")
    path = cache_dir / key

    if path.exists() and ttl_hours > 0:
        age = time.time() - path.stat().st_mtime
        if age <= float(ttl_hours) * 3600.0:
            df = pd.read_csv(path, encoding="utf-8")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df

    df2 = fetch_daily(params)
    try:
        df2.to_csv(path, index=False, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return df2


def _align_daily_filter_ma20(df_daily, df_weekly):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    if df_daily is None or getattr(df_daily, "empty", True):
        dfw = df_weekly.copy()
        dfw["daily_close"] = None
        dfw["daily_ma20"] = None
        dfw["daily_ok"] = False
        return dfw

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dfd["ma20"] = dfd["close"].astype(float).rolling(window=20, min_periods=1).mean()

    dfw = df_weekly.copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    dfd2 = dfd[["date", "close", "ma20"]].rename(columns={"close": "daily_close", "ma20": "daily_ma20"})
    aligned = pd.merge_asof(dfw[["date"]], dfd2, on="date", direction="backward")
    dfw["daily_close"] = aligned["daily_close"]
    dfw["daily_ma20"] = aligned["daily_ma20"]
    dc = pd.to_numeric(dfw["daily_close"], errors="coerce")
    dm = pd.to_numeric(dfw["daily_ma20"], errors="coerce")
    dfw["daily_ok"] = (dc >= dm).fillna(False)
    return dfw


def _align_daily_filter_macd(df_daily, df_weekly):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    if df_daily is None or getattr(df_daily, "empty", True):
        dfw = df_weekly.copy()
        dfw["daily_close"] = None
        dfw["daily_macd"] = None
        dfw["daily_macd_signal"] = None
        dfw["daily_macd_hist"] = None
        dfw["daily_ok"] = False
        return dfw

    dfd = df_daily.copy()
    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

    dfw = df_weekly.copy()
    dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce")
    dfw = dfw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    dfd2 = dfd[["date", "close", "macd", "macd_signal", "macd_hist"]].rename(
        columns={
            "close": "daily_close",
            "macd": "daily_macd",
            "macd_signal": "daily_macd_signal",
            "macd_hist": "daily_macd_hist",
        }
    )
    aligned = pd.merge_asof(dfw[["date"]], dfd2, on="date", direction="backward")
    dfw["daily_close"] = aligned["daily_close"]
    dfw["daily_macd"] = aligned["daily_macd"]
    dfw["daily_macd_signal"] = aligned["daily_macd_signal"]
    dfw["daily_macd_hist"] = aligned["daily_macd_hist"]

    dm = pd.to_numeric(dfw["daily_macd"], errors="coerce")
    ds = pd.to_numeric(dfw["daily_macd_signal"], errors="coerce")
    dfw["daily_ok"] = (dm > ds).fillna(False)
    return dfw


def _score_from_stats(stats) -> float:
    # 艹，这玩意儿就是排序用的“经验分”，别当圣经。
    if stats is None or getattr(stats, "trades", 0) <= 0:
        return 0.0
    mae_penalty = 0.0
    if stats.avg_mae is not None:
        mae_penalty = abs(float(stats.avg_mae)) * 100.0
    return float(stats.win_rate) * 100.0 + float(stats.avg_return) * 80.0 - mae_penalty * 0.6 + math.log(stats.trades + 1.0) * 3.0


def analyze_stock_symbol(
    item: StockUniverseItem,
    *,
    freq: ScanFreq = "weekly",
    window: int = 500,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str | None = None,
    daily_filter: DailyFilter = "macd",
    base_filters: list[str] | None = None,
    horizons: list[int] | None = None,
    rank_horizon: int = 8,
    buy_cost: float = 0.001,
    sell_cost: float = 0.002,
    min_weeks: int = 120,
    non_overlapping: bool = True,
    include_samples: bool = False,
    cache_dir: Path | None = None,
    cache_ttl_hours: float = 24.0,
) -> dict[str, Any]:
    horizons2 = horizons or [4, 8, 12]
    horizons2 = sorted({int(x) for x in horizons2 if int(x) > 0})
    if not horizons2:
        horizons2 = [8]

    try:
        df_daily = _fetch_daily_cached(
            FetchParams(asset="stock", symbol=item.symbol, start_date=start_date, end_date=end_date, adjust=adjust),
            cache_dir=cache_dir or (Path("data") / "cache" / "stock"),
            ttl_hours=float(cache_ttl_hours),
        )
    except DataSourceError as exc:
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}

    if freq == "weekly":
        try:
            df = resample_to_weekly(df_daily)
        except Exception as exc:  # noqa: BLE001
            return {"symbol": item.symbol, "name": item.name, "error": f"转周K失败：{exc}"}
    else:
        df = df_daily.copy()

    df = _ensure_ohlc(df)

    if len(df) < int(min_weeks):
        return {"symbol": item.symbol, "name": item.name, "error": f"K线太少：{len(df)} < {min_weeks}"}

    if window and len(df) > int(window):
        df = df.tail(int(window)).reset_index(drop=True)

    df = add_moving_averages(df, ma_fast=50, ma_slow=200)
    df = add_donchian_channels(df, window=20, upper_col="donchian_upper_20", lower_col="donchian_lower_20", shift=1)

    # daily 辅助过滤（默认开：别在日线明显弱势的时候瞎冲）
    if freq == "weekly" and daily_filter == "ma20":
        df = _align_daily_filter_ma20(df_daily, df)
    elif freq == "weekly" and daily_filter == "macd":
        df = _align_daily_filter_macd(df_daily, df)
    else:
        df = df.copy()
        df["daily_ok"] = True

    # 策略（可插拔）：信号 + 基础过滤
    from .strategy_registry import combine_masks, compute_series

    sig_trend_base = compute_series(df, key="trend")
    sig_swing_base = compute_series(df, key="swing")
    sig_dip_base = compute_series(df, key="dip")

    try:
        mask_base, filter_states = combine_masks(df, filter_keys=list(base_filters or []))
    except Exception as exc:  # noqa: BLE001
        return {"symbol": item.symbol, "name": item.name, "error": f"基础过滤器配置错误：{exc}"}

    # 注意：基础过滤（C：趋势模板）更像“环境过滤器”，它决定“现在能不能上车”，但不应该把历史样本砍到只剩几条。
    # 所以：胜率/磨损统计只用 daily_ok 过滤（策略定义的一部分），不把 base_filters 叠进回测。
    daily_ok = df["daily_ok"].fillna(False)
    sig_trend_bt = (sig_trend_base & daily_ok).fillna(False)
    sig_swing_bt = (sig_swing_base & daily_ok).fillna(False)
    sig_dip_bt = (sig_dip_base & daily_ok).fillna(False)

    # 当前时点信号（用于“能买候选”）：raw_signal AND base_filter_now
    base_ok_now = bool(mask_base.iloc[-1]) if len(mask_base) else True
    sig_trend_now_raw = bool(sig_trend_bt.iloc[-1]) if len(sig_trend_bt) else False
    sig_swing_now_raw = bool(sig_swing_bt.iloc[-1]) if len(sig_swing_bt) else False
    sig_dip_now_raw = bool(sig_dip_bt.iloc[-1]) if len(sig_dip_bt) else False
    sig_trend_now = bool(sig_trend_now_raw and base_ok_now)
    sig_swing_now = bool(sig_swing_now_raw and base_ok_now)
    sig_dip_now = bool(sig_dip_now_raw and base_ok_now)

    last = df.iloc[-1]

    def fnum(v):
        try:
            return None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None

    close_last = fnum(last.get("close"))
    amount_last = fnum(last.get("amount"))
    if amount_last is None:
        vvol = fnum(last.get("volume")) or 0.0
        amount_last = (close_last or 0.0) * vvol

    def _ma200_bonus(close_v: float | None, ma50_v: float | None, ma200_v: float | None) -> float:
        # MA200 软过滤：加分项，不是一票否决
        if close_v is None or ma200_v is None or ma50_v is None:
            return 0.0
        bonus = 0.0
        if ma50_v > ma200_v:
            bonus += 3.0
        if close_v > ma200_v:
            bonus += 3.0
        return bonus

    out: dict[str, Any] = {
        "symbol": item.symbol,
        "name": item.name,
        "date": str(last.get("date").date()) if hasattr(last.get("date"), "date") else str(last.get("date")),
        "close": close_last,
        "amount": amount_last,
        "levels": {
            "ma50": fnum(last.get("ma50")),
            "ma200": fnum(last.get("ma200")),
            "donchian_upper_20": fnum(last.get("donchian_upper_20")),
            "donchian_lower_20": fnum(last.get("donchian_lower_20")),
            "daily_ma20": fnum(last.get("daily_ma20")),
            "daily_macd": fnum(last.get("daily_macd")),
            "daily_macd_signal": fnum(last.get("daily_macd_signal")),
            "daily_macd_hist": fnum(last.get("daily_macd_hist")),
        },
        "daily": {
            "filter": daily_filter,
            "ok": bool(last.get("daily_ok")) if daily_filter != "none" else True,
            "close": fnum(last.get("daily_close")),
            "macd": fnum(last.get("daily_macd")),
            "macd_signal": fnum(last.get("daily_macd_signal")),
        },
        "base_filters": list(base_filters or []),
        "filters": filter_states if isinstance(locals().get("filter_states"), dict) else {},
        "signals_raw": {"trend": sig_trend_now_raw, "swing": sig_swing_now_raw, "dip": sig_dip_now_raw},
        "signals": {"trend": sig_trend_now, "swing": sig_swing_now, "dip": sig_dip_now},
        "forward": {"trend": {}, "swing": {}, "dip": {}},
        "scores": {"trend": 0.0, "swing": 0.0, "dip": 0.0},
    }

    ma200_bonus = _ma200_bonus(out["close"], out["levels"].get("ma50"), out["levels"].get("ma200"))

    # 计算前向收益统计
    for which, sig in [("trend", sig_trend_bt), ("swing", sig_swing_bt), ("dip", sig_dip_bt)]:
        stats_rank: dict[int, Any] = {}
        for h in horizons2:
            try:
                stats, sample = forward_holding_backtest(
                    df,
                    entry_signal=sig,
                    horizon_weeks=int(h),
                    buy_cost=float(buy_cost),
                    sell_cost=float(sell_cost),
                    non_overlapping=bool(non_overlapping),
                )
            except Exception as exc:  # noqa: BLE001
                out["forward"][which][f"{h}w"] = {"error": str(exc)}
                continue
            out["forward"][which][f"{h}w"] = {
                "horizon_weeks": stats.horizon_weeks,
                "trades": stats.trades,
                "wins": stats.wins,
                "win_rate": stats.win_rate,
                "avg_return": stats.avg_return,
                "median_return": stats.median_return,
                "avg_mae": stats.avg_mae,
                "worst_mae": stats.worst_mae,
                "avg_mfe": stats.avg_mfe,
                "best_mfe": stats.best_mfe,
            }
            if include_samples:
                out["forward"][which][f"{h}w"]["sample"] = sample
            stats_rank[h] = stats

        # 默认用 8w 做排序口径（你想改就传 rank_horizon）
        rank_h = int(rank_horizon)
        if rank_h not in horizons2:
            rank_h = 8 if 8 in horizons2 else horizons2[-1]
        st = stats_rank.get(rank_h)
        out["scores"][which] = _score_from_stats(st) + float(ma200_bonus)

    return out
