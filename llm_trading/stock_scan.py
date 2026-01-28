from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .akshare_source import DataSourceError, FetchParams
from .backtest import forward_holding_backtest, score_forward_stats, shrunk_win_rate
from .data_cache import fetch_daily_cached
from .indicators import add_donchian_channels, add_macd, add_moving_averages
from .resample import resample_to_weekly
from .quality_gate import StockQualityGate, forbid_by_symbol_name, passes_stock_quality_gate


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
    # 优先读本地缓存（避免每次都打 AkShare，且网络不稳时 scan-stock 直接崩）
    # 缓存由 symbol_names.py 写出：data/cache/universe/stock_names.json
    try:
        import json

        cache_p = Path("data") / "cache" / "universe" / "stock_names.json"
        if cache_p.exists():
            obj = json.loads(cache_p.read_text(encoding="utf-8"))
            items_map = obj.get("items") if isinstance(obj, dict) else None
            if isinstance(items_map, dict) and items_map:
                out: list[StockUniverseItem] = []
                for sym, name in items_map.items():
                    s = str(sym or "").strip().lower()
                    n = str(name or "").strip().replace(" ", "")
                    if not s or not n:
                        continue
                    if (not include_st) and ("ST" in n.upper()):
                        continue
                    if (not include_bj) and s.startswith("bj"):
                        continue
                    # 缓存里已是 sh/sz/bj 前缀
                    if not (len(s) == 8 and s[:2] in {"sh", "sz", "bj"} and s[2:].isdigit()):
                        continue
                    out.append(StockUniverseItem(symbol=s, name=n))

                def sort_key(x: StockUniverseItem):
                    sym2 = x.symbol.lower()
                    market_rank = 9
                    code = 0
                    if sym2.startswith("sh"):
                        market_rank = 0
                        code = int(sym2[2:] or "0")
                    elif sym2.startswith("sz"):
                        market_rank = 1
                        code = int(sym2[2:] or "0")
                    elif sym2.startswith("bj"):
                        market_rank = 2
                        code = int(sym2[2:] or "0")
                    return (market_rank, code, sym2)

                out.sort(key=sort_key)
                return out
    except (OSError, TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
        # 缓存坏了就回退在线抓取
        pass

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


def load_index_stock_universe(*, index_symbol: str = "000300", cache_ttl_hours: float = 24.0) -> list[StockUniverseItem]:
    """
    指数成分股列表（更适合作为“自动股票池”）：
    - 默认：000300=沪深300（流动性相对更好，避免全A太慢/太杂）
    """
    idx = str(index_symbol or "").strip()
    if idx.isdigit():
        idx = idx.zfill(6)
    if not idx:
        idx = "000300"

    # 优先读本地缓存：避免每次都打 AkShare（不稳+慢），也能减少“跑批时拉不到成分”的偶发失败。
    try:
        import json
        import time

        cache_dir = Path("data") / "cache" / "universe"
        cache_p = cache_dir / f"index_{idx}.json"
        ttl = float(cache_ttl_hours or 0.0)
        if ttl > 0 and cache_p.exists():
            age_h = (time.time() - float(cache_p.stat().st_mtime)) / 3600.0
            if age_h <= ttl:
                obj = json.loads(cache_p.read_text(encoding="utf-8"))
                items_map = obj.get("items") if isinstance(obj, dict) else None
                if isinstance(items_map, dict) and items_map:
                    out: list[StockUniverseItem] = []
                    for sym, name in items_map.items():
                        s = str(sym or "").strip().lower()
                        n = str(name or "").strip().replace(" ", "")
                        if not s:
                            continue
                        if not (len(s) == 8 and s[:2] in {"sh", "sz", "bj"} and s[2:].isdigit()):
                            continue
                        out.append(StockUniverseItem(symbol=s, name=(n or s)))

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

                    out.sort(key=sort_key)
                    return out
    except (OSError, TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
        pass

    _require_akshare()
    import akshare as ak

    df = None
    try:
        df = ak.index_stock_cons_csindex(symbol=idx)
    except Exception:  # noqa: BLE001
        df = None
    if df is None or getattr(df, "empty", True):
        try:
            df = ak.index_stock_cons_sina(symbol=idx)
        except Exception:  # noqa: BLE001
            df = None

    if df is None or getattr(df, "empty", True):
        return []

    code_col = "成分券代码" if "成分券代码" in df.columns else None
    name_col = "成分券名称" if "成分券名称" in df.columns else None
    if code_col is None:
        return []

    items: list[StockUniverseItem] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        code = str(row.get(code_col, "")).strip()
        if not code:
            continue
        sym = _code_to_prefixed_symbol(code)
        if not sym or sym in seen:
            continue
        seen.add(sym)

        name = str(row.get(name_col, "")).strip().replace(" ", "") if name_col is not None else ""
        if not name:
            name = sym
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

    items.sort(key=sort_key)

    # 写缓存（失败不影响结果）
    try:
        import json
        from datetime import datetime

        cache_dir = Path("data") / "cache" / "universe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_p = cache_dir / f"index_{idx}.json"
        obj = {
            "generated_at": datetime.now().isoformat(),
            "index_symbol": str(idx),
            "items": {it.symbol: it.name for it in items},
        }
        cache_p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
        pass

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

    # 硬过滤：先用 symbol/name 做快筛，别浪费时间抓一堆杂毛/妖股数据
    gate = StockQualityGate()
    ok_basic, reasons_basic = forbid_by_symbol_name(symbol=item.symbol, name=item.name, gate=gate)
    if not bool(ok_basic):
        # 注意：这里不是 error（源站失败），是“系统拒绝交易的标的”
        return {
            "symbol": item.symbol,
            "name": item.name,
            "filtered": True,
            "filter_reason": list(reasons_basic),
            "quality_gate": {"ok": False, "reasons": list(reasons_basic), "stage": "symbol_name"},
        }

    try:
        df_daily = fetch_daily_cached(
            FetchParams(asset="stock", symbol=item.symbol, start_date=start_date, end_date=end_date, adjust=adjust),
            cache_dir=cache_dir or (Path("data") / "cache" / "stock"),
            ttl_hours=float(cache_ttl_hours),
        )
    except DataSourceError as exc:
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        return {"symbol": item.symbol, "name": item.name, "error": str(exc)}

    # 硬过滤：流动性/低价等（需要日线才能算）
    q = passes_stock_quality_gate(symbol=item.symbol, name=item.name, df_daily=df_daily, gate=gate)
    if not bool(q.get("ok")):
        return {
            "symbol": item.symbol,
            "name": item.name,
            "filtered": True,
            "filter_reason": list(q.get("reasons") or []),
            "quality_gate": q,
        }

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
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
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
        "quality_gate": q,
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
            except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
                out["forward"][which][f"{h}w"] = {"error": str(exc)}
                continue
            out["forward"][which][f"{h}w"] = {
                "horizon_weeks": stats.horizon_weeks,
                "trades": stats.trades,
                "wins": stats.wins,
                "win_rate": stats.win_rate,
                "win_rate_shrunk": shrunk_win_rate(wins=int(stats.wins), trades=int(stats.trades)),
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
        out["scores"][which] = score_forward_stats(st) + float(ma200_bonus)

    # Phase2：OpportunityScore（0~1），用于 scan-stock 的 --min-score 过滤（默认不影响原有信号口径）
    try:
        from datetime import datetime

        as_of_s = str(out.get("date") or "").strip()
        try:
            as_of_d = datetime.strptime(as_of_s, "%Y-%m-%d").date() if as_of_s else datetime.now().date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_d = datetime.now().date()

        # trap_risk：liquidity_trap.score（0~1）
        trap_risk = None
        try:
            from .factors.game_theory import LiquidityTrapFactor

            r_trap = LiquidityTrapFactor().compute(df)
            try:
                trap_risk = None if r_trap.score is None else float(r_trap.score)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                trap_risk = None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            trap_risk = None

        # key_level：默认 ma50（缺就 close）
        lv = out.get("levels") if isinstance(out.get("levels"), dict) else {}
        kl_name = "ma50"
        kl_value = lv.get("ma50")
        if kl_value is None:
            kl_name = "close"
            kl_value = out.get("close")

        from .opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        opp = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol=str(item.symbol),
                asset="stock",
                as_of=as_of_d,
                ref_date=as_of_d,
                min_score=0.70,
                t_plus_one=True,
                trap_risk=trap_risk,
                fund_flow=None,
                expected_holding_days=int(max(1, int(rank_horizon) * 5)),
            ),
            key_level_name=str(kl_name),
            key_level_value=(None if kl_value is None else float(kl_value)),
        )

        out["trap_risk"] = trap_risk
        if isinstance(opp, dict):
            try:
                out["opp_score"] = None if opp.get("total_score") is None else float(opp.get("total_score"))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                out["opp_score"] = None
            out["opp_bucket"] = str(opp.get("bucket") or "").strip() or None
            out["opp_verdict"] = str(opp.get("verdict") or "").strip() or None
        else:
            out["opp_score"] = None
            out["opp_bucket"] = None
            out["opp_verdict"] = None
    except (AttributeError):  # noqa: BLE001
        out["opp_score"] = None
        out["opp_bucket"] = None
        out["opp_verdict"] = None
        out["trap_risk"] = None

    return out
