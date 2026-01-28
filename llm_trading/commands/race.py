from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ..akshare_source import DataSourceError, FetchParams, fetch_daily
from ..chanlun import ChanlunError, compute_chanlun_structure
from ..config import load_config
from ..csv_loader import CsvSchemaError, load_ohlcv_csv
from ..dow import DowError, compute_dow_structure
from ..etf_scan import analyze_etf_symbol, load_etf_universe
from ..indicators import (
    add_accumulation_distribution_line,
    add_adx,
    add_atr,
    add_donchian_channels,
    add_ichimoku,
    add_macd,
    add_moving_averages,
    add_rsi,
)
from ..pipeline import run_llm_analysis, write_json
from ..plotting import (
    plot_chanlun_chart,
    plot_dow_chart,
    plot_ichimoku_chart,
    plot_momentum_chart,
    plot_turtle_chart,
    plot_vsa_chart,
    plot_wyckoff_chart,
)
from ..resample import resample_to_weekly
from ..vsa import compute_vsa_report
from ..stock_scan import DailyFilter, ScanFreq, analyze_stock_symbol, load_stock_universe

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

def cmd_race_strategies(args: argparse.Namespace) -> int:
    """
    经典量化策略赛马（按牛熊/震荡分段输出，研究用途）。
    """
    import json
    import math

    from ..akshare_source import FetchParams
    from ..data_cache import fetch_daily_cached

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("outputs") / f"race_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    asset = str(getattr(args, "asset", "etf") or "etf").strip().lower()

    # universe（可选）：全ETF赛马榜
    universe_mode = str(getattr(args, "universe", "") or "").strip().lower()
    symbol_name: dict[str, str] = {}
    if universe_mode in {"etf", "etf_all"}:
        if asset != "etf":
            raise SystemExit("--universe 目前只支持 asset=etf")
        try:
            from ..etf_scan import load_etf_universe

            uni = load_etf_universe(include_all_funds=bool(universe_mode == "etf_all"))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"ETF 列表拉取失败：{exc}") from exc
        for it in uni:
            sym = str(getattr(it, "symbol", "") or "").strip()
            if not sym:
                continue
            nm = str(getattr(it, "name", "") or "").strip()
            if nm:
                symbol_name[sym] = nm

    # 输入：symbol 或 top_*.json
    syms: list[str] = []
    if symbol_name:
        syms.extend(list(symbol_name.keys()))
    for s in list(getattr(args, "symbol", []) or []):
        s2 = str(s or "").strip()
        if s2:
            syms.append(s2)

    input_path = str(getattr(args, "input", "") or "").strip()
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise SystemExit(f"找不到输入文件：{p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"读取 JSON 失败：{p} {exc}") from exc
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            for it in raw.get("items") or []:
                if not isinstance(it, dict):
                    continue
                sym = str(it.get("symbol") or "").strip()
                if sym:
                    syms.append(sym)

    # 去重（保持顺序）
    seen = set()
    syms2: list[str] = []
    for s in syms:
        if s in seen:
            continue
        seen.add(s)
        syms2.append(s)
    syms = syms2

    if not syms:
        raise SystemExit("请传 --symbol 或 --input( top_*.json )")

    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        syms = syms[: int(limit)]

    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else (Path("data") / "cache" / asset)
    cache_ttl_hours = float(args.cache_ttl_hours) if getattr(args, "cache_ttl_hours", None) is not None else 24.0

    analysis_cache = bool(getattr(args, "analysis_cache", True))
    analysis_cache_dir = (
        Path(args.analysis_cache_dir)
        if getattr(args, "analysis_cache_dir", None)
        else (Path("data") / "cache" / "analysis" / "race")
    )

    # 成本口径（尽量沿用 scan-etf）
    try:
        from ..costs import cost_model_from_roundtrip
    except (AttributeError):  # noqa: BLE001
        cost_model_from_roundtrip = None  # type: ignore[assignment]

    capital_yuan = float(getattr(args, "capital_yuan", 3000.0) or 3000.0)
    roundtrip_cost_yuan = float(getattr(args, "roundtrip_cost_yuan", 10.0) or 10.0)
    cm = cost_model_from_roundtrip(capital_yuan=capital_yuan, roundtrip_cost_yuan=roundtrip_cost_yuan) if cost_model_from_roundtrip else None
    buy_cost = float(args.buy_cost) if getattr(args, "buy_cost", None) is not None else float(getattr(cm, "buy_cost", 0.0) or 0.0)
    sell_cost = float(args.sell_cost) if getattr(args, "sell_cost", None) is not None else float(getattr(cm, "sell_cost", 0.0) or 0.0)

    # 策略列表
    from ..strategy_race import list_default_weekly_strategies, race_weekly_strategies

    wanted_raw = str(getattr(args, "strategies", "") or "").strip()
    wanted: list[str] = []
    if wanted_raw:
        for part in wanted_raw.split(","):
            p = part.strip()
            if p:
                wanted.append(p)

    defs = list_default_weekly_strategies()
    if wanted:
        defs = [d for d in defs if d.key in set(wanted)]
        if not defs:
            raise SystemExit(f"未知 strategies：{wanted_raw}（可用：{', '.join([d.key for d in list_default_weekly_strategies()])}）")

    include_buyhold = bool(getattr(args, "include_buyhold", False))
    min_weeks_total = int(getattr(args, "min_weeks_total", 104) or 0)
    min_weeks_total = max(0, min(min_weeks_total, 5000))
    min_regime_weeks = int(getattr(args, "min_regime_weeks", 26) or 0)
    min_regime_weeks = max(0, min(min_regime_weeks, 5000))
    min_trades = int(getattr(args, "min_trades", 3) or 0)
    min_trades = max(0, min(min_trades, 1000000))
    top_n = int(getattr(args, "top_n", 10) or 10)
    top_n = max(1, min(top_n, 200))
    min_amount_avg20 = float(getattr(args, "min_amount_avg20", 0.0) or 0.0)
    min_amount_avg20 = max(0.0, min(min_amount_avg20, 1e12))

    # 牛熊指数（一次算好）
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    df_reg_w = None
    regime_error = None
    if regime_index.lower() not in {"", "off", "none", "0"}:
        try:
            from ..market_regime import compute_market_regime_weekly_series

            df_idx = fetch_daily_cached(
                FetchParams(asset="index", symbol=str(regime_index)),
                cache_dir=Path("data") / "cache" / "index",
                ttl_hours=6.0,
            )
            df_reg_w = compute_market_regime_weekly_series(index_symbol=str(regime_index), df_daily=df_idx)
        except Exception as exc:  # noqa: BLE001
            df_reg_w = None
            regime_error = str(exc)

    from ..resample import resample_to_weekly

    start_date = str(getattr(args, "start_date", "") or "").strip() or None
    end_date = str(getattr(args, "end_date", "") or "").strip() or None

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def _compute_amount_avg20(df_daily) -> float | None:
        try:
            import pandas as pd
        except ModuleNotFoundError:
            return None
        if df_daily is None or getattr(df_daily, "empty", True):
            return None
        df = df_daily.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            return None
        try:
            if "amount" in df.columns:
                v = float(df["amount"].tail(20).astype(float).mean())
                return v if math.isfinite(v) else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass
        try:
            if "volume" in df.columns and "close" in df.columns:
                amt = (df["close"].astype(float) * df["volume"].astype(float)).tail(20).mean()
                v = float(amt)
                return v if math.isfinite(v) else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return None
        return None

    strategy_name = {d.key: d.name for d in list_default_weekly_strategies()}

    # 派生结果缓存参数：影响单标的赛马结果的东西都放进来
    cache_params = {
        "cmd": "race",
        "asset": str(asset),
        "strategies": [d.key for d in defs],
        "regime_index": (None if df_reg_w is None else str(regime_index)),
        "start_date": start_date,
        "end_date": end_date,
        "buy_cost": float(buy_cost),
        "sell_cost": float(sell_cost),
    }

    def run_one(sym: str) -> dict[str, Any]:
        sym2 = str(sym).strip()
        if not sym2:
            return {"symbol": sym2, "error": "empty symbol"}

        df_daily = fetch_daily_cached(
            FetchParams(asset=str(asset), symbol=str(sym2), start_date=start_date, end_date=end_date),
            cache_dir=cache_dir,
            ttl_hours=float(cache_ttl_hours),
        )

        # 流动性过滤（可选）
        amt20 = _compute_amount_avg20(df_daily)
        if min_amount_avg20 > 0 and (amt20 is None or float(amt20) < float(min_amount_avg20)):
            return {"symbol": sym2, "_skip": "min_amount_avg20", "amount_avg20": amt20}

        # last_date（用于派生缓存 key）
        last_date = None
        try:
            import pandas as pd

            d0 = df_daily.copy()
            d0["date"] = pd.to_datetime(d0["date"], errors="coerce")
            d0 = d0.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            if not d0.empty:
                dt0 = d0["date"].iloc[-1]
                last_date = dt0.strftime("%Y-%m-%d") if hasattr(dt0, "strftime") else str(dt0)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            last_date = None

        cache_hit = False
        cached_obj = None
        if analysis_cache and analysis_cache_dir is not None and last_date:
            try:
                from ..analysis_cache import ANALYSIS_CACHE_VERSION, cache_path, compute_params_hash, read_cached_json
                from .. import __version__ as _ver

                params_hash = compute_params_hash(
                    {
                        "v": int(ANALYSIS_CACHE_VERSION),
                        "pkg": str(_ver),
                        "symbol": str(sym2),
                        **cache_params,
                    }
                )
                p = cache_path(cache_dir=Path(analysis_cache_dir), symbol=str(sym2), last_date=str(last_date), params_hash=str(params_hash))
                cached = read_cached_json(p)
                if cached is not None and cached.get("symbol") == sym2 and cached.get("last_daily_date") == last_date:
                    cached_obj = cached
                    cached_obj["name"] = str(symbol_name.get(sym2) or cached_obj.get("name") or "")
                    cached_obj["_analysis_cache"] = {"hit": True, "path": str(p)}
                    cache_hit = True
            except (AttributeError):  # noqa: BLE001
                cached_obj = None

        if cached_obj is None:
            df_weekly = resample_to_weekly(df_daily)
            payload = race_weekly_strategies(
                df_weekly=df_weekly,
                df_daily=df_daily,
                df_regime_weekly=df_reg_w,
                strategies=defs,
                buy_cost=float(buy_cost),
                sell_cost=float(sell_cost),
                strategy_cfg=None,
            )

            cached_obj = {
                "generated_at": datetime.now().isoformat(),
                "asset": str(asset),
                "symbol": str(sym2),
                "name": str(symbol_name.get(sym2) or ""),
                "last_daily_date": str(last_date or payload.get("as_of") or ""),
                "data": {"last_date": str(payload.get("as_of") or ""), "amount_avg20": amt20},
                "costs": {
                    "capital_yuan": float(capital_yuan),
                    "roundtrip_cost_yuan": float(roundtrip_cost_yuan),
                    "buy_cost": float(buy_cost),
                    "sell_cost": float(sell_cost),
                },
                "regime": {"index": (None if df_reg_w is None else str(regime_index)), "error": regime_error},
                "race": payload,
                "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
            }

            # 写派生缓存
            if analysis_cache and analysis_cache_dir is not None and last_date:
                try:
                    from ..analysis_cache import ANALYSIS_CACHE_VERSION, cache_path, compute_params_hash, write_cached_json
                    from .. import __version__ as _ver

                    params_hash = compute_params_hash(
                        {
                            "v": int(ANALYSIS_CACHE_VERSION),
                            "pkg": str(_ver),
                            "symbol": str(sym2),
                            **cache_params,
                        }
                    )
                    p = cache_path(cache_dir=Path(analysis_cache_dir), symbol=str(sym2), last_date=str(last_date), params_hash=str(params_hash))
                    cached_obj["_analysis_cache"] = {"hit": False, "path": str(p), "v": int(ANALYSIS_CACHE_VERSION), "pkg": str(_ver)}
                    write_cached_json(p, cached_obj)
                except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                    pass

        # 写单标的文件（用于检查）
        out_file = out_dir / f"race_{str(sym2).replace('/', '_')}.json"
        try:
            write_json(out_file, cached_obj)
        except (AttributeError):  # noqa: BLE001
            pass

        payload = cached_obj.get("race") if isinstance(cached_obj, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        data_obj = cached_obj.get("data") if isinstance(cached_obj, dict) and isinstance(cached_obj.get("data"), dict) else {}

        # summary 摘要：按 CAGR 排一下最能打的策略
        best_key = None
        best_cagr = float("-inf")
        by_key: dict[str, Any] = {}
        by_label_best: dict[str, Any] = {}

        weeks_total = int(payload.get("weeks") or 0)
        for it in payload.get("strategies") or []:
            if not isinstance(it, dict):
                continue
            k = str(it.get("key") or "")
            st = it.get("stats") if isinstance(it.get("stats"), dict) else {}
            cagr = st.get("cagr")
            cagr2 = float(cagr) if cagr is not None else float("-inf")
            by_key[k] = {
                "cagr": cagr,
                "total_return": st.get("total_return"),
                "max_drawdown": st.get("max_drawdown"),
                "trades": st.get("trades"),
                "trade_win_rate": st.get("trade_win_rate"),
                "by_regime": it.get("by_regime") if isinstance(it.get("by_regime"), dict) else {},
            }
            if cagr2 > best_cagr:
                best_cagr = cagr2
                best_key = k

        # 每个 regime 选“最强策略”（按 ann_return）
        for lb in ["bull", "bear", "neutral"]:
            best_k = None
            best_ann = float("-inf")
            best_rec = None
            for k, rec in by_key.items():
                if (not include_buyhold) and k == "buyhold":
                    continue
                st = rec if isinstance(rec, dict) else {}
                trades = int(st.get("trades") or 0)
                if int(min_trades) > 0 and trades < int(min_trades) and k != "buyhold":
                    continue
                by_reg = st.get("by_regime") if isinstance(st.get("by_regime"), dict) else {}
                seg = by_reg.get(lb) if isinstance(by_reg.get(lb), dict) else {}
                periods = int(seg.get("periods") or 0)
                if int(min_regime_weeks) > 0 and periods < int(min_regime_weeks):
                    continue
                ann = seg.get("ann_return")
                ann2 = float(ann) if ann is not None else float("-inf")
                if ann2 > best_ann:
                    best_ann = ann2
                    best_k = k
                    best_rec = {"ann_return": ann, "periods": periods, "compounded_return": seg.get("compounded_return")}
            if best_k:
                by_label_best[lb] = {"key": best_k, "name": strategy_name.get(best_k, best_k), **(best_rec or {})}

        return {
            "symbol": str(sym2),
            "name": str(symbol_name.get(sym2) or ""),
            "weeks": int(weeks_total),
            "amount_avg20": amt20,
            "last_date": str(payload.get("as_of") or data_obj.get("last_date") or cached_obj.get("last_daily_date") or ""),
            "best_by_cagr": {"key": best_key, "name": strategy_name.get(best_key or "", best_key), "cagr": (None if best_key is None else by_key.get(best_key, {}).get("cagr"))},
            "best_by_regime": by_label_best,
            "by_strategy": {k: {kk: vv for kk, vv in v.items() if kk != "by_regime"} for k, v in by_key.items()},
            "by_strategy_regime": {k: (v.get("by_regime") if isinstance(v, dict) else {}) for k, v in by_key.items()},
            "file": str(out_file.name),
            "_analysis_cache_hit": bool(cache_hit),
        }

    workers = int(getattr(args, "workers", 8) or 8)
    workers = max(1, min(workers, 32))

    # 并行跑（全ETF会很慢，不并行你等到明年）
    if len(syms) <= 1 or workers <= 1:
        for sym in syms:
            if bool(getattr(args, "verbose", False)):
                print(f"[1/{len(syms)}] {sym} ...")
            try:
                r = run_one(sym)
                if r.get("_skip"):
                    continue
                if r.get("error"):
                    errors.append({"symbol": str(sym), "error": str(r.get("error"))})
                else:
                    results.append(r)
            except (AttributeError) as exc:  # noqa: BLE001
                errors.append({"symbol": str(sym), "error": str(exc)})
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = {ex.submit(run_one, sym): sym for sym in syms}
            done = 0
            for fut in as_completed(futs):
                sym = futs[fut]
                done += 1
                if bool(getattr(args, "verbose", False)) and (done % 50 == 0 or done == len(syms)):
                    print(f"[{done}/{len(syms)}] ...")
                try:
                    r = fut.result()
                    if r.get("_skip"):
                        continue
                    if r.get("error"):
                        errors.append({"symbol": str(sym), "error": str(r.get("error"))})
                    else:
                        results.append(r)
                except (AttributeError) as exc:  # noqa: BLE001
                    errors.append({"symbol": str(sym), "error": str(exc)})

    # leaderboards（按 bull/bear/neutral 出 TopN）
    def _pick_top_by_label(label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            weeks = int(r.get("weeks") or 0)
            if int(min_weeks_total) > 0 and weeks < int(min_weeks_total):
                continue
            b = (r.get("best_by_regime") or {}).get(label) if isinstance(r.get("best_by_regime"), dict) else None
            if not isinstance(b, dict):
                continue
            ann = b.get("ann_return")
            ann2 = float(ann) if ann is not None else float("-inf")
            if not math.isfinite(ann2):
                continue
            rows.append(
                {
                    "symbol": r.get("symbol"),
                    "name": r.get("name"),
                    "weeks": weeks,
                    "amount_avg20": r.get("amount_avg20"),
                    "best_strategy": b,
                    "file": r.get("file"),
                    "score": ann2,
                }
            )
        rows.sort(key=lambda x: float(x.get("score") or float("-inf")), reverse=True)
        for x in rows:
            x.pop("score", None)
        return rows[: int(top_n)]

    leaderboards = {
        "generated_at": datetime.now().isoformat(),
        "params": {
            "asset": str(asset),
            "universe": universe_mode or None,
            "strategies": [d.key for d in defs],
            "regime_index": (None if df_reg_w is None else str(regime_index)),
            "min_weeks_total": int(min_weeks_total),
            "min_regime_weeks": int(min_regime_weeks),
            "min_trades": int(min_trades),
            "include_buyhold": bool(include_buyhold),
            "min_amount_avg20": float(min_amount_avg20),
            "top_n": int(top_n),
        },
        "top": {
            "bull": _pick_top_by_label("bull"),
            "bear": _pick_top_by_label("bear"),
            "neutral": _pick_top_by_label("neutral"),
        },
    }

    write_json(out_dir / "leaderboards.json", leaderboards)
    write_json(out_dir / "summary.json", {"generated_at": datetime.now().isoformat(), "results": results, "errors": errors, "leaderboards": leaderboards.get("top")})

    # 运行 meta + 标准化 report
    as_of = None
    for it in results:
        if not isinstance(it, dict):
            continue
        ld = str(it.get("last_date") or "").strip()
        if not ld:
            continue
        if as_of is None or ld > as_of:
            as_of = ld
    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "race", "as_of": as_of})
    run_config = _write_run_config(out_dir, args, note="race", extra={"cmd": "race"})
    try:
        from ..reporting import build_report_v1

        write_json(
            out_dir / "report.json",
            build_report_v1(
                cmd="race",
                run_meta=run_meta,
                run_config=run_config,
                artifacts={
                    "run_meta": "run_meta.json",
                    "run_config": "run_config.json",
                    "summary": "summary.json",
                    "leaderboards": "leaderboards.json",
                },
                counts={"symbols": int(len(results)), "errors": int(len(errors)), "universe": int(len(syms))},
                summary={"results": results[:50], "errors": errors[:50]},
                extra={
                    "asset": str(asset),
                    "regime_index": (None if df_reg_w is None else str(regime_index)),
                    "regime_error": regime_error,
                    "as_of": as_of,
                    "leaderboards_top": leaderboards.get("top"),
                },
            ),
        )
    except (AttributeError):  # noqa: BLE001
        pass

    print(str(out_dir.resolve()))
    return 0



