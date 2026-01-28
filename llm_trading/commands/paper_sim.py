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
from ..logger import get_logger

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

_LOG = get_logger(__name__)

def cmd_paper_sim(args: argparse.Namespace) -> int:
    """
    组合级模拟盘/回测（paper sim，研究用途）。

    目的：把“单票回测爽一下”升级成“账户级别”能复现、能对比的结果。
    """
    import json
    import math

    from ..akshare_source import FetchParams, resolve_symbol
    from ..data_cache import fetch_daily_cached
    from ..paper_sim import simulate_portfolio_paper

    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else (Path("outputs") / f"paper_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # P0 吞错治理：不中断主流程的降级必须可见（写入 diagnostics + logger）。
    warnings: list[str] = []
    errors: list[dict[str, Any]] = []
    _warn_seen: set[str] = set()
    _err_seen: set[str] = set()

    def _warn(msg: str, *, dedupe_key: str | None = None) -> None:
        m = str(msg or "").strip()
        if not m:
            return
        k = str(dedupe_key or m)
        if k in _warn_seen:
            return
        _warn_seen.add(k)
        if len(warnings) < 200:
            warnings.append(m)
        try:
            _LOG.warning("%s", m)
        except (AttributeError):  # noqa: BLE001
            pass

    def _record(stage: str, exc: BaseException, *, note: str | None = None, dedupe_key: str | None = None) -> None:
        k = str(dedupe_key or stage)
        if k in _err_seen:
            return
        _err_seen.add(k)
        if len(errors) < 200:
            errors.append(
                {
                    "ts": datetime.now().isoformat(),
                    "stage": str(stage),
                    "type": exc.__class__.__name__,
                    "error": str(exc),
                    "note": (str(note) if note else None),
                }
            )
        _warn((note or f"{stage} failed: {exc}"), dedupe_key=k)

    strat = str(getattr(args, "strategy", "bbb_etf") or "bbb_etf").strip()
    if strat not in {"bbb_etf", "bbb_stock", "rot_stock_weekly"}:
        raise SystemExit(f"未知 strategy：{strat}（可选：bbb_etf/bbb_stock/rot_stock_weekly）")

    asset = "etf" if strat == "bbb_etf" else "stock"
    adjust = None if asset == "etf" else str(getattr(args, "adjust", "qfq") or "qfq").strip()
    if asset == "stock" and adjust not in {"", "qfq", "hfq"}:
        adjust = "qfq"

    # watchlist：--signals + --symbol 合并去重（保持顺序）
    syms: list[str] = []
    for s in list(getattr(args, "symbol", []) or []):
        s2 = str(s or "").strip()
        if s2:
            syms.append(s2)

    # 额外 watchlist 来源：指数成分股（仅股票）
    universe_index = str(getattr(args, "universe_index", "") or "").strip()
    if universe_index:
        if asset != "stock":
            raise SystemExit("--universe-index 仅股票策略支持（bbb_stock）")
        try:
            import re

            from ..stock_scan import load_index_stock_universe
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"加载指数成分股失败：{exc}") from exc

        alias = {
            "hs300": "000300",
            "csi300": "000300",
            "zz500": "000905",
            "csi500": "000905",
            "zz1000": "000852",
            "csi1000": "000852",
        }

        idx_list: list[str] = []
        for part in re.split(r"[,+;]", universe_index):
            x = str(part or "").strip().lower()
            if not x:
                continue
            if x in alias:
                x = alias[x]
            if x.startswith(("sh", "sz")) and len(x) >= 8 and x[2:].isdigit():
                x = x[2:]
            if x.isdigit():
                x = x.zfill(6)
            else:
                continue
            if x not in idx_list:
                idx_list.append(x)

        if not idx_list:
            raise SystemExit(f"--universe-index 无法解析：{universe_index}（例：000300 或 sh000300+sh000905）")

        for idx in idx_list:
            items = load_index_stock_universe(index_symbol=idx)
            if not items:
                raise SystemExit(f"指数成分股为空：{idx}（源站无数据/接口变更）")
            for it in items:
                syms.append(str(it.symbol))

    sig_path = str(getattr(args, "signals", "") or "").strip()
    if sig_path:
        p = Path(sig_path)
        if not p.exists():
            raise SystemExit(f"找不到 signals 文件：{p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"读取 JSON 失败：{p} {exc}") from exc

        if isinstance(raw, dict) and int(raw.get("schema_version") or 0) == 1 and isinstance(raw.get("items"), list):
            for it in raw.get("items") or []:
                if not isinstance(it, dict):
                    continue
                a = str(it.get("asset") or "").strip().lower()
                if a and a != asset:
                    # signals 里可能混了别的资产，先跳过（别硬算）
                    continue
                s2 = str(it.get("symbol") or "").strip()
                if s2:
                    syms.append(s2)
        else:
            raise SystemExit(f"{p} 不是 signals schema_version=1（请传 scan-* 输出的 signals.json）")

    # core：用“宽基”填满闲置仓位（吃beta，减少现金拖累；研究用途）
    core_spec = str(getattr(args, "core", "") or "").strip()
    core_holdings: dict[str, float] = {}
    if core_spec:
        # 支持：sh510300=0.5,sh510500=0.5 或 510300,510500（等权）
        if "=" in core_spec:
            for part in str(core_spec).split(","):
                p2 = str(part or "").strip()
                if not p2 or "=" not in p2:
                    continue
                k, v = p2.split("=", 1)
                kk = str(k or "").strip()
                if not kk:
                    continue
                try:
                    w = float(v)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
                if (not math.isfinite(w)) or w <= 0:
                    continue
                try:
                    sym_res = resolve_symbol(asset, kk)
                except (AttributeError) as exc:  # noqa: BLE001
                    raise SystemExit(f"--core symbol 解析失败：{kk} {exc}") from exc
                core_holdings[str(sym_res)] = float(w)
        else:
            import re

            for part in re.split(r"[,+]", core_spec):
                kk = str(part or "").strip()
                if not kk:
                    continue
                try:
                    sym_res = resolve_symbol(asset, kk)
                except (AttributeError) as exc:  # noqa: BLE001
                    raise SystemExit(f"--core symbol 解析失败：{kk} {exc}") from exc
                core_holdings[str(sym_res)] = 1.0

        sw = float(sum(core_holdings.values()))
        if sw > 0:
            core_holdings = {k: float(v) / float(sw) for k, v in core_holdings.items()}
            for sym in core_holdings:
                syms.append(str(sym))
        else:
            core_holdings = {}
            core_spec = ""

    # 去重保序
    seen: set[str] = set()
    syms2: list[str] = []
    for s in syms:
        if s in seen:
            continue
        seen.add(s)
        syms2.append(s)
    syms = syms2

    if not syms:
        raise SystemExit("请传 --symbol/--signals（signals.json）或 --core")

    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        syms = syms[: int(limit)]

    cache_dir = Path(getattr(args, "cache_dir", "") or "") if getattr(args, "cache_dir", None) else (Path("data") / "cache" / asset)
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)

    start_date = str(getattr(args, "start_date", "") or "").strip() or None
    end_date = str(getattr(args, "end_date", "") or "").strip() or None

    dfs_by_symbol: dict[str, Any] = {}
    fetch_errors: list[dict[str, Any]] = []
    quality_filtered: list[dict[str, Any]] = []
    for s in syms:
        try:
            sym = resolve_symbol(asset, s)
            # 硬过滤（无未来函数）：北交所直接剔除，别碰（小资金+风控纪律下性价比太低）
            if asset == "stock" and str(sym).lower().startswith("bj"):
                quality_filtered.append({"symbol": str(sym), "name": None, "reasons": ["exclude_bj"], "snapshot": {}})
                continue
            df = fetch_daily_cached(
                FetchParams(asset=asset, symbol=sym, start_date=start_date, end_date=end_date, adjust=adjust),
                cache_dir=cache_dir,
                ttl_hours=float(cache_ttl_hours),
            )
            if df is None or getattr(df, "empty", True):
                fetch_errors.append({"symbol": str(sym), "error": "empty_df"})
                continue
            dfs_by_symbol[str(sym)] = df
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append({"symbol": str(s), "error": str(exc)})

    if not dfs_by_symbol:
        raise SystemExit(f"所有标的都没拉到数据：errors={fetch_errors[:10]} filtered={quality_filtered[:10]}")

    # regime index：用于按日期打牛熊/震荡标签（可选）
    df_idx = None
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    if regime_index and regime_index.lower() not in {"off", "none", "0"}:
        try:
            idx_sym = resolve_symbol("index", regime_index)
            # BBB 回测/组合模拟：别截断指数历史，不然 63D动量/波动/回撤这些前面全是 NaN，等于瞎算。
            idx_start = None if strat in {"bbb_etf", "bbb_stock", "rot_stock_weekly"} else start_date
            df_idx = fetch_daily_cached(
                FetchParams(asset="index", symbol=idx_sym, start_date=idx_start, end_date=end_date),
                cache_dir=Path("data") / "cache" / "index",
                ttl_hours=float(cache_ttl_hours),
            )
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append({"symbol": str(regime_index), "error": f"regime_index_failed: {exc}"})
            df_idx = None

    # BBB factor7：RS 基准指数（允许和 regime-index 不同；支持 '+' 合成）
    df_rs = None
    rs_spec_eff = None
    try:
        rank_mode = str(getattr(args, "bbb_entry_rank_mode", "ma20_dist") or "ma20_dist").strip().lower() or "ma20_dist"
        rot_rank_mode = str(getattr(args, "rot_rank_mode", "factor7") or "factor7").strip().lower() or "factor7"
        rs_spec = str(getattr(args, "bbb_rs_index", "") or "").strip() or "sh000300+sh000905"
        need_rs = bool((rank_mode == "factor7" and strat in {"bbb_etf", "bbb_stock"}) or (rot_rank_mode == "factor7" and strat in {"rot_stock_weekly"}))
        if need_rs:
            if rs_spec.lower() == "auto":
                rs_spec = str(regime_index or "").strip()
            if rs_spec and rs_spec.lower() not in {"off", "none", "0"}:
                from ..index_composite import fetch_index_daily_spec

                rs_start = None if strat in {"bbb_etf", "bbb_stock", "rot_stock_weekly"} else start_date
                df_rs, rs_eff = fetch_index_daily_spec(
                    rs_spec,
                    cache_dir=Path("data") / "cache" / "index",
                    ttl_hours=float(cache_ttl_hours),
                    start_date=rs_start,
                    end_date=end_date,
                )
                rs_spec_eff = rs_eff or rs_spec
    except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
        _record(
            "fetch_rs_index",
            exc,
            note=f"RS 基准指数获取失败（bbb_rs_index={str(getattr(args, 'bbb_rs_index', '') or '').strip() or 'default'}），将降级为不使用 RS 基准",
        )
        df_rs = None
        rs_spec_eff = None

    core_min_pct = float(getattr(args, "core_min_pct", 0.0) or 0.0)
    if (not math.isfinite(core_min_pct)) or core_min_pct < 0:
        core_min_pct = 0.0
    core_min_pct = float(max(0.0, min(core_min_pct, 1.0)))

    min_trade_notional_yuan = float(getattr(args, "min_trade_notional_yuan", 0.0) or 0.0)
    if (not math.isfinite(min_trade_notional_yuan)) or min_trade_notional_yuan < 0:
        min_trade_notional_yuan = 0.0
    min_trade_notional_yuan = float(max(0.0, min_trade_notional_yuan))

    res = simulate_portfolio_paper(
        dfs_by_symbol,
        strategy=strat,  # type: ignore[arg-type]
        start_date=start_date,
        end_date=end_date,
        capital_yuan=float(getattr(args, "capital_yuan", 100000.0) or 100000.0),
        roundtrip_cost_yuan=float(getattr(args, "roundtrip_cost_yuan", 10.0) or 0.0),
        min_fee_yuan=float(getattr(args, "min_fee_yuan", 0.0) or 0.0),
        buy_cost=float(getattr(args, "buy_cost", 0.0) or 0.0),
        sell_cost=float(getattr(args, "sell_cost", 0.0) or 0.0),
        slippage_mode=str(getattr(args, "slippage_mode", "none") or "none"),
        slippage_bps=float(getattr(args, "slippage_bps", 0.0) or 0.0),
        slippage_ref_amount_yuan=float(getattr(args, "slippage_ref_amount_yuan", 1e8) or 1e8),
        slippage_bps_min=float(getattr(args, "slippage_bps_min", 0.0) or 0.0),
        slippage_bps_max=float(getattr(args, "slippage_bps_max", 30.0) or 30.0),
        slippage_unknown_bps=float(getattr(args, "slippage_unknown_bps", 10.0) or 10.0),
        slippage_vol_mult=float(getattr(args, "slippage_vol_mult", 0.0) or 0.0),
        lot_size=int(getattr(args, "lot_size", 100) or 100),
        max_positions=int(getattr(args, "max_positions", 0) or 0),
        max_exposure_pct=float(getattr(args, "max_exposure_pct", 0.0) or 0.0),
        vol_target=float(getattr(args, "vol_target", 0.0) or 0.0),
        vol_lookback_days=int(getattr(args, "vol_lookback_days", 20) or 20),
        max_turnover_pct=float(getattr(args, "max_turnover_pct", 0.0) or 0.0),
        max_corr=float(getattr(args, "max_corr", 0.0) or 0.0),
        max_per_theme=int(getattr(args, "max_per_theme", 0) or 0),
        limit_up_pct=float(getattr(args, "limit_up_pct", 0.0) or 0.0),
        limit_down_pct=float(getattr(args, "limit_down_pct", 0.0) or 0.0),
        halt_vol_zero=bool(getattr(args, "halt_vol_zero", True)),
        df_regime_index_daily=df_idx,
        df_rs_index_daily=df_rs,
        core_holdings=(core_holdings if core_holdings else None),
        core_min_pct=float(core_min_pct),
        min_trade_notional_yuan=float(min_trade_notional_yuan),
        portfolio_dd_stop=float(getattr(args, "portfolio_dd_stop", 0.0) or 0.0),
        portfolio_dd_cooldown_days=int(getattr(args, "portfolio_dd_cooldown_days", 0) or 0),
        portfolio_dd_restart_ma_days=int(getattr(args, "portfolio_dd_restart_ma_days", 0) or 0),
        rot_rebalance_weeks=int(getattr(args, "rot_rebalance_weeks", 1) or 1),
        rot_hold_n=int(getattr(args, "rot_hold_n", 6) or 0),
        rot_buffer_n=int(getattr(args, "rot_buffer_n", 2) or 0),
        rot_rank_mode=str(getattr(args, "rot_rank_mode", "factor7") or "factor7"),
        rot_gap_max=float(getattr(args, "rot_gap_max", 0.015) or 0.0),
        rot_split_exec_days=int(getattr(args, "rot_split_exec_days", 1) or 1),
        bbb_entry_gap_max=float(getattr(args, "bbb_entry_gap_max", 0.015) or 0.0),
        bbb_entry_rank_mode=str(getattr(args, "bbb_entry_rank_mode", "ma20_dist") or "ma20_dist"),
        bbb_factor7_weights=str(getattr(args, "bbb_factor7_weights", "") or ""),
        bbb_entry_ma=int(getattr(args, "bbb_entry_ma", 20) or 20),
        bbb_dist_ma_max=float(getattr(args, "bbb_dist_ma_max", 0.12) or 0.12),
        bbb_max_above_20w=float(getattr(args, "bbb_max_above_20w", 0.05) or 0.05),
        bbb_min_weeks=int(getattr(args, "bbb_min_weeks", 60) or 60),
        bbb_min_hold_days=int(getattr(args, "bbb_min_hold_days", 5) or 5),
        bbb_cooldown_days=int(getattr(args, "bbb_cooldown_days", 0) or 0),
        # shortline 参数先走默认值（你要用再加 flag，别过度设计）
    )

    # 输出：paper_sim.json + report.md + report.json（尽量可复现）
    if isinstance(res, dict):
        # 这俩是排查“为什么没交易/为什么少标的”的关键证据，别让别的字段组装失败把它们吞了。
        res["fetch_errors"] = fetch_errors[:200]
        res["quality_gate_filtered"] = quality_filtered[:200]

    try:
        if isinstance(res, dict):
            res["input"] = {
                "signals": (sig_path if sig_path else None),
                "universe_index": (universe_index if universe_index else None),
                "core": (core_spec if core_spec else None),
                "core_holdings": (core_holdings if core_holdings else None),
                "symbols_input": syms,
                "symbols_used": sorted(dfs_by_symbol),
                "asset": asset,
                "adjust": adjust,
                "cache_dir": str(cache_dir),
                "cache_ttl_hours": float(cache_ttl_hours),
                "start_date": start_date,
                "end_date": end_date,
                "regime_index": (regime_index if regime_index and regime_index.lower() not in {"off", "none", "0"} else None),
                "bbb_rs_index": (rs_spec_eff if rs_spec_eff else None),
                "cost": {
                    "roundtrip_cost_yuan": float(getattr(args, "roundtrip_cost_yuan", 10.0) or 0.0),
                    "min_fee_yuan": float(getattr(args, "min_fee_yuan", 0.0) or 0.0),
                    "buy_cost": float(getattr(args, "buy_cost", 0.0) or 0.0),
                    "sell_cost": float(getattr(args, "sell_cost", 0.0) or 0.0),
                    "slippage": {
                        "mode": str(getattr(args, "slippage_mode", "none") or "none"),
                        "bps": float(getattr(args, "slippage_bps", 0.0) or 0.0),
                        "ref_amount_yuan": float(getattr(args, "slippage_ref_amount_yuan", 1e8) or 1e8),
                        "bps_min": float(getattr(args, "slippage_bps_min", 0.0) or 0.0),
                        "bps_max": float(getattr(args, "slippage_bps_max", 30.0) or 30.0),
                        "unknown_bps": float(getattr(args, "slippage_unknown_bps", 10.0) or 10.0),
                        "vol_mult": float(getattr(args, "slippage_vol_mult", 0.0) or 0.0),
                    },
                },
                "constraints": {
                    "lot_size": int(getattr(args, "lot_size", 100) or 100),
                    "limit_up_pct": float(getattr(args, "limit_up_pct", 0.0) or 0.0),
                    "limit_down_pct": float(getattr(args, "limit_down_pct", 0.0) or 0.0),
                    "halt_vol_zero": bool(getattr(args, "halt_vol_zero", True)),
                    "core_min_pct": float(core_min_pct) or None,
                    "min_trade_notional_yuan": float(min_trade_notional_yuan) or None,
                    "portfolio_dd_stop": float(getattr(args, "portfolio_dd_stop", 0.0) or 0.0) or None,
                    "portfolio_dd_cooldown_days": int(getattr(args, "portfolio_dd_cooldown_days", 0) or 0) or None,
                    "max_turnover_pct_buy_side": float(getattr(args, "max_turnover_pct", 0.0) or 0.0) or None,
                    "max_corr": float(getattr(args, "max_corr", 0.0) or 0.0) or None,
                    "max_per_theme": int(getattr(args, "max_per_theme", 0) or 0) or None,
                    "vol_target_ann": float(getattr(args, "vol_target", 0.0) or 0.0) or None,
                    "vol_lookback_days": int(getattr(args, "vol_lookback_days", 20) or 20),
                },
                "bbb": {
                    "entry_rank_mode": str(getattr(args, "bbb_entry_rank_mode", "ma20_dist") or "ma20_dist"),
                    "factor7_weights": (str(getattr(args, "bbb_factor7_weights", "") or "").strip() or None),
                    "entry_gap_max": float(getattr(args, "bbb_entry_gap_max", 0.015) or 0.0),
                    "entry_ma": int(getattr(args, "bbb_entry_ma", 20) or 20),
                    "dist_ma_max": float(getattr(args, "bbb_dist_ma_max", 0.12) or 0.12),
                    "max_above_20w": float(getattr(args, "bbb_max_above_20w", 0.05) or 0.05),
                    "min_weeks": int(getattr(args, "bbb_min_weeks", 60) or 60),
                    "min_hold_days": int(getattr(args, "bbb_min_hold_days", 5) or 5),
                    "cooldown_days": int(getattr(args, "bbb_cooldown_days", 0) or 0),
                },
                "rot": {
                    "rebalance_weeks": int(getattr(args, "rot_rebalance_weeks", 1) or 1),
                    "hold_n": int(getattr(args, "rot_hold_n", 6) or 0),
                    "buffer_n": int(getattr(args, "rot_buffer_n", 2) or 0),
                    "rank_mode": str(getattr(args, "rot_rank_mode", "factor7") or "factor7"),
                    "gap_max": float(getattr(args, "rot_gap_max", 0.015) or 0.0),
                    "split_exec_days": int(getattr(args, "rot_split_exec_days", 1) or 1),
                },
            }
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record("paper_sim.attach_input", exc, note="组装 paper_sim.input 元信息失败（不影响回测结果）")

    if isinstance(res, dict):
        res["diagnostics"] = {"warnings": warnings[:200], "errors": errors[:200]}

    write_json(out_dir / "paper_sim.json", res)
    if quality_filtered:
        write_json(out_dir / "quality_gate_filtered.json", {"generated_at": datetime.now().isoformat(), "filtered": quality_filtered})

    # report.md（KISS：先把关键信息写出来，后续再美化）
    try:
        s = res.get("summary") if isinstance(res, dict) else None
        s = s if isinstance(s, dict) else {}
        turn = s.get("turnover") if isinstance(s.get("turnover"), dict) else {}
        cap = s.get("capacity") if isinstance(s.get("capacity"), dict) else {}
        reg = s.get("regime_stats") if isinstance(s.get("regime_stats"), dict) else {}
        by_label = reg.get("by_label") if isinstance(reg.get("by_label"), dict) else {}
        lines = [
            "# paper-sim\n",
            "",
            f"- strategy: {strat}",
            f"- as_of: {str(res.get('as_of') if isinstance(res, dict) else '')}",
            f"- symbols_used: {len(dfs_by_symbol)}",
            f"- filtered_by_quality_gate: {len(quality_filtered)}",
            f"- start_date: {start_date or ''}",
            f"- end_date: {end_date or ''}",
            f"- capital_yuan: {s.get('capital_yuan')}",
            f"- equity_last: {s.get('equity_last')}",
            f"- total_return: {s.get('total_return')}",
            f"- cagr: {s.get('cagr')}",
            f"- max_drawdown: {s.get('max_drawdown')}",
            f"- trades: {s.get('trades')}",
            f"- win_rate: {s.get('win_rate')}",
            f"- last_regime: {s.get('last_regime')}",
            "",
            "## turnover/capacity (rough)\n",
            f"- period_years: {s.get('period_years')}",
            f"- turnover_pct_of_capital: {turn.get('turnover_pct_of_capital')}",
            f"- turnover_annualized: {turn.get('turnover_annualized')}",
            f"- participation_p95: {cap.get('p95_participation')}",
            f"- participation_max: {cap.get('max_participation')}",
            "",
            "## regime_stats.by_label (daily)\n",
            *[f"- {k}: days={v.get('days')} total_return={v.get('total_return')}" for k, v in sorted(by_label.items()) if isinstance(v, dict)],
            "",
            "免责声明：研究工具输出，不构成投资建议；买卖自负。",
            "",
        ]
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        _record("write_report_md", exc, note="写出 report.md 失败（不影响 paper_sim/report.json 主流程）")

    data_hash = None
    try:
        from ..analysis_cache import compute_params_hash

        fp: list[dict[str, Any]] = []
        for sym, df in dfs_by_symbol.items():
            last = None
            try:
                last = str(df["date"].max().date())
            except (AttributeError):  # noqa: BLE001
                last = None
            fp.append({"symbol": str(sym), "rows": int(len(df)), "last_date": last})
        fp.sort(key=lambda x: str(x.get("symbol") or ""))
        data_hash = compute_params_hash({"asset": asset, "adjust": adjust, "items": fp}) if fp else None
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record("data_hash.compute", exc, note="计算 data_hash 失败（不影响主流程）")
        data_hash = None

    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "paper-sim", "as_of": (res.get("as_of") if isinstance(res, dict) else None), "data_hash": data_hash})
    run_config = _write_run_config(out_dir, args, note="paper-sim", extra={"cmd": "paper-sim"})
    try:
        from ..reporting import build_report_v1

        write_json(
            out_dir / "report.json",
            build_report_v1(
                cmd="paper-sim",
                run_meta=run_meta,
                run_config=run_config,
                artifacts={"run_meta": "run_meta.json", "run_config": "run_config.json", "paper_sim": "paper_sim.json", "report_md": "report.md"},
                summary=(res.get("summary") if isinstance(res, dict) else res),
                extra={"fetch_errors": fetch_errors[:50]},
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _record("write_report_json", exc, note="写出 report.json 失败（不影响 paper_sim 主流程）")

    # 追加写回 paper_sim.json 的 diagnostics（让单文件也能自解释）
    if isinstance(res, dict):
        res["diagnostics"] = {"warnings": warnings[:200], "errors": errors[:200]}
        try:
            write_json(out_dir / "paper_sim.json", res)
        except (AttributeError) as exc:  # noqa: BLE001
            _record("rewrite_paper_sim_json", exc, note="更新 paper_sim.json(diagnostics) 失败")

    # diagnostics：给排查留证据（比 stdout 可靠）
    try:
        write_json(
            out_dir / "diagnostics.json",
            {
                "schema": "llm_trading.diagnostics.v1",
                "generated_at": datetime.now().isoformat(),
                "cmd": "paper-sim",
                "warnings": warnings[:200],
                "errors": errors[:200],
            },
        )
    except (AttributeError) as exc:  # noqa: BLE001
        try:
            _LOG.warning("写出 diagnostics.json 失败: %s", exc)
        except (AttributeError):  # noqa: BLE001
            pass

    print(str(out_dir.resolve()))
    return 0
