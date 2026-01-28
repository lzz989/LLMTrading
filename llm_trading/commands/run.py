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
from ..orders_next_open import apply_order_estimates, basic_enrich_orders_next_open, merge_orders_next_open, sort_orders_next_open

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

_LOG = get_logger(__name__)

def cmd_run(args: argparse.Namespace) -> int:
    """
    日常跑批（研究用途）：scan-strategy(默认,因子库) -> (可选 legacy 对照/兜底) -> holdings-user -> rebalance-user -> report。

    说明：
    - 先把“每天/每周固定产物”做出来：signals/portfolio/orders/report。
    - 先半自动：生成 orders.json，人手下单；别一上来就接券商 API。
    """
    import json
    import shutil

    from ..costs import TradeCost, bps_to_rate, cash_buy, cash_sell, estimate_slippage_bps, trade_cost_from_params
    from ..data_cache import fetch_daily_cached
    from ..tradeability import TradeabilityConfig, tradeability_flags

    # P0 吞错治理：所有“不中断主流程的降级”都要可见（写入 warnings/errors + logger），避免结果默默变样。
    run_warnings: list[str] = []
    run_errors: list[dict[str, Any]] = []
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
        if len(run_warnings) < 200:
            run_warnings.append(m)
        try:
            _LOG.warning("%s", m)
        except (AttributeError):  # noqa: BLE001
            pass

    def _record(stage: str, exc: BaseException, *, note: str | None = None, dedupe_key: str | None = None) -> None:
        k = str(dedupe_key or stage)
        if k in _err_seen:
            return
        _err_seen.add(k)
        if len(run_errors) < 200:
            run_errors.append(
                {
                    "ts": datetime.now().isoformat(),
                    "stage": str(stage),
                    "type": exc.__class__.__name__,
                    "error": str(exc),
                    "note": (str(note) if note else None),
                }
            )
        _warn((note or f"{stage} failed: {exc}"), dedupe_key=k)

    # out_dir：默认 outputs/run_YYYYMMDD；同一天重复跑就自动加 _2/_3
    today = datetime.now().strftime("%Y%m%d")
    base = str(getattr(args, "out_dir", "") or "").strip()
    out_dir = Path(base) if base else (Path("outputs") / f"run_{today}")
    if out_dir.exists():
        for i in range(2, 2000):
            cand = Path(f"{out_dir}_{i}")
            if not cand.exists():
                out_dir = cand
                break
    out_dir.mkdir(parents=True, exist_ok=True)

    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    holdings_path = str(getattr(args, "holdings_path", "") or "").strip() or str(Path("data") / "user_holdings.json")

    # 调仓执行窗：优先 CLI；否则读 user_holdings.trade_rules.rebalance_schedule
    reb_schedule = str(getattr(args, "rebalance_schedule", "") or "").strip().lower()
    if not reb_schedule:
        try:
            obj = json.loads(Path(holdings_path).read_text(encoding="utf-8"))
            rules = obj.get("trade_rules") if isinstance(obj, dict) else None
            if not isinstance(rules, dict):
                rules = obj.get("rules") if isinstance(obj, dict) else None
            if isinstance(rules, dict):
                reb_schedule = str(rules.get("rebalance_schedule") or "").strip().lower()
        except (AttributeError) as exc:  # noqa: BLE001
            _record("load_user_holdings.rebalance_schedule", exc, note=f"读取 {holdings_path} 失败，rebalance_schedule 将回退为 any_day")
            reb_schedule = ""
    if reb_schedule not in {"", "any_day", "fri_close_mon_open"}:
        reb_schedule = "any_day"

    reb_mode = str(getattr(args, "rebalance_mode", "add") or "add").strip().lower()
    if reb_mode not in {"add", "rotate"}:
        reb_mode = "add"

    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
    stock_adjust = str(getattr(args, "stock_adjust", "qfq") or "qfq").strip() or "qfq"
    tb_cfg = TradeabilityConfig(
        limit_up_pct=float(getattr(args, "limit_up_pct", 0.0) or 0.0),
        limit_down_pct=float(getattr(args, "limit_down_pct", 0.0) or 0.0),
        halt_vol_zero=bool(getattr(args, "halt_vol_zero", True)),
    )

    # 复用同一套 parser，避免手搓 Namespace（少犯错）。
    from ..cli import build_parser

    parser = build_parser()

    def _invoke(argv2: list[str]) -> int:
        ns = parser.parse_args(argv2)
        try:
            setattr(ns, "_argv", list(argv2))
        except (AttributeError):  # noqa: BLE001
            pass
        return int(ns.func(ns))

    # 1) signals：优先用 --signals；否则按 scan-mode 自动生成（默认走 scan-strategy；失败回退 legacy scan-etf）。
    signals_in_raw = getattr(args, "signals", None)
    sig_inputs: list[str] = []
    if isinstance(signals_in_raw, list):
        for x in signals_in_raw:
            s = str(x or "").strip()
            if not s:
                continue
            # 兼容逗号分隔（少让你打一堆 --signals）
            for part in s.split(","):
                part2 = str(part or "").strip()
                if part2:
                    sig_inputs.append(part2)
    else:
        s = str(signals_in_raw or "").strip()
        if s:
            for part in s.split(","):
                part2 = str(part or "").strip()
                if part2:
                    sig_inputs.append(part2)
    # 扫描产物目录：
    # - scan_etf/: legacy scan-etf（兼容旧报告结构；也可作为对照/兜底）
    # - scan_strategy/: 因子库 scan-strategy（新默认）
    scan_dir = out_dir / "scan_etf"
    scan_dir.mkdir(parents=True, exist_ok=True)
    scan_strategy_dir = out_dir / "scan_strategy"
    scan_strategy_dir.mkdir(parents=True, exist_ok=True)
    scan_strategy_left_dir = out_dir / "scan_strategy_left"
    scan_strategy_left_dir.mkdir(parents=True, exist_ok=True)

    if sig_inputs:
        sig_paths: list[Path] = []
        for s in sig_inputs:
            p = Path(str(s))
            if not p.exists():
                raise SystemExit(f"找不到 --signals：{p}")
            sig_paths.append(p)

        if len(sig_paths) == 1:
            sig_src = sig_paths[0]
            shutil.copyfile(sig_src, out_dir / "signals.json")
            signals_path = out_dir / "signals.json"
        else:
            from ..signals_merge import merge_signals_files, parse_priority, parse_strategy_weights

            sig_inputs_dir = out_dir / "signals_inputs"
            sig_inputs_dir.mkdir(parents=True, exist_ok=True)
            copied: list[Path] = []
            for i, src in enumerate(sig_paths, start=1):
                dst = sig_inputs_dir / f"{i:02d}_{src.name}"
                shutil.copyfile(src, dst)
                copied.append(dst)

            weights = parse_strategy_weights(str(getattr(args, "signals_merge_weights", "") or ""))
            priority = parse_priority(str(getattr(args, "signals_merge_priority", "") or ""))
            conflict = str(getattr(args, "signals_merge_conflict", "risk_first") or "risk_first").strip().lower()
            if conflict not in {"risk_first", "priority", "vote"}:
                conflict = "risk_first"

            merged = merge_signals_files(
                copied,
                conflict=conflict,  # type: ignore[arg-type]
                weights=weights,
                priority=priority,
                top_k=int(getattr(args, "signals_merge_top_k", 0) or 0),
            )

            # signals.json：下游默认入口；signals_merged.json：给人类看的显式文件名（便于 grep）
            write_json(out_dir / "signals_merged.json", merged)
            write_json(out_dir / "signals.json", merged)
            signals_path = out_dir / "signals.json"
    else:
        scan_freq = str(getattr(args, "scan_freq", "weekly") or "weekly").strip().lower()
        if scan_freq not in {"daily", "weekly"}:
            scan_freq = "weekly"
        scan_limit = int(getattr(args, "scan_limit", 200) or 200)
        scan_min_weeks = int(getattr(args, "scan_min_weeks", 60) or 60)
        scan_top_k = int(getattr(args, "scan_top_k", 30) or 30)
        scan_top_k = max(1, min(scan_top_k, 500))

        scan_mode = str(getattr(args, "scan_mode", "auto") or "auto").strip().lower() or "auto"
        if scan_mode not in {"auto", "strategy", "legacy"}:
            scan_mode = "auto"

        shadow_legacy = bool(getattr(args, "scan_shadow_legacy", True))
        align_top_k = int(getattr(args, "scan_align_top_k", 30) or 30)
        align_top_k = max(1, min(align_top_k, 5000))

        strategy_cfg = str(getattr(args, "scan_strategy_config", "") or "").strip() or str(Path("config") / "strategy_configs.yaml")
        strategy_key = str(getattr(args, "scan_strategy", "") or "").strip() or "bbb_weekly"

        def _signals_items_count(p: Path) -> int | None:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    return None
                items0 = obj.get("items")
                if not isinstance(items0, list):
                    return None
                return int(len(items0))
            except Exception:  # noqa: BLE001
                return None

        # 优先因子库（scan-strategy）；失败就回退 legacy（scan-etf）。
        used_scan_mode = None  # "strategy" | "legacy"
        strategy_ok = False
        if scan_mode in {"auto", "strategy"}:
            try:
                _invoke(
                    [
                        "scan-strategy",
                        "--asset",
                        "etf",
                        "--freq",
                        str(scan_freq),
                        "--limit",
                        str(scan_limit),
                        "--top-k",
                        str(scan_top_k),
                        "--strategy-config",
                        str(strategy_cfg),
                        "--strategy",
                        str(strategy_key),
                        "--out-dir",
                        str(scan_strategy_dir),
                        "--regime-index",
                        str(regime_index),
                    ]
                    + ([] if bool(regime_canary) else ["--no-regime-canary"])
                )
                sp = scan_strategy_dir / "signals.json"
                n = _signals_items_count(sp) if sp.exists() else None
                if n is None:
                    raise RuntimeError(f"scan-strategy 未产出合法 signals.json：{sp}")
                if n <= 0:
                    raise RuntimeError("scan-strategy 产出 signals.items=0（视为无效结果）")

                shutil.copyfile(sp, out_dir / "signals.json")
                shutil.copyfile(sp, out_dir / "signals_strategy.json")
                used_scan_mode = "strategy"
                strategy_ok = True
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                _record("scan_strategy", exc, note="scan-strategy 失败，将回退 legacy（若 scan_mode=auto）")
                strategy_ok = False
                if scan_mode == "strategy":
                    raise

        legacy_ok = False
        if scan_mode == "legacy" or (scan_mode == "auto" and not strategy_ok):
            try:
                _invoke(
                    [
                        "scan-etf",
                        "--freq",
                        str(scan_freq),
                        "--limit",
                        str(scan_limit),
                        "--top-k",
                        str(scan_top_k),
                        "--min-weeks",
                        str(scan_min_weeks),
                        "--out-dir",
                        str(scan_dir),
                        "--regime-index",
                        str(regime_index),
                    ]
                    + ([] if bool(regime_canary) else ["--no-regime-canary"])
                )

                lp = scan_dir / "signals.json"
                if not lp.exists():
                    raise RuntimeError(f"scan-etf 未产出 signals.json：{lp}")
                shutil.copyfile(lp, out_dir / "signals.json")
                shutil.copyfile(lp, out_dir / "signals_legacy.json")
                used_scan_mode = "legacy"
                legacy_ok = True

                top_bbb = scan_dir / "top_bbb.json"
                if top_bbb.exists():
                    shutil.copyfile(top_bbb, out_dir / "top_bbb.json")
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                _record("scan_etf", exc, note="scan-etf(legacy) 失败；无法产出 signals.json，run 将中止")
                raise

        # shadow legacy：稳健切换时，保留一份对照 + 自动生成对齐报告（不影响主流程）。
        if used_scan_mode == "strategy" and shadow_legacy and (not legacy_ok):
            try:
                _invoke(
                    [
                        "scan-etf",
                        "--freq",
                        str(scan_freq),
                        "--limit",
                        str(scan_limit),
                        "--top-k",
                        str(scan_top_k),
                        "--min-weeks",
                        str(scan_min_weeks),
                        "--out-dir",
                        str(scan_dir),
                        "--regime-index",
                        str(regime_index),
                    ]
                    + ([] if bool(regime_canary) else ["--no-regime-canary"])
                )
                lp2 = scan_dir / "signals.json"
                if lp2.exists():
                    shutil.copyfile(lp2, out_dir / "signals_legacy.json")
                top_bbb2 = scan_dir / "top_bbb.json"
                if top_bbb2.exists():
                    shutil.copyfile(top_bbb2, out_dir / "top_bbb_legacy.json")
                legacy_ok = True
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                _record("scan_etf.shadow", exc, note="shadow legacy scan-etf 失败（已跳过，不影响 orders/report 主流程）")

        # strategy-align：新旧信号对齐报告（可审计、可量化）
        try:
            base_p = out_dir / "signals_legacy.json"
            new_p = out_dir / "signals_strategy.json"
            if base_p.exists() and new_p.exists():
                align_dir = out_dir / "strategy_alignment"
                align_dir.mkdir(parents=True, exist_ok=True)
                _invoke(["strategy-align", "--base", str(base_p), "--new", str(new_p), "--top-k", str(align_top_k), "--out-dir", str(align_dir)])
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            _record("strategy_align", exc, note="strategy-align 失败（已跳过，不影响 orders/report 主流程）")

    # 1.2) （可选）左侧低吸候选：再跑一份 scan-strategy（不影响主 signals/orders，只给你“高赔率试错清单”）
    signals_left_path: Path | None = None
    if bool(getattr(args, "scan_left", True)):
        left_key = str(getattr(args, "scan_left_strategy", "") or "left_dip_rr").strip()
        if left_key:
            scan_freq_l = str(getattr(args, "scan_freq", "weekly") or "weekly").strip().lower()
            if scan_freq_l not in {"daily", "weekly"}:
                scan_freq_l = "weekly"
            scan_limit_l = int(getattr(args, "scan_limit", 200) or 200)
            scan_top_k_l = int(getattr(args, "scan_left_top_k", 30) or 30)
            scan_top_k_l = max(1, min(scan_top_k_l, 500))
            strategy_cfg_l = str(getattr(args, "scan_strategy_config", "") or "").strip() or str(Path("config") / "strategy_configs.yaml")

            def _signals_items_count(p: Path) -> int | None:
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(obj, dict):
                        return None
                    items0 = obj.get("items")
                    if not isinstance(items0, list):
                        return None
                    return int(len(items0))
                except Exception:  # noqa: BLE001
                    return None

            try:
                _invoke(
                    [
                        "scan-strategy",
                        "--asset",
                        "etf",
                        "--freq",
                        str(scan_freq_l),
                        "--limit",
                        str(scan_limit_l),
                        "--top-k",
                        str(scan_top_k_l),
                        "--strategy-config",
                        str(strategy_cfg_l),
                        "--strategy",
                        str(left_key),
                        "--out-dir",
                        str(scan_strategy_left_dir),
                        "--regime-index",
                        str(regime_index),
                    ]
                    + ([] if bool(regime_canary) else ["--no-regime-canary"])
                )
                sp = scan_strategy_left_dir / "signals.json"
                n = _signals_items_count(sp) if sp.exists() else None
                if n is None:
                    raise RuntimeError(f"scan-left 未产出合法 signals.json：{sp}")
                if n <= 0:
                    raise RuntimeError("scan-left 产出 signals.items=0（视为无效结果）")
                shutil.copyfile(sp, out_dir / "signals_left.json")
                signals_left_path = out_dir / "signals_left.json"
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                _record("scan_strategy_left", exc, note="scan-left 失败（已跳过，不影响 orders/report 主流程）")
                signals_left_path = None

    # 1.5) （可选）stock signals：额外跑一份 scan-strategy(stock) 做观察池（不影响 ETF 的 rebalance/orders 主流程）
    signals_stock_path: Path | None = None
    if bool(getattr(args, "scan_stock", False)):
        scan_stock_dir = out_dir / "scan_strategy_stock"
        scan_stock_dir.mkdir(parents=True, exist_ok=True)

        scan_freq2 = str(getattr(args, "scan_freq", "weekly") or "weekly").strip().lower()
        if scan_freq2 not in {"daily", "weekly"}:
            scan_freq2 = "weekly"

        stock_universe = str(getattr(args, "scan_stock_universe", "hs300") or "hs300").strip() or "hs300"
        stock_limit = int(getattr(args, "scan_stock_limit", 300) or 300)
        stock_top_k = int(getattr(args, "scan_stock_top_k", 30) or 30)
        stock_top_k = max(1, min(stock_top_k, 5000))
        stock_strategy = str(getattr(args, "scan_stock_strategy", "") or "").strip() or str(getattr(args, "scan_strategy", "") or "").strip() or "bbb_weekly"
        stock_source = str(getattr(args, "scan_stock_source", "auto") or "auto").strip().lower() or "auto"

        strategy_cfg2 = str(getattr(args, "scan_strategy_config", "") or "").strip() or str(Path("config") / "strategy_configs.yaml")

        def _signals_items_count(p: Path) -> int | None:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    return None
                items0 = obj.get("items")
                if not isinstance(items0, list):
                    return None
                return int(len(items0))
            except Exception:  # noqa: BLE001
                return None

        try:
            _invoke(
                [
                    "scan-strategy",
                    "--asset",
                    "stock",
                    "--universe",
                    str(stock_universe),
                    "--freq",
                    str(scan_freq2),
                    "--limit",
                    str(stock_limit),
                    "--top-k",
                    str(stock_top_k),
                    "--strategy-config",
                    str(strategy_cfg2),
                    "--strategy",
                    str(stock_strategy),
                    "--source",
                    str(stock_source),
                    "--out-dir",
                    str(scan_stock_dir),
                    "--regime-index",
                    str(regime_index),
                ]
                + ([] if bool(regime_canary) else ["--no-regime-canary"])
            )
            sp = scan_stock_dir / "signals.json"
            n = _signals_items_count(sp) if sp.exists() else None
            if n is None:
                raise RuntimeError(f"scan-strategy(stock) 未产出合法 signals.json：{sp}")
            if n <= 0:
                raise RuntimeError("scan-strategy(stock) 产出 signals.items=0（视为无效结果）")
            shutil.copyfile(sp, out_dir / "signals_stock.json")
            signals_stock_path = out_dir / "signals_stock.json"
        except (SystemExit, Exception) as exc:  # noqa: BLE001
            _record("scan_strategy_stock", exc, note="scan-strategy(stock) 失败（已跳过，不影响 ETF orders/report 主流程）")
            signals_stock_path = None

    # 1.6) （可选）stock 左侧低吸候选：同一次 run 顺手产出（减少你“再戳一次”的体力活）
    signals_left_stock_path: Path | None = None
    if bool(getattr(args, "scan_stock", False)) and bool(getattr(args, "scan_left", True)):
        left_key2 = str(getattr(args, "scan_left_strategy", "") or "left_dip_rr").strip()
        if left_key2:
            scan_stock_left_dir = out_dir / "scan_strategy_stock_left"
            scan_stock_left_dir.mkdir(parents=True, exist_ok=True)

            scan_freq3 = str(getattr(args, "scan_freq", "weekly") or "weekly").strip().lower()
            if scan_freq3 not in {"daily", "weekly"}:
                scan_freq3 = "weekly"

            stock_universe2 = str(getattr(args, "scan_stock_universe", "hs300") or "hs300").strip() or "hs300"
            stock_limit2 = int(getattr(args, "scan_stock_limit", 300) or 300)
            stock_top_k_l = int(getattr(args, "scan_left_top_k", 30) or 30)
            stock_top_k_l = max(1, min(stock_top_k_l, 5000))
            stock_source2 = str(getattr(args, "scan_stock_source", "auto") or "auto").strip().lower() or "auto"
            strategy_cfg3 = str(getattr(args, "scan_strategy_config", "") or "").strip() or str(Path("config") / "strategy_configs.yaml")

            def _signals_items_count(p: Path) -> int | None:
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    if not isinstance(obj, dict):
                        return None
                    items0 = obj.get("items")
                    if not isinstance(items0, list):
                        return None
                    return int(len(items0))
                except Exception:  # noqa: BLE001
                    return None

            try:
                _invoke(
                    [
                        "scan-strategy",
                        "--asset",
                        "stock",
                        "--universe",
                        str(stock_universe2),
                        "--freq",
                        str(scan_freq3),
                        "--limit",
                        str(stock_limit2),
                        "--top-k",
                        str(stock_top_k_l),
                        "--strategy-config",
                        str(strategy_cfg3),
                        "--strategy",
                        str(left_key2),
                        "--source",
                        str(stock_source2),
                        "--out-dir",
                        str(scan_stock_left_dir),
                        "--regime-index",
                        str(regime_index),
                    ]
                    + ([] if bool(regime_canary) else ["--no-regime-canary"])
                )
                sp = scan_stock_left_dir / "signals.json"
                n = _signals_items_count(sp) if sp.exists() else None
                if n is None:
                    raise RuntimeError(f"scan-left(stock) 未产出合法 signals.json：{sp}")
                if n <= 0:
                    raise RuntimeError("scan-left(stock) 产出 signals.items=0（视为无效结果）")
                shutil.copyfile(sp, out_dir / "signals_left_stock.json")
                signals_left_stock_path = out_dir / "signals_left_stock.json"
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                _record("scan_strategy_left_stock", exc, note="scan-left(stock) 失败（已跳过，不影响 ETF orders/report 主流程）")
                signals_left_stock_path = None

    # 2) holdings-user：持仓 + 组合层汇总
    holdings_out = out_dir / "holdings_user.json"
    _invoke(
        [
            "holdings-user",
            "--path",
            str(holdings_path),
            "--regime-index",
            str(regime_index),
            "--cache-ttl-hours",
            str(cache_ttl_hours),
            "--stock-adjust",
            str(stock_adjust),
            "--out",
            str(holdings_out),
        ]
        + ([] if bool(regime_canary) else ["--no-regime-canary"])
    )

    # 2.5) national-team：国家队/托底代理指标（研究用途；缺数据也不能炸主流程）
    nt_out = out_dir / "national_team.json"
    try:
        holdings_asof2 = ""
        try:
            hold_obj0 = json.loads(holdings_out.read_text(encoding="utf-8"))
            holdings_asof2 = str(hold_obj0.get("as_of") or "") if isinstance(hold_obj0, dict) else ""
        except (AttributeError) as exc:  # noqa: BLE001
            _record("national_team.read_holdings_asof", exc, note="读取 holdings_user.json 的 as_of 失败，national-team 将不传 --as-of")
            holdings_asof2 = ""

        idx0 = str(regime_index or "sh000300")
        # multi-index/canary 格式里，取第一个作为“盯盘指数”
        idx0 = idx0.split(";", 1)[0].split(",", 1)[0].strip() or "sh000300"

        argv_nt = ["national-team", "--cache-ttl-hours", str(cache_ttl_hours), "--index-symbol", str(idx0), "--out", str(nt_out)]
        if holdings_asof2:
            argv_nt += ["--as-of", str(holdings_asof2)]
        _invoke(argv_nt)
    except (SystemExit, Exception) as exc:  # noqa: BLE001
        # 可选模块：失败就记录，主流程继续。
        _record("national_team.invoke", exc, note="national-team 执行失败，已跳过（不影响 orders/report 主流程）")

    # 3) rebalance-user：给出次日开盘订单清单
    rebalance_out = out_dir / "rebalance_user.json"
    _invoke(
        [
            "rebalance-user",
            "--path",
            str(holdings_path),
            "--signals",
            str(out_dir / "signals.json"),
            "--mode",
            str(reb_mode),
            "--regime-index",
            str(regime_index),
            "--cache-ttl-hours",
            str(cache_ttl_hours),
            "--stock-adjust",
            str(stock_adjust),
            "--limit-up-pct",
            str(getattr(args, "limit_up_pct", 0.0) or 0.0),
            "--limit-down-pct",
            str(getattr(args, "limit_down_pct", 0.0) or 0.0),
            "--vol-target",
            str(getattr(args, "vol_target", 0.0) or 0.0),
            "--vol-lookback-days",
            str(getattr(args, "vol_lookback_days", 20) or 20),
            "--max-turnover-pct",
            str(getattr(args, "max_turnover_pct", 0.0) or 0.0),
            "--max-per-theme",
            str(getattr(args, "max_per_theme", 0) or 0),
            "--out",
            str(rebalance_out),
        ]
        + ([] if bool(regime_canary) else ["--no-regime-canary"])
        + ([] if bool(getattr(args, "halt_vol_zero", True)) else ["--no-halt-vol-zero"])
        + (
            ["--max-exposure-pct", str(getattr(args, "max_exposure_pct"))]
            if getattr(args, "max_exposure_pct", None) is not None
            else []
        )
        + (
            ["--min-trade-notional-yuan", str(getattr(args, "min_trade_notional_yuan"))]
            if getattr(args, "min_trade_notional_yuan", None) is not None
            else []
        )
        + (
            ["--max-positions", str(getattr(args, "max_positions"))]
            if getattr(args, "max_positions", None) is not None
            else []
        )
        + (
            ["--max-position-pct", str(getattr(args, "max_position_pct"))]
            if getattr(args, "max_position_pct", None) is not None
            else []
        )
        + (
            ["--max-corr", str(getattr(args, "max_corr"))]
            if getattr(args, "max_corr", None) is not None
            else []
        )
    )

    # 3.5) （可选）持仓深度复盘：逐标的跑 analyze --method all，并聚合一份 report_holdings.md
    if bool(getattr(args, "deep_holdings", False)):
        deep_dir = out_dir / "holdings_deep"
        deep_dir.mkdir(parents=True, exist_ok=True)
        deep_report_md = out_dir / "report_holdings.md"
        deep_summary_json = out_dir / "holdings_deep_summary.json"

        # frozen 标的：从 user_holdings.json 里读（holdings_user.json 不一定保留 frozen 字段）
        frozen_syms: set[str] = set()
        try:
            u = json.loads(Path(str(holdings_path)).read_text(encoding="utf-8"))
            pos = u.get("positions") if isinstance(u, dict) else None
            pos = pos if isinstance(pos, list) else []
            for it in pos:
                if not isinstance(it, dict) or not bool(it.get("frozen")):
                    continue
                s = str(it.get("symbol") or "").strip()
                if s:
                    frozen_syms.add(s)
        except Exception:  # noqa: BLE001
            frozen_syms = set()

        deep_items: list[dict[str, Any]] = []
        deep_errors: list[str] = []

        try:
            hold_obj_d = json.loads(holdings_out.read_text(encoding="utf-8"))
            holds_d = hold_obj_d.get("holdings") if isinstance(hold_obj_d, dict) else None
            holds_d = holds_d if isinstance(holds_d, list) else []
        except Exception as exc:  # noqa: BLE001
            _record("deep_holdings.read_holdings_user", exc, note="读取 holdings_user.json 失败：deep-holdings 已跳过")
            holds_d = []

        for it in holds_d:
            if not isinstance(it, dict) or not bool(it.get("ok")):
                continue
            sym = str(it.get("symbol") or "").strip()
            asset = str(it.get("asset") or "").strip().lower() or "etf"
            name = str(it.get("name") or "").strip()
            if not sym:
                continue

            # 每个持仓一个目录（目录名=asset_symbol，避免潜在重名）
            sym_dir = deep_dir / f"{asset}_{sym}"
            sym_dir.mkdir(parents=True, exist_ok=True)

            argv_a = [
                "analyze",
                "--asset",
                str(asset),
                "--symbol",
                str(sym),
                "--method",
                "all",
                "--out-dir",
                str(sym_dir),
            ]

            # ETF：默认走 AkShare（复权/口径更友好）；stock：默认 auto（优先 TuShare）
            if asset == "etf":
                argv_a += ["--source", "akshare"]
            elif asset == "stock":
                argv_a += ["--source", "auto", "--adjust", str(stock_adjust)]

            try:
                _invoke(argv_a)
            except Exception as exc:  # noqa: BLE001
                msg = f"{asset}:{sym} analyze failed: {exc}"
                deep_errors.append(msg)
                _record("deep_holdings.analyze", exc, note=msg, dedupe_key=f"deep_holdings.analyze:{asset}:{sym}")
                continue

            # 聚合：只抽关键结论（别把 report.md 写成论文）
            decision_action = None
            decision_reasons: list[str] = []
            risk_signals = None
            try:
                sb = json.loads((sym_dir / "signal_backtest.json").read_text(encoding="utf-8"))
                if isinstance(sb, dict):
                    dec = sb.get("decision") if isinstance(sb.get("decision"), dict) else {}
                    decision_action = dec.get("action")
                    rs = dec.get("reasons") if isinstance(dec.get("reasons"), list) else []
                    decision_reasons = [str(x) for x in rs if str(x).strip()][:3]
                    risk_signals = sb.get("risk_signals") if isinstance(sb.get("risk_signals"), dict) else None
            except Exception:  # noqa: BLE001
                decision_action = None
                decision_reasons = []
                risk_signals = None

            opp_score = None
            opp_verdict = None
            try:
                opp = json.loads((sym_dir / "opportunity_score.json").read_text(encoding="utf-8"))
                if isinstance(opp, dict):
                    opp_score = opp.get("total_score")
                    opp_verdict = opp.get("verdict") or opp.get("bucket")
            except Exception:  # noqa: BLE001
                opp_score = None
                opp_verdict = None

            stops = it.get("stops") if isinstance(it.get("stops"), dict) else {}
            deep_items.append(
                {
                    "asset": asset,
                    "symbol": sym,
                    "name": name or None,
                    "status": it.get("status"),
                    "shares": it.get("shares"),
                    "close": it.get("close"),
                    "pnl_net_pct": it.get("pnl_net_pct"),
                    "effective_stop": stops.get("effective_stop"),
                    "effective_ref": stops.get("effective_ref"),
                    "frozen": bool(sym in frozen_syms),
                    "analyze_dir": str(sym_dir.relative_to(out_dir)),
                    "decision_action": decision_action,
                    "decision_reasons": decision_reasons,
                    "risk_signals": risk_signals,
                    "opportunity_score": opp_score,
                    "opportunity_verdict": opp_verdict,
                }
            )

        try:
            write_json(
                deep_summary_json,
                {
                    "schema": "llm_trading.holdings_deep_summary.v1",
                    "generated_at": datetime.now().isoformat(),
                    "as_of": str((json.loads(holdings_out.read_text(encoding="utf-8")) if holdings_out.exists() else {}).get("as_of") or ""),
                    "items": deep_items,
                    "errors": deep_errors[:200],
                },
            )
        except Exception as exc:  # noqa: BLE001
            _record("deep_holdings.write_summary_json", exc, note="写出 holdings_deep_summary.json 失败（已跳过）")

        try:
            lines = [
                "# holdings_deep\n",
                "",
                f"- generated_at: {datetime.now().isoformat()}",
                f"- holdings_asof: {holdings_asof2 or ''}",
                f"- items: {len(deep_items)}",
                "",
                "## items\n",
            ]
            for d in deep_items:
                if not isinstance(d, dict):
                    continue
                lines.extend(
                    [
                        f"- {d.get('asset')} {d.get('symbol')} {d.get('name') or ''} frozen={d.get('frozen')}",
                        f"  - holdings: status={d.get('status')} shares={d.get('shares')} close={d.get('close')} pnl_net_pct={d.get('pnl_net_pct')}",
                        f"  - stop: {d.get('effective_stop')} ({d.get('effective_ref') or ''})",
                        f"  - decision: {d.get('decision_action') or ''} reasons={'; '.join((d.get('decision_reasons') or []))}",
                        f"  - opportunity: score={d.get('opportunity_score')} verdict={d.get('opportunity_verdict')}",
                        f"  - dir: {d.get('analyze_dir')}",
                        "",
                    ]
                )

            if deep_errors:
                lines.extend(["## errors\n", ""] + [f"- {e}" for e in deep_errors[:50]] + [""])

            deep_report_md.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _record("deep_holdings.write_report_md", exc, note="写出 report_holdings.md 失败（已跳过）")

    # 4) report.md：把最关键信息写出来（KISS）
    as_of = ""
    sig_n = 0
    sig_left_n = 0
    sig_left_as_of = ""
    sig_left_strategy = None
    sig_stock_n = 0
    sig_stock_as_of = ""
    sig_stock_strategy = None
    sig_left_stock_n = 0
    sig_left_stock_as_of = ""
    sig_left_stock_strategy = None
    sig_strategy = None
    sig_merge = None
    ord_n = 0
    orders_next_open: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    alerts_counts: dict[str, int] = {"stop": 0, "take_profit": 0, "watch": 0, "hold": 0, "other": 0}
    portfolio_summary: dict[str, Any] = {}
    sig_cfg: dict[str, Any] = {}
    try:
        sig_obj = json.loads((out_dir / "signals.json").read_text(encoding="utf-8"))
        sig_n = int(len((sig_obj.get("items") if isinstance(sig_obj, dict) else None) or []))
        as_of = str(sig_obj.get("generated_at") or "")
        sig_strategy = (str(sig_obj.get("strategy") or "").strip() or None) if isinstance(sig_obj, dict) else None
        sig_cfg = sig_obj.get("config") if isinstance(sig_obj.get("config"), dict) else {}
        # 把 signals-merge 的关键信息透出到 report（方便你复盘“谁压过谁”）
        if isinstance(sig_cfg, dict) and isinstance(sig_cfg.get("merge"), dict):
            m = sig_cfg.get("merge") if isinstance(sig_cfg.get("merge"), dict) else {}
            sig_merge = {
                "conflict": m.get("conflict"),
                "weights": m.get("weights"),
                "priority": m.get("priority"),
                "conflicts": (list((sig_cfg.get("conflicts") or {}).keys()) if isinstance(sig_cfg.get("conflicts"), dict) else None),
            }
    except (AttributeError) as exc:  # noqa: BLE001
        _record("read_signals_for_report", exc, note="读取 signals.json 失败（report 将缺少 signals 摘要）")

    # 可选：左侧低吸候选池（不影响 orders）
    try:
        p_left = out_dir / "signals_left.json"
        if p_left.exists():
            sig_obj_l = json.loads(p_left.read_text(encoding="utf-8"))
            if isinstance(sig_obj_l, dict):
                sig_left_n = int(len((sig_obj_l.get("items") if isinstance(sig_obj_l.get("items"), list) else []) or []))
                sig_left_as_of = str(sig_obj_l.get("as_of") or sig_obj_l.get("generated_at") or "")
                sig_left_strategy = (str(sig_obj_l.get("strategy") or "").strip() or None) if isinstance(sig_obj_l, dict) else None
    except Exception as exc:  # noqa: BLE001
        _record("read_signals_left_for_report", exc, note="读取 signals_left.json 失败（report 将缺少 left 摘要）")

    # 可选：stock 候选池（不影响 orders）
    try:
        p_stock = out_dir / "signals_stock.json"
        if p_stock.exists():
            sig_obj_s = json.loads(p_stock.read_text(encoding="utf-8"))
            if isinstance(sig_obj_s, dict):
                sig_stock_n = int(len((sig_obj_s.get("items") if isinstance(sig_obj_s.get("items"), list) else []) or []))
                sig_stock_as_of = str(sig_obj_s.get("as_of") or sig_obj_s.get("generated_at") or "")
                sig_stock_strategy = (str(sig_obj_s.get("strategy") or "").strip() or None) if isinstance(sig_obj_s, dict) else None
    except Exception as exc:  # noqa: BLE001
        _record("read_signals_stock_for_report", exc, note="读取 signals_stock.json 失败（report 将缺少 stock 摘要）")

    # 可选：stock 左侧低吸候选池（不影响 orders）
    try:
        p_left_stock = out_dir / "signals_left_stock.json"
        if p_left_stock.exists():
            sig_obj_ls = json.loads(p_left_stock.read_text(encoding="utf-8"))
            if isinstance(sig_obj_ls, dict):
                sig_left_stock_n = int(len((sig_obj_ls.get("items") if isinstance(sig_obj_ls.get("items"), list) else []) or []))
                sig_left_stock_as_of = str(sig_obj_ls.get("as_of") or sig_obj_ls.get("generated_at") or "")
                sig_left_stock_strategy = (str(sig_obj_ls.get("strategy") or "").strip() or None) if isinstance(sig_obj_ls, dict) else None
    except Exception as exc:  # noqa: BLE001
        _record("read_signals_left_stock_for_report", exc, note="读取 signals_left_stock.json 失败（report 将缺少 stock_left 摘要）")

    # signals-merge 如果发现 config 冲突，会写到 config.conflicts（这里不瞎猜，直接提示你手动统一参数）
    try:
        if isinstance(sig_cfg, dict) and isinstance(sig_cfg.get("conflicts"), dict) and sig_cfg.get("conflicts"):
            ks = ",".join(sorted([str(k) for k in (sig_cfg.get("conflicts") or {}).keys()]))
            run_warnings.append(f"signals.config 有冲突字段：{ks}（建议用 CLI 覆盖或统一 signals 生成参数）")
    except (AttributeError) as exc:  # noqa: BLE001
        _record("signals_config.conflicts_check", exc, note="signals.config.conflicts 检查失败（将忽略该提示）")

    # 成本/约束：默认继承 signals.config（如果 signals 缺字段就按 0 处理；别让我瞎猜你券商费率）
    try:
        cost_rt = float(sig_cfg.get("roundtrip_cost_yuan") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.roundtrip_cost_yuan",
            exc,
            note=f"signals.config.roundtrip_cost_yuan 非法，已按 0 处理：{sig_cfg.get('roundtrip_cost_yuan')!r}",
        )
        cost_rt = 0.0
    try:
        cost_min_fee = float(sig_cfg.get("min_fee_yuan") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.min_fee_yuan",
            exc,
            note=f"signals.config.min_fee_yuan 非法，已按 0 处理：{sig_cfg.get('min_fee_yuan')!r}",
        )
        cost_min_fee = 0.0
    try:
        cost_buy = float(sig_cfg.get("buy_cost") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.buy_cost",
            exc,
            note=f"signals.config.buy_cost 非法，已按 0 处理：{sig_cfg.get('buy_cost')!r}",
        )
        cost_buy = 0.0
    try:
        cost_sell = float(sig_cfg.get("sell_cost") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.sell_cost",
            exc,
            note=f"signals.config.sell_cost 非法，已按 0 处理：{sig_cfg.get('sell_cost')!r}",
        )
        cost_sell = 0.0

    slip_mode = str(sig_cfg.get("slippage_mode") or "none").strip().lower() or "none"
    try:
        slip_bps = float(sig_cfg.get("slippage_bps") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_bps",
            exc,
            note=f"signals.config.slippage_bps 非法，已按 0 处理：{sig_cfg.get('slippage_bps')!r}",
        )
        slip_bps = 0.0
    try:
        slip_ref_amt = float(sig_cfg.get("slippage_ref_amount_yuan") or 1e8)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_ref_amount_yuan",
            exc,
            note=f"signals.config.slippage_ref_amount_yuan 非法，已按 1e8 处理：{sig_cfg.get('slippage_ref_amount_yuan')!r}",
        )
        slip_ref_amt = 1e8
    try:
        slip_bps_min = float(sig_cfg.get("slippage_bps_min") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_bps_min",
            exc,
            note=f"signals.config.slippage_bps_min 非法，已按 0 处理：{sig_cfg.get('slippage_bps_min')!r}",
        )
        slip_bps_min = 0.0
    try:
        slip_bps_max = float(sig_cfg.get("slippage_bps_max") or 30.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_bps_max",
            exc,
            note=f"signals.config.slippage_bps_max 非法，已按 30 处理：{sig_cfg.get('slippage_bps_max')!r}",
        )
        slip_bps_max = 30.0
    try:
        slip_unknown_bps = float(sig_cfg.get("slippage_unknown_bps") or 10.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_unknown_bps",
            exc,
            note=f"signals.config.slippage_unknown_bps 非法，已按 10 处理：{sig_cfg.get('slippage_unknown_bps')!r}",
        )
        slip_unknown_bps = 10.0
    try:
        slip_vm = float(sig_cfg.get("slippage_vol_mult") or 0.0)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        _record(
            "signals_config.slippage_vol_mult",
            exc,
            note=f"signals.config.slippage_vol_mult 非法，已按 0 处理：{sig_cfg.get('slippage_vol_mult')!r}",
        )
        slip_vm = 0.0

    cost_base = trade_cost_from_params(roundtrip_cost_yuan=float(cost_rt), min_fee_yuan=float(cost_min_fee), buy_cost=float(cost_buy), sell_cost=float(cost_sell))
    slip_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _slip_for(*, asset: str, sym: str) -> dict[str, Any]:
        k = (str(asset), str(sym))
        if k in slip_cache:
            return slip_cache[k]
        amt_avg20 = None
        tb = None
        try:
            adj = None if asset == "etf" else str(stock_adjust or "qfq").strip() or "qfq"
            df = fetch_daily_cached(
                FetchParams(asset=str(asset), symbol=str(sym), adjust=adj),
                cache_dir=Path("data") / "cache" / str(asset),
                ttl_hours=float(cache_ttl_hours),
            )
            if df is not None and (not getattr(df, "empty", True)):
                import pandas as pd

                dfd = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                if not dfd.empty:
                    close_s = pd.to_numeric(dfd["close"], errors="coerce").astype(float)
                    amount_s = None
                    if "amount" in dfd.columns:
                        amount_s = pd.to_numeric(dfd["amount"], errors="coerce").astype(float)
                    elif "volume" in dfd.columns:
                        vol_s = pd.to_numeric(dfd["volume"], errors="coerce").astype(float)
                        amount_s = close_s * vol_s
                    if amount_s is not None:
                        v = amount_s.rolling(window=20, min_periods=20).mean().iloc[-1]
                        amt_avg20 = None if v is None else float(v)

                    # tradeability（最后一根已知日线；非预测）
                    last = dfd.iloc[-1]
                    prev = dfd.iloc[-2] if len(dfd) >= 2 else None
                    prev_close = None if prev is None else prev.get("close")
                    close_last = last.get("close")
                    op = last.get("open") if "open" in dfd.columns else close_last
                    hp = last.get("high") if "high" in dfd.columns else close_last
                    lp = last.get("low") if "low" in dfd.columns else close_last
                    vol_last = last.get("volume") if "volume" in dfd.columns else None
                    amt_last = last.get("amount") if "amount" in dfd.columns else None
                    if amt_last is None and vol_last is not None and close_last is not None:
                        try:
                            amt_last = float(close_last) * float(vol_last)
                        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                            amt_last = None
                    dt = last.get("date")
                    dt_s = str(dt.date()) if hasattr(dt, "date") else str(dt)
                    tb = {
                        "ref_date": dt_s,
                        "flags": tradeability_flags(
                            open_price=(None if op is None else float(op)),
                            high_price=(None if hp is None else float(hp)),
                            low_price=(None if lp is None else float(lp)),
                            prev_close=(None if prev_close is None else float(prev_close)),
                            volume=(None if vol_last is None else float(vol_last)),
                            amount=(None if amt_last is None else float(amt_last)),
                            cfg=tb_cfg,
                        ),
                    }
        except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
            _record(
                "slippage_snapshot",
                exc,
                note=f"计算滑点/可交易性快照失败：{asset}:{sym}（将回退为 unknown_slippage + 无 tradeability_last_bar）",
                dedupe_key=f"slippage_snapshot:{asset}:{sym}",
            )
            amt_avg20 = None
            tb = None

        slip_on = str(slip_mode) not in {"", "none", "off", "0", "false"}
        slip_bps2 = (
            estimate_slippage_bps(
                mode=str(slip_mode),
                amount_avg20_yuan=(float(amt_avg20) if amt_avg20 is not None else None),
                atr_pct=None,
                bps=float(slip_bps),
                ref_amount_yuan=float(slip_ref_amt),
                min_bps=float(slip_bps_min),
                max_bps=float(slip_bps_max),
                unknown_bps=float(slip_unknown_bps),
                vol_mult=float(slip_vm),
            )
            if slip_on
            else 0.0
        )
        out = {
            "slippage_mode": (str(slip_mode) if slip_on else "none"),
            "slippage_bps": float(slip_bps2),
            "slippage_rate": float(bps_to_rate(float(slip_bps2))),
            "amount_avg20_yuan": (float(amt_avg20) if amt_avg20 is not None else None),
            "tradeability_last_bar": tb,
        }
        slip_cache[k] = out
        return out

    def _cost_for(*, asset: str, sym: str) -> tuple[TradeCost, dict[str, Any]]:
        slip = _slip_for(asset=str(asset), sym=str(sym))
        r = float(slip.get("slippage_rate") or 0.0)
        c = TradeCost(
            buy_cost=float(cost_base.buy_cost) + float(r),
            sell_cost=float(cost_base.sell_cost) + float(r),
            buy_fee_yuan=float(cost_base.buy_fee_yuan),
            sell_fee_yuan=float(cost_base.sell_fee_yuan),
            buy_fee_min_yuan=float(cost_base.buy_fee_min_yuan),
            sell_fee_min_yuan=float(cost_base.sell_fee_min_yuan),
        )
        return c, slip

    # 4.1) alerts：从 holdings-user 抽“止损/止盈/观察”提示（执行层要看的就是这玩意）
    orders_from_holdings: list[dict[str, Any]] = []
    holdings_asof = None
    try:
        hold_obj = json.loads(holdings_out.read_text(encoding="utf-8"))
        items = hold_obj.get("holdings") if isinstance(hold_obj, dict) else None
        items = items if isinstance(items, list) else []
        portfolio_summary = hold_obj.get("portfolio") if isinstance(hold_obj.get("portfolio"), dict) else {}

        for it in items:
            if not isinstance(it, dict) or not bool(it.get("ok")):
                continue
            st = str(it.get("status") or "").strip()
            if not st:
                st = "other"
            alerts_counts[st] = int(alerts_counts.get(st, 0)) + 1

            asof2 = str(it.get("asof") or "").strip()
            if asof2:
                holdings_asof = asof2 if holdings_asof is None else max(str(holdings_asof), asof2)

            if st not in {"stop", "take_profit", "watch"}:
                continue

            sym = str(it.get("symbol") or "").strip()
            name = str(it.get("name") or "").strip()
            asset = str(it.get("asset") or "").strip().lower()
            shares = int(it.get("shares") or 0)
            close = it.get("close")
            try:
                close2 = float(close) if close is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                close2 = None

            pnl_pct = it.get("pnl_net_pct")
            try:
                pnl_pct2 = float(pnl_pct) if pnl_pct is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pnl_pct2 = None

            stops = it.get("stops") if isinstance(it.get("stops"), dict) else {}
            eff = stops.get("effective_stop")
            eff_ref = stops.get("effective_ref")

            tp = it.get("take_profit") if isinstance(it.get("take_profit"), dict) else {}
            tp_plan = tp.get("plan")
            tp_sell_shares = tp.get("sell_shares")
            try:
                tp_sell_shares2 = int(tp_sell_shares) if tp_sell_shares is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                tp_sell_shares2 = None

            severity = "high" if st in {"stop", "take_profit"} else "medium"
            alerts.append(
                {
                    "severity": severity,
                    "status": st,
                    "asset": asset,
                    "symbol": sym,
                    "name": name,
                    "asof": asof2 or None,
                    "close": close2,
                    "pnl_net_pct": pnl_pct2,
                    "effective_stop": eff,
                    "effective_ref": eff_ref,
                    "take_profit_plan": tp_plan,
                    "take_profit_sell_shares": tp_sell_shares2,
                }
            )

            # orders：执行层给“次日开盘”参考单（研究用途；你自己下单前再校验一遍）
            if sym and shares > 0 and st in {"stop", "take_profit"}:
                sell_sh = int(tp_sell_shares2) if (st == "take_profit" and tp_sell_shares2 and tp_sell_shares2 > 0) else int(shares)
                orders_from_holdings.append(
                    {
                        "side": "sell",
                        "asset": asset or "etf",
                        "symbol": sym,
                        "name": name or None,
                        "shares": int(sell_sh),
                        "signal_date": asof2 or "",
                        "exec": "next_open",
                        "price_ref": float(close2) if close2 is not None else None,
                        "price_ref_type": "close",
                        "order_type": "market",
                        "limit_price": None,
                        "est_cash": None,
                        "est_fee_yuan": None,
                        "reason": f"holdings: status={st} plan={tp_plan} eff_ref={eff_ref}",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        _record(
            "parse_holdings_user_alerts",
            exc,
            note="读取/解析 holdings_user.json 失败：alerts 与 holdings 风险单将为空（不影响 rebalance 主流程）",
        )

    try:
        write_json(out_dir / "alerts.json", {"generated_at": datetime.now().isoformat(), "asof": holdings_asof, "items": alerts})
    except (AttributeError) as exc:  # noqa: BLE001
        _record("write_alerts_json", exc, note="写出 alerts.json 失败（不影响 orders/report 主流程）")

    try:
        reb_obj = json.loads(rebalance_out.read_text(encoding="utf-8"))
        orders_reb = ((reb_obj.get("rebalance") if isinstance(reb_obj, dict) else None) or {}).get("orders_next_open")  # type: ignore[union-attr]
        orders_reb = orders_reb if isinstance(orders_reb, list) else []

        # 调仓执行窗：只限制 rebalance 单（risk 单=止损/止盈信号，任何一天都允许输出）
        if str(reb_schedule) == "fri_close_mon_open":
            # 约定：周五收盘后出单 => 周一开盘执行。
            # 实操上你周末跑也很正常，所以这里不再死板地“只允许周五”：
            # - 优先用 holdings.as_of 作为“最近交易日”参考（更贴近行情数据口径）
            # - 周末（周六/周日）默认放行（方便提前准备周一开盘单）
            now_wd = datetime.now().weekday()  # Mon=0 ... Sun=6
            ref_wd = None
            try:
                if isinstance(holdings_asof, str) and holdings_asof.strip():
                    ref_wd = datetime.fromisoformat(holdings_asof.strip()).weekday()
            except (TypeError, ValueError):  # noqa: BLE001
                ref_wd = None

            allowed = False
            if now_wd in {5, 6}:  # weekend
                allowed = True
            else:
                allowed = (ref_wd == 4) if ref_wd is not None else (now_wd == 4)

            if not allowed:
                run_warnings.append(
                    f"rebalance blocked by schedule=fri_close_mon_open（as_of={holdings_asof or 'unknown'}；只保留 holdings 风险单，已清空 rebalance 单）"
                )
                orders_reb = []
        # 把 rebalance 的 warnings 透出到 run 报告里（否则你只看 report.md 会一脸懵）
        try:
            reb_w = reb_obj.get("warnings") if isinstance(reb_obj, dict) else None
            reb_w = reb_w if isinstance(reb_w, list) else []
            for w in reb_w[:50]:
                if isinstance(w, str) and w.strip():
                    run_warnings.append(f"rebalance: {w.strip()}")
        except (AttributeError) as exc:  # noqa: BLE001
            _record("rebalance_user.warnings_extract", exc, note="读取 rebalance_user.warnings 失败（将忽略该提示）")

        # 合并 orders：先 risk(holdings) 再 rebalance；冲突（先卖后买）直接丢掉买单（保命优先）
        orders_next_open = merge_orders_next_open(orders_from_holdings, orders_reb, warnings=run_warnings)

        # 纯本地补齐（不依赖行情/数据拉取）
        basic_enrich_orders_next_open(
            orders_next_open,
            buy_cost=float(cost_buy),
            sell_cost=float(cost_sell),
            min_fee_yuan=float(cost_min_fee),
        )

        # 成本/滑点/可交易性估算：用 callback 注入（估算失败不影响订单输出，但必须可见）
        def _estimator(o: dict[str, Any]) -> dict[str, Any]:
            side = str(o.get("side") or "").strip().lower()
            asset = str(o.get("asset") or "").strip().lower() or "etf"
            sym = str(o.get("symbol") or "").strip()
            sh = int(o.get("shares") or 0)
            px = float(o.get("price_ref"))

            cost2, slip2 = _cost_for(asset=asset, sym=sym)
            if side == "buy":
                cash_v, fee_v = cash_buy(shares=int(sh), price=float(px), cost=cost2)
            else:
                cash_v, fee_v = cash_sell(shares=int(sh), price=float(px), cost=cost2)

            tb = (slip2.get("tradeability_last_bar") if isinstance(slip2, dict) else None)
            flags = tb.get("flags") if isinstance(tb, dict) else {}

            out: dict[str, Any] = {
                "slippage": slip2,
                "tradeability_last_bar": tb,
                "halt_risk": bool(flags.get("halted")) if isinstance(flags, dict) else None,
                "limit_up_risk": (bool(flags.get("locked_limit_up")) if isinstance(flags, dict) else None) if side == "buy" else None,
                "limit_down_risk": (bool(flags.get("locked_limit_down")) if isinstance(flags, dict) else None) if side == "sell" else None,
                "est_fee_yuan": float(fee_v),
                "est_cash": float(cash_v),
            }
            return out

        est_errors = apply_order_estimates(orders_next_open, estimator=_estimator)
        for e in est_errors[:200]:
            side = str(e.get("side") or "")
            asset = str(e.get("asset") or "")
            sym = str(e.get("symbol") or "")
            sh = e.get("shares")
            px = e.get("price_ref")
            _record(
                "estimate_order_cost",
                RuntimeError(str(e.get("error") or "")),
                note=f"订单成本/滑点估算失败：side={side} {asset}:{sym} shares={sh} price_ref={px}（订单仍输出但缺 est_cash/est_fee/slippage）",
                dedupe_key=f"estimate_order_cost:{side}:{asset}:{sym}",
            )

        sort_orders_next_open(orders_next_open)
        ord_n = int(len(orders_next_open))
    except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
        _record(
            "parse_rebalance_user_orders",
            exc,
            note="读取/解析 rebalance_user.json 失败：orders_next_open 将仅包含 holdings 风险单（或为空）",
        )

    try:
        write_json(out_dir / "orders_next_open.json", {"generated_at": datetime.now().isoformat(), "orders": orders_next_open, "warnings": run_warnings})
    except (AttributeError) as exc:  # noqa: BLE001
        _record("write_orders_next_open_json", exc, note="写出 orders_next_open.json 失败（建议检查 outputs 目录权限/磁盘空间）")

    try:
        pf = portfolio_summary if isinstance(portfolio_summary, dict) else {}
        eq = pf.get("equity_yuan")
        exp = pf.get("exposure_pct")
        risk2 = pf.get("risk_to_stop_yuan")
        pf_warn = pf.get("warnings") if isinstance(pf.get("warnings"), list) else []

        nt_pct = None
        nt_used = None
        nt_warn2: list[str] = []
        try:
            nt_obj = json.loads((out_dir / "national_team.json").read_text(encoding="utf-8"))
            s = nt_obj.get("score") if isinstance(nt_obj, dict) else None
            s = s if isinstance(s, dict) else {}
            nt_pct = s.get("composite_pct")
            used = s.get("used") if isinstance(s.get("used"), dict) else {}
            if used:
                # 只把“参与打分的项”抖出来；别让 report 太啰嗦
                parts = []
                for k in sorted(list(used.keys())):
                    it = used.get(k) if isinstance(used, dict) else None
                    if not isinstance(it, dict):
                        continue
                    parts.append(f"{k}={it.get('score01')}")
                nt_used = ",".join(parts) if parts else None
            nt_warn2 = [str(x) for x in (nt_obj.get("warnings") or [])[:10]] if isinstance(nt_obj, dict) else []
        except (AttributeError) as exc:  # noqa: BLE001
            _record("read_national_team_for_report", exc, note="读取 national_team.json 失败（report 将不展示 national_team_proxy）")
            nt_pct = None
            nt_used = None
            nt_warn2 = []

        # signals Top：把“候选清单”直接写进 report（否则你得另开 json 才知道值得看啥）
        sig_top_lines: list[str] = []
        try:
            sig_obj3 = json.loads((out_dir / "signals.json").read_text(encoding="utf-8"))
            items3 = sig_obj3.get("items") if isinstance(sig_obj3, dict) else None
            items3 = items3 if isinstance(items3, list) else []

            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    return None

            def _score(it: dict) -> float:
                v = _f(it.get("score"))
                return float(v) if v is not None else 0.0

            items3s = sorted([it for it in items3 if isinstance(it, dict)], key=_score, reverse=True)
            for it in items3s[:15]:
                sym = str(it.get("symbol") or "").strip()
                name = str(it.get("name") or "").strip()
                act = str(it.get("action") or "").strip()
                sc = _score(it)
                cf = _f(it.get("confidence"))
                meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                entry = it.get("entry") if isinstance(it.get("entry"), dict) else {}
                close = None
                if isinstance(meta, dict):
                    close = meta.get("close")
                if close is None and isinstance(entry, dict):
                    close = entry.get("price_ref")
                sig_top_lines.append(
                    f"- {act} {sym} {name} score={sc:.3f} conf={(cf if cf is not None else '')} close={close}"
                )
        except Exception as exc:  # noqa: BLE001
            _record("read_signals_top_for_report", exc, note="读取 signals Top 失败（report 将不展示候选清单）")
            sig_top_lines = []

        # 可选：左侧低吸候选池（不影响 orders，只给你“试错清单”）
        sig_top_left_lines: list[str] = []
        try:
            sp_left = out_dir / "signals_left.json"
            if sp_left.exists():
                sig_obj_l = json.loads(sp_left.read_text(encoding="utf-8"))
                items_l = sig_obj_l.get("items") if isinstance(sig_obj_l, dict) else None
                items_l = items_l if isinstance(items_l, list) else []

                def _f(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        return None

                def _score(it: dict) -> float:
                    v = _f(it.get("score"))
                    return float(v) if v is not None else 0.0

                items_ls = sorted([it for it in items_l if isinstance(it, dict)], key=_score, reverse=True)
                for it in items_ls[:15]:
                    sym = str(it.get("symbol") or "").strip()
                    name = str(it.get("name") or "").strip()
                    act = str(it.get("action") or "").strip()
                    sc = _score(it)
                    cf = _f(it.get("confidence"))
                    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                    entry = it.get("entry") if isinstance(it.get("entry"), dict) else {}
                    close = None
                    if isinstance(meta, dict):
                        close = meta.get("close")
                    if close is None and isinstance(entry, dict):
                        close = entry.get("price_ref")
                    sig_top_left_lines.append(
                        f"- {act} {sym} {name} score={sc:.3f} conf={(cf if cf is not None else '')} close={close}"
                    )
        except Exception as exc:  # noqa: BLE001
            _record("read_signals_left_top_for_report", exc, note="读取 signals_left Top 失败（report 将不展示 left 候选）")
            sig_top_left_lines = []

        # 可选：stock 候选池（不影响 orders，只给你“看盘清单”）
        sig_top_stock_lines: list[str] = []
        try:
            sp = out_dir / "signals_stock.json"
            if sp.exists():
                sig_obj_s = json.loads(sp.read_text(encoding="utf-8"))
                items_s = sig_obj_s.get("items") if isinstance(sig_obj_s, dict) else None
                items_s = items_s if isinstance(items_s, list) else []

                def _f(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        return None

                def _score(it: dict) -> float:
                    v = _f(it.get("score"))
                    return float(v) if v is not None else 0.0

                items_ss = sorted([it for it in items_s if isinstance(it, dict)], key=_score, reverse=True)
                for it in items_ss[:15]:
                    sym = str(it.get("symbol") or "").strip()
                    name = str(it.get("name") or "").strip()
                    act = str(it.get("action") or "").strip()
                    sc = _score(it)
                    cf = _f(it.get("confidence"))
                    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                    entry = it.get("entry") if isinstance(it.get("entry"), dict) else {}
                    close = None
                    if isinstance(meta, dict):
                        close = meta.get("close")
                    if close is None and isinstance(entry, dict):
                        close = entry.get("price_ref")
                    sig_top_stock_lines.append(
                        f"- {act} {sym} {name} score={sc:.3f} conf={(cf if cf is not None else '')} close={close}"
                    )
        except Exception as exc:  # noqa: BLE001
            _record("read_signals_stock_top_for_report", exc, note="读取 signals_stock Top 失败（report 将不展示 stock 候选）")
            sig_top_stock_lines = []

        # 可选：stock 左侧低吸候选池（不影响 orders，只给你“高赔率试错清单”）
        sig_top_left_stock_lines: list[str] = []
        try:
            sp = out_dir / "signals_left_stock.json"
            if sp.exists():
                sig_obj_s = json.loads(sp.read_text(encoding="utf-8"))
                items_s = sig_obj_s.get("items") if isinstance(sig_obj_s, dict) else None
                items_s = items_s if isinstance(items_s, list) else []

                def _f(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        return None

                def _score(it: dict) -> float:
                    v = _f(it.get("score"))
                    return float(v) if v is not None else 0.0

                items_ss = sorted([it for it in items_s if isinstance(it, dict)], key=_score, reverse=True)
                for it in items_ss[:15]:
                    sym = str(it.get("symbol") or "").strip()
                    name = str(it.get("name") or "").strip()
                    act = str(it.get("action") or "").strip()
                    sc = _score(it)
                    cf = _f(it.get("confidence"))
                    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                    entry = it.get("entry") if isinstance(it.get("entry"), dict) else {}
                    close = None
                    if isinstance(meta, dict):
                        close = meta.get("close")
                    if close is None and isinstance(entry, dict):
                        close = entry.get("price_ref")
                    sig_top_left_stock_lines.append(
                        f"- {act} {sym} {name} score={sc:.3f} conf={(cf if cf is not None else '')} close={close}"
                    )
        except Exception as exc:  # noqa: BLE001
            _record("read_signals_left_stock_top_for_report", exc, note="读取 signals_left_stock Top 失败（report 将不展示 stock_left 候选）")
            sig_top_left_stock_lines = []

        # position_plan：把“目标仓位/止损口径”写出来（即便 orders=0，也要让你看见“系统想买啥、为啥买不了”）
        plan_lines: list[str] = []
        blocker_lines: list[str] = []
        try:
            reb_obj3 = json.loads(rebalance_out.read_text(encoding="utf-8"))
            pp3 = reb_obj3.get("position_plan") if isinstance(reb_obj3, dict) else None
            pp3 = pp3 if isinstance(pp3, dict) else {}
            plans3 = pp3.get("plans") if isinstance(pp3.get("plans"), list) else []
            watch3 = pp3.get("watch") if isinstance(pp3.get("watch"), list) else []
            for p in plans3[:10]:
                if not isinstance(p, dict):
                    continue
                plan_lines.append(
                    f"- {p.get('symbol')} {p.get('name') or ''} entry={p.get('entry')} stop={p.get('stop')}({p.get('stop_ref') or ''}) shares={p.get('shares')} pos_yuan={p.get('position_yuan')} mode={p.get('stop_mode')}"
                )
            if (not plan_lines) and watch3:
                for w in watch3[:10]:
                    if not isinstance(w, dict):
                        continue
                    plan_lines.append(
                        f"- watch {w.get('symbol')} {w.get('name') or ''} reason={w.get('reason') or ''} entry={w.get('entry')} stop={w.get('stop')}({w.get('stop_ref') or ''})"
                    )

            # orders_next_open=0 时，把“阻塞原因”讲人话：别逼你去翻一堆 warnings 猜。
            if int(ord_n) == 0 and plans3:
                acc = reb_obj3.get("account") if isinstance(reb_obj3, dict) else {}
                acc = acc if isinstance(acc, dict) else {}
                cash_yuan = acc.get("cash_yuan")

                inp = reb_obj3.get("inputs") if isinstance(reb_obj3, dict) else {}
                inp = inp if isinstance(inp, dict) else {}
                cons = inp.get("constraints") if isinstance(inp.get("constraints"), dict) else {}
                min_trade = cons.get("min_trade_notional_yuan")

                reb2 = reb_obj3.get("rebalance") if isinstance(reb_obj3, dict) else {}
                reb2 = reb2 if isinstance(reb2, dict) else {}
                exb = reb2.get("exposure_buy") if isinstance(reb2.get("exposure_buy"), dict) else {}
                max_expo = exb.get("max_exposure_pct")
                budget_yuan = exb.get("budget_yuan")
                cur_mv = exb.get("current_positions_mv_yuan")
                max_pos_yuan = exb.get("max_positions_yuan")

                # 1) 关键数字（可审计）
                blocker_lines.append(
                    f"- exposure_buy: budget≈{budget_yuan} (max_exposure_pct={max_expo}; current_mv≈{cur_mv}; max_mv≈{max_pos_yuan})"
                )
                blocker_lines.append(f"- cash_yuan: {cash_yuan} ; min_trade_notional_yuan: {min_trade}")

                # 2) 主要阻塞结论
                try:
                    if (budget_yuan is not None) and (min_trade is not None) and float(budget_yuan) < float(min_trade):
                        blocker_lines.append("- blocker: exposure_buy_budget < min_trade_notional（max_exposure_pct 留现金 => 本次预算太小）")
                    elif (cash_yuan is not None) and (min_trade is not None) and float(cash_yuan) < float(min_trade):
                        blocker_lines.append("- blocker: cash < min_trade_notional（现金不够，别硬上）")
                except Exception:  # noqa: BLE001
                    pass

                # 3) 可执行解法（不构成建议，只是解释系统怎么动）
                blocker_lines.append("- hint: 想强行出单=>提高 --max-exposure-pct 或改 --rebalance-mode rotate；否则就接受系统留现金。")
        except Exception as exc:  # noqa: BLE001
            _record("read_rebalance_position_plan_for_report", exc, note="读取 rebalance_user.position_plan 失败（report 将不展示仓位计划）")
            plan_lines = []
            blocker_lines = []

        lines = [
            "# run\n",
            "",
            f"- generated_at: {datetime.now().isoformat()}",
            f"- as_of(signals.generated_at): {as_of}",
            f"- as_of(holdings.asof): {holdings_asof or ''}",
            f"- signals.items: {sig_n}",
            f"- signals_left.items: {sig_left_n}",
            *([f"- as_of(signals_left.as_of): {sig_left_as_of}"] if int(sig_left_n) > 0 and str(sig_left_as_of).strip() else []),
            f"- signals_stock.items: {sig_stock_n}",
            *([f"- as_of(signals_stock.as_of): {sig_stock_as_of}"] if int(sig_stock_n) > 0 and str(sig_stock_as_of).strip() else []),
            f"- signals_left_stock.items: {sig_left_stock_n}",
            *(
                [f"- as_of(signals_left_stock.as_of): {sig_left_stock_as_of}"]
                if int(sig_left_stock_n) > 0 and str(sig_left_stock_as_of).strip()
                else []
            ),
            f"- orders_next_open: {ord_n}",
            f"- mode: {reb_mode}",
            f"- rebalance_schedule: {reb_schedule or 'any_day'}",
            f"- regime_index: {regime_index}",
            f"- equity_yuan: {eq}",
            f"- exposure_pct: {exp}",
            f"- risk_to_stop_yuan: {risk2}",
            "",
            "## national_team_proxy\n",
            f"- score_pct: {nt_pct}",
            f"- used: {nt_used or ''}",
            *([f"- warn: {w}" for w in nt_warn2] if nt_warn2 else []),
            "",
            "## alerts\n",
            f"- stop: {alerts_counts.get('stop', 0)}",
            f"- take_profit: {alerts_counts.get('take_profit', 0)}",
            f"- watch: {alerts_counts.get('watch', 0)}",
            "",
            *[
                f"- {a.get('status')} {a.get('symbol')} {a.get('name') or ''} close={a.get('close')} pnl_pct={a.get('pnl_net_pct')} eff={a.get('effective_stop')} {a.get('effective_ref') or ''} plan={a.get('take_profit_plan') or ''}"
                for a in alerts[:20]
                if isinstance(a, dict)
            ],
            "",
            "## signals_top\n",
            *sig_top_lines,
            "",
            "## signals_top_left\n",
            *sig_top_left_lines,
            "",
            "## signals_top_stock\n",
            *sig_top_stock_lines,
            "",
            "## signals_top_left_stock\n",
            *sig_top_left_stock_lines,
            "",
            "## position_plan\n",
            *plan_lines,
            "",
            "## blockers\n",
            *blocker_lines,
            "",
            "## orders_next_open\n",
            *[
                f"- {o.get('side')} {o.get('asset')} {o.get('symbol')} shares={o.get('shares')} reason={o.get('reason')}"
                for o in orders_next_open[:30]
                if isinstance(o, dict)
            ],
            "",
            "## warnings\n",
            *[f"- {w}" for w in (run_warnings + pf_warn)[:50]],
            "",
            "产物：",
            f"- signals: signals.json",
            *([f"- signals_left: signals_left.json"] if int(sig_left_n) > 0 else []),
            *([f"- signals_stock: signals_stock.json"] if int(sig_stock_n) > 0 else []),
            *([f"- signals_left_stock: signals_left_stock.json"] if int(sig_left_stock_n) > 0 else []),
            f"- holdings: holdings_user.json",
            *([f"- holdings_deep: report_holdings.md"] if (out_dir / "report_holdings.md").exists() else []),
            *([f"- holdings_deep_summary: holdings_deep_summary.json"] if (out_dir / "holdings_deep_summary.json").exists() else []),
            f"- rebalance/orders: rebalance_user.json",
            f"- national_team: national_team.json",
            f"- alerts: alerts.json",
            f"- orders_next_open: orders_next_open.json",
            f"- scan raw: scan_etf/",
            *([f"- scan raw left: scan_strategy_left/"] if int(sig_left_n) > 0 else []),
            *([f"- scan raw stock: scan_strategy_stock/"] if int(sig_stock_n) > 0 else []),
            *([f"- scan raw stock left: scan_strategy_stock_left/"] if int(sig_left_stock_n) > 0 else []),
            "",
            "免责声明：研究工具输出，不构成投资建议；买卖自负。",
            "",
        ]
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        _record("write_report_md", exc, note="写出 report.md 失败（不影响 orders/report.json 主流程）")

    data_hash = None
    try:
        from ..analysis_cache import compute_params_hash

        sig_fp = None
        try:
            sig_obj2 = json.loads((out_dir / "signals.json").read_text(encoding="utf-8"))
            if isinstance(sig_obj2, dict):
                syms = []
                for it in (sig_obj2.get("items") if isinstance(sig_obj2.get("items"), list) else []):
                    if not isinstance(it, dict):
                        continue
                    s = str(it.get("symbol") or "").strip()
                    if s:
                        syms.append(s)
                sig_fp = {"strategy": sig_obj2.get("strategy"), "as_of": sig_obj2.get("as_of"), "generated_at": sig_obj2.get("generated_at"), "symbols": sorted(set(syms))}
        except (AttributeError) as exc:  # noqa: BLE001
            _record("data_hash.read_signals", exc, note="data_hash 读取 signals.json 失败（将忽略该指纹）")
            sig_fp = None

        hold_fp = None
        try:
            hold_obj2 = json.loads((out_dir / "holdings_user.json").read_text(encoding="utf-8"))
            if isinstance(hold_obj2, dict):
                syms = []
                for it in (hold_obj2.get("holdings") if isinstance(hold_obj2.get("holdings"), list) else []):
                    if not isinstance(it, dict) or not bool(it.get("ok")):
                        continue
                    s = str(it.get("symbol") or "").strip()
                    if s:
                        syms.append(s)
                hold_fp = {"as_of": hold_obj2.get("as_of"), "generated_at": hold_obj2.get("generated_at"), "symbols": sorted(set(syms))}
        except (AttributeError) as exc:  # noqa: BLE001
            _record("data_hash.read_holdings", exc, note="data_hash 读取 holdings_user.json 失败（将忽略该指纹）")
            hold_fp = None

        if sig_fp or hold_fp:
            data_hash = compute_params_hash({"signals": sig_fp, "holdings": hold_fp})
    except (AttributeError) as exc:  # noqa: BLE001
        _record("data_hash.compute", exc, note="计算 data_hash 失败（不影响主流程）")
        data_hash = None

    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "run", "as_of": as_of, "data_hash": data_hash})
    run_config = _write_run_config(out_dir, args, note="run", extra={"cmd": "run"})
    try:
        from ..reporting import build_report_v1

        artifacts = {
            "signals": "signals.json",
            "signals_strategy": "signals_strategy.json",
            "signals_legacy": "signals_legacy.json",
            "holdings_user": "holdings_user.json",
            "rebalance_user": "rebalance_user.json",
            "national_team": "national_team.json",
            "alerts": "alerts.json",
            "orders_next_open": "orders_next_open.json",
            "scan_strategy_dir": "scan_strategy/",
            "scan_etf_dir": "scan_etf/",
            "strategy_alignment_dir": "strategy_alignment/",
            "report_md": "report.md",
        }
        if (out_dir / "signals_left.json").exists():
            artifacts["signals_left"] = "signals_left.json"
            artifacts["scan_strategy_left_dir"] = "scan_strategy_left/"
        if (out_dir / "signals_stock.json").exists():
            artifacts["signals_stock"] = "signals_stock.json"
            artifacts["scan_strategy_stock_dir"] = "scan_strategy_stock/"
        if (out_dir / "signals_left_stock.json").exists():
            artifacts["signals_left_stock"] = "signals_left_stock.json"
            artifacts["scan_strategy_stock_left_dir"] = "scan_strategy_stock_left/"
        if (out_dir / "report_holdings.md").exists():
            artifacts["report_holdings_md"] = "report_holdings.md"
            artifacts["holdings_deep_dir"] = "holdings_deep/"
        if (out_dir / "holdings_deep_summary.json").exists():
            artifacts["holdings_deep_summary"] = "holdings_deep_summary.json"

        write_json(
            out_dir / "report.json",
            build_report_v1(
                cmd="run",
                run_meta=run_meta,
                run_config=run_config,
                artifacts=artifacts,
                summary={
                    "signals_items": sig_n,
                    "signals_strategy": sig_strategy,
                    "signals_merge": sig_merge,
                    "signals_left_items": sig_left_n,
                    "signals_left_strategy": sig_left_strategy,
                    "signals_stock_items": sig_stock_n,
                    "signals_stock_strategy": sig_stock_strategy,
                    "signals_left_stock_items": sig_left_stock_n,
                    "signals_left_stock_strategy": sig_left_stock_strategy,
                    "capital_policy": "single_pool",
                    "orders_next_open": ord_n,
                    "as_of": as_of,
                    "holdings_asof": holdings_asof,
                    "mode": reb_mode,
                    "alerts_counts": alerts_counts,
                    "warnings": run_warnings[:50],
                },
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _record("write_report_json", exc, note="写出 report.json 失败（不影响 orders/report.md 主流程）")

    # diagnostics：给排查留证据（比 stdout 可靠）
    try:
        write_json(
            out_dir / "diagnostics.json",
            {
                "schema": "llm_trading.diagnostics.v1",
                "generated_at": datetime.now().isoformat(),
                "cmd": "run",
                "warnings": run_warnings[:200],
                "errors": run_errors[:200],
            },
        )
    except (AttributeError) as exc:  # noqa: BLE001
        try:
            _LOG.warning("写出 diagnostics.json 失败: %s", exc)
        except (AttributeError):  # noqa: BLE001
            pass

    # memory：自动把“这次 run 输出了啥”落盘，防止每次新对话都像失忆一样重来。
    try:
        from ..memory_store import (
            append_run_daily_brief,
            resolve_memory_paths,
            sync_trade_rules_from_user_holdings,
        )

        mp = resolve_memory_paths()
        # trade_rules 是硬约束：从 user_holdings 同步到 profile，保证 LLM prompt 能稳定读到。
        sync_trade_rules_from_user_holdings(
            mp,
            holdings_path=Path(str(holdings_path)),
            source={"type": "auto", "cmd": "run", "holdings_path": str(holdings_path)},
        )
        append_run_daily_brief(
            mp,
            out_dir=out_dir,
            as_of=str(as_of or "") or None,
            holdings_asof=str(holdings_asof or "") or None,
            orders_next_open_count=int(ord_n) if ord_n is not None else None,
            alerts_counts=alerts_counts,
            warnings=run_warnings[:30],
        )
    except Exception:  # noqa: BLE001
        # 可选：记忆写入失败不影响主流程（但你会少一份“可复盘证据”）。
        pass

    print(str(out_dir.resolve()))
    return 0
