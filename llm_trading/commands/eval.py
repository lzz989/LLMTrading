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

def cmd_eval_shortline(args: argparse.Namespace) -> int:
    raise SystemExit("eval-shortline 已从主框架精简移除（超短线/周内短线模块不再维护）。")


def cmd_eval_bbb(args: argparse.Namespace) -> int:
    """
    BBB 稳健性评估（walk-forward + 参数敏感性，研究用途）。
    """
    import json

    from ..akshare_source import FetchParams
    from ..data_cache import fetch_daily_cached

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"eval_bbb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else (Path("data") / "cache" / "etf")
    cache_ttl_hours = float(args.cache_ttl_hours) if getattr(args, "cache_ttl_hours", None) is not None else 24.0

    # 输入：symbol 或 top_bbb.json
    syms: list[str] = []
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

    syms = [s for s in syms if s]
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
        raise SystemExit("请传 --symbol 或 --input( top_bbb.json )")

    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        syms = syms[: int(limit)]

    # 成本口径（尽量沿用 scan-etf）
    try:
        from ..costs import cost_model_from_roundtrip
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        cost_model_from_roundtrip = None  # type: ignore[assignment]

    capital_yuan = float(getattr(args, "capital_yuan", 3000.0) or 3000.0)
    roundtrip_cost_yuan = float(getattr(args, "roundtrip_cost_yuan", 10.0) or 10.0)
    cm = cost_model_from_roundtrip(capital_yuan=capital_yuan, roundtrip_cost_yuan=roundtrip_cost_yuan) if cost_model_from_roundtrip else None
    buy_cost = float(args.buy_cost) if getattr(args, "buy_cost", None) is not None else float(getattr(cm, "buy_cost", 0.0) or 0.0)
    sell_cost = float(args.sell_cost) if getattr(args, "sell_cost", None) is not None else float(getattr(cm, "sell_cost", 0.0) or 0.0)

    # BBB 参数
    bbb_mode = str(getattr(args, "bbb_mode", "strict") or "strict").strip().lower()
    bbb_entry_ma = int(getattr(args, "bbb_entry_ma", 50) or 50)
    bbb_dist_ma_max = float(getattr(args, "bbb_dist_ma_max", 0.12) or 0.12)
    bbb_max_above_20w = float(getattr(args, "bbb_max_above_20w", 0.05) or 0.05)
    min_weeks = int(getattr(args, "min_weeks", 60) or 60)

    horizon_weeks = int(getattr(args, "horizon_weeks", 8) or 8)
    score_mode = str(getattr(args, "score_mode", "annualized") or "annualized").strip().lower()
    non_overlapping = not bool(getattr(args, "allow_overlap", False))

    train_weeks = int(getattr(args, "train_weeks", 156) or 156)
    test_weeks = int(getattr(args, "test_weeks", 26) or 26)
    step_weeks = int(getattr(args, "step_weeks", 26) or 26)
    include_mode_variants = bool(getattr(args, "include_mode_variants", True))

    start_date = str(getattr(args, "start_date", "") or "").strip() or None
    end_date = str(getattr(args, "end_date", "") or "").strip() or None

    from ..backtest import score_forward_stats
    from ..resample import resample_to_weekly
    from ..robustness_bbb import build_oat_variants, bbb_params_from_mode, summarize_stats, walk_forward_select_and_eval

    summary: list[dict[str, object]] = []

    for idx, sym in enumerate(syms, start=1):
        if bool(getattr(args, "verbose", False)):
            print(f"[{idx}/{len(syms)}] {sym} ...")

        df_daily = fetch_daily_cached(
            FetchParams(asset="etf", symbol=sym, start_date=start_date, end_date=end_date),
            cache_dir=cache_dir,
            ttl_hours=float(cache_ttl_hours),
        )
        df_weekly = resample_to_weekly(df_daily)
        last_date = None
        try:
            last_date = str(df_daily["date"].max().date())
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            last_date = None

        base = bbb_params_from_mode(
            mode=str(bbb_mode),
            entry_ma=int(bbb_entry_ma),
            dist_ma50_max=float(bbb_dist_ma_max),
            max_above_20w=float(bbb_max_above_20w),
            min_weekly_bars_total=max(10, int(min_weeks)),
        )
        variants = build_oat_variants(base, include_modes=bool(include_mode_variants))

        # 参数敏感性：全样本对比（只算 forward holding，避免出场策略 trade 过滤复杂度）
        var_rows: list[dict[str, Any]] = []
        for v in variants:
            sig = None
            try:
                from ..bbb import compute_bbb_entry_signal

                sig = compute_bbb_entry_signal(df_weekly, df_daily, params=v.params)
            except (AttributeError) as exc:  # noqa: BLE001
                var_rows.append({"key": v.key, "error": str(exc)})
                continue
            try:
                from ..backtest import forward_holding_backtest

                st, _ = forward_holding_backtest(
                    df_weekly,
                    entry_signal=sig,
                    horizon_weeks=int(horizon_weeks),
                    buy_cost=float(buy_cost),
                    sell_cost=float(sell_cost),
                    non_overlapping=bool(non_overlapping),
                )
            except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
                var_rows.append({"key": v.key, "error": str(exc)})
                continue
            var_rows.append(
                {
                    "key": v.key,
                    "score": float(score_forward_stats(st, mode=str(score_mode))),
                    "stats": summarize_stats(st),
                    "params": {
                        "entry_ma": int(v.params.entry_ma),
                        "dist_ma50_max": float(v.params.dist_ma50_max),
                        "max_above_20w": float(v.params.max_above_20w),
                        "min_weekly_bars_total": int(v.params.min_weekly_bars_total),
                        "require_weekly_macd_bullish": bool(v.params.require_weekly_macd_bullish),
                        "require_weekly_macd_above_zero": bool(v.params.require_weekly_macd_above_zero),
                        "require_daily_macd_bullish": bool(v.params.require_daily_macd_bullish),
                    },
                }
            )

        var_rows2 = [x for x in var_rows if isinstance(x, dict) and "score" in x]
        var_rows2.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

        walk = walk_forward_select_and_eval(
            df_weekly,
            df_daily,
            variants=variants,
            horizon_weeks=int(horizon_weeks),
            score_mode=str(score_mode),  # type: ignore[arg-type]
            buy_cost=float(buy_cost),
            sell_cost=float(sell_cost),
            non_overlapping=bool(non_overlapping),
            train_weeks=int(train_weeks),
            test_weeks=int(test_weeks),
            step_weeks=int(step_weeks),
        )

        # 简单告警：OOS trades 多但均值为负/胜率很差
        alerts: list[str] = []
        oos = walk.get("oos_summary") if isinstance(walk, dict) else None
        if isinstance(oos, dict):
            t = int(oos.get("trades") or 0)
            wr_s = float(oos.get("win_rate_shrunk") or 0.0)
            ar = float(oos.get("avg_return") or 0.0)
            if t >= 8 and ar < 0:
                alerts.append("OOS 平均收益为负（可能阶段失效）")
            if t >= 12 and wr_s < 0.45:
                alerts.append("OOS 收缩胜率偏低（可能过拟合/走样）")

        payload = {
            "generated_at": datetime.now().isoformat(),
            "symbol": str(sym),
            "data": {
                "last_date": last_date,
                "rows_daily": int(len(df_daily)),
                "rows_weekly": int(len(df_weekly)),
            },
            "input": {
                "start_date": start_date,
                "end_date": end_date,
                "cache_dir": str(cache_dir),
                "cache_ttl_hours": float(cache_ttl_hours),
            },
            "costs": {
                "capital_yuan": float(capital_yuan),
                "roundtrip_cost_yuan": float(roundtrip_cost_yuan),
                "buy_cost": float(buy_cost),
                "sell_cost": float(sell_cost),
            },
            "bbb": {
                "mode": str(bbb_mode),
                "entry_ma": int(bbb_entry_ma),
                "dist_ma50_max": float(bbb_dist_ma_max),
                "max_above_20w": float(bbb_max_above_20w),
                "min_weekly_bars_total": int(min_weeks),
                "horizon_weeks": int(horizon_weeks),
                "score_mode": str(score_mode),
                "non_overlapping": bool(non_overlapping),
            },
            "variants": var_rows2[:30],
            "walk_forward": walk,
            "alerts": alerts,
            "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
        }

        out_file = out_dir / f"bbb_robust_{sym}.json"
        write_json(out_file, payload)

        summary.append(
            {
                "symbol": str(sym),
                "last_date": last_date,
                "stability_score": float((walk.get("stability_score") if isinstance(walk, dict) else 0.0) or 0.0),
                "oos_trades": int(((walk.get("oos_summary") or {}).get("trades") if isinstance(walk, dict) else 0) or 0),
                "oos_win_rate_shrunk": float(((walk.get("oos_summary") or {}).get("win_rate_shrunk") if isinstance(walk, dict) else 0.0) or 0.0),
                "oos_avg_return": float(((walk.get("oos_summary") or {}).get("avg_return") if isinstance(walk, dict) else 0.0) or 0.0),
                "alerts": alerts,
            }
        )

    summary.sort(key=lambda x: float(x.get("stability_score") or 0.0), reverse=True)
    write_json(out_dir / "summary.json", {"generated_at": datetime.now().isoformat(), "summary": summary})
    as_of = None
    for it in summary:
        if not isinstance(it, dict):
            continue
        ld = str(it.get("last_date") or "").strip()
        if not ld:
            continue
        if as_of is None or ld > as_of:
            as_of = ld

    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "eval-bbb", "as_of": as_of})
    run_config = _write_run_config(out_dir, args, note="eval-bbb", extra={"cmd": "eval-bbb"})
    try:
        from ..reporting import build_report_v1

        write_json(
            out_dir / "report.json",
            build_report_v1(
                cmd="eval-bbb",
                run_meta=run_meta,
                run_config=run_config,
                artifacts={"run_meta": "run_meta.json", "run_config": "run_config.json", "summary": "summary.json"},
                summary={"items": summary[:50]},
                extra={"as_of": as_of, "symbols": int(len(summary))},
            ),
        )
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass
    print(str(out_dir.resolve()))
    return 0


