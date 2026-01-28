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

def cmd_factor_research(args: argparse.Namespace) -> int:
    """
    Phase1(P0)：因子研究最小闭环（IC/IR/衰减/成本/样本外的地基先打好）。
    输出目录：默认 outputs/factor_reports_<timestamp>（内含 summary/ic/whitelist 三件套）。
    """
    from ..factors.research import FactorResearchParams, run_factor_research
    from ..utils_time import parse_date_any_opt

    asset = str(getattr(args, "asset", "") or "").strip().lower()
    if asset not in {"etf", "stock", "index"}:
        raise SystemExit("参数错误：--asset 只能是 etf/stock/index")

    freq = str(getattr(args, "freq", "") or "daily").strip().lower()
    if freq not in {"daily", "weekly"}:
        raise SystemExit("参数错误：--freq 只能是 daily/weekly")

    # 日期
    as_of_dt = parse_date_any_opt(str(getattr(args, "as_of", "") or "").strip() or None)
    start_dt = parse_date_any_opt(str(getattr(args, "start_date", "") or "").strip() or None)
    as_of = as_of_dt.date() if as_of_dt is not None else None
    start_date = start_dt.date() if start_dt is not None else None

    # horizons
    hz_raw = str(getattr(args, "horizons", "") or "1,5,10,20").strip()
    horizons: list[int] = []
    for p in hz_raw.split(","):
        s = p.strip()
        if not s:
            continue
        try:
            horizons.append(int(s))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        horizons = [1, 5, 10, 20]

    # universe
    uni_raw = str(getattr(args, "universe", "") or "").strip().lower()
    limit = int(getattr(args, "limit", 200) or 200)
    if limit <= 0:
        limit = 200

    symbols: list[str] = []
    if asset == "etf":
        include_all_funds = bool(getattr(args, "include_all_funds", False))
        items = load_etf_universe(include_all_funds=include_all_funds)
        symbols = [it.symbol for it in items][:limit]
    elif asset == "stock":
        # 默认用沪深300成分（更可控，别一上来全A把自己跑吐）
        if (not uni_raw) or uni_raw in {"hs300", "000300"}:
            from ..stock_scan import load_index_stock_universe

            items = load_index_stock_universe(index_symbol="000300")
            symbols = [it.symbol for it in items][:limit]
        elif uni_raw.startswith("index:"):
            from ..stock_scan import load_index_stock_universe

            idx = uni_raw.split(":", 1)[-1].strip() or "000300"
            items = load_index_stock_universe(index_symbol=idx)
            symbols = [it.symbol for it in items][:limit]
        elif uni_raw in {"all", "a"}:
            include_st = bool(getattr(args, "include_st", False))
            include_bj = bool(getattr(args, "include_bj", True))
            items = load_stock_universe(include_st=include_st, include_bj=include_bj)
            symbols = [it.symbol for it in items][:limit]
        else:
            raise SystemExit("参数错误：--universe(stock) 仅支持 hs300 / index:000300 / all")
    else:
        # index：研究通常是“单序列”，这里只跑一个 symbol
        idx_sym = str(getattr(args, "symbol", "") or "").strip()
        if not idx_sym:
            idx_sym = "sh000300"
        symbols = [idx_sym]

    if not symbols:
        raise SystemExit("universe 为空：没拿到任何 symbol")

    # cache/out
    cache_dir = Path(str(getattr(args, "cache_dir", "") or "").strip() or (Path("data") / "cache" / asset))
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)

    out_dir_raw = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"factor_reports_{asset}_{freq}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tradeability / cost
    # ETF 没有硬涨跌停：默认不开；股票默认 9.5%（主板口径，研究用途粗估）
    default_lim = 0.095 if asset == "stock" else 0.0
    limit_up_pct = float(getattr(args, "limit_up_pct", default_lim) or default_lim)
    limit_down_pct = float(getattr(args, "limit_down_pct", default_lim) or default_lim)
    min_fee_yuan = float(getattr(args, "min_fee_yuan", 5.0) or 5.0)
    slippage_bps = float(getattr(args, "slippage_bps", 10.0) or 10.0)
    notional_yuan = float(getattr(args, "notional_yuan", 2000.0) or 2000.0)

    # TuShare 因子包（按需开启；接口挂了也要能降级）
    include_tushare_micro = bool(getattr(args, "include_tushare_micro", False))
    include_tushare_macro = bool(getattr(args, "include_tushare_macro", False))
    max_tushare_symbols = int(getattr(args, "max_tushare_symbols", 80) or 80)
    if max_tushare_symbols < 0:
        max_tushare_symbols = 0

    ctx_raw = str(getattr(args, "context_index", "") or "sh000300").strip()
    ctx = ctx_raw.lower()
    # 支持 000300 / 000300.SH / sh000300
    if "." in ctx_raw:
        try:
            from ..tushare_source import ts_code_to_symbol

            ctx2 = ts_code_to_symbol(ctx_raw)
            if ctx2:
                ctx = str(ctx2).lower()
        except (TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
            pass
    if ctx.isdigit() and len(ctx) == 6:
        ctx = f"sh{ctx}"
    if not (len(ctx) == 8 and ctx.startswith(("sh", "sz")) and ctx[2:].isdigit()):
        raise SystemExit(f"参数错误：--context-index 必须是 sh/sz 前缀的 6 位指数代码（例 sh000300），当前={ctx_raw}")

    p = FactorResearchParams(
        asset=asset,  # type: ignore[arg-type]
        freq=freq,  # type: ignore[arg-type]
        universe=symbols,
        start_date=start_date,
        as_of=as_of,
        horizons=horizons,
        limit_up_pct=limit_up_pct,
        limit_down_pct=limit_down_pct,
        min_fee_yuan=min_fee_yuan,
        slippage_bps_each_side=slippage_bps,
        notional_yuan=notional_yuan,
        include_tushare_micro=bool(include_tushare_micro),
        include_tushare_macro=bool(include_tushare_macro),
        context_index_symbol=str(ctx),
        max_tushare_symbols=int(max_tushare_symbols),
    )

    res = run_factor_research(
        params=p,
        cache_dir=cache_dir,
        cache_ttl_hours=cache_ttl_hours,
        out_dir=out_dir,
        source="auto",
    )

    _write_run_meta(out_dir, args, extra={"cmd": "factor-research"})
    _write_run_config(out_dir, args, note="factor research", extra={"cmd": "factor-research"})
    write_json(out_dir / "report.json", {"schema": "llm_trading.report.v1", "cmd": "factor-research", "generated_at": datetime.now().isoformat(), "summary": res})

    print(str(out_dir.resolve()))
    return 0


