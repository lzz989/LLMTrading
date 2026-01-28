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

def cmd_national_team(args: argparse.Namespace) -> int:
    """
    国家队/托底代理指标（研究用途）：
    - A: 宽基ETF份额Δ（需要每日落盘 spot 快照）+ 宽基ETF主力净流入（近120日）
    - C: 指数尾盘护盘强度（分时；近5个交易日）
    - B: 北向日度净买在当前免费源上经常缺失：保留字段但默认不计入综合分
    """
    import json

    from ..national_team import NationalTeamProxyConfig, compute_national_team_proxy
    from ..utils_time import parse_date_any_opt

    as_of_raw = str(getattr(args, "as_of", "") or "").strip()
    as_of_dt = parse_date_any_opt(as_of_raw) if as_of_raw else None
    as_of = (as_of_dt.date() if as_of_dt is not None else datetime.now().date())

    idx = str(getattr(args, "index_symbol", "sh000300") or "sh000300").strip() or "sh000300"
    wide_raw = str(getattr(args, "wide_etfs", "") or "").strip()
    if wide_raw:
        wide = tuple([x.strip() for x in wide_raw.split(",") if x.strip()])
    else:
        wide = NationalTeamProxyConfig().wide_etfs

    cfg = NationalTeamProxyConfig(
        index_symbol=str(idx),
        wide_etfs=tuple(wide),
        etf_flow_lookback_days=int(getattr(args, "flow_lookback_days", 120) or 120),
        tail_window_minutes=int(getattr(args, "tail_window_minutes", 30) or 30),
        w_etf_flow=float(getattr(args, "w_etf_flow", 0.55) or 0.55),
        w_etf_shares=float(getattr(args, "w_etf_shares", 0.25) or 0.25),
        w_tail=float(getattr(args, "w_tail", 0.20) or 0.20),
        w_northbound=float(getattr(args, "w_northbound", 0.0) or 0.0),
    )

    out = compute_national_team_proxy(as_of=as_of, cfg=cfg, cache_ttl_hours=float(getattr(args, "cache_ttl_hours", 6.0) or 6.0))

    if getattr(args, "out", None):
        out_path = Path(str(args.out))
        write_json(out_path, out)
        print(str(out_path.resolve()))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def cmd_national_team_backtest(args: argparse.Namespace) -> int:
    """
    最小回测（研究用途）：只回测 ETF 主力净流入 proxy（近 ~120 交易日，受数据源限制）。
    输出：
    - scores.csv：每日 sum_inflow / z / score01 / 指数次日收益
    - report.md：简要统计（high/mid/low 分位的次日收益对比）
    - report.json：同上（机器可读）
    """
    import json

    from ..national_team import NationalTeamProxyConfig, backtest_etf_flow_proxy
    from ..utils_time import parse_date_any_opt

    today = datetime.now().strftime("%Y%m%d")
    base = str(getattr(args, "out_dir", "") or "").strip()
    out_dir = Path(base) if base else (Path("outputs") / f"nt_backtest_{today}")
    if out_dir.exists():
        for i in range(2, 2000):
            cand = Path(f"{out_dir}_{i}")
            if not cand.exists():
                out_dir = cand
                break
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = str(getattr(args, "index_symbol", "sh000300") or "sh000300").strip() or "sh000300"
    wide_raw = str(getattr(args, "wide_etfs", "") or "").strip()
    if wide_raw:
        wide = tuple([x.strip() for x in wide_raw.split(",") if x.strip()])
    else:
        wide = NationalTeamProxyConfig().wide_etfs

    start_raw = str(getattr(args, "start_date", "") or "").strip()
    end_raw = str(getattr(args, "end_date", "") or "").strip()
    start_dt = parse_date_any_opt(start_raw) if start_raw else None
    end_dt = parse_date_any_opt(end_raw) if end_raw else None

    res = backtest_etf_flow_proxy(
        index_symbol=str(idx),
        watchlist=tuple(wide),
        start=(start_dt.date() if start_dt is not None else None),
        end=(end_dt.date() if end_dt is not None else None),
        lookback_days=int(getattr(args, "lookback_days", 60) or 60),
        cache_ttl_hours=float(getattr(args, "cache_ttl_hours", 24.0) or 24.0),
    )

    df = res.pop("df", None)
    if df is not None:
        try:
            df.to_csv(out_dir / "scores.csv", index=False, encoding="utf-8")
        except (AttributeError):  # noqa: BLE001
            pass

    # report.md（KISS）
    try:
        st = res.get("stats") if isinstance(res, dict) else None
        st = st if isinstance(st, dict) else {}
        lines = [
            "# national_team_backtest\n",
            "",
            f"- generated_at: {res.get('generated_at')}",
            f"- index_symbol: {res.get('index_symbol')}",
            f"- lookback_days: {res.get('lookback_days')}",
            f"- range.start: {(res.get('range') or {}).get('start')}",
            f"- range.end: {(res.get('range') or {}).get('end')}",
            "",
            "## stats(next_day_return)\n",
            f"- all: {st.get('all')}",
            f"- high(score01>=0.67): {st.get('high')}",
            f"- mid: {st.get('mid')}",
            f"- low(score01<=0.33): {st.get('low')}",
            "",
            "产物：",
            "- scores: scores.csv",
            "- report_json: report.json",
            "",
            "免责声明：研究工具输出，不构成投资建议。",
            "",
        ]
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        pass

    try:
        write_json(out_dir / "report.json", res)
    except (AttributeError):  # noqa: BLE001
        pass

    print(str(out_dir.resolve()))
    return 0



