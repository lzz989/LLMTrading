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

def cmd_replay(args: argparse.Namespace) -> int:
    """
    一键复跑：从 run_config.json / run_meta.json / report.json 复现一次运行。
    默认写到新的 outputs 目录，避免把老产物覆盖掉。
    """
    import json

    from ..run_config import extract_argv_from_any, load_any_config

    src_raw = str(getattr(args, "src", "") or "").strip()
    if not src_raw:
        raise SystemExit("请传 --from（run_config.json/run_meta.json/report.json 或包含这些的目录）")
    src = Path(src_raw)
    if not src.exists():
        raise SystemExit(f"找不到：{src}")

    p = src
    if src.is_dir():
        # 优先 run_config，其次 run_meta，再其次 report
        for name in ["run_config.json", "run_meta.json", "report.json"]:
            cand = src / name
            if cand.exists():
                p = cand
                break
        else:
            raise SystemExit(f"目录下没找到 run_config.json/run_meta.json/report.json：{src}")

    try:
        payload = load_any_config(p)
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"读取失败：{p} {exc}") from exc

    argv0, cmd0 = extract_argv_from_any(payload)
    argv0 = [str(x) for x in (argv0 or []) if str(x).strip()]
    if not argv0:
        raise SystemExit(f"没从配置里读到 argv：{p}")

    cmd = str(cmd0 or argv0[0] or "").strip()
    if not cmd:
        raise SystemExit(f"配置里 cmd 为空：{p}")
    if cmd == "replay":
        raise SystemExit("replay 不能复跑 replay（你搁这套娃呢？）")

    # 输出目录：默认新建，避免覆盖
    out_dir = Path(str(getattr(args, "out_dir", "") or "").strip()) if getattr(args, "out_dir", None) else None
    if out_dir is None:
        out_dir = Path("outputs") / f"replay_{cmd}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 规避撞名
    out_dir_eff = out_dir
    for _ in range(20):
        if not out_dir_eff.exists():
            break
        out_dir_eff = Path(str(out_dir_eff) + "_1")

    # 清理旧 --out-dir 参数，统一改成新的
    def strip_out_dir(argv: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(argv):
            a = str(argv[i])
            if a == "--out-dir":
                i += 2
                continue
            if a.startswith("--out-dir="):
                i += 1
                continue
            out.append(a)
            i += 1
        return out

    argv1 = strip_out_dir(argv0)
    # 这些命令支持 --out-dir，复跑就强制指定新目录
    cmd_support_out_dir = {"analyze", "scan-etf", "scan-stock", "eval-bbb"}
    if cmd in cmd_support_out_dir:
        argv1 = list(argv1) + ["--out-dir", str(out_dir_eff)]

    if bool(getattr(args, "print_only", False)):
        print(json.dumps({"from": str(p), "cmd": cmd, "argv": argv1, "out_dir": str(out_dir_eff)}, ensure_ascii=False, indent=2))
        return 0

    # 直接在本进程里复跑（避免再起 python）
    # 延迟导入：避免 cli.py <-> cli_commands.py 循环依赖在 import 时炸掉。
    from ..cli import main as _main

    return int(_main(argv1))


