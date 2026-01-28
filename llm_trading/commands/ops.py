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

def cmd_clean_outputs(args: argparse.Namespace) -> int:
    """
    清理 outputs 里历史产物（结果有时效性，别让它一直占空间）。
    默认只 dry-run，真删必须加 --apply。
    """
    root = Path(str(getattr(args, "path", None) or "outputs"))
    if not root.exists():
        print(str(root))
        print("outputs 不存在，没啥可清。")
        return 0
    if not root.is_dir():
        raise SystemExit(f"不是目录：{root}")

    keep_days = float(getattr(args, "keep_days", 7.0) or 0.0)
    keep_last = int(getattr(args, "keep_last", 20) or 0)
    include_logs = bool(getattr(args, "include_logs", False))
    apply = bool(getattr(args, "apply", False))

    keep_days = max(0.0, min(keep_days, 3650.0))
    keep_last = max(0, min(keep_last, 5000))

    now = time.time()
    cutoff = now - keep_days * 86400.0 if keep_days > 0 else None

    entries: list[tuple[float, Path]] = []
    for p in root.iterdir():
        name = p.name
        if name in {".gitkeep", "_ssh", "_duckdb_sentinel"}:
            # 这里可能存着用于 git push 的 SSH key（别删，删了你就等着抓狂吧）
            continue
        if p.is_file() and (not include_logs):
            # 默认只清“结果目录”，日志别乱动
            continue
        try:
            st = p.stat()
        except (AttributeError):  # noqa: BLE001
            continue
        entries.append((float(st.st_mtime), p))

    entries.sort(key=lambda x: x[0], reverse=True)

    keep: set[Path] = set()
    if keep_last > 0:
        keep.update([p for _, p in entries[:keep_last]])
    if cutoff is not None:
        for mt, p in entries:
            if mt >= cutoff:
                keep.add(p)

    to_delete = [p for _, p in entries if p not in keep]

    def fmt_ts(ts: float) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except (AttributeError):  # noqa: BLE001
            return str(ts)

    print(str(root.resolve()))
    print(f"规则：保留最近 {keep_days:g} 天 + 最近 {keep_last} 个；include_logs={include_logs}; apply={apply}")
    print(f"扫描到 {len(entries)} 个条目；计划删除 {len(to_delete)} 个；保留 {len(keep)} 个。")
    if entries:
        mt_new, p_new = entries[0]
        mt_old, p_old = entries[-1]
        print(f"最新：{p_new.name} ({fmt_ts(mt_new)})")
        print(f"最旧：{p_old.name} ({fmt_ts(mt_old)})")

    if not to_delete:
        print("没有需要删除的条目。")
        return 0

    if not apply:
        print("dry-run：以下将被删除（前 30 个）：")
        for p in to_delete[:30]:
            print(f"- {p.name}")
        if len(to_delete) > 30:
            print(f"... 还有 {len(to_delete) - 30} 个")
        print("真要删就加 --apply（小心点，删了就没了）。")
        return 0

    import shutil

    deleted = 0
    failed = 0
    for p in to_delete:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            deleted += 1
        except (AttributeError):  # noqa: BLE001
            failed += 1

    print(f"完成：删除 {deleted} 个；失败 {failed} 个。")
    return 0


def cmd_data_doctor(args: argparse.Namespace) -> int:
    """
    数据质量/可复现性体检（研究用途）：
    - 检查 data/cache 里的 OHLCV CSV（抽样：默认只查最近修改的一批，避免 1w+ 文件拖死）
    - 检查 outputs 里的 run/paper-sim/scan 等产物是否带 run_meta/run_config/signals 基本字段
    """
    import json

    from ..data_doctor import DataDoctorConfig, validate_repo_data

    cache_dir = Path(str(getattr(args, "cache_dir", "") or "").strip() or (Path("data") / "cache"))
    outputs_dir = Path(str(getattr(args, "outputs_dir", "") or "").strip() or Path("outputs"))
    cfg = DataDoctorConfig(
        cache_dir=cache_dir,
        outputs_dir=outputs_dir,
        cache_recent_days=int(getattr(args, "cache_recent_days", 3) or 3),
        cache_max_files=int(getattr(args, "cache_max_files", 200) or 200),
        outputs_max_dirs=int(getattr(args, "outputs_max_dirs", 30) or 30),
        include_cache=bool(getattr(args, "include_cache", True)),
        include_outputs=bool(getattr(args, "include_outputs", True)),
    )

    res = validate_repo_data(cfg)

    if getattr(args, "out", None):
        out_path = Path(str(getattr(args, "out")))
        write_json(out_path, res)
        print(str(out_path.resolve()))
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False))

    fail_on = str(getattr(args, "fail_on", "never") or "never").strip().lower()
    errs = int(((res.get("counts") or {}).get("errors") if isinstance(res, dict) else 0) or 0)
    warns = int(((res.get("counts") or {}).get("warnings") if isinstance(res, dict) else 0) or 0)

    if errs > 0:
        return 2 if fail_on in {"error", "warn"} else 0
    if warns > 0:
        return 1 if fail_on in {"warn"} else 0
    return 0



def cmd_monitor(args: argparse.Namespace) -> int:
    """
    outputs 监控/回顾（研究用途）：
    - 扫描 outputs/*/report.json（一级目录，避免递归扫爆）
    - 汇总成一个 monitor.json + summary.csv + report.md
    """
    from ..monitor import build_monitor_payload, write_monitor_artifacts

    outputs_dir = Path(str(getattr(args, "outputs_dir", "") or "").strip() or "outputs")
    max_dirs = int(getattr(args, "max_dirs", 200) or 200)

    include_cmds: set[str] | None = None
    exclude_cmds: set[str] | None = None
    inc_raw = str(getattr(args, "include_cmds", "") or "").strip()
    exc_raw = str(getattr(args, "exclude_cmds", "") or "").strip()
    if inc_raw:
        include_cmds = {p.strip() for p in inc_raw.split(",") if p.strip()}
    if exc_raw:
        exclude_cmds = {p.strip() for p in exc_raw.split(",") if p.strip()}

    # out_dir：默认 outputs/monitor_YYYYMMDD；同日重复跑自动加后缀
    today = datetime.now().strftime("%Y%m%d")
    base = str(getattr(args, "out_dir", "") or "").strip()
    out_dir = Path(base) if base else (Path("outputs") / f"monitor_{today}")
    if out_dir.exists():
        for i in range(2, 2000):
            cand = Path(f"{out_dir}_{i}")
            if not cand.exists():
                out_dir = cand
                break
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_monitor_payload(
        outputs_dir=outputs_dir,
        max_dirs=max_dirs,
        include_cmds=include_cmds,
        exclude_cmds=exclude_cmds,
    )

    artifacts = write_monitor_artifacts(out_dir=out_dir, payload=payload)
    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "monitor"})
    run_config = _write_run_config(out_dir, args, note="monitor", extra={"cmd": "monitor"})
    report_obj = {
        "schema": "llm_trading.report.v1",
        "generated_at": datetime.now().isoformat(),
        "cmd": "monitor",
        "run_meta": run_meta,
        "run_config": run_config,
        "artifacts": artifacts,
        "summary": (payload.get("counts") if isinstance(payload, dict) else None),
        "warnings": (payload.get("warnings") if isinstance(payload, dict) else []),
        "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
    }
    write_json(out_dir / "report.json", report_obj)

    print(str(out_dir.resolve()))
    return 0



