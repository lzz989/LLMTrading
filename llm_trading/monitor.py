from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .json_utils import sanitize_for_json


def _parse_date(s: str | None) -> date | None:
    t = str(s or "").strip()
    if not t:
        return None
    # YYYYMMDD
    if len(t) == 8 and t.isdigit():
        try:
            return datetime.strptime(t, "%Y%m%d").date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            return None
    # YYYY-MM-DD...
    if len(t) >= 10 and t[4] == "-" and t[7] == "-":
        head = t[:10]
        try:
            return datetime.strptime(head, "%Y-%m-%d").date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            pass
    # ISO datetime
    try:
        return datetime.fromisoformat(t).date()
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (AttributeError):  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


@dataclass(frozen=True, slots=True)
class MonitorRow:
    out_dir: str
    cmd: str
    generated_at: str | None
    warnings_count: int
    # 兼容不同 cmd 的关键摘要字段（缺就 None）
    as_of: str | None = None
    holdings_asof: str | None = None
    signals_items: int | None = None
    orders_next_open: int | None = None
    mode: str | None = None
    alerts_stop: int | None = None
    alerts_take_profit: int | None = None
    alerts_watch: int | None = None
    # paper-sim
    cagr: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    trades: int | None = None
    turnover_annualized: float | None = None
    capacity_max_participation: float | None = None
    # eval-bbb
    stability_score_avg: float | None = None
    oos_trades_sum: int | None = None
    oos_win_rate_avg: float | None = None

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "cmd": self.cmd,
            "generated_at": self.generated_at,
            "warnings_count": self.warnings_count,
            "as_of": self.as_of,
            "holdings_asof": self.holdings_asof,
            "signals_items": self.signals_items,
            "orders_next_open": self.orders_next_open,
            "mode": self.mode,
            "alerts_stop": self.alerts_stop,
            "alerts_take_profit": self.alerts_take_profit,
            "alerts_watch": self.alerts_watch,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "trades": self.trades,
            "turnover_annualized": self.turnover_annualized,
            "capacity_max_participation": self.capacity_max_participation,
            "stability_score_avg": self.stability_score_avg,
            "oos_trades_sum": self.oos_trades_sum,
            "oos_win_rate_avg": self.oos_win_rate_avg,
        }


def _extract_row(report: dict[str, Any], *, out_dir: Path) -> MonitorRow | None:
    cmd = str(report.get("cmd") or "").strip()
    if not cmd:
        return None
    generated_at = str(report.get("generated_at") or "").strip() or None
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    warnings_count = int(len(warnings))

    summ = report.get("summary") if isinstance(report.get("summary"), dict) else {}

    if cmd == "run":
        alerts_counts = summ.get("alerts_counts") if isinstance(summ.get("alerts_counts"), dict) else {}
        return MonitorRow(
            out_dir=str(out_dir),
            cmd=cmd,
            generated_at=generated_at,
            warnings_count=warnings_count,
            as_of=str(summ.get("as_of") or "").strip() or None,
            holdings_asof=str(summ.get("holdings_asof") or "").strip() or None,
            signals_items=int(summ.get("signals_items")) if summ.get("signals_items") is not None else None,
            orders_next_open=int(summ.get("orders_next_open")) if summ.get("orders_next_open") is not None else None,
            mode=str(summ.get("mode") or "").strip() or None,
            alerts_stop=int(alerts_counts.get("stop")) if alerts_counts.get("stop") is not None else None,
            alerts_take_profit=int(alerts_counts.get("take_profit")) if alerts_counts.get("take_profit") is not None else None,
            alerts_watch=int(alerts_counts.get("watch")) if alerts_counts.get("watch") is not None else None,
        )

    if cmd == "paper-sim":
        turnover = summ.get("turnover") if isinstance(summ.get("turnover"), dict) else {}
        capacity = summ.get("capacity") if isinstance(summ.get("capacity"), dict) else {}
        return MonitorRow(
            out_dir=str(out_dir),
            cmd=cmd,
            generated_at=generated_at,
            warnings_count=warnings_count,
            as_of=str((report.get("run_meta") or {}).get("extra", {}).get("as_of") or "").strip() or None,
            cagr=float(summ.get("cagr")) if summ.get("cagr") is not None else None,
            max_drawdown=float(summ.get("max_drawdown")) if summ.get("max_drawdown") is not None else None,
            win_rate=float(summ.get("win_rate")) if summ.get("win_rate") is not None else None,
            trades=int(summ.get("trades")) if summ.get("trades") is not None else None,
            turnover_annualized=float(turnover.get("turnover_annualized")) if turnover.get("turnover_annualized") is not None else None,
            capacity_max_participation=float(capacity.get("max_participation")) if capacity.get("max_participation") is not None else None,
        )

    if cmd == "eval-bbb":
        items = summ.get("items") if isinstance(summ.get("items"), list) else []
        stab: list[float] = []
        oos_trades_sum = 0
        oos_wr: list[float] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            v = it.get("stability_score")
            if v is not None:
                try:
                    stab.append(float(v))
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    pass
            t = it.get("oos_trades")
            if t is not None:
                try:
                    oos_trades_sum += int(t)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    pass
            wr = it.get("oos_win_rate_shrunk")
            if wr is not None:
                try:
                    oos_wr.append(float(wr))
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    pass
        stab_avg = (sum(stab) / len(stab)) if stab else None
        oos_wr_avg = (sum(oos_wr) / len(oos_wr)) if oos_wr else None
        return MonitorRow(
            out_dir=str(out_dir),
            cmd=cmd,
            generated_at=generated_at,
            warnings_count=warnings_count,
            as_of=str((report.get("extra") or {}).get("as_of") or "").strip() or None,
            stability_score_avg=stab_avg,
            oos_trades_sum=int(oos_trades_sum) if oos_trades_sum else None,
            oos_win_rate_avg=oos_wr_avg,
        )

    # 其它命令：只保留基础信息
    return MonitorRow(out_dir=str(out_dir), cmd=cmd, generated_at=generated_at, warnings_count=warnings_count)


def scan_reports(*, outputs_dir: Path, max_dirs: int = 200) -> list[Path]:
    """
    扫描 outputs 目录下“一级子目录”的 report.json（KISS，避免递归扫爆）。
    """
    if not outputs_dir.exists():
        return []
    dirs = [p for p in outputs_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if int(max_dirs or 0) > 0:
        dirs = dirs[: int(max_dirs)]
    out: list[Path] = []
    for d in dirs:
        rp = d / "report.json"
        if rp.exists():
            out.append(rp)
    return out


def build_monitor_payload(
    *,
    outputs_dir: Path,
    max_dirs: int = 200,
    include_cmds: set[str] | None = None,
    exclude_cmds: set[str] | None = None,
) -> dict[str, Any]:
    include = {str(x).strip() for x in (include_cmds or set()) if str(x).strip()}
    exclude = {str(x).strip() for x in (exclude_cmds or set()) if str(x).strip()}

    report_paths = scan_reports(outputs_dir=outputs_dir, max_dirs=max_dirs)
    rows: list[MonitorRow] = []
    warnings: list[str] = []

    today = datetime.now().date()
    for rp in report_paths:
        obj = _load_json(rp)
        if not obj:
            continue
        cmd = str(obj.get("cmd") or "").strip()
        if not cmd:
            continue
        if include and cmd not in include:
            continue
        if exclude and cmd in exclude:
            continue

        row = _extract_row(obj, out_dir=rp.parent)
        if row is None:
            continue
        rows.append(row)

        # 轻量预警：只做“明显问题”，别搞学术。
        if row.warnings_count > 0:
            warnings.append(f"{rp.parent}: report.warnings={row.warnings_count}")

        if row.cmd == "run":
            d_hold = _parse_date(row.holdings_asof)
            if d_hold is not None:
                days = (today - d_hold).days
                if days >= 5:
                    warnings.append(f"{rp.parent}: holdings_asof={row.holdings_asof} 已滞后 {days} 天（数据可能过期）")

        if row.cmd == "paper-sim":
            if row.max_drawdown is not None and row.max_drawdown <= -0.20:
                warnings.append(f"{rp.parent}: max_drawdown={row.max_drawdown:.2%} 偏大（回撤压力）")
            if row.turnover_annualized is not None and row.turnover_annualized >= 4.0:
                warnings.append(f"{rp.parent}: turnover_annualized={row.turnover_annualized:.2f} 偏高（磨损压力）")
            if row.capacity_max_participation is not None and row.capacity_max_participation >= 0.01:
                warnings.append(f"{rp.parent}: capacity.max_participation={row.capacity_max_participation:.2%} 偏高（容量/冲击成本风险）")

    # 统计
    by_cmd: dict[str, int] = {}
    for r in rows:
        by_cmd[r.cmd] = int(by_cmd.get(r.cmd, 0) + 1)

    rows_csv = [r.to_csv_row() for r in rows]
    return sanitize_for_json(
        {
            "schema": "llm_trading.monitor.v1",
            "generated_at": datetime.now().isoformat(),
            "outputs_dir": str(outputs_dir),
            "counts": {"reports": int(len(rows)), "by_cmd": by_cmd},
            "warnings": warnings,
            "rows": rows_csv,
        }
    )


def write_monitor_artifacts(*, out_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monitor.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    csv_path = out_dir / "summary.csv"
    if rows:
        keys = list(rows[0].keys()) if isinstance(rows[0], dict) else []
        if keys:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    if isinstance(r, dict):
                        w.writerow(r)

    # 简单 md，给人看的
    md_lines: list[str] = []
    md_lines.append("# outputs 监控摘要（研究用途）")
    md_lines.append("")
    md_lines.append(f"- generated_at: {payload.get('generated_at')}")
    md_lines.append(f"- outputs_dir: {payload.get('outputs_dir')}")
    md_lines.append(f"- reports: {((payload.get('counts') or {}).get('reports'))}")
    md_lines.append(f"- by_cmd: {json.dumps(((payload.get('counts') or {}).get('by_cmd') or {}), ensure_ascii=False)}")
    md_lines.append("")
    warns = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warns:
        md_lines.append("## Warnings")
        for w0 in warns[:50]:
            md_lines.append(f"- {w0}")
        if len(warns) > 50:
            md_lines.append(f"- ... 省略 {len(warns) - 50} 条（看 monitor.json）")
        md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append("- monitor.json")
    md_lines.append("- summary.csv")
    md_lines.append("")
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "monitor_json": "monitor.json",
        "summary_csv": "summary.csv",
        "report_md": "report.md",
    }

