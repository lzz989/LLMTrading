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

def cmd_reconcile(args: argparse.Namespace) -> int:
    """
    对账闭环（研究用途）：
    - 输入：真实成交 fills（CSV/JSON/JSONL，可来自你“手动点确认”后的 API 导出）
    - 输出：user_holdings_next.json + ledger_trades_append.jsonl + report.md

    说明：
    - 默认 dry-run（只生成对账产物，不写回 data/user_holdings.json / data/ledger_trades.jsonl）
    - 真要落盘必须显式加 --apply（别手滑）
    """
    import json

    from ..json_utils import sanitize_for_json
    from ..reconcile import (
        load_ledger_trade_ids,
        load_orders_next_open,
        load_trade_fills,
        load_user_holdings_snapshot,
        reconcile_user_holdings,
    )

    today = datetime.now().strftime("%Y%m%d")
    base = str(getattr(args, "out_dir", "") or "").strip()
    out_dir = Path(base) if base else (Path("outputs") / f"reconcile_{today}")
    if out_dir.exists():
        for i in range(2, 2000):
            cand = Path(f"{out_dir}_{i}")
            if not cand.exists():
                out_dir = cand
                break
    out_dir.mkdir(parents=True, exist_ok=True)

    holdings_path = Path(str(getattr(args, "holdings_path", "") or "").strip() or (Path("data") / "user_holdings.json"))
    fills_path = Path(str(getattr(args, "fills", "") or "").strip())
    if not str(fills_path):
        raise SystemExit("缺少 --fills：请提供真实成交明细文件（csv/json/jsonl）。")

    orders_path = str(getattr(args, "orders", "") or "").strip() or None
    ledger_path = Path(str(getattr(args, "ledger_path", "") or "").strip() or (Path("data") / "ledger_trades.jsonl"))
    apply = bool(getattr(args, "apply", False))

    fills_format = str(getattr(args, "fills_format", "") or "").strip().lower() or None
    encoding = str(getattr(args, "encoding", "") or "").strip() or None

    snap = load_user_holdings_snapshot(holdings_path)
    fills = load_trade_fills(fills_path, fmt=fills_format, encoding=encoding)
    orders = load_orders_next_open(orders_path) if orders_path else None
    existing_ids = load_ledger_trade_ids(ledger_path)

    res, next_snap, ledger_appends = reconcile_user_holdings(
        holdings_snapshot=snap,
        fills=fills,
        existing_ledger_trade_ids=existing_ids,
        orders_next_open=orders,
    )

    # 产物（不管 apply 与否都写）
    write_json(
        out_dir / "reconcile.json",
        {
            "generated_at": datetime.now().isoformat(),
            "apply": bool(apply),
            "inputs": {
                "holdings_path": str(holdings_path),
                "fills_path": str(fills_path),
                "fills_format": fills_format,
                "orders_path": orders_path,
                "ledger_path": str(ledger_path),
            },
            "result": res,
        },
    )
    write_json(out_dir / "user_holdings_next.json", next_snap)

    # ledger append preview（jsonl）
    try:
        lines = [json.dumps(sanitize_for_json(x), ensure_ascii=False, allow_nan=False) for x in ledger_appends]
        (out_dir / "ledger_trades_append.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        pass

    # report.md（KISS：先把关键信息写清楚）
    try:
        r = res if isinstance(res, dict) else {}
        warnings = r.get("warnings") if isinstance(r.get("warnings"), list) else []
        changes = r.get("changes") if isinstance(r.get("changes"), list) else []
        cash0 = r.get("cash_before")
        cash1 = r.get("cash_after")
        delta = None
        try:
            delta = (float(cash1) - float(cash0)) if (cash0 is not None and cash1 is not None) else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            delta = None

        lines = [
            "# reconcile\n",
            "",
            f"- generated_at: {datetime.now().isoformat()}",
            f"- apply: {bool(apply)}",
            f"- holdings_path: {str(holdings_path)}",
            f"- fills_path: {str(fills_path)}",
            f"- orders_path: {orders_path or ''}",
            f"- ledger_path: {str(ledger_path)}",
            f"- fills_total: {r.get('fills_total')}",
            f"- fills_new: {r.get('fills_new')}",
            f"- cash_before: {cash0}",
            f"- cash_after: {cash1}",
            f"- cash_delta: {delta}",
            "",
            "## changes (latest)\n",
            *[
                f"- {c.get('datetime') or ''} {c.get('side')} {c.get('symbol')} shares={c.get('shares')} price={c.get('price')} fee={c.get('fee')} tax={c.get('tax')} realized={c.get('realized_pnl')}"
                for c in changes[-30:]
                if isinstance(c, dict)
            ],
            "",
            "## warnings\n",
            *[f"- {w}" for w in warnings[:80]],
            "",
            "产物：",
            "- reconcile: reconcile.json",
            "- next snapshot: user_holdings_next.json",
            "- ledger append preview: ledger_trades_append.jsonl",
            "",
            "免责声明：研究工具输出，不构成投资建议；买卖自负。",
            "",
        ]
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        pass

    # --apply：写回快照 + 追加 ledger（危险操作必须显式开关）
    if apply:
        write_json(holdings_path, next_snap)
        if ledger_appends:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as f:
                for x in ledger_appends:
                    f.write(json.dumps(sanitize_for_json(x), ensure_ascii=False, allow_nan=False))
                    f.write("\n")

    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "reconcile", "apply": bool(apply)})
    run_config = _write_run_config(out_dir, args, note="reconcile", extra={"cmd": "reconcile", "apply": bool(apply)})
    try:
        from ..reporting import build_report_v1

        write_json(
            out_dir / "report.json",
            build_report_v1(
                cmd="reconcile",
                run_meta=run_meta,
                run_config=run_config,
                artifacts={
                    "reconcile": "reconcile.json",
                    "user_holdings_next": "user_holdings_next.json",
                    "ledger_append_preview": "ledger_trades_append.jsonl",
                    "report_md": "report.md",
                },
                counts=(
                    {"fills_total": res.get("fills_total"), "fills_new": res.get("fills_new")} if isinstance(res, dict) else None
                ),
                summary=(res if isinstance(res, dict) else None),
                warnings=(res.get("warnings") if isinstance(res, dict) and isinstance(res.get("warnings"), list) else None),
            ),
        )
    except (AttributeError):  # noqa: BLE001
        pass

    print(str(out_dir.resolve()))
    return 0



