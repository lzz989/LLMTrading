# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return int(default)


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if v != v:  # NaN
        return None
    return float(v)


def _symbol_name(sym: str, name: str | None) -> str:
    s = str(sym or "").strip()
    n = str(name or "").strip()
    if not n:
        n = "名称未知"
    return f"{s}（{n}）" if s else f"{n}"


def _top_entries(obj: Any, *, max_n: int) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    items = obj.get("items")
    items = items if isinstance(items, list) else []
    rows: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if str(it.get("action") or "").strip().lower() != "entry":
            continue
        rows.append(it)
    rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for it in rows[: max(1, int(max_n))]:
        meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
        entry = it.get("entry") if isinstance(it.get("entry"), dict) else {}
        out.append(
            {
                "symbol": it.get("symbol"),
                "name": it.get("name"),
                "score": _as_float(it.get("score")),
                "close": _as_float(meta.get("close")) if isinstance(meta, dict) else None,
                "entry_ref": _as_float(entry.get("price_ref")) if isinstance(entry, dict) else None,
                "tags": it.get("tags") if isinstance(it.get("tags"), list) else None,
            }
        )
    return out


def _top_plans(obj: Any, *, max_n: int) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    plan = obj.get("position_plan") if isinstance(obj.get("position_plan"), dict) else {}
    plans = plan.get("plans") if isinstance(plan, dict) else []
    plans = plans if isinstance(plans, list) else []
    rows = [p for p in plans if isinstance(p, dict) and bool(p.get("ok"))]
    rows.sort(key=lambda x: float(x.get("position_yuan") or 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for p in rows[: max(1, int(max_n))]:
        out.append(
            {
                "symbol": p.get("symbol"),
                "name": p.get("name"),
                "entry": _as_float(p.get("entry")),
                "stop": _as_float(p.get("stop")),
                "stop_ref": p.get("stop_ref"),
                "shares": _as_int(p.get("shares")),
                "position_yuan": _as_float(p.get("position_yuan")),
            }
        )
    return out


def build_daily_brief(
    *,
    run_dir: Path,
    max_candidates: int = 6,
    max_portfolio: int = 3,
    max_warnings: int = 10,
) -> str:
    signals = _load_json(run_dir / "signals.json")
    signals_stock = _load_json(run_dir / "signals_stock.json")
    rebalance = _load_json(run_dir / "rebalance_user.json")
    orders = _load_json(run_dir / "orders_next_open.json")
    holdings = _load_json(run_dir / "holdings_user.json")

    etf_top = _top_entries(signals, max_n=max_candidates)
    stock_top = _top_entries(signals_stock, max_n=max_candidates)
    plans = _top_plans(rebalance, max_n=max_portfolio)

    as_of_sig = None
    as_of_hold = None
    if isinstance(signals, dict):
        as_of_sig = signals.get("as_of") or signals.get("generated_at")
    if isinstance(holdings, dict):
        as_of_hold = holdings.get("as_of")

    regime = None
    if isinstance(holdings, dict):
        regime = (holdings.get("market_regime") or {}).get("label") if isinstance(holdings.get("market_regime"), dict) else None

    portfolio = holdings.get("portfolio") if isinstance(holdings, dict) else None

    warns: list[str] = []
    if isinstance(rebalance, dict):
        w = rebalance.get("warnings")
        if isinstance(w, list):
            warns.extend([str(x) for x in w if str(x).strip()])
    if isinstance(portfolio, dict):
        w2 = portfolio.get("warnings")
        if isinstance(w2, list):
            warns.extend([str(x) for x in w2 if str(x).strip()])
    warns = warns[: max(0, int(max_warnings))]

    lines: list[str] = []
    lines.append("# 每日量化简报（club-style）")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- data_as_of(signals): {as_of_sig}")
    lines.append(f"- data_as_of(holdings): {as_of_hold}")
    lines.append(f"- market_regime: {regime}")
    lines.append("")

    if isinstance(portfolio, dict):
        eq = _as_float(portfolio.get("equity_yuan"))
        exp = _as_float(portfolio.get("exposure_pct"))
        cash_pct = _as_float(portfolio.get("cash_pct"))
        risk_pct = _as_float(portfolio.get("risk_to_stop_pct_equity"))
        lines.append("## 账户概览")
        lines.append("")
        lines.append(f"- equity_yuan: {eq}")
        lines.append(f"- exposure_pct: {exp}")
        lines.append(f"- cash_pct: {cash_pct}")
        lines.append(f"- risk_to_stop_pct_equity: {risk_pct}")
        lines.append("")

    lines.append("## 候选池（ETF Top）")
    lines.append("")
    if etf_top:
        for it in etf_top:
            sym = str(it.get("symbol") or "").strip()
            name = _symbol_name(sym, str(it.get("name") or "").strip() or None)
            score = it.get("score")
            close = it.get("close")
            entry_ref = it.get("entry_ref")
            lines.append(f"- {name} score={score} close={close} entry_ref={entry_ref}")
    else:
        lines.append("- （空）")
    lines.append("")

    lines.append("## 候选池（个股 Top）")
    lines.append("")
    if stock_top:
        for it in stock_top:
            sym = str(it.get("symbol") or "").strip()
            name = _symbol_name(sym, str(it.get("name") or "").strip() or None)
            score = it.get("score")
            close = it.get("close")
            entry_ref = it.get("entry_ref")
            lines.append(f"- {name} score={score} close={close} entry_ref={entry_ref}")
    else:
        lines.append("- （空）")
    lines.append("")

    lines.append("## 模拟持仓（<=3，规则版）")
    lines.append("")
    if plans:
        for p in plans:
            sym = str(p.get("symbol") or "").strip()
            name = _symbol_name(sym, str(p.get("name") or "").strip() or None)
            entry = p.get("entry")
            stop = p.get("stop")
            stop_ref = p.get("stop_ref")
            sh = p.get("shares")
            pos = p.get("position_yuan")
            lines.append(f"- {name} entry={entry} stop={stop} stop_ref={stop_ref} shares={sh} pos_yuan={pos}")
    else:
        lines.append("- （空）")
    lines.append("")

    lines.append("## 执行草案（orders_next_open）")
    lines.append("")
    if isinstance(orders, list) and orders:
        for o in orders:
            if not isinstance(o, dict):
                continue
            side = str(o.get("side") or "").strip().lower()
            sym = str(o.get("symbol") or "").strip()
            name = _symbol_name(sym, str(o.get("name") or "").strip() or None)
            sh = o.get("shares")
            reason = str(o.get("reason") or "").strip()
            lines.append(f"- {side} {name} shares={sh} reason={reason}")
    else:
        lines.append("- （空）")
    lines.append("")

    lines.append("## 风控/警告")
    lines.append("")
    if warns:
        for w in warns:
            lines.append(f"- {w}")
    else:
        lines.append("- （无）")
    lines.append("")

    lines.append("## 免责声明")
    lines.append("")
    lines.append("研究工具输出，不构成投资建议；买卖自负。")
    lines.append("")
    return "\n".join(lines)


def cmd_daily_brief(args: argparse.Namespace) -> int:
    run_dir = Path(str(getattr(args, "run_dir", "") or "").strip())
    if not run_dir.exists():
        raise SystemExit(f"run_dir 不存在：{run_dir}")

    max_candidates = _as_int(getattr(args, "max_candidates", 6), default=6)
    max_portfolio = _as_int(getattr(args, "max_portfolio", 3), default=3)
    max_warnings = _as_int(getattr(args, "max_warnings", 10), default=10)

    out_raw = str(getattr(args, "out", "") or "").strip()
    if out_raw:
        out_path = Path(out_raw)
    else:
        out_path = Path("outputs") / "agents" / "daily_brief.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = build_daily_brief(
        run_dir=run_dir,
        max_candidates=max_candidates,
        max_portfolio=max_portfolio,
        max_warnings=max_warnings,
    )
    out_path.write_text(md, encoding="utf-8")
    print(str(out_path.resolve()))
    return 0

