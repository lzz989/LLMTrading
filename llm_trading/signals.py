from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from .json_utils import sanitize_for_json


SignalAction = Literal["entry", "watch", "avoid", "exit"]


def _fnum(x) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def _max_date_str(values: list[Any]) -> str | None:
    """
    取最大日期字符串（优先假设是 YYYY-MM-DD 这种可字典序比较的格式）。
    """
    best = None
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        if best is None or s > best:
            best = s
    return best


def signals_from_top_bbb(top_bbb: dict[str, Any]) -> dict[str, Any]:
    """
    scan-etf 的 top_bbb.json -> 统一 signals schema（研究用途）。

    约定：signals 是“让组合层吃”的中间产物：
    - 不承诺完美字段，只承诺 schema 稳定 + 能复现。
    - 原始 top_bbb 的关键字段仍会塞进 meta，别怕丢信息。
    """
    bbb_cfg = top_bbb.get("bbb") if isinstance(top_bbb, dict) else None
    bbb_cfg = bbb_cfg if isinstance(bbb_cfg, dict) else {}
    rank_h = int(bbb_cfg.get("rank_horizon") or 8)

    items0 = top_bbb.get("items") if isinstance(top_bbb, dict) else None
    items0 = items0 if isinstance(items0, list) else []

    out_items: list[dict[str, Any]] = []
    for it in items0:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue
        name = str(it.get("name") or "").strip()
        close = _fnum(it.get("close"))

        bbb = it.get("bbb") if isinstance(it.get("bbb"), dict) else {}
        ok = bool((bbb or {}).get("ok"))
        score = _fnum((bbb or {}).get("score"))
        why = str((bbb or {}).get("why") or "").strip() or None

        fwd = it.get("bbb_forward") if isinstance(it.get("bbb_forward"), dict) else {}
        st = fwd.get(f"{rank_h}w") if isinstance(fwd.get(f"{rank_h}w"), dict) else {}
        conf = _fnum(st.get("win_rate_shrunk"))
        trades = None
        try:
            trades = int(st.get("trades")) if st.get("trades") is not None else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            trades = None

        action: SignalAction = "entry" if ok else "watch"

        out_items.append(
            {
                "asset": "etf",
                "symbol": sym,
                "name": name,
                "action": action,
                "score": score,
                "confidence": conf,
                "confidence_ref": f"win_rate_shrunk_{rank_h}w" if conf is not None else None,
                "entry": {"price_ref": close, "price_ref_type": "close", "notes": why},
                "meta": {
                    "close": close,
                    "pct_chg": _fnum(it.get("pct_chg")),
                    "amount": _fnum(it.get("amount")),
                    "liquidity": it.get("liquidity"),
                    "levels": it.get("levels"),
                    "exit": it.get("exit"),
                    "bbb": bbb,
                    "bbb_forward": fwd,
                    "trades_ref": trades,
                },
                "tags": ["bbb"],
            }
        )

    as_of = None
    try:
        vals = []
        for it in items0:
            if not isinstance(it, dict):
                continue
            v = it.get("last_daily_date") or it.get("last_date")
            if v is not None:
                vals.append(v)
        as_of = _max_date_str(vals)
    except (AttributeError):  # noqa: BLE001
        as_of = None

    return sanitize_for_json(
        {
            "schema_version": 1,
            "generated_at": str(top_bbb.get("generated_at") or datetime.now().isoformat()),
            "as_of": as_of,
            "strategy": "bbb_etf",
            "source": {"type": "scan-etf", "file": "top_bbb.json"},
            "market_regime": bbb_cfg.get("market_regime"),
            "config": bbb_cfg,
            "counts": {"items": int(len(out_items))},
            "items": out_items,
        }
    )


def signals_from_stock_scan_results(
    results: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    rank_horizon_weeks: int = 8,
    market_regime: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    scan-stock 的结果列表 -> 统一 signals schema（研究用途）。

    说明：
    - 只输出“当前触发任何信号”的标的（trend/swing/dip），避免把全A塞爆 signals.json。
    - score 取触发信号里最大的 scores[signal]；confidence 取对应 rank_horizon 的 win_rate_shrunk 最大值。
    """
    rh = max(1, int(rank_horizon_weeks or 8))
    out_items: list[dict[str, Any]] = []

    for it in results or []:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue
        name = str(it.get("name") or "").strip()
        close = _fnum(it.get("close"))

        sigs = it.get("signals") if isinstance(it.get("signals"), dict) else {}
        tags = [k for k in ("trend", "swing", "dip") if bool(sigs.get(k))]
        if not tags:
            continue

        scores = it.get("scores") if isinstance(it.get("scores"), dict) else {}
        score_vals: list[float] = []
        for k in tags:
            v = _fnum(scores.get(k))
            if v is not None:
                score_vals.append(float(v))
        score = float(max(score_vals)) if score_vals else None

        forward = it.get("forward") if isinstance(it.get("forward"), dict) else {}
        best_conf: float | None = None
        for k in tags:
            st = forward.get(k) if isinstance(forward.get(k), dict) else {}
            st2 = st.get(f"{rh}w") if isinstance(st.get(f"{rh}w"), dict) else {}
            v = _fnum(st2.get("win_rate_shrunk"))
            if v is None:
                continue
            if best_conf is None or float(v) > float(best_conf):
                best_conf = float(v)

        action: SignalAction = "entry"
        out_items.append(
            {
                "asset": "stock",
                "symbol": sym,
                "name": name,
                "action": action,
                "score": score,
                "confidence": best_conf,
                "confidence_ref": f"win_rate_shrunk_{rh}w" if best_conf is not None else None,
                "entry": {"price_ref": close, "price_ref_type": "close", "notes": None},
                "meta": {
                    "date": it.get("date"),
                    "close": close,
                    "amount": _fnum(it.get("amount")),
                    "levels": it.get("levels"),
                    "daily": it.get("daily"),
                    "filters": it.get("filters"),
                    "signals": sigs,
                    "scores": scores,
                    "forward": forward,
                },
                "tags": tags,
            }
        )

    as_of = None
    try:
        as_of = _max_date_str([it.get("date") for it in results or [] if isinstance(it, dict)])
    except (AttributeError):  # noqa: BLE001
        as_of = None

    return sanitize_for_json(
        {
            "schema_version": 1,
            "generated_at": str(generated_at or datetime.now().isoformat()),
            "as_of": as_of,
            "strategy": "stock_scan",
            "source": {"type": "scan-stock", "file": "scan-stock"},
            "market_regime": market_regime or None,
            "config": config or {},
            "counts": {"items": int(len(out_items))},
            "items": out_items,
        }
    )
