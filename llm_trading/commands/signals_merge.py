from __future__ import annotations

import argparse
from pathlib import Path


def cmd_signals_merge(args: argparse.Namespace) -> int:
    """
    合并多份 signals.json（多策略聚合；研究用途）。
    """
    import json

    from ..pipeline import write_json
    from ..signals_merge import merge_signals_files, parse_priority, parse_strategy_weights

    ins = list(getattr(args, "inputs", []) or [])
    paths: list[Path] = []
    for p in ins:
        s = str(p or "").strip()
        if not s:
            continue
        paths.append(Path(s))
    if not paths:
        raise SystemExit("缺少 --in（可重复传多次）")

    for p in paths:
        if not p.exists():
            raise SystemExit(f"找不到输入：{p}")

    weights = parse_strategy_weights(str(getattr(args, "weights", "") or ""))
    priority = parse_priority(str(getattr(args, "priority", "") or ""))
    conflict = str(getattr(args, "conflict", "risk_first") or "risk_first").strip().lower()
    if conflict not in {"risk_first", "priority", "vote"}:
        conflict = "risk_first"

    out_obj = merge_signals_files(
        paths,
        conflict=conflict,  # type: ignore[arg-type]
        weights=weights,
        priority=priority,
        top_k=int(getattr(args, "top_k", 0) or 0),
    )

    if getattr(args, "out", None):
        out_path = Path(str(getattr(args, "out")))
        write_json(out_path, out_obj)
        print(str(out_path.resolve()))
    else:
        print(json.dumps(out_obj, ensure_ascii=False, indent=2, allow_nan=False))
    return 0

