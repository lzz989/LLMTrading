from __future__ import annotations

from pathlib import Path

from ..commodity_chain import render_chain_md, scan_commodity_chain
from ..pipeline import write_json


def cmd_commodity_chain(args) -> int:
    out_dir = Path(getattr(args, "out_dir", None) or Path("outputs") / "agents")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_days = int(getattr(args, "min_days", 80) or 80)
    top_k = int(getattr(args, "top_k", 3) or 3)
    top_k = max(1, min(top_k, 10))

    report = scan_commodity_chain(min_days=min_days, top_k=top_k)
    write_json(out_dir / "commodity_chain_scan.json", report)
    md = render_chain_md(report)
    (out_dir / "commodity_chain_scan.md").write_text(md, encoding="utf-8")
    return 0
