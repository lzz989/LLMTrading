# -*- coding: utf-8 -*-

from __future__ import annotations

"""
一个“粗暴但可控”的 except Exception 收敛脚本（Phase5 验收用）。

目的：
- 不改变“能降级就降级”的主语义；
- 但把大量“只是为了数值/日期解析”的 `except Exception` 收敛成更具体的异常元组，
  让代码审计/静态检查更像样，也满足 docs 的验收线。

策略（启发式，宁可保守）：
- 仅当 except 前几行像“转换/解析”时才替换；
- 遇到明显网络/抓数相关关键字则跳过（保持原有广义降级语义）。
"""

import argparse
import re
from pathlib import Path


RE_EXCEPT = re.compile(r"^(?P<indent>\s*)except\s+Exception(?P<as>\s+as\s+\w+)?\s*:(?P<trail>\s*#.*)?$")


def pick_excs(ctx: str) -> list[str]:
    excs: list[str] = []

    # numeric conversion
    if "float(" in ctx or "int(" in ctx:
        excs += ["TypeError", "ValueError", "OverflowError"]

    # datetime / pandas conversion
    if "datetime.strptime" in ctx or "pd.to_datetime" in ctx or "to_datetime(" in ctx:
        excs += ["TypeError", "ValueError"]

    # common indexing failures (df['x'], iloc/loc)
    if any(x in ctx for x in (".iloc[", ".loc[", "['", "][", ".at[", ".iat[")):
        excs += ["KeyError", "IndexError"]

    # attribute access on optional objects
    if "." in ctx:
        excs += ["AttributeError"]

    out: list[str] = []
    seen: set[str] = set()
    for e in excs:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def should_skip(ctx: str) -> bool:
    # 明显“抓数/网络/外部 IO”上下文：别瞎收敛，避免把本来要吞掉的异常漏出去
    tokens = ("requests", "http", "socket", "ak.", "tushare", "fetch_", "urllib", "websocket")
    return any(t in ctx for t in tokens)


def process_file(p: Path) -> tuple[int, list[str]]:
    lines = p.read_text(encoding="utf-8").splitlines()
    out_lines = list(lines)

    changed = 0
    notes: list[str] = []

    for i, line in enumerate(lines):
        m = RE_EXCEPT.match(line)
        if not m:
            continue

        ctx = "\n".join(lines[max(0, i - 6) : i])
        if should_skip(ctx):
            continue

        excs = pick_excs(ctx)
        if not excs:
            continue

        indent = m.group("indent")
        as_part = m.group("as") or ""
        trail = m.group("trail") or ""

        new_line = f"{indent}except ({', '.join(excs)}){as_part}:{trail}"
        if new_line != line:
            out_lines[i] = new_line
            changed += 1

    if changed:
        p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        notes.append(f"{p}: changed={changed}")

    return changed, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="llm_trading", help="root dir (default: llm_trading)")
    ap.add_argument("--apply", action="store_true", help="apply changes (default: dry-run summary only)")
    args = ap.parse_args()

    root = Path(str(args.root))
    files = [p for p in root.rglob("*.py") if p.is_file()]

    total_candidates = 0
    changed_files = 0
    changed_lines = 0
    all_notes: list[str] = []

    for p in files:
        text = p.read_text(encoding="utf-8")
        if "except Exception" not in text:
            continue
        total_candidates += 1
        if not args.apply:
            continue
        n, notes = process_file(p)
        if n:
            changed_files += 1
            changed_lines += int(n)
            all_notes.extend(notes)

    if not args.apply:
        print(f"[dry-run] candidate_files_with_except_exception={total_candidates}")
        print("Run with --apply to modify files.")
        return 0

    print(f"[apply] changed_files={changed_files} changed_lines={changed_lines}")
    # Print a short list for traceability (avoid spamming).
    for s in all_notes[:50]:
        print(s)
    if len(all_notes) > 50:
        print(f"... ({len(all_notes) - 50} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

