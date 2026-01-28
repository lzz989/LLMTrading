from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _try_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (AttributeError):  # noqa: BLE001
        return None


def _try_get_git_head(root: Path) -> dict[str, Any] | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None

    head = _try_read_text(git_dir / "HEAD")
    if not head:
        return None

    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        commit = _try_read_text(git_dir / ref)
        if commit:
            return {"ref": ref, "commit": commit}

        # packed-refs 兜底
        packed = _try_read_text(git_dir / "packed-refs")
        if packed:
            for line in packed.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("^"):
                    continue
                parts = s.split()
                if len(parts) == 2 and parts[1] == ref:
                    return {"ref": ref, "commit": parts[0]}

        return {"ref": ref, "commit": None}

    # detached HEAD
    if len(head) >= 7:
        return {"ref": None, "commit": head}
    return None


def collect_run_meta(*, argv: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    产物可复现的最小 meta：
    - 版本（llm_trading / python）
    - git commit（若可取到）
    - 运行参数 argv
    """
    try:
        from . import __version__ as pkg_ver
    except (AttributeError):  # noqa: BLE001
        pkg_ver = "unknown"

    root = _project_root()
    git = _try_get_git_head(root)

    meta: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "tool": "llm_trading",
        "version": str(pkg_ver),
        "cwd": os.getcwd(),
        "argv": list(argv),
        "command": "python -m llm_trading " + " ".join(argv),
        "python": {
            "executable": sys.executable,
            "version": sys.version.splitlines()[0],
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": git,
    }
    if extra:
        meta["extra"] = extra
    return meta

