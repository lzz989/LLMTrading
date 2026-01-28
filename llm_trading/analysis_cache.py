from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# bump when cached payload schema/meaning changes (avoid reusing stale derived results)
ANALYSIS_CACHE_VERSION = 2


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        out: dict[str, Any] = {}
        for k, v in x.items():
            out[str(k)] = _to_jsonable(v)
        return out
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def compute_params_hash(params: dict[str, Any]) -> str:
    payload = _to_jsonable(params)
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_name(s: str) -> str:
    # 文件名别整那些奇怪字符，Windows/WSL/容器里都能用
    return str(s).replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def cache_path(*, cache_dir: Path, symbol: str, last_date: str, params_hash: str) -> Path:
    h = str(params_hash)[:12]
    name = f"{_safe_name(symbol)}_{_safe_name(last_date)}_{h}.json"
    return cache_dir / name


def read_cached_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (OSError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        return None


def write_cached_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except (OSError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        try:
            Path(str(path) + ".tmp").unlink(missing_ok=True)
        except (OSError, AttributeError):  # noqa: BLE001
            pass
        return False
