from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


_LOCK = threading.Lock()
_MEM_CACHE: dict[str, tuple[float, dict[str, str]]] = {}


def _cache_path(asset: str) -> Path:
    a = str(asset or "").strip().lower() or "unknown"
    safe = "".join(ch for ch in a if ch.isalnum() or ch in {"_", "-"})
    return Path("data") / "cache" / "universe" / f"{safe}_names.json"


def _read_cached_map(path: Path) -> dict[str, str] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw) if raw.strip() else {}
    except (AttributeError):  # noqa: BLE001
        return None
    if not isinstance(obj, dict):
        return None
    items = obj.get("items")
    if not isinstance(items, dict):
        return None
    out: dict[str, str] = {}
    for k, v in items.items():
        ks = str(k or "").strip().lower()
        vs = str(v or "").strip()
        if ks and vs:
            out[ks] = vs
    return out


def _write_cached_map(path: Path, m: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now().isoformat(), "items": m}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_universe_name_map(asset: str, *, ttl_hours: float = 24.0) -> dict[str, str]:
    """
    读取（并必要时刷新）symbol->name 的映射。

    - 缓存位置：data/cache/universe/{asset}_names.json
    - TTL 默认 24h：名称变化不频繁，别每次都去拉全量列表
    """
    a = str(asset or "").strip().lower()
    if a not in {"etf", "stock"}:
        return {}

    ttl = float(ttl_hours or 0.0)
    ttl = max(0.0, min(ttl, 24 * 30))
    now = time.time()

    with _LOCK:
        hit = _MEM_CACHE.get(a)
        if hit is not None:
            ts, m = hit
            if ttl > 0 and (now - float(ts)) <= ttl * 3600.0:
                return dict(m)

    path = _cache_path(a)
    if ttl > 0 and path.exists():
        try:
            age = now - float(path.stat().st_mtime)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            age = 9e18
        if age <= ttl * 3600.0:
            m2 = _read_cached_map(path)
            if m2 is not None:
                with _LOCK:
                    _MEM_CACHE[a] = (now, dict(m2))
                return dict(m2)

    # 刷新缓存（可能会触发一次全量列表拉取）
    m_new: dict[str, str] = {}
    try:
        if a == "stock":
            from .stock_scan import load_stock_universe

            items = load_stock_universe(include_st=True, include_bj=True)
            for it in items:
                sym = str(getattr(it, "symbol", "") or "").strip().lower()
                name = str(getattr(it, "name", "") or "").strip()
                if sym and name:
                    m_new[sym] = name
        else:
            from .etf_scan import load_etf_universe

            items = load_etf_universe(include_all_funds=True)
            for it in items:
                sym = str(getattr(it, "symbol", "") or "").strip().lower()
                name = str(getattr(it, "name", "") or "").strip()
                if sym and name:
                    m_new[sym] = name
    except (AttributeError):  # noqa: BLE001
        # 刷新失败：尽量退回旧缓存（如果有）
        m_old = _read_cached_map(path) if path.exists() else None
        if m_old is not None:
            with _LOCK:
                _MEM_CACHE[a] = (now, dict(m_old))
            return dict(m_old)
        return {}

    try:
        _write_cached_map(path, m_new)
    except (OSError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        pass

    with _LOCK:
        _MEM_CACHE[a] = (now, dict(m_new))
    return dict(m_new)


def get_symbol_name(asset: str, symbol: str) -> str | None:
    """
    给定 asset(etf/stock) + symbol(推荐 sh/sz/bj 前缀)，返回中文名；失败返回 None。
    """
    a = str(asset or "").strip().lower()
    sym = str(symbol or "").strip().lower()
    if a not in {"etf", "stock"} or not sym:
        return None

    m = load_universe_name_map(a, ttl_hours=24.0)
    name = m.get(sym)
    return str(name) if name else None
