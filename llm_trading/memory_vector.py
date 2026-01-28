from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig
from .llm_client import LlmError, openai_embeddings
from .memory_store import MemoryPaths, ensure_memory_layout, load_user_profile


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_ts() -> float:
    return time.time()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, AttributeError):  # noqa: BLE001
        return ""


@dataclass(frozen=True)
class Chunk:
    path: str
    source: str
    start_line: int
    end_line: int
    text: str
    hash: str


def chunk_markdown(
    content: str,
    *,
    chunk_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[tuple[int, int, str]]:
    """
    简单 chunk：按行累计字符数（约等于 token*4），并在 chunk 边界保留 overlap。
    返回 (start_line, end_line, text) 列表。
    """

    lines = (content or "").split("\n")
    if not lines:
        return []

    max_chars = max(32, int(chunk_tokens) * 4)
    overlap_chars = max(0, int(overlap_tokens) * 4)

    chunks: list[tuple[int, int, str]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        start_line = current[0][0]
        end_line = current[-1][0]
        text = "\n".join([x[1] for x in current]).strip("\n")
        if text.strip():
            chunks.append((start_line, end_line, text))

    def carry_overlap() -> None:
        nonlocal current, current_chars
        if overlap_chars <= 0 or not current:
            current = []
            current_chars = 0
            return
        acc = 0
        kept: list[tuple[int, str]] = []
        for ln, line in reversed(current):
            acc += len(line) + 1
            kept.append((ln, line))
            if acc >= overlap_chars:
                break
        kept.reverse()
        current = kept
        current_chars = sum(len(line) + 1 for _, line in current)

    for i, raw in enumerate(lines):
        ln = i + 1
        line = raw if raw is not None else ""
        # 超长行切片，避免单行把 chunk 撑爆
        if line:
            segments = [line[j : j + max_chars] for j in range(0, len(line), max_chars)]
        else:
            segments = [""]
        for seg in segments:
            seg_size = len(seg) + 1
            if current and current_chars + seg_size > max_chars:
                flush()
                carry_overlap()
            current.append((ln, seg))
            current_chars += seg_size

    flush()
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        av = float(a[i] or 0.0)
        bv = float(b[i] or 0.0)
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:  # noqa: BLE001
        return str(p)


def _iter_index_documents(paths: MemoryPaths, *, project_root: Path) -> Iterable[dict[str, Any]]:
    """
    统一把可检索内容抽象成“文档”：
    - long_term_md: 全文
    - daily/*.md: 全文
    - archive/**/*.md: 全文（归档不进 prompt，但需要可检索）
    - user_profile.json: 只取 preferences（避免 events 变动导致频繁重嵌）
    """

    ensure_memory_layout(paths)

    # 1) long-term
    if paths.long_term_md.exists():
        text = _read_text(paths.long_term_md)
        yield {
            "path": _relpath(paths.long_term_md, project_root),
            "source": "memory_long_term",
            "text": text,
            "hash": _sha256(text),
            "mtime": float(paths.long_term_md.stat().st_mtime) if paths.long_term_md.exists() else 0.0,
            "size": int(paths.long_term_md.stat().st_size) if paths.long_term_md.exists() else 0,
        }

    # 2) daily
    try:
        for fp in sorted(paths.daily_dir.glob("*.md")):
            text = _read_text(fp)
            yield {
                "path": _relpath(fp, project_root),
                "source": "memory_daily",
                "text": text,
                "hash": _sha256(text),
                "mtime": float(fp.stat().st_mtime) if fp.exists() else 0.0,
                "size": int(fp.stat().st_size) if fp.exists() else 0,
            }
    except Exception:  # noqa: BLE001
        pass

    # 2.5) archive (rollup)
    archive_dir = paths.base_dir / "archive"
    try:
        for fp in sorted(archive_dir.rglob("*.md")) if archive_dir.exists() else []:
            text = _read_text(fp)
            yield {
                "path": _relpath(fp, project_root),
                "source": "memory_archive",
                "text": text,
                "hash": _sha256(text),
                "mtime": float(fp.stat().st_mtime) if fp.exists() else 0.0,
                "size": int(fp.stat().st_size) if fp.exists() else 0,
            }
    except Exception:  # noqa: BLE001
        pass

    # 3) profile (preferences only)
    profile = load_user_profile(paths)
    prefs = profile.get("preferences") if isinstance(profile, dict) else None
    prefs_text = json.dumps(prefs, ensure_ascii=False, indent=2) if isinstance(prefs, dict) else "{}"
    yield {
        "path": _relpath(paths.user_profile_json, project_root),
        "source": "profile",
        "text": prefs_text,
        "hash": _sha256(prefs_text),
        "mtime": float(paths.user_profile_json.stat().st_mtime) if paths.user_profile_json.exists() else 0.0,
        "size": int(paths.user_profile_json.stat().st_size) if paths.user_profile_json.exists() else 0,
    }


def _load_index(paths: MemoryPaths) -> dict[str, Any] | None:
    try:
        raw = paths.vector_index_json.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError, TypeError, AttributeError):  # noqa: BLE001
        return None


def _write_index(paths: MemoryPaths, index: dict[str, Any]) -> None:
    paths.vector_index_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(paths.vector_index_json) + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(paths.vector_index_json)


def build_or_update_vector_index(
    cfg: AppConfig,
    *,
    paths: MemoryPaths,
    chunk_tokens: int = 400,
    overlap_tokens: int = 80,
    force: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    构建/增量更新 vector index（JSON 存储，避免 sqlite3 依赖）。
    需要配置 EMBEDDINGS_API_KEY / EMBEDDINGS_MODEL / EMBEDDINGS_BASE_URL。
    """

    emb_cfg = cfg.embeddings()
    if not emb_cfg:
        raise RuntimeError("embeddings 未配置：请设置 EMBEDDINGS_API_KEY（以及 EMBEDDINGS_MODEL/BASE_URL 可选）。")

    ensure_memory_layout(paths)
    old = _load_index(paths)
    old_files = {f.get("path"): f for f in (old.get("files") or []) if isinstance(f, dict)} if isinstance(old, dict) else {}
    old_chunks = old.get("chunks") if isinstance(old, dict) else None
    old_chunks = old_chunks if isinstance(old_chunks, list) else []

    # embedding cache：chunk_hash -> embedding
    cache: dict[str, list[float]] = {}
    for it in old_chunks:
        if not isinstance(it, dict):
            continue
        h = str(it.get("hash") or "").strip()
        emb = it.get("embedding")
        if h and isinstance(emb, list) and emb:
            try:
                cache[h] = [float(x) for x in emb]
            except (TypeError, ValueError):  # noqa: BLE001
                continue

    docs = list(_iter_index_documents(paths, project_root=cfg.project_root))
    new_files: list[dict[str, Any]] = []
    new_chunks: list[dict[str, Any]] = []

    # 先把没变的文件 chunk 直接搬过去，变的文件重算
    reused_paths = set()
    if not force and old_files and old_chunks:
        for d in docs:
            p = str(d.get("path") or "")
            h = str(d.get("hash") or "")
            prev = old_files.get(p)
            if prev and str(prev.get("hash") or "") == h:
                reused_paths.add(p)
                for it in old_chunks:
                    if isinstance(it, dict) and str(it.get("path") or "") == p:
                        new_chunks.append(it)

    # 对变化的文件重新 chunk + 嵌入
    need_embed: list[tuple[str, str]] = []  # (chunk_hash, text)
    pending_chunks: list[dict[str, Any]] = []
    for d in docs:
        p = str(d.get("path") or "")
        src = str(d.get("source") or "")
        text = str(d.get("text") or "")
        new_files.append(
            {
                "path": p,
                "source": src,
                "hash": str(d.get("hash") or ""),
                "mtime": float(d.get("mtime") or 0.0),
                "size": int(d.get("size") or 0),
            }
        )

        if p in reused_paths:
            continue

        if p.endswith(".json"):
            chunks0 = [(1, max(1, text.count("\n") + 1), text)]
        else:
            chunks0 = chunk_markdown(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

        for start_ln, end_ln, ctext in chunks0:
            ctext2 = (ctext or "").strip()
            if not ctext2:
                continue
            ch = _sha256(ctext2)
            cid = _sha256(f"{p}:{start_ln}:{end_ln}:{ch}")
            if ch not in cache:
                need_embed.append((ch, ctext2))
            pending_chunks.append(
                {
                    "id": cid,
                    "path": p,
                    "source": src,
                    "start_line": int(start_ln),
                    "end_line": int(end_ln),
                    "hash": ch,
                    "text": ctext2,
                    # embedding later
                }
            )

    # 去重，避免相同 chunk 反复嵌入
    if need_embed:
        uniq: dict[str, str] = {}
        for h, t in need_embed:
            if h not in uniq:
                uniq[h] = t
        need_embed = list(uniq.items())

    if verbose:
        print(
            json.dumps(
                {
                    "docs": len(docs),
                    "reused_files": len(reused_paths),
                    "need_embed_chunks": len(need_embed),
                    "existing_cache_chunks": len(cache),
                },
                ensure_ascii=False,
            )
        )

    # batch embeddings
    BATCH = 64
    for i in range(0, len(need_embed), BATCH):
        batch = need_embed[i : i + BATCH]
        texts = [t for _, t in batch]
        try:
            vecs = openai_embeddings(emb_cfg, inputs=texts)
        except LlmError as exc:
            raise RuntimeError(f"embeddings 调用失败：{exc}") from exc
        if len(vecs) != len(batch):
            raise RuntimeError(f"embeddings 返回数量异常：expect={len(batch)} got={len(vecs)}")
        for (h, _), v in zip(batch, vecs, strict=False):
            cache[h] = v

    # assemble new chunks (reused + rebuilt)
    for it in pending_chunks:
        h = str(it.get("hash") or "")
        emb = cache.get(h) or []
        it2 = dict(it)
        it2["embedding"] = emb
        new_chunks.append(it2)

    index = {
        "schema": "llm_trading.memory_vector_index.v1",
        "updated_at": _now_ts(),
        "embedding": {
            "provider": "openai_compat",
            "model": emb_cfg.model,
            "base_url": emb_cfg.base_url,
        },
        "chunking": {"chunk_tokens": int(chunk_tokens), "overlap_tokens": int(overlap_tokens)},
        "files": new_files,
        "chunks": new_chunks,
    }
    _write_index(paths, index)
    return index


def vector_search(
    cfg: AppConfig,
    *,
    paths: MemoryPaths,
    query: str,
    mode: str = "vector",
    max_results: int = 8,
    min_score: float = 0.15,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    reindex: bool = False,
) -> list[dict[str, Any]]:
    """
    mode:
      - vector: 纯向量相似度
      - hybrid: 向量 + substring 关键词（简易混合）
    """

    q = (query or "").strip()
    if not q:
        return []

    emb_cfg = cfg.embeddings()
    if not emb_cfg:
        # 没 embeddings 就别装，直接返回空（上层可降级 keyword）
        return []

    # 索引缺失/需要重建
    idx = _load_index(paths)
    if reindex or not isinstance(idx, dict):
        idx = build_or_update_vector_index(cfg, paths=paths, force=bool(reindex))

    chunks = idx.get("chunks") if isinstance(idx, dict) else None
    chunks = chunks if isinstance(chunks, list) else []

    # embed query
    try:
        qvecs = openai_embeddings(emb_cfg, inputs=[q])
    except LlmError:
        return []
    qvec = qvecs[0] if qvecs and isinstance(qvecs[0], list) else []
    if not qvec:
        return []

    # vector ranking
    scored: list[tuple[float, dict[str, Any]]] = []
    q_lower = q.lower()
    for it in chunks:
        if not isinstance(it, dict):
            continue
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        try:
            vscore = cosine_similarity(qvec, [float(x) for x in emb])
        except (TypeError, ValueError):  # noqa: BLE001
            continue

        tscore = 0.0
        if mode == "hybrid":
            try:
                text = str(it.get("text") or "")
                occ = text.lower().count(q_lower) if q_lower else 0
                tscore = min(1.0, float(occ) / 3.0) if occ > 0 else 0.0
            except Exception:  # noqa: BLE001
                tscore = 0.0

        score = vscore if mode != "hybrid" else (float(vector_weight) * vscore + float(text_weight) * tscore)
        if score < float(min_score):
            continue
        scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, it in scored[: int(max_results)]:
        text = str(it.get("text") or "")
        snippet = text[:700]
        out.append(
            {
                "path": str(it.get("path") or ""),
                "source": str(it.get("source") or ""),
                "start_line": int(it.get("start_line") or 1),
                "end_line": int(it.get("end_line") or 1),
                "score": float(score),
                "snippet": snippet,
            }
        )
    return out
