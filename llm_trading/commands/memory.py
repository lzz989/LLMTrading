from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import load_config
from ..memory_store import (
    archive_daily_memory,
    append_daily_memory,
    append_long_term_memory,
    build_prompt_memory_context,
    keyword_search_memory,
    resolve_memory_paths,
    sync_trade_rules_from_user_holdings,
)


def _parse_value(raw: str) -> Any:
    s = (raw or "").strip()
    if not s:
        return ""
    # 尝试按 JSON 解析：true/false/null/数字/对象/数组
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return s


def cmd_memory(args: argparse.Namespace) -> int:
    paths = resolve_memory_paths()
    subcmd = str(getattr(args, "memory_cmd", "") or "").strip().lower()

    if subcmd == "status":
        cfg = load_config()
        # 少说废话，给路径 + 最近文件数就行。
        daily_files = []
        try:
            daily_files = sorted(paths.daily_dir.glob("*.md"))
        except Exception:  # noqa: BLE001
            daily_files = []
        payload = {
            "memory_dir": str(paths.base_dir),
            "long_term_md": str(paths.long_term_md),
            "daily_dir": str(paths.daily_dir),
            "daily_files": len(daily_files),
            "user_profile_json": str(paths.user_profile_json),
            "ledger_jsonl": str(paths.ledger_jsonl),
            "vector_index_json": str(paths.vector_index_json),
            "embeddings_enabled": bool(cfg.embeddings()),
        }
        if paths.vector_index_json.exists():
            try:
                idx = json.loads(paths.vector_index_json.read_text(encoding="utf-8"))
                chunks = idx.get("chunks") if isinstance(idx, dict) else None
                payload["vector_index_chunks"] = int(len(chunks)) if isinstance(chunks, list) else 0
            except Exception:  # noqa: BLE001
                payload["vector_index_chunks"] = None
        if bool(getattr(args, "json", False)):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                "\n".join(
                    [
                        f"memory_dir: {payload['memory_dir']}",
                        f"long_term_md: {payload['long_term_md']}",
                        f"daily_dir: {payload['daily_dir']} (files={payload['daily_files']})",
                        f"user_profile_json: {payload['user_profile_json']}",
                        f"ledger_jsonl: {payload['ledger_jsonl']}",
                        f"vector_index_json: {payload['vector_index_json']}"
                        + (
                            f" (chunks={payload.get('vector_index_chunks')})"
                            if payload.get("vector_index_chunks") is not None
                            else ""
                        ),
                        f"embeddings_enabled: {payload['embeddings_enabled']}",
                    ]
                )
            )
        return 0

    if subcmd == "remember":
        text = str(getattr(args, "text", "") or "").strip()
        to_daily = bool(getattr(args, "daily", False))

        # 结构化更新：--set k=v
        updates: dict[str, Any] = {}
        sets = getattr(args, "set", None)
        if isinstance(sets, list):
            for raw in sets:
                s = str(raw or "").strip()
                if not s or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                if not k:
                    continue
                updates[k] = _parse_value(v)

        source = {"type": "cli", "cmd": "memory remember"}

        # 写 text（偏好/备注）到文件记忆：每日 or 长期
        if text:
            if to_daily:
                append_daily_memory(paths, text=text, title=str(getattr(args, "title", "") or "").strip() or None, source=source)
            else:
                append_long_term_memory(paths, text=text, source=source)

        # 写结构化偏好
        if updates:
            from ..memory_store import update_user_profile

            update_user_profile(paths, updates=updates, source=source)

        if not text and not updates:
            raise SystemExit("啥也没写：至少给 --text 或 --set k=v。")
        return 0

    if subcmd == "search":
        cfg = load_config()
        q = str(getattr(args, "query", "") or "").strip()
        mode = str(getattr(args, "mode", "") or "keyword").strip().lower()
        max_results_raw = getattr(args, "max_results", None)
        max_results = int(max_results_raw) if max_results_raw is not None else 20
        context_lines_raw = getattr(args, "context_lines", None)
        context_lines = int(context_lines_raw) if context_lines_raw is not None else 2
        min_score_raw = getattr(args, "min_score", None)
        min_score = float(min_score_raw) if min_score_raw is not None else 0.15
        reindex = bool(getattr(args, "reindex", False))

        res: list[dict[str, Any]] = []
        if mode in {"vector", "hybrid"}:
            try:
                from ..memory_vector import vector_search

                res = vector_search(
                    cfg,
                    paths=paths,
                    query=q,
                    mode=mode,
                    max_results=max_results,
                    min_score=min_score,
                    reindex=reindex,
                )
            except Exception:  # noqa: BLE001
                res = []

        # fallback：没 embeddings / 向量检索失败，就老老实实关键词。
        if not res:
            res = keyword_search_memory(
                paths,
                query=q,
                max_results=max_results,
                context_lines=context_lines,
            )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if subcmd == "index":
        cfg = load_config()
        try:
            from ..memory_vector import build_or_update_vector_index

            idx = build_or_update_vector_index(
                cfg,
                paths=paths,
                force=bool(getattr(args, "force", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
            chunks = idx.get("chunks") if isinstance(idx, dict) else None
            print(
                json.dumps(
                    {
                        "ok": True,
                        "vector_index_json": str(paths.vector_index_json),
                        "chunks": int(len(chunks)) if isinstance(chunks, list) else None,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"index failed: {exc}") from exc
        return 0

    if subcmd == "export-prompt":
        txt = build_prompt_memory_context(
            paths,
            include_long_term=not bool(getattr(args, "no_long_term", False)),
            include_profile=not bool(getattr(args, "no_profile", False)),
            include_daily_days=int(getattr(args, "daily_days", 2) or 2),
            max_chars=int(getattr(args, "max_chars", 6000) or 6000),
        )
        print(txt)
        return 0

    if subcmd == "sync":
        holdings_path = Path(str(getattr(args, "holdings_path", "") or "data/user_holdings.json"))
        out = sync_trade_rules_from_user_holdings(
            paths,
            holdings_path=holdings_path,
            source={"type": "cli", "cmd": "memory sync", "holdings_path": str(holdings_path)},
        )
        if out is None:
            raise SystemExit(f"未找到有效 trade_rules：{holdings_path}")
        print("ok")
        return 0

    if subcmd == "archive":
        keep_days = int(getattr(args, "keep_days", 7) or 7)
        group = str(getattr(args, "group", "month") or "month").strip().lower()
        apply = bool(getattr(args, "apply", False))
        plan = archive_daily_memory(paths, keep_days=keep_days, group=group, apply=apply)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    raise SystemExit(f"未知 memory 子命令：{subcmd}")
