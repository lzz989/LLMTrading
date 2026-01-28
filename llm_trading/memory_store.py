from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .json_utils import sanitize_for_json


@dataclass(frozen=True)
class MemoryPaths:
    base_dir: Path
    long_term_md: Path
    daily_dir: Path
    user_profile_json: Path
    ledger_jsonl: Path
    vector_index_json: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_memory_paths(*, project_root: Path | None = None) -> MemoryPaths:
    """
    记忆库默认放到 data/（gitignore），避免把个人偏好/复盘塞进 git。
    可用环境变量覆盖：
    - LLM_TRADING_MEMORY_DIR：记忆目录（默认 data/memory）
    - LLM_TRADING_PROFILE_PATH：偏好 profile JSON（默认 data/user_profile.json）
    """

    root = (project_root or _repo_root()).resolve()
    mem_dir = os.getenv("LLM_TRADING_MEMORY_DIR", "").strip()
    base_dir = (Path(mem_dir) if mem_dir else (root / "data" / "memory")).resolve()

    profile_path = os.getenv("LLM_TRADING_PROFILE_PATH", "").strip()
    user_profile_json = (Path(profile_path) if profile_path else (root / "data" / "user_profile.json")).resolve()

    return MemoryPaths(
        base_dir=base_dir,
        long_term_md=base_dir / "MEMORY.md",
        daily_dir=base_dir / "daily",
        user_profile_json=user_profile_json,
        ledger_jsonl=base_dir / "ledger.jsonl",
        vector_index_json=base_dir / "vector_index.json",
    )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


AUTO_PROFILE_ALLOWED_PREFIXES: set[str] = {
    # “自动记忆”只允许写这些分区，避免模型一激动把硬风控改了。
    "workflow",
    "output",
    "memory",
}

PROFILE_EVENTS_MAX = 200

def ensure_memory_layout(paths: MemoryPaths) -> None:
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.daily_dir.mkdir(parents=True, exist_ok=True)

    if not paths.long_term_md.exists():
        paths.long_term_md.write_text(
            "# 长期记忆（MEMORY）\n\n"
            "- 说明：这里放“长期有效”的偏好/约束/共识（比如交易口径、风险纪律、输出偏好）。\n"
            "- 原则：如果与用户最新输入冲突，以最新输入为准。\n",
            encoding="utf-8",
        )

    if not paths.user_profile_json.exists():
        # 结构化偏好：给程序/LLM prompt 用，别靠正则瞎猜。
        payload = {
            "schema": "llm_trading.user_profile.v1",
            "updated_at": _now_iso(),
            "preferences": {},
            "events": [],
        }
        paths.user_profile_json.parent.mkdir(parents=True, exist_ok=True)
        paths.user_profile_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError, TypeError, AttributeError):  # noqa: BLE001
        return None


def load_user_profile(paths: MemoryPaths) -> dict[str, Any]:
    ensure_memory_layout(paths)
    obj = _read_json(paths.user_profile_json)
    if isinstance(obj, dict) and str(obj.get("schema") or "").startswith("llm_trading.user_profile."):
        return obj
    # schema 乱了就兜底成空（别让主流程炸）
    return {
        "schema": "llm_trading.user_profile.v1",
        "updated_at": _now_iso(),
        "preferences": {},
        "events": [],
    }


def save_user_profile(paths: MemoryPaths, profile: dict[str, Any]) -> None:
    ensure_memory_layout(paths)
    profile2 = sanitize_for_json(profile)
    if not isinstance(profile2, dict):
        raise ValueError("profile 必须是 object")
    profile2.setdefault("schema", "llm_trading.user_profile.v1")
    profile2["updated_at"] = _now_iso()

    tmp = Path(str(paths.user_profile_json) + ".tmp")
    tmp.write_text(json.dumps(profile2, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(paths.user_profile_json)


def _set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        raise ValueError("key 不能为空")
    cur: dict[str, Any] = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _get_nested(d: dict[str, Any], dotted_key: str) -> tuple[bool, Any]:
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        return False, None
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return False, None
        cur = cur.get(p)
    return True, cur


def _delete_nested(d: dict[str, Any], dotted_key: str) -> bool:
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        return False
    cur: Any = d
    stack: list[tuple[dict[str, Any], str]] = []
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            return False
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            return False
        stack.append((cur, p))
        cur = nxt
    if not isinstance(cur, dict):
        return False
    last = parts[-1]
    if last not in cur:
        return False
    del cur[last]
    # 清理空 dict，避免残留一堆空壳
    for parent, key in reversed(stack):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
            continue
        break
    return True


def _is_allowed_update_key(key: str, *, mode: str, allow_prefixes: set[str] | None) -> bool:
    k = (key or "").strip()
    if not k:
        return False
    if allow_prefixes is None:
        # manual：不设限（用户自己写的，锅自己背）
        return True
    for p in allow_prefixes:
        if k == p or k.startswith(p + "."):
            return True
    return False


def update_user_profile(
    paths: MemoryPaths,
    *,
    updates: dict[str, Any],
    source: dict[str, Any] | None = None,
    mode: str = "manual",
    allow_prefixes: set[str] | None = None,
) -> dict[str, Any]:
    """
    updates: dotted_key -> value
      e.g. {"memory.auto_write_preferences": True}
    """

    profile = load_user_profile(paths)
    prefs = profile.get("preferences")
    if not isinstance(prefs, dict):
        prefs = {}
        profile["preferences"] = prefs

    mode2 = (mode or "manual").strip().lower()
    allow = allow_prefixes
    if allow is None:
        if mode2 == "auto":
            allow = set(AUTO_PROFILE_ALLOWED_PREFIXES)
        elif mode2 == "sync":
            allow = {"trade_rules"}
        else:
            allow = None

    applied: dict[str, Any] = {}
    rejected: dict[str, Any] = {}
    diff: list[dict[str, Any]] = []

    for k0, v in (updates or {}).items():
        k = str(k0).strip()
        if not k:
            continue
        if not _is_allowed_update_key(k, mode=mode2, allow_prefixes=allow):
            rejected[k] = v
            continue
        existed, before = _get_nested(prefs, k)
        _set_nested(prefs, k, v)
        applied[k] = v
        diff.append(
            {
                "key": k,
                "before_exists": bool(existed),
                "before": sanitize_for_json(before) if existed else None,
                "after": sanitize_for_json(v),
            }
        )

    ts = _now_iso()
    event_id = _sha256_hex(json.dumps({"ts": ts, "mode": mode2, "diff": diff}, ensure_ascii=False, sort_keys=True))[:12]

    event = {
        "id": event_id,
        "ts": ts,
        "mode": mode2,
        "updates": sanitize_for_json(applied),
        "rejected": sanitize_for_json(rejected),
        "diff": sanitize_for_json(diff),
        "source": sanitize_for_json(source or {"type": "unknown"}),
    }
    evs = profile.get("events")
    if not isinstance(evs, list):
        evs = []
        profile["events"] = evs
    evs.append(event)
    if len(evs) > PROFILE_EVENTS_MAX:
        profile["events"] = evs[-PROFILE_EVENTS_MAX:]

    save_user_profile(paths, profile)
    append_ledger_event(paths, {"type": "profile_update", **event})
    return profile


def rollback_user_profile(
    paths: MemoryPaths,
    *,
    steps: int = 1,
    mode: str = "auto",
    apply: bool = False,
) -> dict[str, Any]:
    """
    回滚最近 N 次 profile_update（默认只回滚 auto 模式）。
    默认 dry-run；真写回必须显式 apply=True（跟 reconcile/clean 一样，防手滑）。
    """

    n = max(1, int(steps) if steps is not None else 1)
    mode2 = (mode or "auto").strip().lower()
    ensure_memory_layout(paths)

    try:
        lines = paths.ledger_jsonl.read_text(encoding="utf-8").splitlines()
    except (OSError, AttributeError):  # noqa: BLE001
        lines = []

    events: list[dict[str, Any]] = []
    for raw in reversed(lines):
        raw2 = raw.strip()
        if not raw2:
            continue
        try:
            obj = json.loads(raw2)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type") or "") != "profile_update":
            continue
        e_mode = str(obj.get("mode") or "manual").strip().lower()
        if mode2 not in {"any", "all"} and e_mode != mode2:
            continue
        events.append(obj)
        if len(events) >= n:
            break

    # 注意：events 是从最新往旧收集的，回滚要按这个顺序逐个撤销（先撤最新）。
    plan_ops: list[dict[str, Any]] = []
    for ev in events:
        diff = ev.get("diff")
        diff = diff if isinstance(diff, list) else []
        for it in diff:
            if not isinstance(it, dict):
                continue
            key = str(it.get("key") or "").strip()
            if not key:
                continue
            plan_ops.append(
                {
                    "key": key,
                    "before_exists": bool(it.get("before_exists")),
                    "before": it.get("before"),
                    "after": it.get("after"),
                    "event_id": str(ev.get("id") or ""),
                    "event_ts": str(ev.get("ts") or ""),
                    "event_mode": str(ev.get("mode") or ""),
                }
            )

    result = {
        "ok": True,
        "apply": bool(apply),
        "mode": mode2,
        "steps": n,
        "matched_events": [
            {"id": str(ev.get("id") or ""), "ts": str(ev.get("ts") or ""), "mode": str(ev.get("mode") or "")}
            for ev in events
        ],
        "ops": plan_ops,
    }

    if not apply:
        return result

    profile = load_user_profile(paths)
    prefs = profile.get("preferences")
    if not isinstance(prefs, dict):
        prefs = {}
        profile["preferences"] = prefs

    # 按撤销顺序执行：plan_ops 已经是“最新事件的 diff 在前”，逐条恢复 before。
    for op in plan_ops:
        key = str(op.get("key") or "").strip()
        if not key:
            continue
        if bool(op.get("before_exists")):
            _set_nested(prefs, key, op.get("before"))
        else:
            _delete_nested(prefs, key)

    ts = _now_iso()
    rollback_id = _sha256_hex(json.dumps({"ts": ts, "mode": mode2, "steps": n, "events": [e.get("id") for e in events]}, ensure_ascii=False, sort_keys=True))[:12]
    rb_event = {
        "id": rollback_id,
        "ts": ts,
        "mode": mode2,
        "steps": n,
        "rolled_back": sanitize_for_json(result.get("matched_events") or []),
        "ops": sanitize_for_json(plan_ops),
        "source": {"type": "cli", "cmd": "memory rollback"},
    }
    evs = profile.get("events")
    if not isinstance(evs, list):
        evs = []
        profile["events"] = evs
    evs.append(rb_event)
    if len(evs) > PROFILE_EVENTS_MAX:
        profile["events"] = evs[-PROFILE_EVENTS_MAX:]

    save_user_profile(paths, profile)
    append_ledger_event(paths, {"type": "profile_rollback", **rb_event})
    return result


def append_long_term_memory(paths: MemoryPaths, *, text: str, source: dict[str, Any] | None = None) -> None:
    ensure_memory_layout(paths)
    t = (text or "").strip()
    if not t:
        return
    line = f"- [{_now_iso()}] {t}\n"
    paths.long_term_md.write_text(paths.long_term_md.read_text(encoding="utf-8") + line, encoding="utf-8")
    append_ledger_event(
        paths,
        {"type": "long_term_append", "ts": _now_iso(), "text": t, "source": sanitize_for_json(source or {"type": "unknown"})},
    )


def _daily_path(paths: MemoryPaths, d: date) -> Path:
    return paths.daily_dir / f"{d.isoformat()}.md"


def append_daily_memory(
    paths: MemoryPaths,
    *,
    text: str,
    d: date | None = None,
    title: str | None = None,
    source: dict[str, Any] | None = None,
) -> Path:
    ensure_memory_layout(paths)
    dd = d or date.today()
    p = _daily_path(paths, dd)
    header = ""
    if not p.exists():
        header = f"# 每日记忆（{dd.isoformat()}）\n\n"

    t = (text or "").strip()
    if not t:
        return p

    block_title = (title or "").strip()
    if block_title:
        block = f"\n## {block_title}\n\n- [{_now_iso()}] {t}\n"
    else:
        block = f"\n- [{_now_iso()}] {t}\n"

    p.parent.mkdir(parents=True, exist_ok=True)
    prev = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(prev + (header if not prev else "") + block, encoding="utf-8")
    append_ledger_event(
        paths,
        {"type": "daily_append", "ts": _now_iso(), "date": dd.isoformat(), "title": block_title or None, "text": t, "source": sanitize_for_json(source or {"type": "unknown"})},
    )
    return p


_DAILY_FILENAME_RE = re.compile(r"^(?P<d>\d{4}-\d{2}-\d{2})\.md$", re.IGNORECASE)


def _parse_daily_filename_date(path: Path) -> date | None:
    """
    Parse date from daily filename: YYYY-MM-DD.md
    Return None if not match / invalid date.
    """

    name = str(getattr(path, "name", "") or "")
    m = _DAILY_FILENAME_RE.match(name.strip())
    if not m:
        return None
    try:
        return date.fromisoformat(m.group("d"))
    except ValueError:
        return None


def archive_daily_memory(
    paths: MemoryPaths,
    *,
    keep_days: int = 7,
    group: str = "month",
    apply: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """
    归档 daily/*.md，避免无限膨胀污染 prompt。

    - keep_days: 保留最近 N 天 daily 原文（其余归档）
    - group: month | week（归档 rollup 的分组方式）
    - apply: 默认 dry-run；True 才会写入归档并删除原 daily
    - today: 便于测试注入固定日期

    归档文件落在：{memory_dir}/archive/rollup/{group}/<key>.md
    归档索引：{memory_dir}/archive/index.json（用于幂等与审计）
    """

    ensure_memory_layout(paths)

    keep = max(0, int(keep_days) if keep_days is not None else 0)
    today2 = today or date.today()
    # Keep last N days including today: [today-(N-1), ..., today]
    cutoff = date.fromordinal(today2.toordinal() - (keep - 1)) if keep > 0 else today2

    group2 = (group or "month").strip().lower()
    if group2 not in {"month", "week"}:
        raise ValueError("group must be month|week")

    archive_dir = paths.base_dir / "archive"
    rollup_dir = archive_dir / "rollup" / group2
    rollup_dir.mkdir(parents=True, exist_ok=True)

    index_path = archive_dir / "index.json"
    try:
        idx_obj = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    except Exception:  # noqa: BLE001
        idx_obj = {}
    if not isinstance(idx_obj, dict):
        idx_obj = {}
    archived: dict[str, Any] = idx_obj.get("archived") if isinstance(idx_obj.get("archived"), dict) else {}

    candidates: list[dict[str, Any]] = []
    try:
        daily_files = sorted(paths.daily_dir.glob("*.md"))
    except Exception:  # noqa: BLE001
        daily_files = []

    for fp in daily_files:
        dd = _parse_daily_filename_date(fp)
        if dd is None:
            continue
        # keep last N days (dd >= cutoff) untouched
        if dd >= cutoff:
            continue

        rel = str(fp)
        try:
            content = fp.read_text(encoding="utf-8")
        except (OSError, AttributeError):  # noqa: BLE001
            content = ""
        h = _sha256_hex(content)
        if rel in archived and isinstance(archived.get(rel), dict):
            # 已归档：幂等跳过
            continue

        if group2 == "month":
            key = dd.strftime("%Y-%m")
        else:
            iso = dd.isocalendar()
            key = f"{iso.year}-W{int(iso.week):02d}"

        target = rollup_dir / f"{key}.md"
        candidates.append(
            {
                "date": dd.isoformat(),
                "path": str(fp),
                "sha256": h,
                "size": int(fp.stat().st_size) if fp.exists() else 0,
                "target": str(target),
                "group_key": key,
            }
        )

    # 统一排序，保证可复现
    candidates.sort(key=lambda x: str(x.get("date") or "") + "::" + str(x.get("path") or ""))

    plan = {
        "ok": True,
        "apply": bool(apply),
        "group": group2,
        "keep_days": keep,
        "today": today2.isoformat(),
        "cutoff": cutoff.isoformat(),
        "archive_dir": str(archive_dir),
        "rollup_dir": str(rollup_dir),
        "candidates": candidates,
        "archived_count": int(len(candidates)),
    }

    if not apply:
        return plan

    # apply: append into rollup + delete originals
    # group_key -> list[candidate]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for it in candidates:
        k = str(it.get("group_key") or "")
        grouped.setdefault(k, []).append(it)

    wrote_targets: list[str] = []
    for k, items in grouped.items():
        if not items:
            continue
        target = Path(str(items[0].get("target") or ""))
        # Header once
        if not target.exists():
            header = f"# 每日记忆归档（{group2}={k}）\n\n"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(header, encoding="utf-8")

        buf: list[str] = []
        for it in items:
            src = Path(str(it.get("path") or ""))
            try:
                content = src.read_text(encoding="utf-8").rstrip() + "\n"
            except (OSError, AttributeError):  # noqa: BLE001
                content = ""
            sha = str(it.get("sha256") or "")
            dstr = str(it.get("date") or "")
            buf.append(f"\n\n<!-- archived_from: {src.name} date={dstr} sha256={sha} -->\n")
            buf.append(content)

        # Atomic-ish append (best effort): write tmp then replace.
        prev = target.read_text(encoding="utf-8") if target.exists() else ""
        tmp = Path(str(target) + ".tmp")
        tmp.write_text(prev + "".join(buf), encoding="utf-8")
        tmp.replace(target)
        wrote_targets.append(str(target))

    # delete originals after writing succeeded (best effort)
    removed: list[str] = []
    for it in candidates:
        src = Path(str(it.get("path") or ""))
        try:
            src.unlink()
            removed.append(str(src))
        except OSError:
            continue

    # update archive index
    for it in candidates:
        p = str(it.get("path") or "")
        if not p:
            continue
        archived[p] = {
            "date": str(it.get("date") or ""),
            "sha256": str(it.get("sha256") or ""),
            "group": group2,
            "group_key": str(it.get("group_key") or ""),
            "target": str(it.get("target") or ""),
            "archived_at": _now_iso(),
        }

    idx_obj2: dict[str, Any] = dict(idx_obj)
    idx_obj2.setdefault("schema", "llm_trading.memory_archive_index.v1")
    idx_obj2["updated_at"] = _now_iso()
    idx_obj2["archived"] = archived
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_idx = Path(str(index_path) + ".tmp")
    tmp_idx.write_text(json.dumps(idx_obj2, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp_idx.replace(index_path)

    append_ledger_event(
        paths,
        {
            "type": "daily_archive",
            "ts": _now_iso(),
            "apply": True,
            "keep_days": keep,
            "group": group2,
            "cutoff": cutoff.isoformat(),
            "archived": sanitize_for_json(candidates),
            "targets": sanitize_for_json(wrote_targets),
            "removed": sanitize_for_json(removed),
        },
    )

    out = dict(plan)
    out["targets"] = wrote_targets
    out["removed"] = removed
    return out


def append_ledger_event(paths: MemoryPaths, event: dict[str, Any]) -> None:
    ensure_memory_layout(paths)
    try:
        line = json.dumps(sanitize_for_json(event), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):  # noqa: BLE001
        return
    paths.ledger_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledger_jsonl.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _tail_text(path: Path, *, max_chars: int) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, AttributeError):  # noqa: BLE001
        return ""
    if max_chars <= 0:
        return ""
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def build_prompt_memory_context(
    paths: MemoryPaths,
    *,
    include_long_term: bool = True,
    include_profile: bool = True,
    include_daily_days: int = 2,
    max_chars: int = 6000,
) -> str:
    """
    输出一段“可直接塞进 system prompt 的记忆上下文”。
    目标：短、可复核、对 LLM 友好；别把整个仓库喂进去。
    """

    ensure_memory_layout(paths)

    chunks: list[str] = []
    if include_profile:
        profile = load_user_profile(paths)
        prefs = profile.get("preferences") if isinstance(profile, dict) else None
        if isinstance(prefs, dict) and prefs:
            chunks.append("### 用户偏好（结构化）\n" + json.dumps(prefs, ensure_ascii=False, indent=2))

    if include_long_term:
        lt = _tail_text(paths.long_term_md, max_chars=max_chars)
        if lt.strip():
            chunks.append("### 长期记忆（MEMORY.md 摘要/尾部）\n" + lt.strip())

    if include_daily_days > 0:
        today = date.today()
        for i in range(max(0, include_daily_days)):
            dd = today.fromordinal(today.toordinal() - i)
            p = _daily_path(paths, dd)
            if not p.exists():
                continue
            dtxt = _tail_text(p, max_chars=max_chars)
            if dtxt.strip():
                chunks.append(f"### 每日记忆（{dd.isoformat()}，尾部）\n" + dtxt.strip())

    merged = "\n\n".join(chunks).strip()
    if not merged:
        return ""
    if max_chars > 0 and len(merged) > max_chars:
        return merged[-max_chars:]
    return merged


def keyword_search_memory(
    paths: MemoryPaths,
    *,
    query: str,
    max_results: int = 20,
    context_lines: int = 2,
) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []

    ensure_memory_layout(paths)

    files: list[Path] = []
    if paths.long_term_md.exists():
        files.append(paths.long_term_md)
    if paths.daily_dir.exists():
        files.extend(sorted(paths.daily_dir.glob("*.md")))
    # archive：不参与 prompt 注入，但必须可检索（避免归档=失忆）
    archive_dir = paths.base_dir / "archive"
    if archive_dir.exists():
        try:
            files.extend(sorted(archive_dir.rglob("*.md")))
        except Exception:  # noqa: BLE001
            pass
    if paths.user_profile_json.exists():
        files.append(paths.user_profile_json)

    results: list[dict[str, Any]] = []
    for fp in files:
        if len(results) >= max_results:
            break
        try:
            raw = fp.read_text(encoding="utf-8")
        except (OSError, AttributeError):  # noqa: BLE001
            continue
        lines = raw.splitlines()
        for idx, line in enumerate(lines):
            if q not in line:
                continue
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            snippet = "\n".join(lines[start:end]).strip()
            results.append(
                {
                    "path": str(fp),
                    "line": idx + 1,
                    "snippet": snippet,
                }
            )
            if len(results) >= max_results:
                break

    return results


def sync_trade_rules_from_user_holdings(
    paths: MemoryPaths,
    *,
    holdings_path: Path,
    source: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    把 user_holdings.json.trade_rules 同步到 user_profile.preferences.trade_rules。
    这不是“记忆”，是硬约束；让 LLM 每次都能看到。
    """
    hp = Path(holdings_path)
    obj = _read_json(hp)
    tr = obj.get("trade_rules") if isinstance(obj, dict) else None
    if not isinstance(tr, dict):
        return None
    return update_user_profile(
        paths,
        updates={"trade_rules": tr},
        source=source
        or {
            "type": "file_sync",
            "path": str(hp),
            "note": "sync trade_rules from user_holdings.json",
        },
    )


def append_run_daily_brief(
    paths: MemoryPaths,
    *,
    out_dir: Path,
    as_of: str | None,
    holdings_asof: str | None,
    orders_next_open_count: int | None,
    alerts_counts: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> Path:
    """
    给 `run` 自动留痕：让新对话能直接捞到“上次跑批到底输出了啥”。
    """

    parts: list[str] = []
    if as_of:
        parts.append(f"as_of={as_of}")
    if holdings_asof:
        parts.append(f"holdings_asof={holdings_asof}")
    if orders_next_open_count is not None:
        parts.append(f"orders_next_open={int(orders_next_open_count)}")
    if alerts_counts:
        try:
            parts.append("alerts=" + json.dumps(alerts_counts, ensure_ascii=False))
        except (TypeError, ValueError):  # noqa: BLE001
            pass

    meta = "；".join(parts) if parts else ""
    base = f"run 输出已落盘：{out_dir}。" + (f"（{meta}）" if meta else "")
    if warnings:
        w = [str(x).strip() for x in warnings if str(x).strip()]
        if w:
            base += "\n关键 warnings（截断）：\n" + "\n".join(f"- {x}" for x in w[:10])

    return append_daily_memory(
        paths,
        text=base,
        title="run 快照",
        source={"type": "auto", "cmd": "run", "out_dir": str(out_dir)},
    )
