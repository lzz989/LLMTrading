from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .llm_client import ChatMessage
from .pipeline import run_llm_text
from .prompting import extract_first_json


_RE_PCT = re.compile(r"(?P<sign>-)?(?P<num>\d{1,3}(?:\.\d+)?)\s*%")
_RE_LIMIT_DOWN_DAYS = re.compile(r"连续\s*(?P<a>\d+)\s*(?:[-~到至]\s*(?P<b>\d+))?\s*天.*?跌停")


@dataclass(frozen=True)
class ExtractResult:
    updates: dict[str, Any]
    note: str | None = None


def _has_any(text: str, words: list[str]) -> bool:
    t = str(text or "")
    tl = t.lower()
    return any((w in t) or (w.lower() in tl) for w in words)


def _parse_first_pct(text: str) -> float | None:
    """
    Parse first percentage from text.
    Return signed percent points, e.g. "-10%" -> -10.0, "10%" -> 10.0
    """

    s = str(text or "")
    m = _RE_PCT.search(s)
    if not m:
        return None
    try:
        num = float(m.group("num"))
    except Exception:  # noqa: BLE001
        return None
    sign = -1.0 if m.group("sign") else 1.0
    return sign * num


def _rule_extract_updates(user_text: str) -> ExtractResult:
    """
    Minimal heuristic extractor for durable preferences.
    Only writes workflow.* (auto-mode whitelist-safe).
    """

    t = str(user_text or "").strip()
    if not t:
        return ExtractResult(updates={})

    upd: dict[str, Any] = {}
    notes: list[str] = []

    # user level
    if _has_any(t, ["小白", "认知低", "不太懂", "新手"]):
        upd["workflow.user_level"] = "novice"
        # 默认开启教练模式（只写软偏好）
        upd["workflow.coach.enabled"] = True

    # risk appetite / concentration
    if _has_any(t, ["激进", "梭哈", "冲", "满仓", "全仓", "all in", "单吊", "单挑", "重仓"]):
        upd["workflow.risk_appetite"] = "aggressive"
        if _has_any(t, ["全仓", "满仓", "all in", "梭哈", "单吊"]):
            upd["workflow.concentration_preference"] = "all_in_single_idea"
        else:
            upd["workflow.concentration_preference"] = "concentrated"

    if _has_any(t, ["保守", "稳健", "低风险", "不想亏"]):
        upd["workflow.risk_appetite"] = "conservative"

    if _has_any(t, ["集中仓位", "不要开超市", "不想开超市", "别开超市"]):
        upd["workflow.positioning.concentration_style"] = "concentrated_not_all_in"

    # preferred number of positions
    if _has_any(t, ["2~3", "2-3", "两到三", "两三", "三仓", "3仓"]):
        upd["workflow.positioning.preferred_positions"] = 3

    # drawdown tolerance
    if _has_any(t, ["回撤", "亏", "最大亏损", "扛不住", "接受"]):
        pct = _parse_first_pct(t)
        # 只在显式提到百分比时才写，避免误判
        if pct is not None and pct != 0:
            # 统一用负数表示回撤
            upd["workflow.risk_tolerance.max_single_day_drawdown_pct"] = -abs(float(pct))

    # consecutive limit-down tolerance
    if _has_any(t, ["最多只能接受一天", "只能接受一天", "顶多只能接受一天"]):
        if _has_any(t, ["跌停", "连续"]):
            upd["workflow.risk_tolerance.max_consecutive_limit_down_days"] = 1

    m = _RE_LIMIT_DOWN_DAYS.search(t)
    if m and _has_any(t, ["睡不着", "受不了", "扛不住"]):
        # “连续2~3天跌停就睡不着” => 最大可接受天数倾向为 1
        upd.setdefault("workflow.risk_tolerance.max_consecutive_limit_down_days", 1)

    # style preference: pullback vs chase
    if _has_any(t, ["低吸", "回踩", "回调买", "回踩低吸"]):
        upd["workflow.entry_preference"] = "pullback"
        upd["workflow.avoid_chase"] = True
        upd["workflow.dip_buying.definition"] = "right_side_confirmed_pullback"

    if _has_any(t, ["追涨", "追高", "突破买", "打板", "强势突破"]):
        upd["workflow.avoid_chase"] = False

    if upd:
        notes.append("已按关键词抽取并更新 workflow 软偏好（硬风控 trade_rules 不会被自动改）。")

    return ExtractResult(updates=upd, note="\n".join(notes) if notes else None)


def _llm_extract_updates(
    cfg: AppConfig,
    *,
    provider: str,
    user_text: str,
    prefs_snapshot: dict[str, Any],
    mem_ctx: str,
) -> ExtractResult:
    """
    Use LLM to extract durable preferences into workflow.* / output.* / memory.* updates.
    This is *not* allowed to change trade_rules.
    """

    system = (
        "你是一个“用户画像抽取器”，用于把用户的自然语言偏好/约束变化提取成结构化 updates。\n"
        "硬规则：\n"
        "1) 只能输出 JSON，不要输出任何额外文字。\n"
        "2) 只能写 workflow.* / output.* / memory.* 三类键（dotted_key）；禁止写 trade_rules 或任何风控硬约束。\n"
        "3) 只提取“长期有效/可复用”的偏好；不要把一次性的行情观点当偏好。\n"
        "4) 如果用户表达内容与硬风控冲突（例如要求全仓），不要改硬风控；把冲突写到 note 里。\n"
    )
    if mem_ctx.strip():
        system += "\n\n# 现有记忆摘要（供参考；以用户最新输入为准）\n" + mem_ctx.strip()

    user = json.dumps(
        {
            "user_text": str(user_text or "").strip(),
            "current_preferences_snapshot": prefs_snapshot,
            "output_schema": {"updates": {"workflow.xxx": "value"}, "note": "一句话说明（可选）"},
        },
        ensure_ascii=False,
        indent=2,
    )

    raw = run_llm_text(
        cfg,
        messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        provider=provider,
        temperature=0.0,
        max_output_tokens=900,
    )
    obj = extract_first_json(raw)
    if not isinstance(obj, dict):
        return ExtractResult(updates={})

    updates = obj.get("updates") if isinstance(obj.get("updates"), dict) else {}
    # 保险：只保留允许前缀
    clean: dict[str, Any] = {}
    for k, v in updates.items():
        ks = str(k or "").strip()
        if not ks:
            continue
        if ks == "workflow" or ks.startswith("workflow.") or ks == "output" or ks.startswith("output.") or ks == "memory" or ks.startswith("memory."):
            clean[ks] = v
    note = str(obj.get("note") or "").strip() or None
    return ExtractResult(updates=clean, note=note)


def extract_profile_updates(
    cfg: AppConfig,
    *,
    provider: str,
    user_text: str,
    prefs_snapshot: dict[str, Any] | None,
    mem_ctx: str,
    use_llm: bool,
) -> ExtractResult:
    """
    Extract durable preference updates from user text.
    Strategy:
    - Always run a rule-based extractor (cheap, deterministic).
    - If LLM is available and the text looks like preference-change, let LLM refine/override.
    """

    snap = prefs_snapshot if isinstance(prefs_snapshot, dict) else {}

    rule = _rule_extract_updates(user_text)

    # Only invoke LLM when there's a signal it might matter; keep cost controlled.
    t = str(user_text or "")
    want_llm = use_llm and _has_any(
        t,
        [
            "我现在",
            "我变了",
            "以后",
            "改成",
            "不再",
            "偏好",
            "策略",
            "纪律",
            "止损",
            "止盈",
            "仓位",
            "激进",
            "保守",
            "全仓",
            "满仓",
            "单吊",
            "低吸",
            "追涨",
        ],
    )
    if not want_llm:
        return rule

    llm_res = _llm_extract_updates(cfg, provider=provider, user_text=user_text, prefs_snapshot=snap, mem_ctx=mem_ctx)
    # Merge: LLM wins for overlapping keys, but keep rule-only keys.
    merged = dict(rule.updates)
    merged.update(llm_res.updates or {})
    note = llm_res.note or rule.note
    return ExtractResult(updates=merged, note=note)

