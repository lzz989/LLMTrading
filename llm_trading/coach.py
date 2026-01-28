from __future__ import annotations

import json
from typing import Any


def _safe_get(d: dict[str, Any], dotted_key: str) -> Any | None:
    cur: Any = d
    for part in (dotted_key or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _has_any(text: str, words: list[str]) -> bool:
    t = str(text or "")
    tl = t.lower()
    return any((w in t) or (w.lower() in tl) for w in words)


def build_coach_questions(
    *,
    prefs: dict[str, Any],
    user_text: str,
    plan: dict[str, Any] | None = None,
) -> list[str]:
    """
    “教练模式”：给小白强制加一层“执行前复核问题”。
    目标不是嘴炮，而是把最容易亏钱的认知坑提前暴露出来。
    """

    wf = prefs.get("workflow") if isinstance(prefs.get("workflow"), dict) else {}
    wf = wf if isinstance(wf, dict) else {}
    coach = wf.get("coach") if isinstance(wf.get("coach"), dict) else {}
    coach = coach if isinstance(coach, dict) else {}
    if not bool(coach.get("enabled")):
        return []

    max_q = int(coach.get("max_questions") or 3)
    max_q = max(0, min(6, max_q))
    if max_q <= 0:
        return []

    t = str(user_text or "")
    qs: list[str] = []

    # 1) 高频致命坑：消息面扛单
    if _has_any(t, ["小道消息", "内幕", "听说", "群里", "老师说", "一定会", "肯定涨", "必涨", "稳赢"]):
        qs.append("这条消息的“可核验来源”是什么（公告/监管披露/财报/权威媒体）？如果拿不出来源，是否同意按噪音处理？")

    # 2) 风险动作：满仓/单吊/加杠杆
    if _has_any(t, ["满仓", "全仓", "梭哈", "all in", "单吊", "加杠杆", "融资"]):
        qs.append("你这次的失效条件/止损位写清楚了吗？如果明天跳空低开 5% 甚至跌停，你的执行动作是什么？")

    # 3) 纪律：不止损/死扛
    if _has_any(t, ["不止损", "不想止损", "扛一扛", "死扛", "拿到翻倍", "跌了加仓摊平", "摊平"]):
        qs.append("你愿意为“扛单”支付多大的代价？（最多亏多少/最多扛几天）如果触发了，你是否承诺按规则退出？")

    # 4) 你当前画像的硬冲突提示（集中 vs 连续跌停恐惧）
    max_ld = _safe_get(wf, "risk_tolerance.max_consecutive_limit_down_days")
    try:
        max_ld_i = int(max_ld) if max_ld is not None else None
    except Exception:  # noqa: BLE001
        max_ld_i = None
    if max_ld_i is not None and max_ld_i <= 1:
        if _has_any(t, ["小票", "妖股", "连板", "打板", "游资", "题材", "情绪票"]):
            qs.append("你画像里写的是“连续跌停最多扛 1 天”。那你这票属于高跌停链风险吗？如果是，你为什么还要做？")

    # 5) 如果 plan 里包含 run 且可能给出买卖单，提醒“执行前复核”
    if isinstance(plan, dict):
        acts = plan.get("actions") if isinstance(plan.get("actions"), list) else []
        has_run = any(isinstance(a, dict) and str(a.get("type") or "") == "run" for a in acts)
        has_skill_strategy = any(
            isinstance(a, dict)
            and str(a.get("type") or "") == "skill"
            and isinstance(a.get("args"), dict)
            and str((a.get("args") or {}).get("name") or "") == "strategy"
            for a in acts
        )
        if has_run and not _has_any(t, ["只看", "不交易", "纯复盘"]):
            # 给一个“通用复核问题”，别太啰嗦
            qs.append("你明天是“执行计划”还是“只观察”？如果执行，是否同意：只按收盘触发、次日开盘成交，不临盘追涨？")
        if has_skill_strategy:
            # strategy 报告里会给动作/失效条件；提醒看文件
            qs.append("请你先看一眼 strategy_action.md 里的“失效条件”，确认你能做到再执行。")

    # 去重保序 + 截断
    seen: set[str] = set()
    out: list[str] = []
    for q in qs:
        s = str(q or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_q:
            break
    return out


def build_coach_md(
    *,
    prefs: dict[str, Any],
    user_text: str,
    plan: dict[str, Any] | None = None,
) -> str:
    qs = build_coach_questions(prefs=prefs, user_text=user_text, plan=plan)
    if not qs:
        return ""

    wf = prefs.get("workflow") if isinstance(prefs.get("workflow"), dict) else {}
    wf = wf if isinstance(wf, dict) else {}
    snap = {
        "user_level": wf.get("user_level"),
        "risk_appetite": wf.get("risk_appetite"),
        "concentration_preference": wf.get("concentration_preference"),
        "positioning": wf.get("positioning"),
        "risk_tolerance": wf.get("risk_tolerance"),
    }

    lines: list[str] = []
    lines.append("# 执行前复核（教练模式）")
    lines.append("")
    lines.append("## 画像快照（只读）")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(snap, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 我需要你确认的几个问题")
    lines.append("")
    for i, q in enumerate(qs, start=1):
        lines.append(f"{i}. {q}")
    lines.append("")
    return "\n".join(lines)

