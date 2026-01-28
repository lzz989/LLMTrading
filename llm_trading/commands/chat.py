from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_config
from ..llm_client import ChatMessage
from ..memory_store import (
    append_daily_memory,
    build_prompt_memory_context,
    load_user_profile,
    resolve_memory_paths,
    update_user_profile,
)
from ..pipeline import run_llm_text, write_json
from ..prompting import extract_first_json
from ..profile_extractor import extract_profile_updates


_RE_CODE_PREF = re.compile(r"\b(?P<prefix>sh|sz|bj)\s*(?P<code>\d{6})\b", re.IGNORECASE)
_RE_CODE_6 = re.compile(r"\b(?P<code>\d{6})\b")


@dataclass(frozen=True)
class ParsedSymbol:
    asset: str  # stock|etf|index
    symbol: str  # digits or prefixed (for index)


def _infer_asset_from_code(prefix: str | None, code: str) -> str:
    p = (prefix or "").strip().lower()
    c = (code or "").strip()
    if (p in {"sh"} and c.startswith("000")) or (p in {"sz"} and c.startswith("399")):
        return "index"

    # ETF：5xxxx(沪) + 159/16x/513 等（这里不追求完美，够用就行）
    if c.startswith("5") or c.startswith("1"):
        return "etf"
    return "stock"


def _normalize_symbol(asset: str, prefix: str | None, code: str) -> str:
    a = (asset or "").strip().lower()
    p = (prefix or "").strip().lower()
    c = (code or "").strip()
    if a == "index":
        if p in {"sh", "sz"}:
            return f"{p}{c}"
        # 兜底：常见指数规则
        if c.startswith("399"):
            return f"sz{c}"
        return f"sh{c}"
    # stock/etf：交给内部 resolve_symbol 去补前缀
    return c


def parse_symbols_from_text(text: str) -> list[ParsedSymbol]:
    s = str(text or "")
    found: list[tuple[str | None, str]] = []

    for m in _RE_CODE_PREF.finditer(s):
        found.append((m.group("prefix"), m.group("code")))

    # 再抓裸 6 位，避免重复
    pref_codes = {c for _, c in found}
    for m in _RE_CODE_6.finditer(s):
        code = m.group("code")
        if code in pref_codes:
            continue
        found.append((None, code))

    out: list[ParsedSymbol] = []
    seen: set[tuple[str, str]] = set()
    for pref, code in found:
        asset = _infer_asset_from_code(pref, code)
        sym = _normalize_symbol(asset, pref, code)
        key = (asset, sym)
        if key in seen:
            continue
        seen.add(key)
        out.append(ParsedSymbol(asset=asset, symbol=sym))
    return out


def _looks_like_run_request(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    # “分析持仓/复盘/调仓”这类请求，本质上需要先 run 才能靠客观数据说话。
    k = [
        "跑",
        "跑批",
        "run",
        "计划",
        "明天开盘",
        "次日开盘",
        "执行",
        "扫描",
        "选",
        "筛",
        "今天怎么做",
        "持仓",
        "复盘",
        "仓位",
        "调仓",
        "换仓",
        "止盈",
        "止损",
    ]
    tl = t.lower()
    return any((x in t) or (x.lower() in tl) for x in k)


def _detect_skill_requests(text: str, *, syms: list[ParsedSymbol] | None = None) -> list[dict[str, Any]]:
    """
    rule planner 的最小实现：
    - LLM 不可用时，靠关键词触发 skill（并允许用户显式强制）。
    - 注意：这里只做“触发”，不负责跑细节参数（复杂的让 LLM planner 来）。
    """
    t = str(text or "")
    tl = t.lower()

    def _has_any(xs: list[str]) -> bool:
        return any((x in t) or (x.lower() in tl) for x in xs)

    skills: list[str] = []

    # 显式强制（推荐写法：#research / #strategy / #backtest）
    if _has_any(["#research", "强制 research", "强制research", "skill research"]):
        skills.append("research")
    if _has_any(["#strategy", "强制 strategy", "强制strategy", "skill strategy"]):
        skills.append("strategy")
    want_backtest = _has_any(["#backtest", "强制 backtest", "强制backtest", "skill backtest", "回测"])
    if want_backtest:
        # backtest 必须有 symbols；没有就别硬跑（否则只会报错）。
        if syms:
            skills.append("backtest")

    # 半显式：用户说“舆情/新闻/消息面”就默认 research
    if _has_any(["舆情", "新闻", "消息面", "情绪", "盘面", "题材", "热度", "利空", "公告"]) and ("research" not in skills):
        skills.append("research")
    # 用户说“执行清单/终极动作/怎么做”就默认 strategy
    if _has_any(["执行清单", "终极动作", "怎么做", "明天怎么操作", "调仓", "换仓", "止盈", "止损", "上车", "下车", "策略动作"]) and ("strategy" not in skills):
        skills.append("strategy")

    # 去重保序
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for s in skills:
        ss = str(s).strip().lower()
        if not ss or ss in seen:
            continue
        seen.add(ss)
        args: dict[str, Any] = {"name": ss}
        if ss == "backtest" and syms:
            # 只回测同一 asset（混资产没意义，且脚本也不支持混跑）
            asset = syms[0].asset
            symbols = [x.symbol for x in syms if x.asset == asset]
            if symbols:
                args["asset"] = asset
                args["symbols"] = ",".join(symbols)
        out.append({"type": "skill", "args": args})
    return out


def _default_out_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{prefix}_{ts}"


def _safe_get_pref(prefs: dict[str, Any], dotted_key: str) -> Any | None:
    cur: Any = prefs
    for part in (dotted_key or "").split("."):
        if not part:
            continue
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _get_nested(d: dict[str, Any], dotted_key: str) -> tuple[bool, Any]:
    cur: Any = d
    parts = [p for p in (dotted_key or "").split(".") if p]
    if not parts:
        return False, None
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return False, None
        cur = cur.get(p)
    return True, cur


def _filter_updates_by_diff(prefs: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (updates or {}).items():
        ks = str(k or "").strip()
        if not ks:
            continue
        existed, before = _get_nested(prefs, ks)
        # 避免写入“无变化”的更新，减少 profile events 垃圾增长
        if existed and before == v:
            continue
        out[ks] = v
    return out


def _apply_workflow_defaults_to_run_args(prefs: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """
    workflow.run.* 作为 chat/run 的默认值（用户没显式说，就按偏好走）。
    """

    out = dict(args or {})
    wf = prefs.get("workflow") if isinstance(prefs.get("workflow"), dict) else {}
    run0 = wf.get("run") if isinstance(wf.get("run"), dict) else {}
    if not isinstance(run0, dict) or not run0:
        return out

    # 只允许覆盖 chat 常用的少数字段，别让 profile 变成“隐形配置炸弹”
    allow = {
        "scan_mode",
        "scan_strategy",
        "scan_top_k",
        "scan_limit",
        "scan_freq",
        "scan_left",
        "scan_left_strategy",
        "scan_left_top_k",
        "scan_shadow_legacy",
        "deep_holdings",
        "rebalance_mode",
    }
    for k in allow:
        if k in out:
            continue
        v = run0.get(k)
        if v is not None:
            out[k] = v
    return out


def _rule_plan(text: str, *, prefs: dict[str, Any]) -> dict[str, Any]:
    syms = parse_symbols_from_text(text)
    actions: list[dict[str, Any]] = []

    # skill 触发（LLM 不可用时的最小兜底）
    skill_actions = _detect_skill_requests(text, syms=syms)

    need_run = _looks_like_run_request(text) and (not syms or any(x in str(text or "") for x in ["跑", "run", "计划", "扫描", "选"]))
    if need_run:
        run_args = _apply_workflow_defaults_to_run_args(prefs, {})
        run_args.setdefault("out_dir", str(_default_out_dir("chat_run")))
        actions.append({"type": "run", "args": run_args})

    # 默认：有代码就给你分析（最多 3 个，别tm一口吃成胖子把时间全浪费在“看图爽”上）
    for ps in syms[:3]:
        out_dir = str(_default_out_dir(f"chat_analyze_{ps.asset}_{ps.symbol}"))
        actions.append({"type": "analyze", "args": {"asset": ps.asset, "symbol": ps.symbol, "method": "all", "out_dir": out_dir}})

    # 把 skill 放在最后：优先产出 run/analyze，再让 skill 把结果“收敛成报告”
    actions.extend(skill_actions)

    if not actions:
        # 兜底：直接跑 run
        run_args = _apply_workflow_defaults_to_run_args(prefs, {})
        run_args.setdefault("out_dir", str(_default_out_dir("chat_run")))
        actions.append({"type": "run", "args": run_args})

    return {
        "schema": "llm_trading.chat_plan.v1",
        "planner": "rule",
        "user_text": str(text or "").strip(),
        "actions": actions,
        "memory_updates": {},
    }


def _llm_plan(
    cfg,
    *,
    provider: str,
    user_text: str,
    mem_ctx: str,
) -> dict[str, Any]:
    # 允许的 action：越少越稳，别学那些“啥都能干”的 agent，最后只会胡跑。
    allowed_actions = [
        {
            "type": "run",
            "args_help": {
                "out_dir": "输出目录（可选；不传则默认 outputs/run_YYYYMMDD）",
                "scan_mode": "auto|strategy|legacy（可选）",
                "scan_strategy": "scan-strategy key（可选，默认 bbb_weekly）",
                "scan_top_k": "扫描输出 TopK（可选）",
                "scan_limit": "扫描数量（可选）",
                "scan_freq": "weekly|daily（可选）",
                "scan_left": "true/false（可选）",
                "scan_left_strategy": "左侧策略 key（可选）",
                "scan_left_top_k": "左侧 TopK（可选）",
                "scan_shadow_legacy": "true/false（可选）",
                "deep_holdings": "true/false（可选）",
                "rebalance_mode": "add|rotate（可选）",
            },
        },
        {
            "type": "analyze",
            "args_help": {
                "asset": "stock|etf|index",
                "symbol": "如 000725 / 510300 / sh000300",
                "method": "all（默认）",
                "out_dir": "输出目录（可选）",
            },
        },
        {
            "type": "memory_search",
            "args_help": {
                "query": "要搜的内容",
                "mode": "keyword|vector|hybrid（默认 keyword）",
                "max_results": "最多返回条数（默认 20）",
            },
        },
        {
            "type": "skill",
            "args_help": {
                "name": "strategy|research|backtest",
                "run_dir": "基于哪个 outputs/run_* 目录（可选；不传则默认用本次 run 的 out_dir 或自动选最新）",
                "out": "strategy/backtest 的输出路径（可选）",
                "out_dir": "research 的输出目录（可选；默认 outputs/agents）",
                "queries": "research：逗号分隔查询词/代码（可选；为空则从 run_dir/持仓自动提取）",
                "pages": "research：页数（默认 2）",
                "page_size": "research：每页条数（默认 10）",
                "symbols": "backtest：逗号分隔 symbols（必填；例 sh518880,sh159937）",
                "asset": "backtest：etf|stock|index|crypto（默认 etf）",
                "start": "backtest：YYYY-MM-DD（可选）",
                "end": "backtest：YYYY-MM-DD（可选）",
            },
        },
    ]

    system = (
        "你是一个交易框架的“自然语言路由器/调度器”。目标：把用户自然语言请求翻译成可执行的 CLI 动作计划。\n"
        "硬约束：\n"
        "1) 只能输出 JSON，不要输出任何额外文字。\n"
        "2) 只能从允许的 action 列表里选；不允许执行 shell、不允许下真实单。\n"
        "   - 如需新闻/舆情，只能通过 skill(name=research) 触发（由框架去抓取并落盘）。\n"
        "3) 默认优先 run（用策略武器库产出 orders_next_open.json），除非用户明确要分析某个标的。\n"
        "4) 可选 skill：用于把 run 结果落成可复核报告（strategy/research/backtest）；仅当用户需要时才加。\n"
        "5) 你可以提出 memory_updates，但只允许 workflow.* / output.* / memory.* 三类偏好字段。\n"
    )
    if mem_ctx.strip():
        system += "\n\n# 用户偏好/约束（持久记忆；如与本次输入冲突，以本次输入为准）\n" + mem_ctx.strip()

    user = json.dumps(
        {
            "user_text": str(user_text or "").strip(),
            "allowed_actions": allowed_actions,
            "output_schema": {
                "schema": "llm_trading.chat_plan.v1",
                "planner": "llm",
                "actions": [{"type": "run|analyze|memory_search|skill", "args": {}}],
                "memory_updates": {"workflow.xxx": "value"},
                "notes": "给用户的简短说明（可选）",
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    raw = run_llm_text(
        cfg,
        messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        provider=provider,
        temperature=0.0,
        max_output_tokens=1200,
    )
    plan = extract_first_json(raw)
    if not isinstance(plan, dict):
        raise ValueError("LLM plan 不是 JSON object")
    plan.setdefault("schema", "llm_trading.chat_plan.v1")
    plan.setdefault("planner", "llm")
    plan.setdefault("user_text", str(user_text or "").strip())
    plan.setdefault("actions", [])
    plan.setdefault("memory_updates", {})
    return plan


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    actions0 = plan.get("actions")
    actions0 = actions0 if isinstance(actions0, list) else []
    out_actions: list[dict[str, Any]] = []
    for it in actions0:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type") or "").strip().lower()
        args = it.get("args") if isinstance(it.get("args"), dict) else {}
        if t not in {"run", "analyze", "memory_search", "skill"}:
            continue
        if t == "skill":
            name = str(args.get("name") or "").strip().lower()
            if name not in {"strategy", "research", "backtest"}:
                continue
            # 只保留最小必要字段，避免 plan 变成“隐形配置炸弹”
            allow = {
                "name",
                "run_dir",
                "out",
                "out_dir",
                "queries",
                "pages",
                "page_size",
                "symbols",
                "asset",
                "source",
                "cache_ttl_hours",
                "fee_bps",
                "slippage_bps",
                "start",
                "end",
            }
            args2 = {k: v for k, v in args.items() if str(k) in allow}
            args2["name"] = name
            out_actions.append({"type": t, "args": args2})
            continue
        out_actions.append({"type": t, "args": args})

    mem_updates = plan.get("memory_updates") if isinstance(plan.get("memory_updates"), dict) else {}
    # 只保留三大前缀（真正写入时还会再过一遍 update_user_profile 的白名单）
    clean_updates: dict[str, Any] = {}
    for k, v in mem_updates.items():
        ks = str(k or "").strip()
        if not ks:
            continue
        if ks == "workflow" or ks.startswith("workflow.") or ks == "output" or ks.startswith("output.") or ks == "memory" or ks.startswith("memory."):
            clean_updates[ks] = v

    return {
        "schema": "llm_trading.chat_plan.v1",
        "planner": str(plan.get("planner") or "unknown"),
        "user_text": str(plan.get("user_text") or "").strip(),
        "actions": out_actions,
        "memory_updates": clean_updates,
        "notes": str(plan.get("notes") or "").strip() or None,
    }


def _run_subcommand(argv: list[str]) -> dict[str, Any]:
    """
    在同一进程内复用 llm_trading.cli.main，避免 shell subprocess。
    返回：{ok, argv, code, stdout, stderr}
    """

    import io
    import contextlib

    from ..cli import main as cli_main

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    code = 1
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            code = int(cli_main(argv))
        except SystemExit as exc:
            # argparse 可能 raise SystemExit
            try:
                code = int(getattr(exc, "code", 1))
            except Exception:  # noqa: BLE001
                code = 1
        except Exception as exc:  # noqa: BLE001
            buf_err.write(f"{exc.__class__.__name__}: {exc}\n")
            code = 1

    return {
        "ok": code == 0,
        "argv": list(argv),
        "code": int(code),
        "stdout": buf_out.getvalue(),
        "stderr": buf_err.getvalue(),
    }


def _summarize_orders_next_open(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    orders = obj.get("orders") if isinstance(obj, dict) else None
    orders = orders if isinstance(orders, list) else []
    lines: list[str] = []
    for o in orders[:30]:
        if not isinstance(o, dict):
            continue
        side = str(o.get("side") or "").strip().lower()
        asset = str(o.get("asset") or "").strip().lower() or "etf"
        sym = str(o.get("symbol") or "").strip()
        name = str(o.get("name") or "").strip()
        nm = f"{sym}（{name}）" if name else f"{sym}（名称未知）"
        sh = o.get("shares")
        px = o.get("price_ref")
        notional = o.get("est_notional_yuan")
        reason = str(o.get("reason") or "").strip()
        bits = [side, asset, nm]
        if sh is not None:
            bits.append(f"shares={sh}")
        if px is not None:
            bits.append(f"px_ref={px}")
        if notional is not None:
            bits.append(f"notional~{notional:.0f}" if isinstance(notional, (int, float)) else f"notional~{notional}")
        if reason:
            bits.append(f"reason={reason}")
        lines.append(" - " + " ".join(bits))
    return lines


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config()
    provider = str(getattr(args, "provider", "openai") or "openai").strip().lower()
    planner = str(getattr(args, "planner", "auto") or "auto").strip().lower()
    dry_run = bool(getattr(args, "dry_run", False))

    # 读输入：--text > stdin(piped) > 交互
    text = str(getattr(args, "text", "") or "").strip()
    if not text:
        try:
            if not sys.stdin.isatty():
                text = sys.stdin.read().strip()
        except Exception:  # noqa: BLE001
            text = ""

    if not text:
        # 交互式（最小实现）：一问一答；空行退出
        print("进入 chat 模式（空行退出）。")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            ns = argparse.Namespace(**{**vars(args), "text": line})
            cmd_chat(ns)
        return 0

    mp = resolve_memory_paths(project_root=cfg.project_root)
    profile = load_user_profile(mp)
    prefs = profile.get("preferences") if isinstance(profile, dict) else {}
    prefs = prefs if isinstance(prefs, dict) else {}

    # 先判断 LLM 可用性（planner=rule 时强制不用）
    llm_available = False
    if provider == "openai":
        llm_available = bool(cfg.openai())
    elif provider == "gemini":
        llm_available = bool(cfg.gemini())

    use_llm = False  # for plan
    if planner == "llm":
        use_llm = bool(llm_available)
    elif planner == "rule":
        use_llm = False
    else:
        # auto
        use_llm = bool(llm_available)

    # memory：在做任何路由/计划前，先尝试“自动抽取画像变更”（软偏好 only）。
    auto_on = bool(_safe_get_pref(prefs, "memory.auto_write_preferences"))
    mem_ctx = ""
    try:
        mem_ctx = build_prompt_memory_context(mp, include_long_term=True, include_profile=True, include_daily_days=2, max_chars=6000)
    except Exception:  # noqa: BLE001
        mem_ctx = ""

    if auto_on:
        try:
            res = extract_profile_updates(
                cfg,
                provider=provider,
                user_text=text,
                prefs_snapshot=prefs,
                mem_ctx=mem_ctx,
                use_llm=bool(use_llm),
            )
            updates0 = res.updates if isinstance(res.updates, dict) else {}
            updates = _filter_updates_by_diff(prefs, updates0)
            if updates:
                profile = update_user_profile(
                    mp,
                    updates={str(k): v for k, v in updates.items()},
                    source={"type": "chat", "cmd": "chat", "stage": "profile_extract", "user_text": text[:200]},
                    mode="auto",
                )
                prefs = profile.get("preferences") if isinstance(profile, dict) else prefs
                prefs = prefs if isinstance(prefs, dict) else {}
                append_daily_memory(
                    mp,
                    title="chat 用户画像更新",
                    text=(res.note or "chat 自动抽取并更新 workflow 软偏好") + "：" + json.dumps(updates, ensure_ascii=False),
                    source={"type": "chat", "cmd": "chat", "stage": "profile_extract"},
                )
                # 画像更新后，重建 mem_ctx 让后续 LLM plan 看到最新偏好
                try:
                    mem_ctx = build_prompt_memory_context(mp, include_long_term=True, include_profile=True, include_daily_days=2, max_chars=6000)
                except Exception:  # noqa: BLE001
                    mem_ctx = mem_ctx
        except Exception:  # noqa: BLE001
            pass

    plan_raw: dict[str, Any]
    if use_llm:
        try:
            plan_raw = _llm_plan(cfg, provider=provider, user_text=text, mem_ctx=mem_ctx)
        except Exception:
            plan_raw = _rule_plan(text, prefs=prefs)
    else:
        plan_raw = _rule_plan(text, prefs=prefs)

    plan = _validate_plan(plan_raw)
    if not (plan.get("actions") or []):
        # LLM 偶尔会给“空计划”；兜底回到 rule planner，至少跑得起来。
        plan = _validate_plan(_rule_plan(text, prefs=prefs))

    # 输出目录：用于落盘 plan + 执行摘要（可复盘）
    chat_out_dir = Path(str(getattr(args, "out_dir", "") or "").strip() or str(_default_out_dir("chat")))
    chat_out_dir.mkdir(parents=True, exist_ok=True)
    write_json(chat_out_dir / "chat_plan.json", plan)

    # 自动写入偏好（白名单字段），并写 daily 记忆留痕
    mem_updates = plan.get("memory_updates") if isinstance(plan.get("memory_updates"), dict) else {}
    if auto_on and mem_updates:
        try:
            mem_updates2 = _filter_updates_by_diff(prefs, mem_updates)
            if not mem_updates2:
                raise ValueError("no-op memory_updates")
            update_user_profile(
                mp,
                updates={str(k): v for k, v in mem_updates2.items()},
                source={"type": "chat", "cmd": "chat", "user_text": text[:200]},
                mode="auto",
            )
            append_daily_memory(
                mp,
                title="chat 偏好更新",
                text="chat 自动写入偏好（白名单字段）："
                + json.dumps({str(k): v for k, v in mem_updates2.items()}, ensure_ascii=False),
                source={"type": "chat", "cmd": "chat"},
            )
        except Exception:  # noqa: BLE001
            pass

    # coach：输出执行前复核问题（不阻塞执行；但会把坑先摊开）
    try:
        from ..coach import build_coach_md

        coach_md = build_coach_md(prefs=prefs, user_text=text, plan=plan)
        if coach_md.strip():
            (chat_out_dir / "coach.md").write_text(coach_md, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"plan 已落盘：{chat_out_dir / 'chat_plan.json'}")
        try:
            coach_p = chat_out_dir / "coach.md"
            if coach_p.exists():
                print(f"coach 已落盘：{coach_p}")
        except Exception:  # noqa: BLE001
            pass
        return 0

    # 执行动作
    exec_results: list[dict[str, Any]] = []
    last_run_out_dir: str | None = None
    for a in plan.get("actions") or []:
        t = str(a.get("type") or "")
        a_args = a.get("args") if isinstance(a.get("args"), dict) else {}

        if t == "run":
            argv = ["run"]
            # 注入 workflow.run 默认值（仅当 LLM 未填）
            a_args = _apply_workflow_defaults_to_run_args(prefs, a_args)
            if "out_dir" not in a_args:
                a_args["out_dir"] = str(_default_out_dir("chat_run"))
            for k, v in a_args.items():
                if v is None:
                    continue
                kk = str(k).replace("_", "-")
                if isinstance(v, bool):
                    argv.append(f"--{'' if v else 'no-'}{kk}")
                else:
                    argv.extend([f"--{kk}", str(v)])
            res = _run_subcommand(argv)
            exec_results.append({"type": "run", "args": a_args, "result": res})
            try:
                last_run_out_dir = str(a_args.get("out_dir") or "").strip() or last_run_out_dir
            except Exception:  # noqa: BLE001
                pass
            continue

        if t == "analyze":
            argv = ["analyze"]
            asset = str(a_args.get("asset") or "").strip().lower() or "stock"
            symbol = str(a_args.get("symbol") or "").strip()
            method = str(a_args.get("method") or "all").strip()
            out_dir = str(a_args.get("out_dir") or str(_default_out_dir(f"chat_analyze_{asset}_{symbol}")))
            argv.extend(["--asset", asset, "--symbol", symbol, "--method", method, "--out-dir", out_dir])
            res = _run_subcommand(argv)
            exec_results.append({"type": "analyze", "args": {"asset": asset, "symbol": symbol, "method": method, "out_dir": out_dir}, "result": res})
            continue

        if t == "memory_search":
            argv = ["memory", "search", str(a_args.get("query") or "").strip()]
            mode = str(a_args.get("mode") or "keyword").strip().lower()
            max_results = a_args.get("max_results")
            argv.extend(["--mode", mode])
            if max_results is not None:
                argv.extend(["--max-results", str(int(max_results))])
            res = _run_subcommand(argv)
            exec_results.append({"type": "memory_search", "args": a_args, "result": res})
            continue

        if t == "skill":
            name = str(a_args.get("name") or "").strip().lower()
            if not name:
                continue
            argv = ["skill", name]

            # 默认用“本次 run 的 out_dir”作为 run_dir（避免 LLM 不知道路径时跑错目录）
            if (not a_args.get("run_dir")) and last_run_out_dir:
                a_args["run_dir"] = str(last_run_out_dir)

            key_map = {"run_dir": "--run-dir", "out_dir": "--out-dir", "page_size": "--page-size"}
            for k, v in a_args.items():
                if k == "name" or v is None:
                    continue
                if isinstance(v, bool):
                    # skill 目前没暴露 bool flags；保险起见忽略
                    continue
                opt = key_map.get(str(k), f"--{str(k).replace('_','-')}")
                argv.extend([opt, str(v)])
            res = _run_subcommand(argv)
            exec_results.append({"type": "skill", "args": a_args, "result": res})
            continue

    # 落盘执行摘要
    write_json(
        chat_out_dir / "chat_exec.json",
        {
            "schema": "llm_trading.chat_exec.v1",
            "generated_at": datetime.now().isoformat(),
            "plan": plan,
            "results": exec_results,
        },
    )

    # 输出给用户看的“可执行摘要”
    print(f"chat_out_dir: {chat_out_dir.resolve()}")
    if plan.get("notes"):
        print(f"notes: {plan.get('notes')}")
    try:
        coach_p = chat_out_dir / "coach.md"
        if coach_p.exists():
            print(f"coach: {coach_p.resolve()}")
    except Exception:  # noqa: BLE001
        pass

    for it in exec_results:
        t = str(it.get("type") or "")
        res = it.get("result") if isinstance(it.get("result"), dict) else {}
        ok = bool(res.get("ok"))
        if t == "run":
            out_dir = str((it.get("args") or {}).get("out_dir") or "").strip()
            print(f"run: ok={ok} out_dir={out_dir}")
            p = Path(out_dir) / "orders_next_open.json" if out_dir else None
            if p and p.exists():
                lines = _summarize_orders_next_open(p)
                if lines:
                    print("orders_next_open(最多30条)：")
                    print("\n".join(lines))
                else:
                    print("orders_next_open: 0 或解析失败（去 out_dir 看文件）")
            continue
        if t == "analyze":
            out_dir = str((it.get("args") or {}).get("out_dir") or "").strip()
            print(f"analyze: ok={ok} out_dir={out_dir}")
            continue
        if t == "memory_search":
            print(f"memory_search: ok={ok}")
            # 直接把 stdout 打出来（JSON）
            out = str(res.get("stdout") or "").strip()
            if out:
                print(out)
            continue
        if t == "skill":
            name = str((it.get("args") or {}).get("name") or "").strip()
            out = str(res.get("stdout") or "").strip()
            # skill 子命令通常会把产物路径 print 出来（取最后一行）
            last_line = out.splitlines()[-1].strip() if out else ""
            extra = f" out={last_line}" if last_line else ""
            print(f"skill:{name}: ok={ok}{extra}")
            continue

    return 0
