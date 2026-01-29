from __future__ import annotations

import argparse
import json
import runpy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_config
from ..llm_client import ChatMessage
from ..memory_store import append_daily_memory, build_prompt_memory_context, resolve_memory_paths
from ..pipeline import run_llm_text


def _read_text(path: Path, *, max_chars: int | None = None) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        return ""
    if max_chars is not None and max_chars > 0:
        return txt[: int(max_chars)]
    return txt


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _select_latest_run_dir() -> Path | None:
    """
    自动选一个“最近的 run 目录”：
    - outputs/run_* 优先
    - outputs/chat_run_* 次之

    这是给 skill 默认值用的；chat 链路会优先传入“本次 run 的 out_dir”。
    """
    out_root = Path("outputs")
    if not out_root.exists():
        return None
    dirs: list[Path] = []
    try:
        for p in out_root.iterdir():
            if not p.is_dir():
                continue
            n = p.name
            if n.startswith("run_") or n.startswith("chat_run_"):
                dirs.append(p)
    except Exception:  # noqa: BLE001
        dirs = []
    if not dirs:
        return None
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[0]


def _summarize_holdings_user(obj: dict[str, Any]) -> dict[str, Any]:
    holds = obj.get("holdings") if isinstance(obj.get("holdings"), list) else []
    out_holds: list[dict[str, Any]] = []
    for h in holds:
        if not isinstance(h, dict):
            continue
        st = h.get("stops") if isinstance(h.get("stops"), dict) else {}
        tr = h.get("trend") if isinstance(h.get("trend"), dict) else {}
        mom = h.get("momentum") if isinstance(h.get("momentum"), dict) else {}
        sig = h.get("signals") if isinstance(h.get("signals"), dict) else {}

        # 收敛：只保留会影响“可执行动作”的关键字段，别把整坨指标喂给模型浪费 token。
        trend_keep = {
            k: tr.get(k)
            for k in [
                "weekly_slope_12w",
                "weekly_ret_4w",
                "weekly_ret_12w",
                "weekly_ret_26w",
                "daily_slope_20d",
                "daily_ret_5d",
                "daily_ret_20d",
                "daily_ret_60d",
            ]
            if k in tr
        }
        mom_d = mom.get("daily") if isinstance(mom.get("daily"), dict) else {}
        mom_w = mom.get("weekly") if isinstance(mom.get("weekly"), dict) else {}
        mom_keep = {
            "daily": {k: mom_d.get(k) for k in ["rsi14", "adx14", "di_plus14", "di_minus14", "macd", "macd_signal", "macd_hist"] if k in mom_d},
            "weekly": {k: mom_w.get(k) for k in ["rsi14", "adx14", "di_plus14", "di_minus14", "macd", "macd_signal", "macd_hist"] if k in mom_w},
        }

        sig_keep = {k: sig.get(k) for k in ["soft_exit_daily_macd_ma20", "tp1_slow_bull"] if k in sig}

        out_holds.append(
            {
                "asset": h.get("asset"),
                "symbol": h.get("symbol"),
                "name": h.get("name"),
                "close": h.get("close"),
                "cost": h.get("cost"),
                "shares": h.get("shares"),
                "market_value": h.get("market_value"),
                "pnl_net_pct": h.get("pnl_net_pct"),
                "entry_style": h.get("entry_style"),
                "effective_stop": st.get("effective_stop"),
                "effective_ref": st.get("effective_ref"),
                "hard_stop": st.get("hard_stop"),
                "hard_ref": st.get("hard_ref"),
                "hard_enforced": st.get("hard_enforced"),
                "trend": trend_keep,
                "momentum": mom_keep,
                "signals": sig_keep,
            }
        )

    return {
        "as_of": obj.get("as_of"),
        "market_regime": obj.get("market_regime"),
        "portfolio": obj.get("portfolio"),
        "holdings": out_holds,
    }


def _summarize_signals(obj: dict[str, Any] | None, *, top_k: int = 15) -> list[dict[str, Any]]:
    if not obj or not isinstance(obj, dict):
        return []
    items = obj.get("items") if isinstance(obj.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for it in items[: int(top_k)]:
        if not isinstance(it, dict):
            continue
        meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
        ss = meta.get("strategy_signal") if isinstance(meta.get("strategy_signal"), dict) else {}
        fac = ss.get("factors") if isinstance(ss.get("factors"), dict) else {}
        pb = fac.get("pullback") if isinstance(fac.get("pullback"), dict) else {}
        det = pb.get("details") if isinstance(pb.get("details"), dict) else {}
        out.append(
            {
                "asset": it.get("asset"),
                "symbol": it.get("symbol"),
                "name": it.get("name"),
                "action": it.get("action"),
                "score": it.get("score"),
                "confidence": it.get("confidence"),
                "close": meta.get("close"),
                "pct_chg": meta.get("pct_chg"),
                "liquidity": meta.get("liquidity"),
                # pullback 关键信息（趋势回踩低吸最核心）
                "pullback": {
                    "is_valid": det.get("is_valid_pullback"),
                    "prior_gain_pct": det.get("prior_gain_pct"),
                    "pullback_pct": det.get("pullback_pct"),
                    "dist_to_ma_pct": det.get("dist_to_ma_pct"),
                    "days_from_high": det.get("days_from_high"),
                },
            }
        )
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((text or "").rstrip() + "\n", encoding="utf-8")


def _skill_strategy(args: argparse.Namespace) -> int:
    cfg = load_config()
    run_dir_raw = str(getattr(args, "run_dir", "") or "").strip()
    run_dir = Path(run_dir_raw) if run_dir_raw else (_select_latest_run_dir() or None)
    if run_dir is None or (not run_dir.exists()):
        raise SystemExit("找不到 run_dir：请先跑一次 run，或传 --run-dir outputs/run_YYYYMMDD")

    out_path = Path(str(getattr(args, "out", "") or "").strip() or str(Path("outputs") / "agents" / "strategy_action.md"))
    provider = str(getattr(args, "provider", "openai") or "openai").strip().lower()
    no_llm = bool(getattr(args, "no_llm", False))
    if not no_llm:
        # 没配 LLM 就自动降级，别硬炸（chat/rule planner 也会调用 skill）。
        if provider == "openai" and (cfg.openai() is None):
            no_llm = True
        if provider == "gemini" and (cfg.gemini() is None):
            no_llm = True

    # 读产物（尽量用结构化 JSON，report.md 只作为补充线索）
    report_md = _read_text(run_dir / "report.md", max_chars=4000)
    holdings_user = _read_json(run_dir / "holdings_user.json") or {}
    signals = _read_json(run_dir / "signals.json")
    signals_stock = _read_json(run_dir / "signals_stock.json")
    orders = _read_json(run_dir / "orders_next_open.json")

    holdings_sum = _summarize_holdings_user(holdings_user) if isinstance(holdings_user, dict) else {}
    signals_sum = _summarize_signals(signals, top_k=12)
    signals_stock_sum = _summarize_signals(signals_stock, top_k=15)

    if no_llm:
        # 规则兜底：不写花活，先把“能执行的价位”落出来。
        lines: list[str] = []
        lines.append("# 持仓策略动作（strategy）\n")
        lines.append(f"- generated_at: {datetime.now().date().isoformat()}\n")
        lines.append(f"- based_on_run_dir: `{run_dir.as_posix()}`\n")
        lines.append(f"- data_as_of: {holdings_sum.get('as_of')}\n")
        lines.append("\n## 一句话结论\n\n观望（规则模式：不调用 LLM，只输出风控价位）。\n")
        lines.append("## 终极动作（五选一）：观望\n")
        lines.append("## 持仓逐一风控（收盘触发→次日开盘执行）\n")
        hs = holdings_sum.get("holdings") if isinstance(holdings_sum.get("holdings"), list) else []
        for h in hs:
            sym = str(h.get("symbol") or "")
            name = str(h.get("name") or "名称未知")
            close = h.get("close")
            eff = h.get("effective_stop")
            ref = h.get("effective_ref")
            lines.append(f"- `{sym}（{name}）` close={close} effective_stop={eff}({ref})\n")
        _write_text(out_path, "".join(lines))
        print(str(out_path.resolve()))
        return 0

    # LLM：注入记忆（profile+长期+最近 daily），让动作更贴合你的纪律
    mp = resolve_memory_paths(project_root=cfg.project_root)
    mem_ctx = ""
    try:
        mem_ctx = build_prompt_memory_context(mp, include_long_term=True, include_profile=True, include_daily_days=2, max_chars=6000)
    except Exception:  # noqa: BLE001
        mem_ctx = ""

    template = _read_text(Path("references") / "action_template.md", max_chars=4000)
    system = (
        "你是一个交易框架的“策略执行清单生成器”。目标：把 run 产物收敛成一份可执行的动作清单。\n"
        "硬约束：\n"
        "1) 必须中文输出；允许 Markdown。\n"
        "2) 不要输出代码；不要自动下真实单；只给‘收盘触发→次日开盘执行(T+1)’的人工执行草案。\n"
        "3) 必须给出一个‘终极动作（五选一）’：观望 / 试错小仓 / 执行计划 / 减仓 / 退出。\n"
        "4) 所有关键结论必须落在可验证的价位/结构上（effective_stop/均线/关键低点等）。\n"
        "5) 小资金摩擦要考虑：单笔建议>=2000元（止损例外）。\n"
        "6) 输出尽量贴合模板结构。\n"
    )
    if mem_ctx.strip():
        system += "\n\n# 用户偏好/约束（持久记忆；如与本次输入冲突，以本次输入为准）\n" + mem_ctx.strip()
    if template.strip():
        system += "\n\n# 输出模板（参考）\n" + template.strip()

    user_payload = {
        "run_dir": str(run_dir.as_posix()),
        "report_md_excerpt": report_md,
        "holdings_user": holdings_sum,
        "signals_top": signals_sum,
        "signals_stock_top": signals_stock_sum,
        "orders_next_open": orders,
        "note": "请基于以上事实输出 strategy_action.md；不要编造不存在的字段/数据。",
    }

    md = run_llm_text(
        cfg,
        messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2))],
        provider=provider,
        temperature=0.2,
        max_output_tokens=1600,
    )
    _write_text(out_path, md)
    try:
        append_daily_memory(
            mp,
            title="skill:strategy",
            text=f"已生成策略动作清单：{out_path.as_posix()}（based_on={run_dir.as_posix()}）",
            source={"type": "auto", "cmd": "skill strategy", "run_dir": str(run_dir.as_posix())},
        )
    except Exception:  # noqa: BLE001
        pass
    print(str(out_path.resolve()))
    return 0


# --- research skill (news clues) ---


_RISK_PATTERNS: dict[str, list[str]] = {
    "regulatory": ["立案", "调查", "处罚", "罚款", "问询", "警示函", "监管", "通报", "函"],
    "earnings": ["预亏", "亏损", "大幅下降", "下滑", "由盈转亏", "减值", "业绩预告", "业绩快报"],
    "shareholder": ["减持", "质押", "解禁", "回购", "增持", "股东", "举牌"],
    "listing": ["停牌", "复牌", "终止上市", "退市"],
    "macro_sentiment": ["利空", "过热", "降温", "收紧", "加息", "通胀", "风险"],
}


@dataclass(frozen=True, slots=True)
class NewsItem:
    query: str
    title: str
    content: str
    published_at: str
    source: str
    url: str
    flags: list[str]


def _risk_flags(title: str, content: str) -> list[str]:
    text = f"{title} {content}"
    out: list[str] = []
    for k, pats in _RISK_PATTERNS.items():
        for p in pats:
            if p and p in text:
                out.append(k)
                break
    return out


def _fetch_eastmoney_news(*, query: str, pages: int, page_size: int, timeout: float, sleep_sec: float) -> list[NewsItem]:
    """
    MVP：东方财富搜索 jsonp 接口抓“新闻线索”。
    只抓取：标题/时间/来源/链接/摘要；线索≠事实。
    """
    import random
    import re
    import time
    from urllib.parse import quote

    import requests

    q = str(query).strip()
    if not q:
        return []
    pages = max(1, int(pages))
    page_size = int(page_size)
    if page_size <= 0:
        page_size = 10
    if page_size > 50:
        page_size = 50

    def now_ms() -> int:
        return int(time.time() * 1000)

    def jsonp_loads(text: str) -> dict[str, Any]:
        s = (text or "").strip()
        l = s.find("(")
        r = s.rfind(")")
        if l < 0 or r < 0 or r <= l:
            raise ValueError("unexpected jsonp format")
        return json.loads(s[l + 1 : r])

    def clean_em(text: str) -> str:
        t = str(text or "")
        t = re.sub(r"\\(<em>", "", t)
        t = re.sub(r"</em>\\)", "", t)
        t = re.sub(r"<em>", "", t)
        t = re.sub(r"</em>", "", t)
        t = t.replace("\\u3000", "").replace("\\r\\n", " ")
        return t.strip()

    search_url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "referer": f"https://so.eastmoney.com/news/s?keyword={quote(q, safe='')}",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    }

    items: list[NewsItem] = []
    for page_index in range(1, pages + 1):
        inner_param = {
            "uid": "",
            "keyword": q,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": int(page_index),
                    "pageSize": int(page_size),
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }
        cb = f"jQuery{random.randint(10**19, 10**20-1)}_{now_ms()}"
        params = {"cb": cb, "param": json.dumps(inner_param, ensure_ascii=False), "_": str(now_ms())}
        resp = requests.get(search_url, params=params, headers=headers, timeout=float(timeout))
        resp.raise_for_status()
        data = jsonp_loads(resp.text)
        rows = ((data.get("result") or {}).get("cmsArticleWebOld") or []) if isinstance(data, dict) else []

        for r in rows:
            if not isinstance(r, dict):
                continue
            code = str(r.get("code") or "").strip()
            url = str(r.get("url") or "").strip()
            if not url and code:
                url = f"http://finance.eastmoney.com/a/{code}.html"
            title = clean_em(r.get("title"))
            content = clean_em(r.get("content"))
            published_at = str(r.get("date") or "").strip()
            source = str(r.get("mediaName") or "").strip()
            flags = _risk_flags(title, content)
            items.append(NewsItem(query=q, title=title, content=content, published_at=published_at, source=source, url=url, flags=flags))

        if float(sleep_sec) > 0 and page_index < pages:
            time.sleep(float(sleep_sec))

    # 去重（按 URL/标题）
    dedup: dict[str, NewsItem] = {}
    for it in items:
        k = it.url or it.title
        if not k:
            continue
        dedup.setdefault(k, it)
    return list(dedup.values())


def _derive_queries_from_run_dir(run_dir: Path | None) -> list[str]:
    # 优先从 run_dir 读；没有就从 data/user_holdings.json 抓。
    queries: list[str] = []
    if run_dir and run_dir.exists():
        hu = _read_json(run_dir / "holdings_user.json") or {}
        hs = hu.get("holdings") if isinstance(hu.get("holdings"), list) else []
        for h in hs:
            if not isinstance(h, dict):
                continue
            sym = str(h.get("symbol") or "")
            # 用 6 位数字做 query（减少前缀干扰）
            digits = "".join([ch for ch in sym if ch.isdigit()])
            if len(digits) == 6:
                queries.append(digits)

        ss = _read_json(run_dir / "signals_stock.json") or {}
        items = ss.get("items") if isinstance(ss.get("items"), list) else []
        for it in items[:5]:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "")
            digits = "".join([ch for ch in sym if ch.isdigit()])
            if len(digits) == 6:
                queries.append(digits)

    if not queries:
        try:
            uh = _read_json(Path("data") / "user_holdings.json") or {}
            ps = uh.get("positions") if isinstance(uh.get("positions"), list) else []
            for p in ps:
                if not isinstance(p, dict):
                    continue
                sym = str(p.get("symbol") or "")
                digits = "".join([ch for ch in sym if ch.isdigit()])
                if len(digits) == 6:
                    queries.append(digits)
        except Exception:  # noqa: BLE001
            pass

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        s = str(q).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _write_news_outputs(*, out_dir: Path, items: list[NewsItem]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "news_raw.json"
    out_md = out_dir / "news_digest.md"

    payload = {
        "schema": "llm_trading.skill.research.news_raw.v1",
        "generated_at": datetime.now().isoformat(),
        "count": int(len(items)),
        "items": [
            {
                "query": x.query,
                "title": x.title,
                "content": x.content,
                "published_at": x.published_at,
                "source": x.source,
                "url": x.url,
                "flags": list(x.flags),
            }
            for x in items
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # digest：按 query 分组
    byq: dict[str, list[dict[str, Any]]] = {}
    for it in payload["items"]:
        q = str(it.get("query") or "")
        byq.setdefault(q, []).append(it)

    for q in list(byq.keys()):
        byq[q] = sorted(byq[q], key=lambda x: str(x.get("published_at") or ""), reverse=True)

    lines: list[str] = []
    lines.append("# 新闻线索摘要（Eastmoney，多标的汇总）\n")
    lines.append(f"- generated_at: {payload['generated_at']}\n")
    lines.append(f"- total_items: {payload['count']}\n")
    lines.append(f"- queries: {', '.join([q for q in sorted(byq.keys()) if q])}\n")

    from collections import Counter

    for q in [x for x in sorted(byq.keys()) if x]:
        q_items = byq[q]
        times = [str(x.get("published_at") or "") for x in q_items if x.get("published_at")]
        tr0 = min(times) if times else ""
        tr1 = max(times) if times else ""
        c = Counter()
        for it in q_items:
            for f in it.get("flags") or []:
                c[str(f)] += 1
        lines.append("\n---\n")
        lines.append(f"\n## query: {q}\n")
        lines.append(f"- count: {len(q_items)}\n")
        lines.append(f"- time_range: {tr0} ~ {tr1}\n")
        rf = ", ".join([f"{k}:{v}" for k, v in c.most_common(8)]) if c else "(none)"
        lines.append(f"- risk_flags: {rf}\n")
        lines.append("\n| 发布时间 | 来源 | 标题 | flags |\n|---|---|---|---|\n")
        for it in q_items[:15]:
            t = str(it.get("published_at") or "")
            src = str(it.get("source") or "")
            title = str(it.get("title") or "").replace("\n", " ").replace("|", "\\|")
            flags = ",".join(it.get("flags") or [])
            url = str(it.get("url") or "")
            title2 = f"[{title}]({url})" if url else title
            lines.append(f"| {t} | {src} | {title2} | {flags} |\n")

    out_md.write_text("".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_json, out_md


def _skill_research(args: argparse.Namespace) -> int:
    cfg = load_config()
    out_dir = Path(str(getattr(args, "out_dir", "") or "").strip() or str(Path("outputs") / "agents"))
    provider = str(getattr(args, "provider", "openai") or "openai").strip().lower()
    no_llm = bool(getattr(args, "no_llm", False))
    if not no_llm:
        if provider == "openai" and (cfg.openai() is None):
            no_llm = True
        if provider == "gemini" and (cfg.gemini() is None):
            no_llm = True

    run_dir_raw = str(getattr(args, "run_dir", "") or "").strip()
    run_dir = Path(run_dir_raw) if run_dir_raw else None

    queries_raw = str(getattr(args, "queries", "") or "").strip()
    queries = [x.strip() for x in queries_raw.split(",") if x.strip()] if queries_raw else []
    if not queries:
        queries = _derive_queries_from_run_dir(run_dir)
    if not queries:
        raise SystemExit("research: 没有 query：传 --queries 关键词/代码，或提供 --run-dir")

    pages = int(getattr(args, "pages", 2) or 2)
    page_size = int(getattr(args, "page_size", getattr(args, "page-size", 10)) or 10)  # argparse 兼容

    all_items: list[NewsItem] = []
    for q in queries:
        try:
            all_items.extend(_fetch_eastmoney_news(query=q, pages=pages, page_size=page_size, timeout=12.0, sleep_sec=0.2))
        except Exception:  # noqa: BLE001
            continue

    out_json, out_md = _write_news_outputs(out_dir=out_dir, items=all_items)

    # research.md：LLM 可选；没有 LLM 就给规则摘要
    out_report = out_dir / "research.md"
    if no_llm:
        lines: list[str] = []
        lines.append("# 舆情/新闻线索（research）\n\n")
        lines.append(f"- generated_at: {datetime.now().date().isoformat()}\n")
        lines.append(f"- queries: {', '.join(queries)}\n")
        lines.append("- note: 线索≠事实；关键结论需回公告/财报核验。\n")
        lines.append(f"- artifacts: `{out_json.as_posix()}` / `{out_md.as_posix()}`\n")
        _write_text(out_report, "".join(lines))
        print(str(out_report.resolve()))
        return 0

    mp = resolve_memory_paths(project_root=cfg.project_root)
    mem_ctx = ""
    try:
        mem_ctx = build_prompt_memory_context(mp, include_long_term=True, include_profile=True, include_daily_days=2, max_chars=5000)
    except Exception:  # noqa: BLE001
        mem_ctx = ""

    template = _read_text(Path("references") / "report_template.md", max_chars=2500)

    # 把线索压缩成“可喂给模型”的摘要（避免把 140 条全塞进去浪费 token）
    from collections import Counter, defaultdict

    byq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    payload = _read_json(out_json) or {}
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    for it in items:
        if not isinstance(it, dict):
            continue
        byq[str(it.get("query") or "")].append(it)
    for q in list(byq.keys()):
        byq[q] = sorted(byq[q], key=lambda x: str(x.get("published_at") or ""), reverse=True)

    stats: dict[str, Any] = {}
    highlights: dict[str, Any] = {}
    for q in sorted([x for x in byq.keys() if x]):
        c = Counter()
        flagged: list[dict[str, Any]] = []
        for it in byq[q]:
            for f in it.get("flags") or []:
                c[str(f)] += 1
            if it.get("flags"):
                flagged.append(
                    {
                        "published_at": it.get("published_at"),
                        "source": it.get("source"),
                        "title": it.get("title"),
                        "url": it.get("url"),
                        "flags": it.get("flags"),
                    }
                )
        stats[q] = {"count": len(byq[q]), "risk_flags": dict(c)}
        highlights[q] = {"flagged_top": flagged[:8], "recent_top": byq[q][:8]}

    system = (
        "你是交易框架的 research 模块，负责把新闻/舆情线索整理成‘结论-证据-不确定性’。\n"
        "硬约束：\n"
        "1) 必须中文输出；允许 Markdown。\n"
        "2) 媒体稿只能当线索；不得把线索写成事实；必须写‘待核验清单’。\n"
        "3) 不做收益承诺；不输出投资建议（只能写‘对计划的影响/风险点’）。\n"
        "4) 输出尽量贴合模板结构。\n"
    )
    if mem_ctx.strip():
        system += "\n\n# 用户偏好/约束（持久记忆；如与本次输入冲突，以本次输入为准）\n" + mem_ctx.strip()
    if template.strip():
        system += "\n\n# 输出模板（参考）\n" + template.strip()

    user_payload = {
        "queries": queries,
        "stats": stats,
        "highlights": highlights,
        "artifacts": {"news_raw_json": out_json.as_posix(), "news_digest_md": out_md.as_posix()},
        "note": "只基于以上线索写报告；明确哪些是线索、哪些需要核验；别编造公告/财报内容。",
    }
    md = run_llm_text(
        cfg,
        messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2))],
        provider=provider,
        temperature=0.2,
        max_output_tokens=1400,
    )
    _write_text(out_report, md)
    try:
        append_daily_memory(
            mp,
            title="skill:research",
            text=f"已生成舆情线索：{out_report.as_posix()}（queries={','.join(queries)}）",
            source={"type": "auto", "cmd": "skill research"},
        )
    except Exception:  # noqa: BLE001
        pass
    print(str(out_report.resolve()))
    return 0


def _skill_backtest(args: argparse.Namespace) -> int:
    cfg = load_config()
    root = cfg.project_root
    script = root / ".codex" / "skills" / "backtest" / "scripts" / "backtest_exit_signals.py"
    if not script.exists():
        raise SystemExit(f"backtest 脚本不存在：{script}")

    # 直接复用脚本（保留它的可复现口径），这里不重写逻辑。
    argv = [
        str(script),
        "--asset",
        str(getattr(args, "asset", "etf") or "etf"),
        "--symbols",
        str(getattr(args, "symbols", "") or "").strip(),
        "--source",
        str(getattr(args, "source", "akshare") or "akshare"),
        "--out",
        str(getattr(args, "out", "") or str(Path("outputs") / "agents" / "backtest_report.md")),
        "--cache-ttl-hours",
        str(float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)),
        "--fee-bps",
        str(float(getattr(args, "fee_bps", 10.0) or 10.0)),
        "--slippage-bps",
        str(float(getattr(args, "slippage_bps", 5.0) or 5.0)),
    ]
    start = str(getattr(args, "start", "") or "").strip()
    end = str(getattr(args, "end", "") or "").strip()
    if start:
        argv.extend(["--start", start])
    if end:
        argv.extend(["--end", end])

    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code != 0:
            raise
    finally:
        sys.argv = old_argv
    return 0


def _skill_five_schools(args: argparse.Namespace) -> int:
    from ..skills.five_schools import run_five_schools

    asset = str(getattr(args, "asset", "stock") or "stock").strip().lower()
    symbols_raw = str(getattr(args, "symbols", "") or "").strip()
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("symbols 为空：用 --symbols sh513050,sh600188")

    run_dir = str(getattr(args, "run_dir", "") or "").strip() or None
    out_md = str(getattr(args, "out", "") or str(Path("outputs") / "agents" / "five_schools.md")).strip()
    out_json = str(getattr(args, "out_json", "") or str(Path("outputs") / "agents" / "five_schools.json")).strip()
    source = str(getattr(args, "source", "auto") or "auto").strip().lower()
    freq = str(getattr(args, "freq", "weekly") or "weekly").strip().lower()

    res = run_five_schools(
        asset=asset,
        symbols=symbols,
        run_dir=run_dir,
        out_md=out_md,
        out_json=out_json,
        source=source,
        freq=freq,
    )
    print(str(res.out_md.resolve()))
    return 0


def _skill_hotlines(args: argparse.Namespace) -> int:
    from ..skills.hotlines import run_hotlines

    universe = str(getattr(args, "universe", "") or str(Path("config") / "hotlines_universe.yaml")).strip()
    out_md = str(getattr(args, "out", "") or str(Path("outputs") / "agents" / "hotlines.md")).strip()
    out_json = str(getattr(args, "out_json", "") or str(Path("outputs") / "agents" / "hotlines.json")).strip()
    top_n = int(getattr(args, "top", 10) or 10)
    source = str(getattr(args, "source", "auto") or "auto").strip().lower()
    ttl = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)

    res = run_hotlines(
        universe_path=universe,
        out_md=out_md,
        out_json=out_json,
        top_n=top_n,
        source=source,
        cache_ttl_hours=ttl,
    )
    print(str(res.out_md.resolve()))
    return 0


def cmd_skill(args: argparse.Namespace) -> int:
    subcmd = str(getattr(args, "skill_cmd", "") or "").strip().lower()
    if subcmd == "strategy":
        return _skill_strategy(args)
    if subcmd == "research":
        return _skill_research(args)
    if subcmd == "backtest":
        return _skill_backtest(args)
    if subcmd == "five_schools":
        return _skill_five_schools(args)
    if subcmd == "hotlines":
        return _skill_hotlines(args)
    raise SystemExit(f"未知 skill 子命令：{subcmd}")
