#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻线索抓取 + 摘要（MVP）

目的：
- 让“行业/基本面 research”能快速拉一份可复核的新闻线索清单
- 只抓取：标题/时间/来源/链接/摘要（线索≠事实）

数据源：
- 东方财富搜索（https://so.eastmoney.com/news/s?keyword=...）

注意：
- 这个脚本的输出只能当“线索汇总”，关键结论必须回到公告/财报/监管披露核验。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


EM_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"


RISK_PATTERNS: dict[str, list[str]] = {
    # 这些不是“情绪打分”，只是为了快速标红，别过度解读
    "regulatory": ["立案", "调查", "处罚", "罚款", "问询", "警示函", "监管", "通报", "函"],
    "earnings": ["预亏", "亏损", "大幅下降", "下滑", "由盈转亏", "减值", "业绩预告", "业绩快报"],
    "shareholder": ["减持", "质押", "解禁", "回购", "增持", "股东"],
    "listing": ["停牌", "复牌", "终止上市", "退市"],
    "macro_sentiment": ["利空", "过热", "降温", "收紧", "加息", "通胀", "风险"],
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _jsonp_loads(text: str) -> dict[str, Any]:
    # 兼容：callback({...})
    s = (text or "").strip()
    l = s.find("(")
    r = s.rfind(")")
    if l < 0 or r < 0 or r <= l:
        raise ValueError("unexpected jsonp format")
    return json.loads(s[l + 1 : r])


def _clean_em(text: str) -> str:
    t = str(text or "")
    # 去掉搜索高亮
    t = re.sub(r"\(<em>", "", t)
    t = re.sub(r"</em>\)", "", t)
    t = re.sub(r"<em>", "", t)
    t = re.sub(r"</em>", "", t)
    # 常见空白
    t = t.replace("\u3000", "").replace("\r\n", " ")
    return t.strip()


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
    for k, pats in RISK_PATTERNS.items():
        for p in pats:
            if p and p in text:
                out.append(k)
                break
    return out


def fetch_eastmoney_news(
    *,
    query: str,
    pages: int,
    page_size: int,
    timeout: float,
    sleep_sec: float,
) -> list[NewsItem]:
    q = str(query).strip()
    if not q:
        raise ValueError("--query 不能为空")

    pages = max(1, int(pages))
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("--page-size 必须 > 0")
    if page_size > 50:
        page_size = 50  # 保守点，别把人家接口打爆

    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        # Header 必须是 latin-1；keyword 含中文时要先 URL 编码
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
        cb = f"jQuery{random.randint(10000000000000000000, 99999999999999999999)}_{_now_ms()}"
        params = {
            "cb": cb,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": str(_now_ms()),
        }
        resp = requests.get(EM_SEARCH_URL, params=params, headers=headers, timeout=float(timeout))
        resp.raise_for_status()
        data = _jsonp_loads(resp.text)
        rows = ((data.get("result") or {}).get("cmsArticleWebOld") or []) if isinstance(data, dict) else []

        for r in rows:
            if not isinstance(r, dict):
                continue
            code = str(r.get("code") or "").strip()
            url = str(r.get("url") or "").strip()
            if not url and code:
                url = f"http://finance.eastmoney.com/a/{code}.html"
            title = _clean_em(r.get("title"))
            content = _clean_em(r.get("content"))
            published_at = str(r.get("date") or "").strip()
            source = str(r.get("mediaName") or "").strip()
            flags = _risk_flags(title, content)
            items.append(
                NewsItem(
                    query=q,
                    title=title,
                    content=content,
                    published_at=published_at,
                    source=source,
                    url=url,
                    flags=flags,
                )
            )

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


def _md_escape(s: str) -> str:
    t = str(s or "")
    t = t.replace("|", "\\|")
    t = t.replace("\n", " ")
    return t.strip()


def write_outputs(items: list[NewsItem], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "codex.skill.research.news_raw.v1",
        "generated_at": datetime.now().isoformat(),
        "count": int(len(items)),
        "items": [asdict(x) for x in items],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 统计：flag 次数
    flag_counts: dict[str, int] = {}
    for it in items:
        for f in it.flags:
            flag_counts[f] = int(flag_counts.get(f, 0)) + 1

    # markdown digest
    lines: list[str] = []
    lines.append("# 新闻线索摘要（Eastmoney）\n")
    lines.append(f"- generated_at: {payload['generated_at']}\n")
    lines.append(f"- count: {payload['count']}\n")
    if items:
        lines.append(f"- query: {items[0].query}\n")
        # 尽量给个时间范围
        dates = [it.published_at for it in items if it.published_at]
        if dates:
            ds = sorted(dates)
            lines.append(f"- time_range: {ds[0]} ~ {ds[-1]}\n")
    if flag_counts:
        top = sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        lines.append("- risk_flags_top:\n")
        for k, v in top:
            lines.append(f"  - {k}: {v}\n")
    lines.append("\n## 明细\n")
    lines.append("| 发布时间 | 来源 | 标题 | 链接 | flags |\n")
    lines.append("|---|---|---|---|---|\n")
    for it in sorted(items, key=lambda x: x.published_at or "", reverse=True)[:200]:
        flags = ",".join(it.flags or [])
        title = _md_escape(it.title)
        url = _md_escape(it.url)
        lines.append(
            f"| {_md_escape(it.published_at)} | {_md_escape(it.source)} | {title} | {url} | {_md_escape(flags)} |\n"
        )
    lines.append("\n## 备注\n")
    lines.append("- 这是“线索汇总”，不是事实背书；关键点请回到公告/财报/监管披露核验。\n")
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="股票代码或关键词（支持中文）")
    p.add_argument("--pages", type=int, default=3)
    p.add_argument("--page-size", type=int, default=10)
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--sleep-sec", type=float, default=0.2)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    items = fetch_eastmoney_news(
        query=str(args.query),
        pages=int(args.pages),
        page_size=int(args.page_size),
        timeout=float(args.timeout),
        sleep_sec=float(args.sleep_sec),
    )
    write_outputs(items, out_json=Path(str(args.out_json)), out_md=Path(str(args.out_md)))
    print(str(Path(str(args.out_md)).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
