from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .portfolio import infer_theme


@dataclass(frozen=True, slots=True)
class ThemeHotness:
    theme: str
    score_avg: float
    score_max: float
    count: int
    top_etfs: list[dict[str, Any]]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _is_allowed_stock_symbol(sym: str, market_access: dict[str, Any] | None) -> bool:
    """
    过滤买不了的市场：
    - 科创板：sh688*/sh689*
    - 创业板：sz300*/sz301*
    - 北交所：bj*
    """
    s = str(sym or "").strip().lower()
    if not s:
        return False
    if s.startswith("bj"):
        return bool((market_access or {}).get("allow_bj", False))
    if s.startswith("sh") and s[2:].startswith(("688", "689")):
        return bool((market_access or {}).get("allow_star", False))
    if s.startswith("sz") and s[2:].startswith(("300", "301")):
        return bool((market_access or {}).get("allow_cyb", False))
    return True


def compute_theme_hotness_from_signals(
    items: list[dict[str, Any]],
    *,
    top_n: int = 50,
    min_score: float = 0.0,
    max_etfs_per_theme: int = 3,
) -> list[ThemeHotness]:
    """
    基于 ETF signals 计算“主题热度”（主线线索）：
    - 取 score 排名前 top_n 的 ETF
    - 主题由名称关键词粗分（infer_theme）
    - 主题分数=该主题 TopK 的平均 score
    """
    etfs = [it for it in items if str(it.get("asset") or "").lower() == "etf"]
    etfs = sorted(etfs, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
    if top_n > 0:
        etfs = etfs[: int(top_n)]

    buckets: dict[str, list[dict[str, Any]]] = {}
    for it in etfs:
        score = _safe_float(it.get("score", 0.0))
        if score < float(min_score):
            continue
        name = str(it.get("name") or "")
        theme = infer_theme(name)
        buckets.setdefault(theme, []).append(it)

    out: list[ThemeHotness] = []
    for theme, items_t in buckets.items():
        items_t = sorted(items_t, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
        top_items = items_t[: max(1, int(max_etfs_per_theme))]
        scores = [_safe_float(x.get("score", 0.0)) for x in top_items]
        if not scores:
            continue
        score_avg = float(sum(scores) / len(scores))
        score_max = float(max(scores))
        top_etfs = [
            {
                "symbol": it.get("symbol"),
                "name": it.get("name"),
                "score": _safe_float(it.get("score", 0.0)),
                "action": it.get("action"),
                "close": (it.get("meta") or {}).get("close"),
                "as_of": (it.get("meta") or {}).get("as_of"),
            }
            for it in top_items
        ]
        out.append(ThemeHotness(theme=theme, score_avg=score_avg, score_max=score_max, count=len(items_t), top_etfs=top_etfs))

    out.sort(key=lambda x: (x.score_avg, x.score_max, x.count), reverse=True)
    return out


def pick_satellite_stocks(
    stock_items: list[dict[str, Any]],
    *,
    themes: list[str],
    market_access: dict[str, Any] | None,
    min_trade_notional_yuan: float = 3000.0,
    max_picks: int = 2,
    allow_fallback: bool = True,
) -> list[dict[str, Any]]:
    """
    从股票 signals 中选“卫星个股候选”：
    - 过滤买不了的市场
    - 只保留主题匹配（基于名称关键词），否则标记为 other
    - 优先 entry，再 watch
    """
    items_all = []
    for it in stock_items:
        sym = str(it.get("symbol") or "").strip()
        if not _is_allowed_stock_symbol(sym, market_access):
            continue
        name = str(it.get("name") or "")
        theme = infer_theme(name)
        close = _safe_float((it.get("meta") or {}).get("close"), 0.0)
        lot_cost = float(close * 100.0) if close > 0 else None
        items_all.append(
            {
                "symbol": sym,
                "name": name,
                "theme": theme,
                "theme_match": bool((not themes) or theme in themes),
                "score": _safe_float(it.get("score", 0.0)),
                "action": it.get("action"),
                "close": close if close > 0 else None,
                "lot_cost_yuan": lot_cost,
                "min_trade_notional_yuan": float(min_trade_notional_yuan),
            }
        )

    def _rank_key(x: dict[str, Any]):
        action = str(x.get("action") or "")
        action_rank = 2 if action == "entry" else 1 if action == "watch" else 0
        return (action_rank, _safe_float(x.get("score", 0.0)))

    # 先取主题匹配的
    items_theme = [x for x in items_all if bool(x.get("theme_match"))]
    items_theme = sorted(items_theme, key=_rank_key, reverse=True)

    picks = items_theme[: max(0, int(max_picks))]
    if allow_fallback and len(picks) < int(max_picks):
        # 不足则用全量补齐（避免“主题太窄导致无票”）
        items_all = sorted(items_all, key=_rank_key, reverse=True)
        for it in items_all:
            if len(picks) >= int(max_picks):
                break
            if any(x.get("symbol") == it.get("symbol") for x in picks):
                continue
            picks.append(it)
    return picks


def theme_hotness_to_dict(items: list[ThemeHotness]) -> list[dict[str, Any]]:
    return [
        {
            "theme": it.theme,
            "score_avg": it.score_avg,
            "score_max": it.score_max,
            "count": it.count,
            "top_etfs": it.top_etfs,
        }
        for it in items
    ]


def render_theme_hotness_md(
    *,
    hotness: list[ThemeHotness],
    satellite_candidates: list[dict[str, Any]] | None,
    note: str | None = None,
) -> str:
    lines = ["# 主线热度（ETF）& 卫星候选（Stock）", ""]
    if note:
        lines.append(note)
        lines.append("")

    lines.append("## 主线热度（Top 3）")
    if not hotness:
        lines.append("- 暂无可用数据（signals 缺失或 score 为空）。")
    else:
        for i, it in enumerate(hotness[:3], start=1):
            lines.append(f"- {i}. theme={it.theme} | score_avg={it.score_avg:.3f} | count={it.count}")
            for etf in it.top_etfs[:3]:
                lines.append(
                    f"  - {etf.get('symbol')}（{etf.get('name')}） score={etf.get('score'):.3f} close={etf.get('close')}"
                )

    lines.append("")
    lines.append("## 卫星候选（Stock，Top 2）")
    if not satellite_candidates:
        lines.append("- 暂无候选（可能是主题不匹配 / 股票池限制 / market_access 限制）。")
    else:
        for it in satellite_candidates:
            lc = it.get("lot_cost_yuan")
            lc_s = f"{lc:.0f}" if isinstance(lc, (int, float)) and lc else "NA"
            tag = "匹配" if it.get("theme_match") else "fallback"
            lines.append(
                f"- {it.get('symbol')}（{it.get('name')}） theme={it.get('theme')} tag={tag} score={it.get('score'):.3f} action={it.get('action')} lot≈{lc_s} 元"
            )

    return "\n".join(lines).strip() + "\n"
