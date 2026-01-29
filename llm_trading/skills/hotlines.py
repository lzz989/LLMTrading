from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..akshare_source import DataSourceError, FetchParams
from ..data_cache import fetch_daily_cached
from ..indicators import add_atr, add_moving_averages
from ..pipeline import write_json


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _as_bool(x: Any) -> bool:
    return bool(x)


def _as_float(x: Any) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None
    return None if v != v else float(v)  # NaN guard


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _fmt_pct(x: Any) -> str:
    v = _as_float(x)
    if v is None:
        return "?"
    return f"{v*100:.2f}%"


@dataclass(frozen=True, slots=True)
class UniverseItem:
    symbol: str
    name: str
    tags: list[str]  # base tags（官方分类底座：类型/宽基/跨境/商品等），不要天天改含义
    tradable: bool = True


def load_universe_yaml(path: Path) -> tuple[list[UniverseItem], dict[str, Any]]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：请先安装 requirements.txt（需要 pyyaml）") from exc

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _as_dict(obj)
    items0 = _as_list(root.get("items"))

    items: list[UniverseItem] = []
    for it in items0:
        d = _as_dict(it)
        sym = _as_str(d.get("symbol")).strip()
        if not sym:
            continue
        name = _as_str(d.get("name")).strip() or "名称待解析"
        tags = [str(x).strip() for x in _as_list(d.get("tags")) if str(x).strip()]
        tradable = _as_bool(d.get("tradable")) if ("tradable" in d) else True
        items.append(UniverseItem(symbol=sym, name=name, tags=tags, tradable=bool(tradable)))

    meta = {
        "schema": _as_str(root.get("schema")).strip() or "llm_trading.hotlines_universe.v1",
        "updated_at": _as_str(root.get("updated_at")).strip(),
        "note": _as_str(root.get("note")).strip(),
        "count": int(len(items)),
    }
    return items, meta


def _maybe_fill_etf_names(items: list[UniverseItem]) -> dict[str, str]:
    """
    不瞎编名字：如果 universe 里写了“名称待解析”，尝试用本地缓存/数据源补全。
    失败就算了，照原样输出。
    """
    need = [x.symbol for x in items if ("待解析" in x.name or x.name.strip() in {"", "名称未知"})]
    if not need:
        return {}
    try:
        from ..etf_scan import load_etf_universe

        uni = load_etf_universe(include_all_funds=True)
        m = {u.symbol: u.name for u in uni if getattr(u, "symbol", None) and getattr(u, "name", None)}
        return {s: m[s] for s in need if s in m}
    except Exception:  # noqa: BLE001
        return {}


def _derive_topic_tags(name: str) -> list[str]:
    """
    基于 ETF 名称（相对“官方”的文本）派生主题 tags：
    - 这是为了做“主线热度”聚合，不依赖网页热榜；
    - 不追求完美，只求稳定可复核；匹配不到就归为“其他”。
    """
    n = str(name or "").replace(" ", "")
    out: list[str] = []

    def add(x: str) -> None:
        s = str(x or "").strip()
        if s and s not in out:
            out.append(s)

    # 宽基
    if "沪深300" in n:
        add("宽基")
        add("沪深300")
    if "上证50" in n:
        add("宽基")
        add("上证50")
    if "中证500" in n:
        add("宽基")
        add("中证500")

    # 跨境/港股科技（示例：中概互联/恒生科技）
    if ("中国互联" in n) or ("中概" in n):
        add("中概互联")
        add("港股科技")
    if "恒生科技" in n:
        add("恒生科技")
        add("港股科技")
    if "恒生" in n and "恒生科技" not in n:
        add("恒生")

    # 行业/主题（关键词启发式）
    kw_map: list[tuple[str, str]] = [
        ("半导体", "半导体"),
        ("芯片", "半导体"),
        ("证券", "券商"),
        ("券商", "券商"),
        ("传媒", "传媒"),
        ("有色", "有色"),
        ("黄金", "黄金"),
        ("煤炭", "煤炭"),
        ("煤", "煤炭"),
        ("石油", "油气"),
        ("原油", "油气"),
        ("化工", "化工"),
        ("医药", "医药"),
        ("创新药", "创新药"),
        ("军工", "军工"),
        ("银行", "银行"),
        ("消费", "消费"),
        ("白酒", "白酒"),
        ("新能源", "新能源"),
        ("光伏", "光伏"),
        ("锂", "锂电"),
        ("机器人", "机器人"),
        ("人工智能", "AI"),
        ("AI", "AI"),
        ("算力", "算力"),
        ("半导体设备", "半导体设备"),
    ]
    for kw, tag in kw_map:
        if kw and kw in n:
            add(tag)

    if not out:
        add("其他")
    return out


def compute_hotness_metrics(df) -> dict[str, Any]:
    """
    纯行情驱动（可复核）的热度/拥挤度 proxies。
    """
    if df is None or getattr(df, "empty", True):
        return {"ok": False}

    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover
        return {"ok": False}

    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2 = df2.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if df2.empty or "close" not in df2.columns:
        return {"ok": False}

    # 指标：MA20 + ATR14（拥挤度 proxy）
    df2 = add_moving_averages(df2, ma_fast=20, ma_slow=60)
    df2 = add_atr(df2, period=14, out_col="atr14")

    close = df2["close"].astype(float)
    vol = df2["volume"].astype(float) if "volume" in df2.columns else None
    amt = df2["amount"].astype(float) if "amount" in df2.columns else None

    def _ret(n: int) -> float | None:
        if len(df2) < (n + 1):
            return None
        try:
            return float(close.iloc[-1] / close.iloc[-1 - n] - 1.0)
        except Exception:  # noqa: BLE001
            return None

    ret_5 = _ret(5)
    ret_10 = _ret(10)
    ret_20 = _ret(20)

    # vol_ratio_20d：近5日均量 / 20日均量
    vol_ratio = None
    if vol is not None and len(df2) >= 20:
        v5 = float(vol.tail(5).mean())
        v20 = float(vol.tail(20).mean())
        if v20 > 0:
            vol_ratio = float(v5 / v20)

    amt_avg20 = None
    if amt is not None and len(df2) >= 20:
        a20 = float(amt.tail(20).mean())
        if a20 > 0:
            amt_avg20 = float(a20)

    ma20 = _as_float(df2.iloc[-1].get("ma20"))
    close_last = float(close.iloc[-1])
    close_vs_ma20 = None
    if ma20 is not None and ma20 > 0:
        close_vs_ma20 = float(close_last / ma20 - 1.0)

    atr14 = _as_float(df2.iloc[-1].get("atr14"))
    atr_pct_14 = None
    if atr14 is not None and close_last > 0:
        atr_pct_14 = float(atr14 / close_last)

    # 离 20 日高点回撤（追高风险 proxy）
    dd_20h = None
    if len(df2) >= 20 and "high" in df2.columns:
        try:
            h20 = float(df2["high"].astype(float).tail(20).max())
            if h20 > 0:
                dd_20h = float(close_last / h20 - 1.0)
        except Exception:  # noqa: BLE001
            dd_20h = None

    try:
        last_date = df2.iloc[-1]["date"]
        as_of = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
    except Exception:  # noqa: BLE001
        as_of = ""

    return {
        "ok": True,
        "as_of": as_of,
        "close": close_last,
        "ret_5d": ret_5,
        "ret_10d": ret_10,
        "ret_20d": ret_20,
        "vol_ratio_20d": vol_ratio,
        "amount_avg20": amt_avg20,
        "close_vs_ma20_pct": close_vs_ma20,
        "atr_pct_14": atr_pct_14,
        "dd_from_20d_high_pct": dd_20h,
    }


def _score_and_flags(metrics: dict[str, Any]) -> tuple[float, list[str]]:
    if not metrics.get("ok"):
        return 0.0, ["no_data"]

    r10 = float(metrics.get("ret_10d") or 0.0)
    r20 = float(metrics.get("ret_20d") or 0.0)
    vr = float(metrics.get("vol_ratio_20d") or 1.0)
    d_ma20 = float(metrics.get("close_vs_ma20_pct") or 0.0)
    atrp = metrics.get("atr_pct_14")
    dd20 = metrics.get("dd_from_20d_high_pct")

    # 分数：趋势+放量+强度（KISS；只用来排序，不当买卖按钮）
    a = _clamp01((r20 + 0.03) / 0.12)
    b = _clamp01((r10 + 0.02) / 0.08)
    c = _clamp01((vr - 0.8) / 1.4)
    d = _clamp01((d_ma20 + 0.01) / 0.08)
    score = float(0.40 * a + 0.20 * b + 0.25 * c + 0.15 * d)

    flags: list[str] = []
    if atrp is not None and float(atrp) >= 0.06:
        flags.append("crowded_volatility_high")
    if d_ma20 >= 0.10:
        flags.append("extended_from_ma20")
    if dd20 is not None and float(dd20) >= -0.02 and r10 >= 0.06:
        flags.append("chasing_near_20d_high")
    if r20 >= 0.18 and vr >= 1.8:
        flags.append("mania_like")

    return score, flags


def _render_md(report: dict[str, Any]) -> str:
    dt = _as_str(report.get("generated_at")).strip()
    as_of = _as_str(report.get("as_of")).strip()
    top_n = int(_as_float(report.get("top_n")) or 10)
    lines: list[str] = []
    lines.append("# 主线热度（hotlines）\n")
    lines.append(f"- generated_at: {dt}\n")
    lines.append(f"- as_of: {as_of}\n")
    lines.append(f"- top_n: {top_n}\n")

    lines.append("\n## 主线 Top（按 tag 聚合）\n")
    for it0 in _as_list(report.get("tags_rank"))[:top_n]:
        it = _as_dict(it0)
        tag = _as_str(it.get("tag")).strip()
        sc = float(it.get("score") or 0.0)
        flags = [str(x) for x in _as_list(it.get("risk_flags")) if str(x).strip()]
        reps = _as_list(it.get("representatives"))
        rep_s = ", ".join(
            [f"{_as_str(x.get('symbol'))}（{_as_str(x.get('name')) or '名称未知'}）" for x in reps if isinstance(x, dict)]
        )
        lines.append(f"- {tag}: score={sc:.3f} flags={','.join(flags) if flags else '-'} reps={rep_s}\n")

    lines.append("\n## ETF 明细（可复核）\n")
    for it0 in _as_list(report.get("etf_items")):
        it = _as_dict(it0)
        sym = _as_str(it.get("symbol")).strip()
        name = _as_str(it.get("name")).strip() or "名称未知"
        base_tags = [str(x) for x in _as_list(it.get("tags")) if str(x).strip()]
        topic_tags = [str(x) for x in _as_list(it.get("topic_tags")) if str(x).strip()]
        sc = float(it.get("score") or 0.0)
        m = _as_dict(it.get("metrics"))
        flags = [str(x) for x in _as_list(it.get("risk_flags")) if str(x).strip()]
        lines.append(
            f"- {sym}（{name}） base_tags={','.join(base_tags)} topic_tags={','.join(topic_tags)} score={sc:.3f} "
            f"ret20={_fmt_pct(m.get('ret_20d'))} volR={m.get('vol_ratio_20d')} "
            f"flags={','.join(flags) if flags else '-'}\n"
        )

    return "".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class HotlinesRunResult:
    out_md: Path
    out_json: Path
    report: dict[str, Any]


def run_hotlines(
    *,
    universe_path: str,
    out_md: str,
    out_json: str,
    top_n: int = 10,
    source: str = "auto",
    cache_ttl_hours: float = 24.0,
) -> HotlinesRunResult:
    uni_p = Path(universe_path)
    items, uni_meta = load_universe_yaml(uni_p)
    name_map = _maybe_fill_etf_names(items)

    # 拉数：内部走增量缓存（data/ 下，默认 gitignore）
    cache_dir = Path("data") / "cache" / "hotlines"

    etf_items: list[dict[str, Any]] = []
    as_of_max: str | None = None

    for u in items:
        name_eff = name_map.get(u.symbol, u.name)
        topic_tags = _derive_topic_tags(name_eff)
        try:
            df = fetch_daily_cached(
                FetchParams(asset="etf", symbol=u.symbol, source=str(source)),
                cache_dir=cache_dir,
                ttl_hours=float(cache_ttl_hours),
            )
        except (DataSourceError) as exc:
            etf_items.append(
                {
                    "symbol": u.symbol,
                    "name": name_eff,
                    "tags": list(u.tags),
                    "topic_tags": topic_tags,
                    "tradable": bool(u.tradable),
                    "ok": False,
                    "error": str(exc),
                    "metrics": None,
                    "score": 0.0,
                    "risk_flags": ["fetch_failed"],
                }
            )
            continue

        m = compute_hotness_metrics(df)
        score, flags = _score_and_flags(m)
        as_of = _as_str(m.get("as_of")).strip()
        if as_of and ((as_of_max is None) or (as_of > as_of_max)):
            as_of_max = as_of

        etf_items.append(
            {
                "symbol": u.symbol,
                "name": name_eff,
                "tags": list(u.tags),
                "topic_tags": topic_tags,
                "tradable": bool(u.tradable),
                "ok": bool(m.get("ok")),
                "metrics": m,
                "score": float(score),
                "risk_flags": flags,
            }
        )

    # tag 聚合：权重=amount_avg20（没有就等权）
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for it in etf_items:
        tags = [str(x) for x in _as_list(it.get("topic_tags")) if str(x).strip()]
        if not tags:
            tags = ["其他"]
        for t in tags:
            by_tag.setdefault(t, []).append(it)

    tags_rank: list[dict[str, Any]] = []
    for tag, rows in by_tag.items():
        num = 0.0
        den = 0.0
        # 抽 3 个代表 ETF（按 score）
        reps = sorted(rows, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:3]
        risk_union: list[str] = []
        for it in reps:
            for f in _as_list(it.get("risk_flags")):
                s = str(f).strip()
                if s and s not in risk_union:
                    risk_union.append(s)

        # 加权平均
        for it in rows:
            m = _as_dict(it.get("metrics"))
            w = float(m.get("amount_avg20") or 1.0)
            if w <= 0:
                w = 1.0
            sc = float(it.get("score") or 0.0)
            num += sc * w
            den += w
        score_tag = float(num / den) if den > 0 else 0.0

        tags_rank.append(
            {
                "tag": tag,
                "score": float(score_tag),
                "representatives": [
                    {"symbol": _as_str(x.get("symbol")), "name": _as_str(x.get("name")), "tradable": bool(x.get("tradable"))} for x in reps
                ],
                "risk_flags": risk_union,
            }
        )

    tags_rank.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    report = {
        "schema": "llm_trading.skill.hotlines.v1",
        "generated_at": datetime.now().isoformat(),
        "as_of": as_of_max,
        "top_n": int(top_n),
        "universe": {"path": str(uni_p.as_posix()), **uni_meta},
        "tags_rank": tags_rank,
        "etf_items": etf_items,
        "note": (
            "tags(底座)=官方分类口径（类型/宽基/跨境/商品等，尽量稳定）；"
            "topic_tags(派生)=从ETF名称关键词抽取的主题标签（用于主线聚合）。"
            "主线热度=行情驱动 proxy（趋势/放量/强度/波动），用于‘主线识别/拥挤度风险提示’，不是买卖按钮。"
        ),
    }

    out_md_p = Path(out_md)
    out_md_p.parent.mkdir(parents=True, exist_ok=True)
    out_md_p.write_text(_render_md(report), encoding="utf-8")

    out_json_p = Path(out_json)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json_p, report)

    return HotlinesRunResult(out_md=out_md_p, out_json=out_json_p, report=report)
