from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ChainCategory:
    key: str
    name: str
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EtfMetric:
    symbol: str
    name: str
    close: float | None
    as_of: str | None
    ma20: float | None
    ma60: float | None
    ret5: float | None
    ret20: float | None
    ret60: float | None
    dd20: float | None
    heat_score: float | None
    state: str
    amount: float | None


CHAIN_CATEGORIES: tuple[ChainCategory, ...] = (
    ChainCategory("precious", "黄金/白银（避险）", ("黄金", "白银", "贵金属")),
    ChainCategory("base_metals", "铜/铝/有色（工业复苏）", ("有色", "铜", "铝", "锌", "镍", "锡", "稀土", "金属矿业")),
    ChainCategory("energy_chem", "石油/化工（通胀传导）", ("石油", "原油", "油气", "煤炭", "化工", "化学", "油服")),
    ChainCategory("agri", "农产品（全面通胀）", ("农业", "农牧渔", "粮", "粮食", "玉米", "大豆", "豆", "油脂", "棉", "糖", "农产品")),
)


def _load_etf_name_map() -> dict[str, str]:
    from .symbol_names import load_universe_name_map

    return load_universe_name_map("etf", ttl_hours=24.0)


def _categorize_name(name: str) -> ChainCategory | None:
    n = str(name or "")
    for cat in CHAIN_CATEGORIES:
        if any(k in n for k in cat.keywords):
            return cat
    return None


def _load_price_df(symbol: str) -> Any | None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    s = str(symbol or "").strip().lower()
    if not s:
        return None
    path = Path("data") / "cache" / "etf" / f"etf_{s}_qfq.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True):
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _compute_metrics(df, *, min_days: int = 80) -> dict[str, Any] | None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    if df is None or getattr(df, "empty", True):
        return None
    if len(df) < int(min_days):
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    if close is None or close.isna().all():
        return None
    close = close.dropna()
    if close.empty:
        return None

    last = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

    def _ret(n: int) -> float | None:
        if len(close) < n + 1:
            return None
        base = float(close.iloc[-(n + 1)])
        if base == 0:
            return None
        return last / base - 1.0

    ret5 = _ret(5)
    ret20 = _ret(20)
    ret60 = _ret(60)

    high20 = float(close.iloc[-20:].max()) if len(close) >= 20 else None
    dd20 = (last / high20 - 1.0) if (high20 is not None and high20 > 0) else None

    amount = None
    if "amount" in df.columns:
        amount = _safe_float(df["amount"].iloc[-1])

    as_of = None
    if "date" in df.columns:
        try:
            d = df["date"].iloc[-1]
            as_of = d.date().isoformat() if hasattr(d, "date") else str(d)
        except Exception:  # noqa: BLE001
            as_of = None

    # 状态判定（偏“回踩低吸”口径）
    state = "数据不足"
    if ma20 is not None and ma60 is not None:
        dist20 = last / ma20 - 1.0 if ma20 else 0.0
        if last < ma60 * 0.99:
            state = "破位"
        elif dist20 <= -0.05 or (dd20 is not None and dd20 <= -0.08):
            state = "深回踩"
        elif dist20 <= -0.02 or (dd20 is not None and dd20 <= -0.04):
            state = "回踩"
        elif dist20 >= 0.05 and (dd20 is not None and dd20 >= -0.02):
            state = "过热"
        else:
            state = "趋势中"

    # 热度分：近期动量为主，避免只看一两天
    r5 = ret5 if ret5 is not None else 0.0
    r20 = ret20 if ret20 is not None else 0.0
    r60 = ret60 if ret60 is not None else 0.0
    heat_score = 100.0 * (0.5 * r20 + 0.3 * r60 + 0.2 * r5)

    return {
        "close": last,
        "as_of": as_of,
        "ma20": ma20,
        "ma60": ma60,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "dd20": dd20,
        "heat_score": heat_score,
        "state": state,
        "amount": amount,
    }


def scan_commodity_chain(*, min_days: int = 80, top_k: int = 3) -> dict[str, Any]:
    name_map = _load_etf_name_map()
    cats: dict[str, list[EtfMetric]] = {c.key: [] for c in CHAIN_CATEGORIES}
    seen: set[str] = set()

    for sym, name in name_map.items():
        if sym in seen:
            continue
        cat = _categorize_name(name)
        if cat is None:
            continue
        df = _load_price_df(sym)
        met = _compute_metrics(df, min_days=int(min_days))
        if not met:
            continue
        seen.add(sym)
        cats[cat.key].append(
            EtfMetric(
                symbol=str(sym),
                name=str(name),
                close=_safe_float(met.get("close")),
                as_of=str(met.get("as_of") or "") or None,
                ma20=_safe_float(met.get("ma20")),
                ma60=_safe_float(met.get("ma60")),
                ret5=_safe_float(met.get("ret5")),
                ret20=_safe_float(met.get("ret20")),
                ret60=_safe_float(met.get("ret60")),
                dd20=_safe_float(met.get("dd20")),
                heat_score=_safe_float(met.get("heat_score")),
                state=str(met.get("state") or "数据不足"),
                amount=_safe_float(met.get("amount")),
            )
        )

    items_out: list[dict[str, Any]] = []
    for cat in CHAIN_CATEGORIES:
        items = sorted(
            cats.get(cat.key, []),
            key=lambda x: float(x.heat_score or -9e9),
            reverse=True,
        )
        top = items[: max(1, int(top_k))]
        score_avg = None
        if top:
            score_avg = sum(float(x.heat_score or 0.0) for x in top) / float(len(top))
        items_out.append(
            {
                "key": cat.key,
                "name": cat.name,
                "score_avg": score_avg,
                "count": len(items),
                "leaders": [
                    {
                        "symbol": it.symbol,
                        "name": it.name,
                        "as_of": it.as_of,
                        "close": it.close,
                        "ma20": it.ma20,
                        "ma60": it.ma60,
                        "ret5": it.ret5,
                        "ret20": it.ret20,
                        "ret60": it.ret60,
                        "dd20": it.dd20,
                        "heat_score": it.heat_score,
                        "state": it.state,
                        "amount": it.amount,
                    }
                    for it in top
                ],
            }
        )

    # 取所有 leader 中最新日期作为全局 as_of
    as_of = None
    dates = [x.get("as_of") for c in items_out for x in c.get("leaders", []) if x.get("as_of")]
    if dates:
        try:
            as_of = max(dates)
        except Exception:  # noqa: BLE001
            as_of = None

    return {
        "generated_at": datetime.now().isoformat(),
        "as_of": as_of,
        "min_days": int(min_days),
        "top_k": int(top_k),
        "categories": items_out,
    }


def render_chain_md(report: dict[str, Any]) -> str:
    lines = ["# 大宗商品链路热度扫描（ETF）", ""]
    as_of = report.get("as_of") or "未知"
    lines.append(f"- data_as_of: {as_of}")
    lines.append(f"- min_days: {report.get('min_days')}")
    lines.append("")

    for cat in CHAIN_CATEGORIES:
        c = next((x for x in report.get("categories", []) if x.get("key") == cat.key), None)
        if not c:
            continue
        lines.append(f"## {cat.name}")
        lines.append(f"- 样本数: {c.get('count')}, score_avg: {c.get('score_avg')}")
        leaders = c.get("leaders") or []
        if not leaders:
            lines.append("- 无有效标的（缓存不足或数据长度不够）。")
            lines.append("")
            continue
        for it in leaders:
            state = it.get("state") or "未知"
            heat = it.get("heat_score")
            heat_s = f"{heat:.2f}" if isinstance(heat, (int, float)) else "NA"
            ret20 = it.get("ret20")
            ret20_s = f"{ret20*100:.2f}%" if isinstance(ret20, (int, float)) else "NA"
            dd20 = it.get("dd20")
            dd20_s = f"{dd20*100:.2f}%" if isinstance(dd20, (int, float)) else "NA"
            ma20 = it.get("ma20")
            ma60 = it.get("ma60")
            lines.append(
                f"- {it.get('symbol')}（{it.get('name')}） as_of={it.get('as_of')} close={it.get('close')} "
                f"ma20={ma20} ma60={ma60} ret20={ret20_s} dd20={dd20_s} heat={heat_s} state={state}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
