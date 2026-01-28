from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def infer_theme(name: str) -> str:
    """
    用名称做一个非常粗糙的“主题”归类（只用于分散/暴露统计的启发式）。
    追求的是“不把两只一模一样的东西当成分散”，不是学术完美分类。
    """
    n = str(name or "").replace(" ", "")
    if not n:
        return "unknown"

    rules = [
        ("机器人", "robotics"),
        ("人工智能", "ai"),
        ("AI", "ai"),
        ("半导体", "semiconductor"),
        ("芯片", "semiconductor"),
        ("有色", "metals"),
        ("黄金", "gold"),
        ("煤炭", "coal"),
        ("军工", "defense"),
        ("医药", "healthcare"),
        ("医疗", "healthcare"),
        ("消费", "consumer"),
        ("白酒", "consumer"),
        ("银行", "banks"),
        ("证券", "brokers"),
        ("红利", "dividend"),
        ("低波", "low_vol"),
        ("纳指", "nasdaq"),
        ("标普", "sp500"),
        ("恒生", "hang_seng"),
        ("中概", "china_overseas"),
        ("科创", "sci_tech"),
        ("创业板", "chi_next"),
        ("新能源", "new_energy"),
        ("光伏", "new_energy"),
        ("电池", "new_energy"),
    ]
    for kw, theme in rules:
        if kw in n:
            return theme

    # “宽基/大盘/指数类”粗归为 broad
    broad_kw = ["上证", "沪深", "中证", "深证", "A50", "50", "300", "500", "1000", "180", "增强", "价值", "成长"]
    if any(k in n for k in broad_kw):
        return "broad"

    return "other"


def load_weekly_returns_from_cache(
    *,
    asset: str,
    symbol: str,
    adjust: str,
    cache_dir: Path,
    window_weeks: int,
) -> list[float] | None:
    """
    从本地缓存读取日线 → 转周线 → 计算周收益序列（pct_change）。

    返回：周收益列表（长度约 window_weeks）；失败返回 None。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    a = str(asset or "").strip().lower()
    sym = str(symbol or "").strip()
    adj = str(adjust or "").strip()
    if not a or not sym:
        return None

    key = f"{a}_{sym}_{adj}.csv".replace("/", "_").replace("\\", "_")
    path = cache_dir / key
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True) or "date" not in df.columns or "close" not in df.columns:
        return None

    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        return None
    if df.empty:
        return None

    try:
        from .resample import resample_to_weekly

        dfw = resample_to_weekly(df)
        if dfw is None or getattr(dfw, "empty", True):
            return None
        close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
        rets = close.pct_change().replace([float("inf"), float("-inf")], float("nan")).dropna()
        if rets.empty:
            return None
        w = max(0, int(window_weeks))
        if w > 0:
            rets = rets.tail(w)
        xs = [float(x) for x in rets.to_list() if x is not None and math.isfinite(float(x))]
        if len(xs) < 8:
            return None
        return xs
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None


def corr_abs_tail(a: list[float], b: list[float], *, min_overlap: int) -> float | None:
    """
    简单相关系数（用“尾部重叠”近似对齐；用于粗粒度去重/分散评估）。
    """
    n = min(len(a), len(b))
    if n < int(min_overlap):
        return None
    xa = a[-n:]
    xb = b[-n:]
    if n <= 1:
        return None
    ma = sum(xa) / n
    mb = sum(xb) / n
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        va = float(xa[i]) - ma
        vb = float(xb[i]) - mb
        num += va * vb
        da += va * va
        db += vb * vb
    if da <= 0 or db <= 0:
        return None
    c = num / math.sqrt(da * db)
    if not math.isfinite(c):
        return None
    if c > 1:
        c = 1.0
    if c < -1:
        c = -1.0
    return float(abs(c))


def build_portfolio_summary(
    *,
    holdings: list[dict[str, Any]],
    cash_yuan: float | None = None,
    cache_base_dir: Path | None = None,
    corr_window_weeks: int = 104,
    corr_min_overlap_weeks: int = 26,
    warn_single_position_weight_pct: float = 0.40,
    warn_theme_weight_pct: float = 0.60,
    stock_adjust: str = "qfq",
) -> dict[str, Any]:
    """
    基于 analyze_holdings 的输出项，做“组合层”的汇总（研究用途）。
    - 不做交易建议，只输出：暴露/集中度/相关性/到止损线的风险
    """
    items = [it for it in (holdings or []) if isinstance(it, dict) and bool(it.get("ok"))]
    if not items:
        return {
            "cash_yuan": cash_yuan,
            "positions_market_value_yuan": 0.0,
            "positions_cost_yuan": 0.0,
            "positions_pnl_net_yuan": 0.0,
            "positions_pnl_net_pct": None,
            "equity_yuan": cash_yuan if cash_yuan is not None else None,
            "exposure_pct": None,
            "weights": [],
            "themes": [],
            "risk_to_stop_yuan": 0.0,
            "risk_to_stop_pct_equity": None,
            "corr_pairs": [],
            "max_abs_corr": None,
            "warnings": ["持仓为空"],
        }

    def fnum(x) -> float | None:
        try:
            v = None if x is None else float(x)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return None
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        return float(v)

    mv_total = 0.0
    cost_total = 0.0
    pnl_total = 0.0
    risk_to_stop_total = 0.0

    # weights/theme buckets
    weights: list[dict[str, Any]] = []
    theme_mv: dict[str, float] = {}
    theme_syms: dict[str, list[str]] = {}

    for it in items:
        sym = str(it.get("symbol") or "").strip()
        name = str(it.get("name") or "").strip()
        asset = str(it.get("asset") or "").strip().lower()
        shares = int(it.get("shares") or 0)
        close = fnum(it.get("close"))
        mv = fnum(it.get("market_value"))
        if mv is None and close is not None and shares > 0:
            mv = float(close) * float(shares)
        if mv is None:
            mv = 0.0
        mv_total += float(mv)

        cost = fnum(it.get("cost"))
        if cost is not None and shares > 0:
            cost_total += float(cost) * float(shares)

        pnl = fnum(it.get("pnl_net"))
        if pnl is not None:
            pnl_total += float(pnl)

        st = (it.get("stops") or {}) if isinstance(it.get("stops"), dict) else {}
        eff = fnum(st.get("effective_stop"))
        if close is not None and eff is not None and shares > 0 and eff < close:
            risk_to_stop_total += (float(close) - float(eff)) * float(shares)

        theme = infer_theme(name)
        theme_mv[theme] = float(theme_mv.get(theme, 0.0)) + float(mv)
        theme_syms.setdefault(theme, [])
        if sym:
            theme_syms[theme].append(sym)

        weights.append({"asset": asset, "symbol": sym, "name": name, "market_value_yuan": float(mv)})

    pnl_pct = (pnl_total / cost_total) if cost_total > 0 else None

    equity = None
    exposure_pct = None
    cash_pct = None
    if cash_yuan is not None:
        equity = float(cash_yuan) + float(mv_total)
        if equity > 0:
            exposure_pct = float(mv_total) / float(equity)
            cash_pct = float(cash_yuan) / float(equity)

    denom = float(equity) if equity and equity > 0 else float(mv_total)
    if denom <= 0:
        denom = 1.0
    for w in weights:
        w["weight_pct"] = float(w.get("market_value_yuan") or 0.0) / denom
    weights = sorted(weights, key=lambda x: float(x.get("weight_pct") or 0.0), reverse=True)

    themes: list[dict[str, Any]] = []
    for th, mv in theme_mv.items():
        themes.append(
            {
                "theme": th,
                "market_value_yuan": float(mv),
                "weight_pct": float(mv) / denom,
                "symbols": sorted(set(theme_syms.get(th, []))),
            }
        )
    themes = sorted(themes, key=lambda x: float(x.get("weight_pct") or 0.0), reverse=True)

    # 相关性：只做粗粒度“查重/集中度”提示，避免两只高度同涨同跌的东西当分散
    cache_base = cache_base_dir or (Path("data") / "cache")
    rets_by_sym: dict[str, list[float]] = {}
    corr_pairs: list[dict[str, Any]] = []
    max_corr = None
    for i in range(len(items)):
        a = items[i]
        sym_a = str(a.get("symbol") or "").strip()
        asset_a = str(a.get("asset") or "").strip().lower()
        if not sym_a:
            continue
        if sym_a not in rets_by_sym:
            adj_a = "qfq" if asset_a == "etf" else str(stock_adjust or "qfq").strip() or "qfq"
            cache_dir = cache_base / asset_a
            rets_by_sym[sym_a] = load_weekly_returns_from_cache(
                asset=asset_a, symbol=sym_a, adjust=adj_a, cache_dir=cache_dir, window_weeks=int(corr_window_weeks)
            ) or []

        for j in range(i + 1, len(items)):
            b = items[j]
            sym_b = str(b.get("symbol") or "").strip()
            asset_b = str(b.get("asset") or "").strip().lower()
            if not sym_b:
                continue
            if sym_b not in rets_by_sym:
                adj_b = "qfq" if asset_b == "etf" else str(stock_adjust or "qfq").strip() or "qfq"
                cache_dir = cache_base / asset_b
                rets_by_sym[sym_b] = load_weekly_returns_from_cache(
                    asset=asset_b, symbol=sym_b, adjust=adj_b, cache_dir=cache_dir, window_weeks=int(corr_window_weeks)
                ) or []

            ra = rets_by_sym.get(sym_a) or []
            rb = rets_by_sym.get(sym_b) or []
            if not ra or not rb:
                continue
            c = corr_abs_tail(ra, rb, min_overlap=int(corr_min_overlap_weeks))
            if c is None:
                continue
            corr_pairs.append({"a": sym_a, "b": sym_b, "corr_abs": float(c)})
            if max_corr is None or float(c) > float(max_corr):
                max_corr = float(c)

    corr_pairs = sorted(corr_pairs, key=lambda x: float(x.get("corr_abs") or 0.0), reverse=True)[:50]

    warnings: list[str] = []
    if cash_yuan is None:
        warnings.append("cash.amount 为空：无法计算账户权益/仓位比例（请在 data/user_holdings.json 回填现金）")

    if weights:
        top = float(weights[0].get("weight_pct") or 0.0)
        if top >= float(warn_single_position_weight_pct):
            warnings.append(f"单一持仓集中度偏高：top1≈{top:.0%}（建议分散/减小单票暴露）")

    if themes:
        top_th = themes[0]
        w_th = float(top_th.get("weight_pct") or 0.0)
        if w_th >= float(warn_theme_weight_pct):
            warnings.append(f"主题集中度偏高：{top_th.get('theme')}≈{w_th:.0%}（注意同涨同跌）")

    risk_pct_equity = (risk_to_stop_total / equity) if (equity is not None and equity > 0) else None

    return {
        "cash_yuan": float(cash_yuan) if cash_yuan is not None else None,
        "positions_market_value_yuan": float(mv_total),
        "positions_cost_yuan": float(cost_total),
        "positions_pnl_net_yuan": float(pnl_total),
        "positions_pnl_net_pct": float(pnl_pct) if pnl_pct is not None else None,
        "equity_yuan": float(equity) if equity is not None else None,
        "exposure_pct": float(exposure_pct) if exposure_pct is not None else None,
        "cash_pct": float(cash_pct) if cash_pct is not None else None,
        "weights": weights,
        "themes": themes,
        "risk_to_stop_yuan": float(risk_to_stop_total),
        "risk_to_stop_pct_equity": float(risk_pct_equity) if risk_pct_equity is not None else None,
        "corr_pairs": corr_pairs,
        "max_abs_corr": float(max_corr) if max_corr is not None else None,
        "warnings": warnings,
    }

