# -*- coding: utf-8 -*-
"""
OpportunityScore：把“看起来不错”落成可解释的 0~1 分数（先并行输出，不当买卖按钮）。

设计约束：
- KISS：先用现有技术因子 + 少量 proxy（trap/fund_flow）拼一个可解释评分。
- 可复现/可 SQL：输出 schema 固定，字段 snake_case。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd


def _sf(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _clip01(x: float | None) -> float:
    if x is None:
        return 0.0
    return float(max(0.0, min(1.0, float(x))))


def _wavg(xs: list[tuple[float | None, float | None]], *, default: float = 0.5) -> float:
    """
    加权均值：[(value, weight)]；weight 用 confidence。
    """
    num = 0.0
    den = 0.0
    for v, w in xs:
        vv = _sf(v)
        ww = _sf(w)
        if vv is None:
            continue
        if ww is None or ww <= 0:
            ww = 1.0
        num += float(vv) * float(ww)
        den += float(ww)
    if den <= 0:
        return float(default)
    return float(num / den)


@dataclass(frozen=True, slots=True)
class OpportunityScoreInputs:
    symbol: str
    asset: str
    as_of: date
    ref_date: date
    min_score: float = 0.70
    t_plus_one: bool = True

    # optional context
    trap_risk: float | None = None  # 0~1，越高越危险
    fund_flow: float | None = None  # 0~1，可选（TuShare microstructure/hsgt 等）

    expected_holding_days: int = 10


def compute_opportunity_score(
    *,
    df: pd.DataFrame,
    inputs: OpportunityScoreInputs,
    key_level_name: str | None = None,
    key_level_value: float | None = None,
) -> dict[str, Any]:
    # 确保因子已注册（import package 会触发 register_factor 装饰器）
    from . import factors as _  # noqa: F401
    from .factors.base import FACTOR_REGISTRY

    df_local = df
    need = {"close", "high", "low", "volume"}
    if df_local is None or getattr(df_local, "empty", True) or (not need.issubset(set(df_local.columns))):
        return {
            "schema": "llm_trading.opportunity_score.v1",
            "symbol": str(inputs.symbol),
            "asset": str(inputs.asset),
            "as_of": str(inputs.as_of),
            "ref_date": str(inputs.ref_date),
            "source": "opportunity_score",
            "total_score": 0.0,
            "min_score": float(inputs.min_score),
            "verdict": "not_tradeable",
            "bucket": "reject",
            "components": {
                "trend": None,
                "regime": None,
                "risk_reward": None,
                "liquidity": None,
                "trap_risk": _sf(inputs.trap_risk),
                "fund_flow": _sf(inputs.fund_flow),
            },
            "key_level": {"name": key_level_name or "unknown", "value": _sf(key_level_value)},
            "invalidation": {"rule": "unknown", "level": _sf(key_level_value), "note": "data_invalid"},
            "expected_holding_days": int(inputs.expected_holding_days),
            "t_plus_one": bool(inputs.t_plus_one),
            "notes": "data_invalid",
        }

    # 计算一组“够用就行”的技术因子：趋势/环境/结构/流动性
    names = [
        "ma_cross",
        "macd",
        "ichimoku",
        "adx",
        "regime",
        "pullback",
        "atr",
        "volume_ratio",
    ]
    rs = FACTOR_REGISTRY.compute_all(df_local, names)

    # trend：方向与一致性（偏多越高）
    trend = _wavg(
        [
            (rs.get("ma_cross").score if rs.get("ma_cross") else None, rs.get("ma_cross").confidence if rs.get("ma_cross") else None),
            (rs.get("macd").score if rs.get("macd") else None, rs.get("macd").confidence if rs.get("macd") else None),
            (rs.get("ichimoku").score if rs.get("ichimoku") else None, rs.get("ichimoku").confidence if rs.get("ichimoku") else None),
            (rs.get("adx").score if rs.get("adx") else None, rs.get("adx").confidence if rs.get("adx") else None),
        ],
        default=0.5,
    )

    # regime：环境是否允许（bull>neutral>bear）
    regime = _sf((rs.get("regime").score if rs.get("regime") else None))
    if regime is None:
        regime = 0.5

    # risk_reward：结构清晰 + 风险不炸（pullback + atr 低波动更好）
    risk_reward = _wavg(
        [
            (rs.get("pullback").score if rs.get("pullback") else None, rs.get("pullback").confidence if rs.get("pullback") else None),
            (rs.get("atr").score if rs.get("atr") else None, rs.get("atr").confidence if rs.get("atr") else None),
        ],
        default=0.5,
    )

    # liquidity：先用量比 proxy（更复杂的“成交额门槛”留给 scan-* 的 min_amount_avg20）
    liquidity = _sf((rs.get("volume_ratio").score if rs.get("volume_ratio") else None))
    if liquidity is None:
        liquidity = 0.5

    trap_risk = _clip01(_sf(inputs.trap_risk))
    fund_flow = _sf(inputs.fund_flow)
    asset2 = str(inputs.asset or "").strip().lower()

    components = {
        "trend": float(trend),
        "regime": float(regime),
        "risk_reward": float(risk_reward),
        "liquidity": float(liquidity),
        "trap_risk": float(trap_risk) if inputs.trap_risk is not None else None,
        "fund_flow": float(fund_flow) if fund_flow is not None else None,
    }

    # total_score：正向组件加权均值 - trap_risk 罚分（风险项不当“反向加分”）
    pos_weights = {
        "trend": 0.32,
        "regime": 0.18,
        "risk_reward": 0.25,
        "liquidity": 0.25,
        # ETF: fund_flow 可做“解释/加权”；stock: 资金更像风控项（见下方 penalty），不当作直接加分项。
        "fund_flow": (0.10 if (fund_flow is not None and asset2 != "stock") else 0.0),
    }
    num = 0.0
    den = 0.0
    for k, w in pos_weights.items():
        if w <= 0:
            continue
        v = _sf(components.get(k))
        if v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    base = float(num / den) if den > 0 else 0.5

    # trap penalty：默认 0.15（留够空间让“趋势/结构”说话）
    # B) fund_flow penalty：stock 更像“主力是否在撤”的风控项；ETF 不强绑（罚分更轻/可忽略）。
    fund_flow_for_penalty = float(fund_flow) if fund_flow is not None else 0.5
    fund_flow_penalty_w = 0.12 if asset2 == "stock" else 0.03
    fund_flow_penalty = float(fund_flow_penalty_w) * max(0.0, 0.5 - float(fund_flow_for_penalty))
    components["fund_flow_penalty"] = float(fund_flow_penalty) if fund_flow is not None else None

    total = _clip01(base - 0.15 * float(trap_risk) - float(fund_flow_penalty))

    min_score = float(inputs.min_score)
    verdict: Literal["tradeable", "not_tradeable"] = "tradeable" if total >= min_score else "not_tradeable"

    bucket: Literal["reject", "probe", "plan"] = "reject"
    if total >= 0.80:
        bucket = "plan"
    elif total >= min_score:
        bucket = "probe"

    # key_level：默认用 MA50（如果外部没传，就尽量从 df/因子里推一个）
    kl_name = str(key_level_name or "").strip() or None
    kl_value = _sf(key_level_value)
    if kl_name is None or kl_value is None:
        # 优先 ma50（如果已经在 df 里）
        kl = None
        if "ma50" in df_local.columns:
            kl = _sf(df_local.iloc[-1].get("ma50"))
            if kl is not None:
                kl_name = "ma50"
                kl_value = kl
        if (kl_name is None or kl_value is None) and "ma20" in df_local.columns:
            kl = _sf(df_local.iloc[-1].get("ma20"))
            if kl is not None:
                kl_name = "ma20"
                kl_value = kl
        if kl_name is None or kl_value is None:
            kl_name = "close"
            kl_value = _sf(df_local.iloc[-1].get("close"))

    return {
        "schema": "llm_trading.opportunity_score.v1",
        "symbol": str(inputs.symbol),
        "asset": str(inputs.asset),
        "as_of": str(inputs.as_of),
        "ref_date": str(inputs.ref_date),
        "source": "opportunity_score",
        "total_score": float(total),
        "min_score": float(min_score),
        "verdict": str(verdict),
        "bucket": str(bucket),
        "components": components,
        "key_level": {"name": str(kl_name), "value": float(kl_value) if kl_value is not None else None},
        "invalidation": {
            "rule": "close_below_level",
            "level": float(kl_value) if kl_value is not None else None,
            "note": "T+1 执行" if inputs.t_plus_one else None,
        },
        "expected_holding_days": int(inputs.expected_holding_days),
        "t_plus_one": bool(inputs.t_plus_one),
        "notes": None,
    }
