# -*- coding: utf-8 -*-
"""
PositionSizing：成本敏感的仓位建议（先并行输出，别拿它当“自动下单”）。

硬约束（默认口径）：
- 最低佣金 5 元（每边）
- 最小交易额 2000 元（仓库既有规则）
- A 股/ETF 默认 T+1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


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


@dataclass(frozen=True, slots=True)
class PositionSizingInputs:
    symbol: str
    asset: str
    as_of: date
    ref_date: date
    opportunity_score: float
    bucket: Literal["reject", "probe", "plan"]
    source: str = "position_sizing"

    # misc
    confidence: float = 0.60
    max_position_pct: float = 0.30
    price: float | None = None

    min_trade_notional_yuan: int = 2000
    min_fee_yuan: float = 5.0

    t_plus_one: bool = True


def _lot_size(asset: str) -> int:
    a = str(asset or "").strip().lower()
    if a == "stock":
        return 100
    return 1  # etf/index/crypto 默认按 1 份


def compute_position_sizing(*, inputs: PositionSizingInputs) -> dict[str, Any]:
    score = _clip01(_sf(inputs.opportunity_score))
    conf = _clip01(_sf(inputs.confidence))
    max_pos = _clip01(_sf(inputs.max_position_pct))

    # bucket -> 建议仓位（先 KISS：别上凯利）
    suggest_pos = 0.0
    suggest_trade_notional = 0.0
    if inputs.bucket == "plan":
        suggest_pos = float(max_pos) * float(conf)
        suggest_trade_notional = 8000.0
    elif inputs.bucket == "probe":
        suggest_pos = float(max_pos) * 0.6 * float(conf)
        suggest_trade_notional = 5000.0
    else:
        suggest_pos = 0.0
        suggest_trade_notional = 0.0

    # 最低交易额门槛（不满足就别硬凑单）
    min_notional = float(inputs.min_trade_notional_yuan)
    if suggest_trade_notional > 0:
        suggest_trade_notional = float(max(suggest_trade_notional, min_notional))

    lot = int(_lot_size(inputs.asset))

    price = _sf(inputs.price)
    shares = None
    if price is not None and price > 0 and suggest_trade_notional > 0:
        raw = int(float(suggest_trade_notional) / float(price))
        if lot > 1:
            raw = (raw // lot) * lot
        shares = int(max(0, raw))

    # 佣金：研究口径先按“最低佣金”保守估计
    est_commission = float(inputs.min_fee_yuan) if suggest_trade_notional > 0 else 0.0

    # 滑点：默认 0（具体模型放 Phase1/回测里），但字段先占位保证 schema 稳定
    est_slippage = 0.0

    return {
        "schema": "llm_trading.position_sizing.v1",
        "symbol": str(inputs.symbol),
        "asset": str(inputs.asset),
        "as_of": str(inputs.as_of),
        "ref_date": str(inputs.ref_date),
        "source": str(inputs.source),
        "opportunity_score": float(score),
        "confidence": float(conf),
        "max_position_pct": float(max_pos),
        "suggest_position_pct": float(suggest_pos),
        "equity_yuan": None,
        "cash_yuan": None,
        "price": float(price) if price is not None else None,
        "lot_size": int(lot),
        "min_trade_notional_yuan": int(inputs.min_trade_notional_yuan),
        "suggest_trade_notional_yuan": float(suggest_trade_notional),
        "suggest_shares": int(shares) if shares is not None else None,
        "est_commission_yuan": float(est_commission),
        "est_slippage_yuan": float(est_slippage),
        "t_plus_one": bool(inputs.t_plus_one),
        "reason": f"bucket={inputs.bucket}, score={score:.2f}, confidence={conf:.2f}",
        "notes": None,
    }
