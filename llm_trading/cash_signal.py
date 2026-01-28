# -*- coding: utf-8 -*-
"""
CashSignal：把“环境不对就空仓”工程化（账户级风险开关）。

注意：
- 这是“现金比例建议”，不是买卖按钮。
- 默认基于大盘 regime + 波动状态；可选叠加 ERP/HSGT（风险温度计）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal
from pathlib import Path


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
class CashSignalInputs:
    as_of: date
    ref_date: date
    scope: str = "portfolio"
    context_index_symbol: str = "sh000300+sh000905"  # 展示用；实现侧会把 '+' 当成 ',' 处理
    source: str = "cash_signal"

    expected_duration_days: int = 10


def compute_cash_signal(
    *,
    inputs: CashSignalInputs,
    tushare_factors: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    cache_dir2 = cache_dir or (Path("data") / "cache" / "index")
    idx_raw = str(inputs.context_index_symbol or "").strip()
    idx_spec = idx_raw.replace("+", ",")

    label: str | None = None
    vol_20d = None
    panic = False
    margin_score01 = None
    margin_overheat = False
    margin_deleveraging = False
    regime_error = None
    try:
        from .market_regime import compute_market_regime_payload

        payload, regime_error, idx_eff = compute_market_regime_payload(
            idx_spec,
            cache_dir=cache_dir2,
            ttl_hours=6.0,
            ensemble_mode="risk_first",
            canary_downgrade=True,
        )
        if payload:
            label = str(payload.get("label") or "unknown").strip().lower() or "unknown"
            vol_20d = _sf(payload.get("vol_20d"))
            panic = bool(payload.get("panic"))
            # A) 两融杠杆温度计（来自 market_regime_payload 的附加字段；可降级）
            try:
                m = payload.get("market_margin") if isinstance(payload, dict) else None
                if isinstance(m, dict) and bool(m.get("ok")):
                    margin_score01 = _sf(m.get("score01"))
                    margin_overheat = bool(m.get("overheat"))
                    margin_deleveraging = bool(m.get("deleveraging"))
            except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                margin_score01 = None
                margin_overheat = False
                margin_deleveraging = False
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        OverflowError,
        AttributeError,
    ) as exc:  # noqa: BLE001
        regime_error = str(exc)

    # vol_state：给个粗粒度（研究用途）
    vol_state = "unknown"
    if vol_20d is not None:
        if vol_20d >= 0.025:
            vol_state = "high"
        elif vol_20d <= 0.012:
            vol_state = "low"
        else:
            vol_state = "normal"

    # TuShare 风险温度计（可选）
    erp_proxy = None
    north_score01 = None
    south_score01 = None
    try:
        tf = tushare_factors or {}
        erp = tf.get("erp") if isinstance(tf, dict) else None
        hsgt = tf.get("hsgt") if isinstance(tf, dict) else None

        # ERP：erp.erp（小数），我们就原样输出（研究用途）
        if isinstance(erp, dict) and bool(erp.get("ok")):
            erp_proxy = _sf(erp.get("erp"))

        if isinstance(hsgt, dict) and bool(hsgt.get("ok")):
            north = hsgt.get("north") if isinstance(hsgt.get("north"), dict) else None
            south = hsgt.get("south") if isinstance(hsgt.get("south"), dict) else None
            if isinstance(north, dict):
                north_score01 = _sf(north.get("score01"))
            if isinstance(south, dict):
                south_score01 = _sf(south.get("score01"))
    except (AttributeError):  # noqa: BLE001
        erp_proxy = None
        north_score01 = None
        south_score01 = None

    # risk_mode / cash_ratio：先按 regime 定，再按 panic/波动做微调
    risk_mode: Literal["risk_off", "neutral", "risk_on"] = "neutral"
    if label == "bear":
        risk_mode = "risk_off"
    elif label == "bull":
        risk_mode = "risk_on"
    else:
        risk_mode = "neutral"

    cash_ratio = 0.5
    if risk_mode == "risk_off":
        cash_ratio = 0.8
    elif risk_mode == "risk_on":
        cash_ratio = 0.2
    else:
        cash_ratio = 0.5

    if panic:
        cash_ratio = max(cash_ratio, 0.9)
        risk_mode = "risk_off"
    if vol_state == "high":
        cash_ratio = max(cash_ratio, 0.7)

    # A) 杠杆过热/去杠杆：只做风险加权（小散保命优先），别当买卖按钮。
    if margin_deleveraging:
        cash_ratio = max(cash_ratio, 0.75)
        risk_mode = "risk_off"
    elif margin_overheat:
        cash_ratio = max(cash_ratio, 0.60)
        if risk_mode == "risk_on":
            risk_mode = "neutral"

    cash_ratio = float(_clip01(cash_ratio))

    should_stay_cash = bool(cash_ratio >= 0.70)

    # reason：别写一堆口播，写可验证字段
    parts: list[str] = []
    if label:
        parts.append(f"regime={label}")
    if vol_state != "unknown":
        parts.append(f"vol={vol_state}")
    if panic:
        parts.append("panic=true")
    if erp_proxy is not None:
        parts.append("erp_proxy=ok")
    if north_score01 is not None or south_score01 is not None:
        parts.append("hsgt=ok")
    if margin_score01 is not None:
        parts.append("margin=ok")
    if margin_overheat:
        parts.append("margin_overheat=true")
    if margin_deleveraging:
        parts.append("margin_deleveraging=true")
    if regime_error:
        parts.append(f"regime_error={regime_error}")
    reason = " + ".join(parts) if parts else "no_signal"

    return {
        "schema": "llm_trading.cash_signal.v1",
        "as_of": str(inputs.as_of),
        "ref_date": str(inputs.ref_date),
        "source": str(inputs.source),
        "scope": str(inputs.scope),
        "context_index_symbol": str(inputs.context_index_symbol),
        "should_stay_cash": bool(should_stay_cash),
        "cash_ratio": float(cash_ratio),
        "risk_mode": str(risk_mode),
        "expected_duration_days": int(inputs.expected_duration_days),
        "evidence": {
            "market_regime": str(label or "unknown"),
            "vol_state": str(vol_state),
            "erp_proxy": erp_proxy,
            "north_score01": north_score01,
            "south_score01": south_score01,
            "margin_score01": margin_score01,
            "margin_overheat": bool(margin_overheat),
            "margin_deleveraging": bool(margin_deleveraging),
        },
        "reason": str(reason),
        "notes": None,
    }
