from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class TradeabilityConfig:
    """
    真实交易约束（研究用途，先做最小可用版）：
    - halt_vol_zero: volume/amount=0 视为停牌/无成交（常见于停牌/一字无量）
    - limit_up_pct / limit_down_pct: 用“前收->今开”粗估是否一字涨跌停
    """

    limit_up_pct: float = 0.0
    limit_down_pct: float = 0.0
    halt_vol_zero: bool = True


def tradeability_flags(
    *,
    open_price: float | None,
    high_price: float | None,
    low_price: float | None,
    prev_close: float | None,
    volume: float | None = None,
    amount: float | None = None,
    cfg: TradeabilityConfig | None = None,
) -> dict[str, Any]:
    """
    用日线 OHLCV 做一个“能不能在开盘成交”的粗估（研究用途）：
    - halted：volume/amount=0（通常就是停牌/无成交）
    - one_word：high==low（近似）且 open==high==low
    - locked_limit_up/down：一字板 + 达到涨跌停阈值

    注意：
    - 这是“最后一根已知日线”的状态，不是预测下一天。
    - ETF 不一定有严格涨跌停，这块默认不开（limit_up/down=0）。
    """
    cfg2 = cfg or TradeabilityConfig()
    lim_up = max(0.0, float(getattr(cfg2, "limit_up_pct", 0.0) or 0.0))
    lim_dn = max(0.0, float(getattr(cfg2, "limit_down_pct", 0.0) or 0.0))
    halt_zero = bool(getattr(cfg2, "halt_vol_zero", True))

    op = None if open_price is None else float(open_price)
    hp = None if high_price is None else float(high_price)
    lp = None if low_price is None else float(low_price)
    cp = None if prev_close is None else float(prev_close)

    halted = False
    if halt_zero:
        try:
            halted = bool((volume is not None and float(volume) == 0.0) or (amount is not None and float(amount) == 0.0))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            halted = False

    pct_open = None
    try:
        if op is not None and cp is not None and float(cp) > 0:
            pct_open = float(op) / float(cp) - 1.0
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pct_open = None

    one_word = False
    try:
        if op is not None and hp is not None and lp is not None:
            op2 = float(op)
            hp2 = float(hp)
            lp2 = float(lp)
            tol = max(1e-9, abs(hp2) * 1e-6)
            one_word = bool(abs(hp2 - lp2) <= tol and abs(op2 - hp2) <= tol and abs(op2 - lp2) <= tol)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        one_word = False

    locked_up = bool(float(lim_up) > 0 and one_word and (pct_open is not None) and float(pct_open) >= float(lim_up) - 1e-6)
    locked_dn = bool(float(lim_dn) > 0 and one_word and (pct_open is not None) and float(pct_open) <= -float(lim_dn) + 1e-6)

    return {
        "halted": bool(halted),
        "one_word": bool(one_word),
        "pct_open": (float(pct_open) if pct_open is not None and math.isfinite(float(pct_open)) else None),
        "locked_limit_up": bool(locked_up),
        "locked_limit_down": bool(locked_dn),
        "cfg": {"limit_up_pct": float(lim_up), "limit_down_pct": float(lim_dn), "halt_vol_zero": bool(halt_zero)},
    }

