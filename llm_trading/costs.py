from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CostModel:
    capital_yuan: float
    roundtrip_cost_yuan: float
    buy_cost: float
    sell_cost: float


def cost_model_from_roundtrip(*, capital_yuan: float, roundtrip_cost_yuan: float) -> CostModel:
    """
    把“来回固定磨损（元）”换算成回测用的比例成本（buy_cost/sell_cost）。

    约定：
    - roundtrip_cost_yuan = 买入 + 卖出 的总磨损
    - 简单起见：均分为买入/卖出各一半
    """
    cap = float(capital_yuan)
    rt = float(roundtrip_cost_yuan)
    if cap <= 0 or rt <= 0:
        return CostModel(capital_yuan=cap, roundtrip_cost_yuan=rt, buy_cost=0.0, sell_cost=0.0)
    pct = rt / cap
    half = pct / 2.0
    return CostModel(capital_yuan=cap, roundtrip_cost_yuan=rt, buy_cost=float(half), sell_cost=float(half))


@dataclass(frozen=True, slots=True)
class TradeCost:
    """
    交易成本/约束统一口径（研究用途，先把闭环做对）：
    - buy_cost/sell_cost：比例成本（例如 0.001=0.10%），用于佣金/滑点/冲击等“随成交额放大”的部分
    - buy_fee_yuan/sell_fee_yuan：固定附加（每次买/卖一笔）
    - buy_fee_min_yuan/sell_fee_min_yuan：最低佣金（每次买/卖一笔）
    """

    buy_cost: float
    sell_cost: float
    buy_fee_yuan: float
    sell_fee_yuan: float
    buy_fee_min_yuan: float
    sell_fee_min_yuan: float


def trade_cost_from_params(*, roundtrip_cost_yuan: float = 0.0, min_fee_yuan: float = 0.0, buy_cost: float = 0.0, sell_cost: float = 0.0) -> TradeCost:
    """
    从 CLI 常见参数拼出 TradeCost：
    - roundtrip_cost_yuan：来回固定磨损（元）=> 每边均分
    - min_fee_yuan：最低佣金（每边）
    """
    fixed_half = max(0.0, float(roundtrip_cost_yuan or 0.0)) / 2.0
    fee_min = max(0.0, float(min_fee_yuan or 0.0))
    return TradeCost(
        buy_cost=max(0.0, float(buy_cost or 0.0)),
        sell_cost=max(0.0, float(sell_cost or 0.0)),
        buy_fee_yuan=float(fixed_half),
        sell_fee_yuan=float(fixed_half),
        buy_fee_min_yuan=float(fee_min),
        sell_fee_min_yuan=float(fee_min),
    )


def effective_fee_yuan(*, notional_yuan: float, cost_rate: float, min_fee_yuan: float, fixed_fee_yuan: float) -> float:
    """
    给定成交额与成本模型，估算“这一边（买/卖）的费用（元）”。
    - fee = max(notional*cost_rate, min_fee) + fixed_fee
    """
    try:
        n = float(notional_yuan)
        r = float(cost_rate)
        m = float(min_fee_yuan)
        f = float(fixed_fee_yuan)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return 0.0
    if (not math.isfinite(n)) or n <= 0:
        return 0.0
    if not math.isfinite(r):
        r = 0.0
    if not math.isfinite(m):
        m = 0.0
    if not math.isfinite(f):
        f = 0.0
    pct = n * max(0.0, r)
    return float(max(pct, max(0.0, m)) + max(0.0, f))


def effective_rate_for_notional(*, notional_yuan: float, cost_rate: float, min_fee_yuan: float, fixed_fee_yuan: float) -> float:
    """
    把“这一边的费用（元）”摊回成“比例成本”（用于 forward/backtest 等比例口径）。
    """
    try:
        n = float(notional_yuan)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return 0.0
    if (not math.isfinite(n)) or n <= 0:
        return 0.0
    fee = effective_fee_yuan(notional_yuan=float(n), cost_rate=float(cost_rate), min_fee_yuan=float(min_fee_yuan), fixed_fee_yuan=float(fixed_fee_yuan))
    try:
        out = float(fee) / float(n)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return float(max(0.0, out))


def cash_buy(*, shares: int, price: float, cost: TradeCost) -> tuple[float, float]:
    """
    买入：返回 (现金流出, fee_yuan)。
    fee_yuan = max(notional*buy_cost, buy_fee_min_yuan) + buy_fee_yuan
    """
    v = float(shares) * float(price)
    pct = v * max(0.0, float(cost.buy_cost))
    fee_min = max(0.0, float(getattr(cost, "buy_fee_min_yuan", 0.0)))
    fee = max(float(pct), float(fee_min)) + max(0.0, float(cost.buy_fee_yuan))
    return float(v + fee), float(fee)


def cash_sell(*, shares: int, price: float, cost: TradeCost) -> tuple[float, float]:
    """
    卖出：返回 (现金流入, fee_yuan)。
    fee_yuan = max(notional*sell_cost, sell_fee_min_yuan) + sell_fee_yuan
    """
    v = float(shares) * float(price)
    pct = v * max(0.0, float(cost.sell_cost))
    fee_min = max(0.0, float(getattr(cost, "sell_fee_min_yuan", 0.0)))
    fee = max(float(pct), float(fee_min)) + max(0.0, float(cost.sell_fee_yuan))
    return float(v - fee), float(fee)


def calc_shares_for_capital(*, capital_yuan: float, price: float, cost: TradeCost, lot_size: int = 100) -> int:
    """
    给定预算，按成本模型反推“最多能买多少股/份”（向下取整到一手）。
    """
    cap = float(capital_yuan)
    px = float(price)
    lot = max(1, int(lot_size))
    if cap <= 0 or px <= 0:
        return 0

    fee_fixed = max(0.0, float(cost.buy_fee_yuan))
    fee_min = max(0.0, float(cost.buy_fee_min_yuan))
    if cap <= fee_fixed + fee_min:
        return 0

    rate = max(0.0, float(cost.buy_cost))

    # 先给一个“足够接近”的候选，再用 cash_buy 校验（不够就按手数递减）
    cand = 0
    if rate > 0:
        unit = px * (1.0 + rate)
        if unit > 0 and cap > fee_fixed:
            cand = int((cap - fee_fixed) // unit)
    if cand <= 0:
        cand = int((cap - fee_fixed - fee_min) // px) if cap > (fee_fixed + fee_min) else 0

    sh = int((cand // lot) * lot)
    if sh <= 0:
        return 0

    for _ in range(50):
        cash_out, _ = cash_buy(shares=sh, price=px, cost=cost)
        if cash_out <= cap + 1e-9:
            return int(sh)
        sh = int(sh - lot)
        sh = int((sh // lot) * lot)
        if sh <= 0:
            return 0
    return 0


def min_notional_for_min_fee(*, cost_rate: float, min_fee_yuan: float) -> float | None:
    """
    触发“最低佣金”的门槛：min_fee / cost_rate。
    cost_rate<=0 或 min_fee<=0 则返回 None。
    """
    try:
        r = float(cost_rate)
        m = float(min_fee_yuan)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if (not math.isfinite(r)) or (not math.isfinite(m)) or r <= 0 or m <= 0:
        return None
    return float(m / r)


def estimate_slippage_bps(
    *,
    mode: str,
    amount_avg20_yuan: float | None = None,
    atr_pct: float | None = None,
    bps: float = 0.0,
    ref_amount_yuan: float = 1e8,
    min_bps: float = 0.0,
    max_bps: float = 30.0,
    unknown_bps: float = 10.0,
    vol_mult: float = 0.0,
) -> float:
    """
    粗糙但可控的滑点/冲击成本近似（单位：bps，1bp=0.01%）。

    - mode=none：不加滑点
    - mode=fixed：固定 bps（每边）
    - mode=liquidity：按近20日均成交额估算，成交额越小 -> bps 越大

    可选：atr_pct（例如 ATR/close）作为“波动放大器”：
    bps *= (1 + vol_mult * clamp(atr_pct/0.02, 0..3))
    """
    m = str(mode or "none").strip().lower()
    if m in {"0", "off", "false", "none", ""}:
        return 0.0

    min_bps2 = max(0.0, float(min_bps))
    max_bps2 = max(min_bps2, float(max_bps))

    if m in {"fixed", "const", "constant"}:
        x = max(0.0, float(bps))
    elif m in {"liquidity", "liq", "amount"}:
        a = None
        try:
            a = None if amount_avg20_yuan is None else float(amount_avg20_yuan)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            a = None

        if a is None or (not math.isfinite(a)) or a <= 0:
            x = max(0.0, float(unknown_bps))
        else:
            base = max(0.0, float(bps))
            ref = max(1.0, float(ref_amount_yuan))
            # 经验近似：bps ~ base * sqrt(ref/amount)
            try:
                x = base * math.sqrt(ref / a)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                x = base
    else:
        # 未知 mode：当 none 处理（别悄悄给你加成本）
        return 0.0

    # 波动放大器（可选）
    vm = float(vol_mult or 0.0)
    if vm > 0 and atr_pct is not None:
        try:
            ap = float(atr_pct)
            if math.isfinite(ap) and ap > 0:
                # 2% ATR 视为 1x，最多放大到 3x
                k = ap / 0.02
                if k < 0:
                    k = 0.0
                if k > 3:
                    k = 3.0
                x = x * (1.0 + vm * k)
        except (AttributeError):  # noqa: BLE001
            pass

    if not math.isfinite(x):
        x = 0.0
    if x < min_bps2:
        x = min_bps2
    if x > max_bps2:
        x = max_bps2
    return float(x)


def bps_to_rate(bps: float) -> float:
    """
    bps -> 比例（1bp=0.0001）。
    """
    try:
        x = float(bps) / 10000.0
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return float(max(0.0, x))
