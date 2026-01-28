from __future__ import annotations

"""
orders_next_open 组装工具（给 run/rebalance/对账复用）。

设计目标：
- 纯函数 / 易测试（不做数据拉取，不碰 IO）
- 不改变现有 orders schema（只做合并/补齐/排序/可选估算回填）
"""

import math
from typing import Any, Callable

from .costs import min_notional_for_min_fee


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return 0


def _to_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    return float(v) if math.isfinite(v) else None


def _norm_side(x: Any) -> str:
    s = str(x or "").strip().lower()
    return s if s in {"buy", "sell"} else "unknown"


def _norm_asset(x: Any) -> str:
    a = str(x or "").strip().lower()
    return a if a else "etf"


def _norm_symbol(x: Any) -> str:
    return str(x or "").strip()


def merge_orders_next_open(
    orders_from_holdings: list[dict[str, Any]] | None,
    orders_rebalance: list[dict[str, Any]] | None,
    *,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    合并 orders：
    - risk(holdings) 优先；rebalance 次之
    - 冲突规则：同一 symbol 同时出现 holdings.sell + rebalance.buy => 丢弃 buy（保命优先）
    - 去重规则：同 side/asset/symbol 重复 => shares 取最大；reason 进行拼接（不丢原因）
    """
    w = warnings if isinstance(warnings, list) else None

    oh = [o for o in (orders_from_holdings or []) if isinstance(o, dict)]
    or2 = [o for o in (orders_rebalance or []) if isinstance(o, dict)]

    sell_syms = {str(o.get("symbol") or "") for o in oh if _norm_side(o.get("side")) == "sell"}

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _merge_one(o: dict[str, Any]) -> None:
        side = _norm_side(o.get("side"))
        asset = _norm_asset(o.get("asset"))
        sym = _norm_symbol(o.get("symbol"))
        if side not in {"buy", "sell"} or not sym:
            return

        key = (side, asset, sym)
        if key not in merged:
            merged[key] = dict(o)
            return

        old = merged[key]
        sh_old = _to_int(old.get("shares"))
        sh_new = _to_int(o.get("shares"))
        if sh_new > sh_old:
            old["shares"] = int(sh_new)

        r0 = str(old.get("reason") or "").strip()
        r1 = str(o.get("reason") or "").strip()
        if r1 and r1 not in r0:
            old["reason"] = (r0 + "; " + r1) if r0 else r1

    # holdings 风险单先入
    for o in oh:
        _merge_one(o)

    # rebalance 单：冲突的 buy 直接丢掉
    for o in or2:
        side = _norm_side(o.get("side"))
        sym = _norm_symbol(o.get("symbol"))
        if side == "buy" and sym and sym in sell_syms:
            if w is not None:
                w.append(f"orders conflict: {sym} 同时出现 sell(holdings) + buy(rebalance)，已丢弃 buy 单")
            continue
        _merge_one(o)

    return list(merged.values())


def basic_enrich_orders_next_open(
    orders: list[dict[str, Any]],
    *,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    min_fee_yuan: float = 0.0,
    lot_size_stock_etf: int = 100,
    lot_size_other: int = 1,
) -> None:
    """
    对订单做“纯本地补齐”（不依赖行情/数据拉取）：
    - lot_size
    - est_notional_yuan
    - min_notional_for_min_fee_yuan（考虑最低佣金）
    - min_trade_notional_yuan（缺失时默认=上一条）

    注意：
    - 只补缺字段（setdefault），不覆盖已有值
    - price_ref 缺失/非法时不计算 notional
    """
    for o in orders:
        if not isinstance(o, dict):
            continue
        side = _norm_side(o.get("side"))
        asset = _norm_asset(o.get("asset"))
        sym = _norm_symbol(o.get("symbol"))
        if side not in {"buy", "sell"} or not sym:
            continue

        sh = _to_int(o.get("shares"))
        if sh <= 0:
            continue

        lot_size = int(lot_size_stock_etf) if asset in {"etf", "stock"} else int(lot_size_other)
        lot_size = max(1, int(lot_size))
        o.setdefault("lot_size", int(lot_size))

        px = _to_float(o.get("price_ref"))
        if px is None or px <= 0:
            continue

        notional = float(sh) * float(px)
        o.setdefault("est_notional_yuan", float(notional))

        cost_rate = float(buy_cost) if side == "buy" else float(sell_cost)
        o.setdefault("min_notional_for_min_fee_yuan", min_notional_for_min_fee(cost_rate=cost_rate, min_fee_yuan=float(min_fee_yuan)))
        o.setdefault("min_trade_notional_yuan", o.get("min_notional_for_min_fee_yuan"))


OrderEstimator = Callable[[dict[str, Any]], dict[str, Any]]


def apply_order_estimates(
    orders: list[dict[str, Any]],
    *,
    estimator: OrderEstimator,
) -> list[dict[str, Any]]:
    """
    可选：给订单补齐“成本/滑点/可交易性”等估算字段（由调用方提供 estimator）。

    - estimator 输入：订单 dict
    - estimator 输出：dict（会 setdefault 回填到订单）
    - estimator 抛异常：不影响订单输出，但会返回一条结构化 error（给上层写 diagnostics）
    """
    errors: list[dict[str, Any]] = []
    for o in orders:
        if not isinstance(o, dict):
            continue

        side = _norm_side(o.get("side"))
        asset = _norm_asset(o.get("asset"))
        sym = _norm_symbol(o.get("symbol"))
        if side not in {"buy", "sell"} or not sym:
            continue

        sh = _to_int(o.get("shares"))
        px = _to_float(o.get("price_ref"))
        if sh <= 0 or px is None or px <= 0:
            continue

        # 已有结果就别重复估算（避免慢/不稳定）
        if (o.get("est_fee_yuan") is not None) and (o.get("est_cash") is not None) and (o.get("slippage") is not None):
            continue

        try:
            out = estimator(o)
            if isinstance(out, dict):
                for k, v in out.items():
                    if k not in o:
                        o[k] = v
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
            errors.append(
                {
                    "stage": "estimate_order",
                    "side": side,
                    "asset": asset,
                    "symbol": sym,
                    "shares": int(sh),
                    "price_ref": float(px),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
    return errors


def sort_orders_next_open(orders: list[dict[str, Any]]) -> None:
    """
    稳定排序：sell 在前；再按 asset/symbol。
    """

    def _k(x: dict[str, Any]):
        side = _norm_side(x.get("side"))
        asset = _norm_asset(x.get("asset"))
        sym = _norm_symbol(x.get("symbol"))
        return (0 if side == "sell" else 1, asset, sym)

    orders.sort(key=_k)
