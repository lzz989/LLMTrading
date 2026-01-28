# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestOrdersNextOpen(unittest.TestCase):
    def test_merge_conflict_drop_buy(self) -> None:
        from llm_trading.orders_next_open import merge_orders_next_open

        warnings: list[str] = []
        orders = merge_orders_next_open(
            orders_from_holdings=[
                {"side": "sell", "asset": "stock", "symbol": "000001", "shares": 100, "reason": "stop"},
            ],
            orders_rebalance=[
                {"side": "buy", "asset": "stock", "symbol": "000001", "shares": 200, "reason": "rebalance"},
            ],
            warnings=warnings,
        )
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["side"], "sell")
        self.assertTrue(any("已丢弃 buy" in w for w in warnings))

    def test_merge_dedupe_keeps_max_shares_and_concat_reason(self) -> None:
        from llm_trading.orders_next_open import merge_orders_next_open

        orders = merge_orders_next_open(
            orders_from_holdings=[
                {"side": "buy", "asset": "etf", "symbol": "sh510300", "shares": 100, "reason": "a"},
                {"side": "buy", "asset": "etf", "symbol": "sh510300", "shares": 200, "reason": "b"},
            ],
            orders_rebalance=[],
        )
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertEqual(int(o.get("shares") or 0), 200)
        self.assertIn("a", str(o.get("reason") or ""))
        self.assertIn("b", str(o.get("reason") or ""))

    def test_basic_enrich_sets_notional_and_min_fee_threshold(self) -> None:
        from llm_trading.orders_next_open import basic_enrich_orders_next_open

        orders = [{"side": "buy", "asset": "etf", "symbol": "sh510300", "shares": 100, "price_ref": 10.0}]
        basic_enrich_orders_next_open(orders, buy_cost=0.001, sell_cost=0.002, min_fee_yuan=5.0)

        o = orders[0]
        self.assertEqual(o.get("lot_size"), 100)
        self.assertAlmostEqual(float(o.get("est_notional_yuan")), 1000.0, places=6)
        # min_fee / cost_rate = 5 / 0.001 = 5000
        self.assertAlmostEqual(float(o.get("min_notional_for_min_fee_yuan")), 5000.0, places=6)
        self.assertAlmostEqual(float(o.get("min_trade_notional_yuan")), 5000.0, places=6)

    def test_apply_order_estimates_success_and_error(self) -> None:
        from llm_trading.orders_next_open import apply_order_estimates

        orders = [
            {"side": "buy", "asset": "etf", "symbol": "sh510300", "shares": 100, "price_ref": 10.0},
            {"side": "sell", "asset": "etf", "symbol": "sh510500", "shares": 100, "price_ref": 10.0},
        ]

        def est(o: dict) -> dict:
            sym = str(o.get("symbol"))
            if sym == "sh510500":
                raise ValueError("boom")
            return {"est_cash": 1001.0, "est_fee_yuan": 1.0, "slippage": {"slippage_bps": 2.0}}

        errs = apply_order_estimates(orders, estimator=est)
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0].get("symbol"), "sh510500")
        self.assertEqual(errs[0].get("error_type"), "ValueError")

        o0 = orders[0]
        self.assertEqual(o0.get("est_cash"), 1001.0)
        self.assertEqual(o0.get("est_fee_yuan"), 1.0)
        self.assertIsInstance(o0.get("slippage"), dict)


if __name__ == "__main__":
    unittest.main()

