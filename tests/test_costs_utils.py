# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestCostsUtils(unittest.TestCase):
    def test_min_notional_for_min_fee(self) -> None:
        from llm_trading.costs import min_notional_for_min_fee

        self.assertIsNone(min_notional_for_min_fee(cost_rate=0.0, min_fee_yuan=5.0))
        self.assertIsNone(min_notional_for_min_fee(cost_rate=0.001, min_fee_yuan=0.0))
        self.assertAlmostEqual(min_notional_for_min_fee(cost_rate=0.001, min_fee_yuan=5.0) or 0.0, 5000.0)

    def test_effective_fee_yuan(self) -> None:
        from llm_trading.costs import effective_fee_yuan

        # notional*rate = 2 < min_fee => use min_fee + fixed_fee
        fee = effective_fee_yuan(notional_yuan=2000, cost_rate=0.001, min_fee_yuan=5, fixed_fee_yuan=1)
        self.assertAlmostEqual(fee, 6.0)

        # notional*rate = 20 > min_fee => use notional*rate + fixed_fee
        fee2 = effective_fee_yuan(notional_yuan=20000, cost_rate=0.001, min_fee_yuan=5, fixed_fee_yuan=1)
        self.assertAlmostEqual(fee2, 21.0)


if __name__ == "__main__":
    unittest.main()

