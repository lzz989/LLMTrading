# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestPaperSimUtils(unittest.TestCase):
    def test_max_drawdown_basic(self) -> None:
        from llm_trading.paper_sim import _max_drawdown

        self.assertIsNone(_max_drawdown([]))
        self.assertAlmostEqual(_max_drawdown([1.0]), 0.0)
        self.assertAlmostEqual(_max_drawdown([1.0, 2.0, 1.0]), -0.5)
        # Worst drawdown is from 2.0 -> 1.0 = -50% (even if later new highs appear).
        self.assertAlmostEqual(_max_drawdown([1.0, 2.0, 1.0, 3.0, 2.4]), -0.5)


if __name__ == "__main__":
    unittest.main()
