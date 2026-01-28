# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestPortfolioGridExempt(unittest.TestCase):
    def test_grid_exempt_syms_from_snapshot(self) -> None:
        from llm_trading.commands.portfolio import _grid_exempt_syms_from_user_holdings_snapshot

        self.assertEqual(_grid_exempt_syms_from_user_holdings_snapshot(None), set())
        self.assertEqual(_grid_exempt_syms_from_user_holdings_snapshot({}), set())

        snap = {
            "positions": [
                {"symbol": "sh510150", "grid_plan": {"enabled": True}},
                {"symbol": "sh513050", "frozen": True},
                {"symbol": "sh512980", "grid_plan": {"enabled": False}},
                {"symbol": "  ", "grid_plan": {"enabled": True}},
            ]
        }
        self.assertEqual(_grid_exempt_syms_from_user_holdings_snapshot(snap), {"sh510150"})


if __name__ == "__main__":
    unittest.main()

