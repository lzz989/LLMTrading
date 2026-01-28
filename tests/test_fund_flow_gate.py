# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestFundFlowGate(unittest.TestCase):
    def test_score_from_meta(self) -> None:
        from llm_trading.commands.portfolio import _fund_flow_score_from_meta

        meta = {"fund_flow": {"ok": True, "score01": 0.42}}
        self.assertAlmostEqual(float(_fund_flow_score_from_meta(meta) or 0.0), 0.42, places=6)

        meta2 = {"fund_flow": {"ok": False, "score01": "0.33"}}
        self.assertAlmostEqual(float(_fund_flow_score_from_meta(meta2) or 0.0), 0.33, places=6)

        self.assertIsNone(_fund_flow_score_from_meta({"fund_flow": {"score01": None}}))
        self.assertIsNone(_fund_flow_score_from_meta({"x": 1}))

    def test_decision_for_score(self) -> None:
        from llm_trading.commands.portfolio import _fund_flow_decision_for_score

        self.assertEqual(_fund_flow_decision_for_score(None, 0.45, 0.55), "skip")
        self.assertEqual(_fund_flow_decision_for_score(0.40, 0.45, 0.55), "block")
        self.assertEqual(_fund_flow_decision_for_score(0.50, 0.45, 0.55), "warn")
        self.assertEqual(_fund_flow_decision_for_score(0.60, 0.45, 0.55), "pass")


if __name__ == "__main__":
    unittest.main()

