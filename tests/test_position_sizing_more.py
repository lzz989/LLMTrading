# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from datetime import date


class TestPositionSizingMore(unittest.TestCase):
    def test_probe_and_reject_buckets_and_round_lot(self) -> None:
        from llm_trading.position_sizing import PositionSizingInputs, compute_position_sizing

        # reject => zero suggestion
        out0 = compute_position_sizing(
            inputs=PositionSizingInputs(
                symbol="sh510300",
                asset="etf",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                opportunity_score=float("nan"),
                bucket="reject",
                confidence=0.5,
                max_position_pct=0.3,
                price=1.0,
            )
        )
        self.assertEqual(float(out0.get("suggest_position_pct") or 0.0), 0.0)
        self.assertEqual(float(out0.get("suggest_trade_notional_yuan") or 0.0), 0.0)
        self.assertIsNone(out0.get("suggest_shares"))

        # probe => has suggestion and respects min_trade_notional + lot size for stock
        out1 = compute_position_sizing(
            inputs=PositionSizingInputs(
                symbol="sz000001",
                asset="stock",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                opportunity_score=0.75,
                bucket="probe",
                confidence=0.6,
                max_position_pct=0.3,
                price=10.0,
                min_trade_notional_yuan=2000,
                min_fee_yuan=5.0,
            )
        )
        self.assertEqual(int(out1.get("lot_size") or 0), 100)
        self.assertGreaterEqual(float(out1.get("suggest_trade_notional_yuan") or 0.0), 2000.0)
        shares = out1.get("suggest_shares")
        self.assertIsNotNone(shares)
        self.assertEqual(int(shares) % 100, 0)


if __name__ == "__main__":
    unittest.main()

