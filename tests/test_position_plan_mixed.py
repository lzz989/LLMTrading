# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestPositionPlanMixed(unittest.TestCase):
    def test_build_plan_with_etf_and_stock_assets(self) -> None:
        from llm_trading.positioning import PositionPlanParams, build_etf_position_plan

        items = [
            {
                "asset": "etf",
                "symbol": "sh510300",
                "name": "沪深300ETF",
                "close": 2.0,
                "levels": {"bbb_ma_entry": 1.8},
                "exit": {"daily": {"ma20": 1.7}},
            },
            {
                "asset": "stock",
                "symbol": "000001",
                "name": "平安银行",
                "close": 10.0,
                "levels": {"bbb_ma_entry": 9.5},
                "exit": {"daily": {"ma20": 9.0}},
            },
        ]

        plan = build_etf_position_plan(
            items=items,
            market_regime_label="bull",
            params=PositionPlanParams(
                capital_yuan=10000,
                roundtrip_cost_yuan=5.0,
                max_positions=2,
                max_position_pct=0.40,
                diversify=False,
            ),
        )

        plans = plan.get("plans") if isinstance(plan, dict) else []
        self.assertEqual(int(plan.get("counts", {}).get("picked") or 0), 2)
        assets = {str(p.get("asset") or "") for p in plans if isinstance(p, dict)}
        self.assertIn("etf", assets)
        self.assertIn("stock", assets)

    def test_build_stock_position_plan_defaults_asset(self) -> None:
        from llm_trading.positioning import PositionPlanParams, build_stock_position_plan

        items = [
            {
                "symbol": "000002",
                "name": "万科A",
                "close": 12.0,
                "levels": {"bbb_ma_entry": 11.0},
                "exit": {"daily": {"ma20": 10.5}},
            }
        ]

        plan = build_stock_position_plan(
            items=items,
            market_regime_label="neutral",
            params=PositionPlanParams(
                capital_yuan=20000,
                roundtrip_cost_yuan=5.0,
                max_positions=1,
                diversify=False,
            ),
        )

        plans = plan.get("plans") if isinstance(plan, dict) else []
        self.assertEqual(len(plans), 1)
        self.assertEqual(str(plans[0].get("asset") or ""), "stock")


if __name__ == "__main__":
    unittest.main()
