# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class TestDailyBrief(unittest.TestCase):
    def test_build_daily_brief_minimal(self) -> None:
        from llm_trading.commands.brief import build_daily_brief

        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "signals.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "strategy": "demo",
                        "items": [
                            {
                                "asset": "etf",
                                "symbol": "sh510010",
                                "name": "示例ETF",
                                "action": "entry",
                                "score": 0.9,
                                "entry": {"price_ref": 1.23},
                                "meta": {"close": 1.22},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "signals_stock.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "strategy": "demo",
                        "items": [
                            {
                                "asset": "stock",
                                "symbol": "sz002236",
                                "name": "示例个股",
                                "action": "entry",
                                "score": 0.8,
                                "entry": {"price_ref": 19.9},
                                "meta": {"close": 19.8},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "rebalance_user.json").write_text(
                json.dumps(
                    {
                        "position_plan": {
                            "plans": [
                                {
                                    "symbol": "sh510010",
                                    "name": "示例ETF",
                                    "ok": True,
                                    "entry": 1.23,
                                    "stop": 1.20,
                                    "stop_ref": "MA20",
                                    "shares": 1000,
                                    "position_yuan": 1230.0,
                                }
                            ]
                        },
                        "warnings": ["demo warning"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "orders_next_open.json").write_text(
                json.dumps(
                    [
                        {
                            "side": "buy",
                            "symbol": "sh510010",
                            "name": "示例ETF",
                            "shares": 1000,
                            "reason": "demo",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "holdings_user.json").write_text(
                json.dumps(
                    {
                        "as_of": "2026-01-27",
                        "market_regime": {"label": "bull"},
                        "portfolio": {
                            "equity_yuan": 10000,
                            "exposure_pct": 0.8,
                            "cash_pct": 0.2,
                            "risk_to_stop_pct_equity": 0.05,
                        },
                    }
                ),
                encoding="utf-8",
            )

            md = build_daily_brief(run_dir=run_dir, max_candidates=6, max_portfolio=3, max_warnings=10)
            self.assertIn("每日量化简报", md)
            self.assertIn("sh510010", md)
            self.assertIn("示例ETF", md)
            self.assertIn("模拟持仓", md)
            self.assertIn("orders_next_open", md)


if __name__ == "__main__":
    unittest.main()

