# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 200, *, seed: int = 11, direction: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    steps = direction + rng.normal(loc=0.0, scale=0.01, size=n)
    close = 10.0 + np.cumsum(steps)
    close = np.maximum(close, 0.1)

    open_ = close * (1.0 + rng.normal(loc=0.0, scale=0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(loc=0.0, scale=0.003, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(loc=0.0, scale=0.003, size=n)))

    volume = (1_000_000 + rng.integers(low=0, high=200_000, size=n)).astype(float)
    amount = close * volume

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )


class TestPhase2Outputs(unittest.TestCase):
    def test_game_theory_factor_pack_schema(self) -> None:
        from llm_trading.factors.game_theory import compute_game_theory_factor_pack

        df = _make_ohlcv()
        pack = compute_game_theory_factor_pack(
            df=df,
            symbol="sh510300",
            asset="etf",
            as_of=date(2026, 1, 23),
            ref_date=date(2026, 1, 23),
            source="factors",
        )

        self.assertEqual(pack.get("schema"), "llm_trading.game_theory_factors.v1")
        self.assertEqual(pack.get("symbol"), "sh510300")
        self.assertIn("factors", pack)

        factors = pack["factors"]
        self.assertIsInstance(factors, dict)
        for k in ["liquidity_trap", "stop_cluster", "capitulation", "fomo", "wyckoff_phase_proxy"]:
            self.assertIn(k, factors)
            item = factors[k]
            self.assertIsInstance(item, dict)
            self.assertIn("name", item)
            self.assertIn("score", item)
            self.assertIn("direction", item)
            self.assertIn("confidence", item)
            self.assertIn("details", item)

    def test_opportunity_score_schema(self) -> None:
        from llm_trading.opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        df = _make_ohlcv()
        out = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol="sh510300",
                asset="etf",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                min_score=0.70,
                t_plus_one=True,
                trap_risk=0.2,
                fund_flow=None,
                expected_holding_days=10,
            ),
            key_level_name="close",
            key_level_value=float(df.iloc[-1]["close"]),
        )

        self.assertEqual(out.get("schema"), "llm_trading.opportunity_score.v1")
        self.assertEqual(out.get("symbol"), "sh510300")
        sc = float(out.get("total_score") or 0.0)
        self.assertTrue(0.0 <= sc <= 1.0)
        self.assertIn(out.get("bucket"), {"reject", "probe", "plan"})
        self.assertIn(out.get("verdict"), {"tradeable", "not_tradeable"})

        comps = out.get("components")
        self.assertIsInstance(comps, dict)
        for k in ["trend", "regime", "risk_reward", "liquidity", "trap_risk", "fund_flow"]:
            self.assertIn(k, comps)

    def test_opportunity_score_stock_fund_flow_penalty(self) -> None:
        """
        覆盖：
        - stock：fund_flow 作为惩罚项（低于0.5扣分）
        """
        from llm_trading.opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        df = _make_ohlcv()
        out_mid = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol="sh600000",
                asset="stock",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                trap_risk=0.0,
                fund_flow=0.5,
            ),
            key_level_name="close",
            key_level_value=float(df.iloc[-1]["close"]),
        )
        out_bad = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol="sh600000",
                asset="stock",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                trap_risk=0.0,
                fund_flow=0.0,
            ),
            key_level_name="close",
            key_level_value=float(df.iloc[-1]["close"]),
        )

        sc_mid = float(out_mid.get("total_score") or 0.0)
        sc_bad = float(out_bad.get("total_score") or 0.0)
        self.assertGreaterEqual(sc_mid, sc_bad)

    def test_position_sizing_plan_bucket(self) -> None:
        from llm_trading.position_sizing import PositionSizingInputs, compute_position_sizing

        out = compute_position_sizing(
            inputs=PositionSizingInputs(
                symbol="sz000001",
                asset="stock",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
                opportunity_score=0.85,
                bucket="plan",
                confidence=0.6,
                max_position_pct=0.3,
                price=10.0,
                min_trade_notional_yuan=2000,
                min_fee_yuan=5.0,
                t_plus_one=True,
            )
        )

        self.assertEqual(out.get("schema"), "llm_trading.position_sizing.v1")
        self.assertEqual(out.get("asset"), "stock")
        self.assertEqual(int(out.get("lot_size") or 0), 100)
        self.assertGreater(float(out.get("suggest_position_pct") or 0.0), 0.0)
        self.assertGreaterEqual(float(out.get("suggest_trade_notional_yuan") or 0.0), 2000.0)
        shares = out.get("suggest_shares")
        if shares is not None:
            self.assertEqual(int(shares) % 100, 0)

    def test_cash_signal_with_mocked_market_regime(self) -> None:
        from llm_trading.cash_signal import CashSignalInputs, compute_cash_signal

        payload = {"label": "bear", "vol_20d": 0.03, "panic": True}
        with patch("llm_trading.market_regime.compute_market_regime_payload", return_value=(payload, None, "sh000300")):
            out = compute_cash_signal(inputs=CashSignalInputs(as_of=date(2026, 1, 23), ref_date=date(2026, 1, 23)))

        self.assertEqual(out.get("schema"), "llm_trading.cash_signal.v1")
        self.assertTrue(bool(out.get("should_stay_cash")))
        self.assertTrue(0.0 <= float(out.get("cash_ratio") or 0.0) <= 1.0)
        self.assertEqual(out.get("risk_mode"), "risk_off")
        ev = out.get("evidence") or {}
        self.assertEqual(ev.get("market_regime"), "bear")

    def test_cash_signal_bull_and_tushare_factors(self) -> None:
        """
        覆盖：
        - bull => risk_on
        - vol_state=high => cash_ratio >= 0.7（should_stay_cash True）
        - tushare_factors erp/hsgt 解析
        """
        from llm_trading.cash_signal import CashSignalInputs, compute_cash_signal

        payload = {"label": "bull", "vol_20d": 0.03, "panic": False}
        tf = {
            "erp": {"ok": True, "erp": 0.02},
            "hsgt": {"ok": True, "north": {"score01": 0.8}, "south": {"score01": 0.2}},
        }
        with patch("llm_trading.market_regime.compute_market_regime_payload", return_value=(payload, None, "sh000300")):
            out = compute_cash_signal(inputs=CashSignalInputs(as_of=date(2026, 1, 23), ref_date=date(2026, 1, 23)), tushare_factors=tf)

        self.assertEqual(out.get("risk_mode"), "risk_on")
        self.assertTrue(bool(out.get("should_stay_cash")))  # vol_state=high caps to >=0.7
        ev = out.get("evidence") or {}
        self.assertEqual(ev.get("market_regime"), "bull")
        self.assertIsNotNone(ev.get("erp_proxy"))
        self.assertEqual(float(ev.get("north_score01") or 0.0), 0.8)

    def test_cash_signal_market_regime_exception_falls_back(self) -> None:
        from llm_trading.cash_signal import CashSignalInputs, compute_cash_signal

        with patch("llm_trading.market_regime.compute_market_regime_payload", side_effect=RuntimeError("boom")):
            out = compute_cash_signal(inputs=CashSignalInputs(as_of=date(2026, 1, 23), ref_date=date(2026, 1, 23)))

        self.assertEqual(out.get("risk_mode"), "neutral")
        self.assertAlmostEqual(float(out.get("cash_ratio") or 0.0), 0.5, places=6)
        self.assertIn("regime_error=", str(out.get("reason") or ""))

    def test_opportunity_score_key_level_fallback(self) -> None:
        from llm_trading.opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        df = _make_ohlcv()
        # inject ma50 so the fallback path chooses it
        df["ma50"] = df["close"].astype(float).rolling(window=50, min_periods=50).mean()

        out = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol="sh510300",
                asset="etf",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
            ),
            key_level_name=None,
            key_level_value=None,
        )

        kl = out.get("key_level") or {}
        self.assertEqual(kl.get("name"), "ma50")
        self.assertIsNotNone(kl.get("value"))

    def test_opportunity_score_data_invalid(self) -> None:
        from llm_trading.opportunity_score import OpportunityScoreInputs, compute_opportunity_score

        df = pd.DataFrame({"close": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5]})  # missing volume
        out = compute_opportunity_score(
            df=df,
            inputs=OpportunityScoreInputs(
                symbol="sh510300",
                asset="etf",
                as_of=date(2026, 1, 23),
                ref_date=date(2026, 1, 23),
            ),
        )
        self.assertEqual(out.get("notes"), "data_invalid")
        self.assertEqual(out.get("verdict"), "not_tradeable")


if __name__ == "__main__":
    unittest.main()
