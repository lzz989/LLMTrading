# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestFactorsBase(unittest.TestCase):
    def test_factor_normalize_score_methods_and_direction(self) -> None:
        from llm_trading.factors.base import Factor, FactorResult

        class _Dummy(Factor):
            name = "_dummy_norm"
            category = "trend"

            def compute(self, df: pd.DataFrame) -> FactorResult:  # pragma: no cover
                return FactorResult(name=self.name, value=0.0, score=0.5, direction="neutral", confidence=0.0, details={})

        f = _Dummy()

        # boolean
        self.assertEqual(f.normalize_score(True, method="boolean"), 1.0)
        self.assertEqual(f.normalize_score(False, method="boolean"), 0.0)

        # minmax
        self.assertEqual(f.normalize_score(1.0, method="minmax", min_val=1.0, max_val=1.0), 0.5)
        self.assertAlmostEqual(f.normalize_score(5.0, method="minmax", min_val=0.0, max_val=10.0), 0.5, places=6)

        # zscore
        hist = pd.Series([0.0, 1.0, 2.0, 3.0], dtype=float)
        z = f.normalize_score(2.0, method="zscore", history=hist)
        self.assertTrue(0.0 < z < 1.0)

        # percentile
        p = f.normalize_score(2.0, method="percentile", history=hist)
        self.assertAlmostEqual(float(p), 0.5, places=6)

        # direction thresholds
        self.assertEqual(f.get_direction(0.7), "bullish")
        self.assertEqual(f.get_direction(0.3), "bearish")
        self.assertEqual(f.get_direction(0.5), "neutral")

    def test_registry_register_duplicate_and_compute_all_error(self) -> None:
        from llm_trading.factors.base import FACTOR_REGISTRY, Factor, FactorResult

        class _Boom(Factor):
            name = "_dummy_boom"
            category = "trend"

            def compute(self, df: pd.DataFrame) -> FactorResult:
                raise RuntimeError("boom")

        # Make sure we clean up even if the test fails.
        reg = FACTOR_REGISTRY
        if _Boom.name in reg._factors:  # type: ignore[attr-defined]
            del reg._factors[_Boom.name]  # type: ignore[attr-defined]

        try:
            reg.register(_Boom)
            with self.assertRaises(ValueError):
                reg.register(_Boom)

            # get unknown
            with self.assertRaises(KeyError):
                reg.get("_no_such_factor")

            df = pd.DataFrame({"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [1.0, 2.0], "volume": [100.0, 120.0]})
            out = reg.compute_all(df, [_Boom.name])
            self.assertIn(_Boom.name, out)
            r = out[_Boom.name]
            self.assertEqual(r.direction, "neutral")
            self.assertIn("error", r.details)
        finally:
            # restore registry
            if _Boom.name in reg._factors:  # type: ignore[attr-defined]
                del reg._factors[_Boom.name]  # type: ignore[attr-defined]

    def test_strategy_config_validate_and_explain(self) -> None:
        from llm_trading.factors.base import StrategyConfig, StrategyEngine

        # invalid config => normalize_weights() is called in engine init
        cfg = StrategyConfig(
            name="t",
            factor_weights={"ma_cross": 1.0, "macd": 1.0},  # sum != 1
            entry_threshold=0.6,
            exit_threshold=0.4,
        )
        self.assertFalse(cfg.validate())

        engine = StrategyEngine(cfg)
        s = engine.explain_signal({"action": "hold", "score": 0.5, "confidence": 0.0, "reason": "x", "factors": {}})
        self.assertIn("策略:", s)
        self.assertIn("信号:", s)


if __name__ == "__main__":
    unittest.main()

