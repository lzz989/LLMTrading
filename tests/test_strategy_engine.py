# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import pandas as pd

from llm_trading.factors.base import FactorResult, StrategyConfig, StrategyEngine


class _DummyRegistry:
    def __init__(self, results: dict[str, FactorResult]):
        self._results = dict(results)

    def compute_all(self, df: pd.DataFrame, factor_names: list[str], params: dict | None = None) -> dict[str, FactorResult]:
        # Minimal protocol used by StrategyEngine.
        return {name: self._results[name] for name in factor_names}


class TestStrategyEngine(unittest.TestCase):
    def test_weight_normalization_on_init(self) -> None:
        cfg = StrategyConfig(name="t", factor_weights={"a": 2.0, "b": 1.0})
        eng = StrategyEngine(cfg, registry=_DummyRegistry({}))
        self.assertAlmostEqual(sum(eng.config.factor_weights.values()), 1.0, places=6)
        self.assertAlmostEqual(eng.config.factor_weights["a"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(eng.config.factor_weights["b"], 1.0 / 3.0, places=6)

    def test_market_regime_block(self) -> None:
        cfg = StrategyConfig(name="t", factor_weights={"a": 1.0})
        eng = StrategyEngine(cfg, registry=_DummyRegistry({}))
        out = eng.generate_signal(pd.DataFrame(), market_regime="bear")
        self.assertEqual(out["action"], "hold")
        self.assertEqual(out["score"], 0.5)
        self.assertEqual(out["confidence"], 0.0)
        self.assertEqual(out["factors"], {})
        self.assertIn("不在允许范围", out["reason"])

    def test_filters_require_factor(self) -> None:
        cfg = StrategyConfig(name="t", factor_weights={"a": 1.0}, require_factors=["a"])
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=0.4, direction="neutral", confidence=1.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        out = eng.generate_signal(pd.DataFrame(), market_regime="neutral")
        self.assertEqual(out["action"], "hold")
        self.assertIn("必须因子 a 未满足", out["reason"])

    def test_filters_exclude_factor(self) -> None:
        cfg = StrategyConfig(name="t", factor_weights={"a": 1.0}, exclude_factors=["a"])
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=0.9, direction="bullish", confidence=1.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        out = eng.generate_signal(pd.DataFrame(), market_regime="neutral")
        self.assertEqual(out["action"], "hold")
        self.assertIn("排除因子 a 触发", out["reason"])

    def test_filters_exclude_factor_not_in_weights_is_enforced(self) -> None:
        # Regression test: exclude_factors previously had no effect if the factor wasn't in factor_weights.
        cfg = StrategyConfig(name="t", factor_weights={"a": 1.0}, exclude_factors=["x"])
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=0.9, direction="bullish", confidence=1.0),
                "x": FactorResult(name="x", value=0.0, score=0.9, direction="bearish", confidence=1.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        out = eng.generate_signal(pd.DataFrame(), market_regime="neutral")
        self.assertEqual(out["action"], "hold")
        self.assertIn("排除因子 x 触发", out["reason"])

    def test_entry_threshold(self) -> None:
        cfg = StrategyConfig(
            name="t",
            factor_weights={"a": 0.6, "b": 0.4},
            entry_threshold=0.6,
            exit_threshold=0.4,
        )
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=0.8, direction="bullish", confidence=1.0),
                "b": FactorResult(name="b", value=0.0, score=0.6, direction="bullish", confidence=1.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        out = eng.generate_signal(pd.DataFrame(), market_regime="neutral")
        self.assertEqual(out["action"], "entry")
        self.assertGreaterEqual(out["score"], cfg.entry_threshold)
        self.assertAlmostEqual(out["confidence"], 1.0, places=6)
        self.assertIn(">= 入场阈值", out["reason"])

    def test_exit_threshold(self) -> None:
        cfg = StrategyConfig(
            name="t",
            factor_weights={"a": 0.5, "b": 0.5},
            entry_threshold=0.7,
            exit_threshold=0.4,
        )
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=0.2, direction="bearish", confidence=1.0),
                "b": FactorResult(name="b", value=0.0, score=0.4, direction="neutral", confidence=1.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        out = eng.generate_signal(pd.DataFrame(), market_regime="neutral")
        self.assertEqual(out["action"], "exit")
        self.assertLessEqual(out["score"], cfg.exit_threshold + 1e-12)
        self.assertIn("<= 出场阈值", out["reason"])

    def test_composite_score_is_confidence_weighted(self) -> None:
        cfg = StrategyConfig(name="t", factor_weights={"a": 0.6, "b": 0.4})
        reg = _DummyRegistry(
            {
                "a": FactorResult(name="a", value=0.0, score=1.0, direction="bullish", confidence=1.0),
                "b": FactorResult(name="b", value=0.0, score=0.0, direction="bearish", confidence=0.0),
            }
        )
        eng = StrategyEngine(cfg, registry=reg)
        score = eng.compute_composite_score(reg.compute_all(pd.DataFrame(), ["a", "b"]))
        self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
