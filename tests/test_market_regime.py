# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_index_df(n: int = 320, *, seed: int, direction: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    steps = direction + rng.normal(loc=0.0, scale=0.002, size=n)
    close = 1000.0 + np.cumsum(steps * 100.0)
    close = np.maximum(close, 10.0)
    open_ = close * (1.0 + rng.normal(loc=0.0, scale=0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(loc=0.0, scale=0.001, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(loc=0.0, scale=0.001, size=n)))
    volume = (10_000_000 + rng.integers(low=0, high=500_000, size=n)).astype(float)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


class TestMarketRegime(unittest.TestCase):
    def test_parse_regime_index_list(self) -> None:
        from llm_trading.market_regime import parse_regime_index_list, parse_regime_index_spec

        self.assertEqual(parse_regime_index_list("off"), [])
        self.assertEqual(parse_regime_index_list("sh000300"), ["sh000300"])
        self.assertEqual(parse_regime_index_list("sh000300,sh000905"), ["sh000300", "sh000905"])
        # Accept "+" as an alias of ","
        self.assertEqual(parse_regime_index_list("sh000300+sh000905"), ["sh000300", "sh000905"])
        # Explicit canary: primary + canary
        primary, canary = parse_regime_index_spec("sh000300,sz399006;sh000852")
        self.assertEqual(primary, ["sh000300", "sz399006"])
        self.assertEqual(canary, ["sh000852"])

    def test_regime_bullish_trend(self) -> None:
        from llm_trading.market_regime import compute_market_regime

        df = _make_index_df(seed=1, direction=0.003)
        r = compute_market_regime(index_symbol="sh000300", df_daily=df)
        self.assertIn(r.label, {"bull", "neutral", "unknown"})
        self.assertNotEqual(r.label, "bear")

    def test_regime_bearish_trend(self) -> None:
        from llm_trading.market_regime import compute_market_regime

        df = _make_index_df(seed=2, direction=-0.003)
        r = compute_market_regime(index_symbol="sh000300", df_daily=df)
        self.assertIn(r.label, {"bear", "neutral", "unknown"})
        self.assertNotEqual(r.label, "bull")


if __name__ == "__main__":
    unittest.main()
