# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 80, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-10-01", periods=n, freq="B")
    steps = 0.002 + rng.normal(loc=0.0, scale=0.01, size=n)
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


class TestHotlinesMetrics(unittest.TestCase):
    def test_compute_hotness_metrics(self) -> None:
        from llm_trading.skills.hotlines import _score_and_flags, compute_hotness_metrics

        df = _make_ohlcv()
        m = compute_hotness_metrics(df)
        self.assertTrue(bool(m.get("ok")))
        for k in ["as_of", "close", "ret_5d", "ret_10d", "ret_20d", "vol_ratio_20d", "close_vs_ma20_pct", "atr_pct_14"]:
            self.assertIn(k, m)

        score, flags = _score_and_flags(m)
        self.assertTrue(0.0 <= float(score) <= 1.0)
        self.assertIsInstance(flags, list)


if __name__ == "__main__":
    unittest.main()

