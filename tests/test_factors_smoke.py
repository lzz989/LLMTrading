# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 320, *, seed: int = 7, direction: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # A tiny drift + noise; keep prices strictly positive.
    steps = direction + rng.normal(loc=0.0, scale=0.01, size=n)
    close = 1.0 + np.cumsum(steps)
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


class TestFactorsSmoke(unittest.TestCase):
    def test_all_registered_factors_compute(self) -> None:
        # Import triggers @register_factor side-effects.
        from llm_trading import factors  # noqa: F401
        from llm_trading.factors.base import FACTOR_REGISTRY

        df = _make_ohlcv()

        names = FACTOR_REGISTRY.list_factors()
        self.assertGreaterEqual(len(names), 17)

        res = FACTOR_REGISTRY.compute_all(df, names)
        self.assertEqual(set(res.keys()), set(names))

        for name, r in res.items():
            self.assertIsNotNone(r, msg=name)
            self.assertEqual(r.name, name)
            # Most factors are normalized to 0..1; accept a small epsilon.
            self.assertTrue(-1.01 <= float(r.score) <= 1.01, msg=f"{name} score={r.score}")
            self.assertTrue(0.0 <= float(r.confidence) <= 1.01, msg=f"{name} conf={r.confidence}")


if __name__ == "__main__":
    unittest.main()

