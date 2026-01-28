# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _make_min_ohlcv(start: str = "2020-01-01", n: int = 5) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B")
    close = pd.Series(range(1, n + 1), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "amount": close * 1000.0,
        }
    )
    return df


class TestErrorVisibility(unittest.TestCase):
    def test_data_cache_suppressed_errors_are_recorded(self) -> None:
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import fetch_daily_cached

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)

            # Create a valid cache file first.
            df_cache = _make_min_ohlcv(start="2020-01-01", n=5)
            key = "etf_sh510300_qfq.csv"
            df_cache.to_csv(cache_dir / key, index=False, encoding="utf-8")

            # Force "need_forward" and make the forward fetch fail.
            with patch("llm_trading.data_cache._LOG.warning", return_value=None):
                with patch("llm_trading.data_cache._expected_latest_bar_dt", return_value=datetime(2020, 1, 31)):
                    with patch("llm_trading.data_cache.fetch_daily", side_effect=RuntimeError("network boom")):
                        out = fetch_daily_cached(FetchParams(asset="etf", symbol="sh510300"), cache_dir=cache_dir, ttl_hours=24.0)

            self.assertIsInstance(out, pd.DataFrame)
            self.assertFalse(out.empty)

            # We should see at least one suppressed error with stage=forward_fill.
            self.assertIn("cache_warnings", out.attrs)
            warns = out.attrs.get("cache_warnings") or []
            self.assertTrue(any(w.get("stage") == "forward_fill" for w in warns))

    def test_fetch_daily_auto_fallback_keeps_error_attr(self) -> None:
        from llm_trading.akshare_source import FetchParams, fetch_daily
        from llm_trading.tushare_source import TushareEnv

        df_fallback = _make_min_ohlcv(start="2020-01-01", n=3)

        with patch("llm_trading.tushare_source.load_tushare_env", return_value=TushareEnv(token="dummy")):
            with patch("llm_trading.tushare_kline.fetch_etf_daily_tushare", side_effect=RuntimeError("ts boom")):
                with patch("llm_trading.akshare_source._require_akshare", return_value=None):
                    with patch("llm_trading.akshare_source._try_fetch_etf", return_value=df_fallback):
                        with patch("llm_trading.akshare_source._warn_auto_fallback_once") as w:
                            out = fetch_daily(FetchParams(asset="etf", symbol="sh510300", source="auto"))

        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(out.attrs.get("data_source"), "akshare")
        self.assertIn("data_source_auto_fallback_error", out.attrs)
        self.assertIn("ts boom", str(out.attrs.get("data_source_auto_fallback_error")))
        self.assertTrue(w.called)


if __name__ == "__main__":
    unittest.main()
