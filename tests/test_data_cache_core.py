# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _make_df(start: str, n: int, *, close0: float = 10.0, step: float = 0.1, extra_cols: dict | None = None) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B")
    close = [close0 + i * step for i in range(n)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000.0,
            "amount": [c * 1000.0 for c in close],
        }
    )
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    return df


class TestDataCacheCore(unittest.TestCase):
    def test_expected_latest_bar_dt_weekend_is_prev_fri(self) -> None:
        from llm_trading.data_cache import _expected_latest_bar_dt

        sat = datetime(2026, 1, 24, 12, 0, 0)  # Saturday
        dt = _expected_latest_bar_dt(sat, asset="stock")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.date().isoformat(), "2026-01-23")  # Friday

    def test_should_expect_new_bar_rules(self) -> None:
        from llm_trading.data_cache import _should_expect_new_bar

        # crypto: always True
        self.assertTrue(_should_expect_new_bar(datetime(2026, 1, 24, 0, 0), asset="crypto"))

        # A-share: before 15:05 -> False
        self.assertFalse(_should_expect_new_bar(datetime(2026, 1, 23, 15, 0), asset="stock"))
        # after 15:05 weekday -> True
        self.assertTrue(_should_expect_new_bar(datetime(2026, 1, 23, 15, 10), asset="stock"))
        # weekend -> False
        self.assertFalse(_should_expect_new_bar(datetime(2026, 1, 24, 15, 10), asset="stock"))

    def test_has_non_ascii_columns(self) -> None:
        from llm_trading.data_cache import _has_non_ascii_columns

        df = _make_df("2020-01-01", 5, extra_cols={"收盘价": [1, 2, 3, 4, 5]})
        self.assertTrue(_has_non_ascii_columns(df))

    def test_prefer_tushare_requires_source_and_env(self) -> None:
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import _prefer_tushare
        from llm_trading.tushare_source import TushareEnv

        # source != auto/tushare -> False
        self.assertFalse(_prefer_tushare(FetchParams(asset="etf", symbol="sh510300", source="akshare")))

        # source=auto but no env -> False
        with patch("llm_trading.tushare_source.load_tushare_env", return_value=None):
            self.assertFalse(_prefer_tushare(FetchParams(asset="etf", symbol="sh510300", source="auto")))

        # source=auto and env exists -> True
        with patch("llm_trading.tushare_source.load_tushare_env", return_value=TushareEnv(token="dummy")):
            self.assertTrue(_prefer_tushare(FetchParams(asset="etf", symbol="sh510300", source="auto")))

    def test_fetch_daily_cached_ttl0_bypasses_cache(self) -> None:
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import fetch_daily_cached

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            # Create a cache file that should be ignored.
            _make_df("2020-01-01", 3, close0=1.0).to_csv(cache_dir / "etf_sh510300_qfq.csv", index=False, encoding="utf-8")

            df_live = _make_df("2020-02-01", 3, close0=2.0)
            with patch("llm_trading.data_cache.fetch_daily", return_value=df_live) as f:
                out = fetch_daily_cached(FetchParams(asset="etf", symbol="sh510300"), cache_dir=cache_dir, ttl_hours=0.0)

            self.assertTrue(f.called)
            self.assertEqual(float(out.iloc[0]["close"]), float(df_live.iloc[0]["close"]))

    def test_fetch_daily_cached_cache_fresh_returns_without_fetch(self) -> None:
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import fetch_daily_cached

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            df_cache = _make_df("2020-01-01", 10, close0=10.0, step=0.0)
            path = cache_dir / "etf_sh510300_qfq.csv"
            df_cache.to_csv(path, index=False, encoding="utf-8")

            with patch("llm_trading.data_cache.fetch_daily", side_effect=RuntimeError("should not fetch")):
                out = fetch_daily_cached(
                    FetchParams(asset="etf", symbol="sh510300", start_date="2020-01-03", end_date="2020-01-10"),
                    cache_dir=cache_dir,
                    ttl_hours=24.0,
                )

            self.assertFalse(out.empty)
            self.assertEqual(out["date"].min().date().isoformat(), "2020-01-03")

    def test_fetch_daily_cached_backfill_and_forward_fill(self) -> None:
        """
        覆盖两个关键分支：
        - start_date 早于缓存起点 => backfill
        - end_date 晚于缓存终点 => forward_fill
        """
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import fetch_daily_cached

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            # cache: 2020-01-06 ~ 2020-01-08 (3 business days)
            df_cache = _make_df("2020-01-06", 3, close0=10.0, step=0.0)
            df_cache.to_csv(cache_dir / "etf_sh510300_qfq.csv", index=False, encoding="utf-8")

            df_pre = _make_df("2020-01-01", 3, close0=9.0, step=0.0)  # 2020-01-01~2020-01-03
            df_new = _make_df("2020-01-09", 2, close0=11.0, step=0.0)  # 2020-01-09~2020-01-10

            def _fake_fetch(p):
                sd = str(getattr(p, "start_date", "") or "")
                ed = str(getattr(p, "end_date", "") or "")
                if sd.startswith("20200101") and ed.startswith("20200105"):
                    return df_pre
                if sd.startswith("20200109") and ed.startswith("20200110"):
                    return df_new
                raise RuntimeError(f"unexpected fetch range: {sd} {ed}")

            with patch("llm_trading.data_cache.fetch_daily", side_effect=_fake_fetch):
                out = fetch_daily_cached(
                    FetchParams(asset="etf", symbol="sh510300", start_date="2020-01-01", end_date="2020-01-10"),
                    cache_dir=cache_dir,
                    ttl_hours=24.0,
                )

            self.assertEqual(out["date"].min().date().isoformat(), "2020-01-01")
            self.assertEqual(out["date"].max().date().isoformat(), "2020-01-10")

    def test_fetch_daily_cached_migrate_to_tushare_drops_non_ascii_cols(self) -> None:
        from llm_trading.akshare_source import FetchParams
        from llm_trading.data_cache import fetch_daily_cached
        from llm_trading.tushare_source import TushareEnv

        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            df_cache = _make_df("2020-01-01", 10, close0=10.0, step=0.0, extra_cols={"收盘价": [1] * 10})
            df_cache.to_csv(cache_dir / "etf_sh510300_qfq.csv", index=False, encoding="utf-8")

            # Tail refresh returns standard columns.
            df_patch = _make_df("2019-12-15", 5, close0=10.0, step=0.0)

            with patch("llm_trading.tushare_source.load_tushare_env", return_value=TushareEnv(token="dummy")):
                with patch("llm_trading.data_cache.fetch_daily", return_value=df_patch):
                    out = fetch_daily_cached(
                        FetchParams(asset="etf", symbol="sh510300", start_date="2020-01-01", end_date="2020-01-10", source="auto"),
                        cache_dir=cache_dir,
                        ttl_hours=24.0,
                    )

            self.assertFalse(any(any(ord(ch) > 127 for ch in str(c)) for c in out.columns), "non-ascii columns should be dropped after migrate")


if __name__ == "__main__":
    unittest.main()

