# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import pandas as pd


def _make_weekly(open_prices: list[float]) -> pd.DataFrame:
    start = datetime(2020, 1, 3)
    dates = [start + timedelta(days=7 * i) for i in range(len(open_prices))]
    open_s = pd.Series(open_prices, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_s,
            "high": open_s * 1.02,
            "low": open_s * 0.98,
            "close": open_s,
        }
    )


class TestBacktest(unittest.TestCase):
    def test_shrunk_win_rate(self) -> None:
        from llm_trading.backtest import shrunk_win_rate

        self.assertEqual(shrunk_win_rate(wins=0, trades=0), 0.0)
        self.assertAlmostEqual(shrunk_win_rate(wins=5, trades=10, prior_strength=0.0), 0.5, places=12)

        # With a strong prior, win-rate should be pulled toward prior_mean.
        raw = 1.0
        shr = shrunk_win_rate(wins=10, trades=10, prior_mean=0.5, prior_strength=20.0)
        self.assertLess(shr, raw)
        self.assertGreater(shr, 0.5)

    def test_forward_holding_backtest_basic(self) -> None:
        from llm_trading.backtest import forward_holding_backtest

        df = _make_weekly([100, 100, 110, 121, 133.1, 146.41])
        sig = [True] + [False] * (len(df) - 1)

        stats, details = forward_holding_backtest(
            df,
            entry_signal=sig,
            horizon_weeks=1,
            buy_cost=0.0,
            sell_cost=0.0,
            non_overlapping=True,
        )

        self.assertEqual(stats.trades, 1)
        self.assertEqual(stats.wins, 1)
        self.assertAlmostEqual(stats.win_rate, 1.0, places=12)
        # Signal at i=0 -> entry at i+1 open=100, exit at i+2 open=110 => +10%
        self.assertAlmostEqual(stats.avg_return, 0.10, places=12)
        self.assertIsInstance(details, dict)
        self.assertEqual(len(details.get("returns") or []), 1)

    def test_forward_holding_backtest_non_overlapping(self) -> None:
        from llm_trading.backtest import forward_holding_backtest

        df = _make_weekly([100, 100, 105, 110, 115, 120, 125, 130])
        sig = [True] * len(df)

        # horizon=2: i=0 uses [1]->[3], then jump i=3 uses [4]->[6] => 2 trades
        stats, _ = forward_holding_backtest(
            df,
            entry_signal=sig,
            horizon_weeks=2,
            buy_cost=0.0,
            sell_cost=0.0,
            non_overlapping=True,
        )
        self.assertEqual(stats.trades, 2)

        # Allow overlap => should produce more trades (as long as window allows)
        stats2, _ = forward_holding_backtest(
            df,
            entry_signal=sig,
            horizon_weeks=2,
            buy_cost=0.0,
            sell_cost=0.0,
            non_overlapping=False,
        )
        self.assertGreater(stats2.trades, stats.trades)

    def test_forward_holding_backtest_signal_length_mismatch(self) -> None:
        from llm_trading.backtest import forward_holding_backtest

        df = _make_weekly([100, 101, 102, 103, 104, 105])
        with self.assertRaises(ValueError):
            forward_holding_backtest(
                df,
                entry_signal=[True, False],  # wrong length
                horizon_weeks=1,
                buy_cost=0.0,
                sell_cost=0.0,
                non_overlapping=True,
            )

    def test_forward_holding_backtest_no_signal_returns_zero_trades(self) -> None:
        from llm_trading.backtest import forward_holding_backtest

        df = _make_weekly([100, 101, 102, 103, 104, 105])
        sig = [False] * len(df)
        stats, sample = forward_holding_backtest(
            df,
            entry_signal=sig,
            horizon_weeks=1,
            buy_cost=0.001,
            sell_cost=0.002,
            non_overlapping=True,
        )
        self.assertEqual(stats.trades, 0)
        self.assertEqual(sample.get("returns"), [])

    def test_forward_holding_backtest_short_history_returns_zero_trades(self) -> None:
        from llm_trading.backtest import forward_holding_backtest

        # n < horizon+3 => early-return branch
        df = _make_weekly([100, 101, 102])
        sig = [True, False, False]
        stats, _ = forward_holding_backtest(
            df,
            entry_signal=sig,
            horizon_weeks=1,
            buy_cost=0.0,
            sell_cost=0.0,
            non_overlapping=True,
        )
        self.assertEqual(stats.trades, 0)

    def test_score_forward_stats_annualized_mode(self) -> None:
        from llm_trading.backtest import ForwardReturnStats, score_forward_stats

        stats = ForwardReturnStats(
            horizon_weeks=4,
            trades=10,
            wins=6,
            win_rate=0.6,
            avg_return=0.05,
            median_return=0.04,
            avg_log_return=None,  # force fallback path
            implied_ann=None,
            gross_wins=6,
            gross_win_rate=0.6,
            gross_avg_return=0.05,
            gross_median_return=0.04,
            gross_avg_log_return=None,
            gross_implied_ann=None,
            avg_mae=-0.08,
            worst_mae=-0.15,
            avg_mfe=0.10,
            best_mfe=0.20,
        )

        s1 = float(score_forward_stats(stats, mode="annualized"))
        s2 = float(score_forward_stats(stats, mode="win_rate"))
        self.assertNotEqual(s1, 0.0)
        self.assertNotEqual(s2, 0.0)


if __name__ == "__main__":
    unittest.main()
