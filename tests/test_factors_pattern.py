# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


def _make_base_df(n: int = 30, *, close0: float = 10.0, step: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = close0 + np.arange(n, dtype=float) * float(step)
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


class TestFactorsPattern(unittest.TestCase):
    def test_zt_type_insufficient_data(self) -> None:
        from llm_trading.factors.pattern import ZTTypeFactor

        df = _make_base_df(5)
        out = ZTTypeFactor().compute(df)
        self.assertEqual(out.direction, "neutral")
        self.assertEqual(out.details.get("error"), "数据不足")

    def _zt_df_with_type(self, kind: str) -> pd.DataFrame:
        """
        构造一个最近 lookback 内有涨停的序列：
        - 涨停日放在倒数第3天 => days_since_zt=2（最佳买点分支）
        """
        df = _make_base_df(12, close0=10.0, step=0.0)
        zt_pos = len(df) - 3

        # 防止后面几天把涨停“覆盖掉”导致最近涨停日不是我们想要的
        for i in range(zt_pos + 1, len(df)):
            df.loc[df.index[i], ["open", "high", "low", "close"]] = 11.0

        # 让涨停日收益率 >= 0.095（默认主板阈值）：prev_close=10 -> close=11（+10%）
        df.loc[df.index[zt_pos - 1], "close"] = 10.0
        df.loc[df.index[zt_pos], "close"] = 11.0
        prev_close = 10.0

        if kind == "一字板":
            # range 极小
            df.loc[df.index[zt_pos], ["open", "high", "low", "close"]] = 11.0
        elif kind == "T字板":
            # low==open 且 body 小于 range 的 30%
            df.loc[df.index[zt_pos], "open"] = 10.9
            df.loc[df.index[zt_pos], "low"] = 10.9
            df.loc[df.index[zt_pos], "high"] = 11.3  # range=0.4
            df.loc[df.index[zt_pos], "close"] = 11.0  # body=0.1 < 0.12
        elif kind == "烂板":
            # range 大 + 不满足 T 字板
            df.loc[df.index[zt_pos], "open"] = 10.0
            df.loc[df.index[zt_pos], "low"] = 9.0
            df.loc[df.index[zt_pos], "high"] = 11.0
            df.loc[df.index[zt_pos], "close"] = 11.0
        elif kind == "换手板":
            # range 中等，落入 else
            df.loc[df.index[zt_pos], "open"] = 10.8
            df.loc[df.index[zt_pos], "low"] = 10.6
            df.loc[df.index[zt_pos], "high"] = 11.0  # range=0.4 <= 0.5
            df.loc[df.index[zt_pos], "close"] = 11.0
        else:
            raise ValueError(kind)
        return df

    def test_zt_type_branches(self) -> None:
        from llm_trading.factors.pattern import ZTTypeFactor

        for kind in ["一字板", "T字板", "烂板", "换手板"]:
            df = self._zt_df_with_type(kind)
            out = ZTTypeFactor().compute(df)
            self.assertTrue(out.details.get("has_zt"))
            self.assertEqual(out.details.get("zt_type"), kind)
            self.assertEqual(int(out.details.get("days_since_zt") or -1), 2)

    def test_pullback_factor_bullish_and_bearish(self) -> None:
        from llm_trading.factors.pattern import PullbackFactor

        # Bullish pullback: prior gain enough, pullback small, close near MA, vol shrink
        df = _make_base_df(30, close0=10.0, step=0.05)
        # set a recent high at day -6
        df.loc[df.index[-6], "high"] = 12.0
        df.loc[df.index[-6], "close"] = 12.0
        df.loc[df.index[-6], "open"] = 11.8
        df.loc[df.index[-6], "low"] = 11.7
        # current close slightly below high => ~4% pullback
        df.loc[df.index[-1], "close"] = 11.52
        df.loc[df.index[-1], "open"] = 11.55
        df.loc[df.index[-1], "high"] = 11.60
        df.loc[df.index[-1], "low"] = 11.40
        # vol shrink in last 5
        df.loc[df.index[-20:-5], "volume"] = 1000.0
        df.loc[df.index[-5:], "volume"] = 400.0

        out = PullbackFactor().compute(df)
        self.assertIn(out.direction, {"bullish", "neutral"})  # score gate may vary slightly
        self.assertIsInstance(out.details.get("is_valid_pullback"), bool)

        # Bearish: deep pullback
        df2 = df.copy()
        df2.loc[df2.index[-1], "close"] = 10.7  # >9% pullback from 12
        out2 = PullbackFactor().compute(df2)
        self.assertEqual(out2.direction, "bearish")

    def test_candle_pattern_detects_core_patterns(self) -> None:
        from llm_trading.factors.pattern import CandlePatternFactor

        # Bullish engulfing
        df = pd.DataFrame(
            [
                {"open": 10.0, "high": 10.2, "low": 9.8, "close": 9.9, "volume": 1000},
                {"open": 10.0, "high": 10.1, "low": 9.6, "close": 9.7, "volume": 1000},  # bearish
                {"open": 9.6, "high": 10.4, "low": 9.5, "close": 10.2, "volume": 1200},  # bullish engulfing
            ]
        )
        out = CandlePatternFactor().compute(df)
        self.assertIn("看涨吞没", out.details.get("patterns") or [])
        self.assertEqual(out.direction, "bullish")

        # Bearish engulfing
        df2 = pd.DataFrame(
            [
                {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 1000},
                {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.4, "volume": 1000},  # bullish
                {"open": 10.5, "high": 10.6, "low": 9.7, "close": 9.8, "volume": 1200},  # bearish engulfing
            ]
        )
        out2 = CandlePatternFactor().compute(df2)
        self.assertIn("看跌吞没", out2.details.get("patterns") or [])
        self.assertEqual(out2.direction, "bearish")

    def test_reward_risk_factor_prefers_high_rr_near_support(self) -> None:
        from llm_trading.factors.pattern import RewardRiskFactor

        # 构造：近期有一个高点(20)，当前价贴近 MA50(≈10)，赔率应很高
        df = _make_base_df(60, close0=10.0, step=0.0)
        df.loc[df.index[8], ["open", "high", "low", "close"]] = 20.0  # 在 lookback_high=52 内，但不进 MA50 的窗口
        df.loc[df.index[-1], ["open", "high", "low", "close"]] = 10.05  # 贴近支撑，风险小

        out = RewardRiskFactor().compute(df)
        self.assertGreater(float(out.score), 0.8)
        self.assertIn(out.direction, {"bullish", "neutral"})  # score+rr 足够高时通常 bullish
        self.assertIsInstance(out.details.get("rr"), float)

        # 离支撑太远：应触发“追涨假装低吸”的惩罚
        df2 = df.copy()
        df2.loc[df2.index[-1], ["open", "high", "low", "close"]] = 12.0  # dist_to_support_pct≈20% > 10%
        out2 = RewardRiskFactor().compute(df2)
        self.assertLess(float(out2.score), 0.2)


if __name__ == "__main__":
    unittest.main()
