# -*- coding: utf-8 -*-
"""
动量类因子

包含:
- RSIFactor: RSI超买超卖因子
- ROCFactor: 变化率因子
- MomentumFactor: 综合动量因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class RSIFactor(Factor):
    """
    RSI相对强弱指标因子

    测量价格变动的速度和幅度
    """

    name = "rsi"
    category = "momentum"
    description = "RSI：>70超买，<30超卖，50为中性"

    default_params = {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        period = self.params["period"]

        if len(df) < period + 1:
            return FactorResult(
                name=self.name,
                value=50.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # RSI标准化为分数
        # RSI 30-70 映射到 0.3-0.7 (中性区)
        # RSI < 30 映射到 0.7-1.0 (超卖 = 看多机会)
        # RSI > 70 映射到 0.0-0.3 (超买 = 看空风险)
        if current_rsi <= self.params["oversold"]:
            # 超卖区域，分数高（反转机会）
            score = 0.7 + 0.3 * (self.params["oversold"] - current_rsi) / self.params["oversold"]
            direction = "bullish"  # 超卖意味着可能反弹
        elif current_rsi >= self.params["overbought"]:
            # 超买区域，分数低（回调风险）
            score = 0.3 - 0.3 * (current_rsi - self.params["overbought"]) / (100 - self.params["overbought"])
            direction = "bearish"  # 超买意味着可能回调
        else:
            # 中性区域
            score = 0.3 + 0.4 * (current_rsi - self.params["oversold"]) / (self.params["overbought"] - self.params["oversold"])
            direction = "neutral"

        score = max(0.0, min(1.0, score))

        # 置信度基于RSI的极端程度
        extreme_distance = max(
            abs(current_rsi - self.params["oversold"]),
            abs(current_rsi - self.params["overbought"]),
        )
        confidence = min(1.0, extreme_distance / 30)

        return FactorResult(
            name=self.name,
            value=current_rsi,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "rsi": round(current_rsi, 2),
                "zone": "overbought" if current_rsi >= 70 else "oversold" if current_rsi <= 30 else "neutral",
                "prev_rsi": round(rsi.iloc[-2], 2) if len(rsi) >= 2 else None,
            },
        )


@register_factor
class ROCFactor(Factor):
    """
    变化率因子 (Rate of Change)

    测量价格在指定周期内的变化百分比
    """

    name = "roc"
    category = "momentum"
    description = "ROC变化率：正为上涨动能，负为下跌动能"

    default_params = {
        "period": 12,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        period = self.params["period"]

        if len(df) < period + 1:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]
        roc = (close - close.shift(period)) / close.shift(period) * 100

        current_roc = roc.iloc[-1]

        # ROC标准化
        # 用历史ROC的分布来计算分位数
        roc_series = roc.dropna()
        if len(roc_series) > 20:
            percentile = (roc_series < current_roc).mean()
            score = percentile
        else:
            # 简单 sigmoid 映射
            score = 1.0 / (1.0 + np.exp(-current_roc / 5))

        # 方向判断
        if current_roc > 5:
            direction = "bullish"
        elif current_roc < -5:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度基于ROC绝对值
        confidence = min(1.0, abs(current_roc) / 20)

        return FactorResult(
            name=self.name,
            value=current_roc,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "roc": round(current_roc, 2),
                "period": period,
                "prev_close": round(close.iloc[-period - 1], 4) if len(close) > period else None,
            },
        )


@register_factor
class MomentumFactor(Factor):
    """
    综合动量因子

    结合多周期动量信号
    """

    name = "momentum"
    category = "momentum"
    description = "综合动量：结合短中长期动量判断"

    default_params = {
        "short_period": 5,
        "medium_period": 20,
        "long_period": 60,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        short_p = self.params["short_period"]
        medium_p = self.params["medium_period"]
        long_p = self.params["long_period"]

        if len(df) < long_p + 1:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]

        # 计算不同周期的动量
        mom_short = (close.iloc[-1] - close.iloc[-short_p - 1]) / close.iloc[-short_p - 1]
        mom_medium = (close.iloc[-1] - close.iloc[-medium_p - 1]) / close.iloc[-medium_p - 1]
        mom_long = (close.iloc[-1] - close.iloc[-long_p - 1]) / close.iloc[-long_p - 1]

        # 加权综合（短期权重低，长期权重高，避免追高）
        weighted_mom = (
            mom_short * 0.2 +
            mom_medium * 0.3 +
            mom_long * 0.5
        )

        # 判断趋势一致性
        trend_aligned = (
            (mom_short > 0 and mom_medium > 0 and mom_long > 0) or
            (mom_short < 0 and mom_medium < 0 and mom_long < 0)
        )

        # 分数计算
        score = 1.0 / (1.0 + np.exp(-weighted_mom * 10))

        # 方向判断
        if weighted_mom > 0.02 and trend_aligned:
            direction = "bullish"
        elif weighted_mom < -0.02 and trend_aligned:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度基于趋势一致性
        confidence = 0.8 if trend_aligned else 0.4

        return FactorResult(
            name=self.name,
            value=weighted_mom,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "mom_short": round(mom_short * 100, 2),
                "mom_medium": round(mom_medium * 100, 2),
                "mom_long": round(mom_long * 100, 2),
                "weighted_mom": round(weighted_mom * 100, 2),
                "trend_aligned": trend_aligned,
            },
        )
