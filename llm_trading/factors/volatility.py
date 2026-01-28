# -*- coding: utf-8 -*-
"""
波动类因子

包含:
- ATRFactor: 真实波动幅度因子
- BollingerFactor: 布林带因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class ATRFactor(Factor):
    """
    ATR真实波动幅度因子

    测量市场波动性
    """

    name = "atr"
    category = "volatility"
    description = "ATR波动率：高ATR表示波动大，风险高"

    default_params = {
        "period": 14,
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

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 计算True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        atr = tr.rolling(period).mean()

        current_atr = atr.iloc[-1]
        current_close = close.iloc[-1]

        # ATR百分比
        atr_pct = current_atr / current_close if current_close > 0 else 0

        # 历史ATR分位数
        atr_history = atr.dropna()
        if len(atr_history) > 20:
            atr_percentile = (atr_history < current_atr).mean()
        else:
            atr_percentile = 0.5

        # 分数计算
        # 低波动率 = 高分 (稳定)
        # 高波动率 = 低分 (风险)
        score = 1.0 - atr_percentile

        # 方向判断 (波动率本身不判断方向)
        direction = "neutral"

        # 置信度
        confidence = min(1.0, len(atr_history) / 100)

        return FactorResult(
            name=self.name,
            value=current_atr,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "atr": round(current_atr, 4),
                "atr_pct": round(atr_pct * 100, 2),
                "atr_percentile": round(atr_percentile * 100, 1),
                "volatility_level": "high" if atr_percentile > 0.8 else "low" if atr_percentile < 0.2 else "normal",
            },
        )


@register_factor
class BollingerFactor(Factor):
    """
    布林带因子

    测量价格相对于波动通道的位置
    """

    name = "bollinger"
    category = "volatility"
    description = "布林带：价格在上轨超买，下轨超卖"

    default_params = {
        "period": 20,
        "std_dev": 2.0,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        period = self.params["period"]
        std_dev = self.params["std_dev"]

        if len(df) < period:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]

        # 布林带计算
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        current_close = close.iloc[-1]
        current_middle = middle.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        # %B 指标 (价格在布林带中的位置)
        band_width = current_upper - current_lower
        if band_width > 0:
            percent_b = (current_close - current_lower) / band_width
        else:
            percent_b = 0.5

        # 带宽百分比
        bandwidth_pct = band_width / current_middle if current_middle > 0 else 0

        # 分数计算
        # %B 接近 0.5 最好 (中轨附近)
        # %B > 1 超买，%B < 0 超卖
        if percent_b <= 0:
            score = 0.8  # 超卖，可能反弹
            direction = "bullish"
        elif percent_b >= 1:
            score = 0.2  # 超买，可能回调
            direction = "bearish"
        elif percent_b < 0.5:
            score = 0.5 + 0.3 * (0.5 - percent_b)  # 下半区，略看多
            direction = "neutral" if percent_b > 0.2 else "bullish"
        else:
            score = 0.5 - 0.3 * (percent_b - 0.5)  # 上半区，略看空
            direction = "neutral" if percent_b < 0.8 else "bearish"

        # 置信度基于带宽
        confidence = min(1.0, abs(percent_b - 0.5) * 2)

        return FactorResult(
            name=self.name,
            value=percent_b,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "upper": round(current_upper, 4),
                "middle": round(current_middle, 4),
                "lower": round(current_lower, 4),
                "percent_b": round(percent_b, 3),
                "bandwidth_pct": round(bandwidth_pct * 100, 2),
                "position": "above_upper" if percent_b > 1 else "below_lower" if percent_b < 0 else "inside",
            },
        )
