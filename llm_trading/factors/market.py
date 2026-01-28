# -*- coding: utf-8 -*-
"""
市场类因子

包含:
- RegimeFactor: 市场牛熊状态因子
- BreadthFactor: 市场广度因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class RegimeFactor(Factor):
    """
    市场牛熊状态因子

    判断当前市场的整体状态
    """

    name = "regime"
    category = "market"
    description = "市场状态：bull牛市/bear熊市/neutral震荡"

    default_params = {
        "ma_fast": 50,
        "ma_slow": 200,
        "deep_drawdown_thresh": -0.25,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        ma_fast = self.params["ma_fast"]
        ma_slow = self.params["ma_slow"]

        if len(df) < ma_slow + 1:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足", "regime": "unknown"},
            )

        close = df["close"]

        # 计算均线
        ma50 = close.rolling(ma_fast).mean()
        ma200 = close.rolling(ma_slow).mean()

        current_close = close.iloc[-1]
        current_ma50 = ma50.iloc[-1]
        current_ma200 = ma200.iloc[-1]

        # 计算MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]

        # 252日回撤
        high_252 = close.iloc[-252:].max() if len(close) >= 252 else close.max()
        drawdown_252 = (current_close - high_252) / high_252

        # 市场状态判断
        ma_bullish = current_ma50 > current_ma200
        macd_bullish = current_macd > current_signal
        macd_above_zero = current_macd > 0
        deep_drawdown = drawdown_252 <= self.params["deep_drawdown_thresh"]

        # 综合判断
        bull_signals = sum([ma_bullish, macd_bullish, macd_above_zero])
        bear_signals = sum([not ma_bullish, not macd_bullish, not macd_above_zero, deep_drawdown])

        if deep_drawdown:
            regime = "bear"
            score = 0.2
        elif bull_signals >= 3:
            regime = "bull"
            score = 0.8
        elif bear_signals >= 3:
            regime = "bear"
            score = 0.2
        elif bull_signals >= 2:
            regime = "neutral"
            score = 0.6
        else:
            regime = "neutral"
            score = 0.4

        # 方向
        direction = "bullish" if regime == "bull" else "bearish" if regime == "bear" else "neutral"

        # 置信度
        confidence = max(bull_signals, bear_signals) / 4

        return FactorResult(
            name=self.name,
            value=bull_signals - bear_signals,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "regime": regime,
                "ma50": round(current_ma50, 4),
                "ma200": round(current_ma200, 4),
                "ma_bullish": ma_bullish,
                "macd": round(current_macd, 4),
                "macd_signal": round(current_signal, 4),
                "macd_bullish": macd_bullish,
                "macd_above_zero": macd_above_zero,
                "drawdown_252d": round(drawdown_252 * 100, 2),
                "deep_drawdown": deep_drawdown,
            },
        )


@register_factor
class BreadthFactor(Factor):
    """
    市场广度因子

    基于成交量和涨跌判断市场参与度
    注：此因子需要单标的数据，用于判断个股相对市场的强弱
    """

    name = "breadth"
    category = "market"
    description = "市场广度：成交量和涨跌比例"

    default_params = {
        "volume_ma_period": 20,
        "price_ma_period": 20,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        vol_period = self.params["volume_ma_period"]
        price_period = self.params["price_ma_period"]

        if len(df) < max(vol_period, price_period) + 1:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]
        volume = df["volume"]

        # 成交量相对强度
        vol_ma = volume.rolling(vol_period).mean()
        current_vol = volume.iloc[-1]
        current_vol_ma = vol_ma.iloc[-1]
        vol_ratio = current_vol / current_vol_ma if current_vol_ma > 0 else 1.0

        # 价格相对强度 (相对20日均线)
        price_ma = close.rolling(price_period).mean()
        current_close = close.iloc[-1]
        current_price_ma = price_ma.iloc[-1]
        price_strength = (current_close - current_price_ma) / current_price_ma

        # 涨跌天数统计
        returns = close.pct_change()
        recent_returns = returns.iloc[-20:]
        up_days = (recent_returns > 0).sum()
        down_days = (recent_returns < 0).sum()
        up_down_ratio = up_days / (down_days + 1)

        # 综合评分
        score = 0.5

        # 放量上涨 = 健康
        if vol_ratio > 1.2 and price_strength > 0:
            score += 0.2
        # 缩量下跌 = 可能见底
        elif vol_ratio < 0.8 and price_strength < 0:
            score += 0.1
        # 放量下跌 = 不健康
        elif vol_ratio > 1.2 and price_strength < -0.02:
            score -= 0.15
        # 缩量上涨 = 假突破风险
        elif vol_ratio < 0.8 and price_strength > 0.02:
            score -= 0.05

        # 涨跌比例调整
        if up_down_ratio > 1.5:
            score += 0.1
        elif up_down_ratio < 0.7:
            score -= 0.1

        score = max(0.0, min(1.0, score))

        # 方向判断
        if score > 0.6:
            direction = "bullish"
        elif score < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = min(1.0, abs(vol_ratio - 1.0) * 0.5 + abs(price_strength) * 5)

        return FactorResult(
            name=self.name,
            value=vol_ratio,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "vol_ratio": round(vol_ratio, 2),
                "price_strength_pct": round(price_strength * 100, 2),
                "up_days_20d": int(up_days),
                "down_days_20d": int(down_days),
                "up_down_ratio": round(up_down_ratio, 2),
            },
        )
