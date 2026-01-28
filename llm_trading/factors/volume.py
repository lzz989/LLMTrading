# -*- coding: utf-8 -*-
"""
量能类因子

包含:
- VolumeRatioFactor: 量比因子
- OBVFactor: 能量潮因子
- MFIFactor: 资金流量指标因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class VolumeRatioFactor(Factor):
    """
    量比因子

    当日成交量与过去平均成交量的比值
    """

    name = "volume_ratio"
    category = "volume"
    description = "量比：>2为放量，<0.5为缩量"

    default_params = {
        "period": 20,
        "high_threshold": 2.0,
        "low_threshold": 0.5,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        period = self.params["period"]

        if len(df) < period + 1:
            return FactorResult(
                name=self.name,
                value=1.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        volume = df["volume"]
        close = df["close"]

        # 计算量比
        avg_volume = volume.rolling(period).mean()
        current_volume = volume.iloc[-1]
        current_avg = avg_volume.iloc[-1]

        volume_ratio = current_volume / current_avg if current_avg > 0 else 1.0

        # 判断价格方向
        price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0

        # 量价配合分析
        # 放量上涨 = bullish
        # 放量下跌 = bearish
        # 缩量上涨 = neutral (可能假突破)
        # 缩量下跌 = neutral (可能企稳)

        high_vol = volume_ratio >= self.params["high_threshold"]
        low_vol = volume_ratio <= self.params["low_threshold"]

        if high_vol and price_change > 0.01:
            direction = "bullish"
            score = min(1.0, 0.6 + volume_ratio * 0.1)
        elif high_vol and price_change < -0.01:
            direction = "bearish"
            score = max(0.0, 0.4 - volume_ratio * 0.1)
        elif low_vol and price_change > 0:
            direction = "neutral"
            score = 0.55  # 缩量上涨，略偏多
        elif low_vol and price_change < 0:
            direction = "neutral"
            score = 0.45  # 缩量下跌，略偏空
        else:
            direction = "neutral"
            score = 0.5

        # 置信度基于量比的极端程度
        confidence = min(1.0, abs(volume_ratio - 1.0) / 2)

        return FactorResult(
            name=self.name,
            value=volume_ratio,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "volume_ratio": round(volume_ratio, 2),
                "current_volume": int(current_volume),
                "avg_volume": int(current_avg),
                "price_change_pct": round(price_change * 100, 2),
                "volume_signal": "放量" if high_vol else "缩量" if low_vol else "正常",
            },
        )


@register_factor
class OBVFactor(Factor):
    """
    能量潮因子 (On Balance Volume)

    累积成交量判断资金流向
    """

    name = "obv"
    category = "volume"
    description = "OBV能量潮：趋势确认和背离信号"

    default_params = {
        "ma_period": 20,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        ma_period = self.params["ma_period"]

        if len(df) < ma_period + 1:
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

        # 计算OBV
        obv = pd.Series(0.0, index=df.index)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # OBV移动平均
        obv_ma = obv.rolling(ma_period).mean()

        current_obv = obv.iloc[-1]
        current_obv_ma = obv_ma.iloc[-1]

        # 价格趋势
        price_trend = close.iloc[-1] > close.iloc[-ma_period] if len(close) > ma_period else True

        # OBV趋势
        obv_trend = current_obv > current_obv_ma

        # 背离检测
        price_new_high = close.iloc[-1] >= close.rolling(ma_period).max().iloc[-1]
        obv_new_high = current_obv >= obv.rolling(ma_period).max().iloc[-1]

        bearish_divergence = price_new_high and not obv_new_high
        bullish_divergence = not price_new_high and obv_new_high

        # 分数计算
        score = 0.5
        if obv_trend:
            score += 0.2
        else:
            score -= 0.2

        if bullish_divergence:
            score += 0.15
            direction = "bullish"
        elif bearish_divergence:
            score -= 0.15
            direction = "bearish"
        elif obv_trend and price_trend:
            direction = "bullish"
        elif not obv_trend and not price_trend:
            direction = "bearish"
        else:
            direction = "neutral"

        score = max(0.0, min(1.0, score))

        # 置信度
        confidence = 0.7 if (obv_trend == price_trend) else 0.5

        return FactorResult(
            name=self.name,
            value=current_obv,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "obv": int(current_obv),
                "obv_ma": int(current_obv_ma),
                "obv_above_ma": obv_trend,
                "bearish_divergence": bearish_divergence,
                "bullish_divergence": bullish_divergence,
            },
        )


@register_factor
class MFIFactor(Factor):
    """
    资金流量指标因子 (Money Flow Index)

    结合价格和成交量的RSI变体
    """

    name = "mfi"
    category = "volume"
    description = "MFI资金流：>80超买，<20超卖"

    default_params = {
        "period": 14,
        "overbought": 80,
        "oversold": 20,
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

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # 典型价格
        typical_price = (high + low + close) / 3

        # 原始资金流
        raw_money_flow = typical_price * volume

        # 正负资金流
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                negative_flow.iloc[i] = raw_money_flow.iloc[i]

        # 资金流比率
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        money_ratio = positive_sum / (negative_sum + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))

        current_mfi = mfi.iloc[-1]

        # MFI标准化 (类似RSI逻辑)
        if current_mfi <= self.params["oversold"]:
            score = 0.7 + 0.3 * (self.params["oversold"] - current_mfi) / self.params["oversold"]
            direction = "bullish"
        elif current_mfi >= self.params["overbought"]:
            score = 0.3 - 0.3 * (current_mfi - self.params["overbought"]) / (100 - self.params["overbought"])
            direction = "bearish"
        else:
            score = 0.3 + 0.4 * (current_mfi - self.params["oversold"]) / (self.params["overbought"] - self.params["oversold"])
            direction = "neutral"

        score = max(0.0, min(1.0, score))

        confidence = min(1.0, abs(current_mfi - 50) / 30)

        return FactorResult(
            name=self.name,
            value=current_mfi,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "mfi": round(current_mfi, 2),
                "zone": "overbought" if current_mfi >= 80 else "oversold" if current_mfi <= 20 else "neutral",
            },
        )
