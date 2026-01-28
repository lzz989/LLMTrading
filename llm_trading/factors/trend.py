# -*- coding: utf-8 -*-
"""
趋势类因子

包含:
- MACrossFactor: 均线交叉因子
- MACDFactor: MACD因子
- ADXFactor: ADX趋势强度因子
- IchimokuFactor: 一目均衡表因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class MACrossFactor(Factor):
    """
    均线交叉因子

    计算快慢均线的相对位置和交叉状态
    """

    name = "ma_cross"
    category = "trend"
    description = "均线交叉：快线在慢线上方为多头，下方为空头"

    default_params = {
        "fast_period": 20,
        "slow_period": 50,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]

        if len(df) < slow_period:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": f"数据不足 {slow_period} 根K线"},
            )

        close = df["close"]
        ma_fast = close.rolling(fast_period).mean()
        ma_slow = close.rolling(slow_period).mean()

        # 当前状态
        current_fast = ma_fast.iloc[-1]
        current_slow = ma_slow.iloc[-1]
        current_close = close.iloc[-1]

        # 计算距离百分比
        dist_pct = (current_fast - current_slow) / current_slow if current_slow > 0 else 0.0

        # 判断交叉
        prev_fast = ma_fast.iloc[-2] if len(ma_fast) >= 2 else current_fast
        prev_slow = ma_slow.iloc[-2] if len(ma_slow) >= 2 else current_slow

        golden_cross = prev_fast <= prev_slow and current_fast > current_slow
        death_cross = prev_fast >= prev_slow and current_fast < current_slow

        # 标准化分数: dist_pct 映射到 0-1
        # dist_pct 正数表示多头，负数表示空头
        # 用 sigmoid 压缩到 (0, 1)
        score = 1.0 / (1.0 + np.exp(-dist_pct * 20))

        # 判断方向
        if current_fast > current_slow and current_close > ma_fast.iloc[-1]:
            direction = "bullish"
        elif current_fast < current_slow and current_close < ma_fast.iloc[-1]:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度: 基于距离和数据量
        confidence = min(1.0, len(df) / (slow_period * 2)) * min(1.0, abs(dist_pct) * 10 + 0.5)

        return FactorResult(
            name=self.name,
            value=dist_pct,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "ma_fast": round(current_fast, 4),
                "ma_slow": round(current_slow, 4),
                "dist_pct": round(dist_pct, 4),
                "golden_cross": golden_cross,
                "death_cross": death_cross,
                "close_above_fast": current_close > current_fast,
            },
        )


@register_factor
class MACDFactor(Factor):
    """
    MACD因子

    计算MACD的方向、强度和柱状图
    """

    name = "macd"
    category = "trend"
    description = "MACD：金叉为多，死叉为空，柱状图强度"

    default_params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        signal = self.params["signal_period"]

        if len(df) < slow + signal:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]

        # 计算MACD
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]

        # 判断金叉/死叉
        prev_macd = macd_line.iloc[-2] if len(macd_line) >= 2 else current_macd
        prev_signal = signal_line.iloc[-2] if len(signal_line) >= 2 else current_signal

        golden_cross = prev_macd <= prev_signal and current_macd > current_signal
        death_cross = prev_macd >= prev_signal and current_macd < current_signal

        # 柱状图方向变化
        prev_hist = histogram.iloc[-2] if len(histogram) >= 2 else current_hist
        hist_expanding = abs(current_hist) > abs(prev_hist)
        hist_positive = current_hist > 0

        # 计算分数
        # 1. MACD在信号线上方 (+0.4)
        # 2. MACD在零轴上方 (+0.3)
        # 3. 柱状图为正且扩张 (+0.3)
        score = 0.5  # 基准
        if current_macd > current_signal:
            score += 0.2
        else:
            score -= 0.2

        if current_macd > 0:
            score += 0.15
        else:
            score -= 0.15

        if hist_positive and hist_expanding:
            score += 0.15
        elif not hist_positive and hist_expanding:
            score -= 0.15

        score = max(0.0, min(1.0, score))

        # 方向判断
        if current_macd > current_signal and current_macd > 0:
            direction = "bullish"
        elif current_macd < current_signal and current_macd < 0:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度基于MACD绝对值
        hist_std = histogram.std() if len(histogram) > 20 else abs(current_hist)
        confidence = min(1.0, abs(current_hist) / hist_std if hist_std > 0 else 0.5)

        return FactorResult(
            name=self.name,
            value=current_hist,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "macd": round(current_macd, 4),
                "signal": round(current_signal, 4),
                "histogram": round(current_hist, 4),
                "golden_cross": golden_cross,
                "death_cross": death_cross,
                "above_zero": current_macd > 0,
                "hist_expanding": hist_expanding,
            },
        )


@register_factor
class ADXFactor(Factor):
    """
    ADX趋势强度因子

    测量趋势强度，不区分方向
    """

    name = "adx"
    category = "trend"
    description = "ADX趋势强度：>25为强趋势，<20为震荡"

    default_params = {
        "period": 14,
        "strong_trend_threshold": 25,
        "weak_trend_threshold": 20,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        period = self.params["period"]

        if len(df) < period * 2:
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
        atr = tr.rolling(period).mean()

        # 计算+DM和-DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # 平滑
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # 趋势强度分数
        # ADX 0-50 映射到 0-1
        strength_score = min(1.0, current_adx / 50)

        # 方向由 +DI vs -DI 决定
        if current_plus_di > current_minus_di and current_adx > self.params["strong_trend_threshold"]:
            direction = "bullish"
            score = 0.5 + strength_score * 0.5  # 强上升趋势
        elif current_minus_di > current_plus_di and current_adx > self.params["strong_trend_threshold"]:
            direction = "bearish"
            score = 0.5 - strength_score * 0.5  # 强下降趋势
        else:
            direction = "neutral"
            score = 0.5

        # 置信度基于ADX值
        confidence = min(1.0, current_adx / 40)

        return FactorResult(
            name=self.name,
            value=current_adx,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "adx": round(current_adx, 2),
                "plus_di": round(current_plus_di, 2),
                "minus_di": round(current_minus_di, 2),
                "trend_strength": "strong" if current_adx > 25 else "weak" if current_adx < 20 else "moderate",
            },
        )


@register_factor
class IchimokuFactor(Factor):
    """
    一目均衡表因子

    综合判断趋势、支撑阻力
    """

    name = "ichimoku"
    category = "trend"
    description = "一目均衡表：价格与云图的相对位置"

    default_params = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        tenkan_p = self.params["tenkan_period"]
        kijun_p = self.params["kijun_period"]
        senkou_b_p = self.params["senkou_b_period"]

        if len(df) < senkou_b_p + kijun_p:
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

        # 转换线 (Tenkan-sen)
        tenkan = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2

        # 基准线 (Kijun-sen)
        kijun = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2

        # 先行带A (Senkou Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)

        # 先行带B (Senkou Span B)
        senkou_b = ((high.rolling(senkou_b_p).max() + low.rolling(senkou_b_p).min()) / 2).shift(kijun_p)

        current_close = close.iloc[-1]
        current_tenkan = tenkan.iloc[-1]
        current_kijun = kijun.iloc[-1]
        current_senkou_a = senkou_a.iloc[-1]
        current_senkou_b = senkou_b.iloc[-1]

        # 云顶和云底
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)

        # 计算分数
        score = 0.5
        signals = []

        # 1. 价格相对于云
        if current_close > cloud_top:
            score += 0.2
            signals.append("价格在云上方")
        elif current_close < cloud_bottom:
            score -= 0.2
            signals.append("价格在云下方")
        else:
            signals.append("价格在云中")

        # 2. 转换线与基准线
        if current_tenkan > current_kijun:
            score += 0.15
            signals.append("转换线在基准线上方")
        else:
            score -= 0.15
            signals.append("转换线在基准线下方")

        # 3. 云的颜色 (A > B = 绿云/多头)
        if current_senkou_a > current_senkou_b:
            score += 0.1
            signals.append("云为多头云")
        else:
            score -= 0.1
            signals.append("云为空头云")

        # 4. 价格与基准线
        if current_close > current_kijun:
            score += 0.05
        else:
            score -= 0.05

        score = max(0.0, min(1.0, score))

        # 方向判断
        if current_close > cloud_top and current_tenkan > current_kijun:
            direction = "bullish"
        elif current_close < cloud_bottom and current_tenkan < current_kijun:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度基于各信号的一致性
        bullish_signals = sum([
            current_close > cloud_top,
            current_tenkan > current_kijun,
            current_senkou_a > current_senkou_b,
            current_close > current_kijun,
        ])
        confidence = bullish_signals / 4 if direction == "bullish" else (4 - bullish_signals) / 4 if direction == "bearish" else 0.5

        return FactorResult(
            name=self.name,
            value=current_close - cloud_top if current_close > cloud_top else current_close - cloud_bottom,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "tenkan": round(current_tenkan, 4),
                "kijun": round(current_kijun, 4),
                "senkou_a": round(current_senkou_a, 4),
                "senkou_b": round(current_senkou_b, 4),
                "cloud_top": round(cloud_top, 4),
                "cloud_bottom": round(cloud_bottom, 4),
                "price_vs_cloud": "above" if current_close > cloud_top else "below" if current_close < cloud_bottom else "inside",
                "signals": signals,
            },
        )
