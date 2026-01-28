# -*- coding: utf-8 -*-
"""
形态类因子

包含:
- ZTTypeFactor: 涨停类型因子
- PullbackFactor: 回踩因子
- CandlePatternFactor: K线形态因子
"""

import pandas as pd
import numpy as np
from .base import Factor, FactorResult, register_factor


@register_factor
class ZTTypeFactor(Factor):
    """
    涨停类型因子

    分析涨停板的类型和质量
    """

    name = "zt_type"
    category = "pattern"
    description = "涨停类型：一字板/T字板/换手板/烂板"

    default_params = {
        "zt_threshold_main": 0.095,   # 主板涨停阈值
        "zt_threshold_cy": 0.195,     # 创业板涨停阈值
        "lookback_days": 10,          # 回看天数
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        lookback = self.params["lookback_days"]

        if len(df) < lookback + 1:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_price = df["open"]
        volume = df["volume"]

        # 使用主板阈值作为默认
        zt_thresh = self.params["zt_threshold_main"]

        # 检查最近N天是否有涨停
        returns = close.pct_change()
        zt_mask = returns >= zt_thresh
        recent_zt = zt_mask.iloc[-lookback:].any()

        if not recent_zt:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.3,
                details={
                    "has_zt": False,
                    "zt_type": None,
                    "days_since_zt": None,
                },
            )

        # 找到最近涨停日
        zt_indices = df.index[zt_mask]
        if len(zt_indices) == 0:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.3,
                details={"has_zt": False},
            )

        zt_idx = list(df.index).index(zt_indices[-1])
        days_since_zt = len(df) - zt_idx - 1

        # 涨停日K线特征
        zt_row = df.iloc[zt_idx]
        prev_close = df.iloc[zt_idx - 1]["close"] if zt_idx > 0 else zt_row["open"]

        zt_open = zt_row["open"]
        zt_high = zt_row["high"]
        zt_low = zt_row["low"]
        zt_close = zt_row["close"]

        price_range = zt_high - zt_low
        body = abs(zt_close - zt_open)

        # 涨停类型判断
        if price_range < prev_close * 0.005:
            zt_type = "一字板"
            type_score = 0.6  # 太强势，追涨风险大
        elif zt_low == zt_open and body < price_range * 0.3:
            zt_type = "T字板"
            type_score = 0.7  # 有分歧但收涨停
        elif price_range > prev_close * 0.05:
            zt_type = "烂板"
            type_score = 0.3  # 波动大，不稳定
        else:
            zt_type = "换手板"
            type_score = 0.8  # 健康换手

        # 涨停后天数评分
        if days_since_zt == 1:
            day_score = 0.7
        elif days_since_zt == 2:
            day_score = 0.85  # 最佳买点
        elif days_since_zt == 3:
            day_score = 0.7
        elif days_since_zt <= 5:
            day_score = 0.5
        else:
            day_score = 0.3

        # 综合分数
        score = type_score * 0.5 + day_score * 0.5

        # 方向判断
        if score > 0.65 and days_since_zt <= 3:
            direction = "bullish"
        elif score < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"

        # 置信度
        confidence = 0.7 if days_since_zt <= 3 else 0.4

        return FactorResult(
            name=self.name,
            value=score,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "has_zt": True,
                "zt_type": zt_type,
                "days_since_zt": days_since_zt,
                "type_score": type_score,
                "day_score": day_score,
            },
        )


@register_factor
class PullbackFactor(Factor):
    """
    回踩因子

    检测强势股回踩到支撑位的情况
    """

    name = "pullback"
    category = "pattern"
    description = "回踩：强势股回踩均线支撑"

    default_params = {
        "ma_period": 10,
        "max_pullback_pct": 0.06,  # 最大回踩6%
        "min_prior_gain": 0.08,    # 之前至少涨8%
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        ma_period = self.params["ma_period"]
        max_pullback = self.params["max_pullback_pct"]
        min_prior_gain = self.params["min_prior_gain"]

        if len(df) < ma_period + 10:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        close = df["close"]
        high = df["high"]
        volume = df["volume"]

        # 计算均线
        ma = close.rolling(ma_period).mean()

        current_close = close.iloc[-1]
        current_ma = ma.iloc[-1]

        # 找最近高点
        recent_high = high.iloc[-20:].max()
        high_idx = high.iloc[-20:].idxmax()
        high_pos = list(df.index).index(high_idx)
        current_pos = len(df) - 1
        days_from_high = current_pos - high_pos

        # 从高点回撤幅度
        pullback_pct = (recent_high - current_close) / recent_high

        # 之前涨幅 (从20日前到高点)
        price_20d_ago = close.iloc[-20] if len(close) >= 20 else close.iloc[0]
        prior_gain = (recent_high - price_20d_ago) / price_20d_ago

        # 当前价格相对均线位置
        dist_to_ma = (current_close - current_ma) / current_ma

        # 回踩时缩量
        avg_vol_20 = volume.iloc[-20:-5].mean() if len(volume) >= 20 else volume.mean()
        recent_vol = volume.iloc[-5:].mean()
        vol_shrink = recent_vol < avg_vol_20 * 0.7

        # 评分逻辑
        score = 0.5

        # 1. 有足够的前期涨幅
        if prior_gain >= min_prior_gain:
            score += 0.15
        else:
            score -= 0.1

        # 2. 回撤幅度适中
        if pullback_pct <= max_pullback:
            if pullback_pct >= 0.02:  # 至少有点回撤
                score += 0.2  # 健康回踩
            else:
                score += 0.1  # 回撤太少
        else:
            score -= 0.15  # 回撤过深

        # 3. 接近均线支撑
        if -0.02 <= dist_to_ma <= 0.02:
            score += 0.15  # 贴近均线
        elif dist_to_ma < -0.02:
            score -= 0.1  # 跌破均线

        # 4. 缩量回踩
        if vol_shrink:
            score += 0.1

        score = max(0.0, min(1.0, score))

        # 方向判断
        is_valid_pullback = (
            prior_gain >= min_prior_gain and
            pullback_pct <= max_pullback and
            days_from_high >= 2 and
            days_from_high <= 10
        )

        if is_valid_pullback and score > 0.6:
            direction = "bullish"
        elif pullback_pct > max_pullback * 1.5:
            direction = "bearish"
        else:
            direction = "neutral"

        confidence = 0.7 if is_valid_pullback else 0.4

        return FactorResult(
            name=self.name,
            value=pullback_pct,
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "recent_high": round(recent_high, 4),
                "pullback_pct": round(pullback_pct * 100, 2),
                "prior_gain_pct": round(prior_gain * 100, 2),
                "days_from_high": days_from_high,
                "dist_to_ma_pct": round(dist_to_ma * 100, 2),
                "vol_shrink": vol_shrink,
                "is_valid_pullback": is_valid_pullback,
            },
        )


@register_factor
class RewardRiskFactor(Factor):
    """
    赔率 proxy（高赔率左侧低吸用）。

    目标：把“离支撑近（风险小）+ 离近期高点远（潜在收益大）”量化成 0~1 分。
    - support: 默认用 MA(50) 作为关键支撑（周线≈一年均线）
    - reward: 近期最高价 - 当前价（默认回看 52 根K）
    - risk: 当前价 - 支撑位（<=0 视为跌破支撑，不做多）
    """

    name = "reward_risk"
    category = "pattern"
    description = "赔率：到关键支撑的风险 vs 到近期高点的潜在收益（左侧低吸筛选）"

    default_params = {
        "support_ma_period": 50,
        "lookback_high": 52,
        "rr_good": 2.5,  # >=2.5 视为“赔率还行”
        "rr_excellent": 4.0,  # >=4 视为“赔率很高”
        "max_dist_support_pct": 0.10,  # 离支撑太远就别叫低吸：超过 10% 开始扣分
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        try:
            ma_p = int(self.params["support_ma_period"])
            lb = int(self.params["lookback_high"])
            rr_good = float(self.params["rr_good"])
            rr_ex = float(self.params["rr_excellent"])
            max_dist = float(self.params["max_dist_support_pct"])
        except (TypeError, ValueError, OverflowError, KeyError, AttributeError):  # noqa: BLE001
            ma_p, lb, rr_good, rr_ex, max_dist = 50, 52, 2.5, 4.0, 0.10

        need = {"close"}
        if df is None or (not need.issubset(set(df.columns))) or len(df) < max(ma_p, lb) + 2:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足或缺列"},
            )

        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        high = close
        if "high" in df.columns:
            high = pd.to_numeric(df["high"], errors="coerce").astype(float)

        c = float(close.iloc[-1])
        support = float(close.rolling(ma_p, min_periods=ma_p).mean().iloc[-1])

        # 只用历史窗口的最高价（包含当前K也不构成未来函数；这里只是“目标位”参考，不是回测打分用）
        recent_high = float(high.iloc[-lb:].max())

        details = {
            "support_ma_period": ma_p,
            "lookback_high": lb,
            "close": c,
            "support": support,
            "recent_high": recent_high,
        }

        if not np.isfinite(c) or c <= 0 or (not np.isfinite(support)) or support <= 0:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={**details, "error": "价格无效"},
            )

        risk = float(c - support)
        reward = float(max(0.0, recent_high - c))
        rr = float(reward / risk) if risk > 0 else 0.0

        # 价格离支撑太远：再高的 rr 也容易是“追涨假装低吸”
        dist_pct = float((c / support) - 1.0)
        dist_penalty = 1.0
        if dist_pct > max_dist and max_dist > 0:
            # 线性扣分：dist=max_dist => 1；dist=2*max_dist => 0
            dist_penalty = max(0.0, 1.0 - (dist_pct - max_dist) / max_dist)

        # rr -> score：要求“止损近 + 上方空间大”
        # - risk<=0：跌破支撑，不做多（score=0）
        # - rr<=1：赔率差（0~0.2）
        # - rr>=rr_excellent：封顶 1.0
        if risk <= 0:
            base_score = 0.0
        elif rr <= 1.0:
            base_score = max(0.0, min(0.2, 0.2 * rr))
        elif rr >= rr_ex:
            base_score = 1.0
        else:
            # rr: (1, rr_ex) -> (0.2, 1.0)
            base_score = 0.2 + 0.8 * (rr - 1.0) / max(1e-9, (rr_ex - 1.0))

        score = float(max(0.0, min(1.0, base_score * dist_penalty)))

        # 方向：这里不是趋势方向，而是“赔率是否站得住”
        if risk <= 0:
            direction = "bearish"
        elif rr >= rr_good and score >= 0.6:
            direction = "bullish"
        else:
            direction = "neutral"

        # 置信度：数据长度 + 风险是否真的“近”
        base_conf = min(1.0, float(len(df)) / float(max(ma_p, lb) * 2))
        risk_pct = float(risk / c) if c > 0 else 0.0
        tight_bonus = 0.0
        if risk > 0 and risk_pct <= 0.04:
            tight_bonus = 0.15
        confidence = float(max(0.0, min(1.0, 0.35 + 0.45 * score + 0.20 * base_conf + tight_bonus)))

        details.update(
            {
                "risk": risk,
                "reward": reward,
                "rr": rr,
                "dist_to_support_pct": round(dist_pct * 100, 2),
                "dist_penalty": round(dist_penalty, 4),
                "rr_good": rr_good,
                "rr_excellent": rr_ex,
            }
        )

        return FactorResult(
            name=self.name,
            value=float(rr),
            score=score,
            direction=direction,
            confidence=confidence,
            details=details,
        )


@register_factor
class CandlePatternFactor(Factor):
    """
    K线形态因子

    识别常见K线反转形态
    """

    name = "candle_pattern"
    category = "pattern"
    description = "K线形态：锤子线/吞没/十字星等"

    default_params = {
        "body_ratio_threshold": 0.3,  # 实体占比阈值
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        if len(df) < 3:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        # 最近3根K线
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        patterns_found = []
        score = 0.5
        direction = "neutral"

        # 当前K线特征
        c_open, c_high, c_low, c_close = current["open"], current["high"], current["low"], current["close"]
        c_body = abs(c_close - c_open)
        c_range = c_high - c_low
        c_body_ratio = c_body / c_range if c_range > 0 else 0
        c_bullish = c_close > c_open

        # 前一K线特征
        p_open, p_high, p_low, p_close = prev["open"], prev["high"], prev["low"], prev["close"]
        p_body = abs(p_close - p_open)
        p_bullish = p_close > p_open

        # 1. 锤子线 (下影线长，实体小，上影线短)
        lower_shadow = min(c_open, c_close) - c_low
        upper_shadow = c_high - max(c_open, c_close)
        if c_range > 0:
            if lower_shadow > c_body * 2 and upper_shadow < c_body * 0.5:
                patterns_found.append("锤子线")
                score += 0.15
                direction = "bullish"

        # 2. 倒锤子线
        if c_range > 0:
            if upper_shadow > c_body * 2 and lower_shadow < c_body * 0.5:
                patterns_found.append("倒锤子线")
                score += 0.1

        # 3. 十字星 (实体很小)
        if c_body_ratio < 0.1:
            patterns_found.append("十字星")
            score += 0.05  # 中性信号

        # 4. 看涨吞没
        if not p_bullish and c_bullish:
            if c_close > p_open and c_open < p_close:
                patterns_found.append("看涨吞没")
                score += 0.2
                direction = "bullish"

        # 5. 看跌吞没
        if p_bullish and not c_bullish:
            if c_close < p_open and c_open > p_close:
                patterns_found.append("看跌吞没")
                score -= 0.2
                direction = "bearish"

        # 6. 早晨之星 (三根K线形态)
        p2_bullish = prev2["close"] > prev2["open"]
        p_body_small = p_body < (prev2["high"] - prev2["low"]) * 0.3
        if not p2_bullish and p_body_small and c_bullish:
            if c_close > (prev2["open"] + prev2["close"]) / 2:
                patterns_found.append("早晨之星")
                score += 0.2
                direction = "bullish"

        # 7. 黄昏之星
        if p2_bullish and p_body_small and not c_bullish:
            if c_close < (prev2["open"] + prev2["close"]) / 2:
                patterns_found.append("黄昏之星")
                score -= 0.2
                direction = "bearish"

        score = max(0.0, min(1.0, score))

        # 置信度基于形态数量
        confidence = min(0.8, 0.3 + len(patterns_found) * 0.2)

        return FactorResult(
            name=self.name,
            value=len(patterns_found),
            score=score,
            direction=direction,
            confidence=confidence,
            details={
                "patterns": patterns_found,
                "current_body_ratio": round(c_body_ratio, 3),
                "current_bullish": c_bullish,
            },
        )
