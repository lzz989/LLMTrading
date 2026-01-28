# -*- coding: utf-8 -*-
"""
因子库模块 - 所有策略的底层基础

设计理念:
- 因子是原子化的市场特征提取器
- 策略 = 因子组合 + 权重配置
- 因子计算与策略逻辑完全解耦

因子分类:
1. 趋势因子 (trend): MA交叉、MACD、ADX等
2. 动量因子 (momentum): RSI、ROC、动量等
3. 量能因子 (volume): 量比、OBV、资金流等
4. 波动因子 (volatility): ATR、布林带、波动率等
5. 形态因子 (pattern): 涨停类型、K线形态等
6. 市场因子 (market): 牛熊状态、市场宽度等
7. 博弈/流动性 proxy (game_theory): 流动性陷阱/止损聚集/情绪极值等
"""

from .base import Factor, FactorResult, FactorRegistry
from .trend import (
    MACrossFactor,
    MACDFactor,
    ADXFactor,
    IchimokuFactor,
)
from .momentum import (
    RSIFactor,
    ROCFactor,
    MomentumFactor,
)
from .volume import (
    VolumeRatioFactor,
    OBVFactor,
    MFIFactor,
)
from .volatility import (
    ATRFactor,
    BollingerFactor,
)
from .pattern import (
    ZTTypeFactor,
    PullbackFactor,
    RewardRiskFactor,
    CandlePatternFactor,
)
from .market import (
    RegimeFactor,
    BreadthFactor,
)
from .game_theory import (
    LiquidityTrapFactor,
    StopClusterFactor,
    CapitulationFactor,
    FomoFactor,
    WyckoffPhaseProxyFactor,
)

__all__ = [
    # Base
    "Factor",
    "FactorResult",
    "FactorRegistry",
    # Trend
    "MACrossFactor",
    "MACDFactor",
    "ADXFactor",
    "IchimokuFactor",
    # Momentum
    "RSIFactor",
    "ROCFactor",
    "MomentumFactor",
    # Volume
    "VolumeRatioFactor",
    "OBVFactor",
    "MFIFactor",
    # Volatility
    "ATRFactor",
    "BollingerFactor",
    # Pattern
    "ZTTypeFactor",
    "PullbackFactor",
    "RewardRiskFactor",
    "CandlePatternFactor",
    # Market
    "RegimeFactor",
    "BreadthFactor",
    # Game theory / liquidity proxy
    "LiquidityTrapFactor",
    "StopClusterFactor",
    "CapitulationFactor",
    "FomoFactor",
    "WyckoffPhaseProxyFactor",
]
