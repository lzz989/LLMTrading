# -*- coding: utf-8 -*-
"""
因子基类和注册表

设计原则:
1. 每个因子是独立的、可测试的原子单元
2. 因子只负责计算，不负责决策
3. 因子输出标准化为 0-1 或 -1 到 1 的分数
4. 策略 = 因子权重配置文件
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import pandas as pd
import numpy as np


@dataclass
class FactorResult:
    """因子计算结果 - 标准化输出"""

    name: str                           # 因子名称
    value: float                        # 原始值
    score: float                        # 标准化分数 (0-1 或 -1到1)
    direction: Literal["bullish", "bearish", "neutral"]  # 方向判断
    confidence: float                   # 置信度 (0-1)
    details: dict = field(default_factory=dict)  # 详细信息

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "score": self.score,
            "direction": self.direction,
            "confidence": self.confidence,
            "details": self.details,
        }


class Factor(ABC):
    """
    因子抽象基类

    所有因子必须实现:
    1. name: 因子唯一标识
    2. category: 因子分类 (trend/momentum/volume/volatility/pattern/market)
    3. compute(): 计算因子值并返回标准化结果
    """

    # 因子元信息
    name: str = "base_factor"
    category: Literal["trend", "momentum", "volume", "volatility", "pattern", "market", "game_theory"] = "trend"
    description: str = "基础因子"

    # 因子参数 (子类可覆盖)
    default_params: dict = {}

    def __init__(self, **params):
        """
        初始化因子

        Args:
            **params: 因子参数，覆盖默认值
        """
        self.params = {**self.default_params, **params}

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> FactorResult:
        """
        计算因子值

        Args:
            df: 包含 OHLCV 的 DataFrame，必须有以下列:
                - open, high, low, close, volume
                - 日期作为索引

        Returns:
            FactorResult: 标准化的因子结果
        """
        pass

    def normalize_score(
        self,
        value: float,
        method: Literal["minmax", "zscore", "percentile", "boolean"] = "minmax",
        min_val: float = 0.0,
        max_val: float = 1.0,
        history: pd.Series | None = None,
    ) -> float:
        """
        将原始值标准化为 0-1 分数

        Args:
            value: 原始值
            method: 标准化方法
                - minmax: 线性映射到 [0, 1]
                - zscore: z-score 后 sigmoid 压缩
                - percentile: 历史分位数
                - boolean: 布尔转 0/1
            min_val: minmax 方法的最小值
            max_val: minmax 方法的最大值
            history: percentile 方法需要的历史数据

        Returns:
            float: 0-1 之间的标准化分数
        """
        if method == "boolean":
            return 1.0 if value else 0.0

        if method == "minmax":
            if max_val == min_val:
                return 0.5
            return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

        if method == "zscore":
            # z-score 后用 sigmoid 压缩到 (0, 1)
            if history is not None and len(history) > 1:
                mean = history.mean()
                std = history.std()
                if std > 0:
                    z = (value - mean) / std
                    return 1.0 / (1.0 + np.exp(-z))
            return 0.5

        if method == "percentile":
            if history is not None and len(history) > 0:
                return (history < value).mean()
            return 0.5

        return 0.5

    def get_direction(self, score: float, bullish_threshold: float = 0.6, bearish_threshold: float = 0.4) -> str:
        """
        根据分数判断方向

        Args:
            score: 标准化分数 (0-1)
            bullish_threshold: 看多阈值
            bearish_threshold: 看空阈值

        Returns:
            str: "bullish" | "bearish" | "neutral"
        """
        if score >= bullish_threshold:
            return "bullish"
        elif score <= bearish_threshold:
            return "bearish"
        return "neutral"

    def __repr__(self) -> str:
        return f"<Factor: {self.name} ({self.category})>"


class FactorRegistry:
    """
    因子注册表 - 单例模式管理所有因子

    使用方法:
        registry = FactorRegistry()
        registry.register(MACrossFactor)

        # 获取因子
        factor = registry.get("ma_cross")
        result = factor.compute(df)

        # 批量计算
        results = registry.compute_all(df, ["ma_cross", "macd", "rsi"])
    """

    _instance = None
    _factors: dict[str, type[Factor]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, factor_class: type[Factor]) -> None:
        """注册因子类"""
        name = factor_class.name
        if name in self._factors:
            raise ValueError(f"因子 '{name}' 已存在，不能重复注册")
        self._factors[name] = factor_class

    def get(self, name: str, **params) -> Factor:
        """获取因子实例"""
        if name not in self._factors:
            raise KeyError(f"因子 '{name}' 未注册，可用因子: {list(self._factors.keys())}")
        return self._factors[name](**params)

    def list_factors(self, category: str | None = None) -> list[str]:
        """列出所有已注册因子"""
        if category:
            return [name for name, cls in self._factors.items() if cls.category == category]
        return list(self._factors.keys())

    def compute_all(
        self,
        df: pd.DataFrame,
        factor_names: list[str],
        params: dict[str, dict] | None = None,
    ) -> dict[str, FactorResult]:
        """
        批量计算多个因子

        Args:
            df: OHLCV 数据
            factor_names: 要计算的因子名称列表
            params: 每个因子的参数 {"factor_name": {param: value}}

        Returns:
            dict: {factor_name: FactorResult}
        """
        params = params or {}
        results = {}

        for name in factor_names:
            factor_params = params.get(name, {})
            factor = self.get(name, **factor_params)
            try:
                results[name] = factor.compute(df)
            except (  # noqa: BLE001
                AttributeError,
                IndexError,
                KeyError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ) as e:
                # 因子计算失败时返回中性结果
                results[name] = FactorResult(
                    name=name,
                    value=0.0,
                    score=0.5,
                    direction="neutral",
                    confidence=0.0,
                    details={"error": str(e)},
                )

        return results

    def clear(self) -> None:
        """清空注册表 (主要用于测试)"""
        self._factors.clear()


# 全局注册表实例
FACTOR_REGISTRY = FactorRegistry()


def register_factor(factor_class: type[Factor]) -> type[Factor]:
    """
    因子注册装饰器

    使用方法:
        @register_factor
        class MACrossFactor(Factor):
            name = "ma_cross"
            ...
    """
    FACTOR_REGISTRY.register(factor_class)
    return factor_class


@dataclass
class StrategyConfig:
    """
    策略配置 - 定义因子组合和权重

    策略 = 因子权重 + 阈值 + 过滤条件
    """

    name: str                                    # 策略名称
    description: str = ""                        # 策略描述

    # 因子权重配置 {因子名: 权重}
    factor_weights: dict[str, float] = field(default_factory=dict)

    # 因子参数配置 {因子名: {参数名: 值}}
    factor_params: dict[str, dict] = field(default_factory=dict)

    # 信号阈值
    entry_threshold: float = 0.6                 # 入场阈值
    exit_threshold: float = 0.4                  # 出场阈值

    # 过滤条件
    require_factors: list[str] = field(default_factory=list)  # 必须满足的因子 (score > 0.5)
    exclude_factors: list[str] = field(default_factory=list)  # 必须不满足的因子 (score < 0.5)

    # 市场环境限制
    allowed_regimes: list[str] = field(default_factory=lambda: ["bull", "neutral"])

    def validate(self) -> bool:
        """验证配置有效性"""
        # 权重必须为正且和为1
        if not self.factor_weights:
            return False
        if any(w < 0 for w in self.factor_weights.values()):
            return False
        weight_sum = sum(self.factor_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            return False
        return True

    def normalize_weights(self) -> None:
        """归一化权重使其和为1"""
        total = sum(self.factor_weights.values())
        if total > 0:
            self.factor_weights = {k: v / total for k, v in self.factor_weights.items()}


class StrategyEngine:
    """
    策略引擎 - 基于因子配置生成交易信号

    使用方法:
        config = StrategyConfig(
            name="bbb_factor",
            factor_weights={
                "ma_cross": 0.3,
                "macd": 0.25,
                "volume_ratio": 0.2,
                "regime": 0.25,
            },
            entry_threshold=0.65,
        )

        engine = StrategyEngine(config)
        signal = engine.generate_signal(df)
    """

    def __init__(self, config: StrategyConfig, registry: FactorRegistry | None = None):
        self.config = config
        self.registry = registry or FACTOR_REGISTRY

        # 验证配置
        if not config.validate():
            config.normalize_weights()

    def compute_factors(self, df: pd.DataFrame) -> dict[str, FactorResult]:
        """计算策略所需的全部因子（含 require/exclude 的过滤因子）"""
        # 过去的实现只算 factor_weights，导致 require/exclude 引用的因子如果没进权重就不会被计算，
        # 过滤条件形同虚设（非常容易产生误报）。
        # 这里按“权重因子 + 过滤因子”的并集来算，评分仍只用 factor_weights，保证行为可解释且可复现。
        factor_names: list[str] = []
        seen: set[str] = set()
        for n in (
            list(self.config.factor_weights.keys())
            + list(self.config.require_factors or [])
            + list(self.config.exclude_factors or [])
        ):
            k = str(n or "").strip()
            if not k or k in seen:
                continue
            seen.add(k)
            factor_names.append(k)
        return self.registry.compute_all(df, factor_names, self.config.factor_params)

    def compute_composite_score(self, factor_results: dict[str, FactorResult]) -> float:
        """
        计算加权综合分数

        Returns:
            float: 0-1 之间的综合分数
        """
        total_score = 0.0
        total_weight = 0.0

        for factor_name, weight in self.config.factor_weights.items():
            if factor_name in factor_results:
                result = factor_results[factor_name]
                # 权重 × 分数 × 置信度
                total_score += weight * result.score * result.confidence
                total_weight += weight * result.confidence

        if total_weight > 0:
            return total_score / total_weight
        return 0.5

    def check_filters(self, factor_results: dict[str, FactorResult]) -> tuple[bool, str]:
        """
        检查过滤条件

        Returns:
            (pass: bool, reason: str)
        """
        # 必须满足的因子
        for factor_name in self.config.require_factors:
            if factor_name in factor_results:
                if factor_results[factor_name].score <= 0.5:
                    return False, f"必须因子 {factor_name} 未满足"

        # 必须不满足的因子
        for factor_name in self.config.exclude_factors:
            if factor_name in factor_results:
                if factor_results[factor_name].score > 0.5:
                    return False, f"排除因子 {factor_name} 触发"

        return True, ""

    def generate_signal(
        self,
        df: pd.DataFrame,
        market_regime: str = "neutral",
    ) -> dict:
        """
        生成交易信号

        Args:
            df: OHLCV 数据
            market_regime: 当前市场状态 ("bull", "bear", "neutral")

        Returns:
            dict: {
                "action": "entry" | "exit" | "hold",
                "score": float,
                "confidence": float,
                "factors": dict[str, FactorResult],
                "reason": str,
            }
        """
        result = {
            "action": "hold",
            "score": 0.5,
            "confidence": 0.0,
            "factors": {},
            "reason": "",
        }

        # 检查市场环境
        if market_regime not in self.config.allowed_regimes:
            result["reason"] = f"市场环境 {market_regime} 不在允许范围 {self.config.allowed_regimes}"
            return result

        # 计算因子
        factor_results = self.compute_factors(df)
        result["factors"] = {k: v.to_dict() for k, v in factor_results.items()}

        # 检查过滤条件
        filter_pass, filter_reason = self.check_filters(factor_results)
        if not filter_pass:
            result["reason"] = filter_reason
            return result

        # 计算综合分数
        composite_score = self.compute_composite_score(factor_results)
        result["score"] = composite_score

        # 计算综合置信度 (因子置信度的加权平均)
        total_conf = sum(
            self.config.factor_weights.get(k, 0) * v.confidence
            for k, v in factor_results.items()
        )
        result["confidence"] = min(1.0, total_conf)

        # 判断信号
        if composite_score >= self.config.entry_threshold:
            result["action"] = "entry"
            result["reason"] = f"综合分数 {composite_score:.3f} >= 入场阈值 {self.config.entry_threshold}"
        elif composite_score <= self.config.exit_threshold:
            result["action"] = "exit"
            result["reason"] = f"综合分数 {composite_score:.3f} <= 出场阈值 {self.config.exit_threshold}"
        else:
            result["action"] = "hold"
            result["reason"] = f"综合分数 {composite_score:.3f} 在阈值区间内"

        return result

    def explain_signal(self, signal: dict) -> str:
        """
        生成信号解释文本

        Returns:
            str: 人类可读的信号解释
        """
        lines = [
            f"策略: {self.config.name}",
            f"信号: {signal['action'].upper()}",
            f"综合分数: {signal['score']:.3f}",
            f"置信度: {signal['confidence']:.3f}",
            f"原因: {signal['reason']}",
            "",
            "因子明细:",
        ]

        for factor_name, factor_data in signal.get("factors", {}).items():
            weight = self.config.factor_weights.get(factor_name, 0)
            lines.append(
                f"  {factor_name}: "
                f"分数={factor_data['score']:.3f}, "
                f"方向={factor_data['direction']}, "
                f"权重={weight:.2f}"
            )

        return "\n".join(lines)
