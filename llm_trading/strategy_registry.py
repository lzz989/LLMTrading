from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

StrategyKind = Literal["signal", "filter"]


@dataclass(frozen=True, slots=True)
class StrategySpec:
    key: str
    kind: StrategyKind
    name: str
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]


_REGISTRY: dict[str, StrategySpec] = {}

_TREND_TEMPLATE_PARAMS: dict[str, float | int] = {
    "near_high": 0.25,  # 距离52周高点不超过25%
    "above_low": 0.30,  # 高于52周低点至少30%
    "slope_weeks": 4,  # MA40 上行判断回看周数
}


def register_strategy(*, key: str, kind: StrategyKind, name: str | None = None, description: str = ""):
    """
    注册一个策略（信号 / 过滤器）。
    - signal: 返回 entry_signal（bool 序列），用于回测/排序
    - filter: 返回 mask（bool 序列），用于“环境过滤”（比如趋势模板）
    """

    def deco(func: Callable[[pd.DataFrame], pd.Series]):
        k = str(key).strip()
        if not k:
            raise ValueError("strategy key 不能为空")
        if k in _REGISTRY:
            raise ValueError(f"strategy key 重复：{k}")
        spec = StrategySpec(
            key=k,
            kind=kind,
            name=str(name or k),
            description=str(description or ""),
            compute=func,
        )
        _REGISTRY[k] = spec
        return func

    return deco


def set_trend_template_params(*, near_high: float | None = None, above_low: float | None = None, slope_weeks: int | None = None) -> None:
    # 参数是“比例”，0.25=25%，别传 25 这种憨批值。
    if near_high is not None:
        v = float(near_high)
        if not (0.0 <= v <= 0.95):
            raise ValueError("near_high 必须在 [0, 0.95]，例如 0.25 表示25%")
        _TREND_TEMPLATE_PARAMS["near_high"] = v
    if above_low is not None:
        v = float(above_low)
        if not (0.0 <= v <= 5.0):
            raise ValueError("above_low 必须在 [0, 5.0]，例如 0.30 表示30%")
        _TREND_TEMPLATE_PARAMS["above_low"] = v
    if slope_weeks is not None:
        v = int(slope_weeks)
        if v <= 0 or v > 52:
            raise ValueError("slope_weeks 必须在 [1, 52]")
        _TREND_TEMPLATE_PARAMS["slope_weeks"] = v


def get_trend_template_params() -> dict[str, float | int]:
    return dict(_TREND_TEMPLATE_PARAMS)


def get_strategy(key: str) -> StrategySpec:
    k = str(key).strip()
    if k not in _REGISTRY:
        raise KeyError(f"未知策略：{k}（已注册：{', '.join(sorted(_REGISTRY))}）")
    return _REGISTRY[k]


def list_strategies(*, kind: StrategyKind | None = None) -> list[StrategySpec]:
    if kind is None:
        return [*_REGISTRY.values()]
    return [s for s in _REGISTRY.values() if s.kind == kind]


def parse_strategy_list(text: str | None) -> list[str]:
    """
    解析策略列表（逗号分隔）。
    - None/""/"none" => []
    """
    if text is None:
        return []
    t = str(text).strip()
    if not t or t.lower() == "none":
        return []
    parts = []
    for p in t.split(","):
        p2 = p.strip()
        if not p2:
            continue
        if p2.lower() == "none":
            continue
        parts.append(p2)
    return parts


def compute_series(df: pd.DataFrame, *, key: str) -> pd.Series:
    spec = get_strategy(key)
    s = spec.compute(df)
    if not isinstance(s, pd.Series):
        raise TypeError(f"策略 {key} 返回值不是 pandas.Series")
    if len(s) != len(df):
        raise ValueError(f"策略 {key} 返回序列长度不匹配：{len(s)} != {len(df)}")
    return s.astype(bool)


def combine_masks(df: pd.DataFrame, *, filter_keys: list[str]) -> tuple[pd.Series, dict[str, bool]]:
    """
    合并多个过滤器（AND）。
    返回：
    - mask: bool 序列
    - states: 每个过滤器最后一根的状态
    """
    if not filter_keys:
        mask = pd.Series([True] * len(df), index=df.index)
        return mask, {}

    mask = pd.Series([True] * len(df), index=df.index)
    states: dict[str, bool] = {}
    for k in filter_keys:
        s = compute_series(df, key=k).fillna(False)
        mask = mask & s
        states[k] = bool(s.iloc[-1]) if len(s) else False
    return mask.fillna(False), states


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


@register_strategy(
    key="trend",
    kind="signal",
    name="趋势突破（20周唐奇安）",
    description="close > Donchian(20W)上轨(前移1)（周线；MA200 不做硬过滤）",
)
def signal_trend_breakout_20w(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    close = _num(df["close"])

    ma50 = _num(df["ma50"]) if "ma50" in df.columns else close.rolling(50, min_periods=1).mean()
    ma200 = _num(df["ma200"]) if "ma200" in df.columns else close.rolling(200, min_periods=1).mean()

    if "donchian_upper_20" in df.columns:
        upper = _num(df["donchian_upper_20"])
    else:
        if "high" in df.columns:
            upper = _num(df["high"]).rolling(20, min_periods=1).max().shift(1)
        else:
            upper = close.rolling(20, min_periods=1).max().shift(1)

    # MA200 改成软过滤：别在信号里一刀切
    sig = close > upper
    return sig.fillna(False).astype(bool)


@register_strategy(
    key="swing",
    kind="signal",
    name="回踩站回（周线重上MA50）",
    description="上周close<=上周MA50 且 本周close>本周MA50（周线；MA200 不做硬过滤）",
)
def signal_swing_reclaim_ma50(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    close = _num(df["close"])
    ma50 = _num(df["ma50"]) if "ma50" in df.columns else close.rolling(50, min_periods=1).mean()
    ma200 = _num(df["ma200"]) if "ma200" in df.columns else close.rolling(200, min_periods=1).mean()

    prev_close = close.shift(1)
    prev_ma50 = ma50.shift(1)
    # MA200 改成软过滤：别在信号里一刀切
    sig = (prev_close <= prev_ma50) & (close > ma50)
    return sig.fillna(False).astype(bool)


@register_strategy(
    key="dip",
    kind="signal",
    name="左侧捡漏（回踩MA50不破）",
    description="low触碰MA50附近后收回 + close不远离MA50（周线；MA200 不做硬过滤）",
)
def signal_left_dip_touch_ma50(df: pd.DataFrame) -> pd.Series:
    """
    左侧“捡漏”不是接飞刀：只抓“回踩 MA50 附近/触碰后收回”的位置，
    MA200 改成软过滤（加分项），别在信号里一刀切。
    """
    if "close" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    close = _num(df["close"])
    low = _num(df["low"]) if "low" in df.columns else close

    ma50 = _num(df["ma50"]) if "ma50" in df.columns else close.rolling(50, min_periods=1).mean()
    ma200 = _num(df["ma200"]) if "ma200" in df.columns else close.rolling(200, min_periods=1).mean()

    # 参数先写死（KISS）；后面你要调成 CLI/Web 可配再说。
    touch_pct = 0.02  # 低点触碰 MA50 附近：允许高出 2%
    max_above = 0.07  # 收盘不许离 MA50 太远：允许高出 7%

    touch = low <= (ma50 * (1.0 + float(touch_pct)))
    not_extended = close <= (ma50 * (1.0 + float(max_above)))

    # MA200 改成软过滤：别在信号里一刀切
    sig = touch & (close >= ma50) & not_extended
    return sig.fillna(False).astype(bool)


@register_strategy(
    key="trend_template",
    kind="filter",
    name="趋势模板（周线版 Weinsten/Minervini）",
    description="用更硬的“环境过滤”避免震荡里瞎冲：MA10>MA30>MA40、MA40上行、接近52周高点、远离52周低点",
)
def filter_trend_template_weekly(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    close = _num(df["close"])

    p = get_trend_template_params()
    try:
        near_high_pct = float(p.get("near_high") or 0.25)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        near_high_pct = 0.25
    try:
        above_low_pct = float(p.get("above_low") or 0.30)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        above_low_pct = 0.30
    try:
        slope_weeks = int(p.get("slope_weeks") or 4)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        slope_weeks = 4
    slope_weeks = max(1, min(int(slope_weeks), 52))

    # 周线趋势模板（简化但可回测）
    ma10 = close.rolling(10, min_periods=10).mean()
    ma30 = close.rolling(30, min_periods=30).mean()
    ma40 = close.rolling(40, min_periods=40).mean()

    high52 = close.rolling(52, min_periods=52).max()
    low52 = close.rolling(52, min_periods=52).min()

    # 这个阈值是偏保守的：宁可少，也别乱给“能买”
    near_high = close >= (high52 * (1.0 - near_high_pct))
    above_low = close >= (low52 * (1.0 + above_low_pct))

    ma40_up = ma40 > ma40.shift(slope_weeks)  # 趋势向上（粗暴但够用）

    cond = (
        (close > 0)
        & (ma10 > ma30)
        & (ma30 > ma40)
        & (close > ma10)
        & near_high
        & above_low
        & ma40_up
    )

    # 防止 inf/nan 搞炸后续 JSON
    cond = cond.replace([math.inf, -math.inf], float("nan")).fillna(False)
    return cond.astype(bool)
