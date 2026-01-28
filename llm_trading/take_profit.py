from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


BullPhase = Literal["hot", "slow"]


@dataclass(frozen=True, slots=True)
class TakeProfitConfig:
    """
    止盈配置（研究用途）。

    约定（你选的风格）：
    - hot bull（疯牛）：不分批，让利润跑（只靠周线锚/硬风控离场）
    - slow bull（慢牛）：只做一次分批（卖 1/3），其余继续让利润跑
    """

    hot_bull_mom_63d: float = 0.25  # 指数近63日动量 >= 25% 视为“疯牛”
    slow_bull_tp1_trigger_ret: float = 0.20  # 慢牛分批触发：浮盈>=20%
    slow_bull_tp1_sell_ratio: float = 1.0 / 3.0  # 慢牛只卖一次：卖 1/3

    # 回撤止盈（盈利保护）：避免 07/15 这种“跌起来一刀把你砍懵”的行情把利润全吐回去。
    # 逻辑：当浮盈 >= min_profit_ret 后，启用 “近 N 日最高收盘价 * (1-dd_pct)” 的保护线；
    #      收盘跌破则视为“该跑了”，次日执行（研究用途）。
    profit_stop_lookback_days: int = 252  # 近一年（交易日）窗口
    profit_stop_min_profit_ret: float = 0.20  # 只有浮盈>=20% 才开启回撤止盈（更少抖飞）
    profit_stop_dd_pct_hot_bull: float = 0.10  # 疯牛：更快保护（默认 10% 回撤）
    profit_stop_dd_pct_slow_bull: float = 0.12  # 慢牛：稍松一点（默认 12% 回撤）
    profit_stop_enabled: bool = True  # 允许关闭（比如你嫌它抖飞）


def classify_bull_phase(*, label: str, mom_63d: float | None, cfg: TakeProfitConfig | None = None) -> BullPhase | None:
    cfg2 = cfg or TakeProfitConfig()
    lb = str(label or "").strip().lower()
    if lb != "bull":
        return None
    m = None if mom_63d is None else float(mom_63d)
    if m is not None and math.isfinite(m) and m >= float(cfg2.hot_bull_mom_63d):
        return "hot"
    return "slow"


def calc_tp1_sell_shares(*, shares: int, lot_size: int, cfg: TakeProfitConfig | None = None) -> int:
    """
    计算慢牛分批卖出的股数（按手数向下取整）。
    - 默认卖 1/3
    - 至少留一手（否则别卖）
    """
    cfg2 = cfg or TakeProfitConfig()
    sh = int(shares)
    lot = max(1, int(lot_size))
    if sh <= 0:
        return 0

    raw = int(math.floor(float(sh) * float(cfg2.slow_bull_tp1_sell_ratio)))
    sell = (raw // lot) * lot
    if sell <= 0:
        return 0
    # 至少留一手
    if sh - sell < lot:
        return 0
    return int(sell)
