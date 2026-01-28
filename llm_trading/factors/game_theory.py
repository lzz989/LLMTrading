# -*- coding: utf-8 -*-
"""
博弈/流动性视角的 proxy 因子（先落地、先验证、别玄学）。

注意：
- 这些因子更偏“过滤器/风险项”，不是买卖按钮。
- score 统一到 [0,1]，这里更接近“现象强度/风险强度”，不是传统 0.5 中性分。
- 必须可复现、可 SQL：details 里可放 zones 之类的列表，但必须同时给 nearest_* 这种扁平字段。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from datetime import date
from typing import Any, Literal

import pandas as pd

from .base import Factor, FactorResult, register_factor


def _sf(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _clip01(x: float | None) -> float:
    if x is None:
        return 0.0
    return float(max(0.0, min(1.0, float(x))))


def _mean_prev_window(s: pd.Series, window: int) -> float | None:
    w = int(window)
    if w <= 0 or len(s) < w + 1:
        return None
    try:
        x = pd.to_numeric(s.iloc[-(w + 1) : -1], errors="coerce").astype(float)
        m = float(x.mean())
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        return None
    if not math.isfinite(m) or m <= 0:
        return None
    return float(m)


def _dyn_price_step(close: float) -> float:
    """
    对“整数/半整数位”的粗糙近似：
    - 低价 ETF（<1）用 0.05 更像“心理位”
    - 1~2 用 0.1
    - 2~10 用 0.5
    - >10 用 1
    """
    c = float(close)
    if c < 1:
        return 0.05
    if c < 2:
        return 0.1
    if c < 10:
        return 0.5
    return 1.0


@register_factor
class LiquidityTrapFactor(Factor):
    """
    流动性陷阱（假突破/假跌破）proxy。

    - bull_trap：向上扫过 swing_high 后收盘回到关键位下方（追涨容易被埋）
    - bear_trap：向下扫过 swing_low 后收盘收回关键位上方（恐慌容易被割）
    """

    name = "liquidity_trap"
    category: Literal["game_theory"] = "game_theory"
    description = "流动性陷阱：假突破/假跌破（扫流动性）"

    default_params = {
        "lookback": 20,
        "sweep_pct": 0.006,
        "vol_window": 20,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        lb = int(self.params["lookback"])
        sweep_pct = float(self.params["sweep_pct"])
        vol_w = int(self.params["vol_window"])

        need = {"high", "low", "close", "volume"}
        if df is None or not need.issubset(set(df.columns)) or len(df) < lb + 2:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足或缺列"},
            )

        hi = _sf(df.iloc[-1].get("high"))
        lo = _sf(df.iloc[-1].get("low"))
        close = _sf(df.iloc[-1].get("close"))
        if hi is None or lo is None or close is None or close <= 0:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "价格无效"},
            )

        # 关键位：只用“上一窗口”数据，避免把当天自己当 swing_high/low
        prev_high = pd.to_numeric(df["high"].iloc[-(lb + 1) : -1], errors="coerce").astype(float)
        prev_low = pd.to_numeric(df["low"].iloc[-(lb + 1) : -1], errors="coerce").astype(float)
        swing_high = _sf(prev_high.max())
        swing_low = _sf(prev_low.min())

        vol_ratio = None
        try:
            v_last = _sf(df.iloc[-1].get("volume"))
            v_mean = _mean_prev_window(df["volume"], vol_w)
            if v_last is not None and v_mean is not None and v_mean > 0:
                vol_ratio = float(v_last / v_mean)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            vol_ratio = None

        amt_ratio = None
        if "amount" in df.columns:
            try:
                a_last = _sf(df.iloc[-1].get("amount"))
                a_mean = _mean_prev_window(df["amount"], vol_w)
                if a_last is not None and a_mean is not None and a_mean > 0:
                    amt_ratio = float(a_last / a_mean)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                amt_ratio = None

        trap_kind: str | None = None
        level: float | None = None
        strength = 0.0
        direction: Literal["bullish", "bearish", "neutral"] = "neutral"

        bull_trap = bool(swing_high is not None and hi > swing_high * (1.0 + sweep_pct) and close < swing_high)
        bear_trap = bool(swing_low is not None and lo < swing_low * (1.0 - sweep_pct) and close > swing_low)

        if bull_trap and swing_high is not None:
            trap_kind = "bull_trap"
            level = float(swing_high)
            strength = float((hi - swing_high) / swing_high) if swing_high > 0 else 0.0
            direction = "bearish"
        elif bear_trap and swing_low is not None:
            trap_kind = "bear_trap"
            level = float(swing_low)
            strength = float((swing_low - lo) / swing_low) if swing_low > 0 else 0.0
            direction = "bullish"

        # 强度归一化：strength 达到 sweep_pct*2 视为很强（再高也 clip）
        base = _clip01(strength / max(1e-9, sweep_pct * 2.0)) if trap_kind else 0.0
        vol_boost = 0.0
        if trap_kind and vol_ratio is not None and vol_ratio > 1.0:
            # 放量更像“扫流动性”而不是随机波动；但没触发 trap 就别瞎加分（不然全是“疑似陷阱”）
            vol_boost = min(0.2, (float(vol_ratio) - 1.0) * 0.1)

        score = _clip01(base * 0.9 + vol_boost)
        confidence = 0.3
        if trap_kind:
            confidence = _clip01(0.55 + 0.35 * base + (0.1 if vol_ratio is not None and vol_ratio > 1.2 else 0.0))

        details = {
            "trap_kind": trap_kind,
            "level": level,
            "sweep_pct": float(sweep_pct),
            "swing_high": swing_high,
            "swing_low": swing_low,
            "high": hi,
            "low": lo,
            "close": close,
            "volume_ratio": vol_ratio,
            "amount_ratio": amt_ratio,
        }

        return FactorResult(
            name=self.name,
            value=float(strength),
            score=float(score),
            direction=direction,
            confidence=float(confidence),
            details=details,
        )


@register_factor
class StopClusterFactor(Factor):
    """
    止损聚集区 proxy：输出“显著位置”和当前距离，给策略做距离约束。
    """

    name = "stop_cluster"
    category: Literal["game_theory"] = "game_theory"
    description = "止损聚集区 proxy：摆动高低/均线/整数位距离"

    default_params = {
        "lookback": 20,
        "ma20": 20,
        "ma60": 60,
        "ma200": 200,
        "score_full_dist_pct": 0.02,  # 距离<=2% 视为“贴近”，score→1；>2% score→0
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        lb = int(self.params["lookback"])
        score_full = float(self.params["score_full_dist_pct"])

        need = {"high", "low", "close"}
        if df is None or not need.issubset(set(df.columns)) or len(df) < lb + 2:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足或缺列"},
            )

        close = _sf(df.iloc[-1].get("close"))
        if close is None or close <= 0:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "close 无效"},
            )

        # swing
        prev_high = pd.to_numeric(df["high"].iloc[-(lb + 1) : -1], errors="coerce").astype(float)
        prev_low = pd.to_numeric(df["low"].iloc[-(lb + 1) : -1], errors="coerce").astype(float)
        swing_high = _sf(prev_high.max())
        swing_low = _sf(prev_low.min())

        def _ma(n: int) -> float | None:
            w = int(n)
            if w <= 1 or len(df) < w:
                return None
            try:
                s = pd.to_numeric(df["close"], errors="coerce").astype(float).rolling(window=w, min_periods=w).mean()
                return _sf(s.iloc[-1])
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                return None

        ma20 = _ma(int(self.params["ma20"]))
        ma60 = _ma(int(self.params["ma60"]))
        ma200 = _ma(int(self.params["ma200"]))

        step = _dyn_price_step(float(close))
        integer_level = round(float(close) / step) * step

        zones: list[dict[str, Any]] = []

        def _add(level: float | None, kind: str) -> None:
            if level is None:
                return
            lv = _sf(level)
            if lv is None or lv <= 0:
                return
            dist = abs(float(close) - float(lv)) / float(close)
            zones.append({"level": float(lv), "kind": str(kind), "distance_pct": float(dist)})

        _add(swing_high, "swing_high")
        _add(swing_low, "swing_low")
        _add(ma20, "ma20")
        _add(ma60, "ma60")
        _add(ma200, "ma200")
        _add(float(integer_level), "integer_level")

        zones.sort(key=lambda x: float(x.get("distance_pct") or 999.0))

        nearest_level = None
        nearest_kind = None
        nearest_dist = None
        if zones:
            z0 = zones[0]
            nearest_level = _sf(z0.get("level"))
            nearest_kind = str(z0.get("kind") or "")
            nearest_dist = _sf(z0.get("distance_pct"))

        # 贴近程度 -> score（越近越高）
        score = 0.0
        if nearest_dist is not None and score_full > 0:
            score = _clip01(1.0 - float(nearest_dist) / float(score_full))

        confidence = 0.6 if zones else 0.0
        if len(zones) >= 4:
            confidence = 0.75

        return FactorResult(
            name=self.name,
            value=float(nearest_dist or 0.0),
            score=float(score),
            direction="neutral",
            confidence=float(confidence),
            details={
                "nearest_level": nearest_level,
                "nearest_kind": nearest_kind,
                "nearest_distance_pct": nearest_dist,
                "ma20": ma20,
                "ma60": ma60,
                "ma200": ma200,
                "swing_high": swing_high,
                "swing_low": swing_low,
                "integer_level": float(integer_level),
                "price_step": float(step),
                "zones": zones[:12],  # 别无限膨胀，SQL 也不好读
            },
        )


@register_factor
class CapitulationFactor(Factor):
    """
    恐慌释放 proxy（capitulation）。
    不是抄底按钮，只表示“可以开始观察反转条件”的窗口。
    """

    name = "capitulation"
    category: Literal["game_theory"] = "game_theory"
    description = "情绪极值：投降式抛售（恐慌释放）"

    default_params = {
        "atr_period": 14,
        "rsi_period": 14,
        "vol_window": 20,
        "move_atr_full": 2.0,  # 下跌 >= 2*ATR 视为很强
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        need = {"high", "low", "close", "volume"}
        if df is None or not need.issubset(set(df.columns)) or len(df) < 3:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足或缺列"},
            )

        # 指标：ATR/RSI
        try:
            from ..indicators import add_atr, add_rsi
        except (AttributeError):  # noqa: BLE001
            add_atr = None  # type: ignore[assignment]
            add_rsi = None  # type: ignore[assignment]

        dfx = df.copy()
        if add_atr is not None:
            dfx = add_atr(dfx, period=int(self.params["atr_period"]), out_col="atr")
        if add_rsi is not None:
            dfx = add_rsi(dfx, period=int(self.params["rsi_period"]), out_col="rsi")

        close = _sf(dfx.iloc[-1].get("close"))
        prev_close = _sf(dfx.iloc[-2].get("close"))
        atr = _sf(dfx.iloc[-1].get("atr"))
        rsi = _sf(dfx.iloc[-1].get("rsi"))

        if close is None or prev_close is None:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "close 无效"},
            )

        down_move = max(0.0, float(prev_close) - float(close))
        move_atr = None
        if atr is not None and atr > 0:
            move_atr = float(down_move / float(atr))

        vol_ratio = None
        try:
            v_last = _sf(dfx.iloc[-1].get("volume"))
            v_mean = _mean_prev_window(dfx["volume"], int(self.params["vol_window"]))
            if v_last is not None and v_mean is not None and v_mean > 0:
                vol_ratio = float(v_last / v_mean)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            vol_ratio = None

        # 子评分：跌幅(ATR倍数) + 放量 + RSI 低位
        mv_full = max(1.0, float(self.params["move_atr_full"]))
        move_score = 0.0
        if move_atr is not None:
            # 1*ATR 开始有点像“异常”，2*ATR 以上算强
            move_score = _clip01((float(move_atr) - 1.0) / (mv_full - 1.0))

        vol_score = 0.0
        if vol_ratio is not None:
            vol_score = _clip01((float(vol_ratio) - 1.0) / 1.5)  # vol_ratio=2.5 -> 1

        rsi_score = 0.0
        if rsi is not None:
            rsi_score = _clip01((35.0 - float(rsi)) / 15.0)  # rsi<=20 -> 1, rsi>=35 -> 0

        score = _clip01(move_score * 0.5 + vol_score * 0.3 + rsi_score * 0.2)
        direction: Literal["bullish", "bearish", "neutral"] = "bullish" if score >= 0.6 else "neutral"
        confidence = _clip01(0.4 + 0.6 * score)

        return FactorResult(
            name=self.name,
            value=float(move_atr or 0.0),
            score=float(score),
            direction=direction,
            confidence=float(confidence),
            details={
                "atr": atr,
                "move_atr": move_atr,
                "volume_ratio": vol_ratio,
                "rsi": rsi,
            },
        )


@register_factor
class FomoFactor(Factor):
    """
    追涨狂热 proxy（fomo）。
    不是做空按钮，只表示“别追/考虑兑现”的风险窗口。
    """

    name = "fomo"
    category: Literal["game_theory"] = "game_theory"
    description = "情绪极值：追涨狂热（冲高风险）"

    default_params = {
        "atr_period": 14,
        "rsi_period": 14,
        "vol_window": 20,
        "move_atr_full": 2.0,  # 上涨 >= 2*ATR 视为很强
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        need = {"high", "low", "close", "volume"}
        if df is None or not need.issubset(set(df.columns)) or len(df) < 3:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足或缺列"},
            )

        try:
            from ..indicators import add_atr, add_rsi
        except (AttributeError):  # noqa: BLE001
            add_atr = None  # type: ignore[assignment]
            add_rsi = None  # type: ignore[assignment]

        dfx = df.copy()
        if add_atr is not None:
            dfx = add_atr(dfx, period=int(self.params["atr_period"]), out_col="atr")
        if add_rsi is not None:
            dfx = add_rsi(dfx, period=int(self.params["rsi_period"]), out_col="rsi")

        close = _sf(dfx.iloc[-1].get("close"))
        prev_close = _sf(dfx.iloc[-2].get("close"))
        atr = _sf(dfx.iloc[-1].get("atr"))
        rsi = _sf(dfx.iloc[-1].get("rsi"))

        if close is None or prev_close is None:
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.0,
                direction="neutral",
                confidence=0.0,
                details={"error": "close 无效"},
            )

        up_move = max(0.0, float(close) - float(prev_close))
        move_atr = None
        if atr is not None and atr > 0:
            move_atr = float(up_move / float(atr))

        vol_ratio = None
        try:
            v_last = _sf(dfx.iloc[-1].get("volume"))
            v_mean = _mean_prev_window(dfx["volume"], int(self.params["vol_window"]))
            if v_last is not None and v_mean is not None and v_mean > 0:
                vol_ratio = float(v_last / v_mean)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            vol_ratio = None

        mv_full = max(1.0, float(self.params["move_atr_full"]))
        move_score = 0.0
        if move_atr is not None:
            move_score = _clip01((float(move_atr) - 1.0) / (mv_full - 1.0))

        vol_score = 0.0
        if vol_ratio is not None:
            vol_score = _clip01((float(vol_ratio) - 1.0) / 1.5)

        rsi_score = 0.0
        if rsi is not None:
            rsi_score = _clip01((float(rsi) - 65.0) / 15.0)  # rsi>=80 -> 1

        score = _clip01(move_score * 0.5 + vol_score * 0.3 + rsi_score * 0.2)
        direction: Literal["bullish", "bearish", "neutral"] = "bearish" if score >= 0.6 else "neutral"
        confidence = _clip01(0.4 + 0.6 * score)

        return FactorResult(
            name=self.name,
            value=float(move_atr or 0.0),
            score=float(score),
            direction=direction,
            confidence=float(confidence),
            details={
                "atr": atr,
                "move_atr": move_atr,
                "volume_ratio": vol_ratio,
                "rsi": rsi,
            },
        )


@register_factor
class WyckoffPhaseProxyFactor(Factor):
    """
    Wyckoff Phase 的“相似度 proxy”（只给分数，不硬分类）。

    输出：
    - accumulation_like / distribution_like（0~1）
    - score：0~1；>0.5 更像吸筹，<0.5 更像派发
    """

    name = "wyckoff_phase_proxy"
    category: Literal["game_theory"] = "game_theory"
    description = "Wyckoff 阶段 proxy：吸筹/派发相似度分数"

    default_params = {
        "lookback": 60,
        "atr_period": 14,
        "obv_ma": 20,
    }

    def compute(self, df: pd.DataFrame) -> FactorResult:
        lb = int(self.params["lookback"])
        if df is None or len(df) < max(30, lb):
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "数据不足"},
            )

        need = {"high", "low", "close", "volume"}
        if not need.issubset(set(df.columns)):
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "缺列"},
            )

        # 取窗口
        win = df.iloc[-lb:].copy()
        close_s = pd.to_numeric(win["close"], errors="coerce").astype(float)
        high_s = pd.to_numeric(win["high"], errors="coerce").astype(float)
        low_s = pd.to_numeric(win["low"], errors="coerce").astype(float)
        vol_s = pd.to_numeric(win["volume"], errors="coerce").fillna(0.0).astype(float)
        if close_s.isna().all():
            return FactorResult(
                name=self.name,
                value=0.0,
                score=0.5,
                direction="neutral",
                confidence=0.0,
                details={"error": "close 全空"},
            )

        last_close = _sf(close_s.iloc[-1]) or 0.0

        # 1) 区间宽度（越窄越像“盘整吸筹/派发”，但不区分方向）
        hi = _sf(high_s.max())
        lo = _sf(low_s.min())
        range_width_pct = None
        if hi is not None and lo is not None and last_close > 0:
            range_width_pct = float((hi - lo) / last_close)
        # 宽度越小，range_score 越高
        range_score = 0.0
        if range_width_pct is not None:
            range_score = _clip01(1.0 - float(range_width_pct) / 0.30)  # 30% 以上视为不“盘整”

        # 2) 波动收缩（ATR 低于历史中位数更像“吸筹/派发阶段”）
        vol_contract_score = 0.0
        atr_last = None
        try:
            from ..indicators import add_atr

            tmp = add_atr(win, period=int(self.params["atr_period"]), out_col="atr")
            atr_s = pd.to_numeric(tmp["atr"], errors="coerce").astype(float)
            atr_last = _sf(atr_s.iloc[-1])
            atr_med = _sf(float(atr_s.median()))
            if atr_last is not None and atr_med is not None and atr_med > 0:
                vol_contract_score = _clip01(1.0 - float(atr_last) / float(atr_med))
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            vol_contract_score = 0.0

        # 3) OBV/A-D proxy（用 OBV 方向当个“确认维度”，先别写背离神功）
        obv_divergence_score = 0.0
        obv_dir = "neutral"
        try:
            from .volume import OBVFactor

            obv = OBVFactor(ma_period=int(self.params["obv_ma"]))
            r = obv.compute(win)
            obv_dir = r.direction
            # divergence 只做一个粗糙 proxy：有背离就给高分
            bd = bool((r.details or {}).get("bullish_divergence")) or bool((r.details or {}).get("bearish_divergence"))
            obv_divergence_score = 1.0 if bd else 0.2
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            obv_divergence_score = 0.0
            obv_dir = "neutral"

        # accumulation / distribution 相似度（共享“盘整/收缩”，区别在 OBV 方向）
        obv_bull = 1.0 if obv_dir == "bullish" else (0.5 if obv_dir == "neutral" else 0.0)
        obv_bear = 1.0 if obv_dir == "bearish" else (0.5 if obv_dir == "neutral" else 0.0)

        accumulation_like = _clip01(range_score * 0.4 + vol_contract_score * 0.4 + obv_bull * 0.2)
        distribution_like = _clip01(range_score * 0.4 + vol_contract_score * 0.4 + obv_bear * 0.2)

        # 按文档口径：0.5 + 0.5*(acc - dist)
        score = _clip01(0.5 + 0.5 * (float(accumulation_like) - float(distribution_like)))

        direction: Literal["bullish", "bearish", "neutral"] = "neutral"
        if score >= 0.55:
            direction = "bullish"
        elif score <= 0.45:
            direction = "bearish"

        confidence = _clip01(0.5 + 0.5 * max(accumulation_like, distribution_like))

        return FactorResult(
            name=self.name,
            value=float(accumulation_like - distribution_like),
            score=float(score),
            direction=direction,
            confidence=float(confidence),
            details={
                "accumulation_like": float(accumulation_like),
                "distribution_like": float(distribution_like),
                "range_width_pct": range_width_pct,
                "vol_contract_score": float(vol_contract_score),
                "obv_divergence_score": float(obv_divergence_score),
                "atr": atr_last,
            },
        )


def compute_game_theory_factor_pack(
    *,
    df: pd.DataFrame,
    symbol: str,
    asset: str,
    as_of: date,
    ref_date: date | None = None,
    source: str = "factors",
) -> dict[str, Any]:
    """
    按 schema 输出一个打包结果（给 analyze/scan 并行输出用）。
    """
    # 注意：这里不做“多因子组合决策”，只负责把 proxy 因子算出来并结构化输出。
    factors: dict[str, dict[str, Any]] = {}
    for cls in (LiquidityTrapFactor, StopClusterFactor, CapitulationFactor, FomoFactor, WyckoffPhaseProxyFactor):
        try:
            r = cls().compute(df)
            factors[str(cls.name)] = r.to_dict()
        except (AttributeError) as exc:  # noqa: BLE001
            factors[str(cls.name)] = {
                "name": str(cls.name),
                "value": 0.0,
                "score": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
                "details": {"error": str(exc)},
            }

    rd = ref_date or as_of
    return {
        "schema": "llm_trading.game_theory_factors.v1",
        "symbol": str(symbol),
        "asset": str(asset),
        "as_of": str(as_of),
        "ref_date": str(rd),
        "source": str(source),
        "factors": factors,
    }
