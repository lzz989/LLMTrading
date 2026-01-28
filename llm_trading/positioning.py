from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal


StopMode = Literal["weekly_entry_ma", "daily_ma20", "atr"]
RegimeLabel = Literal["bull", "bear", "neutral", "unknown"]


@dataclass(frozen=True, slots=True)
class RiskProfile:
    label: RegimeLabel
    max_exposure_pct: float
    risk_per_trade_pct: float
    stop_mode: StopMode
    max_positions: int


def risk_profile_for_regime(label: str | None) -> RiskProfile:
    """
    给小资金 + 有磨损的 ETF 做一个“别把自己磨死”的默认仓位风控配置。

    约定：
    - bull：可以更激进一点（多开仓、止损给空间）
    - neutral：中等（别上头）
    - bear：谨慎（少开仓、止损更紧）
    """
    lb = str(label or "unknown").strip().lower()
    if lb == "bull":
        # “单笔预期仓位 * 5%”换算成“占总资金的比例”：
        # 预期仓位≈capital*0.90/2 => risk_pct≈0.90/2*0.05=0.0225
        return RiskProfile(label="bull", max_exposure_pct=0.90, risk_per_trade_pct=0.0225, stop_mode="weekly_entry_ma", max_positions=2)
    if lb == "bear":
        # 预期仓位≈capital*0.30/1 => risk_pct≈0.30*0.05=0.015
        return RiskProfile(label="bear", max_exposure_pct=0.30, risk_per_trade_pct=0.0150, stop_mode="daily_ma20", max_positions=1)
    if lb == "neutral":
        # 预期仓位≈capital*0.60/2 => risk_pct≈0.60/2*0.05=0.015
        return RiskProfile(label="neutral", max_exposure_pct=0.60, risk_per_trade_pct=0.0150, stop_mode="weekly_entry_ma", max_positions=2)
    # unknown：当中性偏保守处理（别硬上）
    return RiskProfile(label="unknown", max_exposure_pct=0.50, risk_per_trade_pct=0.0150, stop_mode="weekly_entry_ma", max_positions=1)


@dataclass(frozen=True, slots=True)
class PositionPlanParams:
    capital_yuan: float
    roundtrip_cost_yuan: float
    lot_size: int = 100
    max_cost_pct: float = 0.02  # 来回磨损占仓位比例上限（默认 2%）
    risk_min_yuan: float | None = None  # 默认用 roundtrip_cost_yuan*3
    risk_per_trade_yuan: float | None = None  # 直接指定单笔风险预算（元），优先级高于 risk_per_trade_pct
    max_exposure_pct: float | None = None
    risk_per_trade_pct: float | None = None
    stop_mode: StopMode | None = None
    max_positions: int | None = None
    max_position_pct: float | None = None  # 单标的最大仓位占比（例如 0.30=30%）；None=不限制
    returns_cache_dir: str | None = None  # 用于相关性/分散：默认 data/cache/etf
    diversify: bool = True
    diversify_window_weeks: int = 104
    diversify_min_overlap_weeks: int = 26
    diversify_max_corr: float = 0.95  # 只去掉“几乎同一个东西”的ETF
    max_per_theme: int = 0  # 0=不限制；>0=同主题最多持有 N 个（主题用名称关键字粗分）
    atr_mult: float = 2.0  # stop_mode=atr 时：止损=entry-atr_mult*ATR（周线ATR）


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    # 避免 NaN/inf 进 JSON
    try:
        import math

        if not math.isfinite(x):
            return None
    except (AttributeError):  # noqa: BLE001
        pass
    return float(x)


def _floor_to_lot(shares: int, lot: int) -> int:
    lot2 = max(1, int(lot))
    n = int(shares)
    if n <= 0:
        return 0
    return (n // lot2) * lot2


def _infer_theme(name: str) -> str:
    """
    用 ETF 名称做一个非常粗糙的“主题”归类（只用于同主题限仓的启发式过滤）。
    这玩意儿不追求完美，追求“别把两只一模一样的东西当成分散”。
    """
    n = str(name or "").replace(" ", "")
    if not n:
        return "unknown"

    rules = [
        ("机器人", "robotics"),
        ("半导体", "semiconductor"),
        ("芯片", "semiconductor"),
        ("有色", "metals"),
        ("黄金", "gold"),
        ("煤炭", "coal"),
        ("军工", "defense"),
        ("医药", "healthcare"),
        ("医疗", "healthcare"),
        ("消费", "consumer"),
        ("白酒", "consumer"),
        ("银行", "banks"),
        ("证券", "brokers"),
        ("红利", "dividend"),
        ("低波", "low_vol"),
        ("纳指", "nasdaq"),
        ("标普", "sp500"),
        ("恒生", "hang_seng"),
        ("中概", "china_overseas"),
        ("科创", "sci_tech"),
        ("创业板", "chi_next"),
        ("新能源", "new_energy"),
        ("光伏", "new_energy"),
        ("电池", "new_energy"),
    ]
    for kw, theme in rules:
        if kw in n:
            return theme

    # “宽基/大盘/指数类”粗归为 broad（同主题限仓时会更保守）
    broad_kw = ["上证", "沪深", "中证", "深证", "A50", "50", "300", "500", "1000", "180", "增强", "价值", "成长"]
    if any(k in n for k in broad_kw):
        return "broad"

    return "other"


def _item_asset(it: dict[str, Any], *, default: str) -> str:
    a = str(it.get("asset") or default).strip().lower()
    if a in {"etf", "stock", "index", "crypto"}:
        return a
    return str(default or "etf").strip().lower() or "etf"


def _resolve_returns_cache_dir(*, base_dir: Path, asset: str) -> Path:
    a = str(asset or "").strip().lower() or "etf"
    base = Path(str(base_dir))
    if base.name == a:
        return base
    cand = base / a
    if cand.exists():
        return cand
    return base


def _load_weekly_returns_from_cache(*, symbol: str, asset: str, cache_dir: Path, window_weeks: int) -> list[float] | None:
    """
    从本地缓存读取日线 → 转周线 → 计算周收益序列（pct_change）。

    返回：周收益列表（长度约 window_weeks）；失败返回 None。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None

    sym = str(symbol or "").strip()
    if not sym:
        return None
    adjust = "qfq"
    asset2 = str(asset or "etf").strip().lower() or "etf"
    path = cache_dir / f"{asset2}_{sym}_{adjust}.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True) or "date" not in df.columns or "close" not in df.columns:
        return None

    try:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        return None
    if df.empty:
        return None

    try:
        from .resample import resample_to_weekly

        dfw = resample_to_weekly(df)
        if dfw is None or getattr(dfw, "empty", True):
            return None
        close = pd.to_numeric(dfw["close"], errors="coerce").astype(float)
        rets = close.pct_change().replace([float("inf"), float("-inf")], float("nan")).dropna()
        if rets.empty:
            return None
        w = max(0, int(window_weeks))
        if w > 0:
            rets = rets.tail(w)
        xs = [float(x) for x in rets.to_list() if x is not None and math.isfinite(float(x))]
        if len(xs) < 8:
            return None
        return xs
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None


def _corr_abs_tail(a: list[float], b: list[float], *, min_overlap: int) -> float | None:
    """
    简单相关系数（用“尾部重叠”近似对齐；我们只需要粗粒度去重，不搞学术洁癖）。
    """
    n = min(len(a), len(b))
    if n < int(min_overlap):
        return None
    xa = a[-n:]
    xb = b[-n:]
    if n <= 1:
        return None
    ma = sum(xa) / n
    mb = sum(xb) / n
    num = 0.0
    da = 0.0
    db = 0.0
    for i in range(n):
        va = float(xa[i]) - ma
        vb = float(xb[i]) - mb
        num += va * vb
        da += va * va
        db += vb * vb
    if da <= 0 or db <= 0:
        return None
    c = num / math.sqrt(da * db)
    if not math.isfinite(c):
        return None
    if c > 1:
        c = 1.0
    if c < -1:
        c = -1.0
    return float(abs(c))


def build_etf_position_plan(
    *,
    items: list[dict[str, Any]],
    market_regime_label: str | None,
    params: PositionPlanParams,
) -> dict[str, Any]:
    """
    把候选列表变成“明天买多少 + 止损线”的计划（研究用途）。

    注意：
    - 默认以 ETF 口径解释；若 item 里带 asset=stock，将按 stock 的缓存文件读取相关性（lot 仍默认=100）
    - entry 用当前 close 近似（实际你是次日开盘成交，自己再估一下滑点）
    - stop 选一个“硬止损价位”：bull/neutral 通常用周线 entry_ma；bear 用日线 MA20 更紧
      - 也可手动选 stop_mode=atr：用 ATR 做波动自适应止损（更像“风险一致”的仓位）
    """
    cap = float(params.capital_yuan or 0.0)
    rt_cost = float(params.roundtrip_cost_yuan or 0.0)
    if cap <= 0:
        raise ValueError("capital_yuan 必须 > 0")

    profile = risk_profile_for_regime(market_regime_label)
    max_exposure_pct = float(params.max_exposure_pct) if params.max_exposure_pct is not None else float(profile.max_exposure_pct)
    risk_pct = float(params.risk_per_trade_pct) if params.risk_per_trade_pct is not None else float(profile.risk_per_trade_pct)
    stop_mode: StopMode = params.stop_mode or profile.stop_mode
    max_positions = int(params.max_positions) if params.max_positions is not None else int(profile.max_positions)

    lot = max(1, int(params.lot_size))
    max_cost_pct = float(params.max_cost_pct or 0.0)
    max_cost_pct = max(0.0, min(max_cost_pct, 0.20))

    risk_min = float(params.risk_min_yuan) if params.risk_min_yuan is not None else float(max(0.0, rt_cost * 3.0))
    if params.risk_per_trade_yuan is not None:
        risk_yuan = max(0.0, float(params.risk_per_trade_yuan))
    else:
        risk_yuan_base = cap * max(0.0, risk_pct)
        # 小资金（<=2000）你说能接受更大波动：把“5%”放宽成“10%”（仅对默认档生效）
        if params.risk_per_trade_pct is None and cap <= 2000.0:
            risk_yuan_base *= 2.0
        risk_yuan = max(risk_yuan_base, risk_min)
    max_exposure_yuan = cap * max(0.0, min(max_exposure_pct, 1.0))

    remaining = float(max_exposure_yuan)
    picked = 0

    # 单标的最大仓位（针对“船小随便掉头”的集中度纪律：先守住上限，再谈进攻）
    max_position_yuan = None
    if params.max_position_pct is not None:
        try:
            mp = float(params.max_position_pct)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            mp = 0.0
        if mp > 0:
            mp2 = max(0.0, min(mp, 1.0))
            max_position_yuan = float(cap) * float(mp2)

    # 分散/相关性控制（只做“去重级别”的过滤，避免两只ETF几乎一模一样）
    div_enabled = bool(getattr(params, "diversify", True))
    div_window = max(0, int(getattr(params, "diversify_window_weeks", 104) or 104))
    div_min_overlap = max(8, int(getattr(params, "diversify_min_overlap_weeks", 26) or 26))
    div_max_corr = float(getattr(params, "diversify_max_corr", 0.95) or 0.95)
    if div_max_corr < 0:
        div_max_corr = 0.0
    if div_max_corr > 1:
        div_max_corr = 1.0

    returns_cache_root = (
        Path(str(getattr(params, "returns_cache_dir", "") or "").strip())
        if getattr(params, "returns_cache_dir", None)
        else (Path("data") / "cache")
    )
    max_per_theme = max(0, int(getattr(params, "max_per_theme", 0) or 0))

    plans: list[dict[str, Any]] = []
    watch: list[dict[str, Any]] = []

    selected_returns: dict[str, list[float]] = {}
    returns_dir_by_asset: dict[str, Path] = {}
    theme_counts: dict[str, int] = {}
    corr_pairs: list[dict[str, Any]] = []

    def stop_candidates(it: dict[str, Any]) -> dict[StopMode, tuple[float | None, str]]:
        lv = it.get("levels") or {}
        stop_w = _to_float(lv.get("bbb_ma_entry"))
        if stop_w is None:
            stop_w = _to_float(lv.get("ma50"))
            stop_w_ref = "周线MA50(兜底)" if stop_w is not None else "周线entry_ma"
        else:
            stop_w_ref = "周线entry_ma"

        stop_d = _to_float(((it.get("exit") or {}).get("daily") or {}).get("ma20"))
        return {
            "weekly_entry_ma": (stop_w, stop_w_ref),
            "daily_ma20": (stop_d, "日线MA20"),
        }

    def _try_one(
        it: dict[str, Any], *, asset: str, mode: StopMode, remaining_yuan: float
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        sym = str(it.get("symbol") or "").strip()
        name = str(it.get("name") or "").strip()
        entry = _to_float(it.get("close"))
        lv = it.get("levels") or {}
        st = None
        st_ref = None
        if str(mode) == "atr":
            atr = _to_float(lv.get("atr"))
            mult = None
            try:
                mult = float(getattr(params, "atr_mult", 2.0) or 2.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                mult = 2.0
            mult = max(0.1, min(mult, 20.0))
            if entry is not None and atr is not None and entry > 0 and atr > 0:
                st = float(entry) - float(mult) * float(atr)
                st_ref = f"周线ATR*{mult:g}"
            else:
                st = None
                st_ref = "周线ATR"
        else:
            st, st_ref = stop_candidates(it)[mode]

        if not sym or entry is None or entry <= 0 or st is None or st <= 0 or st >= entry:
            return None, {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": False,
                "reason": "缺entry/stop或stop>=entry",
                "entry": entry,
                "stop": st,
                "stop_ref": st_ref,
                "stop_mode": mode,
            }

        stop_pct = (entry - st) / entry
        if stop_pct <= 0:
            return None, {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": False,
                "reason": "止损距离<=0",
                "entry": entry,
                "stop": st,
                "stop_ref": st_ref,
                "stop_mode": mode,
            }

        raw_pos_yuan = risk_yuan / float(stop_pct)
        want_yuan = min(float(raw_pos_yuan), float(remaining_yuan), float(cap))
        if max_position_yuan is not None and max_position_yuan > 0:
            want_yuan = min(float(want_yuan), float(max_position_yuan))
        shares = _floor_to_lot(int(want_yuan / entry), lot)
        pos_yuan = float(shares) * float(entry)

        if shares <= 0 or pos_yuan <= 0:
            return None, {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": False,
                "reason": "资金太小/不够一手",
                "entry": entry,
                "stop": st,
                "stop_ref": st_ref,
                "stop_mode": mode,
            }

        cost_pct = (rt_cost / pos_yuan) if (rt_cost > 0 and pos_yuan > 0) else 0.0
        if max_cost_pct > 0 and cost_pct > max_cost_pct:
            # 给个“至少买多少才不被磨死”的参考（不强行替你加仓）
            need_yuan = (rt_cost / max_cost_pct) if max_cost_pct > 0 else pos_yuan
            need_shares = _floor_to_lot(int(need_yuan / entry + (lot - 1)), lot)  # 向上取整到一手
            need_yuan2 = float(need_shares) * float(entry)
            need_risk = float(need_yuan2) * float(stop_pct)
            return None, {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": False,
                "reason": f"磨损占比过高({cost_pct:.2%} > {max_cost_pct:.2%})",
                "entry": entry,
                "stop": st,
                "stop_ref": st_ref,
                "stop_pct": float(stop_pct),
                "stop_mode": mode,
                "position_yuan": float(pos_yuan),
                "shares": int(shares),
                "min_position_yuan_for_cost": float(need_yuan),
                "min_shares_for_cost": int(need_shares),
                "risk_yuan_if_min_cost": float(need_risk),
            }

        actual_risk_yuan = float(pos_yuan) * float(stop_pct)
        return (
            {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": True,
                "entry_ref": "close",
                "entry": float(entry),
                "stop": float(st),
                "stop_ref": st_ref,
                "stop_pct": float(stop_pct),
                "stop_mode": mode,
                "shares": int(shares),
                "position_yuan": float(pos_yuan),
                "risk_yuan_target": float(risk_yuan),
                "risk_yuan_actual": float(actual_risk_yuan),
                "roundtrip_cost_yuan": float(rt_cost),
                "roundtrip_cost_pct": float(cost_pct),
                "notes": "研究用途：次日开盘再结合盘面/滑点/成交确认",
            },
            None,
        )

    for it in (items or []):
        sym0 = str(it.get("symbol") or "").strip()
        name0 = str(it.get("name") or "").strip()
        asset0 = _item_asset(it, default="etf")
        if not sym0:
            continue

        key0 = f"{asset0}:{sym0}"
        if asset0 not in returns_dir_by_asset:
            returns_dir_by_asset[asset0] = _resolve_returns_cache_dir(base_dir=returns_cache_root, asset=asset0)

        # 同主题限仓（可选）
        theme = _infer_theme(name0)
        if max_per_theme > 0 and int(theme_counts.get(theme, 0)) >= int(max_per_theme):
            watch.append({"asset": asset0, "symbol": sym0, "name": name0, "ok": False, "reason": f"同主题限仓({theme})", "theme": theme})
            continue

        # 相关性过滤（可选）：跟已选的过于相关就跳过
        if div_enabled and div_max_corr < 0.999 and len(plans) >= 1:
            cur_ret = selected_returns.get(key0)
            if cur_ret is None:
                cur_ret = _load_weekly_returns_from_cache(
                    symbol=sym0,
                    asset=asset0,
                    cache_dir=returns_dir_by_asset[asset0],
                    window_weeks=div_window,
                )
                if cur_ret is not None:
                    selected_returns[key0] = cur_ret
            if cur_ret is not None:
                worst = None
                too_corr = False
                for pkey, pret in selected_returns.items():
                    if pkey == key0 or pret is None:
                        continue
                    c = _corr_abs_tail(cur_ret, pret, min_overlap=div_min_overlap)
                    if c is None:
                        continue
                    if worst is None or float(c) > float(worst.get("corr_abs") or 0.0):
                        a_asset = asset0
                        b_asset, b_sym = pkey.split(":", 1) if ":" in pkey else ("", pkey)
                        worst = {"a": sym0, "a_asset": a_asset, "b": b_sym, "b_asset": b_asset, "corr_abs": float(c)}
                    if float(c) >= float(div_max_corr):
                        too_corr = True
                        break
                if too_corr:
                    watch.append(
                        {
                            "asset": asset0,
                            "symbol": sym0,
                            "name": name0,
                            "ok": False,
                            "reason": f"相关性过高(|corr|>={div_max_corr:.2f})",
                            "theme": theme,
                            "corr_worst": worst,
                        }
                    )
                    continue

        plan, diag = _try_one(it, asset=asset0, mode=stop_mode, remaining_yuan=remaining)
        if plan is None:
            # fallback：如果主 stop 不可用/仓位太碎，尝试其它 stop（不然小资金基本没法玩）
            alts: list[StopMode] = []
            if stop_mode == "weekly_entry_ma":
                alts = ["daily_ma20"]
            elif stop_mode == "daily_ma20":
                alts = ["weekly_entry_ma"]
            else:
                # atr：先回到“周线entry_ma”，再兜底日线MA20
                alts = ["weekly_entry_ma", "daily_ma20"]

            ok_alt = False
            best_diag = diag
            for alt in alts:
                plan2, diag2 = _try_one(it, asset=asset0, mode=alt, remaining_yuan=remaining)
                if plan2 is not None:
                    plan2["notes"] = str(plan2.get("notes") or "") + f"；注意：原本计划用 {stop_mode}，但因仓位/磨损约束改用 {alt}"
                    plans.append(plan2)
                    remaining -= float(plan2.get("position_yuan") or 0.0)
                    picked += 1
                    ok_alt = True
                    break
                best_diag = diag2 or best_diag

            if ok_alt:
                if remaining <= 0:
                    break
                if max_positions > 0 and picked >= max_positions:
                    break
            else:
                watch.append(best_diag or {"ok": False, "reason": "无法生成仓位"})
            continue

        plans.append(plan)
        theme_counts[theme] = int(theme_counts.get(theme, 0)) + 1
        # 保存 returns（用于后续相关性计算）
        if div_enabled and key0 not in selected_returns:
            cur_ret = _load_weekly_returns_from_cache(
                symbol=sym0,
                asset=asset0,
                cache_dir=returns_dir_by_asset[asset0],
                window_weeks=div_window,
            )
            if cur_ret is not None:
                selected_returns[key0] = cur_ret
        # 记录 pairwise corr（仅记录新加入与已选的关系）
        if div_enabled:
            cur_ret = selected_returns.get(key0)
            if cur_ret is not None:
                for pkey, pret in selected_returns.items():
                    if pkey == key0 or pret is None:
                        continue
                    c = _corr_abs_tail(cur_ret, pret, min_overlap=div_min_overlap)
                    if c is None:
                        continue
                    b_asset, b_sym = pkey.split(":", 1) if ":" in pkey else ("", pkey)
                    corr_pairs.append({"a": sym0, "a_asset": asset0, "b": b_sym, "b_asset": b_asset, "corr_abs": float(c)})

        remaining -= float(plan.get("position_yuan") or 0.0)
        picked += 1
        if remaining <= 0:
            break
        if max_positions > 0 and picked >= max_positions:
            break

    return {
        "generated_at": None,
        "market_regime_label": str(profile.label),
        "diversification": {
            "enabled": bool(div_enabled),
            "returns_cache_dir": str(returns_cache_root),
            "returns_cache_dir_by_asset": {k: str(v) for k, v in returns_dir_by_asset.items()},
            "window_weeks": int(div_window),
            "min_overlap_weeks": int(div_min_overlap),
            "max_abs_corr": float(div_max_corr),
            "max_per_theme": int(max_per_theme),
            "theme_counts": theme_counts,
            "corr_pairs": corr_pairs[:200],
        },
        "profile": {
            "max_exposure_pct": float(max_exposure_pct),
            "risk_per_trade_pct": float(risk_pct),
            "stop_mode": str(stop_mode),
            "max_positions": int(max_positions),
        },
        "params": {
            "capital_yuan": float(cap),
            "roundtrip_cost_yuan": float(rt_cost),
            "lot_size": int(lot),
            "max_cost_pct": float(max_cost_pct),
            "risk_min_yuan": float(risk_min),
            "risk_per_trade_yuan": float(params.risk_per_trade_yuan) if params.risk_per_trade_yuan is not None else None,
            "max_position_pct": float(params.max_position_pct) if params.max_position_pct is not None else None,
        },
        "budget": {
            "max_exposure_yuan": float(max_exposure_yuan),
            "risk_per_trade_yuan": float(risk_yuan),
            "remaining_exposure_yuan": float(max(0.0, remaining)),
        },
        "counts": {"input_items": int(len(items or [])), "picked": int(len(plans)), "watch": int(len(watch))},
        "plans": plans,
        "watch": watch[:50],
        "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
    }


def build_stock_position_plan(
    *,
    items: list[dict[str, Any]],
    market_regime_label: str | None,
    params: PositionPlanParams,
) -> dict[str, Any]:
    """
    股票仓位计划（研究用途）。

    说明：
    - 逻辑与 build_etf_position_plan 保持一致；只是默认把 item.asset 兜底为 stock。
    """
    items2: list[dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        it2 = dict(it)
        if not it2.get("asset"):
            it2["asset"] = "stock"
        items2.append(it2)
    return build_etf_position_plan(items=items2, market_regime_label=market_regime_label, params=params)
