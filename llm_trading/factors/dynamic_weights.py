# -*- coding: utf-8 -*-
"""
Phase4：动态权重（regime-aware）——先做“研究闭环”，再谈上线。

这里实现的是一个最小可用版本：
- 从 config/regime_weights.yaml 读取每个 regime 的手工权重表
- 用市场 regime 序列把权重切换成“动态 composite factor”
- 产出可复现的 IC/IR + time-split + walk-forward 报告（OOS 说话）

注意：
- 这是研究工具，不是买卖按钮
- 必须遵守 docs/框架升级/00_constraints.md（T+1、无未来函数、样本外）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import math

import numpy as np
import pandas as pd

from ..akshare_source import FetchParams
from ..data_cache import fetch_daily_cached
from ..index_composite import fetch_index_daily_spec
from ..market_regime import compute_market_regime_weekly_series
from ..pipeline import write_json
from ..tradeability import TradeabilityConfig

from ..strategy_config_loader import load_regime_weights_yaml

from .research import (
    CANONICAL_HORIZONS,
    FactorResearchParams,
    _ensure_ohlcv,
    _safe_float,
    _spearman_corr,
    compute_factor_panel,
    compute_forward_returns,
    compute_tradeability_mask,
)


ScanAsset = Literal["etf", "stock", "index"]
ScanFreq = Literal["daily", "weekly"]


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    w2: dict[str, float] = {}
    for k, v in (weights or {}).items():
        try:
            vv = float(v)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        if not math.isfinite(vv) or vv <= 0:
            continue
        w2[str(k)] = float(vv)
    s = float(sum(w2.values()))
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in w2.items()}


def _dot_score(df: pd.DataFrame, *, weights: dict[str, float]) -> pd.Series:
    """
    weights: {"ma_cross":0.2,...}  -> sum(w * df["factor_ma_cross"])
    缺列直接跳过（研究用途，不强行报错）。
    """
    if df is None or df.empty:
        return pd.Series([], dtype=float)
    out = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    for fac, w in (weights or {}).items():
        col = f"factor_{fac}"
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0.0)
        out = out + float(w) * s
    return out.astype(float)


@dataclass(frozen=True, slots=True)
class DynamicWeightsResearchParams:
    factor_params: FactorResearchParams
    regime_weights_path: Path
    baseline_regime: str = "neutral"


def run_dynamic_weights_research(
    *,
    params: DynamicWeightsResearchParams,
    cache_dir: Path,
    cache_ttl_hours: float,
    out_dir: Path,
    source: str = "auto",
) -> dict[str, Any]:
    """
    输出：
    - dynamic_weights_summary.json
    - dynamic_weights_ic.csv（日期粒度）
    """
    p = params.factor_params
    if str(p.freq) != "weekly":
        raise ValueError("dynamic-weights 当前只支持 --freq weekly（先把口径做对，再扩展 daily）")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 拉 universe 数据，构造“横截面面板”
    frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []

    for sym in p.universe:
        fp = FetchParams(
            asset=p.asset,
            symbol=str(sym),
            start_date=p.start_date.strftime("%Y%m%d") if p.start_date else None,
            end_date=p.as_of.strftime("%Y%m%d") if p.as_of else None,
            adjust="qfq" if p.asset in {"etf", "stock"} else None,
            source=str(source),
        )
        df_raw = fetch_daily_cached(fp, cache_dir=cache_dir, ttl_hours=float(cache_ttl_hours))
        if df_raw is None or getattr(df_raw, "empty", True):
            continue

        dfx_daily = _ensure_ohlcv(df_raw)
        if p.as_of is not None:
            dfx_daily = dfx_daily[dfx_daily["date"].dt.date <= p.as_of].reset_index(drop=True)

        # 动态权重研究先统一成周线（跟市场 regime 周序列对齐）
        dfx = _ensure_ohlcv(pd.DataFrame(dfx_daily))
        from ..resample import resample_to_weekly

        dfx = resample_to_weekly(dfx)
        dfx = _ensure_ohlcv(dfx)

        if dfx is None or getattr(dfx, "empty", True) or len(dfx) < 80:
            continue

        last_date = dfx["date"].iloc[-1]
        meta_rows.append({"symbol": str(sym), "rows": int(len(dfx)), "last_date": str(pd.to_datetime(last_date).date())})

        panel = compute_factor_panel(dfx)
        fwd = compute_forward_returns(dfx, horizons=p.horizons, t_plus_one=True)
        trad = compute_tradeability_mask(
            dfx,
            cfg=TradeabilityConfig(limit_up_pct=float(p.limit_up_pct), limit_down_pct=float(p.limit_down_pct), halt_vol_zero=True),
        )
        merged = panel.merge(fwd, on="date", how="left").merge(trad, on="date", how="left")
        merged["symbol"] = str(sym)
        frames.append(merged)

    if not frames:
        raise RuntimeError("universe 全部抓数失败/为空：没有可研究样本")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).reset_index(drop=True)

    # as_of：默认取各 symbol 共同的最小 last_date，确保“同一天横截面”不缺数据
    as_of_final = p.as_of
    if as_of_final is None:
        try:
            as_of_final = min(pd.to_datetime(x["last_date"]).date() for x in meta_rows if x.get("last_date"))
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_final = None
    if as_of_final is not None:
        data = data[data["date"].dt.date <= as_of_final].reset_index(drop=True)

    data["tradeable_t1"] = data.get("tradeable_t1", True).fillna(False).astype(bool)
    df_cs = data[data["tradeable_t1"]].copy()

    horizons_eff = sorted({int(x) for x in (p.horizons or []) if int(x) > 0} | set(CANONICAL_HORIZONS))
    ret_cols = [f"fwd_ret_{h}" for h in horizons_eff if f"fwd_ret_{h}" in df_cs.columns]

    # 2) 市场 regime（周序列）：支持 '+' 合成指数（等权收益合成）
    ctx_spec = str(getattr(p, "context_index_symbol", "sh000300") or "sh000300").strip()
    df_idx, ctx_eff = fetch_index_daily_spec(
        ctx_spec,
        cache_dir=Path("data") / "cache" / "index",
        ttl_hours=float(cache_ttl_hours),
        start_date=p.start_date.strftime("%Y%m%d") if p.start_date else None,
        end_date=as_of_final.strftime("%Y%m%d") if as_of_final else None,
    )

    regime_df = None
    regime_error = None
    if df_idx is None or getattr(df_idx, "empty", True):
        regime_error = "context index 数据为空"
    else:
        try:
            regime_df = compute_market_regime_weekly_series(index_symbol=str(ctx_eff or ctx_spec), df_daily=df_idx)
        except Exception as exc:  # noqa: BLE001
            regime_df = None
            regime_error = str(exc)

    if regime_df is not None and (not getattr(regime_df, "empty", True)) and "date" in regime_df.columns:
        regime_df = regime_df.copy()
        regime_df["date"] = pd.to_datetime(regime_df["date"], errors="coerce")
        regime_df = regime_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        regime_df = regime_df[["date", "label"]].rename(columns={"label": "market_regime_label"})
        df_cs = df_cs.merge(regime_df, on="date", how="left")
    else:
        df_cs["market_regime_label"] = None

    df_cs["market_regime_label"] = df_cs["market_regime_label"].fillna("unknown").astype(str)

    # 3) 计算 composite factor（static vs dynamic）
    raw_regime_weights = load_regime_weights_yaml(Path(params.regime_weights_path))
    norm_regime_weights = {k: _normalize_weights(v) for k, v in raw_regime_weights.items()}

    baseline_regime = str(params.baseline_regime or "neutral").strip().lower() or "neutral"
    base_w = norm_regime_weights.get(baseline_regime) or {}
    if not base_w:
        # baseline 缺失：退化为任意一个 regime（但报告会写清楚）
        for _k, _v in norm_regime_weights.items():
            if _v:
                base_w = dict(_v)
                baseline_regime = str(_k)
                break

    df_cs["factor_dw_static"] = _dot_score(df_cs, weights=base_w)
    df_cs["factor_dw_dynamic"] = np.nan

    for reg, w in norm_regime_weights.items():
        if not w:
            continue
        mask = df_cs["market_regime_label"].astype(str) == str(reg)
        if not bool(mask.any()):
            continue
        df_cs.loc[mask, "factor_dw_dynamic"] = _dot_score(df_cs.loc[mask], weights=w)

    df_cs["factor_dw_dynamic"] = pd.to_numeric(df_cs["factor_dw_dynamic"], errors="coerce")
    df_cs["factor_dw_dynamic"] = df_cs["factor_dw_dynamic"].fillna(df_cs["factor_dw_static"]).astype(float)

    # 4) cost（用于 top20_net）
    rt_fixed = 2.0 * max(0.0, float(p.min_fee_yuan))
    rt_slip = 2.0 * max(0.0, float(p.slippage_bps_each_side)) / 10000.0
    rt_cost_rate = float(rt_fixed / max(1.0, float(p.notional_yuan)) + rt_slip)

    # 5) IC time series（长表，SQL 友好）
    ic_rows: list[dict[str, Any]] = []
    min_cross_n = max(5, int(getattr(p, "min_cross_n", 30) or 30))
    top_q = float(getattr(p, "top_quantile", 0.8) or 0.8)
    if not (0.5 < top_q < 1.0):
        top_q = 0.8

    factor_cols = ["factor_dw_static", "factor_dw_dynamic"]
    grp = df_cs.groupby(df_cs["date"].dt.date, sort=True)

    for fcol in factor_cols:
        fac_name = str(fcol.removeprefix("factor_"))
        for rcol in ret_cols:
            h = int(rcol.removeprefix("fwd_ret_"))
            for d, g in grp:
                x = pd.to_numeric(g[fcol], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(g[rcol], errors="coerce").to_numpy(dtype=float)
                ic = _spearman_corr(x, y)
                if ic is None:
                    continue
                ok = np.isfinite(x) & np.isfinite(y)
                n = int(ok.sum())

                top_g = None
                top_n = None
                if n >= int(min_cross_n):
                    x_ok = x[ok]
                    y_ok = y[ok]
                    try:
                        thr = float(np.nanquantile(x_ok, top_q))
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        thr = float("nan")
                    if math.isfinite(thr):
                        sel = x_ok >= thr
                        if int(sel.sum()) >= 3:
                            gross = float(np.nanmean(y_ok[sel]))
                            net = float(gross - rt_cost_rate)
                            if math.isfinite(gross):
                                top_g = float(gross)
                                top_n = float(net)

                ic_rows.append(
                    {
                        "date": str(d),
                        "factor": fac_name,
                        "horizon": int(h),
                        "ic": float(ic),
                        "n_obs": int(n),
                        "top20_gross": top_g,
                        "top20_net": top_n,
                    }
                )

    ic_df = pd.DataFrame(
        ic_rows,
        columns=[
            "date",
            "factor",
            "horizon",
            "ic",
            "n_obs",
            "top20_gross",
            "top20_net",
            "asset",
            "freq",
            "as_of",
            "ref_date",
            "source",
        ],
    )
    if not ic_df.empty:
        ic_df["date"] = pd.to_datetime(ic_df["date"], errors="coerce").dt.date
        ic_df["horizon"] = pd.to_numeric(ic_df["horizon"], errors="coerce")
        ic_df["ic"] = pd.to_numeric(ic_df["ic"], errors="coerce")
        ic_df["n_obs"] = pd.to_numeric(ic_df["n_obs"], errors="coerce")
        ic_df["top20_gross"] = pd.to_numeric(ic_df["top20_gross"], errors="coerce")
        ic_df["top20_net"] = pd.to_numeric(ic_df["top20_net"], errors="coerce")
        ic_df["asset"] = str(p.asset)
        ic_df["freq"] = str(p.freq)
        ic_df["as_of"] = str(as_of_final) if as_of_final is not None else None
        ic_df["ref_date"] = str(as_of_final) if as_of_final is not None else None
        ic_df["source"] = "dynamic_weights"
    else:
        # 空表也把 schema 字段补齐（避免输出只有一个换行，DuckDB/read_csv_auto 直接炸）
        ic_df["asset"] = str(p.asset)
        ic_df["freq"] = str(p.freq)
        ic_df["as_of"] = str(as_of_final) if as_of_final is not None else None
        ic_df["ref_date"] = str(as_of_final) if as_of_final is not None else None
        ic_df["source"] = "dynamic_weights"

    ic_path = out_dir / "dynamic_weights_ic.csv"
    ic_df.to_csv(ic_path, index=False, encoding="utf-8")

    # 6) 汇总（time-split + walk-forward）
    all_dates = sorted({d for d in df_cs["date"].dt.date.dropna().to_list()})
    split_ratio = 0.7
    split_idx = int(len(all_dates) * split_ratio)
    train_end = all_dates[split_idx - 1] if split_idx >= 1 and split_idx < len(all_dates) else None
    test_start = all_dates[split_idx] if split_idx >= 1 and split_idx < len(all_dates) else None

    wf_enabled = bool(getattr(p, "walk_forward", True))
    wf_train = max(10, int(getattr(p, "train_window", 252) or 252))
    wf_test = max(5, int(getattr(p, "test_window", 63) or 63))
    wf_step = max(1, int(getattr(p, "step_window", wf_test) or wf_test))
    wf_windows: list[dict[str, Any]] = []
    if wf_enabled and len(all_dates) >= (wf_train + wf_test + 1):
        i = 0
        while i + wf_train + wf_test <= len(all_dates):
            tr_start = all_dates[i]
            tr_end = all_dates[i + wf_train - 1]
            te_start = all_dates[i + wf_train]
            te_end = all_dates[i + wf_train + wf_test - 1]
            wf_windows.append({"train_start": str(tr_start), "train_end": str(tr_end), "test_start": str(te_start), "test_end": str(te_end)})
            i += wf_step

    def _mean(x: pd.Series) -> float | None:
        s = pd.to_numeric(x, errors="coerce").dropna().astype(float)
        if s.empty:
            return None
        return float(s.mean())

    def _median(x: pd.Series) -> float | None:
        s = pd.to_numeric(x, errors="coerce").dropna().astype(float)
        if s.empty:
            return None
        return float(s.median())

    def _pos_ratio(x: pd.Series) -> float | None:
        s = pd.to_numeric(x, errors="coerce").dropna().astype(float)
        if s.empty:
            return None
        return float((s > 0).mean())

    summary: list[dict[str, Any]] = []
    for fac in ["dw_static", "dw_dynamic"]:
        row: dict[str, Any] = {"factor": fac}
        sub_f = ic_df[ic_df["factor"] == fac] if (not ic_df.empty) else ic_df

        for hh0 in CANONICAL_HORIZONS:
            # 固定 schema（保持跟 factor_research 一致的字段风格）
            row[f"ic_{hh0}"] = None
            row[f"ir_{hh0}"] = None
            row[f"ic_samples_{hh0}"] = 0
            row[f"avg_cross_n_{hh0}"] = None
            row[f"ic_train_{hh0}"] = None
            row[f"ic_test_{hh0}"] = None
            row[f"top20_gross_mean_{hh0}"] = None
            row[f"top20_net_mean_{hh0}"] = None
            row[f"wf_windows_{hh0}"] = 0
            row[f"wf_ic_train_mean_{hh0}"] = None
            row[f"wf_ic_test_mean_{hh0}"] = None
            row[f"wf_ic_test_median_{hh0}"] = None
            row[f"wf_ic_test_pos_ratio_{hh0}"] = None

            sub = sub_f[sub_f["horizon"] == int(hh0)] if (not sub_f.empty) else sub_f
            if sub is None or getattr(sub, "empty", True):
                continue

            ics = pd.to_numeric(sub["ic"], errors="coerce").dropna().astype(float)
            if ics.empty:
                continue

            ic_mean = float(ics.mean())
            ic_std = float(ics.std(ddof=0)) if len(ics) >= 2 else None
            ir = (float(ic_mean) / float(ic_std)) if (ic_std is not None and ic_std > 1e-12) else None

            row[f"ic_{hh0}"] = ic_mean
            row[f"ir_{hh0}"] = ir
            row[f"ic_samples_{hh0}"] = int(len(ics))
            row[f"avg_cross_n_{hh0}"] = _mean(sub["n_obs"])
            row[f"top20_gross_mean_{hh0}"] = _mean(sub["top20_gross"])
            row[f"top20_net_mean_{hh0}"] = _mean(sub["top20_net"])

            # time split
            if train_end is not None:
                tr = sub[sub["date"] <= train_end]
                te = sub[sub["date"] > train_end]
                row[f"ic_train_{hh0}"] = _mean(tr["ic"])
                row[f"ic_test_{hh0}"] = _mean(te["ic"])

            # walk-forward
            if wf_windows:
                tr_means = []
                te_means = []
                for w in wf_windows:
                    tr_s = date.fromisoformat(str(w["train_start"]))
                    tr_e = date.fromisoformat(str(w["train_end"]))
                    te_s = date.fromisoformat(str(w["test_start"]))
                    te_e = date.fromisoformat(str(w["test_end"]))
                    tr2 = sub[(sub["date"] >= tr_s) & (sub["date"] <= tr_e)]
                    te2 = sub[(sub["date"] >= te_s) & (sub["date"] <= te_e)]
                    if len(tr2) >= 10:
                        tr_means.append(_mean(tr2["ic"]))
                    if len(te2) >= 10:
                        te_means.append(_mean(te2["ic"]))
                tr_means2 = [x for x in tr_means if x is not None]
                te_means2 = [x for x in te_means if x is not None]
                row[f"wf_windows_{hh0}"] = int(min(len(tr_means2), len(te_means2)))
                row[f"wf_ic_train_mean_{hh0}"] = float(np.mean(tr_means2)) if tr_means2 else None
                row[f"wf_ic_test_mean_{hh0}"] = float(np.mean(te_means2)) if te_means2 else None
                row[f"wf_ic_test_median_{hh0}"] = float(np.median(te_means2)) if te_means2 else None
                row[f"wf_ic_test_pos_ratio_{hh0}"] = float((np.array(te_means2) > 0).mean()) if te_means2 else None

        summary.append(row)

    out = {
        "schema": "llm_trading.dynamic_weights_research.v1",
        "asset": str(p.asset),
        "freq": str(p.freq),
        "as_of": str(as_of_final) if as_of_final is not None else None,
        "ref_date": str(as_of_final) if as_of_final is not None else None,
        "start_date": str(p.start_date) if p.start_date is not None else None,
        "horizons": [int(x) for x in horizons_eff],
        "t_plus_one": True,
        "universe_size": int(len(p.universe)),
        "symbols_used": int(len(meta_rows)),
        "baseline_regime": str(baseline_regime),
        "regime_weights": {"path": str(params.regime_weights_path), "weights": norm_regime_weights},
        "market_regime": {"context_index_symbol": str(ctx_eff or ctx_spec), "error": regime_error},
        "oos_split": {"mode": "time_split", "train_ratio": float(split_ratio), "train_end": str(train_end) if train_end else None, "test_start": str(test_start) if test_start else None},
        "walk_forward": {"enabled": bool(wf_enabled), "train_window": int(wf_train), "test_window": int(wf_test), "step_window": int(wf_step), "windows": wf_windows, "window_count": int(len(wf_windows))},
        "tradeability": {
            "total_rows": int(len(data)),
            "tradeable_rows": int(df_cs["tradeable_t1"].sum()),
            "cfg": {"limit_up_pct": float(p.limit_up_pct), "limit_down_pct": float(p.limit_down_pct)},
        },
        "cost": {"min_fee_yuan_each_side": float(p.min_fee_yuan), "slippage_bps_each_side": float(p.slippage_bps_each_side), "notional_yuan": float(p.notional_yuan), "roundtrip_cost_rate": float(rt_cost_rate)},
        "factors": summary,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    write_json(out_dir / "dynamic_weights_summary.json", out)
    return out
