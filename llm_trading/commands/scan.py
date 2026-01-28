from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ..akshare_source import DataSourceError, FetchParams, fetch_daily
from ..chanlun import ChanlunError, compute_chanlun_structure
from ..config import load_config
from ..csv_loader import CsvSchemaError, load_ohlcv_csv
from ..dow import DowError, compute_dow_structure
from ..etf_scan import analyze_etf_symbol, load_etf_universe
from ..indicators import (
    add_accumulation_distribution_line,
    add_adx,
    add_atr,
    add_donchian_channels,
    add_ichimoku,
    add_macd,
    add_moving_averages,
    add_rsi,
)
from ..pipeline import run_llm_analysis, write_json
from ..plotting import (
    plot_chanlun_chart,
    plot_dow_chart,
    plot_ichimoku_chart,
    plot_momentum_chart,
    plot_turtle_chart,
    plot_vsa_chart,
    plot_wyckoff_chart,
)
from ..resample import resample_to_weekly
from ..vsa import compute_vsa_report
from ..stock_scan import DailyFilter, ScanFreq, analyze_stock_symbol, load_stock_universe
from ..diagnostics import Diagnostics

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

def cmd_scan_etf(args: argparse.Namespace) -> int:
    try:
        universe = load_etf_universe(include_all_funds=bool(args.include_all_funds))
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"ETF 列表拉取失败：{exc}") from exc

    if not universe:
        raise SystemExit("ETF 列表为空：AkShare 没给数据，或者源站抽风。")

    min_amount = float(args.min_amount) if args.min_amount is not None else 0.0
    min_amount_avg20 = float(getattr(args, "min_amount_avg20", 0.0) or 0.0)
    # Phase2：OpportunityScore 过滤（0~1；默认 0=不过滤）
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    min_score = max(0.0, min(float(min_score), 1.0))
    min_weeks = int(getattr(args, "min_weeks", 60) or 0)
    min_weeks = max(0, min(min_weeks, 2000))

    # BBB 回测/胜率/磨损：成本口径统一成 TradeCost（固定磨损/最低佣金/比例成本），
    # forward/backtest 侧用“按单笔资金摊回比例成本”的近似（paper-sim/rebalance 走现金口径）。
    try:
        from ..bbb import BBBParams
        from ..costs import effective_rate_for_notional, trade_cost_from_params
    except (AttributeError):  # noqa: BLE001
        BBBParams = None  # type: ignore[assignment]
        effective_rate_for_notional = None  # type: ignore[assignment]
        trade_cost_from_params = None  # type: ignore[assignment]

    bbb_horizons: list[int] = []
    raw_h = str(getattr(args, "bbb_horizons", "") or "")
    for part in raw_h.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            bbb_horizons.append(int(p))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
    bbb_horizons = sorted({h for h in bbb_horizons if h > 0}) or [4, 8, 12]

    bbb_rank_horizon = int(getattr(args, "bbb_rank_horizon", 8) or 8)
    bbb_score_mode = str(getattr(args, "bbb_score_mode", "win_rate") or "win_rate").strip().lower()
    bbb_min_trades = int(getattr(args, "bbb_min_trades", 0) or 0)
    bbb_min_win_rate = float(getattr(args, "bbb_min_win_rate", 0.0) or 0.0)
    bbb_allow_overlap = bool(getattr(args, "bbb_allow_overlap", False))
    bbb_exit_min_hold_days = int(getattr(args, "bbb_exit_min_hold_days", 5) or 5)
    bbb_exit_cooldown_days = int(getattr(args, "bbb_exit_cooldown_days", 0) or 0)
    bbb_exit_trail = bool(getattr(args, "bbb_exit_trail", True))
    bbb_exit_trail_ma = int(getattr(args, "bbb_exit_trail_ma", 20) or 20)
    bbb_exit_profit_stop = bool(getattr(args, "bbb_exit_profit_stop", True))
    bbb_exit_profit_min_ret = float(getattr(args, "bbb_exit_profit_min_ret", 0.20) or 0.20)
    bbb_exit_profit_dd_pct = float(getattr(args, "bbb_exit_profit_dd_pct", 0.12) or 0.12)
    bbb_exit_stop_loss_ret = float(getattr(args, "bbb_exit_stop_loss_ret", 0.0) or 0.0)
    bbb_exit_panic = bool(getattr(args, "bbb_exit_panic", True))
    bbb_exit_panic_vol_mult = float(getattr(args, "bbb_exit_panic_vol_mult", 3.0) or 3.0)
    bbb_exit_panic_min_drop = float(getattr(args, "bbb_exit_panic_min_drop", 0.04) or 0.04)
    bbb_exit_panic_drawdown_252d = float(getattr(args, "bbb_exit_panic_drawdown_252d", 0.25) or 0.25)
    bbb_mode_user = str(getattr(args, "bbb_mode", "auto") or "auto").strip().lower()
    bbb_mode = bbb_mode_user
    # 熊市过滤：大盘处于 bear 时，BBB 不给“能买”候选（默认关掉熊市交易）
    bbb_allow_bear = bool(getattr(args, "bbb_allow_bear", False))
    bbb_entry_ma = int(getattr(args, "bbb_entry_ma", 50) or 50)
    bbb_dist_ma_max = float(getattr(args, "bbb_dist_ma_max", 0.12) or 0.12)
    bbb_max_above_20w = float(getattr(args, "bbb_max_above_20w", 0.05) or 0.05)

    capital_yuan = float(getattr(args, "capital_yuan", 3000.0) or 3000.0)
    roundtrip_cost_yuan = float(getattr(args, "roundtrip_cost_yuan", 10.0) or 10.0)
    min_fee_yuan = float(getattr(args, "min_fee_yuan", 0.0) or 0.0)

    # buy_cost/sell_cost：比例成本（例如 0.001=0.10%）；roundtrip_cost_yuan：固定磨损；min_fee_yuan：最低佣金（每边）。
    # 注意：这里的 buy/sell_cost_rate 是“输入口径”，不要把固定磨损摊进去（否则后面的现金口径会重复算）。
    buy_cost_rate = float(getattr(args, "buy_cost", 0.0) or 0.0) if getattr(args, "buy_cost", None) is not None else 0.0
    sell_cost_rate = float(getattr(args, "sell_cost", 0.0) or 0.0) if getattr(args, "sell_cost", None) is not None else 0.0

    buy_cost_eff = 0.0
    sell_cost_eff = 0.0
    if trade_cost_from_params and effective_rate_for_notional:
        cost_base = trade_cost_from_params(
            roundtrip_cost_yuan=float(roundtrip_cost_yuan),
            min_fee_yuan=float(min_fee_yuan),
            buy_cost=float(buy_cost_rate),
            sell_cost=float(sell_cost_rate),
        )
        buy_cost_eff = effective_rate_for_notional(
            notional_yuan=float(capital_yuan),
            cost_rate=float(cost_base.buy_cost),
            min_fee_yuan=float(cost_base.buy_fee_min_yuan),
            fixed_fee_yuan=float(cost_base.buy_fee_yuan),
        )
        sell_cost_eff = effective_rate_for_notional(
            notional_yuan=float(capital_yuan),
            cost_rate=float(cost_base.sell_cost),
            min_fee_yuan=float(cost_base.sell_fee_min_yuan),
            fixed_fee_yuan=float(cost_base.sell_fee_yuan),
        )

    # BBB 额外滑点/冲击成本近似（默认不加，避免跟“固定磨损”重复算）
    bbb_slippage_mode = str(getattr(args, "bbb_slippage_mode", "none") or "none").strip().lower()
    bbb_slippage_bps = float(getattr(args, "bbb_slippage_bps", 0.0) or 0.0)
    bbb_slippage_ref_amount_yuan = float(getattr(args, "bbb_slippage_ref_amount_yuan", 1e8) or 1e8)
    bbb_slippage_bps_min = float(getattr(args, "bbb_slippage_bps_min", 0.0) or 0.0)
    bbb_slippage_bps_max = float(getattr(args, "bbb_slippage_bps_max", 30.0) or 30.0)
    bbb_slippage_unknown_bps = float(getattr(args, "bbb_slippage_unknown_bps", 10.0) or 10.0)
    bbb_slippage_vol_mult = float(getattr(args, "bbb_slippage_vol_mult", 0.0) or 0.0)

    # 大盘牛熊/风险偏好（用于 bbb-mode=auto）
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    regime_dict, regime_error, regime_index_eff = _compute_market_regime_payload(regime_index, canary_downgrade=regime_canary)
    regime_enabled = bool(regime_index_eff)
    regime_label = str((regime_dict or {}).get("label") or "unknown")

    # BBB 7因子面板里的 RS：默认用 300+500 等权合成当“更中性的尺子”；也支持 auto=跟随 regime-index 第一个指数。
    rs_index_symbol = None
    rs_index_weekly = None
    try:
        from ..market_regime import parse_regime_index_list

        bbb_rs_index = str(getattr(args, "bbb_rs_index", "") or "").strip()
        rs_spec = bbb_rs_index if bbb_rs_index else "sh000300+sh000905"
        rs_spec_l = str(rs_spec).strip().lower()
        if rs_spec_l in {"off", "none", "0"}:
            rs_index_symbol = None
            rs_index_weekly = None
        elif rs_spec_l in {"auto"}:
            idxs = parse_regime_index_list(regime_index)
            rs_spec = (idxs[0] if idxs else None) or "sh000300"
        if rs_index_symbol is None and rs_index_weekly is None and rs_spec_l in {"off", "none", "0"}:
            pass
        else:
            from ..index_composite import fetch_index_daily_spec

            df_idx, rs_eff = fetch_index_daily_spec(
                str(rs_spec),
                cache_dir=Path("data") / "cache" / "index",
                ttl_hours=6.0,
            )
            rs_index_symbol = rs_eff or str(rs_spec)
            rs_index_weekly = resample_to_weekly(df_idx) if df_idx is not None and (not getattr(df_idx, "empty", True)) else None
    except (AttributeError):  # noqa: BLE001
        rs_index_symbol = None
        rs_index_weekly = None

    # bbb-mode=auto：牛市更激进（允许回踩）、熊市更谨慎（严格）
    if bbb_mode_user == "auto":
        if regime_label == "bull":
            bbb_mode = "pullback"
        elif regime_label == "bear":
            bbb_mode = "strict"
        elif regime_label == "neutral":
            bbb_mode = "pullback"
        else:
            bbb_mode = "strict"

    bbb_params = None
    if BBBParams is not None:
        # BBB：给一个简单的模式选择，别让你为了调一堆开关把自己搞疯
        # - strict: 周MACD金叉且>0 + 日MACD为多 + 位置靠均线（最保守）
        # - pullback: 周MACD>0（允许周线回踩造成的“周MACD未转多”）+ 日MACD为多（更贴近“右侧定方向 + 回踩挑位置”）
        # - early: 周MACD金叉（允许发生在0轴下）+ 日MACD为多（更早，但更容易吃回撤）
        req_w_bull = True
        req_w_above0 = True
        req_d_bull = True
        if bbb_mode == "pullback":
            req_w_bull = False
            req_w_above0 = True
            req_d_bull = True
        elif bbb_mode == "early":
            req_w_bull = True
            req_w_above0 = False
            req_d_bull = True
        else:
            bbb_mode = "strict"

        bbb_params = BBBParams(
            entry_ma=max(2, int(bbb_entry_ma)),
            dist_ma50_max=max(0.0, float(bbb_dist_ma_max)),
            max_above_20w=max(0.0, float(bbb_max_above_20w)),
            min_weekly_bars_total=max(60, int(min_weeks) if int(min_weeks) > 0 else 60),
            require_weekly_macd_bullish=bool(req_w_bull),
            require_weekly_macd_above_zero=bool(req_w_above0),
            require_daily_macd_bullish=bool(req_d_bull),
        )

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    cache_dir = Path(args.cache_dir) if getattr(args, "cache_dir", None) else (Path("data") / "cache" / "etf")
    cache_ttl_hours = float(args.cache_ttl_hours) if getattr(args, "cache_ttl_hours", None) is not None else 24.0
    analysis_cache_dir = Path(args.analysis_cache_dir) if getattr(args, "analysis_cache_dir", None) else (Path("data") / "cache" / "analysis" / "etf")
    analysis_cache = bool(getattr(args, "analysis_cache", True))

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"etf_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = Diagnostics()

    # 可选：用 TuShare 拉“宏观/聪明钱”面板（ERP / north/south），只算一次，供解释/风控参考
    # 注意：不作为 BBB 硬过滤条件（避免你没配 TuShare 也跑不起来）。
    tushare_factors = None
    try:
        from ..tushare_factors import compute_tushare_factor_pack

        # as_of：优先用 regime 的 as_of_max（更贴近“最新可得”），否则用今天
        as_of = datetime.now().date()
        try:
            ens = (regime_dict or {}).get("ensemble") if isinstance(regime_dict, dict) else None
            as_of_s = None
            if isinstance(ens, dict):
                as_of_s = ens.get("as_of_max") or ens.get("as_of_min")
            if not as_of_s and isinstance(regime_dict, dict):
                as_of_s = regime_dict.get("last_date")
            if as_of_s:
                as_of = datetime.strptime(str(as_of_s).strip()[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of = datetime.now().date()

        # context index：用 regime-index 的第一个指数（默认 sh000300）
        ctx_idx = "sh000300"
        try:
            from ..market_regime import parse_regime_index_list

            idxs = parse_regime_index_list(str(regime_index))
            if idxs:
                ctx_idx = str(idxs[0])
        except (AttributeError):  # noqa: BLE001
            ctx_idx = "sh000300"

        tushare_factors = compute_tushare_factor_pack(
            as_of=as_of,
            context_index_symbol_prefixed=ctx_idx,
            symbol_prefixed=None,
            daily_amount_by_date=None,
            cache_dir=Path("data") / "cache" / "tushare_factors",
            ttl_hours=6.0,
        )
        try:
            write_json(out_dir / "tushare_factors.json", tushare_factors)
        except Exception as exc:  # noqa: BLE001
            diag.record("write_tushare_factors_json", exc, note="写出 tushare_factors.json 失败（不影响主流程）")
    except Exception as exc:  # noqa: BLE001
        try:
            (out_dir / "tushare_factors_error.txt").write_text(str(exc), encoding="utf-8")
        except Exception as exc2:  # noqa: BLE001
            diag.record("write_tushare_factors_error_txt", exc2, note="写出 tushare_factors_error.txt 失败（吞错治理）")
        diag.record("tushare_factors", exc, note="TuShare 因子包失败（已降级）")

    results: list[dict] = []
    errors: list[dict] = []
    filtered: list[dict] = []
    filtered_by_min_amount = 0
    filtered_by_min_amount_avg20 = 0
    filtered_by_min_score = 0

    import math

    def safe_float(v, *, default: float | None = None) -> float | None:
        try:
            if v is None:
                return default
            x = float(v)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return default
        return x if math.isfinite(x) else default

    workers = int(getattr(args, "workers", 8) or 8)
    workers = max(1, min(workers, 32))

    total = len(universe)

    def run_one(item):
        return analyze_etf_symbol(
            item,
            freq=args.freq,
            window=args.window,
            bbb_params=bbb_params,
            bbb_horizons=bbb_horizons,
            bbb_rank_horizon=int(bbb_rank_horizon),
            bbb_score_mode=str(bbb_score_mode),
            # forward/backtest 侧用“把固定磨损/最低佣金摊回比例成本”的近似口径
            bbb_buy_cost=float(buy_cost_eff),
            bbb_sell_cost=float(sell_cost_eff),
            bbb_slippage_mode=str(bbb_slippage_mode),
            bbb_slippage_bps=float(bbb_slippage_bps),
            bbb_slippage_ref_amount_yuan=float(bbb_slippage_ref_amount_yuan),
            bbb_slippage_bps_min=float(bbb_slippage_bps_min),
            bbb_slippage_bps_max=float(bbb_slippage_bps_max),
            bbb_slippage_unknown_bps=float(bbb_slippage_unknown_bps),
            bbb_slippage_vol_mult=float(bbb_slippage_vol_mult),
            bbb_non_overlapping=not bool(bbb_allow_overlap),
            bbb_exit_min_hold_days=int(bbb_exit_min_hold_days),
            bbb_exit_cooldown_days=int(bbb_exit_cooldown_days),
            bbb_exit_trail_ma=int(bbb_exit_trail_ma),
            bbb_exit_enable_trail=bool(bbb_exit_trail),
            bbb_exit_stop_loss_ret=float(bbb_exit_stop_loss_ret),
            bbb_exit_profit_stop_enabled=bool(bbb_exit_profit_stop),
            bbb_exit_profit_stop_min_profit_ret=float(bbb_exit_profit_min_ret),
            bbb_exit_profit_stop_dd_pct=float(bbb_exit_profit_dd_pct),
            bbb_exit_panic_enabled=bool(bbb_exit_panic),
            bbb_exit_panic_vol_mult=float(bbb_exit_panic_vol_mult),
            bbb_exit_panic_min_drop=float(bbb_exit_panic_min_drop),
            bbb_exit_panic_drawdown_252d=float(bbb_exit_panic_drawdown_252d),
            include_bbb_samples=bool(getattr(args, "bbb_include_samples", False)),
            cache_dir=cache_dir,
            cache_ttl_hours=float(cache_ttl_hours),
            analysis_cache=bool(analysis_cache),
            analysis_cache_dir=analysis_cache_dir,
            rs_index_symbol=(str(rs_index_symbol) if rs_index_symbol else None),
            rs_index_weekly=rs_index_weekly,
        )

    done = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(run_one, item): item for item in universe}
        for fut in as_completed(fut_map):
            item = fut_map[fut]
            done += 1
            if args.verbose:
                print(f"[{done}/{total}] {item.symbol} {item.name} ...")
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                # 单个标的失败不应拖垮整次扫描；把错误记录到 errors 里即可。
                r = {"symbol": item.symbol, "name": item.name, "error": f"{exc.__class__.__name__}: {exc}"}

            if r.get("error"):
                errors.append(r)
                continue

            if min_amount > 0:
                amt = safe_float(r.get("amount"), default=0.0) or 0.0
                if amt < min_amount:
                    filtered_by_min_amount += 1
                    continue

            if min_amount_avg20 > 0:
                liq = r.get("liquidity") or {}
                avg20 = safe_float(liq.get("daily_amount_avg20"), default=None)
                if avg20 is None:
                    avg20 = safe_float(liq.get("amount_avg20"), default=None)
                if avg20 is None:
                    avg20 = 0.0
                if avg20 < min_amount_avg20:
                    filtered_by_min_amount_avg20 += 1
                    continue

            # OpportunityScore 过滤（Phase2；默认不启用）
            if min_score > 0:
                sc = safe_float(r.get("opp_score"), default=0.0) or 0.0
                if float(sc) < float(min_score):
                    filtered_by_min_score += 1
                    continue

            results.append(r)

    # 排名输出（趋势突破 / 回踩波段 两套）
    def weekly_total(x: dict) -> int:
        bars = x.get("bars") or {}
        try:
            return int(bars.get("weekly_total") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 0

    rankable = results
    if min_weeks > 0:
        rankable = [x for x in results if weekly_total(x) >= min_weeks]

    def key_score(which: str):
        def _k(x: dict):
            s = (x.get("scores") or {}).get(which)
            try:
                return float(s or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return 0.0

        return _k

    results_sorted_trend = sorted(rankable, key=key_score("trend"), reverse=True)
    results_sorted_swing = sorted(rankable, key=key_score("swing"), reverse=True)

    top_k = int(args.top_k) if args.top_k else 30
    top_k = max(5, min(top_k, 100))

    # BBB（左侧偏稳健）：位置优先 + 周线MACD在0轴上 + 日线MACD为多
    bbb_fail_stats: dict[str, int] = {}
    bbb_items: list[dict] = []
    bbb_ok_raw = 0
    bbb_blocked_by_bear = 0
    filtered_by_bbb_min_trades = 0
    filtered_by_bbb_min_win_rate = 0
    for x in results:
        bbb = x.get("bbb") if isinstance(x, dict) else None
        if isinstance(bbb, dict) and bool(bbb.get("ok")):
            bbb_ok_raw += 1

            # 熊市直接不做 BBB（你要硬上，显式传 --bbb-allow-bear）
            if (not bbb_allow_bear) and str(regime_label).strip().lower() == "bear":
                bbb_blocked_by_bear += 1
                bbb_fail_stats["大盘熊市过滤"] = int(bbb_fail_stats.get("大盘熊市过滤", 0)) + 1
                continue

            # 额外门槛：样本数 / 胜率（用 rank_horizon 口径）
            rank_h = int(bbb_rank_horizon)
            fwd = x.get("bbb_forward") if isinstance(x, dict) else None
            fwd = fwd if isinstance(fwd, dict) else {}
            st = fwd.get(f"{rank_h}w") if isinstance(fwd, dict) else None
            st = st if isinstance(st, dict) else {}

            trades = int(safe_float(st.get("trades"), default=0.0) or 0.0)
            win_rate = float(safe_float(st.get("win_rate"), default=0.0) or 0.0)
            win_rate_shrunk = float(safe_float(st.get("win_rate_shrunk"), default=win_rate) or win_rate)

            if bbb_min_trades > 0 and trades < int(bbb_min_trades):
                filtered_by_bbb_min_trades += 1
                continue
            # 用“收缩胜率”做阈值：trades 太少时别让它轻易过关
            if bbb_min_win_rate > 0 and win_rate_shrunk < float(bbb_min_win_rate):
                filtered_by_bbb_min_win_rate += 1
                continue

            bbb_items.append(x)
        else:
            for reason in (bbb.get("fails") if isinstance(bbb, dict) else []) or []:
                bbb_fail_stats[reason] = int(bbb_fail_stats.get(reason, 0)) + 1

    # BBB 7因子加权排序（不改 bbb.ok/fails，只影响候选优先级）
    bbb_factor7_enabled = bool(getattr(args, "bbb_factor7", True))
    raw_w = str(getattr(args, "bbb_factor7_weights", "") or "").strip()
    default_w = {"rs": 0.35, "trend": 0.15, "vol": 0.15, "drawdown": 0.15, "liquidity": 0.10, "boll": 0.05, "volume": 0.05}

    def _parse_w(s: str) -> dict[str, float]:
        w = dict(default_w)
        txt = str(s or "").strip()
        if txt:
            for part in txt.split(","):
                p = str(part or "").strip()
                if not p or "=" not in p:
                    continue
                k, v = p.split("=", 1)
                key = str(k or "").strip().lower()
                if key not in w:
                    continue
                try:
                    w[key] = float(v)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
        # 归一化（全是0就算了）
        s2 = float(sum(max(0.0, float(x)) for x in w.values()))
        if s2 > 0:
            w = {k: float(max(0.0, float(v))) / s2 for k, v in w.items()}
        return w

    w7 = _parse_w(raw_w)

    # 计算 percentile rank -> [-1, 1]，避免不同指标量纲把排序搞崩
    import bisect

    def _norm_rank(sorted_vals: list[float], v: float, *, higher_better: bool) -> float:
        if not sorted_vals or v is None:  # type: ignore[truthy-bool]
            return 0.0
        try:
            x = float(v)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 0.0
        if not math.isfinite(x):
            return 0.0
        n = len(sorted_vals)
        if n <= 1:
            base = 0.0
        else:
            lo = bisect.bisect_left(sorted_vals, x)
            hi = bisect.bisect_right(sorted_vals, x)
            pos = (float(lo) + float(hi - 1)) / 2.0
            pct = float(pos) / float(n - 1)
            base = (pct - 0.5) * 2.0
        return float(base if higher_better else -base)

    if bbb_items and bbb_factor7_enabled:
        # 收集原始值（缺失就跳过）
        rs12_vals: list[float] = []
        rs26_vals: list[float] = []
        adx_vals: list[float] = []
        vol20_vals: list[float] = []
        atrp_vals: list[float] = []
        dd_vals: list[float] = []
        liq_vals: list[float] = []
        boll_vals: list[float] = []
        ar_vals: list[float] = []
        vr_vals: list[float] = []

        def _get_fp(it: dict, path: list[str]) -> float | None:
            cur = it.get("factor_panel_7") if isinstance(it.get("factor_panel_7"), dict) else {}
            for k in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            try:
                x = None if cur is None else float(cur)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return None
            return float(x) if (x is not None and math.isfinite(float(x))) else None

        for it in bbb_items:
            rs12 = _get_fp(it, ["rs", "rs_12w"])
            rs26 = _get_fp(it, ["rs", "rs_26w"])
            adx14 = _get_fp(it, ["trend", "adx14"])
            v20 = _get_fp(it, ["vol", "vol_20d"])
            atrp = _get_fp(it, ["vol", "atr14_pct"])
            dd = _get_fp(it, ["drawdown", "dd_252d"])
            liq = _get_fp(it, ["liquidity", "amount_avg20"])
            boll = _get_fp(it, ["boll", "bandwidth_rel"])
            ar = _get_fp(it, ["liquidity", "amount_ratio"])
            vr = _get_fp(it, ["liquidity", "volume_ratio"])

            if rs12 is not None:
                rs12_vals.append(float(rs12))
            if rs26 is not None:
                rs26_vals.append(float(rs26))
            if adx14 is not None:
                adx_vals.append(float(adx14))
            if v20 is not None:
                vol20_vals.append(float(v20))
            if atrp is not None:
                atrp_vals.append(float(atrp))
            if dd is not None:
                dd_vals.append(float(dd))
            if liq is not None and liq > 0:
                # 先 log，避免极端流动性把 rank 拉爆
                liq_vals.append(float(math.log1p(float(liq))))
            if boll is not None and boll > 0:
                boll_vals.append(float(boll))
            if ar is not None and ar > 0:
                ar_vals.append(float(ar))
            if vr is not None and vr > 0:
                vr_vals.append(float(vr))

        rs12_vals.sort()
        rs26_vals.sort()
        adx_vals.sort()
        vol20_vals.sort()
        atrp_vals.sort()
        dd_vals.sort()
        liq_vals.sort()
        boll_vals.sort()
        ar_vals.sort()
        vr_vals.sort()

        for it in bbb_items:
            rs12 = _get_fp(it, ["rs", "rs_12w"])
            rs26 = _get_fp(it, ["rs", "rs_26w"])
            adx14 = _get_fp(it, ["trend", "adx14"])
            v20 = _get_fp(it, ["vol", "vol_20d"])
            atrp = _get_fp(it, ["vol", "atr14_pct"])
            dd = _get_fp(it, ["drawdown", "dd_252d"])
            liq0 = _get_fp(it, ["liquidity", "amount_avg20"])
            liq = float(math.log1p(float(liq0))) if (liq0 is not None and float(liq0) > 0) else None
            boll = _get_fp(it, ["boll", "bandwidth_rel"])
            ar = _get_fp(it, ["liquidity", "amount_ratio"])
            vr = _get_fp(it, ["liquidity", "volume_ratio"])

            rs_parts: list[float] = []
            if rs12 is not None:
                rs_parts.append(_norm_rank(rs12_vals, float(rs12), higher_better=True))
            if rs26 is not None:
                rs_parts.append(_norm_rank(rs26_vals, float(rs26), higher_better=True))
            rs_sc = float(sum(rs_parts) / len(rs_parts)) if rs_parts else 0.0

            trend_sc = _norm_rank(adx_vals, float(adx14), higher_better=True) if adx14 is not None else 0.0

            vol_sc = 0.0
            if v20 is not None or atrp is not None:
                parts = []
                if v20 is not None:
                    parts.append(_norm_rank(vol20_vals, float(v20), higher_better=False))
                if atrp is not None:
                    parts.append(_norm_rank(atrp_vals, float(atrp), higher_better=False))
                vol_sc = float(sum(parts) / len(parts)) if parts else 0.0

            dd_sc = _norm_rank(dd_vals, float(dd), higher_better=True) if dd is not None else 0.0
            liq_sc = _norm_rank(liq_vals, float(liq), higher_better=True) if liq is not None else 0.0
            boll_sc = _norm_rank(boll_vals, float(boll), higher_better=False) if boll is not None else 0.0

            volconf_sc = 0.0
            if ar is not None or vr is not None:
                parts = []
                if ar is not None:
                    parts.append(_norm_rank(ar_vals, float(ar), higher_better=True))
                if vr is not None:
                    parts.append(_norm_rank(vr_vals, float(vr), higher_better=True))
                volconf_sc = float(sum(parts) / len(parts)) if parts else 0.0

            score7 = (
                float(w7.get("rs", 0.0)) * float(rs_sc)
                + float(w7.get("trend", 0.0)) * float(trend_sc)
                + float(w7.get("vol", 0.0)) * float(vol_sc)
                + float(w7.get("drawdown", 0.0)) * float(dd_sc)
                + float(w7.get("liquidity", 0.0)) * float(liq_sc)
                + float(w7.get("boll", 0.0)) * float(boll_sc)
                + float(w7.get("volume", 0.0)) * float(volconf_sc)
            )

            it["bbb_factor7"] = {
                "enabled": True,
                "score": float(score7),
                "weights": dict(w7),
                "components": {
                    "rs": float(rs_sc),
                    "trend": float(trend_sc),
                    "vol": float(vol_sc),
                    "drawdown": float(dd_sc),
                    "liquidity": float(liq_sc),
                    "boll": float(boll_sc),
                    "volume": float(volconf_sc),
                },
                "note": "score=percentile-rank归一化后加权（仅用于排序；不改变BBB硬规则）",
            }

    def key_bbb(x: dict) -> tuple:
        bbb2 = x.get("bbb") if isinstance(x, dict) else None
        score = safe_float((bbb2 or {}).get("score"), default=0.0) or 0.0
        f7 = safe_float(((x.get("bbb_factor7") or {}) if isinstance(x.get("bbb_factor7"), dict) else {}).get("score"), default=0.0) or 0.0

        lv = x.get("levels") or {}
        close = safe_float(x.get("close"), default=0.0) or 0.0
        ma_entry = safe_float(lv.get("bbb_ma_entry"), default=None)
        if ma_entry is None:
            ma_entry = safe_float(lv.get("ma50"), default=0.0) or 0.0

        dist = 9e9
        if close > 0 and ma_entry and float(ma_entry) > 0:
            dist = abs(close - float(ma_entry)) / float(ma_entry)  # 越接近 entry_ma 越不容易“套山上”

        # 更偏“位置优先”：离 20W 上轨越远（room 越大）越稳
        room = 0.0
        upper = lv.get("resistance_20w")
        upper_f = safe_float(upper)
        if upper_f is not None and upper_f > 0 and close > 0:
            room = (upper_f - close) / upper_f

        amt = safe_float(x.get("amount"), default=0.0) or 0.0

        mom = x.get("momentum") or {}
        macd_w = safe_float(mom.get("macd"), default=0.0) or 0.0

        # 排序：胜率/期望(经验分)优先 -> 7因子加权 -> 位置更好 -> room更大 -> 流动性更强 -> 周线动量更强
        return (-float(score), -float(f7), float(dist), -float(room), -float(amt), -float(macd_w))

    bbb_sorted = sorted(bbb_items, key=key_bbb)

    filtered_by_min_weeks = int(len(results) - len(rankable))
    write_json(
        out_dir / "top_trend.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_weeks": int(min_weeks),
            "min_score": float(min_score) if min_score > 0 else None,
            "counts": {
                "results": len(results),
                "rankable": len(rankable),
                "filtered_by_min_weeks": filtered_by_min_weeks,
                "filtered_by_min_score": int(filtered_by_min_score),
            },
            "items": results_sorted_trend[:top_k],
        },
    )
    write_json(
        out_dir / "top_swing.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_weeks": int(min_weeks),
            "min_score": float(min_score) if min_score > 0 else None,
            "counts": {
                "results": len(results),
                "rankable": len(rankable),
                "filtered_by_min_weeks": filtered_by_min_weeks,
                "filtered_by_min_score": int(filtered_by_min_score),
            },
            "items": results_sorted_swing[:top_k],
        },
    )
    top_bbb_obj = {
        "generated_at": datetime.now().isoformat(),
        "freq": args.freq,
        "bbb": {
            "mode": bbb_mode,
            "mode_user": bbb_mode_user,
            "mode_effective": bbb_mode,
            "market_regime": regime_dict,
            "market_regime_error": regime_error,
            "market_regime_index": regime_index_eff,
            "tushare_factors": tushare_factors,
            "allow_bear": bool(bbb_allow_bear),
            "entry_ma": int(getattr(bbb_params, "entry_ma", 50) or 50) if bbb_params else 50,
            "dist_ma50_max": float(getattr(bbb_params, "dist_ma50_max", 0.12) or 0.12) if bbb_params else 0.12,
            "require_weekly_macd_bullish": bool(getattr(bbb_params, "require_weekly_macd_bullish", True)) if bbb_params else True,
            "require_weekly_macd_above_zero": bool(getattr(bbb_params, "require_weekly_macd_above_zero", True)) if bbb_params else True,
            "require_daily_macd_bullish": bool(getattr(bbb_params, "require_daily_macd_bullish", True)) if bbb_params else True,
            "max_above_20w": float(getattr(bbb_params, "max_above_20w", 0.05) or 0.05) if bbb_params else 0.05,
            "min_weekly_bars_total": int(getattr(bbb_params, "min_weekly_bars_total", max(60, min_weeks)) or max(60, min_weeks)) if bbb_params else max(60, min_weeks),
            "horizons": list(bbb_horizons),
            "rank_horizon": int(bbb_rank_horizon),
            "score_mode": str(bbb_score_mode),
            "min_trades": int(bbb_min_trades),
            "min_win_rate": float(bbb_min_win_rate),
            "capital_yuan": float(capital_yuan),
            "roundtrip_cost_yuan": float(roundtrip_cost_yuan),
            "min_fee_yuan": float(min_fee_yuan),
            # buy_cost/sell_cost：比例成本输入口径（不含 roundtrip/min_fee 的摊回；避免现金口径重复算）。
            "buy_cost": float(buy_cost_rate),
            "sell_cost": float(sell_cost_rate),
            "analysis_cache": bool(analysis_cache),
            "analysis_cache_dir": str(analysis_cache_dir),
            "slippage_mode": str(bbb_slippage_mode),
            "slippage_bps": float(bbb_slippage_bps),
            "slippage_ref_amount_yuan": float(bbb_slippage_ref_amount_yuan),
            "slippage_bps_min": float(bbb_slippage_bps_min),
            "slippage_bps_max": float(bbb_slippage_bps_max),
            "slippage_unknown_bps": float(bbb_slippage_unknown_bps),
            "slippage_vol_mult": float(bbb_slippage_vol_mult),
            "allow_overlap": bool(bbb_allow_overlap),
            "exit_min_hold_days": int(bbb_exit_min_hold_days),
            "exit_cooldown_days": int(bbb_exit_cooldown_days),
            "exit_trail": bool(bbb_exit_trail),
            "exit_trail_ma": int(bbb_exit_trail_ma),
            "exit_profit_stop": bool(bbb_exit_profit_stop),
            "exit_profit_min_ret": float(bbb_exit_profit_min_ret),
            "exit_profit_dd_pct": float(bbb_exit_profit_dd_pct),
            "exit_stop_loss_ret": float(bbb_exit_stop_loss_ret),
            "exit_panic": bool(bbb_exit_panic),
            "exit_panic_vol_mult": float(bbb_exit_panic_vol_mult),
            "exit_panic_min_drop": float(bbb_exit_panic_min_drop),
            "exit_panic_drawdown_252d": float(bbb_exit_panic_drawdown_252d),
            "factor7": {
                "enabled": bool(bbb_factor7_enabled),
                "weights": dict(w7),
                "note": "7因子只用于候选排序（模式1：面板解释+排序加权），不改变BBB ok/fails 硬条件。",
            },
            "min_daily_amount_avg20": float(min_amount_avg20) if min_amount_avg20 > 0 else None,
            "min_score": float(min_score) if min_score > 0 else None,
            "fail_stats": bbb_fail_stats,
        },
        "counts": {
            "filtered_by_min_amount": int(filtered_by_min_amount),
            "filtered_by_min_amount_avg20": int(filtered_by_min_amount_avg20),
            "filtered_by_min_score": int(filtered_by_min_score),
            "results": len(results),
            "rankable": len(rankable),
            "filtered_by_min_weeks": filtered_by_min_weeks,
            "errors": len(errors),
            "bbb_ok_raw": int(bbb_ok_raw),
            "bbb_blocked_by_bear": int(bbb_blocked_by_bear),
            "filtered_by_bbb_min_trades": int(filtered_by_bbb_min_trades),
            "filtered_by_bbb_min_win_rate": int(filtered_by_bbb_min_win_rate),
            "bbb_ok": len(bbb_items),
        },
        "items": bbb_sorted[:top_k],
    }
    write_json(out_dir / "top_bbb.json", top_bbb_obj)

    # 给组合层的统一 signals schema（研究用途）：先把 BBB 打通，后面其它策略也会跟上。
    try:
        from ..signals import signals_from_top_bbb

        write_json(out_dir / "signals.json", signals_from_top_bbb(top_bbb_obj))
    except (AttributeError) as exc:  # noqa: BLE001
        diag.record("write_signals_json", exc, note="写出 signals.json 失败（可能影响组合层输入）")

    # 仓位计划：给小白一眼就能执行的“该上车啦/买多少/止损线”
    plan_obj = None
    plan_err = None
    try:
        from ..positioning import PositionPlanParams, build_etf_position_plan

        regime_label2 = str((regime_dict or {}).get("label") or "unknown")
        plan = build_etf_position_plan(
            items=bbb_sorted[:top_k],
            market_regime_label=regime_label2,
            params=PositionPlanParams(
                capital_yuan=float(capital_yuan),
                roundtrip_cost_yuan=float(roundtrip_cost_yuan),
                lot_size=100,
                max_cost_pct=0.02,
                returns_cache_dir=str(cache_dir),
            ),
        )
        plan["generated_at"] = datetime.now().isoformat()
        plan["input"] = {
            "source": "scan-etf",
            "freq": str(args.freq),
            "market_regime_index": regime_index_eff,
            "market_regime_error": regime_error,
            "capital_yuan": float(capital_yuan),
            "roundtrip_cost_yuan": float(roundtrip_cost_yuan),
            "stop_trigger": "close_only",
            "stop_follow_regime": True,
        }
        plan_obj = plan
        write_json(out_dir / "position_plan.json", plan)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        plan_err = str(exc)
        write_json(out_dir / "position_plan_error.json", {"error": str(exc)})

    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})
    write_json(out_dir / "filtered.json", {"generated_at": datetime.now().isoformat(), "filtered": filtered})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8")

    # 运行 meta + 标准化报告（可复现/可回放）
    as_of = None
    for it in results:
        if not isinstance(it, dict):
            continue
        ld = str(it.get("last_date") or "").strip()
        if not ld:
            continue
        if as_of is None or ld > as_of:
            as_of = ld

    data_hash = None
    try:
        from ..analysis_cache import compute_params_hash

        fp: list[dict[str, Any]] = []
        for it in results:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue
            bars = it.get("bars") if isinstance(it.get("bars"), dict) else {}
            fp.append(
                {
                    "symbol": sym,
                    "last_date": it.get("last_date"),
                    "last_daily_date": it.get("last_daily_date"),
                    "daily_bars": bars.get("daily"),
                    "weekly_total": bars.get("weekly_total"),
                }
            )
        fp.sort(key=lambda x: str(x.get("symbol") or ""))
        data_hash = compute_params_hash({"asset": "etf", "items": fp}) if fp else None
    except (AttributeError):  # noqa: BLE001
        data_hash = None

    run_meta = _write_run_meta(out_dir, args, extra={"cmd": "scan-etf", "as_of": as_of, "regime": regime_label, "data_hash": data_hash})
    run_config = _write_run_config(out_dir, args, note="scan-etf", extra={"cmd": "scan-etf"})
    try:
        from ..reporting import build_report_v1

        top_bbb_brief: list[dict[str, Any]] = []
        for it in bbb_sorted[: min(top_k, 10)]:
            if not isinstance(it, dict):
                continue
            bbb2 = it.get("bbb") if isinstance(it.get("bbb"), dict) else {}
            best = it.get("bbb_best") if isinstance(it.get("bbb_best"), dict) else {}
            exit2 = it.get("exit") if isinstance(it.get("exit"), dict) else {}
            top_bbb_brief.append(
                {
                    "symbol": str(it.get("symbol") or ""),
                    "name": str(it.get("name") or ""),
                    "close": it.get("close"),
                    "last_date": it.get("last_date"),
                    "bbb": {
                        "ok": bool(bbb2.get("ok")) if isinstance(bbb2, dict) else False,
                        "why": bbb2.get("why") if isinstance(bbb2, dict) else None,
                        "score": bbb2.get("score") if isinstance(bbb2, dict) else None,
                        "score_mode": bbb2.get("score_mode") if isinstance(bbb2, dict) else None,
                    },
                    "bbb_best": {
                        "horizon_weeks": best.get("horizon_weeks"),
                        "net_win_rate_shrunk": best.get("net_win_rate_shrunk", best.get("win_rate_shrunk")),
                        "net_implied_ann": best.get("net_implied_ann", best.get("implied_ann")),
                        "worst_mae": best.get("worst_mae"),
                    },
                    "exit": {"suggestion": (exit2.get("suggestion") if isinstance(exit2, dict) else None)},
                }
            )

        report = build_report_v1(
            cmd="scan-etf",
            run_meta=run_meta,
            run_config=run_config,
            artifacts={
                "run_meta": "run_meta.json",
                "run_config": "run_config.json",
                "top_trend": "top_trend.json",
                "top_swing": "top_swing.json",
                "top_bbb": "top_bbb.json",
                "position_plan": "position_plan.json" if plan_obj is not None else None,
                "position_plan_error": "position_plan_error.json" if plan_err else None,
                "all_results": "all_results.csv",
                "errors": "errors.json",
            },
            counts={
                "universe": int(total),
                "results": int(len(results)),
                "errors": int(len(errors)),
            },
            summary={
                "top_bbb": top_bbb_brief,
                "position_plan": {"plans": (plan_obj.get("plans") if isinstance(plan_obj, dict) else None), "error": plan_err},
            },
            extra={
                "as_of": as_of,
                "market_regime": regime_dict,
                "market_regime_error": regime_error,
                "market_regime_index": regime_index_eff,
            },
        )
        write_json(out_dir / "report.json", report)
    except (AttributeError) as exc:  # noqa: BLE001
        diag.record("write_report_json", exc, note="写出 report.json 失败（不影响 scan 主流程）")

    # diagnostics：给排查留证据（比 stdout 可靠）
    if errors:
        diag.warn(f"scan-etf errors={len(errors)}（详见 errors.json）", dedupe_key="scan_etf.errors_count")
    if plan_err:
        diag.warn(f"position_plan failed: {plan_err}", dedupe_key="scan_etf.position_plan")
    diag.write(out_dir, cmd="scan-etf")

    print(str(out_dir.resolve()))
    return 0



def cmd_scan_stock(args: argparse.Namespace) -> int:
    def _to_prefixed_symbol(token: str) -> str | None:
        t = str(token or "").strip().lower()
        if not t:
            return None
        if t.startswith("#"):
            return None
        if len(t) == 8 and t[:2] in {"sh", "sz", "bj"} and t[2:].isdigit():
            return t
        if len(t) == 6 and t.isdigit():
            if t.startswith("6"):
                return f"sh{t}"
            if t.startswith(("0", "3")):
                return f"sz{t}"
            if t.startswith("9"):
                return f"bj{t}"
        return None

    def _parse_symbols_from_args() -> set[str]:
        out: set[str] = set()
        raw = str(getattr(args, "symbols", "") or "")
        if raw.strip():
            for part in raw.replace("\n", " ").replace("\t", " ").replace(",", " ").split():
                s = _to_prefixed_symbol(part)
                if s:
                    out.add(s)

        fp = getattr(args, "symbols_file", None)
        if fp:
            try:
                txt = Path(str(fp)).read_text(encoding="utf-8")
            except OSError:
                txt = ""
            for line in txt.splitlines():
                line2 = line.strip()
                if not line2 or line2.startswith("#"):
                    continue
                for part in line2.replace(",", " ").split():
                    s = _to_prefixed_symbol(part)
                    if s:
                        out.add(s)
        return out

    symbols_set = _parse_symbols_from_args()

    try:
        universe_all = load_stock_universe(include_st=bool(args.include_st), include_bj=not bool(args.exclude_bj))
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"全A列表拉取失败：{exc}") from exc

    universe = universe_all
    if symbols_set:
        idx = {x.symbol: x for x in universe_all}
        universe = [idx[s] for s in sorted(symbols_set) if s in idx]

        missing = sorted([s for s in symbols_set if s not in idx])
        if missing and bool(getattr(args, "verbose", False)):
            print(f"[scan-stock] symbols_file/symbols 里有 {len(missing)} 个不在全A列表/被过滤：{missing[:10]}")

    if not universe:
        if symbols_set:
            raise SystemExit("symbols/symbols-file 过滤后 universe 为空：要么符号写错，要么被 include_st/exclude_bj 规则过滤。")
        raise SystemExit("全A列表为空：AkShare 没给数据，或者源站抽风。")

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    workers = int(args.workers) if args.workers else 8
    workers = max(1, min(workers, 32))

    min_amount = float(args.min_amount) if args.min_amount is not None else 0.0
    min_price = float(args.min_price) if getattr(args, "min_price", None) is not None else 0.0
    max_price = float(args.max_price) if getattr(args, "max_price", None) is not None else 0.0
    # Phase2：OpportunityScore 过滤（0~1；默认 0=不过滤）
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    min_score = max(0.0, min(float(min_score), 1.0))
    min_trades = int(args.min_trades) if args.min_trades is not None else 0
    rank_horizon = int(args.rank_horizon) if args.rank_horizon else 8

    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    regime_dict, regime_error, regime_index_eff = _compute_market_regime_payload(regime_index, canary_downgrade=regime_canary)

    try:
        from ..strategy_registry import parse_strategy_list, set_trend_template_params

        base_filters = parse_strategy_list(getattr(args, "base_filters", None))
        set_trend_template_params(
            near_high=float(getattr(args, "tt_near_high", 0.25)),
            above_low=float(getattr(args, "tt_above_low", 0.30)),
            slope_weeks=int(getattr(args, "tt_slope_weeks", 4)),
        )
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        base_filters = []

    horizons: list[int] = []
    if args.horizons:
        for part in str(args.horizons).split(","):
            part2 = part.strip()
            if not part2:
                continue
            try:
                horizons.append(int(part2))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
    horizons = sorted({h for h in horizons if h > 0}) or [4, 8, 12]

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"stock_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = Diagnostics()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("data") / "cache" / "stock"
    cache_ttl_hours = float(args.cache_ttl_hours) if args.cache_ttl_hours is not None else 24.0

    results: list[dict] = []
    errors: list[dict] = []
    filtered: list[dict] = []
    filtered_by_min_score = 0

    total = len(universe)

    def run_one(item):
        return analyze_stock_symbol(
            item,
            freq=str(args.freq),
            window=int(args.window),
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
            daily_filter=str(args.daily_filter),
            base_filters=base_filters,
            horizons=horizons,
            rank_horizon=rank_horizon,
            buy_cost=float(args.buy_cost),
            sell_cost=float(args.sell_cost),
            min_weeks=int(args.min_weeks),
            non_overlapping=not bool(args.allow_overlap),
            include_samples=False,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(run_one, item): item for item in universe}
        for fut in as_completed(fut_map):
            item = fut_map[fut]
            done += 1
            try:
                r = fut.result()
            except (AttributeError) as exc:  # noqa: BLE001
                r = {"symbol": item.symbol, "name": item.name, "error": str(exc)}

            if r.get("error"):
                errors.append(r)
                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} ERROR: {r.get('error')}")
                continue

            if bool(r.get("filtered")):
                filtered.append(r)
                if args.verbose:
                    rs = r.get("filter_reason") or []
                    rs2 = ",".join([str(x) for x in rs]) if isinstance(rs, list) else str(rs)
                    print(f"[{done}/{total}] {item.symbol} {item.name} skip(quality_gate:{rs2})")
                continue

            if min_amount > 0:
                try:
                    amt = float(r.get("amount") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    amt = 0.0
                if amt < min_amount:
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(amount<{min_amount:g})")
                    continue

            if min_price > 0 or max_price > 0:
                try:
                    close = float(r.get("close") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    close = 0.0
                if close <= 0:
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(close<=0)")
                    continue
                if min_price > 0 and close < min_price:
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(close<{min_price:g})")
                    continue
                if max_price > 0 and close > max_price:
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(close>{max_price:g})")
                    continue

            if min_score > 0:
                try:
                    sc = float(r.get("opp_score") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    sc = 0.0
                if float(sc) < float(min_score):
                    filtered_by_min_score += 1
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(score<{min_score:g})")
                    continue

            results.append(r)
            if args.verbose:
                sig = r.get("signals") or {}
                flag = []
                if sig.get("trend"):
                    flag.append("trend")
                if sig.get("swing"):
                    flag.append("swing")
                if sig.get("dip"):
                    flag.append("dip")
                flag2 = ",".join(flag) if flag else "-"
                print(f"[{done}/{total}] {item.symbol} {item.name} ok signals={flag2}")

    def key_score(which: str):
        def _k(x: dict):
            s = (x.get("scores") or {}).get(which)
            try:
                return float(s or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return 0.0

        return _k

    def get_trades(x: dict, which: str) -> int:
        key = f"{rank_horizon}w"
        st = (x.get("forward") or {}).get(which, {}).get(key, {})
        try:
            return int(st.get("trades") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 0

    def is_trend_context(x: dict) -> bool:
        lv = x.get("levels") or {}
        try:
            close = float(x.get("close") or 0.0)
            ma50 = float(lv.get("ma50") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return False
        if close <= 0 or ma50 <= 0:
            return False
        # MA200 软过滤：这里只留 MA50 做兜底
        return close > ma50

    def pick_top(items: list[dict], which: str) -> list[dict]:
        return sorted(items, key=key_score(which), reverse=True)[:top_k]

    # 主榜单：当前触发信号（买入参考）
    trend_now_all = [x for x in results if (x.get("signals") or {}).get("trend")]
    swing_now_all = [x for x in results if (x.get("signals") or {}).get("swing")]
    dip_now_all = [x for x in results if (x.get("signals") or {}).get("dip")]

    # 过滤低样本：默认只用于“输出榜单”，不再把结果整批丢掉
    if min_trades > 0:
        trend_now = [x for x in trend_now_all if get_trades(x, "trend") >= min_trades]
        swing_now = [x for x in swing_now_all if get_trades(x, "swing") >= min_trades]
        dip_now = [x for x in dip_now_all if get_trades(x, "dip") >= min_trades]
    else:
        trend_now = trend_now_all
        swing_now = swing_now_all
        dip_now = dip_now_all

    top_k = int(args.top_k) if args.top_k else 30
    top_k = max(5, min(top_k, 200))

    trend_mode = "signal_now"
    swing_mode = "signal_now"
    dip_mode = "signal_now"
    trend_items = trend_now
    swing_items = swing_now
    dip_items = dip_now

    # 如果“当前信号榜单”为空，给个 fallback，别让你以为程序坏了
    if not trend_items:
        if trend_now_all:
            trend_mode = "signal_now_low_sample"
            trend_items = trend_now_all
        else:
            trend_mode = "watchlist"
            cand = [x for x in results if is_trend_context(x)]
            if min_trades > 0:
                cand2 = [x for x in cand if get_trades(x, "trend") >= min_trades]
                cand = cand2 or cand
            trend_items = cand or results

    def is_dip_context(x: dict) -> bool:
        lv = x.get("levels") or {}
        try:
            close = float(x.get("close") or 0.0)
            ma50 = float(lv.get("ma50") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return False
        if close <= 0 or ma50 <= 0:
            return False
        # “捡漏”也得讲位置：不能离 MA50 太远
        return close <= ma50 * 1.10

    if not swing_items:
        if swing_now_all:
            swing_mode = "signal_now_low_sample"
            swing_items = swing_now_all
        else:
            swing_mode = "watchlist"
            cand = [x for x in results if is_trend_context(x)]
            if min_trades > 0:
                cand2 = [x for x in cand if get_trades(x, "swing") >= min_trades]
                cand = cand2 or cand
            swing_items = cand or results

    if not dip_items:
        if dip_now_all:
            dip_mode = "signal_now_low_sample"
            dip_items = dip_now_all
        else:
            dip_mode = "watchlist"
            cand = [x for x in results if is_dip_context(x)]
            if min_trades > 0:
                cand2 = [x for x in cand if get_trades(x, "dip") >= min_trades]
                cand = cand2 or cand
            dip_items = cand or results

    write_json(
        out_dir / "top_trend.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_score": float(min_score) if min_score > 0 else None,
            "market_regime": regime_dict,
            "market_regime_error": regime_error,
            "market_regime_index": regime_index_eff,
            "rank_horizon_weeks": rank_horizon,
            "horizons": horizons,
            "base_filters": base_filters,
            "trend_template": {
                "near_high": float(getattr(args, "tt_near_high", 0.25)),
                "above_low": float(getattr(args, "tt_above_low", 0.30)),
                "slope_weeks": int(getattr(args, "tt_slope_weeks", 4)),
            },
            "mode": trend_mode,
            "counts": {
                "results": len(results),
                "errors": len(errors),
                "filtered": len(filtered),
                "filtered_by_min_score": int(filtered_by_min_score),
                "signal_now": len(trend_now_all),
            },
            "min_trades": min_trades,
            "items": pick_top(trend_items, "trend"),
        },
    )
    write_json(
        out_dir / "top_swing.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_score": float(min_score) if min_score > 0 else None,
            "market_regime": regime_dict,
            "market_regime_error": regime_error,
            "market_regime_index": regime_index_eff,
            "rank_horizon_weeks": rank_horizon,
            "horizons": horizons,
            "base_filters": base_filters,
            "trend_template": {
                "near_high": float(getattr(args, "tt_near_high", 0.25)),
                "above_low": float(getattr(args, "tt_above_low", 0.30)),
                "slope_weeks": int(getattr(args, "tt_slope_weeks", 4)),
            },
            "mode": swing_mode,
            "counts": {
                "results": len(results),
                "errors": len(errors),
                "filtered": len(filtered),
                "filtered_by_min_score": int(filtered_by_min_score),
                "signal_now": len(swing_now_all),
            },
            "min_trades": min_trades,
            "items": pick_top(swing_items, "swing"),
        },
    )
    write_json(
        out_dir / "top_dip.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_score": float(min_score) if min_score > 0 else None,
            "market_regime": regime_dict,
            "market_regime_error": regime_error,
            "market_regime_index": regime_index_eff,
            "rank_horizon_weeks": rank_horizon,
            "horizons": horizons,
            "base_filters": base_filters,
            "trend_template": {
                "near_high": float(getattr(args, "tt_near_high", 0.25)),
                "above_low": float(getattr(args, "tt_above_low", 0.30)),
                "slope_weeks": int(getattr(args, "tt_slope_weeks", 4)),
            },
            "mode": dip_mode,
            "counts": {
                "results": len(results),
                "errors": len(errors),
                "filtered": len(filtered),
                "filtered_by_min_score": int(filtered_by_min_score),
                "signal_now": len(dip_now_all),
            },
            "min_trades": min_trades,
            "items": pick_top(dip_items, "dip"),
        },
    )
    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})
    write_json(out_dir / "filtered.json", {"generated_at": datetime.now().isoformat(), "filtered": filtered})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8")

    # signals.json：给组合层/回测层吃的“中间产物”（只输出当前触发信号的标的，避免塞爆文件）。
    try:
        from ..signals import signals_from_stock_scan_results

        sig_cfg = {
            "freq": str(args.freq),
            "rank_horizon_weeks": int(rank_horizon),
            "horizons": list(horizons),
            "base_filters": list(base_filters or []),
            "trend_template": {
                "near_high": float(getattr(args, "tt_near_high", 0.25)),
                "above_low": float(getattr(args, "tt_above_low", 0.30)),
                "slope_weeks": int(getattr(args, "tt_slope_weeks", 4)),
            },
            "filters": {
                "min_amount": float(min_amount) if min_amount > 0 else None,
                "min_price": float(min_price) if min_price > 0 else None,
                "max_price": float(max_price) if max_price > 0 else None,
                "min_trades": int(min_trades) if min_trades > 0 else None,
            },
            "notes": "scan-stock 输出包含 3 个信号：trend/swing/dip；signals.json 仅收口“当前触发”的候选",
        }
        write_json(
            out_dir / "signals.json",
            signals_from_stock_scan_results(
                results,
                generated_at=datetime.now().isoformat(),
                rank_horizon_weeks=int(rank_horizon),
                market_regime=regime_dict if isinstance(regime_dict, dict) else None,
                config=sig_cfg,
            ),
        )
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        diag.record("write_signals_json", exc, note="写出 signals.json 失败（可能影响组合层输入）")

    # diagnostics：给排查留证据（比 stdout 可靠）
    if errors:
        diag.warn(f"scan-stock errors={len(errors)}（详见 errors.json）", dedupe_key="scan_stock.errors_count")
    diag.write(out_dir, cmd="scan-stock")

    print(str(out_dir.resolve()))
    return 0


def cmd_scan_short(args: argparse.Namespace) -> int:
    """
    周内短线扫描（强势股 + 洗盘缩量回调 + T+1~T+3 纪律）。
    """
    # 先算大盘牛熊：熊市默认不做 shortline，别把自己磨死（想硬刚就 --allow-bear）
    allow_bear = bool(getattr(args, "allow_bear", False))
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    regime_dict, regime_error, regime_index_eff = _compute_market_regime_payload(regime_index, canary_downgrade=regime_canary)
    regime_label = str((regime_dict or {}).get("label") or "unknown").strip().lower()

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"short_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = Diagnostics()

    if (not allow_bear) and regime_label == "bear":
        write_json(
            out_dir / "top_short.json",
            {
                "generated_at": datetime.now().isoformat(),
                "strategy": "shortline_t1t3",
                "blocked": True,
                "blocked_reason": "market_regime=bear",
                "market_regime": regime_dict,
                "market_regime_error": regime_error,
                "market_regime_index": regime_index_eff,
                "counts": {"candidates": 0, "watch": 0, "errors": 0, "scanned": 0},
                "items": [],
            },
        )
        write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": []})
        print(str(out_dir.resolve()))
        return 0

    # universe：默认用“涨停股池”预筛，别扫全A把源站薅炸；需要全量就显式 --universe all
    universe_meta: dict | None = None
    if getattr(args, "universe", "zt_pool") == "all":
        try:
            universe = load_stock_universe(include_st=bool(args.include_st), include_bj=not bool(args.exclude_bj))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"全A列表拉取失败：{exc}") from exc

        if not universe:
            raise SystemExit("全A列表为空：AkShare 没给数据，或者源站抽风。")
        universe_meta = {"mode": "all_a", "unique_symbols": int(len(universe))}
    else:
        from ..shortline_universe import load_recent_limitup_universe

        prefilter_days = int(getattr(args, "limitup_lookback_days", 10) or 10)
        max_after = int(getattr(args, "max_days_after_limitup", 7) or 7)
        prefilter_days = max(1, min(prefilter_days, max_after))

        try:
            items, meta = load_recent_limitup_universe(
                end_date=args.end_date,
                lookback_trading_days=prefilter_days,
                include_st=bool(args.include_st),
                include_bj=not bool(args.exclude_bj),
            )
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"涨停股池拉取失败：{exc}") from exc

        if not items:
            raise SystemExit("涨停股池为空：要么近期没涨停（不太可能），要么源站抽风。")

        # 适配后续代码：只要 symbol/name 字段
        # 同时：默认优先扫“更老一点的涨停”（更可能进入洗盘回踩阶段），否则 --limit 取到的全是当日/次日连板没法低吸。
        universe = sorted(items, key=lambda x: (x.last_limitup_date, x.symbol))
        universe_meta = dict(meta)

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    workers = int(args.workers) if args.workers else 8
    workers = max(1, min(workers, 32))

    min_price = float(args.min_price) if getattr(args, "min_price", None) is not None else 0.0
    max_price = float(args.max_price) if getattr(args, "max_price", None) is not None else 0.0
    min_amount_avg20 = float(getattr(args, "min_amount_avg20", 0.0) or 0.0)
    min_trades = int(getattr(args, "min_trades", 0) or 0)
    min_win_rate = float(getattr(args, "min_win_rate", 0.0) or 0.0)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("data") / "cache" / "stock"
    cache_ttl_hours = float(args.cache_ttl_hours) if args.cache_ttl_hours is not None else 12.0

    from ..shortline import ShortlineParams
    from ..shortline_scan import analyze_shortline_symbol

    params = ShortlineParams(
        limitup_lookback_days=int(getattr(args, "limitup_lookback_days", 10) or 10),
        max_days_after_limitup=int(getattr(args, "max_days_after_limitup", 7) or 7),
        pullback_max=float(getattr(args, "pullback_max", 0.12) or 0.12),
        vol_shrink_days=int(getattr(args, "vol_shrink_days", 3) or 3),
        vol_shrink_ratio=float(getattr(args, "vol_shrink_ratio", 0.70) or 0.70),
        min_yang_ratio=float(getattr(args, "min_yang_ratio", 0.50) or 0.50),
        support_ma=int(getattr(args, "support_ma", 10) or 10),
        support_tol=float(getattr(args, "support_tol", 0.02) or 0.02),
        require_pullback_day=not bool(getattr(args, "allow_red_day", False)),
    )

    target_ret = float(getattr(args, "target_ret", 0.05) or 0.05)
    max_hold_days = int(getattr(args, "max_hold_days", 3) or 3)
    min_hold_days = int(getattr(args, "min_hold_days", 1) or 1)
    stop_loss_ret = float(getattr(args, "stop_loss_ret", 0.0) or 0.0)
    stop_loss_ret2 = None if stop_loss_ret <= 0 else float(stop_loss_ret)

    buy_cost = float(getattr(args, "buy_cost", 0.001) or 0.001)
    sell_cost = float(getattr(args, "sell_cost", 0.002) or 0.002)
    non_overlapping = not bool(getattr(args, "allow_overlap", False))

    results: list[dict] = []
    errors: list[dict] = []
    watch: list[dict] = []
    fail_reason_counts: dict[str, int] = {}

    total = len(universe)

    def run_one(item):
        return analyze_shortline_symbol(
            symbol=item.symbol,
            name=item.name,
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
            params=params,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            target_ret=target_ret,
            max_hold_days=max_hold_days,
            min_hold_days=min_hold_days,
            stop_loss_ret=stop_loss_ret2,
            non_overlapping=non_overlapping,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(run_one, item): item for item in universe}
        for fut in as_completed(fut_map):
            item = fut_map[fut]
            done += 1
            try:
                r = fut.result()
            except (AttributeError) as exc:  # noqa: BLE001
                r = {"symbol": item.symbol, "name": item.name, "error": str(exc)}

            if r.get("error"):
                errors.append(r)
                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} ERROR: {r.get('error')}")
                continue

            if not bool((r.get("shortline") or {}).get("ok")):
                # 统计失败原因，方便你调参（不然一堆 skip 你以为程序坏了）
                for reason in (r.get("shortline") or {}).get("fails") or []:
                    try:
                        k = str(reason)
                    except (AttributeError):  # noqa: BLE001
                        continue
                    fail_reason_counts[k] = int(fail_reason_counts.get(k, 0)) + 1

                # “涨停窗口内但没到位”的，丢到 watchlist（给你盯盘/明天再看）
                lu_dt = (r.get("shortline") or {}).get("limitup_date")
                if lu_dt:
                    # 价格过滤
                    try:
                        close = float(r.get("close") or 0.0)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        close = 0.0
                    if close > 0:
                        if (min_price <= 0 or close >= min_price) and (max_price <= 0 or close <= max_price):
                            if min_amount_avg20 > 0:
                                liq = r.get("liquidity") or {}
                                try:
                                    amt20 = float(liq.get("amount_avg20") or 0.0)
                                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                                    amt20 = 0.0
                                if amt20 >= min_amount_avg20:
                                    watch.append(r)
                            else:
                                watch.append(r)

                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} skip(no_signal)")
                continue

            # 价格过滤
            try:
                close = float(r.get("close") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                close = 0.0
            if close <= 0:
                continue
            if min_price > 0 and close < min_price:
                continue
            if max_price > 0 and close > max_price:
                continue

            # 流动性过滤（20日均成交额）
            if min_amount_avg20 > 0:
                liq = r.get("liquidity") or {}
                try:
                    amt20 = float(liq.get("amount_avg20") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    amt20 = 0.0
                if amt20 < min_amount_avg20:
                    continue

            # 回测门槛：样本数 + 收缩胜率
            bt = r.get("shortline_bt") or {}
            try:
                trades = int(bt.get("trades") or 0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                trades = 0
            try:
                wr_s = float(bt.get("win_rate_shrunk") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                wr_s = 0.0

            if min_trades > 0 and trades < min_trades:
                continue
            if min_win_rate > 0 and wr_s < min_win_rate:
                continue

            results.append(r)
            if args.verbose:
                lu = str((r.get("shortline") or {}).get("limitup_date") or "")
                print(f"[{done}/{total}] {item.symbol} {item.name} HIT limitup={lu} score={r.get('score')}")

    # 排名：按 score 倒序
    def key_score(x: dict):
        try:
            return float(x.get("score") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 0.0

    results_sorted = sorted(results, key=key_score, reverse=True)

    # watchlist：按“更接近满足条件”排序（fails 少 + 涨停更近）
    def _watch_key(x: dict):
        sl = x.get("shortline") or {}
        fails = sl.get("fails") or []
        try:
            nfail = int(len(fails))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            nfail = 999
        days = sl.get("days_since_limitup")
        try:
            d = int(days) if days is not None else 999
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            d = 999
        # 成交额越大越靠前（更好进出）
        liq = x.get("liquidity") or {}
        try:
            amt20 = float(liq.get("amount_avg20") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            amt20 = 0.0
        return (nfail, d, -amt20)

    watch_sorted = sorted(watch, key=_watch_key)

    top_k = int(args.top_k) if args.top_k else 50
    top_k = max(5, min(top_k, 200))

    # fail reason topN
    fails_top = sorted(fail_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]

    top_short_obj = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "shortline_t1t3",
        "market_regime": regime_dict,
        "market_regime_error": regime_error,
        "market_regime_index": regime_index_eff,
        "universe": universe_meta,
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "adjust": args.adjust,
            "target_ret": float(target_ret),
            "min_hold_days": int(min_hold_days),
            "max_hold_days": int(max_hold_days),
            "stop_loss_ret": stop_loss_ret2,
            "buy_cost": float(buy_cost),
            "sell_cost": float(sell_cost),
            "non_overlapping": bool(non_overlapping),
            "signal": {
                "limitup_lookback_days": int(params.limitup_lookback_days),
                "max_days_after_limitup": int(params.max_days_after_limitup),
                "pullback_max": float(params.pullback_max),
                "vol_shrink_days": int(params.vol_shrink_days),
                "vol_shrink_ratio": float(params.vol_shrink_ratio),
                "min_yang_ratio": float(params.min_yang_ratio),
                "support_ma": int(params.support_ma),
                "support_tol": float(params.support_tol),
                "require_pullback_day": bool(params.require_pullback_day),
            },
            "filters": {
                "min_price": float(min_price) if min_price > 0 else None,
                "max_price": float(max_price) if max_price > 0 else None,
                "min_amount_avg20": float(min_amount_avg20) if min_amount_avg20 > 0 else None,
                "min_trades": int(min_trades) if min_trades > 0 else None,
                "min_win_rate_shrunk": float(min_win_rate) if min_win_rate > 0 else None,
            },
        },
        "counts": {"candidates": len(results), "watch": len(watch_sorted), "errors": len(errors), "scanned": total},
        "diagnostics": {"fails_top": [{"reason": k, "count": int(v)} for k, v in fails_top]},
        "items": results_sorted[:top_k],
    }
    write_json(out_dir / "top_short.json", top_short_obj)

    # 统一 signals schema（研究用途）：给组合层/执行层消费。
    try:
        from ..signals import signals_from_top_short

        write_json(out_dir / "signals.json", signals_from_top_short(top_short_obj))
    except (AttributeError) as exc:  # noqa: BLE001
        diag.record("write_signals_json", exc, note="写出 signals.json 失败（可能影响组合层输入）")
    # watchlist 单独输出：只盯“涨停窗口内”的近似机会
    if watch_sorted:
        watch_k = min(200, max(20, int(top_k)))

        def _pick_watch(x: dict) -> dict:
            sl = x.get("shortline") or {}
            liq = x.get("liquidity") or {}
            return {
                "symbol": x.get("symbol"),
                "name": x.get("name"),
                "date": x.get("date"),
                "close": x.get("close"),
                "pct_chg": x.get("pct_chg"),
                "liquidity": {"amount_avg20": liq.get("amount_avg20")},
                "limitup_date": sl.get("limitup_date"),
                "days_since_limitup": sl.get("days_since_limitup"),
                "fails": sl.get("fails"),
                "metrics": sl.get("metrics"),
            }

        write_json(
            out_dir / "watch_limitup.json",
            {
                "generated_at": datetime.now().isoformat(),
                "strategy": "shortline_watch_limitup",
                "market_regime": regime_dict,
                "market_regime_error": regime_error,
                "market_regime_index": regime_index_eff,
                "universe": universe_meta,
                "counts": {"watch": len(watch_sorted), "scanned": total},
                "items": [_pick_watch(x) for x in watch_sorted[:watch_k]],
            },
        )
    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df_out = pd.DataFrame(results_sorted)
        if not df_out.empty:
            df_out.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8")

    # diagnostics：给排查留证据（比 stdout 可靠）
    if errors:
        diag.warn(f"scan-short errors={len(errors)}（详见 errors.json）", dedupe_key="scan_short.errors_count")
    diag.write(out_dir, cmd="scan-short")

    print(str(out_dir.resolve()))
    return 0


def cmd_scan_sunrise(args: argparse.Namespace) -> int:
    """
    旭日东升（两K反转/看涨吞没）扫描（研究用途）。
    """
    # 大盘牛熊：默认熊市禁入（旭日东升在熊市容易变“下跌中继”的反抽幻觉）
    allow_bear = bool(getattr(args, "allow_bear", False))
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    regime_dict, regime_error, regime_index_eff = _compute_market_regime_payload(regime_index, canary_downgrade=regime_canary)
    regime_label = str((regime_dict or {}).get("label") or "unknown").strip().lower()

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"sunrise_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = Diagnostics()

    if (not allow_bear) and regime_label == "bear":
        write_json(
            out_dir / "top_sunrise.json",
            {
                "generated_at": datetime.now().isoformat(),
                "strategy": "sunrise_engulfing",
                "blocked": True,
                "blocked_reason": "market_regime=bear",
                "market_regime": regime_dict,
                "market_regime_error": regime_error,
                "market_regime_index": regime_index_eff,
                "counts": {"candidates": 0, "errors": 0, "scanned": 0},
                "items": [],
            },
        )
        write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": []})
        diag.warn("market_regime=bear: blocked (use --allow-bear to override)", dedupe_key="sunrise.blocked_bear")
        diag.write(out_dir, cmd="scan-sunrise")
        print(str(out_dir.resolve()))
        return 0

    # universe：全A。旭日东升信号泛滥，别搞花里胡哨的“伪聪明”预筛，先把规则做扎实再说。
    try:
        universe = load_stock_universe(include_st=bool(args.include_st), include_bj=not bool(args.exclude_bj))
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"全A列表拉取失败：{exc}") from exc
    if not universe:
        raise SystemExit("全A列表为空：AkShare 没给数据，或者源站抽风。")
    universe_meta: dict[str, Any] = {"mode": "all_a", "unique_symbols": int(len(universe))}

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    workers = int(args.workers) if args.workers else 8
    workers = max(1, min(workers, 32))

    min_price = float(args.min_price) if getattr(args, "min_price", None) is not None else 0.0
    max_price = float(args.max_price) if getattr(args, "max_price", None) is not None else 0.0
    min_amount_avg20 = float(getattr(args, "min_amount_avg20", 0.0) or 0.0)
    min_trades = int(getattr(args, "min_trades", 0) or 0)
    min_win_rate = float(getattr(args, "min_win_rate", 0.0) or 0.0)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("data") / "cache" / "stock"
    cache_ttl_hours = float(args.cache_ttl_hours) if args.cache_ttl_hours is not None else 12.0

    from ..sunrise import SunriseParams
    from ..sunrise_scan import analyze_sunrise_symbol

    params = SunriseParams(
        trend_lookback_days=int(getattr(args, "trend_lookback_days", 10) or 10),
        down_ret_min=float(getattr(args, "down_ret_min", 0.05) or 0.05),
        min_body_pct=float(getattr(args, "min_body_pct", 0.003) or 0.003),
        engulf_tol=float(getattr(args, "engulf_tol", 0.001) or 0.001),
        body_engulf_ratio=float(getattr(args, "body_engulf_ratio", 1.0) or 1.0),
        near_low_days=int(getattr(args, "near_low_days", 30) or 30),
        near_low_tol=float(getattr(args, "near_low_tol", 0.06) or 0.06),
        require_volume_increase=bool(getattr(args, "require_volume_increase", False)),
        vol_ratio_min=float(getattr(args, "vol_ratio_min", 1.10) or 1.10),
    )

    target_ret = float(getattr(args, "target_ret", 0.05) or 0.05)
    max_hold_days = int(getattr(args, "max_hold_days", 3) or 3)
    min_hold_days = int(getattr(args, "min_hold_days", 1) or 1)
    stop_loss_ret = float(getattr(args, "stop_loss_ret", 0.0) or 0.0)
    stop_loss_ret2 = None if stop_loss_ret <= 0 else float(stop_loss_ret)

    buy_cost = float(getattr(args, "buy_cost", 0.001) or 0.001)
    sell_cost = float(getattr(args, "sell_cost", 0.002) or 0.002)
    non_overlapping = not bool(getattr(args, "allow_overlap", False))

    results: list[dict] = []
    errors: list[dict] = []
    fail_reason_counts: dict[str, int] = {}

    total = len(universe)

    def run_one(item):
        return analyze_sunrise_symbol(
            symbol=item.symbol,
            name=item.name,
            start_date=args.start_date,
            end_date=args.end_date,
            adjust=args.adjust,
            params=params,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            target_ret=target_ret,
            max_hold_days=max_hold_days,
            min_hold_days=min_hold_days,
            stop_loss_ret=stop_loss_ret2,
            non_overlapping=non_overlapping,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(run_one, item): item for item in universe}
        for fut in as_completed(fut_map):
            item = fut_map[fut]
            done += 1
            try:
                r = fut.result()
            except (AttributeError) as exc:  # noqa: BLE001
                r = {"symbol": item.symbol, "name": item.name, "error": str(exc)}

            if r.get("error"):
                errors.append(r)
                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} ERROR: {r.get('error')}")
                continue

            if not bool((r.get("sunrise") or {}).get("ok")):
                for reason in (r.get("sunrise") or {}).get("fails") or []:
                    try:
                        k = str(reason)
                    except (AttributeError):  # noqa: BLE001
                        continue
                    fail_reason_counts[k] = int(fail_reason_counts.get(k, 0)) + 1
                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} skip(no_signal)")
                continue

            # 价格过滤
            try:
                close = float(r.get("close") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                close = 0.0
            if close <= 0:
                continue
            if min_price > 0 and close < min_price:
                continue
            if max_price > 0 and close > max_price:
                continue

            # 流动性过滤（20日均成交额）
            if min_amount_avg20 > 0:
                liq = r.get("liquidity") or {}
                try:
                    amt20 = float(liq.get("amount_avg20") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    amt20 = 0.0
                if amt20 < min_amount_avg20:
                    continue

            # 回测门槛：样本数 + 收缩胜率
            bt = r.get("sunrise_bt") or {}
            try:
                trades = int(bt.get("trades") or 0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                trades = 0
            try:
                wr_s = float(bt.get("win_rate_shrunk") or 0.0)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                wr_s = 0.0

            if min_trades > 0 and trades < min_trades:
                continue
            if min_win_rate > 0 and wr_s < min_win_rate:
                continue

            results.append(r)
            if args.verbose:
                print(f"[{done}/{total}] {item.symbol} {item.name} HIT score={r.get('score')}")

    def key_score(x: dict):
        try:
            return float(x.get("score") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            return 0.0

    results_sorted = sorted(results, key=key_score, reverse=True)

    top_k = int(args.top_k) if args.top_k else 50
    top_k = max(5, min(top_k, 200))

    fails_top = sorted(fail_reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]

    top_sunrise_obj = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "sunrise_engulfing",
        "market_regime": regime_dict,
        "market_regime_error": regime_error,
        "market_regime_index": regime_index_eff,
        "universe": universe_meta,
        "config": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "adjust": args.adjust,
            "trade": {
                "target_ret": float(target_ret),
                "min_hold_days": int(min_hold_days),
                "max_hold_days": int(max_hold_days),
                "stop_loss_ret": stop_loss_ret2,
                "buy_cost": float(buy_cost),
                "sell_cost": float(sell_cost),
                "non_overlapping": bool(non_overlapping),
            },
            "signal": {
                "trend_lookback_days": int(params.trend_lookback_days),
                "down_ret_min": float(params.down_ret_min),
                "min_body_pct": float(params.min_body_pct),
                "engulf_tol": float(params.engulf_tol),
                "body_engulf_ratio": float(params.body_engulf_ratio),
                "near_low_days": int(params.near_low_days),
                "near_low_tol": float(params.near_low_tol),
                "require_volume_increase": bool(params.require_volume_increase),
                "vol_ratio_min": float(params.vol_ratio_min),
            },
            "filters": {
                "min_price": float(min_price) if min_price > 0 else None,
                "max_price": float(max_price) if max_price > 0 else None,
                "min_amount_avg20": float(min_amount_avg20) if min_amount_avg20 > 0 else None,
                "min_trades": int(min_trades) if min_trades > 0 else None,
                "min_win_rate_shrunk": float(min_win_rate) if min_win_rate > 0 else None,
            },
        },
        "counts": {"candidates": len(results_sorted), "errors": len(errors), "scanned": total},
        "diagnostics": {"fails_top": [{"reason": k, "count": int(v)} for k, v in fails_top]},
        "items": results_sorted[:top_k],
    }
    write_json(out_dir / "top_sunrise.json", top_sunrise_obj)

    # 统一 signals schema（研究用途）：给组合层/执行层消费。
    try:
        from ..signals import signals_from_top_sunrise

        write_json(out_dir / "signals.json", signals_from_top_sunrise(top_sunrise_obj))
    except (AttributeError) as exc:  # noqa: BLE001
        diag.record("write_signals_json", exc, note="写出 signals.json 失败（可能影响组合层输入）")
    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df_out = pd.DataFrame(results_sorted)
        if not df_out.empty:
            df_out.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8")

    # diagnostics：给排查留证据（比 stdout 可靠）
    if errors:
        diag.warn(f"scan-sunrise errors={len(errors)}（详见 errors.json）", dedupe_key="scan_sunrise.errors_count")
    diag.write(out_dir, cmd="scan-sunrise")

    print(str(out_dir.resolve()))
    return 0
