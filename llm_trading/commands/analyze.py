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
from ..logger import get_logger
from ..diagnostics import Diagnostics

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

_LOG = get_logger(__name__)

def cmd_analyze(args: argparse.Namespace) -> int:
    if args.csv:
        try:
            df = load_ohlcv_csv(
                args.csv,
                date_col=args.date_col,
                open_col=args.open_col,
                high_col=args.high_col,
                low_col=args.low_col,
                close_col=args.close_col,
                volume_col=args.volume_col,
                encoding=args.encoding,
            )
        except CsvSchemaError as exc:
            raise SystemExit(str(exc)) from exc
        default_out = _default_out_dir(args.csv)
        title = args.title or f"Wyckoff - {Path(args.csv).name}"
    else:
        try:
            df = fetch_daily(
                FetchParams(
                    asset=args.asset,
                    symbol=args.symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adjust=args.adjust,
                    source=getattr(args, "source", None),
                )
            )
        except DataSourceError as exc:
            raise SystemExit(str(exc)) from exc
        default_out = _default_out_dir_for_symbol(args.asset, args.symbol, args.freq)
        title = args.title or f"{args.asset.upper()} - {args.symbol}"

        # 留一份日线原始数据：后面算“宏观/聪明钱/微观结构”会用到（周线 resample 后就没了）
        try:
            df_daily_raw = df.copy()
        except (AttributeError):  # noqa: BLE001
            df_daily_raw = None

    source_used = None
    try:
        source_used = getattr(df, "attrs", {}).get("data_source")
    except (AttributeError):  # noqa: BLE001
        source_used = None

    if args.freq == "weekly":
        df = resample_to_weekly(df)

    if args.window and len(df) > args.window:
        df = df.tail(args.window).reset_index(drop=True)

    df = add_moving_averages(df, ma_fast=50, ma_slow=200)
    df = add_accumulation_distribution_line(df)

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = Diagnostics()
    try:
        write_json(
            out_dir / "meta.json",
            {
                "generated_at": datetime.now().isoformat(),
                "source": "csv" if args.csv else (source_used or getattr(args, "source", None) or "akshare"),
                "source_requested": None if args.csv else getattr(args, "source", None),
                "csv": args.csv if args.csv else None,
                "asset": args.asset if not args.csv else None,
                "symbol": args.symbol if not args.csv else None,
                "freq": args.freq,
                "method": args.method,
                "title": title,
                "window": int(args.window) if args.window else None,
                "rows": int(len(df)),
                "start_date": str(df["date"].min().date()),
                "end_date": str(df["date"].max().date()),
                "columns": list(df.columns),
            },
        )
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        diag.record("write_meta_json", exc, note="写出 meta.json 失败（不影响主流程）")

    # TuShare 额外因子包（ERP / HSGT north/south / 个股微观结构）
    # - 单标的分析才跑（CSV 模式不跑；scan-* 也不在这跑）
    try:
        if (not args.csv) and ("df_daily_raw" in locals()) and df_daily_raw is not None and (not getattr(df_daily_raw, "empty", True)):
            from ..pipeline import write_json as _write_json
            from ..tushare_factors import compute_tushare_factor_pack

            # as_of：用数据最后一根K线日期，别用今天（周末/节假日会乱）
            try:
                as_of_dt = df_daily_raw["date"].max()
                as_of = as_of_dt.date() if hasattr(as_of_dt, "date") else datetime.now().date()
            except (TypeError, ValueError, KeyError, AttributeError):  # noqa: BLE001
                as_of = datetime.now().date()

            # context index：默认用沪深300；如果你分析的本身就是指数，就用它自己（尽量归一成 sh/sz 前缀）
            context_index = "sh000300"
            try:
                if str(getattr(args, "asset", "")).strip().lower() == "index":
                    from ..akshare_source import resolve_symbol

                    context_index = resolve_symbol("index", str(getattr(args, "symbol", "") or "sh000300"))
            except (AttributeError):  # noqa: BLE001
                context_index = "sh000300"

            # symbol_prefixed：个股微观结构需要它
            sym_prefixed = None
            try:
                if str(getattr(args, "asset", "")).strip().lower() == "stock":
                    from ..akshare_source import resolve_symbol

                    sym_prefixed = resolve_symbol("stock", str(getattr(args, "symbol", "") or ""))
            except (AttributeError):  # noqa: BLE001
                sym_prefixed = None

            # 日成交额映射（给 moneyflow 做归一化）
            amt_map = None
            try:
                amt_map = {}
                for _, row in df_daily_raw[["date", "amount"]].dropna().iterrows():
                    d = row.get("date")
                    if d is None:
                        continue
                    dd = d.date() if hasattr(d, "date") else None
                    a = row.get("amount")
                    if dd is None:
                        continue
                    try:
                        av = float(a)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        continue
                    if av <= 0:
                        continue
                    amt_map[str(dd)] = float(av)
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                amt_map = None

            pack = compute_tushare_factor_pack(
                as_of=as_of,
                context_index_symbol_prefixed=context_index,
                symbol_prefixed=sym_prefixed,
                daily_amount_by_date=amt_map,
                cache_dir=Path("data") / "cache" / "tushare_factors",
                ttl_hours=6.0,
            )
            _write_json(out_dir / "tushare_factors.json", pack)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:  # noqa: BLE001
        try:
            (out_dir / "tushare_factors_error.txt").write_text(str(exc), encoding="utf-8")
        except (OSError, TypeError, ValueError, AttributeError) as exc2:  # noqa: BLE001
            diag.record("write_tushare_factors_error_txt", exc2, note="写出 tushare_factors_error.txt 失败（吞错治理）")
        diag.record("tushare_factors", exc, note="TuShare 因子包失败（已降级）")

    # ETF “前十大重仓”（季度披露，用于解释成分事件影响；例如 TikTok -> 蓝色光标）
    try:
        if (not args.csv) and str(getattr(args, "asset", "") or "").strip().lower() == "etf":
            from ..etf_holdings import fetch_etf_top_holdings_em
            from ..pipeline import write_json as _write_json

            # 用价格数据最后一根日线日期当“参考 as_of”，内部会自动回退到上一年找最新披露季度。
            try:
                base_df = df_daily_raw if ("df_daily_raw" in locals() and df_daily_raw is not None) else df  # type: ignore[name-defined]
                as_of_dt = base_df["date"].max()
                as_of_d = as_of_dt.date() if hasattr(as_of_dt, "date") else datetime.now().date()
            except (AttributeError):  # noqa: BLE001
                as_of_d = datetime.now().date()

            pack = fetch_etf_top_holdings_em(
                symbol=str(getattr(args, "symbol", "") or ""),
                as_of=as_of_d,
                cache_dir=Path("data") / "cache" / "etf_holdings",
                ttl_hours=24.0 * 14,  # 季度披露：缓存久一点，别天天刷网页
                top_n=10,
            )
            _write_json(out_dir / "etf_holdings_top10.json", pack)
    except (AttributeError) as exc:  # noqa: BLE001
        try:
            (out_dir / "etf_holdings_top10_error.txt").write_text(str(exc), encoding="utf-8")
        except (AttributeError) as exc2:  # noqa: BLE001
            diag.record("write_etf_holdings_top10_error_txt", exc2, note="写出 etf_holdings_top10_error.txt 失败（吞错治理）")
        diag.record("etf_holdings_top10", exc, note="ETF 前十大重仓拉取失败（已降级）")

    def ensure_ohlc(df_in):
        df_local = df_in
        if "high" not in df_local.columns or "low" not in df_local.columns:
            df_local = df_local.copy()
            df_local["open"] = df_local.get("open", df_local["close"])
            df_local["high"] = df_local["close"]
            df_local["low"] = df_local["close"]
        if "open" not in df_local.columns:
            df_local = df_local.copy()
            df_local["open"] = df_local["close"]
        return df_local

    # 胜率/磨损统计（胜率优先的“终极动作”需要这玩意儿撑腰）
    def write_signal_backtest() -> None:
        if str(args.freq) != "weekly":
            return
        try:
            import pandas as pd  # noqa: F401
        except ModuleNotFoundError:
            return

        try:
            from ..backtest import forward_holding_backtest
        except (AttributeError):  # noqa: BLE001
            return

        df_local = ensure_ohlc(df)
        df_local = add_moving_averages(df_local, ma_fast=50, ma_slow=200)
        df_local = add_donchian_channels(
            df_local, window=20, upper_col="donchian_upper_20", lower_col="donchian_lower_20", shift=1
        )

        # 跟 scan-stock 的策略保持一致：统一走 strategy_registry，别两套逻辑打架
        try:
            from ..strategy_registry import compute_series
        except (AttributeError):  # noqa: BLE001
            return

        sig_trend = compute_series(df_local, key="trend").fillna(False)
        sig_swing = compute_series(df_local, key="swing").fillna(False)

        horizons = [4, 8, 12]
        buy_cost = 0.001
        sell_cost = 0.002
        min_trades = 20
        rank_horizon = 8

        # 额外：给“持仓视角”的减仓/退出信号（KISS：周线管生死，日线管提前预警）
        weekly_below_ma50_confirm2 = False
        try:
            if len(df_local) >= 2 and "ma50" in df_local.columns:
                last = df_local.iloc[-1]
                prev = df_local.iloc[-2]
                c1 = last.get("close")
                m1 = last.get("ma50")
                c0 = prev.get("close")
                m0 = prev.get("ma50")
                weekly_below_ma50_confirm2 = bool(
                    c1 is not None
                    and m1 is not None
                    and c0 is not None
                    and m0 is not None
                    and float(m1) > 0
                    and float(m0) > 0
                    and float(c1) < float(m1)
                    and float(c0) < float(m0)
                )
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            weekly_below_ma50_confirm2 = False

        daily_macd_bearish_2d = False
        daily_close_below_ma20_2d = False
        try:
            # 仅在单标的（非CSV）模式下可用：我们在 cmd_analyze 里保留了 df_daily_raw
            try:
                dfd0 = df_daily_raw  # type: ignore[name-defined]
            except NameError:  # noqa: BLE001
                dfd0 = None

            if dfd0 is not None and (not getattr(dfd0, "empty", True)):
                from ..indicators import add_macd

                dfd = dfd0.copy()
                if "date" in dfd.columns:
                    import pandas as pd

                    dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
                    dfd = dfd.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                if len(dfd) >= 2:
                    # ma20
                    dfd["ma20"] = dfd["close"].astype(float).rolling(window=20, min_periods=20).mean()
                    dfd = add_macd(dfd, fast=12, slow=26, signal=9, macd_col="macd", signal_col="macd_signal", hist_col="macd_hist")

                    last_d = dfd.iloc[-1]
                    prev_d = dfd.iloc[-2]
                    try:
                        daily_macd_bearish_2d = bool(
                            float(prev_d.get("macd")) < float(prev_d.get("macd_signal"))
                            and float(last_d.get("macd")) < float(last_d.get("macd_signal"))
                        )
                    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                        daily_macd_bearish_2d = False

                    try:
                        daily_close_below_ma20_2d = bool(
                            prev_d.get("close") is not None
                            and prev_d.get("ma20") is not None
                            and last_d.get("close") is not None
                            and last_d.get("ma20") is not None
                            and float(prev_d.get("close")) < float(prev_d.get("ma20"))
                            and float(last_d.get("close")) < float(last_d.get("ma20"))
                        )
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        daily_close_below_ma20_2d = False
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            daily_macd_bearish_2d = False
            daily_close_below_ma20_2d = False

        def one_stat(sig, h: int) -> dict:
            stats, _sample = forward_holding_backtest(
                df_local,
                entry_signal=sig,
                horizon_weeks=int(h),
                buy_cost=float(buy_cost),
                sell_cost=float(sell_cost),
                non_overlapping=True,
            )
            return {
                "horizon_weeks": int(stats.horizon_weeks),
                "trades": int(stats.trades),
                "wins": int(stats.wins),
                "win_rate": float(stats.win_rate),
                "avg_return": float(stats.avg_return),
                "median_return": float(stats.median_return),
                "avg_mae": stats.avg_mae,
                "worst_mae": stats.worst_mae,
                "avg_mfe": stats.avg_mfe,
                "best_mfe": stats.best_mfe,
            }

        stats_out: dict[str, dict[str, dict]] = {"trend": {}, "swing": {}}
        for h in horizons:
            stats_out["trend"][f"{h}w"] = one_stat(sig_trend, h)
            stats_out["swing"][f"{h}w"] = one_stat(sig_swing, h)

        def stat_at(which: str, h: int) -> dict | None:
            return (stats_out.get(which) or {}).get(f"{h}w")

        def ok_for_trade(st: dict | None) -> bool:
            if not st:
                return False
            try:
                return int(st.get("trades") or 0) >= int(min_trades) and float(st.get("avg_return") or 0.0) > 0.0
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return False

        def pick_action(which: str) -> tuple[str, list[str]]:
            st = stat_at(which, rank_horizon)
            if not ok_for_trade(st):
                why = ["样本不足或统计不佳：胜率优先 => 默认观望"]
                if st:
                    try:
                        why.append(
                            f"{rank_horizon}W trades={int(st.get('trades') or 0)} win_rate={float(st.get('win_rate') or 0.0):.3f} avg_return={float(st.get('avg_return') or 0.0):.4f}"
                        )
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        pass
                return "观望", why

            win_rate = float(st.get("win_rate") or 0.0)
            trades = int(st.get("trades") or 0)
            avg_ret = float(st.get("avg_return") or 0.0)
            avg_mae = st.get("avg_mae")
            try:
                avg_mae_f = None if avg_mae is None else float(avg_mae)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                avg_mae_f = None

            strong = (win_rate >= 0.60) and (trades >= 30)
            ok_mae = (avg_mae_f is None) or (avg_mae_f >= -0.10)

            line = (
                f"{rank_horizon}W win_rate={win_rate:.1%} trades={trades} avg_return={avg_ret:.2%} avg_mae={avg_mae_f:.2%}"
                if avg_mae_f is not None
                else f"{rank_horizon}W win_rate={win_rate:.1%} trades={trades} avg_return={avg_ret:.2%}"
            )

            if strong and ok_mae:
                return "执行计划", [line]
            if (win_rate >= 0.55) and ok_mae:
                return "试错小仓", [line]
            return "观望", [line + "（磨损偏大/胜率不够稳）"]

        trend_now = bool(sig_trend.iloc[-1]) if len(sig_trend) else False
        swing_now = bool(sig_swing.iloc[-1]) if len(sig_swing) else False

        action = "观望"
        chosen = "none"
        reasons: list[str] = []

        # 退出/减仓优先级最高：这俩是“持仓风控”信号，不跟入场信号抢戏
        if weekly_below_ma50_confirm2:
            action = "退出"
            chosen = "exit"
            reasons = ["周线 close<MA50 连续2周确认：趋势走坏 => 退出（持仓视角）"]
        elif daily_macd_bearish_2d and daily_close_below_ma20_2d:
            action = "减仓"
            chosen = "reduce"
            reasons = ["日线 MACD死叉连续2天 + close<MA20 连续2天：提前风控 => 减仓（持仓视角）"]
        elif trend_now or swing_now:
            candidates: list[tuple[str, float]] = []
            if trend_now:
                st = stat_at("trend", rank_horizon) or {}
                candidates.append(("trend", float(st.get("win_rate") or 0.0)))
            if swing_now:
                st = stat_at("swing", rank_horizon) or {}
                candidates.append(("swing", float(st.get("win_rate") or 0.0)))
            candidates.sort(key=lambda x: x[1], reverse=True)
            chosen = candidates[0][0] if candidates else ("trend" if trend_now else "swing")
            action, reasons = pick_action(chosen)
        else:
            reasons = ["当前未触发趋势/回踩信号：胜率优先 => 观望"]

        # 给一个“最优持有周数”参考：在 trades>=min_trades 且 avg_return>0 前提下，选 win_rate 最大的 horizon
        best_h = rank_horizon
        best_wr = 0.0
        best_which = chosen if chosen in {"trend", "swing"} else "trend"
        for h in horizons:
            st = stat_at(best_which, h)
            if not ok_for_trade(st):
                continue
            wr = float(st.get("win_rate") or 0.0)
            if wr > best_wr:
                best_wr = wr
                best_h = h

        out = {
            "mode": "win_rate_first",
            "params": {
                "horizons": horizons,
                "rank_horizon_weeks": int(rank_horizon),
                "min_trades": int(min_trades),
                "buy_cost": float(buy_cost),
                "sell_cost": float(sell_cost),
            },
            "risk_signals": {
                "weekly_below_ma50_confirm2": bool(weekly_below_ma50_confirm2),
                "daily_macd_bearish_2d": bool(daily_macd_bearish_2d),
                "daily_close_below_ma20_confirm2": bool(daily_close_below_ma20_2d),
            },
            "signals_now": {"trend": trend_now, "swing": swing_now},
            "stats": stats_out,
            "decision": {
                "action": action,  # 观望 / 试错小仓 / 执行计划 / 减仓 / 退出
                "chosen_signal": chosen,  # trend / swing / reduce / exit / none
                "suggested_horizon_weeks": int(best_h),
                "reasons": reasons,
            },
        }

        try:
            write_json(out_dir / "signal_backtest.json", out)
        except (AttributeError):  # noqa: BLE001
            return

    write_signal_backtest()

    # Phase2：并行输出（评分/过滤器），默认不改变任何老信号口径
    # - game_theory_factors.json
    # - opportunity_score.json
    # - cash_signal.json
    # - position_sizing.json
    try:
        if not args.csv:
            import json

            def _read_json_silent(p: Path) -> dict | None:
                try:
                    if not p.exists():
                        return None
                    return json.loads(p.read_text(encoding="utf-8"))
                except (AttributeError):  # noqa: BLE001
                    return None

            # as_of/ref_date：用数据最后一根K线日期，别用“今天”（周末/节假日会乱）
            try:
                base_df = df_daily_raw if ("df_daily_raw" in locals() and df_daily_raw is not None) else df  # type: ignore[name-defined]
                as_of_dt = base_df["date"].max()
                as_of_d = as_of_dt.date() if hasattr(as_of_dt, "date") else datetime.now().date()
            except (AttributeError):  # noqa: BLE001
                as_of_d = datetime.now().date()

            ref_date_d = as_of_d

            # 读 TuShare 因子包（如果存在）；不存在就 None（不强制）
            tushare_pack = _read_json_silent(out_dir / "tushare_factors.json")

            # 1) 博弈/流动性 proxy 因子
            try:
                from ..factors.game_theory import compute_game_theory_factor_pack

                df_gt = df_daily_raw if ("df_daily_raw" in locals() and df_daily_raw is not None) else df  # type: ignore[name-defined]
                df_gt = ensure_ohlc(df_gt)
                gt_pack = compute_game_theory_factor_pack(
                    df=df_gt,
                    symbol=str(getattr(args, "symbol", "") or ""),
                    asset=str(getattr(args, "asset", "") or ""),
                    as_of=as_of_d,
                    ref_date=ref_date_d,
                    source="factors",
                )
                write_json(out_dir / "game_theory_factors.json", gt_pack)
            except (AttributeError) as exc:  # noqa: BLE001
                gt_pack = None
                (out_dir / "game_theory_factors_error.txt").write_text(str(exc), encoding="utf-8")

            # 2) OpportunityScore（排序/过滤/解释层）
            try:
                from ..opportunity_score import OpportunityScoreInputs, compute_opportunity_score

                # trap_risk：优先用 liquidity_trap.score（0~1，越高越危险）
                trap_risk = None
                try:
                    if isinstance(gt_pack, dict):
                        lr = ((gt_pack.get("factors") or {}) if isinstance(gt_pack.get("factors"), dict) else {}).get("liquidity_trap")
                        if isinstance(lr, dict):
                            trap_risk = lr.get("score")
                except (AttributeError):  # noqa: BLE001
                    trap_risk = None

                # fund_flow：优先 microstructure.score01，其次 north_score01（都没有就 None）
                fund_flow = None
                try:
                    if isinstance(tushare_pack, dict):
                        micro = tushare_pack.get("microstructure")
                        if isinstance(micro, dict) and bool(micro.get("ok")):
                            fund_flow = micro.get("score01")
                        if fund_flow is None:
                            hsgt = tushare_pack.get("hsgt")
                            if isinstance(hsgt, dict) and bool(hsgt.get("ok")):
                                    north = hsgt.get("north") if isinstance(hsgt.get("north"), dict) else None
                                    if isinstance(north, dict):
                                        fund_flow = north.get("score01")
                except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                    fund_flow = None

                # expected_holding_days：优先用 signal_backtest 的 suggested_horizon_weeks（粗略*5）
                expected_holding_days = 10
                try:
                    sb = _read_json_silent(out_dir / "signal_backtest.json")
                    hw = None
                    if isinstance(sb, dict):
                        hw = (((sb.get("decision") or {}) if isinstance(sb.get("decision"), dict) else {}).get("suggested_horizon_weeks"))
                    if hw is not None:
                        expected_holding_days = max(1, int(float(hw) * 5))
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    expected_holding_days = 10

                # key_level：默认 ma50（来自 add_moving_averages），缺就用 close
                kl_name = "ma50"
                kl_value = None
                try:
                    last = df.iloc[-1]
                    if "ma50" in df.columns:
                        kl_value = last.get("ma50")
                    if kl_value is None:
                        kl_name = "close"
                        kl_value = last.get("close")
                except (KeyError, IndexError, AttributeError):  # noqa: BLE001
                    kl_name = "close"
                    kl_value = None

                t1 = bool(str(getattr(args, "asset", "") or "").strip().lower() in {"stock", "etf", "index"})
                o_in = OpportunityScoreInputs(
                    symbol=str(getattr(args, "symbol", "") or ""),
                    asset=str(getattr(args, "asset", "") or ""),
                    as_of=as_of_d,
                    ref_date=ref_date_d,
                    min_score=0.70,
                    t_plus_one=t1,
                    trap_risk=trap_risk,
                    fund_flow=fund_flow,
                    expected_holding_days=int(expected_holding_days),
                )
                opp = compute_opportunity_score(df=ensure_ohlc(df), inputs=o_in, key_level_name=str(kl_name), key_level_value=kl_value)
                write_json(out_dir / "opportunity_score.json", opp)
            except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
                opp = None
                (out_dir / "opportunity_score_error.txt").write_text(str(exc), encoding="utf-8")

            # 3) CashSignal（账户级风险开关；研究用途）
            try:
                from ..cash_signal import CashSignalInputs, compute_cash_signal

                cs_in = CashSignalInputs(as_of=as_of_d, ref_date=ref_date_d, context_index_symbol="sh000300+sh000905")
                cs = compute_cash_signal(inputs=cs_in, tushare_factors=tushare_pack)
                write_json(out_dir / "cash_signal.json", cs)
            except (OSError, RuntimeError, TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
                (out_dir / "cash_signal_error.txt").write_text(str(exc), encoding="utf-8")

            # 4) PositionSizing（成本敏感仓位建议）
            try:
                from ..position_sizing import PositionSizingInputs, compute_position_sizing

                price = None
                try:
                    price = df.iloc[-1].get("close")
                except (KeyError, IndexError, AttributeError):  # noqa: BLE001
                    price = None

                bucket = "reject"
                total_score = 0.0
                try:
                    if isinstance(opp, dict):
                        bucket = str(opp.get("bucket") or "reject").strip().lower() or "reject"
                        total_score = float(opp.get("total_score") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    bucket = "reject"
                    total_score = 0.0

                # confidence：先给一个保守映射（Phase1/回测再做更严谨的胜率/赔率口径）
                conf = max(0.4, min(0.9, 0.4 + 0.6 * float(total_score)))

                ps_in = PositionSizingInputs(
                    symbol=str(getattr(args, "symbol", "") or ""),
                    asset=str(getattr(args, "asset", "") or ""),
                    as_of=as_of_d,
                    ref_date=ref_date_d,
                    opportunity_score=float(total_score),
                    bucket=bucket if bucket in {"reject", "probe", "plan"} else "reject",  # type: ignore[arg-type]
                    confidence=float(conf),
                    max_position_pct=0.30,
                    price=float(price) if price is not None else None,
                    min_trade_notional_yuan=2000,
                    min_fee_yuan=5.0,
                    t_plus_one=bool(str(getattr(args, "asset", "") or "").strip().lower() in {"stock", "etf", "index"}),
                )
                ps = compute_position_sizing(inputs=ps_in)
                write_json(out_dir / "position_sizing.json", ps)
            except (AttributeError) as exc:  # noqa: BLE001
                (out_dir / "position_sizing_error.txt").write_text(str(exc), encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        try:
            (out_dir / "phase2_error.txt").write_text(str(exc), encoding="utf-8")
        except (AttributeError) as exc2:  # noqa: BLE001
            diag.record("write_phase2_error_txt", exc2, note="写出 phase2_error.txt 失败（吞错治理）")
        diag.record("phase2_outputs", exc, note="Phase2 并行输出失败（已降级）")

    def run_wyckoff(target_dir: Path) -> None:
        analysis = None
        if args.llm:
            cfg = load_config()
            analysis = run_llm_analysis(
                cfg,
                df=df,
                prompt_path=args.prompt,
                max_rows=args.max_rows_llm,
                system_text="你是一个严谨的威科夫技术分析助手。你必须按用户要求输出。",
            )
            write_json(target_dir / "analysis.json", analysis)

        try:
            last = df.iloc[-1]

            def f(x):
                try:
                    return None if x is None else float(x)
                except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                    return None

            close = f(last.get("close"))
            ma50 = f(last.get("ma50"))
            ma200 = f(last.get("ma200"))
            ad_line = f(last.get("ad_line"))

            ret_4 = None
            ret_12 = None
            if close is not None and len(df) >= 5:
                ret_4 = f(close / float(df.iloc[-5]["close"]) - 1.0)
            if close is not None and len(df) >= 13:
                ret_12 = f(close / float(df.iloc[-13]["close"]) - 1.0)

            ad_delta_20 = None
            if "ad_line" in df.columns and ad_line is not None and len(df) >= 21:
                ad_delta_20 = f(ad_line - float(df.iloc[-21]["ad_line"]))

            write_json(
                target_dir / "wyckoff_features.json",
                {
                    "method": "wyckoff_features",
                    "last": {
                        "date": str(last["date"].date()),
                        "close": close,
                        "ma50": ma50,
                        "ma200": ma200,
                        "ad_line": ad_line,
                    },
                    "derived": {
                        "close_vs_ma200": f(close - ma200) if (close is not None and ma200 is not None) else None,
                        "ma50_vs_ma200": f(ma50 - ma200) if (ma50 is not None and ma200 is not None) else None,
                        "ret_4": ret_4,
                        "ret_12": ret_12,
                        "ad_delta_20": ad_delta_20,
                    },
                },
            )
        except (OSError, TypeError, ValueError, AttributeError):  # noqa: BLE001
            pass

        plot_wyckoff_chart(
            df,
            analysis=analysis,
            out_path=str(target_dir / "chart.png"),
            title=title,
            font_path=args.font_path,
        )

    def run_chan(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)

        try:
            structure = compute_chanlun_structure(df_local, min_gap=args.chan_min_gap)
        except ChanlunError as exc:
            raise SystemExit(str(exc)) from exc

        write_json(target_dir / "chan_structure.json", structure)
        if args.llm:
            cfg = load_config()
            llm_analysis = run_llm_analysis(
                cfg,
                df=df_local,
                prompt_path=args.chan_prompt,
                max_rows=args.max_rows_llm,
                extra_json=structure,
                system_text="你是一个严谨的缠论结构解读助手。你必须按用户要求输出。",
            )
            write_json(target_dir / "llm_analysis.json", llm_analysis)
        plot_chanlun_chart(
            df_local,
            structure=structure,
            out_path=str(target_dir / "chart.png"),
            title=f"缠论 - {title}",
            font_path=args.font_path,
        )

    def run_ichimoku(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)

        df_local = add_ichimoku(
            df_local,
            tenkan=args.ichimoku_tenkan,
            kijun=args.ichimoku_kijun,
            span_b=args.ichimoku_spanb,
            displacement=args.ichimoku_disp,
        )

        def fval(key: str):
            v = df_local.iloc[-1].get(key)
            try:
                x = None if v is None else float(v)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                return None
            try:
                import math

                return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
            except (AttributeError):  # noqa: BLE001
                return x

        close = fval("close")
        tenkan = fval("ichimoku_tenkan")
        kijun = fval("ichimoku_kijun")
        # span_a/span_b 是前移画图用的；状态判断用 raw（当前时点）不然末尾全是 NaN。
        span_a = fval("ichimoku_span_a_raw") if "ichimoku_span_a_raw" in df_local.columns else fval("ichimoku_span_a")
        span_b = fval("ichimoku_span_b_raw") if "ichimoku_span_b_raw" in df_local.columns else fval("ichimoku_span_b")

        cloud_top = None
        cloud_bottom = None
        position = "unknown"
        if span_a is not None and span_b is not None:
            cloud_top = float(max(span_a, span_b))
            cloud_bottom = float(min(span_a, span_b))
            if close is not None:
                if close > cloud_top:
                    position = "above"
                elif close < cloud_bottom:
                    position = "below"
                else:
                    position = "inside"

        tk_cross = "none"
        if len(df_local) >= 2:
            prev = df_local.iloc[-2]
            try:
                prev_diff = float(prev["ichimoku_tenkan"]) - float(prev["ichimoku_kijun"])
                cur_diff = float(df_local.iloc[-1]["ichimoku_tenkan"]) - float(df_local.iloc[-1]["ichimoku_kijun"])
                if prev_diff <= 0 < cur_diff:
                    tk_cross = "bullish"
                elif prev_diff >= 0 > cur_diff:
                    tk_cross = "bearish"
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                tk_cross = "unknown"

        out = {
            "method": "ichimoku",
            "params": {
                "tenkan": int(args.ichimoku_tenkan),
                "kijun": int(args.ichimoku_kijun),
                "span_b": int(args.ichimoku_spanb),
                "displacement": int(args.ichimoku_disp),
            },
            "last": {
                "date": str(df_local.iloc[-1]["date"].date()),
                "close": close,
                "tenkan": tenkan,
                "kijun": kijun,
                "span_a": span_a,
                "span_b": span_b,
                "cloud_top": cloud_top,
                "cloud_bottom": cloud_bottom,
                "position": position,
                "tk_cross": tk_cross,
            },
        }
        write_json(target_dir / "ichimoku.json", out)
        plot_ichimoku_chart(
            df_local, out_path=str(target_dir / "chart.png"), title=f"Ichimoku - {title}", font_path=args.font_path
        )

    def run_turtle(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)

        df_local = add_donchian_channels(
            df_local,
            window=args.turtle_entry,
            upper_col="donchian_entry_upper",
            lower_col="donchian_entry_lower",
            shift=1,
        )
        df_local = add_donchian_channels(
            df_local,
            window=args.turtle_exit,
            upper_col="donchian_exit_upper",
            lower_col="donchian_exit_lower",
            shift=1,
        )
        df_local = add_atr(df_local, period=args.turtle_atr, out_col="atr")

        last = df_local.iloc[-1]

        def f(x):
            try:
                return None if x is None else float(x)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                return None

        close = f(last.get("close"))
        entry_u = f(last.get("donchian_entry_upper"))
        entry_l = f(last.get("donchian_entry_lower"))
        exit_u = f(last.get("donchian_exit_upper"))
        exit_l = f(last.get("donchian_exit_lower"))
        atr = f(last.get("atr"))

        long_entry = bool(close is not None and entry_u is not None and close > entry_u)
        long_exit = bool(close is not None and exit_l is not None and close < exit_l)
        short_entry = bool(close is not None and entry_l is not None and close < entry_l)
        short_exit = bool(close is not None and exit_u is not None and close > exit_u)

        stop_atr = float(args.turtle_stop_atr)
        long_stop = f(close - stop_atr * atr) if (close is not None and atr is not None) else None
        short_stop = f(close + stop_atr * atr) if (close is not None and atr is not None) else None

        out = {
            "method": "turtle",
            "params": {
                "entry": int(args.turtle_entry),
                "exit": int(args.turtle_exit),
                "atr": int(args.turtle_atr),
                "stop_atr": stop_atr,
            },
            "last": {
                "date": str(last["date"].date()),
                "close": close,
                "donchian_entry_upper": entry_u,
                "donchian_entry_lower": entry_l,
                "donchian_exit_upper": exit_u,
                "donchian_exit_lower": exit_l,
                "atr": atr,
            },
            "signals": {
                "long_entry_breakout": long_entry,
                "long_exit_breakdown": long_exit,
                "short_entry_breakdown": short_entry,
                "short_exit_breakout": short_exit,
            },
            "risk": {
                "long_stop": long_stop,
                "short_stop": short_stop,
            },
        }
        write_json(target_dir / "turtle.json", out)
        plot_turtle_chart(df_local, out_path=str(target_dir / "chart.png"), title=f"Turtle - {title}", font_path=args.font_path)

    def run_momentum(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)
        df_local = add_rsi(df_local, period=args.rsi_period, out_col="rsi")
        df_local = add_macd(
            df_local,
            fast=args.macd_fast,
            slow=args.macd_slow,
            signal=args.macd_signal,
            macd_col="macd",
            signal_col="macd_signal",
            hist_col="macd_hist",
        )
        df_local = add_adx(
            df_local,
            period=args.adx_period,
            adx_col="adx",
            di_plus_col="di_plus",
            di_minus_col="di_minus",
        )

        last = df_local.iloc[-1]

        def f(key: str):
            try:
                v = last.get(key)
                return None if v is None else float(v)
            except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
                return None

        close = f("close")
        rsi = f("rsi")
        macd = f("macd")
        macd_sig = f("macd_signal")
        macd_hist = f("macd_hist")
        adx = f("adx")
        di_p = f("di_plus")
        di_m = f("di_minus")

        rsi_state = "unknown"
        if rsi is not None:
            if rsi >= 70:
                rsi_state = "overbought"
            elif rsi <= 30:
                rsi_state = "oversold"
            else:
                rsi_state = "neutral"

        macd_state = "unknown"
        if macd is not None and macd_sig is not None:
            macd_state = "bullish" if macd > macd_sig else ("bearish" if macd < macd_sig else "neutral")

        trend_strength = "unknown"
        if adx is not None:
            trend_strength = "strong" if adx >= 25 else ("weak" if adx <= 20 else "medium")

        direction = "unknown"
        if di_p is not None and di_m is not None:
            direction = "up" if di_p > di_m else ("down" if di_p < di_m else "neutral")

        out = {
            "method": "momentum",
            "params": {
                "rsi_period": int(args.rsi_period),
                "macd_fast": int(args.macd_fast),
                "macd_slow": int(args.macd_slow),
                "macd_signal": int(args.macd_signal),
                "adx_period": int(args.adx_period),
            },
            "last": {
                "date": str(last["date"].date()),
                "close": close,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_sig,
                "macd_hist": macd_hist,
                "adx": adx,
                "di_plus": di_p,
                "di_minus": di_m,
            },
            "state": {
                "rsi": rsi_state,
                "macd": macd_state,
                "trend_strength": trend_strength,
                "direction": direction,
            },
        }
        write_json(target_dir / "momentum.json", out)
        plot_momentum_chart(
            df_local,
            out_path=str(target_dir / "chart.png"),
            title=f"Momentum - {title}",
            font_path=args.font_path,
        )

    def run_dow(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)
        try:
            structure = compute_dow_structure(df_local, lookback=args.dow_lookback, min_gap=args.dow_min_gap)
        except DowError as exc:
            raise SystemExit(str(exc)) from exc
        write_json(target_dir / "dow.json", structure)
        plot_dow_chart(
            df_local,
            swings=structure.get("swings"),
            out_path=str(target_dir / "chart.png"),
            title=f"Dow - {title}",
            font_path=args.font_path,
        )

    def run_vsa(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)
        df_feat, report = compute_vsa_report(
            df_local,
            vol_window=args.vsa_vol_window,
            spread_window=args.vsa_spread_window,
            lookback_events=args.vsa_lookback,
        )
        write_json(target_dir / "vsa_features.json", report)
        if args.llm:
            cfg = load_config()
            llm_analysis = run_llm_analysis(
                cfg,
                df=df_local,
                prompt_path=args.vsa_prompt,
                max_rows=args.max_rows_llm,
                extra_json=report,
                system_text="你是一个严谨的 VSA（量价行为）分析助手。你必须按用户要求输出。",
            )
            write_json(target_dir / "llm_analysis.json", llm_analysis)

        plot_vsa_chart(
            df_feat,
            vsa_report=report,
            out_path=str(target_dir / "chart.png"),
            title=f"VSA - {title}",
            font_path=args.font_path,
        )

    def run_institution(target_dir: Path) -> None:
        df_local = ensure_ohlc(df)

        # 仅在“有真实标的”的情况下才尝试拉资金流（CSV 模式别瞎搞）
        symbol_prefixed = None
        asset = None
        if not args.csv:
            asset = str(args.asset)
            try:
                from ..akshare_source import resolve_symbol

                symbol_prefixed = resolve_symbol(asset, str(args.symbol))
            except (AttributeError):  # noqa: BLE001
                symbol_prefixed = str(args.symbol)

        try:
            from ..institution import compute_institution_report

            report = compute_institution_report(
                df_local,
                asset=asset,
                symbol_prefixed=symbol_prefixed,
                freq=str(args.freq),
                vsa_vol_window=int(args.vsa_vol_window),
                vsa_spread_window=int(args.vsa_spread_window),
            )
            write_json(target_dir / "institution.json", report)
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            diag.record("institution_report", exc, note="institution 模块失败（已降级）")
            return

    if args.method == "wyckoff":
        run_wyckoff(out_dir)
    elif args.method == "chan":
        run_chan(out_dir)
    elif args.method == "ichimoku":
        run_ichimoku(out_dir)
    elif args.method == "turtle":
        run_turtle(out_dir)
    elif args.method == "momentum":
        run_momentum(out_dir)
    elif args.method == "dow":
        run_dow(out_dir)
    elif args.method == "vsa":
        run_vsa(out_dir)
    elif args.method == "institution":
        run_institution(out_dir)
    elif args.method == "both":
        wyckoff_dir = out_dir / "wyckoff"
        chan_dir = out_dir / "chan"
        wyckoff_dir.mkdir(parents=True, exist_ok=True)
        chan_dir.mkdir(parents=True, exist_ok=True)
        run_wyckoff(wyckoff_dir)
        run_chan(chan_dir)
    elif args.method == "all":
        methods = [
            ("wyckoff", run_wyckoff),
            ("chan", run_chan),
            ("ichimoku", run_ichimoku),
            ("turtle", run_turtle),
            ("momentum", run_momentum),
            ("dow", run_dow),
            ("vsa", run_vsa),
            ("institution", run_institution),
        ]
        for name, fn in methods:
            target = out_dir / name
            target.mkdir(parents=True, exist_ok=True)
            fn(target)
    else:
        raise SystemExit(f"未知 method：{args.method}")

    # Phase3：可选输出（不影响原有口径）：按 strategy_configs.yaml 额外算一份 strategy_signal.json
    strategy_cfg = str(getattr(args, "strategy_config", "") or "").strip()
    strategy_key = str(getattr(args, "strategy", "") or "").strip()
    if strategy_cfg and strategy_key:
        try:
            from ..factors.base import StrategyEngine
            from ..strategy_config_loader import load_strategy_configs_yaml

            cfgs = load_strategy_configs_yaml(Path(strategy_cfg))
            if strategy_key not in cfgs:
                raise ValueError(f"未知 strategy={strategy_key}（可用：{', '.join(sorted(cfgs))}）")

            # 市场 regime：用指数算一个“环境标签”（用于 allowed_regimes）
            idx = str(getattr(args, "strategy_regime_index", "sh000300") or "sh000300").strip()
            canary = bool(getattr(args, "strategy_regime_canary", True))
            regime_dict, regime_error, regime_idx_eff = _compute_market_regime_payload(idx, canary_downgrade=canary)
            regime_label = str((regime_dict or {}).get("label") or "unknown")

            engine = StrategyEngine(cfgs[strategy_key])
            sig = engine.generate_signal(df, market_regime=regime_label)

            # as_of：用最后一根K线日期（别用“今天”，周末会乱）
            as_of = None
            try:
                last_dt = df["date"].max()
                as_of = str(last_dt.date()) if hasattr(last_dt, "date") else None
            except (AttributeError):  # noqa: BLE001
                as_of = None

            write_json(
                out_dir / "strategy_signal.json",
                {
                    "schema": "llm_trading.strategy_signal.v1",
                    "symbol": args.symbol if not args.csv else None,
                    "asset": args.asset if not args.csv else None,
                    "freq": args.freq,
                    "as_of": as_of,
                    "strategy_key": str(strategy_key),
                    "strategy_config": str(strategy_cfg),
                    "market_regime": {"index": str(regime_idx_eff or idx), "payload": regime_dict, "error": regime_error},
                    "signal": sig,
                },
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:  # noqa: BLE001
            (out_dir / "strategy_signal_error.txt").write_text(str(exc), encoding="utf-8")

    if getattr(args, "narrate", False):
        try:
            from ..narrative import DEFAULT_SCHOOLS, generate_narrative_text

            cfg = load_config()
            schools_raw = str(getattr(args, "narrate_schools", "") or "")
            schools = [s.strip() for s in schools_raw.split(",") if s.strip()]
            if len(schools) == 1 and schools[0].lower() == "all":
                schools = ["chan", "wyckoff", "ichimoku", "turtle", "momentum", "dow", "vsa", "institution"]
            if not schools:
                schools = DEFAULT_SCHOOLS
            print(f"[narrate] provider={args.narrate_provider} schools={','.join(schools)}")
            text = generate_narrative_text(
                cfg,
                out_dir=out_dir,
                provider=str(args.narrate_provider),
                prompt_path=str(getattr(args, "narrate_prompt")),
                schools=schools,
                temperature=float(getattr(args, "narrate_temperature")),
                max_output_tokens=int(getattr(args, "narrate_max_output_tokens")),
            )
            (out_dir / "summary.md").write_text(text, encoding="utf-8")
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            (out_dir / "summary_error.txt").write_text(str(exc), encoding="utf-8")
            print(f"[narrate] failed: {exc}")

    # diagnostics：给排查留证据（比 stdout 可靠）
    diag.write(out_dir, cmd="analyze")

    print(str(out_dir.resolve()))
    return 0
