from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 12):
    raise SystemExit(
        "艹，别用 Python 3.8 了，这仓库现在基线是 Python 3.12+。\n"
        "用：\"/home/root_zzl/miniconda3/bin/python\" -m venv \".venv\" 然后装 requirements.txt"
    )

from .akshare_source import DataSourceError, FetchParams, fetch_daily
from .chanlun import ChanlunError, compute_chanlun_structure
from .config import load_config
from .csv_loader import CsvSchemaError, load_ohlcv_csv
from .dow import DowError, compute_dow_structure
from .etf_scan import analyze_etf_symbol, load_etf_universe
from .indicators import (
    add_accumulation_distribution_line,
    add_adx,
    add_atr,
    add_donchian_channels,
    add_ichimoku,
    add_macd,
    add_moving_averages,
    add_rsi,
)
from .pipeline import run_llm_analysis, write_json
from .plotting import (
    plot_chanlun_chart,
    plot_dow_chart,
    plot_ichimoku_chart,
    plot_momentum_chart,
    plot_turtle_chart,
    plot_vsa_chart,
    plot_wyckoff_chart,
)
from .resample import resample_to_weekly
from .vsa import compute_vsa_report
from .stock_scan import DailyFilter, ScanFreq, analyze_stock_symbol, load_stock_universe


def _default_out_dir(csv_path: str) -> Path:
    stem = Path(csv_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{stem}_{ts}"


def _default_out_dir_for_symbol(asset: str, symbol: str, freq: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(" ", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{asset}_{safe_symbol}_{freq}_{ts}"


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        df = fetch_daily(
            FetchParams(
                asset=args.asset,
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
            )
        )
    except DataSourceError as exc:
        raise SystemExit(str(exc)) from exc

    if args.freq == "weekly":
        df = resample_to_weekly(df)

    out_path = Path(args.out) if args.out else Path("data") / f"{args.asset}_{args.symbol}_{args.freq}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(str(out_path.resolve()))
    return 0


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
                )
            )
        except DataSourceError as exc:
            raise SystemExit(str(exc)) from exc
        default_out = _default_out_dir_for_symbol(args.asset, args.symbol, args.freq)
        title = args.title or f"{args.asset.upper()} - {args.symbol}"

    if args.freq == "weekly":
        df = resample_to_weekly(df)

    if args.window and len(df) > args.window:
        df = df.tail(args.window).reset_index(drop=True)

    df = add_moving_averages(df, ma_fast=50, ma_slow=200)
    df = add_accumulation_distribution_line(df)

    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        write_json(
            out_dir / "meta.json",
            {
                "generated_at": datetime.now().isoformat(),
                "source": "csv" if args.csv else "akshare",
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
    except Exception:  # noqa: BLE001
        pass

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
            from .backtest import forward_holding_backtest
        except Exception:  # noqa: BLE001
            return

        df_local = ensure_ohlc(df)
        df_local = add_moving_averages(df_local, ma_fast=50, ma_slow=200)
        df_local = add_donchian_channels(
            df_local, window=20, upper_col="donchian_upper_20", lower_col="donchian_lower_20", shift=1
        )

        # 跟 scan-stock 的策略保持一致：统一走 strategy_registry，别两套逻辑打架
        try:
            from .strategy_registry import compute_series
        except Exception:  # noqa: BLE001
            return

        sig_trend = compute_series(df_local, key="trend").fillna(False)
        sig_swing = compute_series(df_local, key="swing").fillna(False)

        horizons = [4, 8, 12]
        buy_cost = 0.001
        sell_cost = 0.002
        min_trades = 20
        rank_horizon = 8

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
            except Exception:  # noqa: BLE001
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
                    except Exception:  # noqa: BLE001
                        pass
                return "观望", why

            win_rate = float(st.get("win_rate") or 0.0)
            trades = int(st.get("trades") or 0)
            avg_ret = float(st.get("avg_return") or 0.0)
            avg_mae = st.get("avg_mae")
            try:
                avg_mae_f = None if avg_mae is None else float(avg_mae)
            except Exception:  # noqa: BLE001
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

        if trend_now or swing_now:
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
            "signals_now": {"trend": trend_now, "swing": swing_now},
            "stats": stats_out,
            "decision": {
                "action": action,  # 观望 / 试错小仓 / 执行计划
                "chosen_signal": chosen,  # trend / swing / none
                "suggested_horizon_weeks": int(best_h),
                "reasons": reasons,
            },
        }

        try:
            write_json(out_dir / "signal_backtest.json", out)
        except Exception:  # noqa: BLE001
            return

    write_signal_backtest()

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
                except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                return None
            try:
                import math

                return None if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else x
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
                from .akshare_source import resolve_symbol

                symbol_prefixed = resolve_symbol(asset, str(args.symbol))
            except Exception:  # noqa: BLE001
                symbol_prefixed = str(args.symbol)

        try:
            from .institution import compute_institution_report

            report = compute_institution_report(
                df_local,
                asset=asset,
                symbol_prefixed=symbol_prefixed,
                freq=str(args.freq),
                vsa_vol_window=int(args.vsa_vol_window),
                vsa_spread_window=int(args.vsa_spread_window),
            )
            write_json(target_dir / "institution.json", report)
        except Exception:  # noqa: BLE001
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

    if getattr(args, "narrate", False):
        try:
            from .narrative import DEFAULT_SCHOOLS, generate_narrative_text

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
        except Exception as exc:  # noqa: BLE001
            (out_dir / "summary_error.txt").write_text(str(exc), encoding="utf-8")
            print(f"[narrate] failed: {exc}")

    print(str(out_dir.resolve()))
    return 0


def cmd_scan_etf(args: argparse.Namespace) -> int:
    try:
        universe = load_etf_universe(include_all_funds=bool(args.include_all_funds))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ETF 列表拉取失败：{exc}") from exc

    if not universe:
        raise SystemExit("ETF 列表为空：AkShare 没给数据，或者源站抽风。")

    min_amount = float(args.min_amount) if args.min_amount is not None else 0.0
    min_amount_avg20 = float(getattr(args, "min_amount_avg20", 0.0) or 0.0)
    min_weeks = int(getattr(args, "min_weeks", 60) or 0)
    min_weeks = max(0, min(min_weeks, 2000))

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"etf_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    errors: list[dict] = []
    filtered_by_min_amount = 0
    filtered_by_min_amount_avg20 = 0

    import math

    def safe_float(v, *, default: float | None = None) -> float | None:
        try:
            if v is None:
                return default
            x = float(v)
        except Exception:  # noqa: BLE001
            return default
        return x if math.isfinite(x) else default

    workers = int(getattr(args, "workers", 8) or 8)
    workers = max(1, min(workers, 32))

    total = len(universe)

    def run_one(item):
        return analyze_etf_symbol(item, freq=args.freq, window=args.window)

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
                r = {"symbol": item.symbol, "name": item.name, "error": str(exc)}

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

            results.append(r)

    # 排名输出（趋势突破 / 回踩波段 两套）
    def weekly_total(x: dict) -> int:
        bars = x.get("bars") or {}
        try:
            return int(bars.get("weekly_total") or 0)
        except Exception:  # noqa: BLE001
            return 0

    rankable = results
    if min_weeks > 0:
        rankable = [x for x in results if weekly_total(x) >= min_weeks]

    def key_score(which: str):
        def _k(x: dict):
            s = (x.get("scores") or {}).get(which)
            try:
                return float(s or 0.0)
            except Exception:  # noqa: BLE001
                return 0.0

        return _k

    results_sorted_trend = sorted(rankable, key=key_score("trend"), reverse=True)
    results_sorted_swing = sorted(rankable, key=key_score("swing"), reverse=True)

    top_k = int(args.top_k) if args.top_k else 30
    top_k = max(5, min(top_k, 100))

    # BBB（左侧偏稳健）：位置优先 + 周线MACD在0轴上 + 日线MACD为多
    dist_ma50_max = 0.12
    max_above_20w = 0.05
    min_weekly_bars_total = max(60, min_weeks)

    def bbb_eval(x: dict) -> dict:
        fails: list[str] = []

        wk_total = weekly_total(x)
        if wk_total < min_weekly_bars_total:
            fails.append("周K不足")

        lv = x.get("levels") or {}
        close = safe_float(x.get("close"))
        ma50 = safe_float(lv.get("ma50"))
        if close is None or ma50 is None or close <= 0 or ma50 <= 0:
            fails.append("缺close/MA50")
        else:
            dist = abs(close - ma50) / ma50
            if dist > dist_ma50_max:
                fails.append("离MA50太远")

        upper_f = safe_float(lv.get("resistance_20w"))
        if upper_f is None or upper_f <= 0:
            fails.append("缺20W上轨")
        else:
            if close is not None and close > upper_f * (1.0 + max_above_20w):
                fails.append("追高(高于20W上轨)")

        mom = x.get("momentum") or {}
        macd_state = str(mom.get("macd_state") or "")
        macd_f = safe_float(mom.get("macd"))
        if macd_state != "bullish":
            fails.append("周MACD未转多")
        if macd_f is None or macd_f <= 0:
            fails.append("周MACD<=0")

        daily = x.get("daily") or {}
        if str(daily.get("macd_state") or "") != "bullish":
            fails.append("日MACD未转多")

        ok = len(fails) == 0

        why = ""
        if ok:
            dist_pct = ""
            room_pct = ""
            try:
                if close is not None and ma50 is not None and ma50 > 0:
                    dist_pct = f"{((close - ma50) / ma50) * 100:.1f}%"
            except Exception:  # noqa: BLE001
                dist_pct = ""

            try:
                if close is not None and upper_f is not None and upper_f > 0:
                    room_pct = f"{((upper_f - close) / upper_f) * 100:.1f}%"
            except Exception:  # noqa: BLE001
                room_pct = ""

            why = "通过：周MACD多且>0 / 日MACD多 / 位置靠MA50"
            if dist_pct:
                why += f"({dist_pct})"
            why += " / 未追高"
            if room_pct:
                why += f"(离20W上轨{room_pct})"

        return {"ok": ok, "fails": fails, "why": why}

    bbb_fail_stats: dict[str, int] = {}
    bbb_items: list[dict] = []
    for x in results:
        ev = bbb_eval(x)
        if ev["ok"]:
            bbb_items.append({**x, "bbb": ev})
        else:
            for reason in ev.get("fails") or []:
                bbb_fail_stats[reason] = int(bbb_fail_stats.get(reason, 0)) + 1

    def key_bbb(x: dict) -> tuple:
        lv = x.get("levels") or {}
        close = safe_float(x.get("close"), default=0.0) or 0.0
        ma50 = safe_float(lv.get("ma50"), default=0.0) or 0.0

        dist = 9e9
        if close > 0 and ma50 > 0:
            dist = abs(close - ma50) / ma50  # 越接近 MA50 越不容易“套山上”

        # 更偏“位置优先”：离 20W 上轨越远（room 越大）越稳
        room = 0.0
        upper = lv.get("resistance_20w")
        upper_f = safe_float(upper)
        if upper_f is not None and upper_f > 0 and close > 0:
            room = (upper_f - close) / upper_f

        amt = safe_float(x.get("amount"), default=0.0) or 0.0

        mom = x.get("momentum") or {}
        macd_w = safe_float(mom.get("macd"), default=0.0) or 0.0

        # 排序：位置更好 -> room更大 -> 流动性更强 -> 周线动量更强
        return (float(dist), -float(room), -float(amt), -float(macd_w))

    bbb_sorted = sorted(bbb_items, key=key_bbb)

    filtered_by_min_weeks = int(len(results) - len(rankable))
    write_json(
        out_dir / "top_trend.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_weeks": int(min_weeks),
            "counts": {"results": len(results), "rankable": len(rankable), "filtered_by_min_weeks": filtered_by_min_weeks},
            "items": results_sorted_trend[:top_k],
        },
    )
    write_json(
        out_dir / "top_swing.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "min_weeks": int(min_weeks),
            "counts": {"results": len(results), "rankable": len(rankable), "filtered_by_min_weeks": filtered_by_min_weeks},
            "items": results_sorted_swing[:top_k],
        },
    )
    write_json(
        out_dir / "top_bbb.json",
        {
            "generated_at": datetime.now().isoformat(),
            "freq": args.freq,
            "bbb": {
                "dist_ma50_max": 0.12,
                "require_weekly_macd_bullish": True,
                "require_weekly_macd_above_zero": True,
                "require_daily_macd_bullish": True,
                "max_above_20w": 0.05,
                "min_weekly_bars_total": max(60, min_weeks),
                "min_daily_amount_avg20": float(min_amount_avg20) if min_amount_avg20 > 0 else None,
                "fail_stats": bbb_fail_stats,
            },
            "counts": {
                "filtered_by_min_amount": int(filtered_by_min_amount),
                "filtered_by_min_amount_avg20": int(filtered_by_min_amount_avg20),
                "results": len(results),
                "rankable": len(rankable),
                "filtered_by_min_weeks": filtered_by_min_weeks,
                "errors": len(errors),
                "bbb_ok": len(bbb_items),
            },
            "items": bbb_sorted[:top_k],
        },
    )
    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8")

    print(str(out_dir.resolve()))
    return 0


def cmd_scan_stock(args: argparse.Namespace) -> int:
    try:
        universe = load_stock_universe(include_st=bool(args.include_st), include_bj=not bool(args.exclude_bj))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"全A列表拉取失败：{exc}") from exc

    if not universe:
        raise SystemExit("全A列表为空：AkShare 没给数据，或者源站抽风。")

    if args.limit and int(args.limit) > 0:
        universe = universe[: int(args.limit)]

    workers = int(args.workers) if args.workers else 8
    workers = max(1, min(workers, 32))

    min_amount = float(args.min_amount) if args.min_amount is not None else 0.0
    min_price = float(args.min_price) if getattr(args, "min_price", None) is not None else 0.0
    max_price = float(args.max_price) if getattr(args, "max_price", None) is not None else 0.0
    min_trades = int(args.min_trades) if args.min_trades is not None else 0
    rank_horizon = int(args.rank_horizon) if args.rank_horizon else 8

    try:
        from .strategy_registry import parse_strategy_list, set_trend_template_params

        base_filters = parse_strategy_list(getattr(args, "base_filters", None))
        set_trend_template_params(
            near_high=float(getattr(args, "tt_near_high", 0.25)),
            above_low=float(getattr(args, "tt_above_low", 0.30)),
            slope_weeks=int(getattr(args, "tt_slope_weeks", 4)),
        )
    except Exception:  # noqa: BLE001
        base_filters = []

    horizons: list[int] = []
    if args.horizons:
        for part in str(args.horizons).split(","):
            part2 = part.strip()
            if not part2:
                continue
            try:
                horizons.append(int(part2))
            except Exception:  # noqa: BLE001
                continue
    horizons = sorted({h for h in horizons if h > 0}) or [4, 8, 12]

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"stock_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("data") / "cache" / "stock"
    cache_ttl_hours = float(args.cache_ttl_hours) if args.cache_ttl_hours is not None else 24.0

    results: list[dict] = []
    errors: list[dict] = []

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
            except Exception as exc:  # noqa: BLE001
                r = {"symbol": item.symbol, "name": item.name, "error": str(exc)}

            if r.get("error"):
                errors.append(r)
                if args.verbose:
                    print(f"[{done}/{total}] {item.symbol} {item.name} ERROR: {r.get('error')}")
                continue

            if min_amount > 0:
                try:
                    amt = float(r.get("amount") or 0.0)
                except Exception:  # noqa: BLE001
                    amt = 0.0
                if amt < min_amount:
                    if args.verbose:
                        print(f"[{done}/{total}] {item.symbol} {item.name} skip(amount<{min_amount:g})")
                    continue

            if min_price > 0 or max_price > 0:
                try:
                    close = float(r.get("close") or 0.0)
                except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                return 0.0

        return _k

    def get_trades(x: dict, which: str) -> int:
        key = f"{rank_horizon}w"
        st = (x.get("forward") or {}).get(which, {}).get(key, {})
        try:
            return int(st.get("trades") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def is_trend_context(x: dict) -> bool:
        lv = x.get("levels") or {}
        try:
            close = float(x.get("close") or 0.0)
            ma50 = float(lv.get("ma50") or 0.0)
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
                "signal_now": len(dip_now_all),
            },
            "min_trades": min_trades,
            "items": pick_top(dip_items, "dip"),
        },
    )
    write_json(out_dir / "errors.json", {"generated_at": datetime.now().isoformat(), "errors": errors})

    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(out_dir / "all_results.csv", index=False, encoding="utf-8")

    print(str(out_dir.resolve()))
    return 0


def cmd_clean_outputs(args: argparse.Namespace) -> int:
    """
    清理 outputs 里历史产物（结果有时效性，别让它一直占空间）。
    默认只 dry-run，真删必须加 --apply。
    """
    root = Path(str(getattr(args, "path", None) or "outputs"))
    if not root.exists():
        print(str(root))
        print("outputs 不存在，没啥可清。")
        return 0
    if not root.is_dir():
        raise SystemExit(f"不是目录：{root}")

    keep_days = float(getattr(args, "keep_days", 7.0) or 0.0)
    keep_last = int(getattr(args, "keep_last", 20) or 0)
    include_logs = bool(getattr(args, "include_logs", False))
    apply = bool(getattr(args, "apply", False))

    keep_days = max(0.0, min(keep_days, 3650.0))
    keep_last = max(0, min(keep_last, 5000))

    now = time.time()
    cutoff = now - keep_days * 86400.0 if keep_days > 0 else None

    entries: list[tuple[float, Path]] = []
    for p in root.iterdir():
        name = p.name
        if name in {".gitkeep"}:
            continue
        if p.is_file() and (not include_logs):
            # 默认只清“结果目录”，日志别乱动
            continue
        try:
            st = p.stat()
        except Exception:  # noqa: BLE001
            continue
        entries.append((float(st.st_mtime), p))

    entries.sort(key=lambda x: x[0], reverse=True)

    keep: set[Path] = set()
    if keep_last > 0:
        keep.update([p for _, p in entries[:keep_last]])
    if cutoff is not None:
        for mt, p in entries:
            if mt >= cutoff:
                keep.add(p)

    to_delete = [p for _, p in entries if p not in keep]

    def fmt_ts(ts: float) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:  # noqa: BLE001
            return str(ts)

    print(str(root.resolve()))
    print(f"规则：保留最近 {keep_days:g} 天 + 最近 {keep_last} 个；include_logs={include_logs}; apply={apply}")
    print(f"扫描到 {len(entries)} 个条目；计划删除 {len(to_delete)} 个；保留 {len(keep)} 个。")
    if entries:
        mt_new, p_new = entries[0]
        mt_old, p_old = entries[-1]
        print(f"最新：{p_new.name} ({fmt_ts(mt_new)})")
        print(f"最旧：{p_old.name} ({fmt_ts(mt_old)})")

    if not to_delete:
        print("没有需要删除的条目。")
        return 0

    if not apply:
        print("dry-run：以下将被删除（前 30 个）：")
        for p in to_delete[:30]:
            print(f"- {p.name}")
        if len(to_delete) > 30:
            print(f"... 还有 {len(to_delete) - 30} 个")
        print("真要删就加 --apply（小心点，删了就没了）。")
        return 0

    import shutil

    deleted = 0
    failed = 0
    for p in to_delete:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            deleted += 1
        except Exception:  # noqa: BLE001
            failed += 1

    print(f"完成：删除 {deleted} 个；失败 {failed} 个。")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("没装 uvicorn？先跑：pip install -r \"requirements.txt\"") from exc

    host = str(args.host or "127.0.0.1")
    port = int(args.port or 8000)
    reload = bool(args.reload)

    uvicorn.run(
        "llm_trading.webapp:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=str(args.log_level or "info"),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_trading", description="LLM辅助交易：威科夫读图 + 自动标注出图")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="用 AkShare 抓 A股 ETF/指数 日线/周线数据并落 CSV")
    p_fetch.add_argument("--asset", choices=["etf", "index", "stock"], required=True, help="数据类型：etf / index / stock")
    p_fetch.add_argument(
        "--symbol",
        required=True,
        help="代码或名称：ETF 支持 510300 或 sh510300；指数支持 sh000300 / sz399006；个股支持 000725 / sz000725 / 京东方A",
    )
    p_fetch.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_fetch.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p_fetch.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="输出频率（默认 weekly）")
    p_fetch.add_argument("--adjust", default=None, help="仅个股：复权方式（qfq/hfq/空，可选；默认 qfq）")
    p_fetch.add_argument("--out", default=None, help="输出 CSV 路径（默认 data/<asset>_<symbol>_<freq>.csv）")
    p_fetch.set_defaults(func=cmd_fetch)

    p = sub.add_parser("analyze", help="读取CSV -> 计算均线 -> (可选) LLM 生成标注 -> 出图")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", default=None, help="CSV 文件路径")
    g.add_argument("--symbol", default=None, help="直接抓数分析：ETF/指数代码（如 510300 / sh000300）")
    p.add_argument("--encoding", default=None, help="CSV 编码（可选）")
    p.add_argument("--date-col", default=None, help="日期列名（可选）")
    p.add_argument("--open-col", default=None, help="开盘列名（可选）")
    p.add_argument("--high-col", default=None, help="最高列名（可选）")
    p.add_argument("--low-col", default=None, help="最低列名（可选）")
    p.add_argument("--close-col", default=None, help="收盘列名（可选）")
    p.add_argument("--volume-col", default=None, help="成交量列名（可选）")
    p.add_argument("--asset", choices=["etf", "index", "stock"], default="etf", help="当使用 --symbol 时必须指定资产类型（默认 etf）")
    p.add_argument("--start-date", default=None, help="开始日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p.add_argument("--end-date", default=None, help="结束日期（YYYYMMDD 或 YYYY-MM-DD，可选）")
    p.add_argument("--adjust", default=None, help="仅个股：复权方式（qfq/hfq/空，可选；默认 qfq）")
    p.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="分析频率（默认 weekly）")
    p.add_argument("--window", type=int, default=500, help="只取最近 N 行（默认 500；周线≈10年）")
    p.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/<name>_<timestamp>）")
    p.add_argument("--title", default=None, help="图标题（可选）")
    p.add_argument("--font-path", default=None, help="中文字体文件路径(.ttf/.otf)，用于解决缺字问题（可选）")
    p.add_argument(
        "--method",
        choices=["wyckoff", "chan", "ichimoku", "turtle", "momentum", "dow", "vsa", "institution", "both", "all"],
        default="wyckoff",
        help="分析方法（默认 wyckoff）",
    )
    p.add_argument("--chan-min-gap", type=int, default=4, help="缠论：分型成笔的最小间隔（默认 4，越大越稳但越少）")
    p.add_argument("--ichimoku-tenkan", type=int, default=9, help="一目：转换线周期（默认 9）")
    p.add_argument("--ichimoku-kijun", type=int, default=26, help="一目：基准线周期（默认 26）")
    p.add_argument("--ichimoku-spanb", type=int, default=52, help="一目：先行B周期（默认 52）")
    p.add_argument("--ichimoku-disp", type=int, default=26, help="一目：位移周期（默认 26）")
    p.add_argument("--turtle-entry", type=int, default=20, help="海龟：入场 Donchian 周期（默认 20）")
    p.add_argument("--turtle-exit", type=int, default=10, help="海龟：出场 Donchian 周期（默认 10）")
    p.add_argument("--turtle-atr", type=int, default=20, help="海龟：ATR 周期（默认 20）")
    p.add_argument("--turtle-stop-atr", type=float, default=2.0, help="海龟：止损倍数（默认 2.0*ATR）")
    p.add_argument("--rsi-period", type=int, default=14, help="Momentum：RSI 周期（默认 14）")
    p.add_argument("--macd-fast", type=int, default=12, help="Momentum：MACD 快线 EMA（默认 12）")
    p.add_argument("--macd-slow", type=int, default=26, help="Momentum：MACD 慢线 EMA（默认 26）")
    p.add_argument("--macd-signal", type=int, default=9, help="Momentum：MACD 信号线 EMA（默认 9）")
    p.add_argument("--adx-period", type=int, default=14, help="Momentum：ADX 周期（默认 14）")
    p.add_argument("--dow-lookback", type=int, default=2, help="Dow：分型 lookback（默认 2；越大越稳但越少）")
    p.add_argument("--dow-min-gap", type=int, default=2, help="Dow：swing 最小间隔（默认 2）")
    p.add_argument("--vsa-vol-window", type=int, default=20, help="VSA：相对成交量窗口（默认 20）")
    p.add_argument("--vsa-spread-window", type=int, default=20, help="VSA：相对 spread 窗口（默认 20）")
    p.add_argument("--vsa-lookback", type=int, default=120, help="VSA：最近事件回看根数（默认 120）")
    p.add_argument("--llm", action="store_true", help="启用 LLM 结构化分析（需要 OPENAI_API_KEY/OPENAI_MODEL）")
    p.add_argument(
        "--prompt",
        default=str(Path("prompts") / "wyckoff_json_prompt.md"),
        help="威科夫提示词路径（默认 prompts/wyckoff_json_prompt.md）",
    )
    p.add_argument(
        "--chan-prompt",
        default=str(Path("prompts") / "chanlun_json_prompt.md"),
        help="缠论解读提示词路径（默认 prompts/chanlun_json_prompt.md）",
    )
    p.add_argument(
        "--vsa-prompt",
        default=str(Path("prompts") / "vsa_json_prompt.md"),
        help="VSA 解读提示词路径（默认 prompts/vsa_json_prompt.md）",
    )
    p.add_argument("--max-rows-llm", type=int, default=300, help="喂给 LLM 的最大行数（默认 300，会等距抽样）")
    p.add_argument("--narrate", action="store_true", help="生成“多流派自然语言解读”（默认用 Gemini，需要 GEMINI_API_KEY/MODEL）")
    p.add_argument("--narrate-provider", choices=["gemini", "openai"], default="openai", help="自然语言解读的 LLM 提供方（默认 openai）")
    p.add_argument(
        "--narrate-prompt",
        default=str(Path("prompts") / "synthesis_prompt.md"),
        help="自然语言解读提示词路径（默认 prompts/synthesis_prompt.md）",
    )
    p.add_argument(
        "--narrate-schools",
        default="chan,wyckoff,ichimoku,turtle,momentum",
        help="参与综合解读的流派列表，逗号分隔（默认 5 派：chan,wyckoff,ichimoku,turtle,momentum；可加 dow,vsa,institution）",
    )
    p.add_argument("--narrate-temperature", type=float, default=0.2, help="自然语言解读 temperature（默认 0.2）")
    p.add_argument("--narrate-max-output-tokens", type=int, default=1200, help="自然语言解读最大输出 token（默认 1200）")
    p.set_defaults(func=cmd_analyze)

    p_scan = sub.add_parser("scan-etf", help="扫描场内 ETF/基金，输出波段候选排名（研究用途）")
    p_scan.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly）")
    p_scan.add_argument("--window", type=int, default=400, help="每个标的取最近 N 根K线（默认 400）")
    p_scan.add_argument("--min-weeks", type=int, default=60, help="周K 少于该值不进榜（默认 60；填 0 关闭）")
    p_scan.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="过滤最后一根成交额小于该值的标的（默认 0=不过滤；优先用数据源 amount，缺失才用 close*volume 估算）",
    )
    p_scan.add_argument(
        "--min-amount-avg20",
        type=float,
        default=0.0,
        help="过滤最近20日均成交额小于该值的标的（默认 0=不过滤；更适合当流动性门槛）",
    )
    p_scan.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（默认 0=全量；按代码排序）")
    p_scan.add_argument("--top-k", type=int, default=30, help="输出 Top K（默认 30）")
    p_scan.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/etf_scan_<timestamp>）")
    p_scan.add_argument("--workers", type=int, default=8, help="并发线程数（默认 8）")
    p_scan.add_argument(
        "--include-all-funds",
        action="store_true",
        help="把 LOF/固收/其它场内基金也算进去（默认只扫股票/海外股票 ETF：15xxxx + 5[1/2/3/6/8/9]xxxx）",
    )
    p_scan.add_argument("--verbose", action="store_true", help="打印扫描进度（可选）")
    p_scan.set_defaults(func=cmd_scan_etf)

    p_scan_stock = sub.add_parser("scan-stock", help="扫描全A个股，输出“当前买入信号 + 历史胜率/磨损”（研究用途）")
    p_scan_stock.add_argument("--freq", choices=["daily", "weekly"], default="weekly", help="扫描频率（默认 weekly）")
    p_scan_stock.add_argument("--window", type=int, default=500, help="每个标的取最近 N 根K线（默认 500；周线≈10年）")
    p_scan_stock.add_argument("--start-date", default="20100101", help="开始日期（默认 20100101；可选）")
    p_scan_stock.add_argument("--end-date", default=None, help="结束日期（可选）")
    p_scan_stock.add_argument("--adjust", default=None, help="复权方式（qfq/hfq/空，可选；默认 qfq）")
    p_scan_stock.add_argument(
        "--daily-filter",
        choices=["none", "ma20", "macd"],
        default="macd",
        help="日线辅助过滤（默认 macd）",
    )
    p_scan_stock.add_argument(
        "--base-filters",
        default="trend_template",
        help="基础环境过滤器，逗号分隔（默认 trend_template；填 none 关闭）",
    )
    p_scan_stock.add_argument("--tt-near-high", type=float, default=0.25, help="趋势模板：距离52周高点最大回撤比例（默认 0.25）")
    p_scan_stock.add_argument("--tt-above-low", type=float, default=0.30, help="趋势模板：高于52周低点最小涨幅比例（默认 0.30）")
    p_scan_stock.add_argument("--tt-slope-weeks", type=int, default=4, help="趋势模板：MA40 上行判断回看周数（默认 4）")
    p_scan_stock.add_argument("--horizons", default="4,8,12", help="胜率统计持有周数，逗号分隔（默认 4,8,12）")
    p_scan_stock.add_argument("--rank-horizon", type=int, default=8, help="榜单排序使用的 horizon（默认 8）")
    p_scan_stock.add_argument("--min-weeks", type=int, default=120, help="周K 少于该值直接跳过（默认 120）")
    p_scan_stock.add_argument("--min-trades", type=int, default=12, help="排序口径 trades 少于该值的直接过滤（默认 12）")
    p_scan_stock.add_argument("--min-price", type=float, default=0.0, help="过滤股价低于该值的标的（默认 0=不过滤）")
    p_scan_stock.add_argument("--max-price", type=float, default=0.0, help="过滤股价高于该值的标的（默认 0=不过滤）")
    p_scan_stock.add_argument(
        "--min-amount",
        type=float,
        default=0.0,
        help="过滤周线成交额(优先用 amount；否则 close*volume) 小于该值的标的（默认 0=不过滤）",
    )
    p_scan_stock.add_argument("--limit", type=int, default=0, help="只扫描前 N 个（默认 0=全量；按代码排序）")
    p_scan_stock.add_argument("--top-k", type=int, default=50, help="输出 Top K（默认 50）")
    p_scan_stock.add_argument("--workers", type=int, default=8, help="并发抓数/计算线程数（默认 8）")
    p_scan_stock.add_argument("--buy-cost", type=float, default=0.001, help="买入成本（默认 0.001=0.10%%）")
    p_scan_stock.add_argument("--sell-cost", type=float, default=0.002, help="卖出成本（默认 0.002=0.20%%，含印花税的保守估计）")
    p_scan_stock.add_argument("--allow-overlap", action="store_true", help="允许信号样本重叠（默认不允许，避免假高胜率）")
    p_scan_stock.add_argument("--include-st", action="store_true", help="包含 ST/*ST（默认排除）")
    p_scan_stock.add_argument("--exclude-bj", action="store_true", help="排除北交所（默认包含）")
    p_scan_stock.add_argument("--cache-dir", default=None, help="缓存目录（默认 data/cache/stock）")
    p_scan_stock.add_argument("--cache-ttl-hours", type=float, default=24.0, help="缓存有效期（小时，默认 24）")
    p_scan_stock.add_argument("--out-dir", default=None, help="输出目录（默认 outputs/stock_scan_<timestamp>）")
    p_scan_stock.add_argument("--verbose", action="store_true", help="打印扫描进度（可选）")
    p_scan_stock.set_defaults(func=cmd_scan_stock)

    p_clean = sub.add_parser("clean-outputs", help="清理 outputs 历史产物（默认 dry-run）")
    p_clean.add_argument("--path", default="outputs", help="输出目录（默认 outputs）")
    p_clean.add_argument("--keep-days", type=float, default=7.0, help="保留最近 N 天（默认 7）")
    p_clean.add_argument("--keep-last", type=int, default=20, help="额外保留最近 N 个条目（默认 20）")
    p_clean.add_argument("--include-logs", action="store_true", help="连 outputs 下的 .log 文件也一起清（默认不动日志）")
    p_clean.add_argument("--apply", action="store_true", help="真正执行删除（默认只 dry-run）")
    p_clean.set_defaults(func=cmd_clean_outputs)

    p_srv = sub.add_parser("serve", help="启动 Web 前端（本机浏览器打开）")
    p_srv.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    p_srv.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    p_srv.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载（可选）")
    p_srv.add_argument("--log-level", default="info", help="uvicorn 日志级别（默认 info）")
    p_srv.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        os.environ.setdefault("PYTHONUTF8", "1")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
