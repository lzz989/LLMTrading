from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..pipeline import write_json

from .common import _write_run_config, _write_run_meta


def cmd_dynamic_weights(args: argparse.Namespace) -> int:
    """
    Phase4：动态权重（regime-aware）研究闭环。

    输出目录默认：outputs/walk_forward_<asset>_<timestamp>
    - dynamic_weights_summary.json
    - dynamic_weights_ic.csv
    """
    from ..factors.dynamic_weights import DynamicWeightsResearchParams, run_dynamic_weights_research
    from ..factors.research import FactorResearchParams
    from ..utils_time import parse_date_any_opt

    asset = str(getattr(args, "asset", "") or "").strip().lower()
    if asset not in {"etf", "stock", "index"}:
        raise SystemExit("参数错误：--asset 只能是 etf/stock/index")

    freq = str(getattr(args, "freq", "") or "weekly").strip().lower()
    if freq not in {"weekly"}:
        raise SystemExit("参数错误：dynamic-weights 当前只支持 --freq weekly")

    # 日期
    as_of_dt = parse_date_any_opt(str(getattr(args, "as_of", "") or "").strip() or None)
    start_dt = parse_date_any_opt(str(getattr(args, "start_date", "") or "").strip() or None)
    as_of = as_of_dt.date() if as_of_dt is not None else None
    start_date = start_dt.date() if start_dt is not None else None

    # horizons
    hz_raw = str(getattr(args, "horizons", "") or "1,5,10,20").strip()
    horizons: list[int] = []
    for p in hz_raw.split(","):
        s = p.strip()
        if not s:
            continue
        try:
            horizons.append(int(s))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        horizons = [1, 5, 10, 20]

    # universe
    uni_raw = str(getattr(args, "universe", "") or "").strip().lower()
    limit = int(getattr(args, "limit", 200) or 200)
    if limit <= 0:
        limit = 200

    symbols: list[str] = []
    if asset == "etf":
        include_all_funds = bool(getattr(args, "include_all_funds", False))
        from ..etf_scan import load_etf_universe

        items = load_etf_universe(include_all_funds=include_all_funds)
        symbols = [it.symbol for it in items][:limit]
    elif asset == "stock":
        if (not uni_raw) or uni_raw in {"hs300", "000300"}:
            from ..stock_scan import load_index_stock_universe

            items = load_index_stock_universe(index_symbol="000300")
            symbols = [it.symbol for it in items][:limit]
        elif uni_raw.startswith("index:"):
            from ..stock_scan import load_index_stock_universe

            idx = uni_raw.split(":", 1)[-1].strip() or "000300"
            items = load_index_stock_universe(index_symbol=idx)
            symbols = [it.symbol for it in items][:limit]
        elif uni_raw in {"all", "a"}:
            include_st = bool(getattr(args, "include_st", False))
            include_bj = bool(getattr(args, "include_bj", True))
            from ..stock_scan import load_stock_universe

            items = load_stock_universe(include_st=include_st, include_bj=include_bj)
            symbols = [it.symbol for it in items][:limit]
        else:
            raise SystemExit("参数错误：--universe(stock) 仅支持 hs300 / index:000300 / all")
    else:
        idx_sym = str(getattr(args, "symbol", "") or "").strip() or "sh000300"
        symbols = [idx_sym]

    if not symbols:
        raise SystemExit("universe 为空：没拿到任何 symbol")

    # cache/out
    cache_dir = Path(str(getattr(args, "cache_dir", "") or "").strip() or (Path("data") / "cache" / asset))
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)

    out_dir_raw = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"walk_forward_{asset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tradeability / cost
    default_lim = 0.095 if asset == "stock" else 0.0
    limit_up_pct = float(getattr(args, "limit_up_pct", default_lim) or default_lim)
    limit_down_pct = float(getattr(args, "limit_down_pct", default_lim) or default_lim)
    min_fee_yuan = float(getattr(args, "min_fee_yuan", 5.0) or 5.0)
    slippage_bps = float(getattr(args, "slippage_bps", 10.0) or 10.0)
    notional_yuan = float(getattr(args, "notional_yuan", 2000.0) or 2000.0)

    # walk-forward config
    walk_forward = bool(getattr(args, "walk_forward", True))
    train_window = int(getattr(args, "train_window", 252) or 252)
    test_window = int(getattr(args, "test_window", 63) or 63)
    step_window = int(getattr(args, "step_window", test_window) or test_window)
    min_cross_n = int(getattr(args, "min_cross_n", 30) or 30)
    top_quantile = float(getattr(args, "top_quantile", 0.8) or 0.8)

    ctx = str(getattr(args, "context_index", "") or "sh000300").strip()

    fr = FactorResearchParams(
        asset=asset,  # type: ignore[arg-type]
        freq="weekly",  # type: ignore[arg-type]
        universe=symbols,
        start_date=start_date,
        as_of=as_of,
        horizons=horizons,
        limit_up_pct=limit_up_pct,
        limit_down_pct=limit_down_pct,
        min_fee_yuan=min_fee_yuan,
        slippage_bps_each_side=slippage_bps,
        notional_yuan=notional_yuan,
        walk_forward=bool(walk_forward),
        train_window=int(train_window),
        test_window=int(test_window),
        step_window=int(step_window),
        min_cross_n=int(min_cross_n),
        top_quantile=float(top_quantile),
        context_index_symbol=str(ctx),
    )

    regime_weights_path = Path(str(getattr(args, "regime_weights", "") or "").strip() or (Path("config") / "regime_weights.yaml"))
    baseline_regime = str(getattr(args, "baseline_regime", "") or "neutral").strip().lower() or "neutral"
    dp = DynamicWeightsResearchParams(
        factor_params=fr,
        regime_weights_path=regime_weights_path,
        baseline_regime=baseline_regime,
    )

    res = run_dynamic_weights_research(params=dp, cache_dir=cache_dir, cache_ttl_hours=cache_ttl_hours, out_dir=out_dir, source="auto")

    _write_run_meta(out_dir, args, extra={"cmd": "dynamic-weights"})
    _write_run_config(out_dir, args, note="dynamic weights research", extra={"cmd": "dynamic-weights"})
    write_json(out_dir / "report.json", {"schema": "llm_trading.report.v1", "cmd": "dynamic-weights", "generated_at": datetime.now().isoformat(), "summary": res})

    print(str(out_dir.resolve()))
    return 0

