from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


class WarehouseError(RuntimeError):
    pass


@dataclass(frozen=True)
class WarehousePaths:
    root_dir: Path
    db_path: Path

    # sentinel files: make DuckDB globs never go empty, otherwise it会报错（艹，体验极差）
    sentinel_etf_csv: Path
    sentinel_stock_csv: Path
    sentinel_index_csv: Path
    sentinel_crypto_csv: Path
    sentinel_outputs_dir: Path


def _repo_root() -> Path:
    # llm_trading/warehouse.py -> repo root
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text_if_missing(path: Path, text: str) -> None:
    """
    写哨兵文件：默认“缺了就写”；但如果我们升级了哨兵 schema，也需要能自动覆盖旧版本。
    - 如果文件存在且内容一致：不动（避免无意义刷新 mtime）
    - 如果文件存在但内容不同：覆盖（让 sql-init 不被旧哨兵卡住）
    """
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except (AttributeError):  # noqa: BLE001
            # 读失败就别碰了，避免把未知状态写坏（用户手工改过/权限问题）
            return
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def build_warehouse_paths(
    *,
    root_dir: Path | None = None,
    db_path: Path | None = None,
) -> WarehousePaths:
    root = (root_dir or _repo_root()).resolve()
    dbp = (db_path or (root / "data" / "warehouse.duckdb")).resolve()
    return WarehousePaths(
        root_dir=root,
        db_path=dbp,
        sentinel_etf_csv=root / "data" / "cache" / "etf" / "etf__duckdb_sentinel.csv",
        sentinel_stock_csv=root / "data" / "cache" / "stock" / "stock__duckdb_sentinel.csv",
        sentinel_index_csv=root / "data" / "cache" / "index" / "index__duckdb_sentinel.csv",
        sentinel_crypto_csv=root / "data" / "cache" / "crypto" / "crypto__duckdb_sentinel.csv",
        sentinel_outputs_dir=root / "outputs" / "_duckdb_sentinel",
    )


def ensure_warehouse_sentinels(paths: WarehousePaths) -> None:
    """
    DuckDB 的 read_csv_auto/read_json_auto：glob 匹配不到文件就直接炸。
    所以我给每个关键目录放一个“空哨兵文件”，保证视图永远能跑出来（哪怕是空表）。
    """

    # Bars CSV sentinels (empty, header only)
    _write_text_if_missing(
        paths.sentinel_etf_csv,
        "date,open,close,high,low,volume,amount\n",
    )
    _write_text_if_missing(
        paths.sentinel_stock_csv,
        "date,股票代码,open,close,high,low,volume,amount\n",
    )
    _write_text_if_missing(
        paths.sentinel_index_csv,
        "date,open,high,low,close,volume\n",
    )
    _write_text_if_missing(
        paths.sentinel_crypto_csv,
        "date,open,high,low,close,amount\n",
    )

    # Outputs JSON sentinels
    paths.sentinel_outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) 单标的分析产物
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "meta.json",
        (
            "{\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "source": "sentinel",\n'
            '  "source_requested": "sentinel",\n'
            '  "csv": null,\n'
            '  "asset": "etf",\n'
            '  "symbol": "000000",\n'
            '  "freq": "weekly",\n'
            '  "method": "all",\n'
            '  "title": "sentinel",\n'
            '  "window": 0,\n'
            '  "rows": 0,\n'
            '  "start_date": "1970-01-01",\n'
            '  "end_date": "1970-01-01",\n'
            '  "columns": []\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "signal_backtest.json",
        (
            "{\n"
            '  "mode": "win_rate_first",\n'
            '  "params": {"horizons": [4, 8, 12], "rank_horizon_weeks": 8, "min_trades": 20, "buy_cost": 0.001, "sell_cost": 0.002},\n'
            '  "risk_signals": {"weekly_below_ma50_confirm2": false, "daily_macd_bearish_2d": false, "daily_close_below_ma20_confirm2": false},\n'
            '  "signals_now": {"trend": false, "swing": false},\n'
            '  "stats": {},\n'
            '  "decision": {"action": "观望", "chosen_signal": "none", "suggested_horizon_weeks": 8, "reasons": ["sentinel"]}\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "tushare_factors.json",
        (
            "{\n"
            '  "ok": true,\n'
            '  "as_of": "1970-01-01",\n'
            '  "erp": {\n'
            '    "ok": false,\n'
            '    "equity_yield": null,\n'
            '    "rf": {"tenor": "1y", "value_pct": null, "yield": null},\n'
            '    "erp": null,\n'
            '    "ref_date_index": null,\n'
            '    "ref_date_rf": null,\n'
            '    "erp_alt_10y": null,\n'
            '    "rf_alt_10y": {"rf": {"value_pct": null, "yield": null, "score01": null}}\n'
            "  },\n"
            '  "hsgt": {\n'
            '    "ok": false,\n'
            '    "ref_date": null,\n'
            '    "north": {"money_yuan": null, "score01": null},\n'
            '    "south": {"money_yuan": null, "score01": null}\n'
            "  },\n"
            '  "microstructure": {\n'
            '    "ok": false,\n'
            '    "ref_date": null,\n'
            '    "last": {"net_big_amount_yuan": null, "net_mf_amount_yuan": null, "amount_yuan": null, "net_big_ratio": null, "net_total_ratio": null},\n'
            '    "z": null,\n'
            '    "score01": null\n'
            "  },\n"
            '  "source": {"name": "sentinel"}\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "etf_holdings_top10.json",
        (
            "{\n"
            '  "schema": "llm_trading.etf_holdings_top10.v1",\n'
            '  "ok": true,\n'
            '  "asset": "etf",\n'
            '  "symbol": "sh000000",\n'
            '  "fund_code": "000000",\n'
            '  "as_of": "1970-01-01",\n'
            '  "report_period": "1970Q1",\n'
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "source": {"name": "sentinel"},\n'
            '  "rows": [\n'
            "    {\n"
            '      "rank": 0,\n'
            '      "stock_code": "000000",\n'
            '      "stock_symbol": "sz000000",\n'
            '      "stock_ts_code": "000000.SZ",\n'
            '      "stock_name": "sentinel",\n'
            '      "weight_pct": null,\n'
            '      "weight": null,\n'
            '      "shares_wan": null,\n'
            '      "hold_value_wan": null\n'
            "    }\n"
            "  ]\n"
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "game_theory_factors.json",
        (
            "{\n"
            '  "schema": "llm_trading.game_theory_factors.v1",\n'
            '  "symbol": "000000",\n'
            '  "asset": "etf",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "source": "sentinel",\n'
            '  "factors": {\n'
            '    "liquidity_trap": {"name": "liquidity_trap", "value": 0, "score": 0, "direction": "neutral", "confidence": 0, "details": {"trap_kind": null, "level": null, "sweep_pct": null, "swing_high": null, "swing_low": null, "volume_ratio": null, "amount_ratio": null}},\n'
            '    "stop_cluster": {"name": "stop_cluster", "value": 0, "score": 0, "direction": "neutral", "confidence": 0, "details": {"nearest_level": null, "nearest_kind": null, "nearest_distance_pct": null, "ma20": null, "ma60": null, "ma200": null, "swing_high": null, "swing_low": null, "integer_level": null, "zones": []}},\n'
            '    "capitulation": {"name": "capitulation", "value": 0, "score": 0, "direction": "neutral", "confidence": 0, "details": {"atr": null, "move_atr": null, "volume_ratio": null, "rsi": null}},\n'
            '    "fomo": {"name": "fomo", "value": 0, "score": 0, "direction": "neutral", "confidence": 0, "details": {"atr": null, "move_atr": null, "volume_ratio": null, "rsi": null}},\n'
            '    "wyckoff_phase_proxy": {"name": "wyckoff_phase_proxy", "value": 0, "score": 0.5, "direction": "neutral", "confidence": 0, "details": {"accumulation_like": null, "distribution_like": null, "range_width_pct": null, "vol_contract_score": null, "obv_divergence_score": null}}\n'
            "  }\n"
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "opportunity_score.json",
        (
            "{\n"
            '  "schema": "llm_trading.opportunity_score.v1",\n'
            '  "symbol": "000000",\n'
            '  "asset": "etf",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "source": "sentinel",\n'
            '  "total_score": 0,\n'
            '  "min_score": 0.7,\n'
            '  "verdict": "not_tradeable",\n'
            '  "bucket": "reject",\n'
            '  "components": {"trend": null, "regime": null, "risk_reward": null, "liquidity": null, "trap_risk": null, "fund_flow": null},\n'
            '  "key_level": {"name": "ma50", "value": null},\n'
            '  "invalidation": {"rule": "close_below_level", "level": null, "note": "sentinel"},\n'
            '  "expected_holding_days": 10,\n'
            '  "t_plus_one": true,\n'
            '  "notes": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "cash_signal.json",
        (
            "{\n"
            '  "schema": "llm_trading.cash_signal.v1",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "source": "sentinel",\n'
            '  "scope": "portfolio",\n'
            '  "context_index_symbol": "sh000300+sh000905",\n'
            '  "should_stay_cash": true,\n'
            '  "cash_ratio": 0.8,\n'
            '  "risk_mode": "risk_off",\n'
            '  "expected_duration_days": 10,\n'
            '  "evidence": {"market_regime": "unknown", "vol_state": "unknown", "erp_proxy": null, "north_score01": null, "south_score01": null},\n'
            '  "reason": "sentinel",\n'
            '  "notes": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "position_sizing.json",
        (
            "{\n"
            '  "schema": "llm_trading.position_sizing.v1",\n'
            '  "symbol": "000000",\n'
            '  "asset": "etf",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "source": "sentinel",\n'
            '  "opportunity_score": 0,\n'
            '  "confidence": 0,\n'
            '  "max_position_pct": 0.3,\n'
            '  "suggest_position_pct": 0,\n'
            '  "equity_yuan": null,\n'
            '  "cash_yuan": null,\n'
            '  "price": null,\n'
            '  "lot_size": 1,\n'
            '  "min_trade_notional_yuan": 2000,\n'
            '  "suggest_trade_notional_yuan": 0,\n'
            '  "suggest_shares": null,\n'
            '  "est_commission_yuan": 0,\n'
            '  "est_slippage_yuan": 0,\n'
            '  "t_plus_one": true,\n'
            '  "reason": "sentinel",\n'
            '  "notes": "sentinel"\n'
            "}\n"
        ),
    )

    _write_text_if_missing(
        paths.sentinel_outputs_dir / "strategy_signal.json",
        (
            "{\n"
            '  "schema": "llm_trading.strategy_signal.v1",\n'
            '  "symbol": "000000",\n'
            '  "asset": "etf",\n'
            '  "freq": "weekly",\n'
            '  "as_of": "1970-01-01",\n'
            '  "strategy_key": "sentinel",\n'
            '  "strategy_config": "config/strategy_configs.yaml",\n'
            '  "market_regime": {"index": "sh000300", "payload": {"symbol": "sh000300", "label": "unknown"}, "error": null},\n'
            '  "signal": {"action": "hold", "score": 0.5, "confidence": 0.0, "factors": {}, "reason": "sentinel"}\n'
            "}\n"
        ),
    )

    # 3) 因子研究（Phase1 / P0）
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "factor_research_summary.json",
        (
            "{\n"
            '  "schema": "llm_trading.factor_research.v1",\n'
            '  "asset": "etf",\n'
            '  "freq": "daily",\n'
            '  "as_of": "1970-01-01",\n'
            '  "start_date": null,\n'
            '  "horizons": [1, 5, 10, 20],\n'
            '  "universe_size": 0,\n'
            '  "symbols_used": 0,\n'
            '  "tradeability": {"total_rows": 0, "tradeable_rows": 0, "blocked_rows": 0, "blocked_breakdown": {}, "cfg": {"limit_up_pct": 0, "limit_down_pct": 0}},\n'
            '  "cost": {"min_fee_yuan_each_side": 5, "slippage_bps_each_side": 10, "notional_yuan": 2000, "roundtrip_cost_rate": 0.0},\n'
            '  "factors": [\n'
            '    {"factor": "sentinel", "ic_1": null, "ir_1": null, "ic_samples_1": 0, "avg_cross_n_1": null, "ic_5": null, "ir_5": null, "ic_samples_5": 0, "avg_cross_n_5": null, "ic_10": null, "ir_10": null, "ic_samples_10": 0, "avg_cross_n_10": null, "ic_20": null, "ir_20": null, "ic_samples_20": 0, "avg_cross_n_20": null}\n'
            "  ],\n"
            '  "source": {"name": "sentinel"},\n'
            '  "generated_at": "1970-01-01T00:00:00"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "factor_research_macro.json",
        (
            "{\n"
            '  "schema": "llm_trading.factor_research_macro.v1",\n'
            '  "ok": true,\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "asset": "index",\n'
            '  "freq": "daily",\n'
            '  "context_index_symbol": "sh000300",\n'
            '  "t_plus_one": true,\n'
            '  "horizons": [1, 5, 10, 20],\n'
            '  "start_date": null,\n'
            '  "source": {"name": "sentinel"},\n'
            '  "cost": {"roundtrip_cost_rate": 0.0},\n'
            '  "components": {},\n'
            '  "factors": [\n'
            '    {"factor": "sentinel", "ic_1": null, "ir_1": null, "ic_samples_1": 0, "avg_cross_n_1": null, "ic_5": null, "ir_5": null, "ic_samples_5": 0, "avg_cross_n_5": null, "ic_10": null, "ir_10": null, "ic_samples_10": 0, "avg_cross_n_10": null, "ic_20": null, "ir_20": null, "ic_samples_20": 0, "avg_cross_n_20": null}\n'
            "  ],\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "notes": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "factor_research_whitelist.json",
        (
            "{\n"
            '  "schema": "llm_trading.factor_research_whitelist.v1",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "asset": "etf",\n'
            '  "freq": "daily",\n'
            '  "source": {"name": "sentinel"},\n'
            '  "whitelist": [],\n'
            '  "blacklist": [],\n'
            '  "notes": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "factor_research_symbols.json",
        (
            "{\n"
            '  "schema": "llm_trading.factor_research_symbols.v1",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "asset": "etf",\n'
            '  "freq": "daily",\n'
            '  "source": {"name": "sentinel"},\n'
            '  "symbols": []\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "factor_research_ic.csv",
        "date,factor,horizon,ic,n_obs\n",
    )

    # 4) 动态权重（Phase4）
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "dynamic_weights_summary.json",
        (
            "{\n"
            '  "schema": "llm_trading.dynamic_weights_research.v1",\n'
            '  "asset": "etf",\n'
            '  "freq": "weekly",\n'
            '  "as_of": "1970-01-01",\n'
            '  "ref_date": "1970-01-01",\n'
            '  "start_date": null,\n'
            '  "universe_size": 0,\n'
            '  "symbols_used": 0,\n'
            '  "baseline_regime": "neutral",\n'
            '  "regime_weights": {"path": "config/regime_weights.yaml", "weights": {}},\n'
            '  "market_regime": {"context_index_symbol": "sh000300", "error": null},\n'
            '  "cost": {"roundtrip_cost_rate": 0.0},\n'
            '  "factors": [\n'
            "    {\n"
            '      "factor": "dw_static",\n'
            '      "ic_1": null, "ir_1": null, "ic_samples_1": 0, "ic_train_1": null, "ic_test_1": null, "top20_gross_mean_1": null, "top20_net_mean_1": null,\n'
            '      "wf_windows_1": 0, "wf_ic_train_mean_1": null, "wf_ic_test_mean_1": null, "wf_ic_test_median_1": null, "wf_ic_test_pos_ratio_1": null,\n'
            '      "ic_5": null, "ir_5": null, "ic_samples_5": 0, "ic_train_5": null, "ic_test_5": null, "top20_gross_mean_5": null, "top20_net_mean_5": null,\n'
            '      "wf_windows_5": 0, "wf_ic_train_mean_5": null, "wf_ic_test_mean_5": null, "wf_ic_test_median_5": null, "wf_ic_test_pos_ratio_5": null,\n'
            '      "ic_10": null, "ir_10": null, "ic_samples_10": 0, "ic_train_10": null, "ic_test_10": null, "top20_gross_mean_10": null, "top20_net_mean_10": null,\n'
            '      "wf_windows_10": 0, "wf_ic_train_mean_10": null, "wf_ic_test_mean_10": null, "wf_ic_test_median_10": null, "wf_ic_test_pos_ratio_10": null,\n'
            '      "ic_20": null, "ir_20": null, "ic_samples_20": 0, "ic_train_20": null, "ic_test_20": null, "top20_gross_mean_20": null, "top20_net_mean_20": null,\n'
            '      "wf_windows_20": 0, "wf_ic_train_mean_20": null, "wf_ic_test_mean_20": null, "wf_ic_test_median_20": null, "wf_ic_test_pos_ratio_20": null\n'
            "    },\n"
            "    {\n"
            '      "factor": "dw_dynamic",\n'
            '      "ic_1": null, "ir_1": null, "ic_samples_1": 0, "ic_train_1": null, "ic_test_1": null, "top20_gross_mean_1": null, "top20_net_mean_1": null,\n'
            '      "wf_windows_1": 0, "wf_ic_train_mean_1": null, "wf_ic_test_mean_1": null, "wf_ic_test_median_1": null, "wf_ic_test_pos_ratio_1": null,\n'
            '      "ic_5": null, "ir_5": null, "ic_samples_5": 0, "ic_train_5": null, "ic_test_5": null, "top20_gross_mean_5": null, "top20_net_mean_5": null,\n'
            '      "wf_windows_5": 0, "wf_ic_train_mean_5": null, "wf_ic_test_mean_5": null, "wf_ic_test_median_5": null, "wf_ic_test_pos_ratio_5": null,\n'
            '      "ic_10": null, "ir_10": null, "ic_samples_10": 0, "ic_train_10": null, "ic_test_10": null, "top20_gross_mean_10": null, "top20_net_mean_10": null,\n'
            '      "wf_windows_10": 0, "wf_ic_train_mean_10": null, "wf_ic_test_mean_10": null, "wf_ic_test_median_10": null, "wf_ic_test_pos_ratio_10": null,\n'
            '      "ic_20": null, "ir_20": null, "ic_samples_20": 0, "ic_train_20": null, "ic_test_20": null, "top20_gross_mean_20": null, "top20_net_mean_20": null,\n'
            '      "wf_windows_20": 0, "wf_ic_train_mean_20": null, "wf_ic_test_mean_20": null, "wf_ic_test_median_20": null, "wf_ic_test_pos_ratio_20": null\n'
            "    }\n"
            "  ],\n"
            '  "generated_at": "1970-01-01T00:00:00"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "dynamic_weights_ic.csv",
        "date,factor,horizon,ic,n_obs,top20_gross,top20_net,asset,freq,as_of,ref_date,source\n",
    )

    # 5) 策略对齐（Phase3）
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "alignment.json",
        (
            "{\n"
            '  "schema": "llm_trading.strategy_alignment.v1",\n'
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "base": {"file": "sentinel", "strategy": "sentinel", "as_of": "1970-01-01"},\n'
            '  "new": {"file": "sentinel", "strategy": "sentinel", "as_of": "1970-01-01"},\n'
            '  "universe": {"symbols": 0},\n'
            '  "entry_confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},\n'
            '  "mismatch": {"count": 0, "rate": 0.0},\n'
            '  "top_k": {"k": 30, "base_entry": 0, "new_entry": 0, "overlap_rate": 0.0, "overlap_symbols": []}\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "mismatches.csv",
        "symbol,base_action,new_action,base_score,new_score\n",
    )

    # 2) 单标的各流派子目录（可选，但先放着，免得你哪天只跑了某个流派，SQL 直接报错）
    (paths.sentinel_outputs_dir / "wyckoff").mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "wyckoff" / "wyckoff_features.json",
        (
            "{\n"
            '  "method": "wyckoff_features",\n'
            '  "last": {"date": "1970-01-01", "close": null, "ma50": null, "ma200": null, "ad_line": null},\n'
            '  "derived": {}\n'
            "}\n"
        ),
    )

    (paths.sentinel_outputs_dir / "turtle").mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "turtle" / "turtle.json",
        (
            "{\n"
            '  "method": "turtle",\n'
            '  "params": {"entry": 20, "exit": 10, "atr": 20, "stop_atr": 2.0},\n'
            '  "last": {"date": "1970-01-01", "close": null},\n'
            '  "signals": {},\n'
            '  "risk": {}\n'
            "}\n"
        ),
    )

    (paths.sentinel_outputs_dir / "ichimoku").mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "ichimoku" / "ichimoku.json",
        (
            "{\n"
            '  "method": "ichimoku",\n'
            '  "params": {"tenkan": 9, "kijun": 26, "span_b": 52, "displacement": 26},\n'
            '  "last": {"date": "1970-01-01", "close": null, "position": "unknown", "tk_cross": "none"}\n'
            "}\n"
        ),
    )

    (paths.sentinel_outputs_dir / "chan").mkdir(parents=True, exist_ok=True)
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "chan" / "chan_structure.json",
        "{\n  \"method\": \"chan\",\n  \"ok\": true,\n  \"note\": \"sentinel\"\n}\n",
    )

    # 3) 跑批类产物（scan/run/paper/monitor 等）
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "run_meta.json",
        "{\n  \"argv\": [],\n  \"generated_at\": \"1970-01-01T00:00:00\",\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "run_config.json",
        "{\n  \"argv\": [],\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "report.json",
        "{\n  \"ok\": true,\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "errors.json",
        "{\n  \"errors\": [],\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "signals.json",
        (
            "{\n"
            '  "schema_version": 1,\n'
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "as_of": "1970-01-01",\n'
            '  "strategy": "sentinel",\n'
            '  "source": {"type": "sentinel", "file": null},\n'
            '  "market_regime": {"symbol": "sh000300", "label": "unknown"},\n'
            '  "items": [\n'
            "    {\n"
            '      "asset": "etf",\n'
            '      "symbol": "000000",\n'
            '      "name": "sentinel",\n'
            '      "action": "entry",\n'
            '      "score": null,\n'
            '      "confidence": null,\n'
            '      "confidence_ref": null,\n'
            '      "entry": {"price_ref": null, "price_ref_type": "close", "notes": null},\n'
            '      "meta": {\n'
            '        "close": null,\n'
            '        "pct_chg": null,\n'
            '        "amount": null,\n'
            '        "liquidity": {\n'
            '          "volume_last": null,\n'
            '          "volume_avg20": null,\n'
            '          "amount_last": null,\n'
            '          "amount_avg20": null,\n'
            '          "turnover_est_last": null,\n'
            '          "daily_volume_last": null,\n'
            '          "daily_volume_avg20": null,\n'
            '          "daily_amount_last": null,\n'
            '          "daily_amount_avg20": null\n'
            "        },\n"
            '        "levels": {\n'
            '          "support_20w": null,\n'
            '          "resistance_20w": null,\n'
            '          "atr": null,\n'
            '          "ma50": null,\n'
            '          "ma200": null\n'
            "        },\n"
            '        "exit": {\n'
            '          "suggestion": "hold",\n'
            '          "weekly": {"below_ma50_confirm2": false},\n'
            '          "daily": {"bearish_confirm2": false}\n'
            "        }\n"
            "      },\n"
            '      "tags": []\n'
            "    }\n"
            "  ],\n"
            '  "note": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "top_bbb.json",
        (
            "{\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "freq": "weekly",\n'
            '  "bbb": {"mode_effective": "sentinel", "market_regime": {"label": "unknown"}},\n'
            '  "items": [\n'
            "    {\n"
            '      "symbol": "000000",\n'
            '      "name": "sentinel",\n'
            '      "fund_type": null,\n'
            '      "last_date": "1970-01-01",\n'
            '      "last_daily_date": "1970-01-01",\n'
            '      "close": null,\n'
            '      "pct_chg": null,\n'
            '      "amount": null,\n'
            '      "liquidity": {"amount_avg20": null},\n'
            '      "levels": {"support_20w": null, "resistance_20w": null, "ma50": null, "atr": null},\n'
            '      "momentum": {"macd_state": "unknown", "adx": null},\n'
            '      "scores": {"trend": 0, "swing": 0},\n'
            '      "exit": {"suggestion": "hold", "weekly": {"below_ma50_confirm2": false}, "daily": {"bearish_confirm2": false}},\n'
            '      "bbb": {"score": null, "why": null},\n'
            '      "bbb_factor7": {"score": null},\n'
            '      "factor_panel": {\n'
            '        "ok": false,\n'
            '        "as_of": "1970-01-01",\n'
            '        "rs": {"index": null, "rs_12w": null, "rs_26w": null},\n'
            '        "mom": {"mom_12w": null, "mom_26w": null},\n'
            '        "trend": {"adx14": null},\n'
            '        "vol": {"vol_20d": null, "atr14_pct": null},\n'
            '        "drawdown": {"dd_252d": null, "from_low_252d": null},\n'
            '        "liquidity": {"amount_avg20": null, "amount_ratio": null, "volume_ratio": null},\n'
            '        "boll": {"bandwidth": null, "bandwidth_rel": null, "squeeze": null}\n'
            "      },\n"
            '      "factor_panel_7": {\n'
            '        "ok": false,\n'
            '        "as_of": "1970-01-01",\n'
            '        "rs": {"index": null, "rs_12w": null, "rs_26w": null},\n'
            '        "mom": {"mom_12w": null, "mom_26w": null},\n'
            '        "trend": {"adx14": null},\n'
            '        "vol": {"vol_20d": null, "atr14_pct": null},\n'
            '        "drawdown": {"dd_252d": null, "from_low_252d": null},\n'
            '        "liquidity": {"amount_avg20": null, "amount_ratio": null, "volume_ratio": null},\n'
            '        "boll": {"bandwidth": null, "bandwidth_rel": null, "squeeze": null}\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "note": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "position_plan.json",
        "{\n  \"plan\": [],\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "orders_next_open.json",
        (
            "{\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "orders": [\n'
            "    {\n"
            '      "side": "buy",\n'
            '      "asset": "etf",\n'
            '      "symbol": "000000",\n'
            '      "shares": 0,\n'
            '      "lot_size": 100,\n'
            '      "signal_date": "1970-01-01",\n'
            '      "exec": "next_open",\n'
            '      "price_ref": null,\n'
            '      "price_ref_type": "close",\n'
            '      "order_type": "market",\n'
            '      "est_notional_yuan": null,\n'
            '      "est_cash": null,\n'
            '      "est_fee_yuan": null,\n'
            '      "halt_risk": false,\n'
            '      "limit_up_risk": false,\n'
            '      "reason": "sentinel"\n'
            "    }\n"
            "  ],\n"
            '  "note": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "holdings_user.json",
        (
            "{\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "as_of": "1970-01-01",\n'
            '  "regime_index": "sh000300",\n'
            '  "market_regime": {"label": "unknown"},\n'
            '  "holdings": [\n'
            "    {\n"
            '      "asset": "etf",\n'
            '      "symbol": "000000",\n'
            '      "name": "sentinel",\n'
            '      "ok": false,\n'
            '      "asof": "1970-01-01",\n'
            '      "close": null,\n'
            '      "shares": 0,\n'
            '      "cost": null,\n'
            '      "market_value": null,\n'
            '      "pnl_net": null,\n'
            '      "pnl_net_pct": null,\n'
            '      "stops": {\n'
            '        "hard_stop": null,\n'
            '        "hard_ref": null,\n'
            '        "loss_stop": null,\n'
            '        "loss_tol_pct": null,\n'
            '        "profit_enabled": false,\n'
            '        "profit_stop": null,\n'
            '        "profit_ref": null,\n'
            '        "effective_stop": null,\n'
            '        "effective_ref": null\n'
            "      },\n"
            '      "factor_panel": {\n'
            '        "ok": false,\n'
            '        "as_of": "1970-01-01",\n'
            '        "rs": {"index": null, "rs_12w": null, "rs_26w": null},\n'
            '        "mom": {"mom_12w": null, "mom_26w": null},\n'
            '        "trend": {"adx14": null},\n'
            '        "vol": {"vol_20d": null, "atr14_pct": null},\n'
            '        "drawdown": {"dd_252d": null, "from_low_252d": null},\n'
            '        "liquidity": {"amount_avg20": null, "amount_ratio": null, "volume_ratio": null},\n'
            '        "boll": {"bandwidth": null, "bandwidth_rel": null, "squeeze": null}\n'
            "      },\n"
            '      "factor_panel_7": {\n'
            '        "ok": false,\n'
            '        "as_of": "1970-01-01",\n'
            '        "rs": {"index": null, "rs_12w": null, "rs_26w": null},\n'
            '        "mom": {"mom_12w": null, "mom_26w": null},\n'
            '        "trend": {"adx14": null},\n'
            '        "vol": {"vol_20d": null, "atr14_pct": null},\n'
            '        "drawdown": {"dd_252d": null, "from_low_252d": null},\n'
            '        "liquidity": {"amount_avg20": null, "amount_ratio": null, "volume_ratio": null},\n'
            '        "boll": {"bandwidth": null, "bandwidth_rel": null, "squeeze": null}\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "portfolio": {},\n'
            '  "note": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "rebalance_user.json",
        (
            "{\n"
            '  "generated_at": "1970-01-01T00:00:00",\n'
            '  "as_of": {"signals": "1970-01-01", "holdings": "1970-01-01"},\n'
            '  "mode": "add",\n'
            '  "inputs": {"holdings_path": null, "signals_path": null},\n'
            '  "account": {"cash_yuan": null, "equity_yuan": null},\n'
            '  "market_regime": {"label": "unknown"},\n'
            '  "position_plan": {"profile": {}, "budget": {}, "plans": [], "watch": []},\n'
            '  "rebalance": {\n'
            '    "orders_next_open": [\n'
            "      {\n"
            '        "side": "buy",\n'
            '        "asset": "etf",\n'
            '        "symbol": "000000",\n'
            '        "shares": 0,\n'
            '        "lot_size": 100,\n'
            '        "signal_date": "1970-01-01",\n'
            '        "exec": "next_open",\n'
            '        "price_ref": null,\n'
            '        "price_ref_type": "close",\n'
            '        "order_type": "market",\n'
            '        "est_notional_yuan": null,\n'
            '        "est_cash": null,\n'
            '        "est_fee_yuan": null,\n'
            '        "halt_risk": false,\n'
            '        "limit_up_risk": false,\n'
            '        "reason": "sentinel"\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "note": "sentinel"\n'
            "}\n"
        ),
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "monitor.json",
        "{\n  \"ok\": true,\n  \"note\": \"sentinel\"\n}\n",
    )
    _write_text_if_missing(
        paths.sentinel_outputs_dir / "paper_sim.json",
        "{\n  \"ok\": true,\n  \"note\": \"sentinel\"\n}\n",
    )


def _classify_kind(rel_path: str) -> str:
    # 只要能稳定分类就行：给 SQL 做过滤用；别搞得太花。
    p = rel_path.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]

    if p == "data/user_holdings.json":
        return "user_holdings"

    if p.startswith("data/cache/etf/") and base.endswith(".csv"):
        return "cache_etf_bars"
    if p.startswith("data/cache/stock/") and base.endswith(".csv"):
        return "cache_stock_bars"
    if p.startswith("data/cache/index/") and base.endswith(".csv"):
        return "cache_index_bars"
    if p.startswith("data/cache/crypto/") and base.endswith(".csv"):
        return "cache_crypto_bars"

    if p.startswith("data/cache/"):
        # 其他 cache：比如 etf_spot/etf_share/northbound/tushare_factors 等
        return "data_cache_other"

    if p.startswith("outputs/"):
        if base == "meta.json":
            return "out_meta"
        if base == "signal_backtest.json":
            return "out_signal_backtest"
        if base == "tushare_factors.json":
            return "out_tushare_factors"
        if base == "etf_holdings_top10.json":
            return "out_etf_holdings_top10"
        if base == "game_theory_factors.json":
            return "out_game_theory_factors"
        if base == "opportunity_score.json":
            return "out_opportunity_score"
        if base == "cash_signal.json":
            return "out_cash_signal"
        if base == "position_sizing.json":
            return "out_position_sizing"
        if base == "strategy_signal.json":
            return "out_strategy_signal"
        if base == "factor_research_summary.json":
            return "out_factor_research_summary"
        if base == "factor_research_macro.json":
            return "out_factor_research_macro"
        if base == "factor_research_whitelist.json":
            return "out_factor_research_whitelist"
        if base == "factor_research_symbols.json":
            return "out_factor_research_symbols"
        if base == "factor_research_ic.csv":
            return "out_factor_research_ic"
        if base == "dynamic_weights_summary.json":
            return "out_dynamic_weights_summary"
        if base == "dynamic_weights_ic.csv":
            return "out_dynamic_weights_ic"
        if base == "alignment.json":
            return "out_strategy_alignment"
        if base == "mismatches.csv":
            return "out_strategy_alignment_mismatches"

        if base == "wyckoff_features.json":
            return "out_wyckoff_features"
        if base == "turtle.json":
            return "out_turtle"
        if base == "ichimoku.json":
            return "out_ichimoku"
        if base == "chan_structure.json":
            return "out_chan_structure"

        # run/scan/paper/monitor 常见产物
        if base in {
            "run_meta.json",
            "run_config.json",
            "report.json",
            "errors.json",
            "signals.json",
            "signals_merged.json",
            "top_trend.json",
            "top_swing.json",
            "top_bbb.json",
            "position_plan.json",
            "orders_next_open.json",
            "holdings_user.json",
            "rebalance_user.json",
            "alerts.json",
            "monitor.json",
            "paper_sim.json",
        }:
            return f"out_{base.removesuffix('.json')}"
        return "outputs_other"

    return "other"


def scan_structured_files(root: Path) -> list[dict]:
    """
    扫 data/ + outputs/ 下的结构化文件（csv/json）。
    注意：排除 data/cache/_mpl 和 data/cache/_xdg，这俩是字体缓存，放进来纯属恶心人。
    """

    root = root.resolve()
    records: list[dict] = []

    skip_dir_suffixes = {
        "/data/cache/_mpl",
        "/data/cache/_xdg",
    }

    for area in ["data", "outputs"]:
        base_dir = root / area
        if not base_dir.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(base_dir):
            dirpath_p = Path(dirpath)
            dirpath_posix = dirpath_p.as_posix()
            if any(dirpath_posix.endswith(sfx) for sfx in skip_dir_suffixes):
                dirnames[:] = []
                continue

            for fn in filenames:
                if not (fn.endswith(".csv") or fn.endswith(".json")):
                    continue

                abs_path = dirpath_p / fn
                try:
                    st = abs_path.stat()
                except FileNotFoundError:
                    continue

                rel_path = abs_path.relative_to(root).as_posix()
                kind = _classify_kind(rel_path)
                records.append(
                    {
                        "rel_path": rel_path,
                        "abs_path": abs_path.as_posix(),
                        "area": area,
                        "ext": abs_path.suffix.lstrip(".").lower(),
                        "kind": kind,
                        "basename": fn,
                        "parent": abs_path.parent.relative_to(root).as_posix(),
                        "size_bytes": int(st.st_size),
                        "mtime": datetime.fromtimestamp(st.st_mtime),
                    }
                )

    # 稳定排序：方便 diff/排查
    records.sort(key=lambda r: r["rel_path"])
    return records


def _sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _duckdb_import_or_die():
    try:
        import duckdb  # type: ignore

        return duckdb
    except ModuleNotFoundError as exc:
        raise WarehouseError(
            "缺 duckdb 依赖。先装：\n"
            "  \".venv/bin/python\" -m pip install -r \"requirements-py312.txt\"\n"
            "或：\n"
            "  \".venv/bin/python\" -m pip install \"duckdb>=1.4,<2.0\""
        ) from exc


def _create_or_refresh_catalog(con, rows: list[dict]) -> None:
    import pandas as pd

    con.execute("create schema if not exists wh")
    con.execute(
        """
        create table if not exists wh.file_catalog (
          rel_path text primary key,
          abs_path text not null,
          area text not null,
          ext text not null,
          kind text not null,
          basename text not null,
          parent text not null,
          size_bytes bigint,
          mtime timestamp
        )
        """
    )
    con.execute("delete from wh.file_catalog")
    if not rows:
        return

    df = pd.DataFrame(rows)
    con.register("_tmp_file_catalog", df)
    con.execute(
        """
        insert into wh.file_catalog
        select
          rel_path,
          abs_path,
          area,
          ext,
          kind,
          basename,
          parent,
          size_bytes,
          mtime
        from _tmp_file_catalog
        """
    )
    con.unregister("_tmp_file_catalog")


def _create_views(con, paths: WarehousePaths) -> None:
    """
    只建“通用高频”的视图：
    - bars（etf/stock/index/crypto）统一成一个大视图
    - outputs 常见 JSON（meta/signal_backtest/tushare_factors/run_meta/report/signals/top_bbb/…）
    其他你要啥再加，不然视图爆炸了你自己也看不懂。
    """

    root = paths.root_dir

    # CSV bars
    pat_etf = (root / "data" / "cache" / "etf" / "etf_*.csv").as_posix()
    pat_stock = (root / "data" / "cache" / "stock" / "stock_*.csv").as_posix()
    pat_index = (root / "data" / "cache" / "index" / "index_*.csv").as_posix()
    pat_crypto = (root / "data" / "cache" / "crypto" / "crypto_*.csv").as_posix()

    con.execute("create schema if not exists wh")

    con.execute(
        f"""
        create or replace view wh.v_bars_etf as
        select
          'etf' as asset,
          regexp_extract(filename, 'etf_([a-z]{{2}}\\d{{6}})', 1) as symbol,
          regexp_extract(filename, '_(qfq|hfq)\\.csv$', 1) as adjust,
          try_cast(date as date) as date,
          try_cast(open as double) as open,
          try_cast(high as double) as high,
          try_cast(low as double) as low,
          try_cast(close as double) as close,
          try_cast(volume as double) as volume,
          try_cast(amount as double) as amount,
          filename as _file
        from read_csv_auto({_sql_quote(pat_etf)}, filename=true, union_by_name=true)
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_bars_stock as
        select
          'stock' as asset,
          regexp_extract(filename, 'stock_([a-z]{{2}}\\d{{6}})', 1) as symbol,
          regexp_extract(filename, '_(qfq|hfq)\\.csv$', 1) as adjust,
          try_cast(date as date) as date,
          try_cast(open as double) as open,
          try_cast(high as double) as high,
          try_cast(low as double) as low,
          try_cast(close as double) as close,
          try_cast(volume as double) as volume,
          try_cast(amount as double) as amount,
          filename as _file
        from read_csv_auto({_sql_quote(pat_stock)}, filename=true, union_by_name=true)
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_bars_index as
        select
          'index' as asset,
          regexp_extract(filename, 'index_([a-z]{{2}}\\d{{6}})', 1) as symbol,
          regexp_extract(filename, '_(qfq|hfq)\\.csv$', 1) as adjust,
          try_cast(date as date) as date,
          try_cast(open as double) as open,
          try_cast(high as double) as high,
          try_cast(low as double) as low,
          try_cast(close as double) as close,
          try_cast(volume as double) as volume,
          cast(null as double) as amount,
          filename as _file
        from read_csv_auto({_sql_quote(pat_index)}, filename=true, union_by_name=true)
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_bars_crypto as
        select
          'crypto' as asset,
          regexp_extract(filename, 'crypto_([a-z0-9]+)', 1) as symbol,
          cast(null as text) as adjust,
          try_cast(date as date) as date,
          try_cast(open as double) as open,
          try_cast(high as double) as high,
          try_cast(low as double) as low,
          try_cast(close as double) as close,
          cast(null as double) as volume,
          try_cast(amount as double) as amount,
          filename as _file
        from read_csv_auto({_sql_quote(pat_crypto)}, filename=true, union_by_name=true)
        """
    )

    con.execute(
        """
        create or replace view wh.v_bars as
        select * from wh.v_bars_etf
        union all
        select * from wh.v_bars_stock
        union all
        select * from wh.v_bars_index
        union all
        select * from wh.v_bars_crypto
        """
    )

    # Outputs JSON (1-level)
    pat_outputs_1 = (root / "outputs" / "*" / "*.json").as_posix()
    con.execute(
        f"""
        create or replace view wh.v_outputs_json as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            regexp_extract(filename, '/([^/]+)\\.json$', 1) as name,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_outputs_1)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # Analysis outputs (top level)
    pat_meta = (root / "outputs" / "*" / "meta.json").as_posix()
    pat_sig = (root / "outputs" / "*" / "signal_backtest.json").as_posix()
    pat_factors = (root / "outputs" / "*" / "tushare_factors.json").as_posix()
    pat_etf_hold = (root / "outputs" / "*" / "etf_holdings_top10.json").as_posix()
    pat_gt = (root / "outputs" / "*" / "game_theory_factors.json").as_posix()
    pat_opp = (root / "outputs" / "*" / "opportunity_score.json").as_posix()
    pat_cash = (root / "outputs" / "*" / "cash_signal.json").as_posix()
    pat_ps = (root / "outputs" / "*" / "position_sizing.json").as_posix()
    pat_strategy_sig = (root / "outputs" / "*" / "strategy_signal.json").as_posix()
    pat_fr_sum = (root / "outputs" / "*" / "factor_research_summary.json").as_posix()
    pat_fr_ic = (root / "outputs" / "*" / "factor_research_ic.csv").as_posix()
    pat_fr_macro = (root / "outputs" / "*" / "factor_research_macro.json").as_posix()
    pat_dw_sum = (root / "outputs" / "*" / "dynamic_weights_summary.json").as_posix()
    pat_dw_ic = (root / "outputs" / "*" / "dynamic_weights_ic.csv").as_posix()
    pat_align = (root / "outputs" / "*" / "alignment.json").as_posix()
    pat_align_mm = (root / "outputs" / "*" / "mismatches.csv").as_posix()
    # run/scan 常见产物（结构化 JSON）
    pat_signals = (root / "outputs" / "*" / "signals*.json").as_posix()
    pat_top_bbb = (root / "outputs" / "*" / "top_bbb.json").as_posix()
    pat_holdings_user = (root / "outputs" / "*" / "holdings_user.json").as_posix()
    pat_rebalance_user = (root / "outputs" / "*" / "rebalance_user.json").as_posix()
    pat_orders_next_open = (root / "outputs" / "*" / "orders_next_open.json").as_posix()
    pat_alerts = (root / "outputs" / "*" / "alerts.json").as_posix()

    con.execute(
        f"""
        create or replace view wh.v_analysis_meta as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_meta)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_signal_backtest as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_sig)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_tushare_factors as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_factors)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_etf_holdings_top10 as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_etf_hold)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_tushare_factors_flat as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            try_cast(ok as boolean) as ok,
            try_cast(as_of as date) as as_of,
            -- ERP proxy (main)
            try_cast(erp.equity_yield as double) as erp_equity_yield,
            try_cast(erp.rf.yield as double) as erp_rf_yield,
            try_cast(erp.erp as double) as erp,
            try_cast(erp.rf.value_pct as double) as erp_rf_value_pct,
            erp.rf.tenor as erp_rf_tenor,
            erp.ref_date_index as erp_ref_date_index,
            erp.ref_date_rf as erp_ref_date_rf,
            -- ERP proxy (alt: 10Y gov bond, for comparison)
            try_cast(erp.erp_alt_10y as double) as erp_alt_10y,
            try_cast(erp.rf_alt_10y.rf.yield as double) as erp_rf_10y_yield,
            try_cast(erp.rf_alt_10y.rf.value_pct as double) as erp_rf_10y_value_pct,
            try_cast(erp.rf_alt_10y.rf.score01 as double) as erp_rf_10y_score01,
            -- HSGT (north/south) scores
            hsgt.ref_date as hsgt_ref_date,
            try_cast(hsgt.north.money_yuan as double) as hsgt_north_money_yuan,
            try_cast(hsgt.north.score01 as double) as hsgt_north_score01,
            try_cast(hsgt.south.money_yuan as double) as hsgt_south_money_yuan,
            try_cast(hsgt.south.score01 as double) as hsgt_south_score01,
            -- Stock microstructure (moneyflow big/superbig proxy)
            microstructure.ref_date as micro_ref_date,
            try_cast(microstructure.last.net_big_amount_yuan as double) as micro_net_big_amount_yuan,
            try_cast(microstructure.last.net_mf_amount_yuan as double) as micro_net_mf_amount_yuan,
            try_cast(microstructure.last.amount_yuan as double) as micro_amount_yuan,
            try_cast(microstructure.last.net_big_ratio as double) as micro_net_big_ratio,
            try_cast(microstructure.last.net_total_ratio as double) as micro_net_total_ratio,
            try_cast(microstructure.z as double) as micro_z,
            try_cast(microstructure.score01 as double) as micro_score01
          from read_json_auto({_sql_quote(pat_factors)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_etf_holdings_top10_items as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            t.schema as schema,
            try_cast(t.ok as boolean) as ok,
            t.asset as asset,
            t.symbol as symbol,
            t.fund_code as fund_code,
            try_cast(t.as_of as date) as as_of,
            t.report_period as report_period,
            try_cast(t.generated_at as timestamp) as generated_at,
            u.r.rank as rank,
            cast(u.r.stock_code as text) as stock_code,
            cast(u.r.stock_symbol as text) as stock_symbol,
            cast(u.r.stock_ts_code as text) as stock_ts_code,
            cast(u.r.stock_name as text) as stock_name,
            try_cast(u.r.weight_pct as double) as weight_pct,
            try_cast(u.r.weight as double) as weight,
            try_cast(u.r.shares_wan as double) as shares_wan,
            try_cast(u.r.hold_value_wan as double) as hold_value_wan
          from read_json_auto({_sql_quote(pat_etf_hold)}, filename=true, union_by_name=true) t
          cross join unnest(t.rows) as u(r)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # run/scan outputs (flattened)
    con.execute(
        f"""
        create or replace view wh.v_signals_items as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            try_cast(t.schema_version as int) as schema_version,
            try_cast(t.generated_at as timestamp) as generated_at,
            try_cast(t.as_of as date) as as_of,
            t.strategy as strategy,
            t.source.type as source_type,
            t.source.file as source_file,
            t.market_regime.symbol as market_regime_symbol,
            t.market_regime.label as market_regime_label,
            u.it.asset as asset,
            u.it.symbol as symbol,
            u.it.name as name,
            u.it.action as action,
            try_cast(u.it.score as double) as score,
            try_cast(u.it.confidence as double) as confidence,
            u.it.confidence_ref as confidence_ref,
            try_cast(u.it.entry.price_ref as double) as entry_price_ref,
            u.it.entry.price_ref_type as entry_price_ref_type,
            u.it.entry.notes as entry_notes,
            try_cast(u.it.meta.close as double) as close,
            try_cast(u.it.meta.pct_chg as double) as pct_chg,
            try_cast(u.it.meta.amount as double) as amount,
            try_cast(u.it.meta.liquidity.amount_last as double) as amount_last,
            try_cast(u.it.meta.liquidity.amount_avg20 as double) as amount_avg20,
            try_cast(u.it.meta.liquidity.amount_last as double) / nullif(try_cast(u.it.meta.liquidity.amount_avg20 as double), 0.0) as amount_ratio,
            try_cast(u.it.meta.liquidity.volume_last as double) as volume_last,
            try_cast(u.it.meta.liquidity.volume_avg20 as double) as volume_avg20,
            try_cast(u.it.meta.liquidity.volume_last as double) / nullif(try_cast(u.it.meta.liquidity.volume_avg20 as double), 0.0) as volume_ratio,
            try_cast(u.it.meta.levels.support_20w as double) as support_20w,
            try_cast(u.it.meta.levels.resistance_20w as double) as resistance_20w,
            try_cast(u.it.meta.levels.atr as double) as atr,
            try_cast(u.it.meta.levels.ma50 as double) as ma50,
            try_cast(u.it.meta.levels.ma200 as double) as ma200,
            u.it.meta.exit.suggestion as exit_suggestion,
            try_cast(u.it.meta.exit.weekly.below_ma50_confirm2 as boolean) as weekly_below_ma50_confirm2,
            try_cast(u.it.meta.exit.daily.bearish_confirm2 as boolean) as daily_bearish_confirm2,
            u.it.tags as tags
          from read_json_auto({_sql_quote(pat_signals)}, filename=true, union_by_name=true) t
          cross join unnest(t.items) as u(it)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_top_bbb_items as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            try_cast(t.generated_at as timestamp) as generated_at,
            t.freq as freq,
            t.bbb.mode_effective as bbb_mode_effective,
            t.bbb.market_regime.label as market_regime_label,
            u.it.symbol as symbol,
            u.it.name as name,
            u.it.fund_type as fund_type,
            try_cast(u.it.last_date as date) as last_date,
            try_cast(u.it.last_daily_date as date) as last_daily_date,
            try_cast(u.it.close as double) as close,
            try_cast(u.it.pct_chg as double) as pct_chg,
            try_cast(u.it.amount as double) as amount,
            try_cast(u.it.liquidity.amount_avg20 as double) as amount_avg20,
            try_cast(u.it.levels.support_20w as double) as support_20w,
            try_cast(u.it.levels.resistance_20w as double) as resistance_20w,
            try_cast(u.it.levels.ma50 as double) as ma50,
            try_cast(u.it.levels.atr as double) as atr,
            u.it.momentum.macd_state as macd_state,
            try_cast(u.it.momentum.adx as double) as adx,
            try_cast(u.it.scores.trend as int) as score_trend,
            try_cast(u.it.scores.swing as int) as score_swing,
            u.it.exit.suggestion as exit_suggestion,
            try_cast(u.it.exit.weekly.below_ma50_confirm2 as boolean) as weekly_below_ma50_confirm2,
            try_cast(u.it.exit.daily.bearish_confirm2 as boolean) as daily_bearish_confirm2,
            try_cast(u.it.bbb.score as double) as bbb_score,
            u.it.bbb.why as bbb_why,
            try_cast(u.it.bbb_factor7.score as double) as bbb_factor7_score,
            -- factor panel: top_bbb 历史产物里字段是 factor_panel_7（scan-etf 新增 factor_panel 只是别名，不保证存在）
            u.it.factor_panel_7.rs.index as fp_rs_index,
            try_cast(u.it.factor_panel_7.rs.rs_12w as double) as fp_rs_12w,
            try_cast(u.it.factor_panel_7.rs.rs_26w as double) as fp_rs_26w,
            try_cast(u.it.factor_panel_7.mom.mom_12w as double) as fp_mom_12w,
            try_cast(u.it.factor_panel_7.mom.mom_26w as double) as fp_mom_26w,
            try_cast(u.it.factor_panel_7.trend.adx14 as double) as fp_adx14,
            try_cast(u.it.factor_panel_7.vol.vol_20d as double) as fp_vol_20d,
            try_cast(u.it.factor_panel_7.vol.atr14_pct as double) as fp_atr14_pct,
            try_cast(u.it.factor_panel_7.drawdown.dd_252d as double) as fp_dd_252d,
            try_cast(u.it.factor_panel_7.drawdown.from_low_252d as double) as fp_from_low_252d,
            try_cast(u.it.factor_panel_7.liquidity.amount_avg20 as double) as fp_amount_avg20,
            try_cast(u.it.factor_panel_7.liquidity.amount_ratio as double) as fp_amount_ratio,
            try_cast(u.it.factor_panel_7.liquidity.volume_ratio as double) as fp_volume_ratio,
            try_cast(u.it.factor_panel_7.boll.bandwidth as double) as fp_boll_bandwidth,
            try_cast(u.it.factor_panel_7.boll.bandwidth_rel as double) as fp_boll_bandwidth_rel,
            try_cast(u.it.factor_panel_7.boll.squeeze as boolean) as fp_boll_squeeze
          from read_json_auto({_sql_quote(pat_top_bbb)}, filename=true, union_by_name=true) t
          cross join unnest(t.items) as u(it)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_holdings_user_holdings as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            try_cast(t.as_of as date) as as_of,
            t.regime_index as regime_index,
            t.market_regime.label as market_regime_label,
            u.h.asset as asset,
            u.h.symbol as symbol,
            u.h.name as name,
            try_cast(u.h.ok as boolean) as ok,
            try_cast(u.h.asof as date) as holding_asof,
            try_cast(u.h.close as double) as close,
            try_cast(u.h.shares as int) as shares,
            try_cast(u.h.cost as double) as cost,
            try_cast(u.h.market_value as double) as market_value,
            try_cast(u.h.pnl_net as double) as pnl_net,
            try_cast(u.h.pnl_net_pct as double) as pnl_net_pct,
            try_cast(u.h.stops.hard_stop as double) as hard_stop,
            u.h.stops.hard_ref as hard_ref,
            try_cast(u.h.stops.loss_stop as double) as loss_stop,
            try_cast(u.h.stops.loss_tol_pct as double) as loss_tol_pct,
            try_cast(u.h.stops.profit_enabled as boolean) as profit_enabled,
            try_cast(u.h.stops.profit_stop as double) as profit_stop,
            u.h.stops.profit_ref as profit_ref,
            try_cast(u.h.stops.effective_stop as double) as effective_stop,
            u.h.stops.effective_ref as effective_ref,
            -- factor panel：holdings-user 输出里两字段同值；但历史产物可能只有 factor_panel_7，所以视图只吃 factor_panel_7
            u.h.factor_panel_7.rs.index as fp_rs_index,
            try_cast(u.h.factor_panel_7.rs.rs_12w as double) as fp_rs_12w,
            try_cast(u.h.factor_panel_7.rs.rs_26w as double) as fp_rs_26w,
            try_cast(u.h.factor_panel_7.mom.mom_12w as double) as fp_mom_12w,
            try_cast(u.h.factor_panel_7.mom.mom_26w as double) as fp_mom_26w,
            try_cast(u.h.factor_panel_7.trend.adx14 as double) as fp_adx14,
            try_cast(u.h.factor_panel_7.vol.vol_20d as double) as fp_vol_20d,
            try_cast(u.h.factor_panel_7.vol.atr14_pct as double) as fp_atr14_pct,
            try_cast(u.h.factor_panel_7.drawdown.dd_252d as double) as fp_dd_252d,
            try_cast(u.h.factor_panel_7.drawdown.from_low_252d as double) as fp_from_low_252d,
            try_cast(u.h.factor_panel_7.liquidity.amount_avg20 as double) as fp_amount_avg20,
            try_cast(u.h.factor_panel_7.liquidity.amount_ratio as double) as fp_amount_ratio,
            try_cast(u.h.factor_panel_7.liquidity.volume_ratio as double) as fp_volume_ratio,
            try_cast(u.h.factor_panel_7.boll.bandwidth as double) as fp_boll_bandwidth,
            try_cast(u.h.factor_panel_7.boll.bandwidth_rel as double) as fp_boll_bandwidth_rel,
            try_cast(u.h.factor_panel_7.boll.squeeze as boolean) as fp_boll_squeeze
          from read_json_auto({_sql_quote(pat_holdings_user)}, filename=true, union_by_name=true) t
          cross join unnest(t.holdings) as u(h)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_orders_next_open_orders as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            try_cast(t.generated_at as timestamp) as generated_at,
            u.o.side as side,
            u.o.asset as asset,
            u.o.symbol as symbol,
            try_cast(u.o.shares as int) as shares,
            try_cast(u.o.lot_size as int) as lot_size,
            try_cast(u.o.signal_date as date) as signal_date,
            u.o.exec as exec,
            try_cast(u.o.price_ref as double) as price_ref,
            u.o.price_ref_type as price_ref_type,
            u.o.order_type as order_type,
            try_cast(u.o.est_notional_yuan as double) as est_notional_yuan,
            try_cast(u.o.est_cash as double) as est_cash,
            try_cast(u.o.est_fee_yuan as double) as est_fee_yuan,
            try_cast(u.o.halt_risk as boolean) as halt_risk,
            try_cast(u.o.limit_up_risk as boolean) as limit_up_risk,
            u.o.reason as reason
          from read_json_auto({_sql_quote(pat_orders_next_open)}, filename=true, union_by_name=true) t
          cross join unnest(t.orders) as u(o)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_rebalance_user_orders_next_open as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            try_cast(t.generated_at as timestamp) as generated_at,
            try_cast(t.as_of.signals as date) as as_of_signals,
            try_cast(t.as_of.holdings as date) as as_of_holdings,
            t.mode as mode,
            t.inputs.holdings_path as holdings_path,
            t.inputs.signals_path as signals_path,
            try_cast(t.account.cash_yuan as double) as cash_yuan,
            try_cast(t.account.equity_yuan as double) as equity_yuan,
            t.market_regime.label as market_regime_label,
            u.o.side as side,
            u.o.asset as asset,
            u.o.symbol as symbol,
            try_cast(u.o.shares as int) as shares,
            try_cast(u.o.lot_size as int) as lot_size,
            try_cast(u.o.signal_date as date) as signal_date,
            u.o.exec as exec,
            try_cast(u.o.price_ref as double) as price_ref,
            u.o.price_ref_type as price_ref_type,
            u.o.order_type as order_type,
            try_cast(u.o.est_notional_yuan as double) as est_notional_yuan,
            try_cast(u.o.est_cash as double) as est_cash,
            try_cast(u.o.est_fee_yuan as double) as est_fee_yuan,
            try_cast(u.o.halt_risk as boolean) as halt_risk,
            try_cast(u.o.limit_up_risk as boolean) as limit_up_risk,
            u.o.reason as reason
          from read_json_auto({_sql_quote(pat_rebalance_user)}, filename=true, union_by_name=true) t
          cross join unnest(t.rebalance.orders_next_open) as u(o)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # Phase2 outputs (flattened)
    con.execute(
        f"""
        create or replace view wh.v_game_theory_factors as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            symbol,
            asset,
            as_of,
            ref_date,
            source,
            try_cast(factors.liquidity_trap.score as double) as liquidity_trap_score,
            try_cast(factors.liquidity_trap.confidence as double) as liquidity_trap_confidence,
            factors.liquidity_trap.direction as liquidity_trap_direction,
            try_cast(factors.liquidity_trap.details.trap_kind as varchar) as liquidity_trap_kind,
            try_cast(factors.liquidity_trap.details.level as double) as liquidity_trap_level,
            try_cast(factors.liquidity_trap.details.sweep_pct as double) as liquidity_trap_sweep_pct,
            try_cast(factors.liquidity_trap.details.swing_high as double) as liquidity_trap_swing_high,
            try_cast(factors.liquidity_trap.details.swing_low as double) as liquidity_trap_swing_low,
            try_cast(factors.liquidity_trap.details.volume_ratio as double) as liquidity_trap_volume_ratio,
            try_cast(factors.liquidity_trap.details.amount_ratio as double) as liquidity_trap_amount_ratio,

            try_cast(factors.stop_cluster.score as double) as stop_cluster_score,
            try_cast(factors.stop_cluster.confidence as double) as stop_cluster_confidence,
            try_cast(factors.stop_cluster.details.nearest_level as double) as stop_cluster_nearest_level,
            try_cast(factors.stop_cluster.details.nearest_kind as varchar) as stop_cluster_nearest_kind,
            try_cast(factors.stop_cluster.details.nearest_distance_pct as double) as stop_cluster_nearest_distance_pct,

            try_cast(factors.capitulation.score as double) as capitulation_score,
            factors.capitulation.direction as capitulation_direction,
            try_cast(factors.capitulation.details.move_atr as double) as capitulation_move_atr,
            try_cast(factors.capitulation.details.volume_ratio as double) as capitulation_volume_ratio,
            try_cast(factors.capitulation.details.rsi as double) as capitulation_rsi,

            try_cast(factors.fomo.score as double) as fomo_score,
            factors.fomo.direction as fomo_direction,
            try_cast(factors.fomo.details.move_atr as double) as fomo_move_atr,
            try_cast(factors.fomo.details.volume_ratio as double) as fomo_volume_ratio,
            try_cast(factors.fomo.details.rsi as double) as fomo_rsi,

            try_cast(factors.wyckoff_phase_proxy.score as double) as wyckoff_score,
            factors.wyckoff_phase_proxy.direction as wyckoff_direction,
            try_cast(factors.wyckoff_phase_proxy.details.accumulation_like as double) as wyckoff_accumulation_like,
            try_cast(factors.wyckoff_phase_proxy.details.distribution_like as double) as wyckoff_distribution_like,
            try_cast(factors.wyckoff_phase_proxy.details.range_width_pct as double) as wyckoff_range_width_pct,
            try_cast(factors.wyckoff_phase_proxy.details.vol_contract_score as double) as wyckoff_vol_contract_score,
            try_cast(factors.wyckoff_phase_proxy.details.obv_divergence_score as double) as wyckoff_obv_divergence_score
          from read_json_auto({_sql_quote(pat_gt)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_opportunity_score as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            symbol,
            asset,
            as_of,
            ref_date,
            source,
            try_cast(total_score as double) as total_score,
            try_cast(min_score as double) as min_score,
            verdict,
            bucket,
            try_cast(components.trend as double) as comp_trend,
            try_cast(components.regime as double) as comp_regime,
            try_cast(components.risk_reward as double) as comp_risk_reward,
            try_cast(components.liquidity as double) as comp_liquidity,
            try_cast(components.trap_risk as double) as comp_trap_risk,
            try_cast(components.fund_flow as double) as comp_fund_flow,
            key_level.name as key_level_name,
            try_cast(key_level.value as double) as key_level_value,
            invalidation.rule as invalidation_rule,
            try_cast(invalidation.level as double) as invalidation_level,
            invalidation.note as invalidation_note,
            try_cast(expected_holding_days as int) as expected_holding_days,
            try_cast(t_plus_one as boolean) as t_plus_one,
            notes
          from read_json_auto({_sql_quote(pat_opp)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_cash_signal as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            as_of,
            ref_date,
            source,
            scope,
            context_index_symbol,
            try_cast(should_stay_cash as boolean) as should_stay_cash,
            try_cast(cash_ratio as double) as cash_ratio,
            risk_mode,
            try_cast(expected_duration_days as int) as expected_duration_days,
            evidence.market_regime as evidence_market_regime,
            evidence.vol_state as evidence_vol_state,
            try_cast(evidence.erp_proxy as double) as evidence_erp_proxy,
            try_cast(evidence.north_score01 as double) as evidence_north_score01,
            try_cast(evidence.south_score01 as double) as evidence_south_score01,
            reason,
            notes
          from read_json_auto({_sql_quote(pat_cash)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_position_sizing as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            symbol,
            asset,
            as_of,
            ref_date,
            source,
            try_cast(opportunity_score as double) as opportunity_score,
            try_cast(confidence as double) as confidence,
            try_cast(max_position_pct as double) as max_position_pct,
            try_cast(suggest_position_pct as double) as suggest_position_pct,
            try_cast(price as double) as price,
            try_cast(lot_size as int) as lot_size,
            try_cast(min_trade_notional_yuan as double) as min_trade_notional_yuan,
            try_cast(suggest_trade_notional_yuan as double) as suggest_trade_notional_yuan,
            try_cast(suggest_shares as int) as suggest_shares,
            try_cast(est_commission_yuan as double) as est_commission_yuan,
            try_cast(est_slippage_yuan as double) as est_slippage_yuan,
            try_cast(t_plus_one as boolean) as t_plus_one,
            reason,
            notes
          from read_json_auto({_sql_quote(pat_ps)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # Factor research (Phase1 / P0)
    con.execute(
        f"""
        create or replace view wh.v_factor_research_summary as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            asset,
            freq,
            as_of,
            start_date,
            try_cast(universe_size as int) as universe_size,
            try_cast(symbols_used as int) as symbols_used,
            try_cast(cost.roundtrip_cost_rate as double) as roundtrip_cost_rate,
            try_cast(tradeability.total_rows as int) as tradeability_total_rows,
            try_cast(tradeability.tradeable_rows as int) as tradeability_tradeable_rows,
            generated_at
          from read_json_auto({_sql_quote(pat_fr_sum)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_factor_research_factors as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            t.schema,
            t.asset,
            t.freq,
            t.as_of,
            t.start_date,
            u.f.factor as factor,
            try_cast(u.f.ic_1 as double) as ic_1,
            try_cast(u.f.ir_1 as double) as ir_1,
            try_cast(u.f.ic_samples_1 as int) as ic_samples_1,
            try_cast(u.f.ic_5 as double) as ic_5,
            try_cast(u.f.ir_5 as double) as ir_5,
            try_cast(u.f.ic_samples_5 as int) as ic_samples_5,
            try_cast(u.f.ic_10 as double) as ic_10,
            try_cast(u.f.ir_10 as double) as ir_10,
            try_cast(u.f.ic_samples_10 as int) as ic_samples_10,
            try_cast(u.f.ic_20 as double) as ic_20,
            try_cast(u.f.ir_20 as double) as ir_20,
            try_cast(u.f.ic_samples_20 as int) as ic_samples_20
          from read_json_auto({_sql_quote(pat_fr_sum)}, filename=true, union_by_name=true) t
          cross join unnest(t.factors) as u(f)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_factor_research_ic as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            date,
            factor,
            try_cast(horizon as int) as horizon,
            try_cast(ic as double) as ic,
            try_cast(n_obs as int) as n_obs
          from read_csv_auto({_sql_quote(pat_fr_ic)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_factor_research_macro_summary as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            context_index_symbol,
            freq,
            as_of,
            start_date,
            try_cast(cost.roundtrip_cost_rate as double) as roundtrip_cost_rate,
            generated_at
          from read_json_auto({_sql_quote(pat_fr_macro)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_factor_research_macro_factors as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            t.schema,
            t.context_index_symbol,
            t.freq,
            t.as_of,
            u.f.factor as factor,
            try_cast(u.f.ic_1 as double) as ic_1,
            try_cast(u.f.ir_1 as double) as ir_1,
            try_cast(u.f.ic_samples_1 as int) as ic_samples_1,
            try_cast(u.f.ic_5 as double) as ic_5,
            try_cast(u.f.ir_5 as double) as ir_5,
            try_cast(u.f.ic_samples_5 as int) as ic_samples_5,
            try_cast(u.f.ic_10 as double) as ic_10,
            try_cast(u.f.ir_10 as double) as ir_10,
            try_cast(u.f.ic_samples_10 as int) as ic_samples_10,
            try_cast(u.f.ic_20 as double) as ic_20,
            try_cast(u.f.ir_20 as double) as ir_20,
            try_cast(u.f.ic_samples_20 as int) as ic_samples_20
          from read_json_auto({_sql_quote(pat_fr_macro)}, filename=true, union_by_name=true) t
          cross join unnest(t.factors) as u(f)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # Phase3/4: strategy-signal / alignment / dynamic-weights (flattened)
    con.execute(
        f"""
        create or replace view wh.v_strategy_signal as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            symbol,
            asset,
            freq,
            try_cast(as_of as date) as as_of,
            strategy_key,
            strategy_config,
            market_regime.index as market_regime_index,
            market_regime.payload.label as market_regime_label,
            market_regime.error as market_regime_error,
            signal.action as signal_action,
            try_cast(signal.score as double) as signal_score,
            try_cast(signal.confidence as double) as signal_confidence,
            signal.reason as signal_reason,
            signal.factors as signal_factors
          from read_json_auto({_sql_quote(pat_strategy_sig)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_strategy_alignment as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            schema,
            try_cast(generated_at as timestamp) as generated_at,
            base.file as base_file,
            base.strategy as base_strategy,
            try_cast(base.as_of as date) as base_as_of,
            new.file as new_file,
            new.strategy as new_strategy,
            try_cast(new.as_of as date) as new_as_of,
            try_cast(universe.symbols as int) as universe_symbols,
            try_cast(entry_confusion.tp as int) as entry_tp,
            try_cast(entry_confusion.fp as int) as entry_fp,
            try_cast(entry_confusion.fn as int) as entry_fn,
            try_cast(entry_confusion.tn as int) as entry_tn,
            try_cast(mismatch.count as int) as mismatch_count,
            try_cast(mismatch.rate as double) as mismatch_rate,
            try_cast(top_k.k as int) as top_k,
            try_cast(top_k.base_entry as int) as top_k_base_entry,
            try_cast(top_k.new_entry as int) as top_k_new_entry,
            try_cast(top_k.overlap_rate as double) as top_k_overlap_rate,
            top_k.overlap_symbols as top_k_overlap_symbols
          from read_json_auto({_sql_quote(pat_align)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_strategy_alignment_mismatches as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            symbol,
            base_action,
            new_action,
            try_cast(base_score as double) as base_score,
            try_cast(new_score as double) as new_score
          from read_csv_auto({_sql_quote(pat_align_mm)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_dynamic_weights_summary as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            t.schema,
            t.asset,
            t.freq,
            try_cast(t.as_of as date) as as_of,
            try_cast(t.ref_date as date) as ref_date,
            try_cast(t.start_date as date) as start_date,
            try_cast(t.universe_size as int) as universe_size,
            try_cast(t.symbols_used as int) as symbols_used,
            t.baseline_regime,
            t.regime_weights.path as regime_weights_path,
            t.market_regime.context_index_symbol as market_regime_context_index_symbol,
            t.market_regime.error as market_regime_error,
            try_cast(t.cost.roundtrip_cost_rate as double) as roundtrip_cost_rate,
            t.generated_at as generated_at
          from read_json_auto({_sql_quote(pat_dw_sum)}, filename=true, union_by_name=true) t
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_dynamic_weights_factors as
        select * from (
          select
            regexp_extract(t.filename, 'outputs/([^/]+)/', 1) as out_dir,
            t.filename as _file,
            t.schema,
            t.asset,
            t.freq,
            try_cast(t.as_of as date) as as_of,
            t.baseline_regime,
            u.f.factor as factor,
            try_cast(u.f.ic_1 as double) as ic_1,
            try_cast(u.f.ir_1 as double) as ir_1,
            try_cast(u.f.ic_samples_1 as int) as ic_samples_1,
            try_cast(u.f.ic_train_1 as double) as ic_train_1,
            try_cast(u.f.ic_test_1 as double) as ic_test_1,
            try_cast(u.f.top20_gross_mean_1 as double) as top20_gross_mean_1,
            try_cast(u.f.top20_net_mean_1 as double) as top20_net_mean_1,
            try_cast(u.f.wf_windows_1 as int) as wf_windows_1,
            try_cast(u.f.wf_ic_train_mean_1 as double) as wf_ic_train_mean_1,
            try_cast(u.f.wf_ic_test_mean_1 as double) as wf_ic_test_mean_1,
            try_cast(u.f.wf_ic_test_median_1 as double) as wf_ic_test_median_1,
            try_cast(u.f.wf_ic_test_pos_ratio_1 as double) as wf_ic_test_pos_ratio_1,

            try_cast(u.f.ic_5 as double) as ic_5,
            try_cast(u.f.ir_5 as double) as ir_5,
            try_cast(u.f.ic_samples_5 as int) as ic_samples_5,
            try_cast(u.f.ic_train_5 as double) as ic_train_5,
            try_cast(u.f.ic_test_5 as double) as ic_test_5,
            try_cast(u.f.top20_gross_mean_5 as double) as top20_gross_mean_5,
            try_cast(u.f.top20_net_mean_5 as double) as top20_net_mean_5,
            try_cast(u.f.wf_windows_5 as int) as wf_windows_5,
            try_cast(u.f.wf_ic_train_mean_5 as double) as wf_ic_train_mean_5,
            try_cast(u.f.wf_ic_test_mean_5 as double) as wf_ic_test_mean_5,
            try_cast(u.f.wf_ic_test_median_5 as double) as wf_ic_test_median_5,
            try_cast(u.f.wf_ic_test_pos_ratio_5 as double) as wf_ic_test_pos_ratio_5,

            try_cast(u.f.ic_10 as double) as ic_10,
            try_cast(u.f.ir_10 as double) as ir_10,
            try_cast(u.f.ic_samples_10 as int) as ic_samples_10,
            try_cast(u.f.ic_train_10 as double) as ic_train_10,
            try_cast(u.f.ic_test_10 as double) as ic_test_10,
            try_cast(u.f.top20_gross_mean_10 as double) as top20_gross_mean_10,
            try_cast(u.f.top20_net_mean_10 as double) as top20_net_mean_10,
            try_cast(u.f.wf_windows_10 as int) as wf_windows_10,
            try_cast(u.f.wf_ic_train_mean_10 as double) as wf_ic_train_mean_10,
            try_cast(u.f.wf_ic_test_mean_10 as double) as wf_ic_test_mean_10,
            try_cast(u.f.wf_ic_test_median_10 as double) as wf_ic_test_median_10,
            try_cast(u.f.wf_ic_test_pos_ratio_10 as double) as wf_ic_test_pos_ratio_10,

            try_cast(u.f.ic_20 as double) as ic_20,
            try_cast(u.f.ir_20 as double) as ir_20,
            try_cast(u.f.ic_samples_20 as int) as ic_samples_20,
            try_cast(u.f.ic_train_20 as double) as ic_train_20,
            try_cast(u.f.ic_test_20 as double) as ic_test_20,
            try_cast(u.f.top20_gross_mean_20 as double) as top20_gross_mean_20,
            try_cast(u.f.top20_net_mean_20 as double) as top20_net_mean_20,
            try_cast(u.f.wf_windows_20 as int) as wf_windows_20,
            try_cast(u.f.wf_ic_train_mean_20 as double) as wf_ic_train_mean_20,
            try_cast(u.f.wf_ic_test_mean_20 as double) as wf_ic_test_mean_20,
            try_cast(u.f.wf_ic_test_median_20 as double) as wf_ic_test_median_20,
            try_cast(u.f.wf_ic_test_pos_ratio_20 as double) as wf_ic_test_pos_ratio_20
          from read_json_auto({_sql_quote(pat_dw_sum)}, filename=true, union_by_name=true) t
          cross join unnest(t.factors) as u(f)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    con.execute(
        f"""
        create or replace view wh.v_dynamic_weights_ic as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            date,
            factor,
            try_cast(horizon as int) as horizon,
            try_cast(ic as double) as ic,
            try_cast(n_obs as int) as n_obs,
            try_cast(top20_gross as double) as top20_gross,
            try_cast(top20_net as double) as top20_net,
            asset,
            freq,
            try_cast(as_of as date) as as_of,
            try_cast(ref_date as date) as ref_date,
            source
          from read_csv_auto({_sql_quote(pat_dw_ic)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )

    # Analysis per-school (2-level)
    pat_wyckoff = (root / "outputs" / "*" / "wyckoff" / "wyckoff_features.json").as_posix()
    pat_turtle = (root / "outputs" / "*" / "turtle" / "turtle.json").as_posix()
    pat_ichimoku = (root / "outputs" / "*" / "ichimoku" / "ichimoku.json").as_posix()
    pat_chan = (root / "outputs" / "*" / "chan" / "chan_structure.json").as_posix()

    con.execute(
        f"""
        create or replace view wh.v_wyckoff_features as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_wyckoff)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_turtle as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_turtle)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_ichimoku as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_ichimoku)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )
    con.execute(
        f"""
        create or replace view wh.v_chan_structure as
        select * from (
          select
            regexp_extract(filename, 'outputs/([^/]+)/', 1) as out_dir,
            filename as _file,
            *
          from read_json_auto({_sql_quote(pat_chan)}, filename=true, union_by_name=true)
        )
        where out_dir <> '_duckdb_sentinel'
        """
    )


def sql_init(*, db_path: str | None = None, root_dir: str | Path | None = None) -> Path:
    duckdb = _duckdb_import_or_die()
    paths = build_warehouse_paths(
        root_dir=(Path(root_dir) if root_dir else None),
        db_path=(Path(db_path) if db_path else None),
    )

    ensure_warehouse_sentinels(paths)
    _ensure_parent(paths.db_path)

    con = duckdb.connect(paths.db_path.as_posix())
    try:
        rows = scan_structured_files(paths.root_dir)
        _create_or_refresh_catalog(con, rows)
        _create_views(con, paths)
    finally:
        con.close()

    return paths.db_path


def sql_sync(*, db_path: str | None = None, root_dir: str | Path | None = None) -> Path:
    duckdb = _duckdb_import_or_die()
    paths = build_warehouse_paths(
        root_dir=(Path(root_dir) if root_dir else None),
        db_path=(Path(db_path) if db_path else None),
    )

    ensure_warehouse_sentinels(paths)
    _ensure_parent(paths.db_path)

    con = duckdb.connect(paths.db_path.as_posix())
    try:
        rows = scan_structured_files(paths.root_dir)
        _create_or_refresh_catalog(con, rows)
    finally:
        con.close()

    return paths.db_path


def sql_query(
    *,
    sql: str,
    db_path: str | None = None,
    limit: int = 50,
    out: str | None = None,
) -> None:
    duckdb = _duckdb_import_or_die()
    paths = build_warehouse_paths(db_path=Path(db_path) if db_path else None)

    con = duckdb.connect(paths.db_path.as_posix())
    try:
        raw = sql.strip().rstrip(";")
        head = raw.lstrip().split(None, 1)[0].lower() if raw.strip() else ""

        # 只对 SELECT/WITH 做 limit 包装；别瞎包，create/pragma 这种你包了必炸。
        q = raw
        is_selectish = head in {"select", "with"}
        if is_selectish and limit >= 0:
            q = f"select * from ({raw}) as _q limit {int(limit)}"

        if out:
            if not is_selectish:
                raise WarehouseError("--out 只支持 SELECT/WITH 查询（你导出个 create table 我咋 copy？）")
            out_path = Path(out)
            _ensure_parent(out_path)
            # COPY 是 DuckDB 原生导出，速度和稳定性都比 pandas 好
            con.execute(f"COPY ({q}) TO {_sql_quote(out_path.resolve().as_posix())} (HEADER, DELIMITER ',')")
            print(out_path.resolve().as_posix())
            return

        if not is_selectish:
            con.execute(raw)
            print("ok")
            return

        df = con.execute(q).df()
        print(df.to_string(index=False))
    finally:
        con.close()
