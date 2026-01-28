from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from ..logger import get_logger
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

_LOG = get_logger(__name__)

from .common import (
    _compute_market_regime_payload,
    _default_out_dir,
    _default_out_dir_for_symbol,
    _write_run_config,
    _write_run_meta,
)

def cmd_plan_etf(args: argparse.Namespace) -> int:
    """
    仓位计划（ETF）：把 scan-etf 的 top_bbb.json 变成“明天买多少 + 止损线”。
    """
    import json

    scan_dir = Path(str(getattr(args, "scan_dir", "") or "").strip()) if getattr(args, "scan_dir", None) else None
    input_path = Path(str(getattr(args, "input", "") or "").strip()) if getattr(args, "input", None) else None

    if input_path is None or str(input_path) == ".":
        if scan_dir is None:
            raise SystemExit("要么传 --scan-dir 指向 scan-etf 输出目录，要么传 --input 指向 top_bbb.json。")
        input_path = scan_dir / "top_bbb.json"

    if not input_path.exists():
        raise SystemExit(f"找不到输入文件：{input_path}")

    try:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"读取 JSON 失败：{input_path} {exc}") from exc

    bbb_cfg = raw.get("bbb") if isinstance(raw, dict) else {}
    bbb_cfg = bbb_cfg if isinstance(bbb_cfg, dict) else {}
    items = raw.get("items") if isinstance(raw, dict) else []
    items = items if isinstance(items, list) else []

    regime = bbb_cfg.get("market_regime") if isinstance(bbb_cfg, dict) else None
    regime = regime if isinstance(regime, dict) else {}
    regime_label = str(regime.get("label") or "unknown")

    # 默认继承 scan-etf 的成本口径，用户也可覆盖
    cap_default = float(bbb_cfg.get("capital_yuan") or 3000.0)
    rt_default = float(bbb_cfg.get("roundtrip_cost_yuan") or 10.0)

    cap = float(getattr(args, "capital_yuan", None) or cap_default)
    rt = float(getattr(args, "roundtrip_cost_yuan", None) or rt_default)

    from ..positioning import PositionPlanParams, build_etf_position_plan, risk_profile_for_regime

    stop_mode = getattr(args, "stop_mode", None)
    stop_mode2 = None if not stop_mode else str(stop_mode).strip()
    if stop_mode2 in {"weekly_entry_ma", "daily_ma20", "atr"}:
        stop_mode3 = stop_mode2  # type: ignore[assignment]
    else:
        stop_mode3 = None

    pp = PositionPlanParams(
        capital_yuan=float(cap),
        roundtrip_cost_yuan=float(rt),
        lot_size=int(getattr(args, "lot_size", 100) or 100),
        max_cost_pct=float(getattr(args, "max_cost_pct", 0.01) or 0.01),
        risk_min_yuan=float(getattr(args, "risk_min_yuan")) if getattr(args, "risk_min_yuan", None) is not None else None,
        risk_per_trade_yuan=float(getattr(args, "risk_per_trade_yuan")) if getattr(args, "risk_per_trade_yuan", None) is not None else None,
        max_exposure_pct=float(getattr(args, "max_exposure_pct")) if getattr(args, "max_exposure_pct", None) is not None else None,
        risk_per_trade_pct=float(getattr(args, "risk_per_trade_pct")) if getattr(args, "risk_per_trade_pct", None) is not None else None,
        stop_mode=stop_mode3,
        max_positions=int(getattr(args, "max_positions")) if getattr(args, "max_positions", None) is not None else None,
        returns_cache_dir=str(getattr(args, "returns_cache_dir")) if getattr(args, "returns_cache_dir", None) else None,
        diversify=bool(getattr(args, "diversify", True)),
        diversify_window_weeks=int(getattr(args, "diversify_window_weeks", 104) or 104),
        diversify_min_overlap_weeks=int(getattr(args, "diversify_min_overlap_weeks", 26) or 26),
        diversify_max_corr=float(getattr(args, "diversify_max_corr", 0.95) or 0.95),
        max_per_theme=int(getattr(args, "max_per_theme", 0) or 0),
        atr_mult=float(getattr(args, "atr_mult", 2.0) or 2.0),
    )

    plan = build_etf_position_plan(items=items, market_regime_label=regime_label, params=pp)
    plan["generated_at"] = datetime.now().isoformat()
    plan["input"] = {
        "source": str(input_path),
        "scan_generated_at": raw.get("generated_at") if isinstance(raw, dict) else None,
        "market_regime_index": bbb_cfg.get("market_regime_index"),
        "market_regime_error": bbb_cfg.get("market_regime_error"),
    }

    out_path = None
    if getattr(args, "out", None):
        out_path = Path(str(args.out))
    else:
        out_path = input_path.parent / "position_plan.json"

    write_json(out_path, plan)
    print(str(out_path.resolve()))
    return 0


def cmd_holdings_etf(args: argparse.Namespace) -> int:
    """
    持仓分析（ETF）：按“收盘价触发 + 跟随牛熊”的风控框架给出止损动作与价位。
    """
    import json

    from ..holdings import analyze_etf_holdings

    raw_items = list(getattr(args, "item", []) or [])
    input_path = str(getattr(args, "input", "") or "").strip()

    holdings: list[dict] = []
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise SystemExit(f"找不到输入文件：{p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"读取 JSON 失败：{p} {exc}") from exc
        hs = data.get("holdings") if isinstance(data, dict) else None
        if isinstance(hs, list):
            for it in hs:
                if not isinstance(it, dict):
                    continue
                holdings.append(
                    {
                        "symbol": str(it.get("symbol") or "").strip(),
                        "shares": int(it.get("shares") or 0),
                        "cost": float(it.get("cost") or 0.0),
                    }
                )

    for s in raw_items:
        s2 = str(s or "").strip()
        if not s2:
            continue
        parts = [p.strip() for p in s2.replace(" ", "").split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(f"--item 格式错误：{s2}；正确例子：512400,500,1.967")
        sym, shares_s, cost_s = parts
        try:
            shares = int(shares_s)
            cost = float(cost_s)
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"--item 数字解析失败：{s2} {exc}") from exc
        holdings.append({"symbol": sym, "shares": shares, "cost": cost})

    if not holdings:
        raise SystemExit("holdings 为空：请传 --item 或 --input")

    out = analyze_etf_holdings(
        holdings=holdings,
        regime_index=str(getattr(args, "regime_index", "sh000300") or "sh000300"),
        regime_canary_downgrade=bool(getattr(args, "regime_canary", True)),
        sell_cost_yuan=float(getattr(args, "sell_cost_yuan", 5.0) or 0.0),
    )

    if getattr(args, "out", None):
        out_path = Path(str(args.out))
        write_json(out_path, out)
        try:
            diag.warnings.extend([str(x) for x in (warnings or []) if str(x).strip()][: diag.max_items])
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass
        diag.write(out_path.parent, cmd="rebalance-user")
        print(str(out_path.resolve()))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _guess_asset_for_user_holding_symbol(symbol: str) -> str:
    """
    给 user_holdings.json 用的粗暴资产类别推断（研究用途）。
    - 6/0/3 开头基本是股票；5/1/15/16/159 之类大概率是 ETF/基金。
    - 兜底：etf（本项目主线就是 ETF 工具，别把默认搞反）
    """
    s = str(symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    if not s.isdigit():
        return "etf"
    if len(s) != 6:
        return "etf"
    if s.startswith(("6", "0", "3")):
        return "stock"
    return "etf"


def _load_user_holdings_snapshot(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], float | None]:
    """
    读取 data/user_holdings.json（或用户指定 path），解析出：
    - 原始 dict
    - holdings（给 analyze_holdings 用）
    - cash_amount（可为空）

    这个函数是给 holdings-user / rebalance-user 复用的，别 tm 到处复制解析逻辑。
    """
    import json
    import math

    p = Path(str(path))
    if not p.exists():
        raise SystemExit(f"找不到持仓文件：{p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"读取 JSON 失败：{p} {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"持仓文件格式不对：根节点必须是 object：{p}")

    positions = data.get("positions")
    if not isinstance(positions, list) or not positions:
        raise SystemExit(f"持仓文件里 positions 为空：{p}")

    cash_amount = None
    cash = data.get("cash")
    if isinstance(cash, dict):
        raw_amt = cash.get("amount")
        try:
            cash_amount = None if raw_amt is None else float(raw_amt)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cash_amount = None
        if cash_amount is not None and (not math.isfinite(float(cash_amount))):
            cash_amount = None

    holdings: list[dict[str, Any]] = []
    for it in positions:
        if not isinstance(it, dict):
            continue

        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue

        try:
            shares = int(it.get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            shares = 0

        # user_holdings.json 用 cost_basis；兼容未来你手动改成 cost
        raw_cost = it.get("cost_basis")
        if raw_cost is None:
            raw_cost = it.get("cost")
        try:
            cost = float(raw_cost or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cost = 0.0

        if shares <= 0 or cost <= 0:
            continue

        asset = str(it.get("asset") or "").strip().lower()
        if asset not in {"etf", "stock"}:
            asset = _guess_asset_for_user_holding_symbol(sym)

        entry_style = str(it.get("entry_style") or it.get("style") or "").strip().lower()
        payload: dict[str, Any] = {"asset": asset, "symbol": sym, "shares": shares, "cost": cost}
        if entry_style:
            payload["entry_style"] = entry_style

        # 冻仓：不自动补仓/不自动调仓（你手动操作不受影响）
        frozen_raw = it.get("frozen")
        if frozen_raw is None:
            frozen_raw = it.get("freeze")
        if isinstance(frozen_raw, bool):
            frozen = frozen_raw
        else:
            frozen_s = str(frozen_raw or "").strip().lower()
            frozen = frozen_s in {"1", "true", "yes", "y", "on"}
        if frozen:
            payload["frozen"] = True
        holdings.append(payload)

    if not holdings:
        raise SystemExit(f"positions 解析后为空（shares/cost_basis 非法？）：{p}")

    return dict(data), holdings, cash_amount


def _grid_exempt_syms_from_user_holdings_snapshot(data: dict[str, Any] | None) -> set[str]:
    """
    从 data/user_holdings.json 快照里提取“网格/组合外标的”集合。

    约定：positions[].grid_plan.enabled=true => rebalance 不自动清仓/不主动轮动（避免拆用户纪律）。
    """
    if not isinstance(data, dict):
        return set()
    ps = data.get("positions")
    if not isinstance(ps, list):
        return set()
    out: set[str] = set()
    for it in ps:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue
        gp = it.get("grid_plan")
        if isinstance(gp, dict) and bool(gp.get("enabled")):
            out.add(sym)
    return out


def cmd_holdings_user(args: argparse.Namespace) -> int:
    """
    读取 data/user_holdings.json 的 positions，一键跑 holdings 分析（ETF/股票）。
    """
    import json

    from ..holdings import analyze_holdings

    path = Path(str(getattr(args, "path", "") or "").strip() or (Path("data") / "user_holdings.json"))
    data, holdings, cash_amount = _load_user_holdings_snapshot(path)
    diag = Diagnostics()

    out = analyze_holdings(
        holdings=holdings,
        regime_index=str(getattr(args, "regime_index", "sh000300") or "sh000300"),
        regime_canary_downgrade=bool(getattr(args, "regime_canary", True)),
        sell_cost_yuan=float(getattr(args, "sell_cost_yuan", 5.0) or 0.0),
        cache_ttl_hours=float(getattr(args, "cache_ttl_hours", 6.0) or 6.0),
        stock_adjust=str(getattr(args, "stock_adjust", "qfq") or "qfq"),
    )

    # 组合层汇总（暴露/集中度/相关性/到止损风险）。
    try:
        from ..portfolio import build_portfolio_summary

        out["portfolio"] = build_portfolio_summary(
            holdings=list(out.get("holdings") or []),
            cash_yuan=cash_amount,
            cache_base_dir=Path("data") / "cache",
            stock_adjust=str(getattr(args, "stock_adjust", "qfq") or "qfq"),
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
    ) as exc:  # noqa: BLE001
        # 不让组合层把主流程炸了：先保证持仓分析能跑出来。
        out["portfolio_error"] = str(exc)
        diag.record("build_portfolio_summary", exc, note="组合层汇总失败（已降级）")

    if getattr(args, "out", None):
        out_path = Path(str(args.out))
        write_json(out_path, out)
        try:
            w2 = out.get("warnings") if isinstance(out, dict) else None
            if isinstance(w2, list):
                diag.warnings.extend([str(x) for x in w2 if str(x).strip()][: diag.max_items])
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass
        diag.write(out_path.parent, cmd="holdings-user")
        print(str(out_path.resolve()))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def cmd_rebalance_user(args: argparse.Namespace) -> int:
    """
    组合层“调仓建议”（研究用途）：
    - 输入：data/user_holdings.json + signals(top_bbb.json/未来 signals.json)
    - 输出：rebalance.json（目标仓位 + 次日开盘买卖清单）

    说明：
    - mode=add：只用现金做“增量加仓”，默认不生成卖出单（更保守，适合小资金/不想被磨损折腾）。
    - mode=rotate：按目标仓位做“轮动/再平衡”，会生成卖出单（包含清掉非目标）。
    """
    import json

    from ..akshare_source import FetchParams, resolve_symbol
    from ..costs import (
        TradeCost,
        bps_to_rate,
        calc_shares_for_capital,
        cash_buy,
        cash_sell,
        estimate_slippage_bps,
        min_notional_for_min_fee,
        trade_cost_from_params,
    )
    from ..data_cache import fetch_daily_cached
    from ..holdings import analyze_holdings
    from ..json_utils import sanitize_for_json
    from ..positioning import PositionPlanParams, build_etf_position_plan, risk_profile_for_regime
    from ..tradeability import TradeabilityConfig, tradeability_flags

    path = Path(str(getattr(args, "path", "") or "").strip() or (Path("data") / "user_holdings.json"))
    data, holdings, cash_amount = _load_user_holdings_snapshot(path)
    diag = Diagnostics()

    # user_holdings.json 的交易规则（可选）
    rules = data.get("trade_rules") if isinstance(data, dict) else None
    if not isinstance(rules, dict):
        rules = data.get("rules") if isinstance(data, dict) else None
    rules = rules if isinstance(rules, dict) else {}

    frozen_syms: set[str] = set()
    for h in (holdings or []):
        if not isinstance(h, dict):
            continue
        if bool(h.get("frozen")):
            sym = str(h.get("symbol") or "").strip()
            if sym:
                frozen_syms.add(sym)

    # 网格/组合外标的：目前先做“最小正确性”——rebalance 不自动清仓它（避免把用户的纪律策略给拆了）。
    # 未来如果要支持“网格仓位的单独订单生成”，再做更细粒度的 base_shares/grid_shares 拆分。
    grid_exempt_syms: set[str] = set()
    try:
        grid_exempt_syms = _grid_exempt_syms_from_user_holdings_snapshot(data)
    except Exception:  # noqa: BLE001
        grid_exempt_syms = set()

    # 单笔买入最小金额：优先 CLI；否则读 user_holdings.json 的 trade_rules
    min_trade_notional_yuan = None
    raw_min_trade = getattr(args, "min_trade_notional_yuan", None)
    if raw_min_trade is not None:
        try:
            v = float(raw_min_trade)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 0.0
        min_trade_notional_yuan = max(0.0, float(v))
    else:
        raw = rules.get("min_trade_notional_yuan")
        try:
            v = float(raw)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 0.0
        if v > 0:
            min_trade_notional_yuan = float(v)
    if min_trade_notional_yuan is None:
        min_trade_notional_yuan = 0.0

    # Phase2：候选质量过滤（OpportunityScore；0~1；默认 0=不过滤）
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    min_score = max(0.0, min(float(min_score), 1.0))

    # Phase2：CashSignal 风控开关（默认关闭：只降不升地限制 max_exposure_pct）
    use_cash_signal = bool(getattr(args, "cash_signal", False))

    # 组合层约束：优先 CLI；否则读 user_holdings.json 的 trade_rules
    max_positions_eff = None
    raw_mp = getattr(args, "max_positions", None)
    if raw_mp is not None:
        try:
            v = int(raw_mp)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 0
        if v > 0:
            max_positions_eff = int(v)
    else:
        raw = rules.get("max_positions")
        try:
            v = int(raw)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 0
        if v > 0:
            max_positions_eff = int(v)

    max_position_pct_eff = None
    raw_mpp = getattr(args, "max_position_pct", None)
    if raw_mpp is not None:
        try:
            v = float(raw_mpp)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            v = 0.0
        if v > 0:
            max_position_pct_eff = max(0.0, min(float(v), 1.0))
    else:
        raw = rules.get("max_position_pct")
        try:
            v = float(raw)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 0.0
        if v > 0:
            max_position_pct_eff = max(0.0, min(float(v), 1.0))

    mode = str(getattr(args, "mode", "add") or "add").strip().lower()
    if mode not in {"add", "rotate"}:
        mode = "add"

    signals_path = Path(str(getattr(args, "signals", "") or "").strip())
    if not str(signals_path) or str(signals_path) == ".":
        raise SystemExit("缺少 --signals：请指向 scan-etf 输出的 top_bbb.json（后续也会支持统一 signals.json）。")
    if not signals_path.exists():
        raise SystemExit(f"找不到 signals 文件：{signals_path}")

    try:
        raw_sig = json.loads(signals_path.read_text(encoding="utf-8"))
    except (AttributeError) as exc:  # noqa: BLE001
        raise SystemExit(f"读取 signals JSON 失败：{signals_path} {exc}") from exc

    # 先跑一遍 holdings 分析：拿 market_regime + 当前市值/现金（用于资本口径）。
    hold_out = analyze_holdings(
        holdings=holdings,
        regime_index=str(getattr(args, "regime_index", "sh000300") or "sh000300"),
        regime_canary_downgrade=bool(getattr(args, "regime_canary", True)),
        sell_cost_yuan=float(getattr(args, "sell_cost_yuan", 5.0) or 0.0),
        cache_ttl_hours=float(getattr(args, "cache_ttl_hours", 6.0) or 6.0),
        stock_adjust=str(getattr(args, "stock_adjust", "qfq") or "qfq"),
    )

    # 组合层汇总：rebalance 需要 equity/cash/exposure 口径（rotate 模式尤其依赖它）。
    port: dict[str, Any] = {}
    try:
        from ..portfolio import build_portfolio_summary

        port = build_portfolio_summary(
            holdings=list((hold_out.get("holdings") if isinstance(hold_out, dict) else None) or []),
            cash_yuan=cash_amount,
            cache_base_dir=Path("data") / "cache",
            stock_adjust=str(getattr(args, "stock_adjust", "qfq") or "qfq"),
        )
    except (AttributeError) as exc:  # noqa: BLE001
        diag.record("build_portfolio_summary", exc, note="组合层汇总失败（已降级；rotate/add 仍可继续）")
        port = {}

    market_regime = hold_out.get("market_regime") if isinstance(hold_out, dict) else None
    market_regime = market_regime if isinstance(market_regime, dict) else {}
    regime_label = str(market_regime.get("label") or "unknown").strip().lower() or "unknown"

    # Phase2：CashSignal（账户级风险开关；默认启用，且只做“只降不升”的 max_exposure 缩放）
    cash_sig: dict[str, Any] | None = None
    cash_sig_error: str | None = None
    cash_exposure_cap_pct: float | None = None  # = 1 - cash_ratio
    tushare_pack: dict[str, Any] | None = None
    tushare_pack_error: str | None = None
    try:
        from datetime import datetime

        # as_of：优先用 holdings-user 输出的 as_of（YYYY-MM-DD）；否则用今天
        as_of_s = str((hold_out.get("as_of") if isinstance(hold_out, dict) else None) or "").strip()
        try:
            as_of_d = datetime.strptime(as_of_s, "%Y-%m-%d").date() if as_of_s else datetime.now().date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            as_of_d = datetime.now().date()

        # tushare 因子包（可选；只在启用 cash-signal 风控时才拉，避免默认浪费积分）
        if bool(use_cash_signal):
            try:
                idx_raw2 = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
                idx_first = idx_raw2.split(";", 1)[0].split(",", 1)[0].strip() or "sh000300"
                if idx_first.lower() not in {"off", "none", "0"}:
                    from ..tushare_factors import compute_tushare_factor_pack

                    tushare_pack = compute_tushare_factor_pack(
                        as_of=as_of_d,
                        context_index_symbol_prefixed=str(idx_first),
                        symbol_prefixed=None,
                        daily_amount_by_date=None,
                        cache_dir=Path("data") / "cache" / "tushare_factors",
                        ttl_hours=6.0,
                    )
            except Exception as exc:  # noqa: BLE001
                tushare_pack = None
                tushare_pack_error = str(exc)
                diag.record("tushare_factor_pack", exc, note="TuShare 因子包失败（CashSignal 已降级）")

        # CashSignal（内部会自己算 market_regime；我们用它的 cash_ratio 来限制 max_exposure）
        try:
            from ..cash_signal import CashSignalInputs, compute_cash_signal

            idx_raw3 = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip() or "sh000300"
            cash_sig = compute_cash_signal(
                inputs=CashSignalInputs(as_of=as_of_d, ref_date=as_of_d, context_index_symbol=idx_raw3.replace(",", "+")),
                tushare_factors=tushare_pack,
            )
            cr = cash_sig.get("cash_ratio") if isinstance(cash_sig, dict) else None
            try:
                cr2 = float(cr) if cr is not None else None
            except Exception:  # noqa: BLE001
                cr2 = None
            if cr2 is not None:
                cash_exposure_cap_pct = max(0.0, min(1.0, 1.0 - float(cr2)))
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            cash_sig = None
            cash_sig_error = str(exc)
            cash_exposure_cap_pct = None
            diag.record("cash_signal", exc, note="CashSignal 计算失败（已降级）")
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
        cash_sig = None
        cash_sig_error = str(exc)
        cash_exposure_cap_pct = None
        tushare_pack = None
        tushare_pack_error = None
        diag.record("cash_signal.unexpected", exc, note="CashSignal unexpected error suppressed (disabled in this run)")
        try:
            _LOG.warning("[cash-signal] unexpected error suppressed (will disable cash_signal in this run): %s", str(exc), exc_info=True)
        except (TypeError, ValueError, OverflowError, AttributeError, RuntimeError):  # noqa: BLE001
            pass

    equity_yuan = port.get("equity_yuan")
    try:
        equity_yuan = float(equity_yuan) if equity_yuan is not None else None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        equity_yuan = None

    # 兜底：如果 portfolio_summary 没算出来 equity，就用 holdings 的市值拼一个（至少别把流程炸了）
    if equity_yuan is None or equity_yuan <= 0:
        try:
            mv = 0.0
            for it in (hold_out.get("holdings") or []):
                if not isinstance(it, dict) or not bool(it.get("ok")):
                    continue
                try:
                    mv += float(it.get("market_value") or 0.0)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    continue
            eq2 = float(cash_amount or 0.0) + float(mv)
            if eq2 > 0:
                equity_yuan = float(eq2)
        except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
            try:
                _LOG.warning("[portfolio] fallback equity_yuan from holdings failed (will keep equity_yuan=None): %s", str(exc), exc_info=True)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                pass

    # 目标仓位口径：
    # - 你的要求：按总权益(equity)算目标仓位；add 模式下“下单预算”仍只用 cash.amount（新钱慢慢补仓）
    # - rotate 模式：本来就按权益做再平衡（会卖出腾挪）
    cap_override = getattr(args, "capital_yuan", None)
    capital_yuan = None
    if cap_override is not None:
        try:
            capital_yuan = float(cap_override)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            capital_yuan = None
    if capital_yuan is None:
        capital_yuan = float(equity_yuan) if equity_yuan is not None else None
    if capital_yuan is None:
        capital_yuan = float(cash_amount) if cash_amount is not None else None

    if capital_yuan is None or capital_yuan <= 0:
        raise SystemExit(f"无法确定 capital_yuan：mode={mode} cash.amount={cash_amount} equity_yuan={equity_yuan}（可用 --capital-yuan 覆盖）")

    # signals：输入支持两类（KISS，不折腾数据库）
    # 1) 旧口径：scan-etf 的 top_bbb.json
    # 2) 统一口径：signals.json（schema_version=1；来源可为 scan-etf/scan-strategy/signals-merge 等）
    sig_items: list[dict[str, Any]] = []
    bbb_cfg: dict[str, Any] = {}

    is_signals_schema = bool(isinstance(raw_sig, dict) and int(raw_sig.get("schema_version") or 0) == 1 and raw_sig.get("strategy") and raw_sig.get("items"))
    if is_signals_schema:
        cfg0 = raw_sig.get("config")
        bbb_cfg = cfg0 if isinstance(cfg0, dict) else {}
        items0 = raw_sig.get("items")
        items0 = items0 if isinstance(items0, list) else []
        for s in items0:
            if not isinstance(s, dict):
                continue
            if str(s.get("asset") or "").strip().lower() != "etf":
                continue
            act = str(s.get("action") or "").strip().lower()
            # 组合层只吃 entry（别拿 watch/avoid/exit 去凑仓位）
            if act != "entry":
                continue
            meta = s.get("meta") if isinstance(s.get("meta"), dict) else {}
            entry = s.get("entry") if isinstance(s.get("entry"), dict) else {}
            sig_items.append(
                {
                    "symbol": s.get("symbol"),
                    "name": s.get("name"),
                    "close": (meta.get("close") if isinstance(meta, dict) else None) or entry.get("price_ref"),
                    "levels": meta.get("levels") if isinstance(meta, dict) else None,
                    "exit": meta.get("exit") if isinstance(meta, dict) else None,
                    "bbb": meta.get("bbb") if isinstance(meta, dict) else None,
                    "bbb_forward": meta.get("bbb_forward") if isinstance(meta, dict) else None,
                }
            )
    else:
        items0 = raw_sig.get("items") if isinstance(raw_sig, dict) else None
        sig_items = items0 if isinstance(items0, list) else []
        cfg0 = raw_sig.get("bbb") if isinstance(raw_sig, dict) else None
        bbb_cfg = cfg0 if isinstance(cfg0, dict) else {}

    # 非 BBB 信号（尤其 scan-strategy）通常不会带 levels/exit；
    # 但仓位计划需要它们来算 stop（entry_ma / MA20 / ATR）。这里补一份“最小可用口径”，让组合层能跑通。
    try:
        cache_ttl_hours_lv = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
        cache_dir_etf = Path("data") / "cache" / "etf"

        def _fnum(x) -> float | None:
            try:
                v = None if x is None else float(x)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return None
            if v is None:
                return None
            if v != v:  # NaN
                return None
            return float(v)

        for it in sig_items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue

            lv = it.get("levels") if isinstance(it.get("levels"), dict) else {}
            ex = it.get("exit") if isinstance(it.get("exit"), dict) else {}
            ex_d = ex.get("daily") if isinstance(ex.get("daily"), dict) else {}

            need_ma50 = (lv.get("ma50") is None) and (lv.get("bbb_ma_entry") is None)
            need_atr = lv.get("atr") is None
            need_ma20 = ex_d.get("ma20") is None
            if not (need_ma50 or need_atr or need_ma20):
                continue

            try:
                sym2 = resolve_symbol("etf", sym)
            except Exception:  # noqa: BLE001
                sym2 = sym

            df_d = fetch_daily_cached(
                FetchParams(asset="etf", symbol=str(sym2), adjust="qfq"),
                cache_dir=cache_dir_etf,
                ttl_hours=float(cache_ttl_hours_lv),
            )
            if df_d is None or getattr(df_d, "empty", True):
                continue

            try:
                import pandas as pd

                dfd = df_d.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            except Exception:  # noqa: BLE001
                dfd = df_d

            if need_ma20:
                try:
                    # 不依赖 ta.SMAIndicator：某些脏缓存/短样本会触发第三方库的奇怪异常；
                    # 这里用 pandas rolling 直接算，口径更可控（不足 window 就留 NaN）。
                    dfd2 = dfd.copy()
                    c = dfd2["close"].astype(float)
                    dfd2["ma20"] = c.rolling(window=20, min_periods=20).mean()
                    last_d = dfd2.iloc[-1]
                    ma20_v = _fnum(last_d.get("ma20"))
                    if ma20_v is not None and ma20_v > 0:
                        ex_d["ma20"] = float(ma20_v)
                except Exception:  # noqa: BLE001
                    pass

            if need_ma50 or need_atr:
                try:
                    dfw = resample_to_weekly(dfd)
                    if dfw is None or getattr(dfw, "empty", True):
                        raise RuntimeError("weekly 为空")

                    dfw2 = dfw.copy()
                    c2 = dfw2["close"].astype(float)
                    # 周线 MA50：不足 50 周就留 NaN（别用假均线骗人）
                    dfw2["ma50"] = c2.rolling(window=50, min_periods=50).mean()
                    # 周线 ATR：用统一函数（内部已做短样本/IndexError 兜底）
                    dfw2 = add_atr(dfw2, period=14, out_col="atr")
                    last_w = dfw2.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True).iloc[-1]

                    ma50_v = _fnum(last_w.get("ma50"))
                    if ma50_v is not None and ma50_v > 0:
                        if lv.get("ma50") is None:
                            lv["ma50"] = float(ma50_v)
                        if lv.get("bbb_ma_entry") is None:
                            # 对非 BBB 策略：entry_ma 兜底用 MA50（不造假，不瞎编“神秘止损线”）
                            lv["bbb_ma_entry"] = float(ma50_v)

                    atr_v = _fnum(last_w.get("atr"))
                    if atr_v is not None and atr_v > 0 and lv.get("atr") is None:
                        lv["atr"] = float(atr_v)
                except Exception:  # noqa: BLE001
                    pass

            if lv:
                it["levels"] = lv
            if ex_d:
                ex["daily"] = ex_d
                it["exit"] = ex
    except Exception as exc:  # noqa: BLE001
        diag.record("signals.ensure_levels_exit", exc, note="补全 levels/exit 失败（可能影响仓位计划/止损口径）")

    if not sig_items:
        raise SystemExit(f"signals.items 为空：{signals_path}")

    # 冻仓标的：从候选里剔除（否则 plan 会把仓位预算浪费在“你不让动”的票上）
    frozen_in_signals: list[str] = []
    if frozen_syms:
        kept: list[dict[str, Any]] = []
        for it in sig_items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").strip()
            if sym and sym in frozen_syms:
                frozen_in_signals.append(sym)
                continue
            kept.append(it)
        sig_items = kept

    # Phase2：OpportunityScore 过滤（只影响候选质量；默认 0=不过滤）
    filtered_by_min_score_sig = 0
    if min_score > 0:
        kept: list[dict[str, Any]] = []
        cache_ttl_hours2 = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
        cache_dir2 = Path("data") / "cache" / "etf"

        for it in sig_items:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue

            sc = None
            try:
                # 取日线 -> 转周线（BBB/仓位计划口径以周线为主）
                df_d = fetch_daily_cached(
                    FetchParams(asset="etf", symbol=resolve_symbol("etf", sym), adjust="qfq"),
                    cache_dir=cache_dir2,
                    ttl_hours=float(cache_ttl_hours2),
                )
                if df_d is None or getattr(df_d, "empty", True):
                    raise RuntimeError("无日线数据")

                from ..resample import resample_to_weekly
                from ..indicators import add_moving_averages
                from ..factors.game_theory import LiquidityTrapFactor
                from ..opportunity_score import OpportunityScoreInputs, compute_opportunity_score

                dfw = resample_to_weekly(df_d)
                if dfw is None or getattr(dfw, "empty", True):
                    raise RuntimeError("转周K失败/为空")

                # 给 key_level 用：ma50（兜底 close）
                dfw = add_moving_averages(dfw, ma_fast=50, ma_slow=200)

                # trap_risk：liquidity_trap.score（0~1）
                trap = None
                try:
                    r_trap = LiquidityTrapFactor().compute(dfw)
                    trap = None if r_trap.score is None else float(r_trap.score)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    trap = None

                # as_of：用最后一根周线日期（避免周末/节假日乱跳）
                try:
                    as_of_dt = dfw["date"].max()
                    as_of_d = as_of_dt.date() if hasattr(as_of_dt, "date") else None
                except (AttributeError):  # noqa: BLE001
                    as_of_d = None
                if as_of_d is None:
                    from datetime import datetime

                    as_of_d = datetime.now().date()

                kl_name = "ma50"
                kl_value = None
                try:
                    kl_value = dfw.iloc[-1].get("ma50")
                except (KeyError, IndexError, AttributeError):  # noqa: BLE001
                    kl_value = None
                if kl_value is None:
                    kl_name = "close"
                    try:
                        kl_value = dfw.iloc[-1].get("close")
                    except (KeyError, IndexError, AttributeError):  # noqa: BLE001
                        kl_value = None

                opp = compute_opportunity_score(
                    df=dfw,
                    inputs=OpportunityScoreInputs(
                        symbol=str(sym),
                        asset="etf",
                        as_of=as_of_d,
                        ref_date=as_of_d,
                        min_score=0.70,
                        t_plus_one=True,
                        trap_risk=trap,
                        fund_flow=None,
                        expected_holding_days=10,
                    ),
                    key_level_name=str(kl_name),
                    key_level_value=(None if kl_value is None else float(kl_value)),
                )
                if isinstance(opp, dict) and opp.get("total_score") is not None:
                    sc = float(opp.get("total_score"))
                    it["opp_score"] = float(sc)
                    it["trap_risk"] = trap
                    it["opp_bucket"] = str(opp.get("bucket") or "").strip() or None
            except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
                it["opp_score"] = None
                it["opp_error"] = str(exc)

            if sc is not None and float(sc) >= float(min_score):
                kept.append(it)
            else:
                filtered_by_min_score_sig += 1

        sig_items = kept

    rt_default = float((bbb_cfg or {}).get("roundtrip_cost_yuan") or 10.0)
    rt = float(getattr(args, "roundtrip_cost_yuan", None) or rt_default)

    # 成本/约束参数：优先 CLI 覆盖；否则继承 signals.config（或留空=0）
    min_fee_default = float((bbb_cfg or {}).get("min_fee_yuan") or 0.0)
    min_fee_yuan = float(getattr(args, "min_fee_yuan", None) or min_fee_default)

    buy_cost_default = float((bbb_cfg or {}).get("buy_cost") or 0.0)
    sell_cost_default = float((bbb_cfg or {}).get("sell_cost") or 0.0)
    buy_cost = float(getattr(args, "buy_cost", None) or buy_cost_default)
    sell_cost = float(getattr(args, "sell_cost", None) or sell_cost_default)

    slip_mode = str(getattr(args, "slippage_mode", None) or (bbb_cfg or {}).get("slippage_mode") or "none").strip().lower() or "none"
    slip_bps = float(getattr(args, "slippage_bps", None) or (bbb_cfg or {}).get("slippage_bps") or 0.0)
    slip_ref_amt = float(getattr(args, "slippage_ref_amount_yuan", None) or (bbb_cfg or {}).get("slippage_ref_amount_yuan") or 1e8)
    slip_bps_min = float(getattr(args, "slippage_bps_min", None) or (bbb_cfg or {}).get("slippage_bps_min") or 0.0)
    slip_bps_max = float(getattr(args, "slippage_bps_max", None) or (bbb_cfg or {}).get("slippage_bps_max") or 30.0)
    slip_unknown_bps = float(getattr(args, "slippage_unknown_bps", None) or (bbb_cfg or {}).get("slippage_unknown_bps") or 10.0)
    slip_vm = float(getattr(args, "slippage_vol_mult", None) or (bbb_cfg or {}).get("slippage_vol_mult") or 0.0)

    stop_mode = getattr(args, "stop_mode", None)
    stop_mode2 = None if not stop_mode else str(stop_mode).strip()
    if stop_mode2 in {"weekly_entry_ma", "daily_ma20", "atr"}:
        stop_mode3 = stop_mode2  # type: ignore[assignment]
    else:
        stop_mode3 = None

    # 组合构建：vol targeting（KISS：用大盘指数的已知波动率做“只降不升”的仓位缩放）
    vol_target = float(getattr(args, "vol_target", 0.0) or 0.0)
    vol_target = max(0.0, min(vol_target, 5.0))
    vol_lookback_days = int(getattr(args, "vol_lookback_days", 20) or 20)
    vol_lookback_days = max(5, min(vol_lookback_days, 252))
    vol_index_symbol = None
    vol_realized_ann = None
    vol_scale = None
    vol_error = None

    max_exposure_eff = None
    if vol_target > 0:
        cache_ttl_hours2 = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
        idx_raw = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
        idx_first = idx_raw.split(",", 1)[0].strip()
        if idx_first and idx_first.lower() not in {"off", "none", "0"}:
            vol_index_symbol = idx_first
            try:
                import math
                import pandas as pd

                df_idx = fetch_daily_cached(
                    FetchParams(asset="index", symbol=str(idx_first)),
                    cache_dir=Path("data") / "cache" / "index",
                    ttl_hours=float(cache_ttl_hours2),
                )
                if df_idx is not None and (not getattr(df_idx, "empty", True)) and "close" in df_idx.columns:
                    dfi = df_idx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                    close = pd.to_numeric(dfi["close"], errors="coerce").astype(float)
                    rets = close.pct_change().replace([math.inf, -math.inf], float("nan")).dropna().tail(int(vol_lookback_days))
                    if not rets.empty:
                        v = float(rets.std()) * float(math.sqrt(252.0))
                        if math.isfinite(v) and v > 0:
                            vol_realized_ann = float(v)
                            # 只降不升：波动高于目标才降仓；低波不加仓（别在“平静”里上头）
                            vol_scale = min(1.0, float(vol_target) / float(vol_realized_ann))
                            base_me = (
                                float(getattr(args, "max_exposure_pct"))
                                if getattr(args, "max_exposure_pct", None) is not None
                                else float(risk_profile_for_regime(regime_label).max_exposure_pct)
                            )
                            max_exposure_eff = float(base_me) * float(vol_scale)
            except (TypeError, ValueError, OverflowError, AttributeError) as exc:  # noqa: BLE001
                vol_error = str(exc)

    # CashSignal：用 cash_ratio 把 max_exposure 做“只降不升”的封顶（默认启用；可 --no-cash-signal 关闭）
    cash_exposure_cap_pct_eff = None
    if bool(use_cash_signal) and cash_exposure_cap_pct is not None:
        try:
            cash_exposure_cap_pct_eff = max(0.0, min(float(cash_exposure_cap_pct), 1.0))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cash_exposure_cap_pct_eff = None

    if cash_exposure_cap_pct_eff is not None:
        if max_exposure_eff is None:
            # 如果用户手动传了 --max-exposure-pct，就先取 min(用户上限, cash_signal 上限)
            me_user = None
            if getattr(args, "max_exposure_pct", None) is not None:
                try:
                    me_user = float(getattr(args, "max_exposure_pct"))
                except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                    me_user = None
            if me_user is not None:
                me_user = max(0.0, min(float(me_user), 1.0))
                max_exposure_eff = float(min(float(me_user), float(cash_exposure_cap_pct_eff)))
            else:
                max_exposure_eff = float(cash_exposure_cap_pct_eff)
        else:
            max_exposure_eff = float(min(float(max_exposure_eff), float(cash_exposure_cap_pct_eff)))

    # max_corr：别名覆盖 diversify_max_corr（兼容旧参数，减少你记命令的心智负担）
    max_corr_override = getattr(args, "max_corr", None)
    div_max_corr = float(getattr(args, "diversify_max_corr", 0.95) or 0.95)
    if max_corr_override is not None:
        try:
            div_max_corr = float(max_corr_override)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    pp = PositionPlanParams(
        capital_yuan=float(capital_yuan),
        roundtrip_cost_yuan=float(rt),
        lot_size=int(getattr(args, "lot_size", 100) or 100),
        max_cost_pct=float(getattr(args, "max_cost_pct", 0.02) or 0.02),
        risk_min_yuan=float(getattr(args, "risk_min_yuan")) if getattr(args, "risk_min_yuan", None) is not None else None,
        risk_per_trade_yuan=float(getattr(args, "risk_per_trade_yuan")) if getattr(args, "risk_per_trade_yuan", None) is not None else None,
        max_exposure_pct=float(max_exposure_eff)
        if max_exposure_eff is not None
        else (float(getattr(args, "max_exposure_pct")) if getattr(args, "max_exposure_pct", None) is not None else None),
        risk_per_trade_pct=float(getattr(args, "risk_per_trade_pct")) if getattr(args, "risk_per_trade_pct", None) is not None else None,
        stop_mode=stop_mode3,
        max_positions=int(max_positions_eff) if max_positions_eff is not None else None,
        max_position_pct=float(max_position_pct_eff) if max_position_pct_eff is not None else None,
        returns_cache_dir=str(getattr(args, "returns_cache_dir")) if getattr(args, "returns_cache_dir", None) else None,
        diversify=bool(getattr(args, "diversify", True)),
        diversify_window_weeks=int(getattr(args, "diversify_window_weeks", 104) or 104),
        diversify_min_overlap_weeks=int(getattr(args, "diversify_min_overlap_weeks", 26) or 26),
        diversify_max_corr=float(div_max_corr),
        max_per_theme=int(getattr(args, "max_per_theme", 0) or 0),
        atr_mult=float(getattr(args, "atr_mult", 2.0) or 2.0),
    )

    plan = build_etf_position_plan(items=sig_items, market_regime_label=str(regime_label), params=pp)

    # --- rebalance: 目标仓位 -> 买卖份额 ---
    warnings: list[str] = []
    if frozen_syms:
        warnings.append(f"冻仓标的：{','.join(sorted(frozen_syms))}（不自动补仓/不自动调仓；你手动下单不受影响）")
    if grid_exempt_syms:
        warnings.append(f"网格/组合外标的：{','.join(sorted(grid_exempt_syms))}（rebalance 不自动清仓/不主动轮动；按你的网格纪律手动执行）")
    if frozen_in_signals:
        warnings.append(f"signals 候选里包含冻仓标的，已剔除：{','.join(sorted(set(frozen_in_signals)))}")
    if min_score > 0 and int(filtered_by_min_score_sig) > 0:
        warnings.append(f"OpportunityScore 过滤：min_score={min_score:g}，剔除候选={int(filtered_by_min_score_sig)}")
    if min_score > 0 and (not sig_items):
        warnings.append(f"OpportunityScore 过滤后候选为空：min_score={min_score:g}（rotate 保护会阻止一键清仓）")
    if bool(use_cash_signal):
        if cash_exposure_cap_pct_eff is not None:
            crx = None
            try:
                crx = cash_sig.get("cash_ratio") if isinstance(cash_sig, dict) else None
            except (AttributeError):  # noqa: BLE001
                crx = None
            warnings.append(f"CashSignal 风控：cash_ratio≈{crx} => max_exposure_cap≈{float(cash_exposure_cap_pct_eff):.2f}（只降不升）")
        elif cash_sig_error:
            warnings.append(f"CashSignal 计算失败（降级不影响主流程）：{cash_sig_error}")
        elif tushare_pack_error:
            warnings.append(f"TuShare 因子包失败（已降级）：{tushare_pack_error}")
    lot = max(1, int(getattr(args, "lot_size", 100) or 100))
    tb_cfg = TradeabilityConfig(
        limit_up_pct=float(getattr(args, "limit_up_pct", 0.0) or 0.0),
        limit_down_pct=float(getattr(args, "limit_down_pct", 0.0) or 0.0),
        halt_vol_zero=bool(getattr(args, "halt_vol_zero", True)),
    )
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
    daily_df_cache: dict[str, Any] = {}

    def _df_daily(sym: str):
        s = str(sym or "").strip()
        if not s:
            return None
        if s in daily_df_cache:
            return daily_df_cache[s]
        try:
            sym2 = resolve_symbol("etf", s)
        except Exception:  # noqa: BLE001
            sym2 = s
        try:
            df = fetch_daily_cached(
                FetchParams(asset="etf", symbol=str(sym2), adjust="qfq"),
                cache_dir=Path("data") / "cache" / "etf",
                ttl_hours=float(cache_ttl_hours),
            )
        except Exception:  # noqa: BLE001
            df = None
        daily_df_cache[s] = df
        return df

    def _tradeability_last_bar(sym: str) -> dict[str, Any] | None:
        df = _df_daily(sym)
        if df is None or getattr(df, "empty", True):
            return None
        try:
            import pandas as pd

            dfd = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if dfd.empty:
                return None
            last = dfd.iloc[-1]
            prev = dfd.iloc[-2] if len(dfd) >= 2 else None
            prev_close = None if prev is None else prev.get("close")
            close = last.get("close")
            op = last.get("open") if "open" in dfd.columns else close
            hp = last.get("high") if "high" in dfd.columns else close
            lp = last.get("low") if "low" in dfd.columns else close
            vol = last.get("volume") if "volume" in dfd.columns else None
            amt = last.get("amount") if "amount" in dfd.columns else None
            if amt is None and vol is not None and close is not None:
                try:
                    amt = float(close) * float(vol)
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    amt = None

            dt = last.get("date")
            dt_s = str(dt.date()) if hasattr(dt, "date") else str(dt)

            flags = tradeability_flags(
                open_price=(None if op is None else float(op)),
                high_price=(None if hp is None else float(hp)),
                low_price=(None if lp is None else float(lp)),
                prev_close=(None if prev_close is None else float(prev_close)),
                volume=(None if vol is None else float(vol)),
                amount=(None if amt is None else float(amt)),
                cfg=tb_cfg,
            )
            return {"ref_date": dt_s, "flags": flags}
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return None

    # 当前持仓 map（只管 ETF；股票先当“手动账户”处理）
    positions_mv_yuan = 0.0
    cur_shares: dict[str, int] = {}
    cur_close: dict[str, float] = {}
    for it in (hold_out.get("holdings") or []):
        if not isinstance(it, dict) or not bool(it.get("ok")):
            continue
        try:
            mv = float(it.get("market_value") or 0.0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            mv = 0.0
        if mv > 0:
            positions_mv_yuan += float(mv)
        if str(it.get("asset") or "").strip().lower() != "etf":
            continue
        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue
        try:
            sh = int(it.get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            sh = 0
        if sh <= 0:
            continue
        cur_shares[sym] = int(sh)
        try:
            c = float(it.get("close")) if it.get("close") is not None else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            c = None
        if c is not None and c > 0:
            cur_close[sym] = float(c)

    # 目标仓位 map（由 plan.plans 决定；按 entry close 估算）
    targets: list[dict[str, Any]] = []
    target_shares: dict[str, int] = {}
    plan_items = plan.get("plans") if isinstance(plan, dict) else None
    plan_items = plan_items if isinstance(plan_items, list) else []
    for p in plan_items:
        if not isinstance(p, dict) or not bool(p.get("ok")):
            continue
        sym = str(p.get("symbol") or "").strip()
        if not sym:
            continue
        try:
            tsh = int(p.get("shares") or 0)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            tsh = 0
        if tsh <= 0:
            continue
        target_shares[sym] = int(tsh)
        cur = int(cur_shares.get(sym, 0))
        delta = int(tsh - cur)
        targets.append(
            {
                "symbol": sym,
                "name": str(p.get("name") or "").strip(),
                "entry": p.get("entry"),
                "stop": p.get("stop"),
                "stop_ref": p.get("stop_ref"),
                "stop_mode": p.get("stop_mode"),
                "target_shares": int(tsh),
                "current_shares": int(cur),
                "delta_shares": int(delta),
                "position_yuan": p.get("position_yuan"),
                "risk_yuan_actual": p.get("risk_yuan_actual"),
            }
        )

    if mode == "add":
        # add 模式：新钱优先补已有仓（current_shares>0），再开新票；组内保持 plan 原顺序（sort 稳定）
        targets.sort(key=lambda x: (0 if int(x.get("current_shares") or 0) > 0 else 1))

    orders: list[dict[str, Any]] = []
    cash_avail = float(cash_amount or 0.0)
    cash_in_est = 0.0
    cash_out_est = 0.0
    max_turnover_pct = float(getattr(args, "max_turnover_pct", 0.0) or 0.0)
    max_turnover_pct = max(0.0, min(max_turnover_pct, 5.0))
    buy_turnover_budget_yuan = None
    buy_turnover_used_yuan = 0.0
    try:
        base_eq = float(equity_yuan) if equity_yuan is not None else float(capital_yuan)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        base_eq = float(capital_yuan)
    if max_turnover_pct > 0 and base_eq > 0:
        buy_turnover_budget_yuan = float(base_eq) * float(max_turnover_pct)

    # 仓位约束：add 模式也要守住“最大总仓位比例”，否则你有存量仓时会被建议把现金打光。
    # rotate 会卖出腾挪到目标；add 不卖，所以这里必须额外卡一下“新增买入的上限”。
    exposure_buy_budget_yuan = None
    exposure_buy_used_yuan = 0.0
    exposure_max_positions_yuan = None
    max_exposure_pct_eff2 = None
    if mode == "add" and base_eq > 0:
        prof0 = risk_profile_for_regime(regime_label)
        me0 = None
        if max_exposure_eff is not None:
            me0 = float(max_exposure_eff)
        elif getattr(args, "max_exposure_pct", None) is not None:
            try:
                me0 = float(getattr(args, "max_exposure_pct"))
            except (TypeError, ValueError, OverflowError):  # noqa: BLE001
                me0 = None
        if me0 is None:
            me0 = float(getattr(prof0, "max_exposure_pct", 0.0) or 0.0)
        me0 = max(0.0, min(float(me0), 1.0))
        max_exposure_pct_eff2 = float(me0)
        exposure_max_positions_yuan = float(base_eq) * float(me0)
        exposure_buy_budget_yuan = float(exposure_max_positions_yuan) - float(positions_mv_yuan)
        if exposure_buy_budget_yuan < 0:
            exposure_buy_budget_yuan = 0.0
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 6.0) or 6.0)
    cache_dir = Path("data") / "cache" / "etf"

    # 估算用成本模型（统一口径，尽量别让“现金估算”瞎飘）
    cost_base = trade_cost_from_params(roundtrip_cost_yuan=float(rt), min_fee_yuan=float(min_fee_yuan), buy_cost=float(buy_cost), sell_cost=float(sell_cost))

    slip_cache: dict[str, dict[str, Any]] = {}

    def _slip_for(sym: str) -> dict[str, Any]:
        sym2 = str(sym)
        if sym2 in slip_cache:
            return slip_cache[sym2]
        if str(slip_mode) in {"", "none", "off", "0", "false"}:
            out = {"slippage_mode": "none", "slippage_bps": 0.0, "slippage_rate": 0.0, "amount_avg20_yuan": None}
            slip_cache[sym2] = out
            return out

        amt_avg20 = None
        try:
            df = fetch_daily_cached(
                FetchParams(asset="etf", symbol=sym2),
                cache_dir=cache_dir,
                ttl_hours=float(cache_ttl_hours),
            )
            if df is not None and (not getattr(df, "empty", True)):
                df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
                if not df.empty:
                    import pandas as pd

                    close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
                    amount_s = None
                    if "amount" in df.columns:
                        amount_s = pd.to_numeric(df["amount"], errors="coerce").astype(float)
                    elif "volume" in df.columns:
                        vol_s = pd.to_numeric(df["volume"], errors="coerce").astype(float)
                        amount_s = close_s * vol_s
                    if amount_s is not None:
                        v = amount_s.rolling(window=20, min_periods=20).mean().iloc[-1]
                        amt_avg20 = None if v is None else float(v)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            amt_avg20 = None

        slip_bps2 = estimate_slippage_bps(
            mode=str(slip_mode),
            amount_avg20_yuan=(float(amt_avg20) if amt_avg20 is not None else None),
            atr_pct=None,
            bps=float(slip_bps),
            ref_amount_yuan=float(slip_ref_amt),
            min_bps=float(slip_bps_min),
            max_bps=float(slip_bps_max),
            unknown_bps=float(slip_unknown_bps),
            vol_mult=float(slip_vm),
        )
        out = {
            "slippage_mode": str(slip_mode),
            "slippage_bps": float(slip_bps2),
            "slippage_rate": float(bps_to_rate(float(slip_bps2))),
            "amount_avg20_yuan": (float(amt_avg20) if amt_avg20 is not None else None),
        }
        slip_cache[sym2] = out
        return out

    def _cost_for(sym: str) -> tuple[TradeCost, dict[str, Any]]:
        slip = _slip_for(sym)
        r = float(slip.get("slippage_rate") or 0.0)
        c = TradeCost(
            buy_cost=float(cost_base.buy_cost) + float(r),
            sell_cost=float(cost_base.sell_cost) + float(r),
            buy_fee_yuan=float(cost_base.buy_fee_yuan),
            sell_fee_yuan=float(cost_base.sell_fee_yuan),
            buy_fee_min_yuan=float(cost_base.buy_fee_min_yuan),
            sell_fee_min_yuan=float(cost_base.sell_fee_min_yuan),
        )
        return c, slip

    def _floor_to_lot(shares: int) -> int:
        n = int(shares)
        if n <= 0:
            return 0
        return (n // int(lot)) * int(lot)

    def _min_shares_for_notional(*, notional_yuan: float, price: float) -> int:
        # 向上取整到一手：保证成交额>=门槛（否则小单被最低佣金磨死）
        try:
            import math

            n = float(notional_yuan)
            px = float(price)
        except (TypeError, ValueError, OverflowError):  # noqa: BLE001
            return 0
        if n <= 0 or px <= 0:
            return 0
        raw = int(math.ceil(n / px))
        if raw <= 0:
            return 0
        lot2 = max(1, int(lot))
        return int(((raw + lot2 - 1) // lot2) * lot2)

    def _append_sell(*, sym: str, sh: int, price_ref: float | None, reason: str) -> float:
        nonlocal cash_in_est
        px = float(price_ref) if price_ref is not None else None
        if px is None or px <= 0 or int(sh) <= 0:
            return 0.0

        cost2, slip2 = _cost_for(sym)
        tb = _tradeability_last_bar(sym)
        flags = tb.get("flags") if isinstance(tb, dict) else {}
        notional = float(sh) * float(px)
        cash_in, fee = cash_sell(shares=int(sh), price=float(px), cost=cost2)
        cash_in_est += float(cash_in)
        orders.append(
            {
                "side": "sell",
                "asset": "etf",
                "symbol": sym,
                "shares": int(sh),
                "lot_size": int(lot),
                "signal_date": str((market_regime or {}).get("last_date") or (market_regime or {}).get("date") or ""),
                "exec": "next_open",
                "price_ref": float(px) if px is not None else None,
                "price_ref_type": "close",
                "order_type": "market",
                "limit_price": None,
                "est_notional_yuan": float(notional),
                "est_cash": float(cash_in),
                "est_fee_yuan": float(fee),
                "min_notional_for_min_fee_yuan": min_notional_for_min_fee(cost_rate=float(sell_cost), min_fee_yuan=float(min_fee_yuan)),
                "min_trade_notional_yuan": min_notional_for_min_fee(cost_rate=float(sell_cost), min_fee_yuan=float(min_fee_yuan)),
                "slippage": slip2,
                "tradeability_last_bar": tb,
                "halt_risk": bool(flags.get("halted")) if isinstance(flags, dict) else None,
                "limit_down_risk": bool(flags.get("locked_limit_down")) if isinstance(flags, dict) else None,
                "reason": str(reason),
            }
        )
        return float(cash_in)

    def _append_buy(*, sym: str, sh: int, price_ref: float | None, reason: str) -> float:
        nonlocal cash_out_est
        px = float(price_ref) if price_ref is not None else None
        if px is None or px <= 0 or int(sh) <= 0:
            return 0.0

        cost2, slip2 = _cost_for(sym)
        tb = _tradeability_last_bar(sym)
        flags = tb.get("flags") if isinstance(tb, dict) else {}
        notional = float(sh) * float(px)
        cash_out, fee = cash_buy(shares=int(sh), price=float(px), cost=cost2)
        cash_out_est += float(cash_out)
        orders.append(
            {
                "side": "buy",
                "asset": "etf",
                "symbol": sym,
                "shares": int(sh),
                "lot_size": int(lot),
                "signal_date": str((market_regime or {}).get("last_date") or (market_regime or {}).get("date") or ""),
                "exec": "next_open",
                "price_ref": float(px) if px is not None else None,
                "price_ref_type": "close",
                "order_type": "market",
                "limit_price": None,
                "est_notional_yuan": float(notional),
                "est_cash": float(cash_out),
                "est_fee_yuan": float(fee),
                "min_notional_for_min_fee_yuan": min_notional_for_min_fee(cost_rate=float(buy_cost), min_fee_yuan=float(min_fee_yuan)),
                "min_trade_notional_yuan": min_notional_for_min_fee(cost_rate=float(buy_cost), min_fee_yuan=float(min_fee_yuan)),
                "slippage": slip2,
                "tradeability_last_bar": tb,
                "halt_risk": bool(flags.get("halted")) if isinstance(flags, dict) else None,
                "limit_up_risk": bool(flags.get("locked_limit_up")) if isinstance(flags, dict) else None,
                "reason": str(reason),
            }
        )
        return float(cash_out)

    if mode == "rotate":
        # rotate 的前提是“目标组合非空”。否则数据源抽风/信号为空时，别把账户一键清仓了。
        if not target_shares:
            warnings.append("rotate 保护：目标组合为空（signals过滤/条件失败/数据缺失），本次不执行任何卖出/买入")
        else:
            # 1) 卖出：先清掉“非目标”，再处理“超配减仓”
            for sym, sh in sorted(cur_shares.items()):
                if sym in frozen_syms or sym in grid_exempt_syms:
                    continue
                if sym not in target_shares:
                    _append_sell(sym=sym, sh=int(sh), price_ref=cur_close.get(sym), reason="rotate: 非目标清仓")
            for t in targets:
                sym = str(t.get("symbol") or "")
                if sym in frozen_syms or sym in grid_exempt_syms:
                    continue
                delta = int(t.get("delta_shares") or 0)
                if delta < 0:
                    _append_sell(sym=sym, sh=int(_floor_to_lot(-delta)), price_ref=cur_close.get(sym), reason="rotate: 减仓到目标")

            cash_avail = float(cash_amount or 0.0) + float(cash_in_est)

            # 2) 买入：按 plan 顺序加仓到目标（现金不够就截断）
            for t in targets:
                sym = str(t.get("symbol") or "")
                if sym in frozen_syms or sym in grid_exempt_syms:
                    continue
                delta = int(t.get("delta_shares") or 0)
                if delta <= 0:
                    continue
                try:
                    entry = float(t.get("entry")) if t.get("entry") is not None else None
                except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                    entry = None
                if entry is None or entry <= 0:
                    warnings.append(f"{sym} 缺 entry，跳过买入")
                    continue
                want = _floor_to_lot(int(delta))
                if want <= 0:
                    continue
                cost2, _ = _cost_for(sym)
                affordable = int(calc_shares_for_capital(capital_yuan=float(cash_avail), price=float(entry), cost=cost2, lot_size=int(lot)))
                buy_sh = int(min(want, affordable))
                if buy_sh <= 0:
                    warnings.append(f"现金不足：{sym} 目标加仓{want}，但可买=0（rotate 也买不起）")
                    continue
                # 换手约束（KISS：当前只限制 buy 侧）
                if buy_turnover_budget_yuan is not None:
                    rem = float(buy_turnover_budget_yuan) - float(buy_turnover_used_yuan)
                    max_by_turn = _floor_to_lot(int(rem / float(entry))) if rem > 0 else 0
                    if max_by_turn <= 0:
                        warnings.append(
                            f"换手约束触发：buy_turnover_used≈{buy_turnover_used_yuan:.0f} >= budget≈{buy_turnover_budget_yuan:.0f}（停止后续买入）"
                        )
                        break
                    buy_sh = int(min(int(buy_sh), int(max_by_turn)))
                # rotate：同样守“单笔金额门槛”（否则卖完再买一丢丢，纯磨损）
                if float(min_trade_notional_yuan or 0.0) > 0 and float(entry) > 0:
                    min_sh = _min_shares_for_notional(notional_yuan=float(min_trade_notional_yuan), price=float(entry))
                    if min_sh > 0 and int(buy_sh) < int(min_sh):
                        est_notional = float(buy_sh) * float(entry)
                        warnings.append(
                            f"单笔金额门槛：{sym} 本次可买{buy_sh}份≈{est_notional:.0f} < {min_trade_notional_yuan:.0f}（需要≥{min_sh}份），跳过"
                        )
                        continue
                cash_need = float(_append_buy(sym=sym, sh=int(buy_sh), price_ref=entry, reason="rotate: 加仓到目标"))
                cash_avail -= float(cash_need)
                buy_turnover_used_yuan += float(buy_sh) * float(entry)
    else:
        # add：只买不卖，现金不够就少买/不买
        if cash_amount is None:
            warnings.append("cash.amount 为空：add 模式无法做现金预算（请在 data/user_holdings.json 回填 cash.amount）")
        cash_avail = float(cash_amount or 0.0)
        for t in targets:
            sym = str(t.get("symbol") or "")
            if sym in frozen_syms or sym in grid_exempt_syms:
                continue
            delta = int(t.get("delta_shares") or 0)
            if delta <= 0:
                if delta < 0:
                    warnings.append(f"{sym} 当前持仓>{t.get('target_shares')}（add 模式不卖，忽略减仓）")
                continue
            try:
                entry = float(t.get("entry")) if t.get("entry") is not None else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                entry = None
            if entry is None or entry <= 0:
                warnings.append(f"{sym} 缺 entry，跳过买入")
                continue
            want = _floor_to_lot(int(delta))
            if want <= 0:
                continue
            cost2, _ = _cost_for(sym)
            affordable = int(calc_shares_for_capital(capital_yuan=float(cash_avail), price=float(entry), cost=cost2, lot_size=int(lot)))
            buy_sh = int(min(want, affordable))
            if buy_sh <= 0:
                warnings.append(f"现金不足：{sym} 目标加仓{want}，但剩余现金≈{cash_avail:.0f} 买不起一手")
                continue
            # 仓位约束（add）：不让“新增买入”把总暴露推到 max_exposure_pct 之上
            if exposure_buy_budget_yuan is not None and float(entry) > 0:
                rem_exp = float(exposure_buy_budget_yuan) - float(exposure_buy_used_yuan)
                max_by_exp = _floor_to_lot(int(rem_exp / float(entry))) if rem_exp > 0 else 0
                if max_by_exp <= 0:
                    warnings.append(
                        f"仓位约束触发：positions_mv≈{positions_mv_yuan:.0f} 已接近/超过 max_exposure≈{(exposure_max_positions_yuan or 0.0):.0f}（停止后续买入）"
                    )
                    break
                buy_sh = int(min(int(buy_sh), int(max_by_exp)))
            # 换手约束（KISS：当前只限制 buy 侧）
            if buy_turnover_budget_yuan is not None:
                rem = float(buy_turnover_budget_yuan) - float(buy_turnover_used_yuan)
                max_by_turn = _floor_to_lot(int(rem / float(entry))) if rem > 0 else 0
                if max_by_turn <= 0:
                    warnings.append(f"换手约束触发：buy_turnover_used≈{buy_turnover_used_yuan:.0f} >= budget≈{buy_turnover_budget_yuan:.0f}（停止后续买入）")
                    break
                buy_sh = int(min(int(buy_sh), int(max_by_turn)))

            # 单笔金额门槛：达不到就别下（小单被 5 元最低佣金磨损得很难看）
            if float(min_trade_notional_yuan or 0.0) > 0 and float(entry) > 0:
                min_sh = _min_shares_for_notional(notional_yuan=float(min_trade_notional_yuan), price=float(entry))
                if min_sh > 0 and int(buy_sh) < int(min_sh):
                    est_notional = float(buy_sh) * float(entry)
                    warnings.append(
                        f"单笔金额门槛：{sym} 本次可买{buy_sh}份≈{est_notional:.0f} < {min_trade_notional_yuan:.0f}（需要≥{min_sh}份），跳过"
                    )
                    continue
            cash_need = float(_append_buy(sym=sym, sh=int(buy_sh), price_ref=entry, reason="add: 增量加仓"))
            cash_avail -= float(cash_need)
            buy_turnover_used_yuan += float(buy_sh) * float(entry)
            exposure_buy_used_yuan += float(buy_sh) * float(entry)

    out = sanitize_for_json(
        {
            "generated_at": datetime.now().isoformat(),
            "as_of": {
                "signals": (raw_sig.get("as_of") if isinstance(raw_sig, dict) else None) or (raw_sig.get("generated_at") if isinstance(raw_sig, dict) else None),
                "holdings": (hold_out.get("as_of") if isinstance(hold_out, dict) else None),
            },
            "mode": mode,
            "inputs": {
                "holdings_path": str(path),
                "signals_path": str(signals_path),
                "signals_generated_at": raw_sig.get("generated_at") if isinstance(raw_sig, dict) else None,
                "capital_yuan": float(capital_yuan),
                "roundtrip_cost_yuan": float(rt),
                "min_fee_yuan": float(min_fee_yuan),
                "buy_cost": float(buy_cost),
                "sell_cost": float(sell_cost),
                "slippage": {
                    "mode": str(slip_mode),
                    "bps": float(slip_bps),
                    "ref_amount_yuan": float(slip_ref_amt),
                    "bps_min": float(slip_bps_min),
                    "bps_max": float(slip_bps_max),
                    "unknown_bps": float(slip_unknown_bps),
                    "vol_mult": float(slip_vm),
                },
                "constraints": {
                    "lot_size": int(lot),
                    "limit_up_pct": float(getattr(tb_cfg, "limit_up_pct", 0.0)),
                    "limit_down_pct": float(getattr(tb_cfg, "limit_down_pct", 0.0)),
                    "halt_vol_zero": bool(getattr(tb_cfg, "halt_vol_zero", True)),
                    "min_trade_notional_yuan": float(min_trade_notional_yuan or 0.0),
                    "max_positions": int(max_positions_eff) if max_positions_eff is not None else None,
                    "max_position_pct": float(max_position_pct_eff) if max_position_pct_eff is not None else None,
                    "max_turnover_pct_buy_side": float(max_turnover_pct) if max_turnover_pct > 0 else None,
                },
                "vol_target": {
                    "target_ann": float(vol_target) if vol_target > 0 else None,
                    "lookback_days": int(vol_lookback_days) if vol_target > 0 else None,
                    "index_symbol": str(vol_index_symbol) if vol_index_symbol else None,
                    "realized_vol_ann": float(vol_realized_ann) if vol_realized_ann is not None else None,
                    "scale": float(vol_scale) if vol_scale is not None else None,
                    "max_exposure_pct_eff": float(max_exposure_eff) if max_exposure_eff is not None else None,
                    "error": str(vol_error) if vol_error else None,
                    "note": "当前实现=只降不升；只用指数历史波动率缩放 max_exposure_pct（研究用途）",
                },
                "phase2": {
                    "min_score": float(min_score) if min_score > 0 else None,
                    "filtered_by_min_score": int(filtered_by_min_score_sig),
                    "cash_signal_enabled": bool(use_cash_signal),
                    "cash_exposure_cap_pct": float(cash_exposure_cap_pct_eff) if cash_exposure_cap_pct_eff is not None else None,
                    "cash_signal": cash_sig,
                    "cash_signal_error": str(cash_sig_error) if cash_sig_error else None,
                    "tushare_pack_error": str(tushare_pack_error) if tushare_pack_error else None,
                },
            },
            "market_regime": market_regime,
            "account": {
                "cash_yuan": float(cash_amount) if cash_amount is not None else None,
                "equity_yuan": float(equity_yuan) if equity_yuan is not None else None,
            },
            "position_plan": {
                "profile": plan.get("profile") if isinstance(plan, dict) else None,
                "budget": plan.get("budget") if isinstance(plan, dict) else None,
                "plans": plan.get("plans") if isinstance(plan, dict) else None,
                "watch": plan.get("watch") if isinstance(plan, dict) else None,
            },
            "rebalance": {
                "targets": targets,
                "orders_next_open": orders,
                "cash_est": {
                    "cash_start_yuan": float(cash_amount) if cash_amount is not None else None,
                    "cash_in_est_yuan": float(cash_in_est),
                    "cash_out_est_yuan": float(cash_out_est),
                    "cash_remaining_est_yuan": float(cash_avail),
                    "note": "估算按 price_ref(close)*shares + 成本模型（比例成本/最低佣金/固定磨损/滑点）计算；仍未计入真实盘口价差/限价成交差异，下单前自己复核。",
                },
                "turnover_buy": {
                    "budget_yuan": float(buy_turnover_budget_yuan) if buy_turnover_budget_yuan is not None else None,
                    "used_yuan": float(buy_turnover_used_yuan),
                    "note": "max_turnover_pct 当前只限制 buy 侧（KISS）；如需同时限制 sell，我们再加。",
                },
                "exposure_buy": {
                    "max_exposure_pct": float(max_exposure_pct_eff2) if max_exposure_pct_eff2 is not None else None,
                    "max_positions_yuan": float(exposure_max_positions_yuan) if exposure_max_positions_yuan is not None else None,
                    "current_positions_mv_yuan": float(positions_mv_yuan),
                    "budget_yuan": float(exposure_buy_budget_yuan) if exposure_buy_budget_yuan is not None else None,
                    "used_yuan": float(exposure_buy_used_yuan),
                    "note": "add 模式下：把 total_exposure 卡在 max_exposure_pct（避免你有存量仓时，系统还建议把现金打光）。",
                },
            },
            "warnings": warnings,
            "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
        }
    )

    if getattr(args, "out", None):
        out_path = Path(str(args.out))
        write_json(out_path, out)
        # 把“组合层提示”也写进 diagnostics，避免只看 diagnostics 时一脸懵（仍会去重/限额）。
        try:
            for w in (warnings or [])[:50]:
                if isinstance(w, str) and w.strip():
                    diag.warn(w.strip())
        except Exception:  # noqa: BLE001
            pass
        diag.write(out_path.parent, cmd="rebalance-user")
        print(str(out_path.resolve()))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
