from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .llm_client import ChatMessage
from .pipeline import run_llm_text
from .prompting import load_prompt_text


DEFAULT_SCHOOLS = ["chan", "wyckoff", "ichimoku", "turtle", "momentum"]


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (AttributeError):  # noqa: BLE001
        return None


def _find_method_file(out_dir: Path, method: str, filename: str) -> Path | None:
    p1 = out_dir / method / filename
    if p1.exists():
        return p1
    p2 = out_dir / filename
    if p2.exists():
        return p2
    return None


def _compact_chan(obj: Any) -> Any | None:
    if not isinstance(obj, dict):
        return None
    strokes = obj.get("strokes") or []
    centers = obj.get("centers") or []
    return {
        "params": obj.get("params"),
        "summary": obj.get("summary"),
        "strokes_tail": strokes[-6:] if isinstance(strokes, list) else [],
        "centers_tail": centers[-3:] if isinstance(centers, list) else [],
    }


def _compact_dow(obj: Any) -> Any | None:
    if not isinstance(obj, dict):
        return None
    swings = obj.get("swings") or []
    return {"params": obj.get("params"), "summary": obj.get("summary"), "swings_tail": swings[-12:] if isinstance(swings, list) else []}


def _compact_vsa(obj: Any) -> Any | None:
    if not isinstance(obj, dict):
        return None
    events = obj.get("events") or []
    return {"params": obj.get("params"), "summary": obj.get("summary"), "last": obj.get("last"), "events": events if isinstance(events, list) else []}


def _compact_institution(obj: Any) -> Any | None:
    if not isinstance(obj, dict):
        return None
    pv = obj.get("price_volume") or {}
    ff = obj.get("fund_flow") or {}
    ff2 = None
    if isinstance(ff, dict):
        keys = [
            "source",
            "last_date",
            "main_net_5d",
            "main_net_20d",
            "main_pct_avg_5d",
            "main_pct_avg_20d",
            "super_net_5d",
            "big_net_5d",
        ]
        ff2 = {k: ff.get(k) for k in keys if k in ff}
    pv2 = None
    if isinstance(pv, dict):
        pv2 = {k: pv.get(k) for k in ["ad_delta_20", "obv_delta_20", "obv_slope_20", "vsa_bias", "vsa_summary"] if k in pv}
    return {"summary": obj.get("summary"), "price_volume": pv2, "fund_flow": ff2}


def _compact_tushare_factors(obj: Any) -> Any | None:
    """
    控制上下文体积：只保留“能用于解释/风控”的关键字段。
    """
    if not isinstance(obj, dict):
        return None

    erp0 = obj.get("erp") if isinstance(obj.get("erp"), dict) else None
    hsgt0 = obj.get("hsgt") if isinstance(obj.get("hsgt"), dict) else None
    micro0 = obj.get("microstructure") if isinstance(obj.get("microstructure"), dict) else None

    erp = None
    if isinstance(erp0, dict):
        rf = erp0.get("rf") if isinstance(erp0.get("rf"), dict) else {}
        rf10_0 = erp0.get("rf_alt_10y") if isinstance(erp0.get("rf_alt_10y"), dict) else None
        rf10 = None
        if isinstance(rf10_0, dict):
            # rf_alt_10y.rf 是我们关心的；其它 cache/source 信息丢掉。
            r10 = rf10_0.get("rf") if isinstance(rf10_0.get("rf"), dict) else {}
            rf10 = {
                "ok": rf10_0.get("ok"),
                "ref_date": rf10_0.get("ref_date"),
                "rf": {"name": r10.get("name"), "tenor": r10.get("tenor"), "yield": r10.get("yield"), "score01": r10.get("score01")},
            }
        erp = {
            "ok": erp0.get("ok"),
            "as_of": erp0.get("as_of"),
            "index_symbol": erp0.get("index_symbol"),
            "ref_date_index": erp0.get("ref_date_index"),
            "ref_date_rf": erp0.get("ref_date_rf"),
            "pe_used": erp0.get("pe_used"),
            "equity_yield": erp0.get("equity_yield"),
            "rf": {"name": rf.get("name"), "tenor": rf.get("tenor"), "yield": rf.get("yield")},
            "erp": erp0.get("erp"),
            "rf_alt_10y": rf10,
            "erp_alt_10y": erp0.get("erp_alt_10y"),
        }

    hsgt = None
    if isinstance(hsgt0, dict):
        n = hsgt0.get("north") if isinstance(hsgt0.get("north"), dict) else {}
        s = hsgt0.get("south") if isinstance(hsgt0.get("south"), dict) else {}
        hsgt = {
            "ok": hsgt0.get("ok"),
            "as_of": hsgt0.get("as_of"),
            "ref_date": hsgt0.get("ref_date"),
            "north": {"money_yuan": n.get("money_yuan"), "z": n.get("z"), "score01": n.get("score01")},
            "south": {"money_yuan": s.get("money_yuan"), "z": s.get("z"), "score01": s.get("score01")},
        }

    micro = None
    if isinstance(micro0, dict):
        last = micro0.get("last") if isinstance(micro0.get("last"), dict) else {}
        micro = {
            "ok": micro0.get("ok"),
            "as_of": micro0.get("as_of"),
            "ref_date": micro0.get("ref_date"),
            "z": micro0.get("z"),
            "score01": micro0.get("score01"),
            "last": {
                "net_big_amount_yuan": last.get("net_big_amount_yuan"),
                "net_big_ratio": last.get("net_big_ratio"),
                "net_total_ratio": last.get("net_total_ratio"),
            },
        }

    return {
        "ok": obj.get("ok"),
        "as_of": obj.get("as_of"),
        "erp": erp,
        "hsgt": hsgt,
        "microstructure": micro,
    }


def _compact_user_holdings(obj: Any, *, asset: str | None, symbol: str | None) -> dict[str, Any] | None:
    """
    user_holdings.json 可能包含用户仓位/现金/计划等敏感信息。
    这里严格裁剪：只保留“组合约束 + 当前标的相关仓位”，够用就行，别把整本账本塞给 LLM。
    """
    if not isinstance(obj, dict):
        return None

    a = str(asset or "").strip().lower()
    sym = str(symbol or "").strip()

    # 尽量把 meta.symbol 统一成持仓快照里常用的 sh/sz/bj 前缀形式（只处理 stock/etf，index 不折腾）。
    symbol_prefixed = None
    if a in {"stock", "etf"} and sym:
        # ⚠️ 这里别为了“按中文名解析”把 AkShare 拉全量名单，narrate 会变得又慢又不稳定。
        # 只在你传的是“代码/前缀代码”时才解析；传中文名就放过它（匹配不上就匹配不上，别折腾）。
        s2 = sym.strip().lower()
        can_fast_resolve = bool((len(s2) == 6 and s2.isdigit()) or s2.startswith(("sh", "sz", "bj")))
        if can_fast_resolve:
            try:
                from .akshare_source import resolve_symbol

                symbol_prefixed = resolve_symbol(a, sym)
            except (AttributeError):  # noqa: BLE001
                symbol_prefixed = None

    positions = obj.get("positions") if isinstance(obj.get("positions"), list) else []
    pos_total = len(positions)
    pos_frozen = 0
    this_positions: list[dict[str, Any]] = []

    for p in positions:
        if not isinstance(p, dict):
            continue
        if bool(p.get("frozen")):
            pos_frozen += 1
        if not symbol_prefixed:
            continue
        ps = str(p.get("symbol") or "").strip().lower()
        if not ps:
            continue
        # 持仓文件一般用 sh510150；如果 meta 里传了 510150，也能对上。
        if ps == symbol_prefixed or (len(ps) == 8 and len(symbol_prefixed) == 8 and ps[2:] == symbol_prefixed[2:]):
            this_positions.append(
                {
                    "symbol": ps,
                    "shares": p.get("shares"),
                    "cost_basis": p.get("cost_basis"),
                    "entry_style": p.get("entry_style"),
                    "frozen": bool(p.get("frozen")),
                    "note": p.get("note"),
                }
            )

    cash = obj.get("cash") if isinstance(obj.get("cash"), dict) else {}
    trade_rules = obj.get("trade_rules") if isinstance(obj.get("trade_rules"), dict) else {}

    # trade_rules 也裁剪一下，避免塞太多碎碎念。
    tr_keys = ["min_trade_notional_yuan", "max_positions", "max_position_pct", "rebalance_schedule", "note"]
    trade_rules2 = {k: trade_rules.get(k) for k in tr_keys if k in trade_rules}

    return {
        "last_updated": obj.get("last_updated"),
        "currency": obj.get("currency"),
        "cash": {"amount": cash.get("amount")},
        "trade_rules": trade_rules2,
        "portfolio": {"positions_total": int(pos_total), "positions_frozen": int(pos_frozen)},
        "this_symbol": {"symbol_prefixed": symbol_prefixed, "positions": this_positions},
    }


def _build_analysis_hints(
    *,
    meta: dict[str, Any] | None,
    signal_backtest: dict[str, Any] | None,
    user_holdings: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    给 LLM 一个“别瞎看/别瞎说”的硬提示：关注优先级 + 交易约束。
    目标：减少噪音、让动作更贴近可执行规则（胜率优先 + 风控优先）。
    """
    if not isinstance(meta, dict):
        return None

    asset = str(meta.get("asset") or "").strip().lower() or None
    symbol = str(meta.get("symbol") or "").strip() or None

    # 当前标的是否持仓/是否冻结
    has_position = False
    frozen = False
    if isinstance(user_holdings, dict):
        try:
            ps = ((user_holdings.get("this_symbol") or {}).get("positions") or []) if isinstance(user_holdings.get("this_symbol"), dict) else []
            if isinstance(ps, list) and ps:
                has_position = True
                frozen = any(bool(p.get("frozen")) for p in ps if isinstance(p, dict))
        except (AttributeError):  # noqa: BLE001
            has_position = False
            frozen = False

    action = None
    risk = {}
    if isinstance(signal_backtest, dict):
        try:
            action = ((signal_backtest.get("decision") or {}).get("action")) if isinstance(signal_backtest.get("decision"), dict) else None
        except (AttributeError):  # noqa: BLE001
            action = None
        try:
            risk = signal_backtest.get("risk_signals") if isinstance(signal_backtest.get("risk_signals"), dict) else {}
        except (AttributeError):  # noqa: BLE001
            risk = {}

    # 关注优先级（先保命，再谈入场）
    priority = []
    if action is not None:
        priority.append("signal_backtest.decision.action（硬规则：必须照抄执行）")
    if has_position:
        priority.append("风控优先：weekly_below_ma50_confirm2 / daily_macd_bearish_2d / daily_close_below_ma20_confirm2")
    priority.append("周线结构优先（日线只做风险确认/入场位置）")
    if asset == "stock":
        priority.append("个股：microstructure(大单/超大单) + institution 只做确认/解释，不当成买卖按钮")
    elif asset == "etf":
        priority.append("ETF：BBB/趋势结构/流动性优先；ERP/HSGT 只做风险温度计")
    elif asset == "index":
        priority.append("指数：当成环境变量，不当成个股买点")

    restrictions = []
    if frozen:
        restrictions.append("该标的 frozen=true：默认不建议加仓/换仓（除非出现明确“退出”风控信号）")
    if (action in {"减仓", "退出"}) and (not has_position):
        restrictions.append("action=减仓/退出 但你当前空仓：等价于‘不新开仓/保持观望’")

    # trade_rules（可选）
    tr = None
    if isinstance(user_holdings, dict) and isinstance(user_holdings.get("trade_rules"), dict):
        tr = user_holdings.get("trade_rules")

    return {
        "asset": asset,
        "symbol": symbol,
        "position_state": {"has_position": bool(has_position), "frozen": bool(frozen)},
        "decision_action": action,
        "risk_signals": dict(risk) if isinstance(risk, dict) else {},
        "priority": priority,
        "restrictions": restrictions,
        "trade_rules": tr,
    }


def collect_analysis_bundle(out_dir: Path, *, schools: list[str] | None = None) -> dict[str, Any]:
    schools2 = schools or DEFAULT_SCHOOLS
    out: dict[str, Any] = {"schools": list(schools2)}

    meta = _read_json(out_dir / "meta.json")
    if isinstance(meta, dict):
        out["meta"] = meta
    sb = _read_json(out_dir / "signal_backtest.json")
    if isinstance(sb, dict):
        out["signal_backtest"] = sb
    tf = _read_json(out_dir / "tushare_factors.json")
    if isinstance(tf, dict):
        out["tushare_factors"] = _compact_tushare_factors(tf)

    # 可选：带上用户持仓约束（只裁剪“组合规则 + 当前标的相关仓位”）
    user_holdings = None
    try:
        u = _read_json(Path("data") / "user_holdings.json")
        if isinstance(meta, dict):
            uh = _compact_user_holdings(u, asset=str(meta.get("asset") or ""), symbol=str(meta.get("symbol") or ""))
        else:
            uh = _compact_user_holdings(u, asset=None, symbol=None)
        if uh is not None:
            out["user_holdings"] = uh
            user_holdings = uh
    except (AttributeError):  # noqa: BLE001
        pass

    # 自动“关注重点”提示：让模型别乱飞（胜率优先 + 风控优先）
    try:
        hints = _build_analysis_hints(meta=meta if isinstance(meta, dict) else None, signal_backtest=sb if isinstance(sb, dict) else None, user_holdings=user_holdings)
        if hints is not None:
            out["analysis_hints"] = hints
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        pass

    for school in schools2:
        s = (school or "").strip().lower()
        if s == "chan":
            p = _find_method_file(out_dir, "chan", "chan_structure.json")
            obj = _read_json(p) if p else None
            out["chan"] = _compact_chan(obj)
        elif s == "wyckoff":
            features_p = _find_method_file(out_dir, "wyckoff", "wyckoff_features.json")
            analysis_p = _find_method_file(out_dir, "wyckoff", "analysis.json")
            out["wyckoff"] = {
                "features": _read_json(features_p) if features_p else None,
                "llm_analysis": _read_json(analysis_p) if analysis_p else None,
            }
        elif s == "ichimoku":
            p = _find_method_file(out_dir, "ichimoku", "ichimoku.json")
            out["ichimoku"] = _read_json(p) if p else None
        elif s == "turtle":
            p = _find_method_file(out_dir, "turtle", "turtle.json")
            out["turtle"] = _read_json(p) if p else None
        elif s == "momentum":
            p = _find_method_file(out_dir, "momentum", "momentum.json")
            out["momentum"] = _read_json(p) if p else None
        elif s == "dow":
            p = _find_method_file(out_dir, "dow", "dow.json")
            obj = _read_json(p) if p else None
            out["dow"] = _compact_dow(obj)
        elif s == "vsa":
            p = _find_method_file(out_dir, "vsa", "vsa_features.json")
            obj = _read_json(p) if p else None
            out["vsa"] = _compact_vsa(obj)
        elif s == "institution":
            p = _find_method_file(out_dir, "institution", "institution.json")
            obj = _read_json(p) if p else None
            out["institution"] = _compact_institution(obj)
        else:
            out.setdefault("unknown_schools", []).append(s)

    return out


def generate_narrative_text(
    cfg: AppConfig,
    *,
    out_dir: Path,
    provider: str,
    prompt_path: str,
    schools: list[str] | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> str:
    prompt_text = load_prompt_text(prompt_path)
    bundle = collect_analysis_bundle(out_dir, schools=schools)
    user = prompt_text.strip() + "\n\n分析结果(JSON)：\n" + json.dumps(bundle, ensure_ascii=False, indent=2)
    mem_ctx = ""
    try:
        # 记忆是“偏好/纪律/复盘”的硬约束提示：只给摘要，别把整本日记塞进 prompt。
        from .memory_store import build_prompt_memory_context, resolve_memory_paths

        mem_ctx = build_prompt_memory_context(
            resolve_memory_paths(project_root=cfg.project_root),
            include_long_term=True,
            include_profile=True,
            include_daily_days=2,
            max_chars=6000,
        )
    except Exception:  # noqa: BLE001
        mem_ctx = ""

    system = "你是一个严谨的交易研究解读助手。你必须用中文输出，不构成投资建议。"
    if mem_ctx.strip():
        system += "\n\n# 用户偏好/约束（持久记忆，供参考；如与用户最新输入冲突，以最新输入为准）\n" + mem_ctx.strip()
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return run_llm_text(
        cfg,
        messages=messages,
        provider=provider,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
    )
