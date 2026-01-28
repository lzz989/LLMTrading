from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .json_utils import sanitize_for_json


MergeConflictMode = Literal["risk_first", "priority", "vote"]


def _fnum(x: Any) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def parse_strategy_weights(text: str | None) -> dict[str, float]:
    """
    解析权重：bbb_weekly=1,trend_pullback_weekly=0.8
    """
    t = str(text or "").strip()
    if not t:
        return {}
    out: dict[str, float] = {}
    for part in t.split(","):
        p = part.strip()
        if not p:
            continue
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kk = str(k).strip()
        if not kk:
            continue
        try:
            w = float(v)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        if not math.isfinite(w) or w <= 0:
            continue
        out[kk] = float(w)
    return out


def parse_priority(text: str | None) -> list[str]:
    """
    解析优先级列表：bbb_weekly,trend_pullback_weekly,left_dip_rr
    """
    t = str(text or "").strip()
    if not t:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in t.split(","):
        k = str(part or "").strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _max_date_str(values: list[Any]) -> str | None:
    best = None
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        if best is None or s > best:
            best = s
    return best


def _load_signals_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (AttributeError) as exc:  # noqa: BLE001
        raise ValueError(f"signals.json 解析失败：{path} {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"signals.json 不是 dict：{path}")
    sv = int(obj.get("schema_version") or 0)
    if sv != 1:
        raise ValueError(f"signals.schema_version!=1：{path} got={sv}")
    if not isinstance(obj.get("items"), list):
        raise ValueError(f"signals.items 不是 list：{path}")
    return obj


def _resolve_action(
    actions: list[str],
    *,
    mode: MergeConflictMode,
    priority: list[str],
    by_strategy_action: dict[str, str],
    weights: dict[str, float],
) -> tuple[str, str]:
    """
    返回：resolved_action, rule
    """
    acts = [str(a or "").strip().lower() for a in actions if str(a or "").strip()]
    if not acts:
        return "watch", "empty_default_watch"

    # 先归一化一下
    norm = []
    for a in acts:
        if a in {"entry", "watch", "avoid", "exit"}:
            norm.append(a)
    if not norm:
        return "watch", "unknown_actions_default_watch"

    if mode == "risk_first":
        if "exit" in norm:
            return "exit", "risk_first.exit"
        if "avoid" in norm:
            return "avoid", "risk_first.avoid"
        if "entry" in norm:
            return "entry", "risk_first.entry"
        return "watch", "risk_first.watch"

    if mode == "priority":
        for k in priority:
            a = str(by_strategy_action.get(k) or "").strip().lower()
            if a in {"entry", "watch", "avoid", "exit"}:
                return a, f"priority.{k}"
        # priority 没命中，退回 risk_first
        if "exit" in norm:
            return "exit", "priority.fallback_exit"
        if "avoid" in norm:
            return "avoid", "priority.fallback_avoid"
        if "entry" in norm:
            return "entry", "priority.fallback_entry"
        return "watch", "priority.fallback_watch"

    # vote：按权重投票（同票按 risk_first）
    score = {"entry": 0.0, "watch": 0.0, "avoid": 0.0, "exit": 0.0}
    for strat, act in by_strategy_action.items():
        a = str(act or "").strip().lower()
        if a not in score:
            continue
        w = float(weights.get(strat, 1.0))
        if not math.isfinite(w) or w <= 0:
            w = 1.0
        score[a] += float(w)

    best = max(score.items(), key=lambda kv: kv[1])[0]
    top = score.get(best, 0.0)
    tied = [k for k, v in score.items() if abs(float(v) - float(top)) <= 1e-12 and float(v) > 0]
    if len(tied) == 1:
        return best, "vote.weighted_majority"

    # tie -> risk_first
    if "exit" in norm:
        return "exit", "vote.tie_exit"
    if "avoid" in norm:
        return "avoid", "vote.tie_avoid"
    if "entry" in norm:
        return "entry", "vote.tie_entry"
    return "watch", "vote.tie_watch"


def _pick_primary_strategy(
    by: dict[str, Any],
    *,
    resolved_action: str,
    conflict: MergeConflictMode,
    priority: list[str],
    weights: dict[str, float],
) -> str | None:
    """
    选一个“主策略”，把它的 entry/meta 透传给下游（rebalance/paper-sim）。

    约定（KISS + 可复现）：
    - 优先选 action==resolved_action 的贡献者（否则退化成全量里挑一个最“强”的）。
    - conflict=priority 时优先命中 priority 列表；
    - 其余情况按 weight*score 最大；再按 weight*confidence；再按 strategy 名字稳定排序。
    """
    if not by:
        return None

    def _w(k: str) -> float:
        try:
            v = float(weights.get(k, 1.0))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            v = 1.0
        if not math.isfinite(v) or v <= 0:
            v = 1.0
        return float(v)

    ra = str(resolved_action or "").strip().lower()
    cand: list[str] = []
    for strat, info in by.items():
        if not isinstance(info, dict):
            continue
        act = str(info.get("action") or "").strip().lower()
        if act == ra:
            cand.append(str(strat))
    if not cand:
        cand = [str(strat) for strat, info in by.items() if isinstance(info, dict)]
    if not cand:
        return None

    if conflict == "priority" and priority:
        for k in priority:
            kk = str(k or "").strip()
            if kk and kk in cand:
                return kk

    best = None
    best_key = None
    for strat in cand:
        info = by.get(strat)
        if not isinstance(info, dict):
            continue
        s = _fnum(info.get("score"))
        c = _fnum(info.get("confidence"))
        w = _w(strat)
        ws = float(s) * float(w) if s is not None else -1.0
        wc = float(c) * float(w) if c is not None else -1.0
        key = (ws, wc, str(strat))
        if best_key is None or key > best_key:
            best_key = key
            best = str(strat)
    return best


def merge_signals_files(
    paths: list[Path],
    *,
    conflict: MergeConflictMode = "risk_first",
    weights: dict[str, float] | None = None,
    priority: list[str] | None = None,
    top_k: int = 0,
) -> dict[str, Any]:
    """
    合并多份 signals.json（schema_version=1）到一个“候选集合”。
    - 不做交易建议；只是让组合层/执行层能“看到多策略全貌”。
    """
    w = dict(weights or {})
    pr = list(priority or [])

    inputs: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for p in paths:
        obj = _load_signals_json(p)
        payloads.append(obj)
        inputs.append(
            {
                "path": str(p),
                "strategy": obj.get("strategy"),
                "generated_at": obj.get("generated_at"),
                "as_of": obj.get("as_of"),
                "source": obj.get("source"),
            }
        )

    items_map: dict[str, dict[str, Any]] = {}
    as_of_all: list[Any] = []
    market_regime_first = None
    config_all: list[dict[str, Any]] = []
    for obj in payloads:
        as_of_all.append(obj.get("as_of") or obj.get("generated_at"))
        if market_regime_first is None and isinstance(obj.get("market_regime"), dict):
            market_regime_first = obj.get("market_regime")
        cfg0 = obj.get("config")
        if isinstance(cfg0, dict):
            config_all.append(cfg0)
        strat = str(obj.get("strategy") or "").strip()
        src = obj.get("source")
        for it in obj.get("items") or []:
            if not isinstance(it, dict):
                continue
            asset = str(it.get("asset") or "").strip().lower()
            sym = str(it.get("symbol") or "").strip()
            if not asset or not sym:
                continue
            key = f"{asset}:{sym}"

            rec = items_map.get(key)
            if rec is None:
                rec = {
                    "asset": asset,
                    "symbol": sym,
                    "name": str(it.get("name") or "").strip() or None,
                    "tags": [],
                    "_by_strategy": {},
                }
                items_map[key] = rec

            # name：优先保留已有；否则用新的非空值
            if (not rec.get("name")) and str(it.get("name") or "").strip():
                rec["name"] = str(it.get("name") or "").strip()

            # tags：union
            tags = it.get("tags") if isinstance(it.get("tags"), list) else []
            for t in tags:
                ts = str(t or "").strip()
                if not ts:
                    continue
                rec.setdefault("tags", [])
                if ts not in rec["tags"]:
                    rec["tags"].append(ts)

            by = rec["_by_strategy"]
            by[strat] = {
                "action": str(it.get("action") or "").strip().lower() or None,
                "score": _fnum(it.get("score")),
                "confidence": _fnum(it.get("confidence")),
                "source": src,
                "generated_at": obj.get("generated_at"),
                "as_of": obj.get("as_of"),
                "entry": (it.get("entry") if isinstance(it.get("entry"), dict) else None),
                "meta": (it.get("meta") if isinstance(it.get("meta"), dict) else None),
            }

    merged_items: list[dict[str, Any]] = []
    for _, rec in items_map.items():
        by = rec.get("_by_strategy") if isinstance(rec.get("_by_strategy"), dict) else {}
        actions: list[str] = []
        by_action: dict[str, str] = {}
        score_max = None
        conf_max = None
        score_weighted_max = None
        contributors: list[dict[str, Any]] = []
        for strat, info in by.items():
            if not isinstance(info, dict):
                continue
            act = str(info.get("action") or "").strip().lower()
            if act:
                actions.append(act)
                by_action[str(strat)] = act
            s = _fnum(info.get("score"))
            c = _fnum(info.get("confidence"))
            if s is not None:
                score_max = s if score_max is None else max(float(score_max), float(s))
                ww = float(w.get(strat, 1.0))
                if not math.isfinite(ww) or ww <= 0:
                    ww = 1.0
                sw = float(s) * float(ww)
                score_weighted_max = sw if score_weighted_max is None else max(float(score_weighted_max), float(sw))
            if c is not None:
                conf_max = c if conf_max is None else max(float(conf_max), float(c))

            contributors.append(
                {
                    "strategy": str(strat),
                    "action": act or None,
                    "score": s,
                    "confidence": c,
                    "weight": float(w.get(strat, 1.0)) if str(strat) in w else None,
                    "as_of": info.get("as_of"),
                    "generated_at": info.get("generated_at"),
                }
            )

        resolved_action, rule = _resolve_action(actions, mode=conflict, priority=pr, by_strategy_action=by_action, weights=w)

        primary_strategy = _pick_primary_strategy(by, resolved_action=resolved_action, conflict=conflict, priority=pr, weights=w)
        primary_entry = None
        primary_meta = None
        if primary_strategy is not None:
            info = by.get(primary_strategy)
            if isinstance(info, dict):
                primary_entry = info.get("entry") if isinstance(info.get("entry"), dict) else None
                primary_meta = info.get("meta") if isinstance(info.get("meta"), dict) else None

        merged_info = {
            "conflict_mode": conflict,
            "rule": rule,
            "score_weighted_max": score_weighted_max,
            "primary_strategy": primary_strategy,
            "strategies": contributors,
        }

        if primary_meta is not None:
            meta_out: dict[str, Any] = dict(primary_meta)
            if "merged" in meta_out and isinstance(meta_out.get("merged"), dict):
                meta_out["merged"]["signals_merge"] = merged_info
            elif "merged" in meta_out:
                meta_out["merged_signals_merge"] = merged_info
            else:
                meta_out["merged"] = merged_info
        else:
            meta_out = {"merged": merged_info}

        merged_items.append(
            {
                "asset": rec.get("asset"),
                "symbol": rec.get("symbol"),
                "name": rec.get("name"),
                "action": resolved_action,
                "score": score_max,
                "confidence": conf_max,
                "confidence_ref": "merged_max",
                # 关键：透传主策略的 entry/meta，组合层要靠它算止损/仓位。
                "entry": primary_entry,
                "meta": meta_out,
                "tags": rec.get("tags") or [],
            }
        )

    # 排序：entry 最前；再按 score/confidence（不保证跨策略可比，只是给你个“先看哪几个”）
    rank = {"entry": 3, "watch": 2, "avoid": 1, "exit": 0}

    def _k(it: dict[str, Any]):
        a = str(it.get("action") or "watch")
        r = int(rank.get(a, 0))
        s = _fnum(it.get("score")) or 0.0
        c = _fnum(it.get("confidence")) or 0.0
        return (-r, -float(c), -float(s), str(it.get("asset") or ""), str(it.get("symbol") or ""))

    merged_items.sort(key=_k)
    if int(top_k or 0) > 0:
        merged_items = merged_items[: int(top_k)]

    # config：成本/滑点属于“账户参数”，理论上应该放到 holdings/config 里；但当前链路依赖 signals.config。
    # 合并规则：多份 config 对同一字段给不同值 => 不瞎猜（写进 conflicts，值留 None）。
    cfg_out: dict[str, Any] = {"merge": {"conflict": conflict, "weights": w or None, "priority": pr or None}}
    cfg_conflicts: dict[str, Any] = {}
    if config_all:
        keys = [
            "roundtrip_cost_yuan",
            "min_fee_yuan",
            "buy_cost",
            "sell_cost",
            "slippage_mode",
            "slippage_bps",
            "slippage_ref_amount_yuan",
            "slippage_bps_min",
            "slippage_bps_max",
            "slippage_unknown_bps",
            "slippage_vol_mult",
        ]
        for k in keys:
            vals: list[Any] = []
            for cfg in config_all:
                if k in cfg:
                    v = cfg.get(k)
                    if v is not None:
                        vals.append(v)
            if not vals:
                continue
            first = vals[0]
            same = True
            for v in vals[1:]:
                if v != first:
                    same = False
                    break
            if same:
                cfg_out[k] = first
            else:
                cfg_out[k] = None
                cfg_conflicts[k] = vals
    if cfg_conflicts:
        cfg_out["conflicts"] = cfg_conflicts

    return sanitize_for_json(
        {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(),
            "as_of": _max_date_str(as_of_all),
            "strategy": "signals_merged",
            "source": {"type": "signals-merge", "inputs": inputs},
            "market_regime": market_regime_first,
            "config": cfg_out,
            "counts": {"inputs": int(len(paths)), "items": int(len(merged_items))},
            "items": merged_items,
            "disclaimer": "研究工具输出，不构成投资建议；买卖自负。",
        }
    )
