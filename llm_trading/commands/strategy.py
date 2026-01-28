from __future__ import annotations

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..akshare_source import FetchParams
from ..data_cache import fetch_daily_cached
from ..json_utils import sanitize_for_json
from ..pipeline import write_json
from ..resample import resample_to_weekly
from ..strategy_config_loader import load_strategy_configs_yaml

from .common import _compute_market_regime_payload, _write_run_config, _write_run_meta


SignalAction = Literal["entry", "watch", "avoid", "exit"]


def _fnum(x) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def _ensure_ohlcv(df):
    if df is None or getattr(df, "empty", True):
        return df
    df2 = df.copy()
    if "open" not in df2.columns:
        df2["open"] = df2.get("close")
    if "high" not in df2.columns:
        df2["high"] = df2.get("close")
    if "low" not in df2.columns:
        df2["low"] = df2.get("close")
    if "volume" not in df2.columns:
        df2["volume"] = 0.0
    if "amount" not in df2.columns:
        try:
            df2["amount"] = df2["close"].astype(float) * df2["volume"].astype(float)
        except (AttributeError):  # noqa: BLE001
            pass
    return df2


def _pct_chg(close: list[float]) -> float | None:
    if len(close) < 2:
        return None
    c0 = close[-2]
    c1 = close[-1]
    if c0 <= 0 or (not math.isfinite(c0)) or (not math.isfinite(c1)):
        return None
    return float((c1 / c0) - 1.0)


def cmd_scan_strategy(args: argparse.Namespace) -> int:
    """
    Phase3：按 config/strategy_configs.yaml（StrategyEngine）生成 signals.json（研究用途）。

    说明：
    - 默认只输出（不替换 scan-etf/scan-stock 的现有口径）。
    - 产物遵守 signals schema_version=1，方便被 run/portfolio/paper-sim 吃。
    """
    import pandas as pd

    from ..factors.base import StrategyEngine

    asset = str(getattr(args, "asset", "") or "").strip().lower()
    if asset not in {"etf", "stock", "index"}:
        raise SystemExit("--asset 只能是 etf/stock/index")

    freq = str(getattr(args, "freq", "") or "weekly").strip().lower()
    if freq not in {"daily", "weekly"}:
        raise SystemExit("--freq 只能是 daily/weekly")

    # 数据源选择（提升 stock/index 扫描稳定性：优先 TuShare，失败回退 AkShare）
    src_raw = str(getattr(args, "source", "") or "").strip().lower()
    if src_raw not in {"akshare", "tushare", "auto"}:
        # 默认：ETF 走 AkShare（复权/折算更友好）；stock/index 默认 auto（有 token 就优先 TuShare）
        src_raw = "auto" if asset in {"stock", "index"} else "akshare"
    source = str(src_raw)

    strategy_key = str(getattr(args, "strategy", "") or "").strip()
    if not strategy_key:
        raise SystemExit("--strategy 不能为空（例：bbb_weekly / conservative）")

    cfg_path = Path(str(getattr(args, "strategy_config", "") or "").strip() or (Path("config") / "strategy_configs.yaml"))
    cfgs = load_strategy_configs_yaml(cfg_path)
    if strategy_key not in cfgs:
        raise SystemExit(f"未知 strategy：{strategy_key}（可用：{', '.join(sorted(cfgs))}）")
    cfg = cfgs[strategy_key]

    out_dir_raw = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"scan_strategy_{asset}_{strategy_key}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # universe
    uni_raw = str(getattr(args, "universe", "") or "").strip().lower()
    limit = int(getattr(args, "limit", 0) or 0)
    top_k = int(getattr(args, "top_k", 30) or 30)
    top_k = max(1, min(top_k, 5000))
    include_all_funds = bool(getattr(args, "include_all_funds", False))
    include_st = bool(getattr(args, "include_st", False))
    include_bj = bool(getattr(args, "include_bj", True))

    symbol_name: dict[str, str] = {}
    syms: list[str] = []

    if asset == "etf":
        from ..etf_scan import load_etf_universe

        items = load_etf_universe(include_all_funds=include_all_funds)
        for it in items:
            sym = str(getattr(it, "symbol", "") or "").strip()
            if not sym:
                continue
            nm = str(getattr(it, "name", "") or "").strip()
            if nm:
                symbol_name[sym] = nm
            syms.append(sym)
    elif asset == "stock":
        if (not uni_raw) or uni_raw in {"hs300", "000300"}:
            from ..stock_scan import load_index_stock_universe

            items = load_index_stock_universe(index_symbol="000300")
        elif uni_raw.startswith("index:"):
            from ..stock_scan import load_index_stock_universe

            idx = uni_raw.split(":", 1)[-1].strip() or "000300"
            items = load_index_stock_universe(index_symbol=idx)
        elif uni_raw in {"all", "a"}:
            from ..stock_scan import load_stock_universe

            items = load_stock_universe(include_st=include_st, include_bj=include_bj)
        else:
            raise SystemExit("--universe(stock) 仅支持 hs300 / index:000300 / all")
        for it in items:
            sym = str(getattr(it, "symbol", "") or "").strip()
            if not sym:
                continue
            nm = str(getattr(it, "name", "") or "").strip()
            if nm:
                symbol_name[sym] = nm
            syms.append(sym)
    else:
        s = str(getattr(args, "symbol", "") or "").strip()
        if not s:
            s = "sh000300"
        syms = [s]

    # allow manual --symbol append
    sym_raw = getattr(args, "symbol", None)
    if isinstance(sym_raw, list):
        for s in sym_raw:
            s2 = str(s or "").strip()
            if s2:
                syms.append(s2)

    # dedupe keep order
    seen = set()
    syms2: list[str] = []
    for s in syms:
        if s in seen:
            continue
        seen.add(s)
        syms2.append(s)
    syms = syms2

    # 统一 symbol 口径 + 黑白名单过滤（让 stock 扫描更像“可跑批”的东西）
    def _norm_symbol(s: str) -> str:
        x = str(s or "").strip().lower()
        if not x:
            return ""
        # already prefixed
        if len(x) == 8 and x[:2] in {"sh", "sz", "bj"} and x[2:].isdigit():
            return x
        # digits-only
        if len(x) == 6 and x.isdigit():
            if asset == "etf":
                return f"sh{x}" if x.startswith("5") else f"sz{x}"
            if asset == "stock":
                if x.startswith("6"):
                    return f"sh{x}"
                if x.startswith(("0", "3")):
                    return f"sz{x}"
                if x.startswith("9"):
                    return f"bj{x}"
            if asset == "index":
                # KISS：对常见 000xxx/399xxx 做个粗前缀；复杂映射留给 resolve_symbol 兜底
                if x.startswith("399"):
                    return f"sz{x}"
                return f"sh{x}"
        return x

    # normalize + dedupe again (600641 vs sh600641)
    seen = set()
    syms3: list[str] = []
    for s in syms:
        s2 = _norm_symbol(str(s))
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        syms3.append(s2)
    syms = syms3

    def _load_symbol_set(spec) -> set[str]:
        import re

        raw = str(spec or "").strip()
        if not raw:
            return set()
        p = Path(raw)
        parts: list[str] = []
        if p.exists() and p.is_file():
            try:
                txt = p.read_text(encoding="utf-8")
            except OSError:
                txt = ""
            for line in txt.splitlines():
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                parts.append(s)
        else:
            parts.extend([x for x in re.split(r"[,\s]+", raw) if str(x).strip()])

        out: set[str] = set()
        for s in parts:
            s2 = _norm_symbol(str(s))
            if s2:
                out.add(s2)
        return out

    wl = _load_symbol_set(getattr(args, "whitelist", None))
    if wl:
        syms = [s for s in syms if s in wl]
    bl = _load_symbol_set(getattr(args, "blacklist", None))
    if bl:
        syms = [s for s in syms if s not in bl]

    if limit > 0:
        syms = syms[: int(limit)]
    if not syms:
        raise SystemExit("universe 为空：没拿到任何 symbol")

    cache_dir = Path(str(getattr(args, "cache_dir", "") or "").strip() or (Path("data") / "cache" / asset))
    cache_ttl_hours = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)
    window = int(getattr(args, "window", 400) or 400)
    min_score = float(getattr(args, "min_score", 0.0) or 0.0)
    min_score = max(0.0, min(min_score, 1.0))
    workers = int(getattr(args, "workers", 8) or 8)
    workers = max(1, min(workers, 64))

    # 市场 regime（用于 allowed_regimes；注意：这是“市场环境”，不是单标的 regime）
    regime_index = str(getattr(args, "regime_index", "sh000300") or "sh000300").strip()
    regime_canary = bool(getattr(args, "regime_canary", True))
    regime_dict, regime_error, regime_index_eff = _compute_market_regime_payload(regime_index, canary_downgrade=regime_canary)
    regime_label = str((regime_dict or {}).get("label") or "unknown")

    engine = StrategyEngine(cfg)
    post_process: dict[str, Any] = {}

    def _compute_one(sym: str) -> dict[str, Any] | None:
        try:
            df_d = fetch_daily_cached(
                FetchParams(asset=asset, symbol=str(sym), source=source),
                cache_dir=cache_dir,
                ttl_hours=cache_ttl_hours,
            )
        except Exception:  # noqa: BLE001
            return None

        if df_d is None or getattr(df_d, "empty", True):
            return None

        try:
            df_d2 = df_d.copy()
            df_d2["date"] = pd.to_datetime(df_d2["date"], errors="coerce")
            df_d2 = df_d2.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            df_d2 = df_d

        df_use = df_d2
        if freq == "weekly":
            try:
                df_use = resample_to_weekly(df_d2)
            except Exception:  # noqa: BLE001
                df_use = df_d2

        df_use = _ensure_ohlcv(df_use)
        if window > 0 and len(df_use) > window:
            df_use = df_use.iloc[-int(window) :].reset_index(drop=True)

        try:
            close_ser = pd.to_numeric(df_use["close"], errors="coerce").astype(float)
            close_last = _fnum(close_ser.iloc[-1])
            close_vals = close_ser.tail(2).tolist()
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            close_last = None
            close_vals = []

        try:
            amt_last = None
            if "amount" in df_d2.columns:
                amt_last = _fnum(pd.to_numeric(df_d2["amount"], errors="coerce").astype(float).iloc[-1])
            if amt_last is None and "volume" in df_d2.columns and close_last is not None:
                vol_last = _fnum(pd.to_numeric(df_d2["volume"], errors="coerce").astype(float).iloc[-1])
                if vol_last is not None:
                    amt_last = float(close_last) * float(vol_last)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            amt_last = None

        try:
            vol_last = _fnum(pd.to_numeric(df_d2.get("volume"), errors="coerce").astype(float).iloc[-1]) if "volume" in df_d2.columns else None
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            vol_last = None

        def _mean_tail(df0, col: str, n: int) -> float | None:
            try:
                if col not in df0.columns:
                    return None
                s = pd.to_numeric(df0[col], errors="coerce").astype(float).tail(int(n))
                v = float(s.mean())
                return v if math.isfinite(v) else None
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                return None

        amt_avg20 = _mean_tail(df_d2, "amount", 20)
        vol_avg20 = _mean_tail(df_d2, "volume", 20)

        try:
            last_dt = df_d2["date"].iloc[-1]
            as_of = str(last_dt.date()) if hasattr(last_dt, "date") else None
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            as_of = None

        sig = engine.generate_signal(df_use, market_regime=regime_label)
        sc = _fnum(sig.get("score"))
        if sc is None:
            sc = 0.0

        # min_score：只过滤“候选质量”，不影响你后面做对齐报告
        if min_score > 0 and float(sc) < float(min_score):
            return None

        act0 = str(sig.get("action") or "").strip().lower()
        reason = str(sig.get("reason") or "").strip() or None

        action: SignalAction = "watch"
        if act0 == "entry":
            action = "entry"
        elif act0 == "exit":
            action = "exit"
        else:
            # hold：如果是“环境不允许”导致的 hold，用 avoid 更诚实
            if "不在允许范围" in str(reason or ""):
                action = "avoid"
            else:
                action = "watch"

        item = {
            "asset": str(asset),
            "symbol": str(sym),
            "name": symbol_name.get(str(sym), str(sym)),
            "action": action,
            "score": float(sc),
            "confidence": _fnum(sig.get("confidence")),
            "confidence_ref": "factor_confidence_wavg" if sig.get("confidence") is not None else None,
            "entry": {"price_ref": close_last, "price_ref_type": "close", "notes": reason},
            "meta": {
                "close": close_last,
                "pct_chg": _pct_chg([float(x) for x in close_vals if x is not None]) if close_vals else None,
                "amount": amt_last,
                "liquidity": {
                    "amount_last": amt_last,
                    "amount_avg20": amt_avg20,
                    "volume_last": vol_last,
                    "volume_avg20": vol_avg20,
                },
                "strategy_signal": sig,
                "as_of": as_of,
            },
            "tags": ["strategy_config", str(strategy_key)],
        }
        return sanitize_for_json(item)

    items: list[dict[str, Any]] = []
    done = 0
    total = len(syms)
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(_compute_one, sym): sym for sym in syms}
        for fut in as_completed(futs):
            done += 1
            it = None
            try:
                it = fut.result()
            except (AttributeError):  # noqa: BLE001
                it = None
            if it is not None:
                items.append(it)

    computed_n = int(len(items))

    # B) stock：资金（主力/大单）作为“惩罚项”（ETF 不强绑）。
    # - 只对 top candidates 取 TuShare moneyflow（有 token 才跑；失败自动降级）
    # - 只降不升：避免“资金因子”变成追涨按钮
    if asset == "stock" and items:
        try:
            from datetime import date as _date
            from datetime import datetime as _datetime

            from ..tushare_factors import compute_stock_microstructure_tushare
            from ..tushare_source import load_tushare_env

            env = load_tushare_env()
            if env is not None:
                # 只对 top 候选做二次打分，减少 API 压力
                cand_n = max(60, int(top_k) * 3)
                cand_n = min(int(cand_n), int(len(items)))

                def _score(it0: dict[str, Any]) -> float:
                    try:
                        return float(it0.get("score") or 0.0)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        return 0.0

                cands = [it for it in items if str(it.get("action") or "") != "avoid"]
                cands.sort(key=_score, reverse=True)
                cands = cands[: int(cand_n)]

                # 参数：保守一点（只做“撤退惩罚”）
                penalty_w = 0.12
                ttl_h = float(cache_ttl_hours)

                def _parse_as_of(it0: dict[str, Any]) -> _date:
                    try:
                        meta = it0.get("meta") if isinstance(it0.get("meta"), dict) else {}
                        s = str(meta.get("as_of") or "").strip()
                        d = _datetime.strptime(s, "%Y-%m-%d").date() if s else None
                    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                        d = None
                    return d if d is not None else _date.today()

                def _clip01(x: float) -> float:
                    return float(max(0.0, min(1.0, float(x))))

                def _run_one(it0: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                    sym = str(it0.get("symbol") or "").strip().lower()
                    as_of_d = _parse_as_of(it0)
                    micro = compute_stock_microstructure_tushare(
                        as_of=as_of_d,
                        symbol_prefixed=sym,
                        daily_amount_by_date=None,  # 扫描期不做归一化，避免重复拉 K 线；z/score01 用历史稳健化
                        cache_dir=Path("data") / "cache" / "tushare_factors" / "micro",
                        ttl_hours=float(ttl_h),
                        lookback_days=60,
                    )
                    return sym, micro

                ff_by_sym: dict[str, dict[str, Any]] = {}
                # 线程别开太大：TuShare 有限流；这里追求“稳”，不是极限速度。
                ff_workers = max(1, min(4, int(workers)))
                with ThreadPoolExecutor(max_workers=int(ff_workers)) as ex2:
                    futs2 = {ex2.submit(_run_one, it): it for it in cands}
                    for fut2 in as_completed(futs2):
                        try:
                            sym, micro = fut2.result()
                        except Exception:  # noqa: BLE001
                            continue
                        if sym:
                            ff_by_sym[str(sym)] = micro if isinstance(micro, dict) else {"ok": False, "error": "microstructure_invalid"}

                # apply penalty
                applied = 0
                for it in items:
                    sym = str(it.get("symbol") or "").strip().lower()
                    if not sym or sym not in ff_by_sym:
                        continue
                    micro = ff_by_sym.get(sym) or {}
                    if not isinstance(micro, dict):
                        continue

                    # 保留 base score（便于复盘）
                    base_sc = _score(it)
                    flow = None
                    try:
                        if bool(micro.get("ok")):
                            flow = micro.get("score01")
                    except (AttributeError):  # noqa: BLE001
                        flow = None
                    try:
                        flow01 = 0.5 if flow is None else float(flow)
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        flow01 = 0.5
                    flow01 = _clip01(flow01)

                    # 惩罚：只对“低于中性(0.5)”的部分扣分
                    penalty = float(penalty_w) * max(0.0, 0.5 - float(flow01))
                    sc2 = _clip01(float(base_sc) - float(penalty))

                    meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
                    meta["score_base"] = float(base_sc)
                    meta["score_adjusted"] = float(sc2)
                    meta["fund_flow"] = {
                        "ok": bool(micro.get("ok")),
                        "ref_date": micro.get("ref_date"),
                        "z": micro.get("z"),
                        "score01": micro.get("score01"),
                        "penalty": float(penalty),
                        "note": "stock_fund_flow_penalty: max(0,0.5-score01)*w",
                    }
                    it["meta"] = meta
                    it["score"] = float(sc2)
                    applied += 1

                    # action 也跟着修正（避免“看起来 entry 但 score 已被打回”）
                    try:
                        base_action = str(it.get("action") or "").strip().lower()
                        meta["action_base"] = base_action
                        if base_action != "avoid":
                            if float(sc2) >= float(cfg.entry_threshold):
                                it["action"] = "entry"
                            elif float(sc2) <= float(cfg.exit_threshold):
                                it["action"] = "exit"
                            else:
                                it["action"] = "watch"
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        pass

                post_process["stock_fund_flow_penalty"] = {
                    "enabled": True,
                    "applied": int(applied),
                    "candidates": int(cand_n),
                    "weight": float(penalty_w),
                    "ttl_hours": float(ttl_h),
                    "note": "仅对候选集取 TuShare moneyflow(大单+超大单) 的 score01；低于0.5才扣分；ETF 不强绑。",
                }
        except Exception:  # noqa: BLE001
            # 可选增强：失败就跳过，别影响主流程
            pass

    # sort & top_k
    items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    if len(items) > int(top_k):
        items = items[: int(top_k)]
    output_n = int(len(items))

    # as_of：取 items 里最大的（YYYY-MM-DD 可字典序比较）
    as_of = None
    try:
        dates = []
        for it in items:
            meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
            s = str(meta.get("as_of") or "").strip()
            if s:
                dates.append(s)
        as_of = max(dates) if dates else None
    except (AttributeError):  # noqa: BLE001
        as_of = None

    out = sanitize_for_json(
        {
            "schema_version": 1,
            "generated_at": datetime.now().isoformat(),
            "as_of": as_of,
            "strategy": f"config:{strategy_key}",
            "source": {"type": "scan-strategy", "file": "signals.json"},
            "market_regime": regime_dict,
            "config": {
                "strategy_key": str(strategy_key),
                "strategy_config": str(cfg_path),
                "strategy": {
                    "name": cfg.name,
                    "description": cfg.description,
                    "factor_weights": cfg.factor_weights,
                    "factor_params": cfg.factor_params,
                    "entry_threshold": cfg.entry_threshold,
                    "exit_threshold": cfg.exit_threshold,
                    "require_factors": cfg.require_factors,
                    "exclude_factors": cfg.exclude_factors,
                    "allowed_regimes": cfg.allowed_regimes,
                },
                "universe": uni_raw or ("etf_all" if (asset == "etf" and include_all_funds) else asset),
                "limit": int(limit) if limit > 0 else None,
                "top_k": int(top_k),
                "min_score": float(min_score) if min_score > 0 else None,
                "whitelist": str(getattr(args, "whitelist", "") or "").strip() or None,
                "blacklist": str(getattr(args, "blacklist", "") or "").strip() or None,
                "freq": str(freq),
                "window": int(window),
                "cache_dir": str(cache_dir),
                "cache_ttl_hours": float(cache_ttl_hours),
                "source": str(source),
                "workers": int(workers),
                "regime_index": str(regime_index_eff or regime_index),
                "regime_canary": bool(regime_canary),
                "regime_error": regime_error,
                "post_process": post_process or None,
            },
            "counts": {
                "items": int(output_n),  # backward compatible: 输出 items 数
                "computed": int(computed_n),  # 实际算出来的（未截断 top_k 前）
                "skipped": int(max(0, int(total) - int(computed_n))),  # fetch/样本不足/过滤导致的 None
                "universe": int(total),
            },
            "items": items,
        }
    )
    write_json(out_dir / "signals.json", out)
    _write_run_meta(out_dir, args, extra={"cmd": "scan-strategy"})
    _write_run_config(out_dir, args, note="scan strategy config", extra={"cmd": "scan-strategy"})

    print(str(out_dir.resolve()))
    return 0


def cmd_strategy_align(args: argparse.Namespace) -> int:
    """
    Phase3：新旧信号对齐报告（对齐=可量化，别靠嘴）。

    输入：两份 signals.json（schema_version=1）
    输出：alignment.json + mismatches.csv（可 SQL/可审计）
    """
    import csv
    import json

    base_path = Path(str(getattr(args, "base", "") or "").strip())
    new_path = Path(str(getattr(args, "new", "") or "").strip())
    if not base_path.exists():
        raise SystemExit(f"找不到 --base：{base_path}")
    if not new_path.exists():
        raise SystemExit(f"找不到 --new：{new_path}")

    out_dir_raw = str(getattr(args, "out_dir", "") or "").strip()
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"strategy_alignment_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(p: Path) -> dict[str, Any]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (AttributeError) as exc:  # noqa: BLE001
            raise SystemExit(f"读取 JSON 失败：{p} {exc}") from exc

    base = _load(base_path)
    new = _load(new_path)

    def _is_signals(obj: Any) -> bool:
        return bool(isinstance(obj, dict) and int(obj.get("schema_version") or 0) == 1 and isinstance(obj.get("items"), list))

    if not _is_signals(base):
        raise SystemExit(f"--base 不是 signals schema_version=1：{base_path}")
    if not _is_signals(new):
        raise SystemExit(f"--new 不是 signals schema_version=1：{new_path}")

    base_items = base.get("items") if isinstance(base, dict) else []
    new_items = new.get("items") if isinstance(new, dict) else []

    def _map(items0: list[Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for it in items0:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol") or "").strip()
            if not sym:
                continue
            out[sym] = it
        return out

    base_map = _map(base_items)
    new_map = _map(new_items)

    syms = sorted(set(base_map) | set(new_map))
    if not syms:
        raise SystemExit("两份 signals 都是空的，别闹。")

    def _act(it: dict[str, Any] | None) -> str:
        if not it:
            return "missing"
        return str(it.get("action") or "missing")

    def _score(it: dict[str, Any] | None) -> float | None:
        if not it:
            return None
        return _fnum(it.get("score"))

    # confusion matrix（entry vs non-entry）
    tp = fp = fn = tn = 0
    mismatches: list[dict[str, Any]] = []

    for sym in syms:
        b = base_map.get(sym)
        n = new_map.get(sym)
        ba = _act(b)
        na = _act(n)

        be = bool(ba == "entry")
        ne = bool(na == "entry")
        if be and ne:
            tp += 1
        elif (not be) and ne:
            fp += 1
        elif be and (not ne):
            fn += 1
        else:
            tn += 1

        if ba != na:
            mismatches.append(
                {
                    "symbol": sym,
                    "base_action": ba,
                    "new_action": na,
                    "base_score": _score(b),
                    "new_score": _score(n),
                }
            )

    total = len(syms)
    mismatch_rate = float(len(mismatches) / total) if total > 0 else 0.0

    # top-k overlap（只看 entry；按 score 降序）
    top_k = int(getattr(args, "top_k", 30) or 30)
    top_k = max(1, min(top_k, 5000))

    def _top_entry(m: dict[str, dict[str, Any]]) -> list[str]:
        pairs: list[tuple[str, float]] = []
        for sym, it in m.items():
            if str(it.get("action") or "") != "entry":
                continue
            sc = _fnum(it.get("score"))
            if sc is None:
                continue
            pairs.append((sym, float(sc)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in pairs[: int(top_k)]]

    base_top = _top_entry(base_map)
    new_top = _top_entry(new_map)
    inter = sorted(set(base_top) & set(new_top))
    overlap = float(len(inter) / max(1, min(len(base_top), len(new_top)))) if (base_top and new_top) else 0.0

    report = sanitize_for_json(
        {
            "schema": "llm_trading.strategy_alignment.v1",
            "generated_at": datetime.now().isoformat(),
            "base": {"file": str(base_path), "strategy": base.get("strategy"), "as_of": base.get("as_of")},
            "new": {"file": str(new_path), "strategy": new.get("strategy"), "as_of": new.get("as_of")},
            "universe": {"symbols": int(total)},
            "entry_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "mismatch": {"count": int(len(mismatches)), "rate": float(mismatch_rate)},
            "top_k": {"k": int(top_k), "base_entry": int(len(base_top)), "new_entry": int(len(new_top)), "overlap_rate": float(overlap), "overlap_symbols": inter},
        }
    )
    write_json(out_dir / "alignment.json", report)

    # csv：只写 mismatch（便于 DuckDB/Excel）
    csv_path = out_dir / "mismatches.csv"
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "base_action", "new_action", "base_score", "new_score"])
            w.writeheader()
            for row in mismatches:
                w.writerow(row)
    except (AttributeError):  # noqa: BLE001
        pass

    _write_run_meta(out_dir, args, extra={"cmd": "strategy-align"})
    _write_run_config(out_dir, args, note="strategy alignment", extra={"cmd": "strategy-align"})

    print(str(out_dir.resolve()))
    return 0
