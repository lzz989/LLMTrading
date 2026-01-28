from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataDoctorConfig:
    cache_dir: Path
    outputs_dir: Path
    cache_recent_days: int
    cache_max_files: int
    outputs_max_dirs: int
    include_cache: bool = True
    include_outputs: bool = True


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cache_key(path: Path) -> dict[str, str | None]:
    """
    data/cache/<asset>/{asset}_{symbol}_{adjust}.csv
    """
    name = path.name
    asset = path.parent.name
    sym = None
    adj = None
    if name.endswith(".csv"):
        stem = name[: -len(".csv")]
        parts = stem.split("_")
        if len(parts) >= 3:
            # asset_{symbol}_{adjust} 里 symbol 可能自带下划线（极少），所以从两端扣
            # - parts[0] = asset
            # - parts[-1] = adjust
            # - 中间拼回 symbol
            adj = parts[-1]
            sym = "_".join(parts[1:-1]) or None
    return {"asset": asset, "symbol": sym, "adjust": adj}


def _validate_ohlcv_csv(path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover
        return {"path": str(path), "kind": "cache_csv", "status": "error", "errors": ["缺 pandas：先装 requirements.txt"], "warnings": []}

    meta = _parse_cache_key(path)
    out: dict[str, Any] = {"path": str(path), "kind": "cache_csv", **meta, "status": "ok", "errors": [], "warnings": []}

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        out["status"] = "error"
        out["errors"].append(f"CSV 读取失败：{exc}")
        return out

    if df is None or getattr(df, "empty", True):
        out["status"] = "error"
        out["errors"].append("CSV 为空")
        return out

    out["rows"] = int(len(df))
    cols = set(str(c) for c in df.columns)
    out["columns"] = sorted(cols)

    if "date" not in cols:
        out["status"] = "error"
        out["errors"].append("缺 date 列")
        return out
    if "close" not in cols:
        out["status"] = "error"
        out["errors"].append("缺 close 列")
        return out

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    bad_date = int(df2["date"].isna().sum())
    if bad_date > 0:
        out["warnings"].append(f"date 解析失败：{bad_date} 行")
    df2 = df2.dropna(subset=["date"]).reset_index(drop=True)
    if df2.empty:
        out["status"] = "error"
        out["errors"].append("date 全部解析失败（空表）")
        return out

    is_sorted = bool(df2["date"].is_monotonic_increasing)
    if not is_sorted:
        out["warnings"].append("date 非递增（乱序）")
    # duplicates（以 date 为准）
    dup = int(df2.duplicated(subset=["date"]).sum())
    if dup > 0:
        out["warnings"].append(f"date 重复：{dup} 行")

    # 仅用于检查：不改文件；排序后再测 OHLC 逻辑更稳定
    df2 = df2.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    try:
        out["date_min"] = str(df2["date"].iloc[0].date())
        out["date_max"] = str(df2["date"].iloc[-1].date())
    except (KeyError, IndexError, AttributeError):  # noqa: BLE001
        out["date_min"] = None
        out["date_max"] = None

    # 数值列（不存在就跳过）
    def num(col: str):
        if col not in df2.columns:
            return None
        return pd.to_numeric(df2[col], errors="coerce").astype(float)

    open_s = num("open")
    high_s = num("high")
    low_s = num("low")
    close_s = num("close")
    vol_s = num("volume")
    amt_s = num("amount")

    # close 合法性
    if close_s is not None:
        bad_close = int(((close_s <= 0) | close_s.isna()).sum())
        if bad_close > 0:
            out["warnings"].append(f"close<=0 或 NaN：{bad_close} 行")
    else:
        out["status"] = "error"
        out["errors"].append("close 列无法转数值")
        return out

    # OHLC 逻辑（缺列就不硬报错，但会给提示）
    ohlc_err = 0
    if high_s is not None and low_s is not None:
        ohlc_err += int((high_s < low_s).sum())
        if open_s is not None:
            ohlc_err += int(((open_s > high_s) | (open_s < low_s)).sum())
        if close_s is not None:
            ohlc_err += int(((close_s > high_s) | (close_s < low_s)).sum())
        if ohlc_err > 0:
            out["warnings"].append(f"OHLC 不合法（high<low 或 open/close 越界）：{ohlc_err} 行")
    else:
        out["warnings"].append("缺 high/low 列：无法做 OHLC 合法性检查")

    # volume/amount：缺列不算错，但会提示
    if vol_s is None:
        out["warnings"].append("缺 volume 列（停牌/流动性判断会变弱）")
    else:
        vol_zero = int((vol_s.fillna(0.0) == 0.0).sum())
        out["volume_zero_rows"] = vol_zero

    if amt_s is None:
        out["warnings"].append("缺 amount 列（成交额缺失会影响滑点/容量估计）")
    else:
        amt_zero = int((amt_s.fillna(0.0) == 0.0).sum())
        out["amount_zero_rows"] = amt_zero

    # 最后更新时间（文件 mtime）
    try:
        out["mtime"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        out["mtime"] = None

    # status：有 error 就 error；否则有 warning 就 warn
    if out["errors"]:
        out["status"] = "error"
    elif out["warnings"]:
        out["status"] = "warn"
    else:
        out["status"] = "ok"
    return out


def _validate_signals_json(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "kind": "signals_json", "status": "ok", "errors": [], "warnings": []}
    try:
        obj = _read_json(path)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["errors"].append(f"signals.json 解析失败：{exc}")
        return out

    if not isinstance(obj, dict):
        out["status"] = "error"
        out["errors"].append("signals.json 不是 dict")
        return out

    sv = _safe_int(obj.get("schema_version"))
    if sv != 1:
        out["warnings"].append(f"schema_version!=1：{sv}")
    if not obj.get("generated_at"):
        out["warnings"].append("缺 generated_at")
    if not obj.get("strategy"):
        out["warnings"].append("缺 strategy")
    src = obj.get("source")
    if not isinstance(src, dict) or not src.get("type"):
        out["warnings"].append("缺 source.type（数据来源不清晰）")

    items = obj.get("items")
    if not isinstance(items, list):
        out["status"] = "error"
        out["errors"].append("items 不是 list")
        return out

    out["items"] = int(len(items))
    bad = 0
    for it in items[:2000]:  # 别无限扫
        if not isinstance(it, dict):
            bad += 1
            continue
        if not str(it.get("symbol") or "").strip():
            bad += 1
        if not str(it.get("asset") or "").strip():
            bad += 1
        if not str(it.get("action") or "").strip():
            bad += 1
    if bad > 0:
        out["warnings"].append(f"items 基础字段缺失（前2000条里）：{bad} 条")

    if out["errors"]:
        out["status"] = "error"
    elif out["warnings"]:
        out["status"] = "warn"
    else:
        out["status"] = "ok"
    return out


def _validate_run_dir(dir_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(dir_path), "kind": "run_dir", "status": "ok", "errors": [], "warnings": []}
    if not dir_path.is_dir():
        out["status"] = "error"
        out["errors"].append("不是目录")
        return out

    run_meta_p = dir_path / "run_meta.json"
    run_cfg_p = dir_path / "run_config.json"
    if not run_meta_p.exists():
        out["warnings"].append("缺 run_meta.json（可复现性差）")
    if not run_cfg_p.exists():
        out["warnings"].append("缺 run_config.json（缺 params_hash）")

    cmd = None
    params_hash = None
    as_of = None
    if run_cfg_p.exists():
        try:
            rc = _read_json(run_cfg_p)
            if isinstance(rc, dict):
                cmd = str(rc.get("cmd") or "") or None
                params_hash = str(rc.get("params_hash") or "") or None
        except (AttributeError) as exc:  # noqa: BLE001
            out["warnings"].append(f"run_config.json 解析失败：{exc}")
    if run_meta_p.exists():
        try:
            rm = _read_json(run_meta_p)
            if isinstance(rm, dict):
                extra = rm.get("extra") if isinstance(rm.get("extra"), dict) else {}
                as_of = str(extra.get("as_of") or "") or None
                cmd = str(extra.get("cmd") or cmd or "") or None
        except (AttributeError) as exc:  # noqa: BLE001
            out["warnings"].append(f"run_meta.json 解析失败：{exc}")

    out["cmd"] = cmd
    out["params_hash"] = params_hash
    out["as_of"] = as_of
    if not params_hash:
        out["warnings"].append("缺 params_hash（run_config.json）")

    # 常见产物校验（存在就校验）
    sig_p = dir_path / "signals.json"
    if sig_p.exists():
        sig_r = _validate_signals_json(sig_p)
        out.setdefault("children", []).append(sig_r)
        if sig_r["status"] == "error":
            out["warnings"].append("signals.json 校验失败（见 children）")

    if out["errors"]:
        out["status"] = "error"
    elif out["warnings"] or (out.get("children") and any(ch.get("status") == "warn" for ch in out.get("children", []))):
        out["status"] = "warn"
    else:
        out["status"] = "ok"
    return out


def _pick_recent_files(paths: list[Path], *, max_files: int, recent_days: int) -> list[Path]:
    """
    取最近修改的一批文件，避免 data/cache 动辄上万文件把你机器拖死。
    """
    max_n = max(0, int(max_files))
    if max_n <= 0:
        return []

    now_ts = time.time()
    days = max(0, int(recent_days))
    min_ts = now_ts - float(days) * 86400.0 if days > 0 else None

    scored: list[tuple[float, Path]] = []
    for p in paths:
        try:
            st = p.stat()
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            continue
        ts = float(st.st_mtime)
        if min_ts is not None and ts < float(min_ts):
            continue
        scored.append((ts, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:max_n]]


def validate_repo_data(cfg: DataDoctorConfig) -> dict[str, Any]:
    t0 = time.time()
    res: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "ok": True,
        "counts": {"errors": 0, "warnings": 0},
        "config": {
            "cache_dir": str(cfg.cache_dir),
            "outputs_dir": str(cfg.outputs_dir),
            "cache_recent_days": int(cfg.cache_recent_days),
            "cache_max_files": int(cfg.cache_max_files),
            "outputs_max_dirs": int(cfg.outputs_max_dirs),
            "include_cache": bool(cfg.include_cache),
            "include_outputs": bool(cfg.include_outputs),
        },
        "checks": [],
        "notes": [
            "提示：universe 通常来自“当前存量列表”（可能有幸存者偏差）；回测结论别当真理。",
            "提示：信号/回测默认按“下一交易日开盘成交”近似；盘中追涨/一字板/停牌会导致偏差。",
        ],
    }

    errors: list[str] = []
    warnings: list[str] = []

    # 1) cache csv（只查最近一批）
    if cfg.include_cache and cfg.cache_dir.exists():
        all_csv = [p for p in cfg.cache_dir.rglob("*.csv") if p.is_file()]
        picked = _pick_recent_files(all_csv, max_files=int(cfg.cache_max_files), recent_days=int(cfg.cache_recent_days))
        res["cache"] = {"total_csv": int(len(all_csv)), "checked_csv": int(len(picked))}
        for p in picked:
            r = _validate_ohlcv_csv(p)
            res["checks"].append(r)
            if r.get("status") == "error":
                errors.append(str(r.get("path")))
            elif r.get("status") == "warn":
                warnings.append(str(r.get("path")))
    else:
        res["cache"] = {"total_csv": 0, "checked_csv": 0}

    # 2) outputs run dirs（只查最近一批）
    if cfg.include_outputs and cfg.outputs_dir.exists():
        dirs = [p for p in cfg.outputs_dir.iterdir() if p.is_dir()]
        # 按 mtime 降序取前 N
        scored: list[tuple[float, Path]] = []
        for d in dirs:
            try:
                scored.append((float(d.stat().st_mtime), d))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                continue
        scored.sort(key=lambda x: x[0], reverse=True)
        picked_dirs = (
            [p for _, p in scored[: max(0, int(cfg.outputs_max_dirs))]] if int(cfg.outputs_max_dirs) > 0 else [p for _, p in scored]
        )
        res["outputs"] = {"total_dirs": int(len(dirs)), "checked_dirs": int(len(picked_dirs))}
        for d in picked_dirs:
            r = _validate_run_dir(d)
            res["checks"].append(r)
            if r.get("status") == "error":
                errors.append(str(r.get("path")))
            elif r.get("status") == "warn":
                warnings.append(str(r.get("path")))
    else:
        res["outputs"] = {"total_dirs": 0, "checked_dirs": 0}

    # 汇总
    res["counts"]["errors"] = int(len(errors))
    res["counts"]["warnings"] = int(len(warnings))
    if errors:
        res["ok"] = False
        res["errors"] = errors[:200]
    if warnings:
        res["warnings"] = warnings[:200]

    res["elapsed_sec"] = float(max(0.0, time.time() - t0))
    return res
