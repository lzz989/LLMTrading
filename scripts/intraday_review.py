#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘中持仓快照（研究用途）

用途：
- 基于最近一次 run 的 holdings_user.json（成本/MA20/昨日收盘）；
- 拉取 Sina 分时分钟线（akshare.stock_zh_a_minute），给出盘中快照；
- 输出到 outputs/agents/ 便于复核与复用。

注意：
- 止损/止盈仍按“收盘触发→次日开盘执行（T+1）”口径；盘中只做监控与预警。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"


def _num(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _mult(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}x"


def _ret(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a / b - 1.0)


def _find_latest_run_dir(outputs_dir: Path) -> Path | None:
    cands = [p for p in outputs_dir.glob("run_*") if p.is_dir()]
    if not cands:
        return None
    # 目录名里通常包含 YYYYMMDD 或 YYYYMMDD_xxx，字典序即可。
    return sorted(cands, key=lambda p: p.name)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="指定 run 目录（默认自动取 outputs/run_* 最新）")
    ap.add_argument("--out-dir", default="outputs/agents", help="输出目录（默认 outputs/agents）")
    ap.add_argument("--stop-loss-pct", type=float, default=0.06, help="单票止损比例（默认 0.06=6%）")
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir).strip()) if args.run_dir else _find_latest_run_dir(Path("outputs"))
    if run_dir is None:
        raise SystemExit("没找到 outputs/run_*；请先跑一次 run 产出 holdings_user.json")

    holdings_path = run_dir / "holdings_user.json"
    if not holdings_path.exists():
        raise SystemExit(f"缺少 {holdings_path}；请先跑一次 run")

    obj = json.loads(holdings_path.read_text(encoding="utf-8"))
    now = datetime.now()
    today = now.date()

    # 延迟 import：避免无 akshare/pandas 环境时报错（但本仓库默认有）
    import pandas as pd  # type: ignore
    import akshare as ak  # type: ignore

    items: list[dict[str, Any]] = []
    for h in obj.get("holdings", []):
        sym = str(h.get("symbol") or "")
        name = str(h.get("name") or sym)
        asset = str(h.get("asset") or "")
        cost = _to_float(h.get("cost"))
        shares = int(h.get("shares") or 0)
        prev_close = _to_float(h.get("close"))
        ma20 = _to_float((h.get("levels") or {}).get("daily_ma20"))

        try:
            df = ak.stock_zh_a_minute(symbol=sym, period="1")
        except Exception as e:  # noqa: BLE001
            items.append({"symbol": sym, "name": name, "asset": asset, "ok": False, "error": f"minute fetch failed: {e}"})
            continue

        if df is None or getattr(df, "empty", True):
            items.append({"symbol": sym, "name": name, "asset": asset, "ok": False, "error": "minute data empty"})
            continue

        df2 = df.copy()
        df2["day"] = pd.to_datetime(df2["day"], errors="coerce")
        # Sina 分时返回的数值列常是 object(str)，不转数字会导致 sum 变成字符串拼接（灾难级错误）。
        for c in ("open", "high", "low", "close", "volume"):
            if c in df2.columns:
                df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2 = df2.dropna(subset=["day"]).sort_values("day")
        df_today = df2[df2["day"].dt.date == today]
        if df_today.empty:
            items.append({"symbol": sym, "name": name, "asset": asset, "ok": False, "error": f"no rows for today={today}"})
            continue

        last = df_today.iloc[-1]
        last_dt = last["day"].to_pydatetime() if hasattr(last["day"], "to_pydatetime") else last["day"]
        last_px = _to_float(last.get("close"))
        open_px = _to_float(df_today.iloc[0].get("open"))

        # last 30m window (relative to latest minute)
        w_end = pd.to_datetime(last_dt)
        w_start = w_end - pd.Timedelta(minutes=30)
        df_w = df_today[(df_today["day"] >= w_start) & (df_today["day"] <= w_end)]

        w_ret = None
        if not df_w.empty:
            w_base = _to_float(df_w.iloc[0].get("open"))
            w_ret = _ret(last_px, w_base)

        vol_sum = _to_float(df_w.get("volume").sum()) if (not df_w.empty and "volume" in df_w.columns) else None
        vol_today_sum = _to_float(df_today.get("volume").sum()) if "volume" in df_today.columns else None
        vol_per_min_ratio = None
        if vol_sum is not None and vol_today_sum is not None and len(df_w) > 0 and len(df_today) > 0:
            vol_per_min_ratio = (vol_sum / len(df_w)) / (vol_today_sum / len(df_today))

        # playbook stops/tps (close-only triggers)
        stop_loss = cost * (1.0 - float(args.stop_loss_pct)) if cost is not None else None
        stop_struct = ma20
        if stop_loss is not None and stop_struct is not None:
            stop = max(stop_loss, stop_struct)
        else:
            stop = stop_loss if stop_loss is not None else stop_struct

        tp1 = cost * (1.0 + float(args.stop_loss_pct)) if cost is not None else None  # TP1 uses +6% by default
        tp2 = cost * 1.10 if cost is not None else None

        items.append(
            {
                "asset": asset,
                "symbol": sym,
                "name": name,
                "ok": True,
                "as_of": last_dt.strftime("%Y-%m-%d %H:%M"),
                "price": last_px,
                "open": open_px,
                "prev_close": prev_close,
                "ret_vs_prev_close": _ret(last_px, prev_close),
                "ret_vs_open": _ret(last_px, open_px),
                "last_30m": {
                    "end": w_end.strftime("%Y-%m-%d %H:%M"),
                    "start": w_start.strftime("%Y-%m-%d %H:%M"),
                    "n": int(len(df_w)),
                    "ret": w_ret,
                    "vol_sum": vol_sum,
                    "vol_per_min_ratio": vol_per_min_ratio,
                },
                "position": {"shares": shares, "cost": cost},
                "levels": {"daily_ma20": ma20},
                "playbook": {
                    "stop": stop,
                    "stop_loss_6pct": stop_loss,
                    "stop_struct_ma20": stop_struct,
                    "tp1_6pct": tp1,
                    "tp2_10pct": tp2,
                    "tp1_hit_intraday": (last_px is not None and tp1 is not None and last_px >= tp1),
                    "tp2_hit_intraday": (last_px is not None and tp2 is not None and last_px >= tp2),
                    "risk_position": (stop is not None and cost is not None and stop < cost),
                },
            }
        )

    out = {
        "schema": "llm_trading.intraday_review.v1",
        "generated_at": now.isoformat(timespec="seconds"),
        "market_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_run_dir": str(run_dir),
        "notes": [
            "Intraday snapshot; if market_time < 11:30 it is not a midday close snapshot.",
            "Stop/TP are close-only triggers; intraday hits require end-of-day confirmation.",
            "Minute data source: akshare.stock_zh_a_minute (Sina).",
        ],
        "items": items,
    }

    out_dir = Path(str(args.out_dir)).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_base = f"intraday_review_{today.strftime('%Y%m%d')}_{now.strftime('%H%M')}"
    json_path = out_dir / f"{fn_base}.json"
    md_path = out_dir / f"{fn_base}.md"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# 盘中持仓快照（{today}）\n\n")
    lines.append(f"- generated_at: {now.isoformat(timespec='seconds')}\n")
    lines.append(f"- market_time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    lines.append(f"- based_on: {holdings_path.as_posix()}（昨日收盘基准 + 成本/MA20）\n")
    lines.append("\n> 注意：若当前时间未到 11:30，则这不是‘午盘收盘’时点，只是盘中快照。止损/止盈均按‘收盘触发→次日开盘执行’口径。\n")
    lines.append("\n| 标的 | 现价@时间 | 较昨收 | 较今开 | 近30m | 30m量能(倍数) | 止损线 | TP1(+6%) | TP2(+10%) | 风险仓 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")

    for it in items:
        if not it.get("ok"):
            lines.append(f"| {it.get('symbol')} {it.get('name')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ? |\n")
            continue
        px = _to_float(it.get("price"))
        ts = str(it.get("as_of") or "")
        r1 = _to_float(it.get("ret_vs_prev_close"))
        r2 = _to_float(it.get("ret_vs_open"))
        r30 = _to_float(((it.get("last_30m") or {}) if isinstance(it.get("last_30m"), dict) else {}).get("ret"))
        vr = _to_float(((it.get("last_30m") or {}) if isinstance(it.get("last_30m"), dict) else {}).get("vol_per_min_ratio"))
        pb = it.get("playbook") if isinstance(it.get("playbook"), dict) else {}
        stop = _to_float(pb.get("stop"))
        tp1 = _to_float(pb.get("tp1_6pct"))
        tp2 = _to_float(pb.get("tp2_10pct"))
        risk = bool(pb.get("risk_position"))
        lines.append(
            f"| {it.get('symbol')} {it.get('name')} | {_num(px)}@{ts.split()[-1]} | {_pct(r1)} | {_pct(r2)} | {_pct(r30)} | {_mult(vr)} | {_num(stop)} | {_num(tp1)} | {_num(tp2)} | {'Y' if risk else 'N'} |\n"
        )

    md_path.write_text("".join(lines), encoding="utf-8")
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
