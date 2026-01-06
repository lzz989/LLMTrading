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
    except Exception:  # noqa: BLE001
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


def collect_analysis_bundle(out_dir: Path, *, schools: list[str] | None = None) -> dict[str, Any]:
    schools2 = schools or DEFAULT_SCHOOLS
    out: dict[str, Any] = {"schools": list(schools2)}

    meta = _read_json(out_dir / "meta.json")
    if isinstance(meta, dict):
        out["meta"] = meta
    sb = _read_json(out_dir / "signal_backtest.json")
    if isinstance(sb, dict):
        out["signal_backtest"] = sb

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
    system = "你是一个严谨的交易研究解读助手。你必须用中文输出，不构成投资建议。"
    messages = [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
    return run_llm_text(
        cfg,
        messages=messages,
        provider=provider,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
    )
