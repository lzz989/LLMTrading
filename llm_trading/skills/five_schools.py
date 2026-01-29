from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..akshare_source import DataSourceError, FetchParams, fetch_daily
from ..chanlun import compute_chanlun_structure
from ..indicators import (
    add_accumulation_distribution_line,
    add_atr,
    add_donchian_channels,
    add_ichimoku,
    add_moving_averages,
)
from ..pipeline import write_json
from ..resample import resample_to_weekly
from ..vsa import compute_vsa_report


School = Literal["wyckoff", "chan", "ichimoku", "turtle", "vsa"]
Stance = Literal["bull", "bear", "neutral"]


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _as_bool(x: Any) -> bool:
    return bool(x)


def _as_float(x: Any) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None
    return None if v != v else float(v)  # NaN guard


def _fmt_price(x: Any) -> str:
    v = _as_float(x)
    if v is None:
        return "?"
    # A 股/ETF 习惯 2~3 位；这里先 3 位，避免价格低的 ETF 全变 0.00
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def _load_leaders_cfg(path: Path | None = None) -> dict[str, Any]:
    p = path or (Path("prompts") / "five_schools_leaders.yaml")
    try:
        import yaml

        obj = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:  # noqa: BLE001
        obj = None
    root = _as_dict(obj)
    leaders = _as_dict(root.get("leaders"))
    return leaders


def _leader_name(leaders: dict[str, Any], school: str) -> str:
    d = _as_dict(leaders.get(str(school)))
    nm = _as_str(d.get("name")).strip()
    if nm:
        return nm
    # fallback（别让输出空着）
    return {
        "wyckoff": "庄线老炮",
        "chan": "老缠",
        "ichimoku": "云图道长",
        "turtle": "军规官",
        "vsa": "量价刑警",
    }.get(str(school), "教主")


def _leader_template(leaders: dict[str, Any], school: str, stance: Stance) -> str:
    d = _as_dict(leaders.get(str(school)))
    t = _as_dict(d.get("templates"))
    s = _as_str(t.get(str(stance))).strip()
    if s:
        return s
    return "{key}不破就行，破了就撤。"


def _extract_names_from_run_dir(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None or (not run_dir.exists()):
        return {}

    out: dict[str, str] = {}
    for fn in ["holdings_user.json", "signals.json", "signals_stock.json", "signals_left.json", "signals_left_stock.json"]:
        obj = _read_json(run_dir / fn) or {}
        holds = obj.get("holdings") if isinstance(obj.get("holdings"), list) else None
        items = holds if holds is not None else (obj.get("items") if isinstance(obj.get("items"), list) else [])
        for it in _as_list(items):
            d = _as_dict(it)
            sym = _as_str(d.get("symbol")).strip()
            name = _as_str(d.get("name")).strip()
            if sym and name and sym not in out:
                out[sym] = name
    return out


def _find_analyze_dir_in_run_dir(*, run_dir: Path, asset: str, symbol: str) -> Path | None:
    """
    复用 run --deep-holdings 产物：
    - outputs/run_*/holdings_deep_summary.json 里会记录 analyze_dir（相对 run_dir）
    - 或者直接猜目录：run_dir/holdings_deep/{asset}_{symbol}
    """
    # 1) summary 优先
    summ = _read_json(run_dir / "holdings_deep_summary.json")
    items = _as_list((_as_dict(summ).get("items")))
    for it in items:
        d = _as_dict(it)
        if str(d.get("asset") or "").strip().lower() != str(asset).strip().lower():
            continue
        if str(d.get("symbol") or "").strip().lower() != str(symbol).strip().lower():
            continue
        rel = str(d.get("analyze_dir") or "").strip()
        if not rel:
            continue
        p = (run_dir / rel).resolve()
        if p.exists() and p.is_dir():
            return p

    # 2) 兜底：猜 deep 目录
    p2 = (run_dir / "holdings_deep" / f"{asset}_{symbol}").resolve()
    if p2.exists() and p2.is_dir():
        return p2
    return None


def _load_school_evidence_from_analyze_dir(analyze_dir: Path, school: School) -> dict[str, Any] | None:
    file_map = {
        "wyckoff": "wyckoff_features.json",
        "chan": "chan_structure.json",
        "ichimoku": "ichimoku.json",
        "turtle": "turtle.json",
        "vsa": "vsa_features.json",
    }
    name = str(school)
    sub = analyze_dir / name
    base = sub if sub.exists() and sub.is_dir() else analyze_dir
    fp = base / file_map[name]
    return _read_json(fp)


def _compute_wyckoff_evidence(df) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    df2 = add_moving_averages(df, ma_fast=50, ma_slow=200)
    df2 = add_accumulation_distribution_line(df2)
    last = df2.iloc[-1]

    close = _as_float(last.get("close"))
    ma50 = _as_float(last.get("ma50"))
    ma200 = _as_float(last.get("ma200"))
    ad_line = _as_float(last.get("ad_line"))

    ret_4 = None
    ret_12 = None
    if close is not None and len(df2) >= 5:
        ret_4 = _as_float(close / float(df2.iloc[-5]["close"]) - 1.0)
    if close is not None and len(df2) >= 13:
        ret_12 = _as_float(close / float(df2.iloc[-13]["close"]) - 1.0)

    ad_delta_20 = None
    if ad_line is not None and "ad_line" in df2.columns and len(df2) >= 21:
        ad_delta_20 = _as_float(ad_line - float(df2.iloc[-21]["ad_line"]))

    try:
        last_date = df2.iloc[-1]["date"]
        last_date_s = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
    except Exception:  # noqa: BLE001
        last_date_s = ""

    return {
        "method": "wyckoff_features",
        "last": {"date": last_date_s, "close": close, "ma50": ma50, "ma200": ma200, "ad_line": ad_line},
        "derived": {
            "close_vs_ma200": _as_float(close - ma200) if (close is not None and ma200 is not None) else None,
            "ma50_vs_ma200": _as_float(ma50 - ma200) if (ma50 is not None and ma200 is not None) else None,
            "ret_4": ret_4,
            "ret_12": ret_12,
            "ad_delta_20": ad_delta_20,
        },
    }


def _compute_ichimoku_evidence(df) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    df2 = add_ichimoku(df, tenkan=9, kijun=26, span_b=52, displacement=26)
    last = df2.iloc[-1]

    close = _as_float(last.get("close"))
    tenkan = _as_float(last.get("ichimoku_tenkan"))
    kijun = _as_float(last.get("ichimoku_kijun"))
    span_a = _as_float(last.get("ichimoku_span_a_raw")) if "ichimoku_span_a_raw" in df2.columns else _as_float(last.get("ichimoku_span_a"))
    span_b = _as_float(last.get("ichimoku_span_b_raw")) if "ichimoku_span_b_raw" in df2.columns else _as_float(last.get("ichimoku_span_b"))

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
    if len(df2) >= 2:
        prev = df2.iloc[-2]
        try:
            prev_diff = float(prev["ichimoku_tenkan"]) - float(prev["ichimoku_kijun"])
            cur_diff = float(last["ichimoku_tenkan"]) - float(last["ichimoku_kijun"])
            if prev_diff <= 0 < cur_diff:
                tk_cross = "bullish"
            elif prev_diff >= 0 > cur_diff:
                tk_cross = "bearish"
        except Exception:  # noqa: BLE001
            tk_cross = "unknown"

    try:
        last_date = last.get("date")
        last_date_s = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
    except Exception:  # noqa: BLE001
        last_date_s = ""

    return {
        "method": "ichimoku",
        "params": {"tenkan": 9, "kijun": 26, "span_b": 52, "displacement": 26},
        "last": {
            "date": last_date_s,
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


def _compute_turtle_evidence(df) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    df2 = add_donchian_channels(df, window=20, upper_col="donchian_entry_upper", lower_col="donchian_entry_lower", shift=1)
    df2 = add_donchian_channels(df2, window=10, upper_col="donchian_exit_upper", lower_col="donchian_exit_lower", shift=1)
    df2 = add_atr(df2, period=20, out_col="atr")
    last = df2.iloc[-1]

    close = _as_float(last.get("close"))
    entry_u = _as_float(last.get("donchian_entry_upper"))
    entry_l = _as_float(last.get("donchian_entry_lower"))
    exit_u = _as_float(last.get("donchian_exit_upper"))
    exit_l = _as_float(last.get("donchian_exit_lower"))
    atr = _as_float(last.get("atr"))

    long_entry = bool(close is not None and entry_u is not None and close > entry_u)
    long_exit = bool(close is not None and exit_l is not None and close < exit_l)
    short_entry = bool(close is not None and entry_l is not None and close < entry_l)
    short_exit = bool(close is not None and exit_u is not None and close > exit_u)

    stop_atr = 2.0
    long_stop = _as_float(close - stop_atr * atr) if (close is not None and atr is not None) else None
    short_stop = _as_float(close + stop_atr * atr) if (close is not None and atr is not None) else None

    try:
        last_date = last.get("date")
        last_date_s = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
    except Exception:  # noqa: BLE001
        last_date_s = ""

    return {
        "method": "turtle",
        "params": {"entry": 20, "exit": 10, "atr": 20, "stop_atr": stop_atr},
        "last": {
            "date": last_date_s,
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
        "risk": {"long_stop": long_stop, "short_stop": short_stop},
    }


def _compute_vsa_evidence(df) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    _df_feat, report = compute_vsa_report(df, vol_window=20, spread_window=20, lookback_events=120)
    # analyze 写的是 vsa_features.json，但结构体里 method= vsa；这里直接复用 report
    return report if isinstance(report, dict) else None


def _compute_school_evidence_from_df(df, school: School) -> dict[str, Any] | None:
    if school == "wyckoff":
        return _compute_wyckoff_evidence(df)
    if school == "chan":
        # 缠论：min_gap 取默认 4（偏稳）
        try:
            return compute_chanlun_structure(df, min_gap=4)
        except Exception:  # noqa: BLE001
            return None
    if school == "ichimoku":
        return _compute_ichimoku_evidence(df)
    if school == "turtle":
        return _compute_turtle_evidence(df)
    if school == "vsa":
        return _compute_vsa_evidence(df)
    return None


def _as_of_from_evidence(evi: dict[str, Any] | None) -> str | None:
    if not evi:
        return None
    # 优先找 last.date / summary.last_date
    last = _as_dict(evi.get("last"))
    d0 = _as_str(last.get("date")).strip()
    if d0:
        return d0
    summ = _as_dict(evi.get("summary"))
    d1 = _as_str(summ.get("last_date")).strip()
    return d1 or None


def render_school_quick_review(
    *,
    symbol: str,
    name: str,
    school: School,
    evidence: dict[str, Any] | None,
    leaders_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    输出：每派 1 句话 + 1 条失效（可验证价位/结构锚点）。
    注意：这是“快评筛子”，不是完整研报；宁可保守也别瞎编。
    """
    leader = _leader_name(leaders_cfg, str(school))

    stance: Stance = "neutral"
    key_level = None
    key_kind = None
    invalid_rule = None

    close = None
    evidence_lines: list[str] = []

    if not evidence:
        one = f"{leader}：缺数据，别硬上。"
        return {
            "school": str(school),
            "leader": leader,
            "stance": stance,
            "one_liner": one,
            "anchors": {"price_ref": None, "key_level": None, "key_level_kind": None, "evidence": ["evidence_missing"]},
            "invalidation": {"rule": "break_structure", "level": None, "note": "缺数据：无法给出可靠失效位"},
        }

    if school == "wyckoff":
        last = _as_dict(evidence.get("last"))
        derived = _as_dict(evidence.get("derived"))
        close = _as_float(last.get("close"))
        ma200 = _as_float(last.get("ma200"))
        ma50 = _as_float(last.get("ma50"))
        ad_d20 = _as_float(derived.get("ad_delta_20"))

        if close is not None and ma200 is not None and ma50 is not None:
            if close >= ma200 and ma50 >= ma200:
                stance = "bull"
            elif close < ma200 and ma50 < ma200:
                stance = "bear"
            else:
                stance = "neutral"
        key_level = ma200
        key_kind = "ma200"
        invalid_rule = "close_below_level" if stance != "bear" else "close_above_level"

        evidence_lines.append(f"close={_fmt_price(close)} ma200={_fmt_price(ma200)} ma50={_fmt_price(ma50)}")
        if ad_d20 is not None:
            evidence_lines.append(f"ad_delta_20={ad_d20:.2f}")

    elif school == "chan":
        summ = _as_dict(evidence.get("summary"))
        close = _as_float(summ.get("last_close"))
        pos = _as_str(summ.get("position_vs_last_center")).strip().lower()
        last_dir = _as_str(summ.get("last_stroke_direction")).strip().lower()
        c0 = _as_dict(summ.get("last_center"))
        cl = _as_float(c0.get("low"))
        ch = _as_float(c0.get("high"))

        if pos == "above" and last_dir == "up":
            stance = "bull"
            key_level = cl
            key_kind = "center_low"
            invalid_rule = "close_below_level"
        elif pos == "below" and last_dir == "down":
            stance = "bear"
            key_level = ch
            key_kind = "center_high"
            invalid_rule = "close_above_level"
        else:
            stance = "neutral"
            # 中枢内：用中枢低做“赌狗止损”更实战
            key_level = cl
            key_kind = "center_low"
            invalid_rule = "close_below_level"

        evidence_lines.append(f"pos={pos} last_stroke={last_dir}")
        if cl is not None and ch is not None:
            evidence_lines.append(f"center=[{_fmt_price(cl)},{_fmt_price(ch)}]")

    elif school == "ichimoku":
        last = _as_dict(evidence.get("last"))
        close = _as_float(last.get("close"))
        pos = _as_str(last.get("position")).strip().lower()
        tk = _as_str(last.get("tk_cross")).strip().lower()
        cloud_top = _as_float(last.get("cloud_top"))
        cloud_bottom = _as_float(last.get("cloud_bottom"))
        kijun = _as_float(last.get("kijun"))

        if pos == "above" and (tk in {"bullish", "none", "unknown"}):
            stance = "bull"
        elif pos == "below" and (tk in {"bearish", "none", "unknown"}):
            stance = "bear"
        else:
            stance = "neutral"

        # 失效：云上看 cloud_top；云下看 cloud_bottom；云里看 cloud_bottom（偏保命）
        if pos == "above" and cloud_top is not None:
            key_level = cloud_top
            key_kind = "cloud_top"
            invalid_rule = "close_below_level"
        elif pos == "below" and cloud_bottom is not None:
            key_level = cloud_bottom
            key_kind = "cloud_bottom"
            invalid_rule = "close_above_level"
        else:
            key_level = cloud_bottom if cloud_bottom is not None else kijun
            key_kind = "cloud_bottom" if cloud_bottom is not None else "kijun"
            invalid_rule = "close_below_level"

        evidence_lines.append(f"pos={pos} tk_cross={tk}")
        evidence_lines.append(f"cloud_top={_fmt_price(cloud_top)} cloud_bottom={_fmt_price(cloud_bottom)}")

    elif school == "turtle":
        last = _as_dict(evidence.get("last"))
        sig = _as_dict(evidence.get("signals"))
        risk = _as_dict(evidence.get("risk"))
        close = _as_float(last.get("close"))
        exit_l = _as_float(last.get("donchian_exit_lower"))
        exit_u = _as_float(last.get("donchian_exit_upper"))
        long_stop = _as_float(risk.get("long_stop"))

        long_entry = _as_bool(sig.get("long_entry_breakout"))
        long_exit = _as_bool(sig.get("long_exit_breakdown"))

        if long_entry:
            stance = "bull"
        elif long_exit:
            stance = "bear"
        else:
            stance = "neutral"

        # long 失效：exit_l 和 long_stop 取更紧的（更高的那个）
        cand: list[tuple[str, float]] = []
        if exit_l is not None:
            cand.append(("donchian_exit_lower", float(exit_l)))
        if long_stop is not None:
            cand.append(("atr_stop", float(long_stop)))
        if cand:
            kind, lvl = max(cand, key=lambda x: x[1])
            key_kind = kind
            key_level = float(lvl)
            invalid_rule = "close_below_level"
        else:
            key_kind = "donchian_exit_lower"
            key_level = exit_l
            invalid_rule = "close_below_level"

        evidence_lines.append(f"long_entry={bool(long_entry)} long_exit={bool(long_exit)}")
        evidence_lines.append(f"exit_l={_fmt_price(exit_l)} exit_u={_fmt_price(exit_u)} atr_stop={_fmt_price(long_stop)}")

    elif school == "vsa":
        summ = _as_dict(evidence.get("summary"))
        last = _as_dict(evidence.get("last"))
        close = _as_float(last.get("close")) if last else _as_float(summ.get("last_close"))
        vol_level = _as_str(summ.get("vol_level")).strip().lower()
        spread_level = _as_str(summ.get("spread_level")).strip().lower()
        events_n = int(_as_float(summ.get("events")) or 0)

        # 极简判定：放量+大波动 在顶部/底部的意义不同，我们这里先只做“偏多/偏空/不确定”倾向
        if vol_level in {"high", "very_high"} and spread_level == "wide":
            stance = "neutral"  # 放量宽幅：两面性强，快评不强行站队
        elif vol_level in {"high", "very_high"}:
            stance = "bull"
        elif vol_level == "low":
            stance = "neutral"
        else:
            stance = "neutral"

        # 失效：VSA 本质还是结构止损；这里用 close 作为占位（真实使用应结合 MA20/关键低点）
        key_level = close
        key_kind = "close"
        invalid_rule = "break_structure"

        evidence_lines.append(f"vol={vol_level} spread={spread_level} events={events_n}")

    tpl = _leader_template(leaders_cfg, str(school), stance)
    one = tpl.format(key=_fmt_price(key_level))
    # 前缀统一：教主名 + 一句话
    one_liner = f"{leader}：{one}"

    invalid_note = ""
    if invalid_rule == "close_below_level":
        invalid_note = f"收盘跌破{_fmt_price(key_level)}视为失效"
    elif invalid_rule == "close_above_level":
        invalid_note = f"收盘站上{_fmt_price(key_level)}视为失效"
    else:
        invalid_note = "结构走坏视为失效（需结合关键低点/均线确认）"

    return {
        "school": str(school),
        "leader": leader,
        "stance": stance,
        "one_liner": one_liner,
        "anchors": {
            "price_ref": close,
            "key_level": key_level,
            "key_level_kind": key_kind,
            "evidence": evidence_lines[:3],
        },
        "invalidation": {"rule": invalid_rule or "break_structure", "level": key_level, "note": invalid_note},
    }


def render_md(report: dict[str, Any]) -> str:
    dt = _as_str(report.get("generated_at")).strip()
    as_of = _as_str(report.get("as_of")).strip()
    asset = _as_str(report.get("asset")).strip()
    lines: list[str] = []
    lines.append("# 五派快评（教主口吻）\n")
    lines.append(f"- generated_at: {dt}\n")
    if as_of:
        lines.append(f"- as_of: {as_of}\n")
    lines.append(f"- asset: {asset}\n")

    syms = _as_list(report.get("symbols"))
    for s0 in syms:
        s = _as_dict(s0)
        sym = _as_str(s.get("symbol")).strip()
        name = _as_str(s.get("name")).strip() or "名称未知"
        lines.append("\n---\n")
        lines.append(f"## {sym}（{name}）\n")
        schools = _as_list(s.get("schools"))
        for r0 in schools:
            r = _as_dict(r0)
            leader = _as_str(r.get("leader")).strip()
            one = _as_str(r.get("one_liner")).strip()
            inv = _as_dict(r.get("invalidation"))
            inv_note = _as_str(inv.get("note")).strip()
            lines.append(f"- {leader}：{one.replace(leader + '：', '').strip()}；失效：{inv_note}\n")
    return "".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class FiveSchoolsRunResult:
    out_md: Path
    out_json: Path | None
    report: dict[str, Any]


def run_five_schools(
    *,
    asset: str,
    symbols: list[str],
    run_dir: str | None,
    out_md: str,
    out_json: str | None,
    source: str = "auto",
    freq: str = "weekly",
) -> FiveSchoolsRunResult:
    leaders = _load_leaders_cfg()
    run_dir_p = Path(run_dir) if run_dir else None
    names_map = _extract_names_from_run_dir(run_dir_p)

    items_out: list[dict[str, Any]] = []
    as_of_max: str | None = None

    schools: list[School] = ["wyckoff", "chan", "ichimoku", "turtle", "vsa"]

    for sym0 in symbols:
        sym = str(sym0 or "").strip()
        if not sym:
            continue
        name = names_map.get(sym) or "名称未知"

        analyze_dir = None
        if run_dir_p is not None:
            try:
                analyze_dir = _find_analyze_dir_in_run_dir(run_dir=run_dir_p, asset=str(asset), symbol=str(sym))
            except Exception:  # noqa: BLE001
                analyze_dir = None

        # 证据：优先复用 analyze 产物；没有就现算（会触发抓行情）
        ev_map: dict[str, dict[str, Any] | None] = {}
        if analyze_dir is not None:
            for sc in schools:
                ev_map[str(sc)] = _load_school_evidence_from_analyze_dir(analyze_dir, sc)

        need_fetch = any(ev_map.get(str(sc)) is None for sc in schools)
        df = None
        if need_fetch:
            try:
                df = fetch_daily(FetchParams(asset=str(asset), symbol=str(sym), source=str(source)))
                if str(freq).strip().lower() == "weekly":
                    df = resample_to_weekly(df)
            except (DataSourceError) as exc:
                # 抓不到就算了：快评要诚实
                df = None
                # 但也要把错误落到 report 里
                ev_map.setdefault("_fetch_error", {"error": str(exc)})  # type: ignore[arg-type]

        for sc in schools:
            if ev_map.get(str(sc)) is None and df is not None:
                ev_map[str(sc)] = _compute_school_evidence_from_df(df, sc)

        reviews: list[dict[str, Any]] = []
        for sc in schools:
            ev = ev_map.get(str(sc))
            rv = render_school_quick_review(symbol=sym, name=name, school=sc, evidence=ev, leaders_cfg=leaders)
            reviews.append(rv)

            d0 = _as_of_from_evidence(ev)
            if d0 and ((as_of_max is None) or (d0 > as_of_max)):
                as_of_max = d0

        items_out.append({"symbol": sym, "name": name, "schools": reviews})

    report = {
        "schema": "llm_trading.skill.five_schools.v1",
        "generated_at": datetime.now().isoformat(),
        "as_of": as_of_max,
        "asset": str(asset),
        "symbols": items_out,
        "note": "快评默认不调用LLM：用于筛子/复核。缺数据会降级输出，不硬编。",
    }

    out_md_p = Path(out_md)
    out_md_p.parent.mkdir(parents=True, exist_ok=True)
    out_md_p.write_text(render_md(report), encoding="utf-8")

    out_json_p: Path | None = None
    if out_json:
        out_json_p = Path(out_json)
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_json_p, report)

    return FiveSchoolsRunResult(out_md=out_md_p, out_json=out_json_p, report=report)

