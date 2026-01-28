from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any


class EtfHoldingsError(RuntimeError):
    pass


def _read_csv_if_fresh(path: Path, *, ttl_hours: float):
    """
    轻量 CSV 缓存读取：ETF 持仓（季度披露）更新频率低，避免每次都去抓网页。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise EtfHoldingsError("缺依赖：pandas 未安装") from exc

    if (not path.exists()) or float(ttl_hours) <= 0:
        return None
    try:
        age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
        if age > float(ttl_hours):
            return None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None


def _write_csv_silent(df, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (AttributeError):  # noqa: BLE001
        return
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        pass


def _parse_quarter_label(s: str) -> tuple[int, int] | None:
    """
    解析东方财富“季度”字段：例如 2025年4季度股票投资明细 -> (2025,4)
    """
    t = str(s or "").strip()
    if not t:
        return None
    m = re.search(r"(?P<y>\d{4})年(?P<q>[1-4])季度", t)
    if not m:
        return None
    try:
        y = int(m.group("y"))
        q = int(m.group("q"))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None
    if q not in {1, 2, 3, 4}:
        return None
    return y, q


def _quarter_end_date(y: int, q: int) -> date:
    if q == 1:
        return date(int(y), 3, 31)
    if q == 2:
        return date(int(y), 6, 30)
    if q == 3:
        return date(int(y), 9, 30)
    return date(int(y), 12, 31)


def _guess_cn_stock_symbol(code6: str) -> str | None:
    """
    6 位股票代码 -> 前缀形式：
    - 6xxxxxx -> sh
    - 0/3xxxxxx -> sz
    - 4/8xxxxxx -> bj（粗略兜底，够用就行）
    """
    c = str(code6 or "").strip()
    if not (len(c) == 6 and c.isdigit()):
        return None
    if c.startswith("6"):
        return f"sh{c}"
    if c.startswith(("0", "3")):
        return f"sz{c}"
    if c.startswith(("4", "8", "9")):
        return f"bj{c}"
    return None


def _prefixed_stock_symbol_to_ts_code(sym: str) -> str | None:
    s = str(sym or "").strip().lower()
    if len(s) != 8:
        return None
    if not s.startswith(("sh", "sz", "bj")):
        return None
    code = s[2:]
    if not (len(code) == 6 and code.isdigit()):
        return None
    return f"{code}.{s[:2].upper()}"


def fetch_etf_top_holdings_em(
    *,
    symbol: str,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float = 24.0 * 14,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    ETF/基金“前十大重仓”（季度披露，研究用途）：
    - 数据源：AkShare -> 天天基金/东方财富 fundf10 页面（fund_portfolio_hold_em）
    - 典型用途：解释“基金为什么会受某个成分事件影响”（例如 TikTok 事件对蓝色光标）

    注意：
    - 披露频率低（季度）；不存在“实时持仓”。
    - 东方财富页面里的“持股数/持仓市值”常见单位为 万股/万元；我们保留原口径并在字段名里标注。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise EtfHoldingsError("缺依赖：pandas 未安装") from exc

    from .akshare_source import DataSourceError, resolve_symbol

    sym_prefixed = resolve_symbol("etf", str(symbol))
    fund_code = sym_prefixed[2:]

    n = int(top_n or 10)
    n = max(1, min(n, 50))

    years = []
    try:
        years = [int(as_of.year), int(as_of.year) - 1]
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        years = []
    if not years:
        years = [datetime.now().year, datetime.now().year - 1]

    df_all = None
    used_year = None
    last_err = None

    cache_dir.mkdir(parents=True, exist_ok=True)

    for y in years:
        path = cache_dir / f"fund_portfolio_hold_em_{fund_code}_{y}.csv"
        df = _read_csv_if_fresh(path, ttl_hours=float(ttl_hours))
        if df is None:
            try:
                import akshare as ak

                df = ak.fund_portfolio_hold_em(symbol=str(fund_code), date=str(int(y)))
            except Exception as exc:  # noqa: BLE001
                df = None
                last_err = str(exc)
            if df is not None and (not getattr(df, "empty", True)):
                _write_csv_silent(df, path)

        if df is not None and (not getattr(df, "empty", True)):
            df_all = df
            used_year = int(y)
            break

    if df_all is None or getattr(df_all, "empty", True):
        return {
            "schema": "llm_trading.etf_holdings_top10.v1",
            "ok": False,
            "asset": "etf",
            "symbol": str(sym_prefixed),
            "fund_code": str(fund_code),
            "as_of": None,
            "report_period": None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": {"name": "akshare", "func": "fund_portfolio_hold_em"},
            "rows": [],
            "error": (last_err or "fund_portfolio_hold_em 返回空（可能该年份无披露/接口抽风）"),
        }

    need_cols = ["股票代码", "股票名称", "占净值比例", "持股数", "持仓市值", "季度"]
    for c in need_cols:
        if c not in df_all.columns:
            return {
                "schema": "llm_trading.etf_holdings_top10.v1",
                "ok": False,
                "asset": "etf",
                "symbol": str(sym_prefixed),
                "fund_code": str(fund_code),
                "as_of": None,
                "report_period": None,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": {"name": "akshare", "func": "fund_portfolio_hold_em", "year": used_year},
                "rows": [],
                "error": f"fund_portfolio_hold_em 缺少列：{c}",
                "columns": [str(x) for x in list(df_all.columns)],
            }

    dfx = df_all.copy()
    dfx["_yq_key"] = None
    dfx["_y"] = None
    dfx["_q"] = None
    for i in range(len(dfx)):
        ql = dfx.iloc[i].get("季度")
        pq = _parse_quarter_label(str(ql))
        if pq is None:
            continue
        yy, qq = pq
        dfx.at[i, "_y"] = int(yy)
        dfx.at[i, "_q"] = int(qq)
        dfx.at[i, "_yq_key"] = int(yy) * 10 + int(qq)

    dfx2 = dfx.dropna(subset=["_yq_key"]).copy()
    if dfx2.empty:
        return {
            "schema": "llm_trading.etf_holdings_top10.v1",
            "ok": False,
            "asset": "etf",
            "symbol": str(sym_prefixed),
            "fund_code": str(fund_code),
            "as_of": None,
            "report_period": None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": {"name": "akshare", "func": "fund_portfolio_hold_em", "year": used_year},
            "rows": [],
            "error": "无法从“季度”字段解析披露期（数据口径变化？）",
        }

    try:
        latest_key = int(pd.to_numeric(dfx2["_yq_key"], errors="coerce").dropna().max())
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        latest_key = None
    if latest_key is None:
        return {
            "schema": "llm_trading.etf_holdings_top10.v1",
            "ok": False,
            "asset": "etf",
            "symbol": str(sym_prefixed),
            "fund_code": str(fund_code),
            "as_of": None,
            "report_period": None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": {"name": "akshare", "func": "fund_portfolio_hold_em", "year": used_year},
            "rows": [],
            "error": "季度解析失败：latest_key 为空",
        }

    latest_y = int(latest_key // 10)
    latest_q = int(latest_key % 10)
    rep_as_of = _quarter_end_date(latest_y, latest_q)
    rep_period = f"{latest_y}Q{latest_q}"

    sub = dfx2[dfx2["_yq_key"] == latest_key].copy()
    sub["weight_pct"] = pd.to_numeric(sub["占净值比例"], errors="coerce")
    sub["shares_wan"] = pd.to_numeric(sub["持股数"], errors="coerce")
    sub["hold_value_wan"] = pd.to_numeric(sub["持仓市值"], errors="coerce")
    sub = sub.dropna(subset=["股票代码", "股票名称"]).copy()
    sub = sub.sort_values("weight_pct", ascending=False).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for idx, r in sub.head(n).iterrows():
        code = str(r.get("股票代码") or "").strip()
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)
        stock_sym = _guess_cn_stock_symbol(code)
        ts_code = _prefixed_stock_symbol_to_ts_code(stock_sym) if stock_sym else None

        w_pct = r.get("weight_pct")
        w_pct2 = float(w_pct) if w_pct is not None else None
        w = (float(w_pct2) / 100.0) if (w_pct2 is not None) else None

        rows.append(
            {
                "rank": int(idx) + 1,
                "stock_code": code if code else None,
                "stock_symbol": stock_sym,
                "stock_ts_code": ts_code,
                "stock_name": str(r.get("股票名称") or "").strip() or None,
                "weight_pct": w_pct2,
                "weight": w,
                "shares_wan": (None if pd.isna(r.get("shares_wan")) else float(r.get("shares_wan"))),
                "hold_value_wan": (None if pd.isna(r.get("hold_value_wan")) else float(r.get("hold_value_wan"))),
            }
        )

    return {
        "schema": "llm_trading.etf_holdings_top10.v1",
        "ok": True,
        "asset": "etf",
        "symbol": str(sym_prefixed),
        "fund_code": str(fund_code),
        "as_of": str(rep_as_of),
        "report_period": str(rep_period),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"name": "akshare", "func": "fund_portfolio_hold_em", "year": used_year},
        "rows": rows,
        "note": "季度披露：持股数/持仓市值常见单位为万股/万元；仅用于研究解释，不代表实时持仓。",
    }
