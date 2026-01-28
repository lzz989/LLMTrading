from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


class TushareFactorsError(RuntimeError):
    pass


def _fmt_yyyymmdd(d: date) -> str:
    return f"{d:%Y%m%d}"


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    try:
        import math

        if not math.isfinite(x):
            return None
    except (AttributeError):  # noqa: BLE001
        pass
    return float(x)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def _mad(xs: list[float], *, center: float | None = None) -> float:
    if not xs:
        return 0.0
    c = float(center) if center is not None else _median(xs)
    dev = [abs(float(x) - c) for x in xs]
    return float(_median(dev))


def robust_zscore(x: float | None, history: list[float]) -> float | None:
    """
    稳健 z-score：median/MAD，抗极端值。
    """
    x2 = _safe_float(x)
    if x2 is None:
        return None
    hs = [float(v) for v in history if _safe_float(v) is not None]
    if len(hs) < 8:
        return None
    med = _median(hs)
    mad = _mad(hs, center=med)
    if mad <= 1e-12:
        return 0.0
    return float((x2 - med) / (1.4826 * mad))


def z_to_score01(z: float | None, *, clip: float = 3.0) -> float | None:
    """
    z-score -> [0,1]（z=0 => 0.5；|z|>=clip => 0/1）
    """
    z2 = _safe_float(z)
    if z2 is None:
        return None
    c = max(0.5, float(clip))
    if z2 >= c:
        return 1.0
    if z2 <= -c:
        return 0.0
    return float((z2 + c) / (2.0 * c))


def rolling_robust_zscore_series(
    values: list[float | None],
    *,
    window: int = 60,
    min_history: int = 8,
) -> list[float | None]:
    """
    逐点 rolling 的 robust z-score（median/MAD），严格用“过去窗口”（不含当前点，避免未来函数）。
    - values[i] 用 values[max(0,i-window):i] 做 history
    """
    w = max(5, int(window))
    out: list[float | None] = []
    for i, v in enumerate(values):
        x = _safe_float(v)
        hist = values[max(0, i - w) : i]
        if x is None or len([h for h in hist if _safe_float(h) is not None]) < int(min_history):
            out.append(None)
            continue
        out.append(robust_zscore(x, hist))
    return out


def rolling_score01_series(
    values: list[float | None],
    *,
    window: int = 60,
    min_history: int = 8,
    clip: float = 3.0,
) -> tuple[list[float | None], list[float | None]]:
    """
    返回 (z, score01) 两个序列；score01 用 z_to_score01 映射到 [0,1]。
    """
    zs = rolling_robust_zscore_series(values, window=int(window), min_history=int(min_history))
    sc = [z_to_score01(z, clip=float(clip)) if z is not None else None for z in zs]
    return zs, sc


def _read_csv_if_fresh(path: Path, *, ttl_hours: float):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

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


@dataclass(frozen=True, slots=True)
class ERPProxyConfig:
    # ERP 口径（研究用途）：
    # - equity_yield: 1 / index_pe_ttm
    # - rf_yield: shibor_1y / 100
    # - erp = equity_yield - rf_yield
    #
    # 注意：这不是“教科书10Y国债ERP”，而是“可复现、可落地”的代理指标。
    shibor_tenor: str = "1y"
    lookback_days: int = 90


def compute_cn_gov_bond_10y_yield_proxy(
    *,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """
    10Y 国债收益率（研究用途）：
    - 优先用 AkShare 东方财富口径（稳定、免费）：ak.bond_zh_us_rate(start_date=...)

    返回：
    - value_pct：百分比（1.5=1.5%）
    - yield：小数（0.015=1.5%）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    try:
        import akshare as ak
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：akshare 未安装") from exc

    win = max(20, int(lookback_days or 0))
    start = as_of - timedelta(days=max(400, win * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    path = cache_dir / f"bond_zh_us_rate_{start_s}_{end_s}.csv"

    df = _read_csv_if_fresh(path, ttl_hours=ttl_hours)
    if df is None or getattr(df, "empty", True):
        try:
            # AkShare 这接口内部爱打 tqdm 进度条，别污染我们的 CLI 输出。
            import io
            from contextlib import redirect_stderr, redirect_stdout

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df = ak.bond_zh_us_rate(start_date=str(start_s))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"AkShare bond_zh_us_rate 调用失败：{exc}"}
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "AkShare bond_zh_us_rate 返回空"}
        _write_csv_silent(df, path)

    if "日期" not in df.columns or "中国国债收益率10年" not in df.columns:
        return {"ok": False, "error": "bond_zh_us_rate 缺少列：日期/中国国债收益率10年"}

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["日期"], errors="coerce").dt.date
    dfx["cn10y_pct"] = pd.to_numeric(dfx["中国国债收益率10年"], errors="coerce")
    dfx = dfx.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
    if dfx.empty:
        return {"ok": False, "error": f"bond_zh_us_rate 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    last = dfx.iloc[-1]
    ref_date = last.get("date")
    rf_pct = _safe_float(last.get("cn10y_pct"))
    rf_yield = (float(rf_pct) / 100.0) if rf_pct is not None else None

    hist = [float(x) for x in dfx["cn10y_pct"].tail(win).to_list() if _safe_float(x) is not None]
    z = robust_zscore(rf_pct, hist) if rf_pct is not None else None
    score01 = z_to_score01(z) if z is not None else None

    return {
        "ok": True,
        "as_of": str(as_of),
        "ref_date": str(ref_date),
        "rf": {
            "name": "cn_gov_bond",
            "tenor": "10y",
            "value_pct": rf_pct,
            "yield": rf_yield,
            "z": z,
            "score01": score01,
        },
        "history_days_used": int(len(hist)),
        "cache_path": str(path),
        "source": {"name": "akshare", "func": "bond_zh_us_rate"},
        "note": "10Y国债收益率来自东方财富口径；用于ERP对照与风险温度计。",
    }


def compute_erp_proxy_tushare(
    *,
    as_of: date,
    index_symbol_prefixed: str,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    cfg: ERPProxyConfig | None = None,
) -> dict[str, Any]:
    """
    计算一个 ERP proxy（研究用途）：
    - index pe_ttm：pro.index_dailybasic
    - rf：pro.shibor（默认 1y）
    """
    cfg2 = cfg or ERPProxyConfig()

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, prefixed_symbol_to_ts_code

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（ERP 跳过）"}

    ts_code = prefixed_symbol_to_ts_code(str(index_symbol_prefixed))
    if ts_code is None:
        return {"ok": False, "error": f"index_symbol 解析失败：{index_symbol_prefixed}"}

    win = max(20, int(cfg2.lookback_days))
    start = as_of - timedelta(days=max(30, win))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)

    path_idx = cache_dir / f"index_dailybasic_{ts_code.replace('.', '_')}_{start_s}_{end_s}.csv"
    path_shibor = cache_dir / f"shibor_{start_s}_{end_s}.csv"

    pro = None
    try:
        pro = get_pro_api(env)
    except TushareSourceError as exc:
        return {"ok": False, "error": str(exc)}

    df_idx = _read_csv_if_fresh(path_idx, ttl_hours=ttl_hours)
    if df_idx is None or getattr(df_idx, "empty", True):
        try:
            df_idx = pro.index_dailybasic(ts_code=str(ts_code), start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare index_dailybasic 调用失败：{exc}"}
        if df_idx is None or getattr(df_idx, "empty", True):
            return {"ok": False, "error": "TuShare index_dailybasic 返回空"}
        _write_csv_silent(df_idx, path_idx)

    df_sh = _read_csv_if_fresh(path_shibor, ttl_hours=ttl_hours)
    if df_sh is None or getattr(df_sh, "empty", True):
        try:
            df_sh = pro.shibor(start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare shibor 调用失败：{exc}"}
        if df_sh is None or getattr(df_sh, "empty", True):
            return {"ok": False, "error": "TuShare shibor 返回空"}
        _write_csv_silent(df_sh, path_shibor)

    # index pe_ttm
    if "trade_date" not in df_idx.columns:
        return {"ok": False, "error": "index_dailybasic 缺少 trade_date"}
    dfi = df_idx.copy()
    dfi["trade_date"] = pd.to_datetime(dfi["trade_date"], format="%Y%m%d", errors="coerce")
    dfi = dfi.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    dfi["pe_ttm"] = pd.to_numeric(dfi.get("pe_ttm"), errors="coerce") if "pe_ttm" in dfi.columns else None
    dfi["pe"] = pd.to_numeric(dfi.get("pe"), errors="coerce") if "pe" in dfi.columns else None

    dfi = dfi[dfi["trade_date"].dt.date <= as_of].reset_index(drop=True)
    if dfi.empty:
        return {"ok": False, "error": f"index_dailybasic 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path_idx)}

    last_i = dfi.iloc[-1]
    ref_date_i = last_i.get("trade_date")
    ref_date_i_s = str(ref_date_i.date()) if hasattr(ref_date_i, "date") else None
    pe_ttm = _safe_float(last_i.get("pe_ttm")) if "pe_ttm" in dfi.columns else None
    pe = _safe_float(last_i.get("pe")) if "pe" in dfi.columns else None
    pe_used = pe_ttm if (pe_ttm is not None and pe_ttm > 0) else pe
    equity_yield = (1.0 / float(pe_used)) if (pe_used is not None and pe_used > 0) else None

    # shibor rf（默认 1y）
    if "date" not in df_sh.columns:
        return {"ok": False, "error": "shibor 缺少 date"}
    tenor = str(cfg2.shibor_tenor or "1y").strip().lower() or "1y"
    if tenor not in {"on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"}:
        tenor = "1y"

    dfs = df_sh.copy()
    _raw_date = dfs["date"].copy()
    # TuShare shibor 的 date 常见是 20260123 这种 int/str：必须按 YYYYMMDD 解析，不然会被当成纳秒时间戳（1970 年）
    dfs["date"] = pd.to_datetime(_raw_date, format="%Y%m%d", errors="coerce")
    if dfs["date"].isna().all():
        dfs["date"] = pd.to_datetime(_raw_date, errors="coerce")
    else:
        m = dfs["date"].isna()
        if bool(m.any()):
            dfs.loc[m, "date"] = pd.to_datetime(_raw_date[m], errors="coerce")
    dfs = dfs.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if tenor not in dfs.columns:
        return {"ok": False, "error": f"shibor 缺少列：{tenor}"}
    dfs[tenor] = pd.to_numeric(dfs[tenor], errors="coerce")
    dfs = dfs[dfs["date"].dt.date <= as_of].reset_index(drop=True)
    if dfs.empty:
        return {"ok": False, "error": f"shibor 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path_shibor)}

    last_s = dfs.iloc[-1]
    ref_date_s = last_s.get("date")
    ref_date_s_s = str(ref_date_s.date()) if hasattr(ref_date_s, "date") else None
    rf_pct = _safe_float(last_s.get(tenor))
    rf_yield = (float(rf_pct) / 100.0) if (rf_pct is not None) else None

    erp = (float(equity_yield) - float(rf_yield)) if (equity_yield is not None and rf_yield is not None) else None

    # 10Y 国债 ERP 对照（可选；主要用于“长期口径”对比，不替代 shibor 口径）
    rf10 = None
    erp10 = None
    try:
        rf10 = compute_cn_gov_bond_10y_yield_proxy(
            as_of=as_of,
            cache_dir=cache_dir / "bond10y",
            ttl_hours=float(ttl_hours),
            lookback_days=int(cfg2.lookback_days),
        )
        rf10_y = None
        try:
            rf10_y = (rf10 or {}).get("rf", {}).get("yield")
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            rf10_y = None
        try:
            erp10 = (float(equity_yield) - float(rf10_y)) if (equity_yield is not None and rf10_y is not None) else None
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            erp10 = None
    except (TypeError, ValueError, OverflowError) as exc:  # noqa: BLE001
        rf10 = {"ok": False, "error": str(exc)}
        erp10 = None

    return {
        "ok": True,
        "as_of": str(as_of),
        "index_symbol": str(index_symbol_prefixed),
        "ts_code": str(ts_code),
        "ref_date_index": ref_date_i_s,
        "ref_date_rf": ref_date_s_s,
        "pe_ttm": pe_ttm,
        "pe": pe,
        "pe_used": pe_used,
        "equity_yield": equity_yield,  # 小数：0.08=8%
        "rf": {"name": "shibor", "tenor": tenor, "value_pct": rf_pct, "yield": rf_yield},
        "erp": erp,  # 小数
        "rf_alt_10y": rf10,
        "erp_alt_10y": erp10,  # 小数
        "cache": {"index_dailybasic": str(path_idx), "shibor": str(path_shibor)},
        "source": {"name": "tushare"},
        "note": "ERP proxy=1/index_pe_ttm - shibor_1y；另附10Y国债口径作对照（不替代主口径）。",
    }


def compute_erp_proxy_series_tushare(
    *,
    as_of: date,
    index_symbol_prefixed: str,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 720,
    shibor_tenor: str = "1y",
    z_window: int = 90,
    z_min_history: int = 20,
) -> dict[str, Any]:
    """
    ERP proxy 的“时间序列版”（用于研究/风控温度计）：
    - equity_yield: 1 / index_pe_ttm
    - rf_yield: shibor(tenor)/100
    - erp = equity_yield - rf_yield
    - z/score01：rolling robust zscore（不含当前点，避免未来函数）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, prefixed_symbol_to_ts_code

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（ERP series 跳过）"}

    ts_code = prefixed_symbol_to_ts_code(str(index_symbol_prefixed))
    if ts_code is None:
        return {"ok": False, "error": f"index_symbol 解析失败：{index_symbol_prefixed}"}

    lb = max(60, int(lookback_days or 0))
    # 多取一些用于 rolling history（但别无限扩大）
    start = as_of - timedelta(days=max(400, lb + int(z_window) * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)

    path_idx = cache_dir / f"index_dailybasic_{ts_code.replace('.', '_')}_{start_s}_{end_s}.csv"
    path_shibor = cache_dir / f"shibor_{start_s}_{end_s}.csv"

    try:
        pro = get_pro_api(env)
    except TushareSourceError as exc:
        return {"ok": False, "error": str(exc)}

    df_idx = _read_csv_if_fresh(path_idx, ttl_hours=ttl_hours)
    if df_idx is None or getattr(df_idx, "empty", True):
        try:
            df_idx = pro.index_dailybasic(ts_code=str(ts_code), start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare index_dailybasic 调用失败：{exc}"}
        if df_idx is None or getattr(df_idx, "empty", True):
            return {"ok": False, "error": "TuShare index_dailybasic 返回空"}
        _write_csv_silent(df_idx, path_idx)

    df_sh = _read_csv_if_fresh(path_shibor, ttl_hours=ttl_hours)
    if df_sh is None or getattr(df_sh, "empty", True):
        try:
            df_sh = pro.shibor(start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare shibor 调用失败：{exc}"}
        if df_sh is None or getattr(df_sh, "empty", True):
            return {"ok": False, "error": "TuShare shibor 返回空"}
        _write_csv_silent(df_sh, path_shibor)

    # index pe
    if "trade_date" not in df_idx.columns:
        return {"ok": False, "error": "index_dailybasic 缺少 trade_date"}
    dfi = df_idx.copy()
    dfi["trade_date"] = pd.to_datetime(dfi["trade_date"], format="%Y%m%d", errors="coerce")
    dfi = dfi.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    dfi["pe_ttm"] = pd.to_numeric(dfi.get("pe_ttm"), errors="coerce") if "pe_ttm" in dfi.columns else None
    dfi["pe"] = pd.to_numeric(dfi.get("pe"), errors="coerce") if "pe" in dfi.columns else None
    dfi["date"] = dfi["trade_date"].dt.date
    dfi = dfi[dfi["date"] <= as_of].reset_index(drop=True)
    if dfi.empty:
        return {"ok": False, "error": f"index_dailybasic 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path_idx)}

    # shibor
    if "date" not in df_sh.columns:
        return {"ok": False, "error": "shibor 缺少 date"}
    tenor = str(shibor_tenor or "1y").strip().lower() or "1y"
    if tenor not in {"on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"}:
        tenor = "1y"

    dfs = df_sh.copy()
    _raw_date = dfs["date"].copy()
    dfs["date"] = pd.to_datetime(_raw_date, format="%Y%m%d", errors="coerce")
    if dfs["date"].isna().all():
        dfs["date"] = pd.to_datetime(_raw_date, errors="coerce")
    else:
        m = dfs["date"].isna()
        if bool(m.any()):
            dfs.loc[m, "date"] = pd.to_datetime(_raw_date[m], errors="coerce")
    dfs = dfs.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if tenor not in dfs.columns:
        return {"ok": False, "error": f"shibor 缺少列：{tenor}"}
    dfs[tenor] = pd.to_numeric(dfs[tenor], errors="coerce")
    dfs = dfs.dropna(subset=["date"]).reset_index(drop=True)
    dfs["date_d"] = dfs["date"].dt.date
    dfs = dfs[dfs["date_d"] <= as_of].reset_index(drop=True)
    if dfs.empty:
        return {"ok": False, "error": f"shibor 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path_shibor)}

    # 对齐：trade_date -> 最近一条 shibor（<=trade_date）
    dfi2 = dfi[["trade_date", "date", "pe_ttm", "pe"]].copy()
    dfs2 = dfs[["date", "date_d", tenor]].copy()
    aligned = pd.merge_asof(
        dfi2.sort_values("trade_date"),
        dfs2.sort_values("date"),
        left_on="trade_date",
        right_on="date",
        direction="backward",
    )

    rows = []
    erp_vals: list[float | None] = []
    date_col = "date_x" if "date_x" in aligned.columns else ("date" if "date" in aligned.columns else None)
    for _, r in aligned.iterrows():
        # merge_asof 会把左侧的 date 变成 date_x（右侧是 date_y），别拿错列导致全是 None
        d = r.get(date_col) if date_col else None
        if d is None:
            try:
                td = r.get("trade_date")
                d = td.date() if hasattr(td, "date") else None
            except (AttributeError):  # noqa: BLE001
                d = None
        pe_ttm = _safe_float(r.get("pe_ttm"))
        pe = _safe_float(r.get("pe"))
        pe_used = pe_ttm if (pe_ttm is not None and pe_ttm > 0) else pe
        eq_y = (1.0 / float(pe_used)) if (pe_used is not None and pe_used > 0) else None
        rf_pct = _safe_float(r.get(tenor))
        rf_y = (float(rf_pct) / 100.0) if rf_pct is not None else None
        erp = (float(eq_y) - float(rf_y)) if (eq_y is not None and rf_y is not None) else None
        erp_vals.append(erp)
        rows.append(
            {
                "date": str(d) if d is not None else None,
                "pe_used": pe_used,
                "equity_yield": eq_y,
                "rf_name": "shibor",
                "rf_tenor": tenor,
                "rf_value_pct": rf_pct,
                "rf_yield": rf_y,
                "erp": erp,
                "z": None,
                "score01": None,
            }
        )

    zs, sc = rolling_score01_series(erp_vals, window=int(z_window), min_history=int(z_min_history))
    for i in range(len(rows)):
        rows[i]["z"] = zs[i]
        rows[i]["score01"] = sc[i]

    # 裁剪到 lookback_days（用于输出体积控制；rolling history 已在 start 中预留）
    if rows:
        try:
            cutoff = as_of - timedelta(days=int(lb))
            rows = [r for r in rows if r.get("date") and pd.to_datetime(r["date"]).date() >= cutoff]
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    return {
        "ok": True,
        "as_of": str(as_of),
        "index_symbol": str(index_symbol_prefixed),
        "ts_code": str(ts_code),
        "rows": rows,
        "cache": {"index_dailybasic": str(path_idx), "shibor": str(path_shibor)},
        "source": {"name": "tushare"},
        "note": "ERP series proxy=1/index_pe_ttm - shibor_1y；z/score01 用 rolling robust z-score（不含当前点）。",
    }


def compute_hsgt_flow_scores_tushare(
    *,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    沪深港通资金流（north/south）：
    - 数据源：pro.moneyflow_hsgt
    - 用 robust z-score + score01 映射到 [0,1]
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（HSGT 跳过）"}

    win = max(8, int(lookback_days or 0))
    start = as_of - timedelta(days=max(400, win * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    path = cache_dir / f"moneyflow_hsgt_{start_s}_{end_s}.csv"

    df = _read_csv_if_fresh(path, ttl_hours=ttl_hours)
    if df is None or getattr(df, "empty", True):
        try:
            pro = get_pro_api(env)
        except TushareSourceError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            df = pro.moneyflow_hsgt(start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare moneyflow_hsgt 调用失败：{exc}"}
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "TuShare moneyflow_hsgt 返回空"}
        _write_csv_silent(df, path)

    if "trade_date" not in df.columns or "north_money" not in df.columns or "south_money" not in df.columns:
        return {"ok": False, "error": "moneyflow_hsgt 缺少 trade_date/north_money/south_money 列"}

    dfx = df.copy()
    dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d", errors="coerce")
    dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    dfx["north_money"] = pd.to_numeric(dfx["north_money"], errors="coerce")
    dfx["south_money"] = pd.to_numeric(dfx["south_money"], errors="coerce")

    dfx["date"] = dfx["trade_date"].dt.date
    dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
    if dfx.empty:
        return {"ok": False, "error": f"moneyflow_hsgt 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    last = dfx.iloc[-1]
    ref_date = last.get("date")

    north_m = _safe_float(last.get("north_money"))
    south_m = _safe_float(last.get("south_money"))
    # 单位：百万元（TuShare 口径）；换成元方便直觉
    north_yuan = (north_m * 1_000_000.0) if north_m is not None else None
    south_yuan = (south_m * 1_000_000.0) if south_m is not None else None

    hist_n = [float(x) for x in dfx["north_money"].tail(win).to_list() if _safe_float(x) is not None]
    hist_s = [float(x) for x in dfx["south_money"].tail(win).to_list() if _safe_float(x) is not None]

    z_n = robust_zscore(north_m, hist_n) if north_m is not None else None
    z_s = robust_zscore(south_m, hist_s) if south_m is not None else None
    sc_n = z_to_score01(z_n) if z_n is not None else None
    sc_s = z_to_score01(z_s) if z_s is not None else None

    return {
        "ok": True,
        "as_of": str(as_of),
        "ref_date": str(ref_date),
        "north": {"money_million": north_m, "money_yuan": north_yuan, "z": z_n, "score01": sc_n},
        "south": {"money_million": south_m, "money_yuan": south_yuan, "z": z_s, "score01": sc_s},
        "history_days_used": {"north": int(len(hist_n)), "south": int(len(hist_s))},
        "cache_path": str(path),
        "source": {"name": "tushare"},
    }


def compute_hsgt_flow_series_tushare(
    *,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 720,
    z_window: int = 60,
    z_min_history: int = 20,
) -> dict[str, Any]:
    """
    沪深港通 north/south 的时间序列版（用于研究/风控温度计）。
    z/score01：rolling robust z-score（不含当前点，避免未来函数）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（HSGT series 跳过）"}

    lb = max(60, int(lookback_days or 0))
    start = as_of - timedelta(days=max(400, lb + int(z_window) * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    path = cache_dir / f"moneyflow_hsgt_{start_s}_{end_s}.csv"

    df = _read_csv_if_fresh(path, ttl_hours=ttl_hours)
    if df is None or getattr(df, "empty", True):
        try:
            pro = get_pro_api(env)
        except TushareSourceError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            df = pro.moneyflow_hsgt(start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare moneyflow_hsgt 调用失败：{exc}"}
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "TuShare moneyflow_hsgt 返回空"}
        _write_csv_silent(df, path)

    if "trade_date" not in df.columns or "north_money" not in df.columns or "south_money" not in df.columns:
        return {"ok": False, "error": "moneyflow_hsgt 缺少 trade_date/north_money/south_money 列"}

    dfx = df.copy()
    dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d", errors="coerce")
    dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    dfx["north_money"] = pd.to_numeric(dfx["north_money"], errors="coerce")
    dfx["south_money"] = pd.to_numeric(dfx["south_money"], errors="coerce")
    dfx["date"] = dfx["trade_date"].dt.date
    dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
    if dfx.empty:
        return {"ok": False, "error": f"moneyflow_hsgt 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    north_vals = [_safe_float(x) for x in dfx["north_money"].to_list()]
    south_vals = [_safe_float(x) for x in dfx["south_money"].to_list()]
    z_n, sc_n = rolling_score01_series(north_vals, window=int(z_window), min_history=int(z_min_history))
    z_s, sc_s = rolling_score01_series(south_vals, window=int(z_window), min_history=int(z_min_history))

    rows = []
    for i, r in dfx.iterrows():
        d = r.get("date")
        nm = north_vals[i]
        sm = south_vals[i]
        rows.append(
            {
                "date": str(d) if d is not None else None,
                "north_money_million": nm,
                "north_money_yuan": (float(nm) * 1_000_000.0) if nm is not None else None,
                "north_z": z_n[i],
                "north_score01": sc_n[i],
                "south_money_million": sm,
                "south_money_yuan": (float(sm) * 1_000_000.0) if sm is not None else None,
                "south_z": z_s[i],
                "south_score01": sc_s[i],
            }
        )

    # 裁剪输出体积
    if rows:
        try:
            cutoff = as_of - timedelta(days=int(lb))
            rows = [r for r in rows if r.get("date") and pd.to_datetime(r["date"]).date() >= cutoff]
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    return {
        "ok": True,
        "as_of": str(as_of),
        "rows": rows,
        "cache_path": str(path),
        "source": {"name": "tushare"},
        "note": "HSGT series：north/south money + rolling robust z-score/score01（不含当前点）。",
    }


def compute_stock_microstructure_tushare(
    *,
    as_of: date,
    symbol_prefixed: str,
    daily_amount_by_date: dict[str, float] | None,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 60,
) -> dict[str, Any]:
    """
    个股“微观交易结构”proxy（研究用途）：
    - 用 TuShare moneyflow 的大单/超大单净额做“聪明钱”代理
    - 用 日成交额(amount_yuan) 做归一化，避免大盘股天然数值更大
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, prefixed_symbol_to_ts_code

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（microstructure 跳过）"}

    ts_code = prefixed_symbol_to_ts_code(str(symbol_prefixed))
    if ts_code is None:
        return {"ok": False, "error": f"symbol 解析失败：{symbol_prefixed}"}

    win = max(8, int(lookback_days or 0))
    start = as_of - timedelta(days=max(200, win * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    safe_name = ts_code.replace(".", "_")
    path = cache_dir / f"moneyflow_{safe_name}_{start_s}_{end_s}.csv"

    df = _read_csv_if_fresh(path, ttl_hours=ttl_hours)
    if df is None or getattr(df, "empty", True):
        try:
            pro = get_pro_api(env)
        except TushareSourceError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            df = pro.moneyflow(ts_code=str(ts_code), start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare moneyflow 调用失败：{exc}"}
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "TuShare moneyflow 返回空"}
        _write_csv_silent(df, path)

    need_cols = [
        "trade_date",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    for c in need_cols:
        if c not in df.columns:
            return {"ok": False, "error": f"moneyflow 缺少列：{c}"}

    dfx = df.copy()
    dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d", errors="coerce")
    dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    for c in ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce")

    dfx["date"] = dfx["trade_date"].dt.date
    dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
    if dfx.empty:
        return {"ok": False, "error": f"moneyflow 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    # 金额单位：TuShare moneyflow amount 通常是“万元”
    WAN_TO_YUAN = 10_000.0

    dfx["net_big_amount_wan"] = (dfx["buy_lg_amount"] + dfx["buy_elg_amount"]) - (dfx["sell_lg_amount"] + dfx["sell_elg_amount"])
    dfx["net_big_amount_yuan"] = dfx["net_big_amount_wan"] * WAN_TO_YUAN
    dfx["net_mf_amount_yuan"] = dfx["net_mf_amount"] * WAN_TO_YUAN

    # 对齐日成交额（元）：来自日线K（ak/tushare 都可），缺了就不给 ratio，别瞎算
    amt_map = daily_amount_by_date or {}
    net_big_ratio = None
    net_total_ratio = None
    last = dfx.iloc[-1]
    ref_date = last.get("date")

    amt_yuan = None
    try:
        amt_yuan = None if ref_date is None else float(amt_map.get(str(ref_date)))
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        amt_yuan = None
    if amt_yuan is not None and amt_yuan > 0:
        try:
            net_big_ratio = float(last.get("net_big_amount_yuan")) / float(amt_yuan)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            net_big_ratio = None
        try:
            net_total_ratio = float(last.get("net_mf_amount_yuan")) / float(amt_yuan)
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            net_total_ratio = None

    # 历史 z-score（用 ratio 优先；ratio 缺失时用 net_big_amount_yuan）
    hist_vals = []
    if amt_map:
        for _, row in dfx.tail(win).iterrows():
            d = row.get("date")
            if d is None:
                continue
            try:
                ay = float(amt_map.get(str(d)))
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                ay = None  # type: ignore[assignment]
            if ay is None or ay <= 0:
                continue
            try:
                r = float(row.get("net_big_amount_yuan")) / float(ay)
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                r = None
            if _safe_float(r) is not None:
                hist_vals.append(float(r))
    if not hist_vals:
        hist_vals = [float(x) for x in dfx["net_big_amount_yuan"].tail(win).to_list() if _safe_float(x) is not None]

    x_cur = net_big_ratio if (net_big_ratio is not None and hist_vals) else _safe_float(last.get("net_big_amount_yuan"))
    z = robust_zscore(x_cur, hist_vals) if x_cur is not None else None
    score01 = z_to_score01(z) if z is not None else None

    return {
        "ok": True,
        "as_of": str(as_of),
        "symbol": str(symbol_prefixed),
        "ts_code": str(ts_code),
        "ref_date": str(ref_date),
        "last": {
            "net_big_amount_yuan": _safe_float(last.get("net_big_amount_yuan")),
            "net_mf_amount_yuan": _safe_float(last.get("net_mf_amount_yuan")),
            "amount_yuan": amt_yuan,
            "net_big_ratio": net_big_ratio,
            "net_total_ratio": net_total_ratio,
        },
        "z": z,
        "score01": score01,
        "history_days_used": int(len(hist_vals)),
        "cache_path": str(path),
        "source": {"name": "tushare"},
        "note": "microstructure proxy=moneyflow(大单+超大单净额)/成交额；缺成交额时退化为净额本身。",
    }


def compute_stock_microstructure_series_tushare(
    *,
    as_of: date,
    symbol_prefixed: str,
    daily_amount_by_date: dict[str, float] | None,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 120,
    z_window: int = 60,
    z_min_history: int = 20,
) -> dict[str, Any]:
    """
    个股“微观交易结构”proxy 的时间序列版（用于因子研究）：
    - big_proxy = (大单+超大单净额)/成交额
    - z/score01：rolling robust z-score（不含当前点）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareFactorsError("缺依赖：pandas 未安装") from exc

    from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, prefixed_symbol_to_ts_code

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（microstructure series 跳过）"}

    ts_code = prefixed_symbol_to_ts_code(str(symbol_prefixed))
    if ts_code is None:
        return {"ok": False, "error": f"symbol 解析失败：{symbol_prefixed}"}

    lb = max(30, int(lookback_days or 0))
    start = as_of - timedelta(days=max(200, lb + int(z_window) * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    safe_name = ts_code.replace(".", "_")
    path = cache_dir / f"moneyflow_{safe_name}_{start_s}_{end_s}.csv"

    df = _read_csv_if_fresh(path, ttl_hours=ttl_hours)
    if df is None or getattr(df, "empty", True):
        try:
            pro = get_pro_api(env)
        except TushareSourceError as exc:
            return {"ok": False, "error": str(exc)}
        try:
            df = pro.moneyflow(ts_code=str(ts_code), start_date=start_s, end_date=end_s)
        except (AttributeError) as exc:  # noqa: BLE001
            return {"ok": False, "error": f"TuShare moneyflow 调用失败：{exc}"}
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "TuShare moneyflow 返回空"}
        _write_csv_silent(df, path)

    need_cols = [
        "trade_date",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    for c in need_cols:
        if c not in df.columns:
            return {"ok": False, "error": f"moneyflow 缺少列：{c}"}

    dfx = df.copy()
    dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d", errors="coerce")
    dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for c in ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"]:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    dfx["date"] = dfx["trade_date"].dt.date
    dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
    if dfx.empty:
        return {"ok": False, "error": f"moneyflow 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    WAN_TO_YUAN = 10_000.0
    dfx["net_big_amount_wan"] = (dfx["buy_lg_amount"] + dfx["buy_elg_amount"]) - (dfx["sell_lg_amount"] + dfx["sell_elg_amount"])
    dfx["net_big_amount_yuan"] = dfx["net_big_amount_wan"] * WAN_TO_YUAN
    dfx["net_mf_amount_yuan"] = dfx["net_mf_amount"] * WAN_TO_YUAN

    amt_map = daily_amount_by_date or {}
    vals: list[float | None] = []
    rows: list[dict[str, Any]] = []
    for _, r in dfx.iterrows():
        d = r.get("date")
        ds = str(d) if d is not None else None
        amt_y = None
        try:
            amt_y = None if ds is None else float(amt_map.get(ds))
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            amt_y = None

        nb = _safe_float(r.get("net_big_amount_yuan"))
        nt = _safe_float(r.get("net_mf_amount_yuan"))
        nb_ratio = (float(nb) / float(amt_y)) if (nb is not None and amt_y is not None and amt_y > 0) else None
        nt_ratio = (float(nt) / float(amt_y)) if (nt is not None and amt_y is not None and amt_y > 0) else None

        # 优先 ratio；缺成交额就退化为净额（但会有“市值效应”，仅作兜底）
        v = nb_ratio if nb_ratio is not None else nb
        vals.append(v)

        rows.append(
            {
                "date": ds,
                "net_big_amount_yuan": nb,
                "net_mf_amount_yuan": nt,
                "amount_yuan": amt_y,
                "net_big_ratio": nb_ratio,
                "net_total_ratio": nt_ratio,
                "z": None,
                "score01": None,
            }
        )

    zs, sc = rolling_score01_series(vals, window=int(z_window), min_history=int(z_min_history))
    for i in range(len(rows)):
        rows[i]["z"] = zs[i]
        rows[i]["score01"] = sc[i]

    # 裁剪输出体积
    if rows:
        try:
            cutoff = as_of - timedelta(days=int(lb))
            rows = [r for r in rows if r.get("date") and pd.to_datetime(r["date"]).date() >= cutoff]
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    return {
        "ok": True,
        "as_of": str(as_of),
        "symbol": str(symbol_prefixed),
        "ts_code": str(ts_code),
        "rows": rows,
        "cache_path": str(path),
        "source": {"name": "tushare"},
        "note": "microstructure series proxy=(大单+超大单净额)/成交额；z/score01 rolling robust z-score（不含当前点）。",
    }


def compute_tushare_factor_pack(
    *,
    as_of: date,
    context_index_symbol_prefixed: str,
    symbol_prefixed: str | None,
    daily_amount_by_date: dict[str, float] | None,
    cache_dir: Path,
    ttl_hours: float = 6.0,
) -> dict[str, Any]:
    """
    一次性把“宏观ERP + 沪深港通 north/south + (可选)个股microstructure”打包出来。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    erp = compute_erp_proxy_tushare(
        as_of=as_of,
        index_symbol_prefixed=str(context_index_symbol_prefixed),
        cache_dir=cache_dir / "erp",
        ttl_hours=float(ttl_hours),
    )
    hsgt = compute_hsgt_flow_scores_tushare(as_of=as_of, cache_dir=cache_dir / "hsgt", ttl_hours=float(ttl_hours))

    micro = None
    if symbol_prefixed:
        micro = compute_stock_microstructure_tushare(
            as_of=as_of,
            symbol_prefixed=str(symbol_prefixed),
            daily_amount_by_date=daily_amount_by_date,
            cache_dir=cache_dir / "micro",
            ttl_hours=float(ttl_hours),
        )

    return {
        "ok": bool(erp.get("ok") or hsgt.get("ok") or (micro or {}).get("ok")),
        "as_of": str(as_of),
        "erp": erp,
        "hsgt": hsgt,
        "microstructure": micro,
        "source": {"name": "tushare"},
    }
