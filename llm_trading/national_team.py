from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class NationalTeamProxyConfig:
    """
    “国家队/托底”代理指标（研究用途）：
    - 我们不做“精确知道国家队买卖”的幻想；只用公开可得的数据做 proxy。
    - 目标：作为组合层的“风险温度计/解释面板”，而不是神谕。
    """

    # 盯盘指数（决定尾盘护盘特征的计算对象）
    index_symbol: str = "sh000300"

    # 宽基ETF：优先选择“可能被用作托底通道”的大票
    wide_etfs: tuple[str, ...] = (
        "sh510300",  # 沪深300ETF
        "sh510050",  # 上证50ETF
        "sh510500",  # 中证500ETF
        "sz159915",  # 创业板ETF
        "sh588000",  # 科创50ETF
        "sz159919",  # 沪深300ETF(深)
    )

    # A: ETF 资金流（主力净流入）历史回看窗口（天，受数据源限制通常≈120天）
    etf_flow_lookback_days: int = 120

    # C: 尾盘窗口（分钟），默认 30min（14:30-15:00）
    tail_window_minutes: int = 30

    # 组合权重：缺数据会自动归一化（不让某一项缺失把总分算炸）
    w_etf_flow: float = 0.55
    w_etf_shares: float = 0.25
    w_tail: float = 0.20
    # 北向（B）在 2024-08 后公开日度数据经常缺失：保留字段但默认不计入
    w_northbound: float = 0.0


def _parse_yyyymmdd_any(s: str | None) -> date | None:
    if not s:
        return None
    t = str(s).strip()
    if not t:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(t, fmt).date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            continue
    return None


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
    # 避免引入 numpy；utils_stats.median 已有，但这里保持模块自洽（减少循环依赖）。
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
    稳健 z-score：用 median/MAD 替代 mean/std，抗极端值（更像“托底日”的那种尖峰）。
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
    # 1.4826: MAD -> std 的一致性系数（正态假设下）
    return float((x2 - med) / (1.4826 * mad))


def z_to_score01(z: float | None, *, clip: float = 3.0) -> float | None:
    """
    把 z-score 映射到 [0,1]，用于多因子加权。
    - z=0 => 0.5
    - |z|>=clip => 0/1
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


def _normalize_symbol_etf(sym: str) -> str:
    s = str(sym or "").strip().lower()
    if not s:
        return ""
    if s.startswith(("sh", "sz")):
        return s
    if s.isdigit() and len(s) == 6:
        # ETF 代码规则：5xxxx=>沪；其他常见=>深
        return f"sh{s}" if s.startswith("5") else f"sz{s}"
    return s


def _symbol_to_market_code(sym: str) -> tuple[str, str] | None:
    """
    把 sh510300 -> (sh, 510300) 这种给 Eastmoney 资金流接口用。
    """
    s = _normalize_symbol_etf(sym)
    if not s or len(s) != 8:
        return None
    pref = s[:2]
    code = s[2:]
    if pref not in {"sh", "sz"} or not code.isdigit():
        return None
    return pref, code


def fetch_etf_spot_snapshot(*, cache_dir: Path, ttl_hours: float) -> dict[str, Any]:
    """
    拉取 ETF 实时行情（含“最新份额/主力净流入”），并落盘为一个“快照”。
    注意：东财是实时接口；这里用 数据日期(YYYYMMDD) 作为快照 key。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    try:
        import akshare as ak
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：akshare 未安装") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 先看缓存：别每次都去薅东财（容易被踢/慢）
    if float(ttl_hours) > 0:
        try:
            latest = None
            for p in cache_dir.glob("etf_spot_*.csv"):
                if latest is None or p.stat().st_mtime > latest.stat().st_mtime:
                    latest = p
            if latest is not None:
                age = (datetime.now().timestamp() - latest.stat().st_mtime) / 3600.0
                if age <= float(ttl_hours):
                    # as_of 从文件名解析
                    try:
                        ds = latest.name.split("_", 2)[-1].split(".", 1)[0]
                        d = datetime.strptime(ds, "%Y%m%d").date()
                    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                        d = datetime.now().date()
                    return {"ok": True, "as_of": str(d), "path": str(latest), "cached": True}
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            pass

    # 直接拉现货：这个接口没法按日期查历史，所以只能“每天跑一次落盘”积累。
    try:
        df = ak.fund_etf_spot_em()
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "path": None, "as_of": None}
    if df is None or getattr(df, "empty", True):
        return {"ok": False, "error": "fund_etf_spot_em 返回空", "path": None, "as_of": None}

    # 数据日期：有些行可能 NaT，用整体 max 兜底
    dmax = None
    try:
        dser = pd.to_datetime(df.get("数据日期"), errors="coerce")
        if dser is not None:
            dmax = dser.dropna().max()
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        dmax = None
    as_of = (dmax.date() if hasattr(dmax, "date") and dmax is not None else datetime.now().date())

    path = cache_dir / f"etf_spot_{_fmt_yyyymmdd(as_of)}.csv"

    # 小心：全量 1350+ 行没必要每次都写全部字段；先留“够用”的列
    keep_cols = [
        "代码",
        "名称",
        "最新价",
        "成交额",
        "主力净流入-净额",
        "主力净流入-净占比",
        "最新份额",
        "流通市值",
        "总市值",
        "数据日期",
        "更新时间",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    out_df = df[cols].copy()
    try:
        out_df.to_csv(path, index=False, encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "as_of": str(as_of), "error": f"写入快照失败：{exc}", "path": str(path)}
    return {"ok": True, "as_of": str(as_of), "path": str(path), "cached": False}


def _latest_snapshot_before(*, cache_dir: Path, as_of: date) -> Path | None:
    """
    找到 cache_dir 里 <as_of 的最近一个 etf_spot_YYYYMMDD.csv（用于计算份额Δ）。
    """
    if not cache_dir.exists():
        return None
    cand: list[tuple[date, Path]] = []
    for p in cache_dir.glob("etf_spot_*.csv"):
        name = p.name
        try:
            ds = name.split("_", 2)[-1].split(".", 1)[0]
            d = datetime.strptime(ds, "%Y%m%d").date()
        except (TypeError, ValueError, AttributeError):  # noqa: BLE001
            continue
        if d < as_of:
            cand.append((d, p))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[-1][1]


def compute_etf_share_deltas(*, spot_path: Path, prev_spot_path: Path | None, watchlist: tuple[str, ...]) -> dict[str, Any]:
    """
    A(1)：宽基ETF份额变动（需要连续两天的 spot 快照）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    def _read(p: Path) -> Any:
        try:
            return pd.read_csv(p, encoding="utf-8")
        except (AttributeError):  # noqa: BLE001
            return None

    df = _read(spot_path)
    if df is None or getattr(df, "empty", True):
        return {"ok": False, "error": f"读取 spot 失败：{spot_path}", "items": []}

    df_prev = _read(prev_spot_path) if prev_spot_path else None
    if prev_spot_path and (df_prev is None or getattr(df_prev, "empty", True)):
        df_prev = None

    # watchlist: 统一成 6 位数字代码
    codes = []
    for s in watchlist:
        ss = _normalize_symbol_etf(s)
        if ss and len(ss) == 8:
            codes.append(ss[2:])
    codes_set = set(codes)

    df2 = df.copy()
    if "代码" not in df2.columns:
        return {"ok": False, "error": "spot 缺少 代码 列", "items": []}
    df2["代码"] = df2["代码"].astype(str).str.zfill(6)
    cur = df2[df2["代码"].isin(codes_set)].copy()

    prev_map: dict[str, float] = {}
    if df_prev is not None and "代码" in df_prev.columns and "最新份额" in df_prev.columns:
        dp = df_prev.copy()
        dp["代码"] = dp["代码"].astype(str).str.zfill(6)
        try:
            dp["最新份额"] = pd.to_numeric(dp["最新份额"], errors="coerce")
        except (AttributeError):  # noqa: BLE001
            pass
        for _, r in dp.iterrows():
            code = str(r.get("代码") or "").zfill(6)
            sh = _safe_float(r.get("最新份额"))
            if code and sh is not None:
                prev_map[code] = float(sh)

    items: list[dict[str, Any]] = []
    for _, r in cur.iterrows():
        code = str(r.get("代码") or "").zfill(6)
        sym = _normalize_symbol_etf(code)
        name = str(r.get("名称") or "")
        shares = _safe_float(r.get("最新份额"))
        px = _safe_float(r.get("最新价"))
        inflow = _safe_float(r.get("主力净流入-净额"))
        amt = _safe_float(r.get("成交额"))

        prev_sh = prev_map.get(code)
        delta = (float(shares) - float(prev_sh)) if (shares is not None and prev_sh is not None) else None
        delta_pct = (delta / float(prev_sh)) if (delta is not None and prev_sh and prev_sh != 0) else None
        notional_delta = (delta * float(px)) if (delta is not None and px is not None) else None

        items.append(
            {
                "symbol": sym,
                "code": code,
                "name": name,
                "price": px,
                "shares": shares,
                "shares_prev": prev_sh,
                "shares_delta": delta,
                "shares_delta_pct": delta_pct,
                "shares_delta_notional": notional_delta,
                "main_inflow_yuan_spot": inflow,
                "turnover_yuan_spot": amt,
            }
        )

    ok = True
    warn = None
    if prev_spot_path is None:
        ok = False
        warn = "缺少上一交易日 spot 快照：份额Δ暂不可用（需要每天收盘跑一次积累）"
    elif not prev_map:
        ok = False
        warn = f"上一交易日 spot 快照缺少 最新份额：{prev_spot_path}"

    return {
        "ok": ok,
        "warning": warn,
        "spot_path": str(spot_path),
        "prev_spot_path": (str(prev_spot_path) if prev_spot_path else None),
        "items": items,
    }


def fetch_etf_main_flow_hist(*, symbol: str, cache_dir: Path, ttl_hours: float) -> dict[str, Any]:
    """
    A(2)：ETF 主力净流入历史（东财资金流向，近 ~120 个交易日）。
    这个接口对 ETF 代码可用（它本质上把 ETF 当成“股票”看）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    mc = _symbol_to_market_code(symbol)
    if mc is None:
        return {"ok": False, "error": f"symbol 非法：{symbol}", "symbol": symbol}
    market, code = mc

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"etf_main_flow_{market}{code}.csv"
    if path.exists() and float(ttl_hours) > 0:
        try:
            age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
            if age <= float(ttl_hours):
                df = pd.read_csv(path, encoding="utf-8")
                return {"ok": True, "symbol": symbol, "market": market, "code": code, "path": str(path), "df": df, "cached": True}
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    try:
        import akshare as ak
    except ModuleNotFoundError as exc:  # pragma: no cover
        return {"ok": False, "symbol": symbol, "error": "akshare 未安装"}  # pragma: no cover

    try:
        df = ak.stock_individual_fund_flow(stock=str(code), market=str(market))
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "symbol": symbol, "market": market, "code": code, "error": str(exc)}

    if df is None or getattr(df, "empty", True):
        return {"ok": False, "symbol": symbol, "market": market, "code": code, "error": "资金流接口返回空"}

    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except (AttributeError):  # noqa: BLE001
        # 写缓存失败不影响计算
        pass
    return {"ok": True, "symbol": symbol, "market": market, "code": code, "path": str(path), "df": df, "cached": False}


def compute_etf_flow_score(*, as_of: date, watchlist: tuple[str, ...], cache_dir: Path, ttl_hours: float, lookback_days: int) -> dict[str, Any]:
    """
    用“宽基ETF主力净流入”构建一个日度强度分数。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    series_list: list[Any] = []
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for sym in watchlist:
        out = fetch_etf_main_flow_hist(symbol=sym, cache_dir=cache_dir, ttl_hours=ttl_hours)
        if not bool(out.get("ok")):
            errors.append(f"{sym}: {out.get('error')}")
            continue
        df = out.get("df")
        if df is None or getattr(df, "empty", True):
            errors.append(f"{sym}: df 为空")
            continue
        dfx = df.copy()
        if "日期" not in dfx.columns or "主力净流入-净额" not in dfx.columns:
            errors.append(f"{sym}: 缺少 日期/主力净流入-净额")
            continue
        dfx["日期"] = pd.to_datetime(dfx["日期"], errors="coerce")
        dfx = dfx.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
        dfx["主力净流入-净额"] = pd.to_numeric(dfx["主力净流入-净额"], errors="coerce")

        s = dfx.set_index(dfx["日期"].dt.date)["主力净流入-净额"]
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s.rename(_normalize_symbol_etf(sym)))

        # 当日值（如果没有当天，就用 <=as_of 最近一天，别瞎补未来）
        s2 = s[s.index <= as_of]
        v = float(s2.iloc[-1]) if len(s2) else None
        items.append(
            {
                "symbol": _normalize_symbol_etf(sym),
                "value_yuan": v,
                "ref_date": (str(s2.index[-1]) if len(s2) else None),
                "cached": bool(out.get("cached")),
                "cache_path": out.get("path"),
            }
        )

    if not series_list:
        return {"ok": False, "error": "无可用 ETF 资金流数据", "items": items, "errors": errors}

    df_all = pd.concat(series_list, axis=1).sort_index()
    sum_series = df_all.sum(axis=1, skipna=True)
    sum_series = sum_series[sum_series.index <= as_of]
    if sum_series.empty:
        return {"ok": False, "error": "资金流合成序列为空(可能 as_of 太早)", "items": items, "errors": errors}

    x = float(sum_series.iloc[-1])
    hist = [float(v) for v in sum_series.tail(max(8, int(lookback_days))).to_list() if _safe_float(v) is not None]
    z = robust_zscore(x, hist)
    score01 = z_to_score01(z)
    return {
        "ok": True,
        "as_of": str(as_of),
        "sum_main_inflow_yuan": float(x),
        "z_main_inflow": z,
        "score01": score01,
        "items": items,
        "errors": errors,
        "history_days_used": int(len(hist)),
    }


def fetch_index_trends2(*, secid: str, timeout_sec: float = 10.0) -> dict[str, Any]:
    """
    C：指数分时（push2his trends2），只给最近 ~5 个交易日（ndays=5）。
    这个接口比 ak.index_zh_a_hist_min_em 更稳（不依赖 index_code_id_map_em）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    import requests

    url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "iscr": "0",
        "ndays": "5",
        "secid": str(secid),
    }
    try:
        r = requests.get(url, params=params, timeout=float(timeout_sec))
        js = r.json()
    except (requests.exceptions.RequestException, ValueError, TypeError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "secid": secid, "df": None}

    data = js.get("data") if isinstance(js, dict) else None
    trends = (data.get("trends") if isinstance(data, dict) else None) or []
    if not trends:
        return {"ok": False, "error": "data.trends 为空", "secid": secid, "df": None}

    rows = [str(x).split(",") for x in trends]
    df = pd.DataFrame(rows, columns=["time", "open", "close", "high", "low", "volume", "amount", "avg"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume", "amount", "avg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    return {"ok": True, "secid": secid, "df": df, "name": (data.get("name") if isinstance(data, dict) else None)}


def compute_northbound_flow_score_tushare(
    *,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float,
    lookback_days: int,
) -> dict[str, Any]:
    """
    北向资金 proxy（TuShare moneyflow_hsgt）：
    - 取 north_money（百万元）
    - 用 robust z-score + score01 映射到 [0,1]

    说明：
    - 只用 <=as_of 的最后一个交易日数据，避免“非交易日/未来函数”。
    - 如未配置 TUSHARE_TOKEN 或接口不可用，返回 ok=False。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    try:
        from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env
    except (ImportError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "error": f"TuShare 模块不可用：{exc}"}

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（北向组件跳过）"}

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 300 条上限一般覆盖 1 年交易日；给 400 天缓冲足够。
    win = max(8, int(lookback_days or 0))
    start = as_of - timedelta(days=max(400, win * 3))
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)
    path = cache_dir / f"moneyflow_hsgt_{start_s}_{end_s}.csv"

    df = None
    if path.exists() and float(ttl_hours) > 0:
        try:
            age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
            if age <= float(ttl_hours):
                df = pd.read_csv(path, encoding="utf-8")
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            df = None

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
        try:
            df.to_csv(path, index=False, encoding="utf-8")
        except (AttributeError):  # noqa: BLE001
            pass

    if "trade_date" not in df.columns or "north_money" not in df.columns:
        return {"ok": False, "error": "moneyflow_hsgt 缺少 trade_date/north_money 列"}

    dfx = df.copy()
    dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], errors="coerce")
    dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    dfx["north_money"] = pd.to_numeric(dfx["north_money"], errors="coerce")

    s = dfx.set_index(dfx["trade_date"].dt.date)["north_money"]
    s = s[~s.index.duplicated(keep="last")]
    s2 = s[s.index <= as_of]
    if s2.empty:
        return {"ok": False, "error": f"moneyflow_hsgt 在 <=as_of={as_of} 范围内无数据", "cache_path": str(path)}

    v = _safe_float(s2.iloc[-1])
    ref_date = s2.index[-1]
    hist = [float(x) for x in s2.tail(win).to_list() if _safe_float(x) is not None]
    z = robust_zscore(v, hist) if v is not None else None
    score01 = z_to_score01(z) if z is not None else None

    # north_money 单位：百万元；同时给一个换算后的“元”
    north_million = float(v) if v is not None else None
    north_yuan = (north_million * 1_000_000.0) if north_million is not None else None

    return {
        "ok": True,
        "as_of": str(as_of),
        "ref_date": str(ref_date),
        "north_money_million": north_million,
        "north_money_yuan": north_yuan,
        "z": z,
        "score01": score01,
        "cache_path": str(path),
        "source": {"name": "tushare"},
        "history_days_used": int(len(hist)),
    }


def compute_etf_share_deltas_tushare(
    *,
    as_of: date,
    watchlist: tuple[str, ...],
    cache_dir: Path,
    ttl_hours: float,
) -> dict[str, Any]:
    """
    宽基ETF份额Δ（TuShare etf_share_size）：
    - total_share: 万份
    - close: 元
    - shares_delta_notional: 估算申赎名义资金量（元）= Δ(万份)*1e4*close

    注意：etf_share_size 属于高积分接口；失败则返回 ok=False（让上层回退到东财 spot 快照方案）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    try:
        from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, normalize_ts_code, ts_code_to_symbol
    except (ImportError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "error": f"TuShare 模块不可用：{exc}", "items": []}

    env = load_tushare_env()
    if env is None:
        return {"ok": False, "error": "未配置 TUSHARE_TOKEN（ETF份额组件跳过）", "items": []}

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        pro = get_pro_api(env)
    except TushareSourceError as exc:
        return {"ok": False, "error": str(exc), "items": []}

    start = as_of - timedelta(days=60)  # 覆盖节假日/海外ETF延迟的缓冲
    start_s = _fmt_yyyymmdd(start)
    end_s = _fmt_yyyymmdd(as_of)

    items: list[dict[str, Any]] = []
    errors: list[str] = []

    for sym in watchlist:
        ts_code = normalize_ts_code(sym)
        if not ts_code:
            errors.append(f"{sym}: 无法转换为 ts_code")
            continue

        safe_name = ts_code.replace(".", "_")
        path = cache_dir / f"etf_share_size_{safe_name}_{start_s}_{end_s}.csv"

        df = None
        if path.exists() and float(ttl_hours) > 0:
            try:
                age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
                if age <= float(ttl_hours):
                    df = pd.read_csv(path, encoding="utf-8")
            except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
                df = None

        if df is None or getattr(df, "empty", True):
            try:
                df = pro.etf_share_size(ts_code=str(ts_code), start_date=start_s, end_date=end_s)
            except (AttributeError) as exc:  # noqa: BLE001
                errors.append(f"{ts_code}: etf_share_size 调用失败：{exc}")
                continue
            if df is None or getattr(df, "empty", True):
                errors.append(f"{ts_code}: etf_share_size 返回空")
                continue
            try:
                df.to_csv(path, index=False, encoding="utf-8")
            except (AttributeError):  # noqa: BLE001
                pass

        if "trade_date" not in df.columns or "total_share" not in df.columns:
            errors.append(f"{ts_code}: 缺少 trade_date/total_share 列")
            continue

        dfx = df.copy()
        dfx["trade_date"] = pd.to_datetime(dfx["trade_date"], errors="coerce")
        dfx = dfx.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
        dfx["total_share"] = pd.to_numeric(dfx["total_share"], errors="coerce")
        if "total_size" in dfx.columns:
            dfx["total_size"] = pd.to_numeric(dfx["total_size"], errors="coerce")
        if "close" in dfx.columns:
            dfx["close"] = pd.to_numeric(dfx["close"], errors="coerce")
        if "nav" in dfx.columns:
            dfx["nav"] = pd.to_numeric(dfx["nav"], errors="coerce")

        dfx["date"] = dfx["trade_date"].dt.date
        dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
        if len(dfx) < 2:
            errors.append(f"{ts_code}: <=as_of 数据不足（需要至少2个交易日）")
            continue

        cur = dfx.iloc[-1]
        prev = dfx.iloc[-2]

        sh_cur = _safe_float(cur.get("total_share"))
        sh_prev = _safe_float(prev.get("total_share"))
        delta = (float(sh_cur) - float(sh_prev)) if (sh_cur is not None and sh_prev is not None) else None
        delta_pct = (delta / float(sh_prev)) if (delta is not None and sh_prev and sh_prev != 0) else None

        px = _safe_float(cur.get("close"))
        if px is None:
            px = _safe_float(cur.get("nav"))

        # total_share 单位：万份；close 单位：元
        notional_delta = (delta * 10_000.0 * float(px)) if (delta is not None and px is not None) else None

        sym2 = ts_code_to_symbol(ts_code) or _normalize_symbol_etf(sym)
        items.append(
            {
                "symbol": sym2,
                "ts_code": ts_code,
                "name": str(cur.get("etf_name") or ""),
                "price": px,
                "shares": sh_cur,  # 单位：万份（见 shares_unit）
                "shares_prev": sh_prev,  # 单位：万份
                "shares_delta": delta,  # 单位：万份
                "shares_delta_pct": delta_pct,
                "shares_delta_notional": notional_delta,  # 单位：元（估算）
                "shares_unit": "wan_shares",
                "ref_date": str(cur.get("date") or ""),
                "prev_ref_date": str(prev.get("date") or ""),
                "cache_path": str(path),
            }
        )

    ok = bool(items) and (len(items) >= max(1, int(len(watchlist) * 0.5)))
    if not ok and not errors:
        errors.append("无可用 etf_share_size 数据")

    return {
        "ok": ok,
        "as_of": str(as_of),
        "items": items,
        "errors": errors,
        "source": {"name": "tushare"},
    }


def _index_secid_candidates(index_symbol: str) -> list[str]:
    s = str(index_symbol or "").strip().lower()
    if s.startswith(("sh", "sz")):
        code = s[2:]
        pref = s[:2]
    else:
        code = s
        pref = ""
    code = code.strip()
    if not code.isdigit():
        return []

    # akshare 的兜底顺序：1 -> 0 -> 47（有些指数挂在 47 市场） -> 2
    if pref == "sz":
        base = ["0", "1", "47", "2"]
    else:
        base = ["1", "0", "47", "2"]
    return [f"{m}.{code}" for m in base]


def compute_tail_support(*, as_of: date, index_symbol: str, cache_dir: Path, ttl_hours: float, window_minutes: int) -> dict[str, Any]:
    """
    尾盘护盘 proxy（研究用途）：
    - 用最后 window_minutes 的涨幅/量能占比做一个“尾盘强度”指标。
    - trends2 只返回最近 5 个交易日；as_of 太早会缺数据。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    code = str(index_symbol or "").strip().lower()
    code2 = code[2:] if code.startswith(("sh", "sz")) else code
    path = cache_dir / f"index_trends2_{code2}.csv"

    df = None
    name = None
    used_secid = None
    errors: list[str] = []

    if path.exists() and float(ttl_hours) > 0:
        try:
            age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
            if age <= float(ttl_hours):
                df = pd.read_csv(path, encoding="utf-8")
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            df = None

    if df is None or getattr(df, "empty", True):
        for secid in _index_secid_candidates(index_symbol):
            out = fetch_index_trends2(secid=secid)
            if bool(out.get("ok")) and out.get("df") is not None:
                df = out.get("df")
                name = out.get("name")
                used_secid = secid
                break
            errors.append(f"{secid}: {out.get('error')}")
        if df is None or getattr(df, "empty", True):
            return {"ok": False, "error": "指数分钟数据获取失败", "errors": errors, "index_symbol": index_symbol}
        try:
            df.to_csv(path, index=False, encoding="utf-8")
        except (AttributeError):  # noqa: BLE001
            pass

    dfx = df.copy()
    dfx["date"] = dfx["time"].dt.date
    day = dfx[dfx["date"] == as_of].copy()
    if day.empty:
        # trends2 只给最近 5 个交易日：as_of 太早就没
        return {
            "ok": False,
            "error": f"trends2 不包含 as_of={as_of}（接口只给近5日）",
            "index_symbol": index_symbol,
            "secid": used_secid,
            "cache_path": str(path),
        }

    day = day.sort_values("time").reset_index(drop=True)
    close_first = _safe_float(day["close"].iloc[0])
    close_last = _safe_float(day["close"].iloc[-1])
    if close_first is None or close_last is None or close_first <= 0:
        return {"ok": False, "error": "close 数据非法", "index_symbol": index_symbol, "secid": used_secid, "cache_path": str(path)}

    # 尾盘窗口：用时间戳做切片（更稳，不依赖分钟根数是否齐全）
    start_tail = datetime.combine(as_of, datetime.strptime("15:00", "%H:%M").time())
    start_tail = start_tail.replace(minute=0)  # 15:00
    # window_minutes 默认 30 => 14:30
    start_tail = start_tail.replace(hour=15, minute=0)  # reset
    start_tail = start_tail.replace(hour=15, minute=0)  # explicit
    # 14:30
    start_tail = datetime.combine(as_of, datetime.strptime("14:30", "%H:%M").time())
    try:
        # 支持用户改 window_minutes（比如 15/60）
        start_tail = datetime.combine(as_of, datetime.strptime("15:00", "%H:%M").time()) - pd.Timedelta(minutes=int(window_minutes))
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        start_tail = datetime.combine(as_of, datetime.strptime("14:30", "%H:%M").time())

    tail = day[day["time"] >= start_tail].copy()
    if tail.empty:
        return {"ok": False, "error": "尾盘窗口数据为空", "index_symbol": index_symbol, "secid": used_secid, "cache_path": str(path)}

    close_tail_start = _safe_float(tail["close"].iloc[0])
    if close_tail_start is None or close_tail_start <= 0:
        return {"ok": False, "error": "尾盘起点 close 非法", "index_symbol": index_symbol, "secid": used_secid, "cache_path": str(path)}

    ret_day = float(close_last / close_first - 1.0)
    ret_tail = float(close_last / close_tail_start - 1.0)

    amt_day = _safe_float(day.get("amount").sum() if "amount" in day.columns else None)
    amt_tail = _safe_float(tail.get("amount").sum() if "amount" in tail.columns else None)

    n_total = int(len(day))
    n_tail = int(len(tail))
    vol_ratio = None
    if amt_day is not None and amt_tail is not None and amt_day > 0 and n_total > 0 and n_tail > 0:
        # (尾盘每分钟成交额) / (全日每分钟成交额)
        vol_ratio = float((amt_tail / n_tail) / (amt_day / n_total))

    # 简单打分：尾盘涨幅为正 + 量能放大 => 分更高（归一化到 0~1）
    raw = 0.0
    raw += max(-0.05, min(0.05, ret_tail)) / 0.05  # [-1,1]
    if vol_ratio is not None:
        raw += max(-1.0, min(1.0, (vol_ratio - 1.0)))  # 量能>1 加分
        raw /= 2.0
    # map [-1,1] -> [0,1]
    score01 = float((raw + 1.0) / 2.0)

    return {
        "ok": True,
        "as_of": str(as_of),
        "index_symbol": index_symbol,
        "index_name": name,
        "secid": used_secid,
        "cache_path": str(path),
        "ret_day": ret_day,
        "ret_tail": ret_tail,
        "tail_start": start_tail.isoformat(sep=" "),
        "amount_day": amt_day,
        "amount_tail": amt_tail,
        "n_total": n_total,
        "n_tail": n_tail,
        "tail_amount_per_min_ratio": vol_ratio,
        "score01": score01,
    }


def compute_national_team_proxy(
    *,
    as_of: date,
    cfg: NationalTeamProxyConfig | None = None,
    cache_ttl_hours: float = 6.0,
) -> dict[str, Any]:
    """
    汇总：A(ETF份额/资金)+B(北向;可能缺)+C(尾盘护盘) -> composite_score。
    """
    conf = cfg or NationalTeamProxyConfig()

    spot_dir = Path("data") / "cache" / "etf_spot"
    flow_dir = Path("data") / "cache" / "etf_flow"
    share_dir = Path("data") / "cache" / "etf_share"
    idx_min_dir = Path("data") / "cache" / "index_min"
    nb_dir = Path("data") / "cache" / "northbound"

    warnings: list[str] = []

    # 1) ETF spot 快照（主要用于“当日快照信息”）+ 份额Δ（优先 TuShare；失败再回退东财快照）
    spot_meta = fetch_etf_spot_snapshot(cache_dir=spot_dir, ttl_hours=float(cache_ttl_hours))
    spot_ok = bool(spot_meta.get("ok"))
    spot_path = Path(str(spot_meta.get("path") or "")) if spot_meta.get("path") else None
    spot_asof = _parse_yyyymmdd_any(str(spot_meta.get("as_of") or "")) or as_of
    if spot_ok and spot_path and spot_asof != as_of:
        warnings.append(f"ETF spot 数据日期={spot_asof} 与 as_of={as_of} 不一致（周末/数据源延迟/盘中运行？）")

    # 1.1) 份额Δ：先试 TuShare（无需你每天落盘快照；但需要高积分权限）
    share_part_ts: dict[str, Any] = compute_etf_share_deltas_tushare(
        as_of=as_of, watchlist=conf.wide_etfs, cache_dir=share_dir, ttl_hours=float(cache_ttl_hours)
    )
    share_part: dict[str, Any] = share_part_ts

    # 1.2) TuShare 不可用/失败 -> 回退东财 spot 快照（需要连续两天快照）
    if not bool(share_part.get("ok")):
        err_ts = str(share_part_ts.get("error") or "").strip()
        if not err_ts and isinstance(share_part_ts.get("errors"), list):
            es = [str(x) for x in (share_part_ts.get("errors") or []) if str(x).strip()]
            if es:
                err_ts = "; ".join(es[:5])
        if err_ts and ("未配置 TUSHARE_TOKEN" not in err_ts):
            warnings.append(f"TuShare ETF份额组件不可用：{err_ts}（已回退到东财 spot 快照）")
        share_part = {"ok": False, "warning": "spot 不可用", "items": []}
        if spot_ok and spot_path and spot_path.exists():
            prev_path = _latest_snapshot_before(cache_dir=spot_dir, as_of=spot_asof)
            share_part = compute_etf_share_deltas(spot_path=spot_path, prev_spot_path=prev_path, watchlist=conf.wide_etfs)
            if share_part.get("warning"):
                warnings.append(str(share_part.get("warning")))
        else:
            warnings.append(f"ETF spot 快照不可用：{spot_meta.get('error')}")

    # 2) ETF 主力净流入历史
    flow_part = compute_etf_flow_score(
        as_of=as_of,
        watchlist=conf.wide_etfs,
        cache_dir=flow_dir,
        ttl_hours=float(cache_ttl_hours),
        lookback_days=int(conf.etf_flow_lookback_days),
    )
    if not bool(flow_part.get("ok")):
        warnings.append(f"ETF资金流计算失败：{flow_part.get('error')}")

    # 3) 尾盘护盘特征
    tail_part = compute_tail_support(
        as_of=as_of,
        index_symbol=str(conf.index_symbol),
        cache_dir=idx_min_dir,
        ttl_hours=float(cache_ttl_hours),
        window_minutes=int(conf.tail_window_minutes),
    )
    if not bool(tail_part.get("ok")):
        warnings.append(f"尾盘特征缺失：{tail_part.get('error')}")

    # 4) 北向（B）：优先 TuShare（moneyflow_hsgt）；没配 token 就跳过
    north_part = compute_northbound_flow_score_tushare(
        as_of=as_of,
        cache_dir=nb_dir,
        ttl_hours=float(cache_ttl_hours),
        lookback_days=int(conf.etf_flow_lookback_days),
    )
    if not bool(north_part.get("ok")) and float(conf.w_northbound) > 0:
        warnings.append(f"北向组件缺失：{north_part.get('error')}")

    # --- composite score ---
    # 分数都用 score01([0,1])，缺失则跳过权重并重新归一化
    comp_inputs: list[tuple[str, float, float | None]] = [
        ("etf_flow", float(conf.w_etf_flow), flow_part.get("score01") if isinstance(flow_part, dict) else None),
        ("etf_shares", float(conf.w_etf_shares), None),
        ("tail", float(conf.w_tail), tail_part.get("score01") if isinstance(tail_part, dict) else None),
        ("northbound", float(conf.w_northbound), north_part.get("score01") if isinstance(north_part, dict) else None),
    ]

    # shares score：用 份额Δ的“名义资金量”做一个粗分（没有历史就不给分，避免瞎编）
    shares_score01 = None
    try:
        if bool(share_part.get("ok")) and isinstance(share_part.get("items"), list):
            xs = []
            for it in share_part.get("items") or []:
                if not isinstance(it, dict):
                    continue
                v = _safe_float(it.get("shares_delta_notional"))
                if v is not None:
                    xs.append(float(v))
            if xs:
                total = float(sum(xs))
                # 经验：用 log 压缩量纲（别让某一天的极端值把分打爆）
                import math

                shares_score01 = float(0.5 + 0.5 * math.tanh(total / 1e9))  # 1e9=10亿名义规模
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        shares_score01 = None

    comp_inputs = [(k, w, (shares_score01 if k == "etf_shares" else v)) for (k, w, v) in comp_inputs]

    w_sum = 0.0
    s_sum = 0.0
    used: dict[str, Any] = {}
    for k, w, v in comp_inputs:
        vv = _safe_float(v)
        if vv is None:
            continue
        ww = max(0.0, float(w))
        if ww <= 0:
            continue
        w_sum += ww
        s_sum += ww * float(vv)
        used[k] = {"weight": ww, "score01": float(vv)}

    composite01 = (s_sum / w_sum) if w_sum > 0 else None

    return {
        "schema": "llm_trading.national_team_proxy.v1",
        "generated_at": datetime.now().isoformat(),
        "as_of": str(as_of),
        "config": {
            "index_symbol": conf.index_symbol,
            "wide_etfs": list(conf.wide_etfs),
            "weights": {"etf_flow": conf.w_etf_flow, "etf_shares": conf.w_etf_shares, "tail": conf.w_tail, "northbound": conf.w_northbound},
            "etf_flow_lookback_days": int(conf.etf_flow_lookback_days),
            "tail_window_minutes": int(conf.tail_window_minutes),
        },
        "score": {
            "composite01": composite01,
            "composite_pct": (float(composite01) * 100.0 if composite01 is not None else None),
            "used": used,
        },
        "components": {
            "etf_spot": spot_meta,
            "etf_shares": {"score01": shares_score01, **share_part},
            "etf_flow": flow_part,
            "tail": tail_part,
            "northbound": north_part,
        },
        "warnings": warnings,
        "disclaimer": "研究工具输出，不构成投资建议；国家队/内幕走势无法被公开数据精确预测。",
    }


def backtest_etf_flow_proxy(
    *,
    index_symbol: str,
    watchlist: tuple[str, ...],
    start: date | None = None,
    end: date | None = None,
    lookback_days: int = 60,
    cache_ttl_hours: float = 24.0,
) -> dict[str, Any]:
    """
    最小回测（研究用途）：只验证 A(ETF主力净流入合成) 这个 proxy 的“统计学体感”。

    注意：
    - 这不是“国家队真值回测”，只是 proxy 信号的有效性检查。
    - ETF资金流接口一般只给近 ~120 交易日，所以回测窗口天然受限。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas 未安装") from exc

    flow_dir = Path("data") / "cache" / "etf_flow"
    series_list: list[Any] = []
    errors: list[str] = []
    for sym in watchlist:
        out = fetch_etf_main_flow_hist(symbol=sym, cache_dir=flow_dir, ttl_hours=float(cache_ttl_hours))
        if not bool(out.get("ok")):
            errors.append(f"{sym}: {out.get('error')}")
            continue
        df = out.get("df")
        if df is None or getattr(df, "empty", True) or "日期" not in df.columns or "主力净流入-净额" not in df.columns:
            errors.append(f"{sym}: df 为空或缺列")
            continue
        dfx = df.copy()
        dfx["日期"] = pd.to_datetime(dfx["日期"], errors="coerce")
        dfx = dfx.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
        dfx["主力净流入-净额"] = pd.to_numeric(dfx["主力净流入-净额"], errors="coerce")
        s = dfx.set_index(dfx["日期"].dt.date)["主力净流入-净额"]
        s = s[~s.index.duplicated(keep="last")]
        series_list.append(s.rename(_normalize_symbol_etf(sym)))

    if not series_list:
        return {"ok": False, "error": "无可用 ETF 资金流历史", "errors": errors}

    df_all = pd.concat(series_list, axis=1).sort_index()
    sum_series = df_all.sum(axis=1, skipna=True)
    sum_series = sum_series.dropna()

    if start is not None:
        sum_series = sum_series[sum_series.index >= start]
    if end is not None:
        sum_series = sum_series[sum_series.index <= end]

    if sum_series.empty:
        return {"ok": False, "error": "过滤后资金流序列为空(日期范围太早/太窄？)", "errors": errors}

    # rolling robust z-score（只用过去窗口，避免未来函数）
    z_vals: list[float | None] = []
    score01_vals: list[float | None] = []
    xs = sum_series.to_list()
    idx = list(sum_series.index)
    win = max(8, int(lookback_days))
    for i, x in enumerate(xs):
        j0 = max(0, i - win + 1)
        hist = [float(v) for v in xs[j0 : i + 1] if _safe_float(v) is not None]
        z = robust_zscore(float(x), hist)
        z_vals.append(z)
        score01_vals.append(z_to_score01(z))

    flow_df = pd.DataFrame(
        {
            "date": idx,
            "sum_main_inflow_yuan": [float(v) for v in xs],
            "z_main_inflow": z_vals,
            "score01": score01_vals,
        }
    )

    # benchmark：指数次日收益（用 close->close）
    idx_sym = str(index_symbol or "").strip() or "sh000300"
    try:
        from .akshare_source import FetchParams
        from .data_cache import fetch_daily_cached

        df_idx = fetch_daily_cached(
            FetchParams(asset="index", symbol=str(idx_sym)),
            cache_dir=Path("data") / "cache" / "index",
            ttl_hours=float(cache_ttl_hours),
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:  # noqa: BLE001
        return {"ok": False, "error": f"指数数据获取失败：{exc}", "errors": errors, "flow_df": flow_df}

    if df_idx is None or getattr(df_idx, "empty", True) or "date" not in df_idx.columns or "close" not in df_idx.columns:
        return {"ok": False, "error": "指数数据为空/缺列", "errors": errors, "flow_df": flow_df}

    dfi = df_idx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")
    dfi = dfi.dropna(subset=["date"]).reset_index(drop=True)
    dfi["date_d"] = dfi["date"].dt.date
    dfi["close"] = pd.to_numeric(dfi["close"], errors="coerce")
    dfi = dfi.dropna(subset=["close"]).reset_index(drop=True)
    dfi["ret_1d_next"] = dfi["close"].shift(-1) / dfi["close"] - 1.0

    idx_df = dfi[["date_d", "close", "ret_1d_next"]].rename(columns={"date_d": "date", "close": "index_close", "ret_1d_next": "index_ret_1d_next"})

    merged = flow_df.merge(idx_df, on="date", how="left")

    # 简单统计：高/低分位的次日收益对比（score01 early days 可能是 None）
    m2 = merged.dropna(subset=["score01", "index_ret_1d_next"]).copy()
    if m2.empty:
        stats = {"note": "可用样本为空(可能指数对齐失败或样本太少)"}
    else:
        hi = m2[m2["score01"] >= 0.67]
        lo = m2[m2["score01"] <= 0.33]
        mid = m2[(m2["score01"] > 0.33) & (m2["score01"] < 0.67)]

        def _summ(df3):
            if df3 is None or df3.empty:
                return {"n": 0, "mean_next_ret": None, "median_next_ret": None}
            xs2 = [float(x) for x in df3["index_ret_1d_next"].to_list() if _safe_float(x) is not None]
            if not xs2:
                return {"n": int(len(df3)), "mean_next_ret": None, "median_next_ret": None}
            return {"n": int(len(df3)), "mean_next_ret": float(sum(xs2) / len(xs2)), "median_next_ret": float(_median(xs2))}

        stats = {"high": _summ(hi), "mid": _summ(mid), "low": _summ(lo), "all": _summ(m2)}

    return {
        "ok": True,
        "generated_at": datetime.now().isoformat(),
        "index_symbol": idx_sym,
        "watchlist": list(watchlist),
        "lookback_days": int(lookback_days),
        "range": {"start": str(start) if start else None, "end": str(end) if end else None},
        "stats": stats,
        "errors": errors,
        "df": merged,
        "disclaimer": "研究工具输出，不构成投资建议；仅评估ETF资金流 proxy 的统计表现。",
    }
