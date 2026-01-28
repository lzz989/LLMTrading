from __future__ import annotations

from typing import Literal


class TushareKlineError(RuntimeError):
    pass


AdjustType = Literal["", "qfq", "hfq"]


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareKlineError("缺依赖：pandas 未安装") from exc
    return pd


def _normalize_adjust(adjust: str | None) -> AdjustType:
    a = str(adjust or "").strip().lower()
    if a in {"", "0", "none", "raw"}:
        return ""
    if a in {"qfq", "hfq"}:
        return a  # type: ignore[return-value]
    # 别整花活：复权就 qfq/hfq，不复权就空
    return "qfq"


def _amount_to_yuan(amount_col):
    """
    TuShare Pro 日线 amount 通常是“千元”，这里统一换算为“元”。
    """
    try:
        import pandas as pd

        s = pd.to_numeric(amount_col, errors="coerce")
        return s * 1000.0
    except (AttributeError):  # noqa: BLE001
        return amount_col


def fetch_stock_daily_tushare(*, ts_code: str, start_date: str, end_date: str, adjust: str | None) -> object:
    """
    TuShare A股日线：
    - pro.daily: OHLCV（未复权）
    - pro.adj_factor: 复权因子（用于 qfq/hfq）

    返回 DataFrame（列：date/open/high/low/close/volume/amount），按 date 升序。
    """
    pd = _require_pandas()

    from .tushare_source import get_pro_api

    a = _normalize_adjust(adjust)
    pro = get_pro_api()

    try:
        df = pro.daily(ts_code=str(ts_code), start_date=str(start_date), end_date=str(end_date))
    except (AttributeError) as exc:  # noqa: BLE001
        raise TushareKlineError(f"TuShare daily 调用失败：{exc}") from exc

    if df is None or getattr(df, "empty", True):
        raise TushareKlineError(f"TuShare daily 返回空：{ts_code}")

    dfx = df.copy()
    if "trade_date" not in dfx.columns:
        raise TushareKlineError("TuShare daily 缺少 trade_date 列")

    # 统一字段
    if "vol" in dfx.columns and "volume" not in dfx.columns:
        dfx = dfx.rename(columns={"vol": "volume"})
    dfx = dfx.rename(columns={"trade_date": "date"})

    dfx["date"] = pd.to_datetime(dfx["date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    if "amount" in dfx.columns:
        dfx["amount"] = _amount_to_yuan(dfx["amount"])

    dfx = dfx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # 复权：只调价格，成交量/成交额不动（够用就行，别在这搞论文级别的“复权成交量”玄学）
    if a in {"qfq", "hfq"} and (not dfx.empty):
        try:
            df_af = pro.adj_factor(ts_code=str(ts_code), start_date=str(start_date), end_date=str(end_date))
        except (AttributeError) as exc:  # noqa: BLE001
            raise TushareKlineError(f"TuShare adj_factor 调用失败：{exc}") from exc

        if df_af is None or getattr(df_af, "empty", True):
            raise TushareKlineError(f"TuShare adj_factor 返回空：{ts_code}")
        if "trade_date" not in df_af.columns or "adj_factor" not in df_af.columns:
            raise TushareKlineError("TuShare adj_factor 缺少 trade_date/adj_factor 列")

        af = df_af[["trade_date", "adj_factor"]].copy()
        af["trade_date"] = pd.to_datetime(af["trade_date"], format="%Y%m%d", errors="coerce")
        af["adj_factor"] = pd.to_numeric(af["adj_factor"], errors="coerce")
        af = af.dropna(subset=["trade_date", "adj_factor"]).sort_values("trade_date").reset_index(drop=True)
        if af.empty:
            raise TushareKlineError(f"TuShare adj_factor 清洗后为空：{ts_code}")

        # 对齐到日线
        dfx2 = dfx.merge(af, how="left", left_on="date", right_on="trade_date")
        dfx2 = dfx2.drop(columns=["trade_date"])
        dfx2["adj_factor"] = pd.to_numeric(dfx2["adj_factor"], errors="coerce")

        # adj_factor 有时会缺一小段“最早期”数据（常见：上市初期），别把整个流程炸了：
        # - 如果缺失只发生在开头：直接丢掉那段（我们也用不着那么早的复权K线）
        # - 如果中间/末尾也缺：先尝试前向填充；还缺就报错（避免把错误数据当真）
        if dfx2["adj_factor"].isna().any():
            first_valid = dfx2["adj_factor"].first_valid_index()
            if first_valid is None:
                raise TushareKlineError(f"TuShare adj_factor 全缺失：{ts_code}")

            tail_has_na = bool(dfx2["adj_factor"].iloc[int(first_valid) :].isna().any())
            if not tail_has_na:
                dfx2 = dfx2.iloc[int(first_valid) :].reset_index(drop=True)
            else:
                dfx2["adj_factor"] = dfx2["adj_factor"].ffill()
                if dfx2["adj_factor"].isna().any():
                    raise TushareKlineError(f"TuShare adj_factor 对齐失败：{ts_code}（存在缺失因子）")

        base = float(dfx2["adj_factor"].iloc[-1]) if a == "qfq" else float(dfx2["adj_factor"].iloc[0])
        if base == 0:
            raise TushareKlineError(f"TuShare adj_factor 基准为 0：{ts_code}")

        ratio = dfx2["adj_factor"] / base
        for c in ["open", "high", "low", "close"]:
            if c in dfx2.columns:
                dfx2[c] = pd.to_numeric(dfx2[c], errors="coerce") * ratio
        dfx = dfx2.drop(columns=["adj_factor"])

    # 输出最小必要列
    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in dfx.columns]
    out = dfx[keep].copy()
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_index_daily_tushare(*, ts_code: str, start_date: str, end_date: str) -> object:
    """
    TuShare 指数日线（index_daily）。
    返回 DataFrame（列：date/open/high/low/close/volume/amount），按 date 升序。
    """
    pd = _require_pandas()

    from .tushare_source import get_pro_api

    pro = get_pro_api()
    try:
        df = pro.index_daily(ts_code=str(ts_code), start_date=str(start_date), end_date=str(end_date))
    except Exception as exc:  # noqa: BLE001
        raise TushareKlineError(f"TuShare index_daily 调用失败：{exc}") from exc

    if df is None or getattr(df, "empty", True):
        raise TushareKlineError(f"TuShare index_daily 返回空：{ts_code}")

    dfx = df.copy()
    if "trade_date" not in dfx.columns:
        raise TushareKlineError("TuShare index_daily 缺少 trade_date 列")

    if "vol" in dfx.columns and "volume" not in dfx.columns:
        dfx = dfx.rename(columns={"vol": "volume"})
    dfx = dfx.rename(columns={"trade_date": "date"})

    dfx["date"] = pd.to_datetime(dfx["date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    if "amount" in dfx.columns:
        dfx["amount"] = _amount_to_yuan(dfx["amount"])

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in dfx.columns]
    out = dfx[keep].copy()
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_etf_daily_tushare(*, ts_code: str, start_date: str, end_date: str) -> object:
    """
    TuShare ETF/场内基金日线（fund_daily）。

    注意：
    - TuShare 这里没有 qfq/hfq 参数；如你对复权连续性特别敏感，ETF 仍建议走 AkShare 的 EM 接口。
    """
    pd = _require_pandas()

    from .tushare_source import get_pro_api

    pro = get_pro_api()
    try:
        df = pro.fund_daily(ts_code=str(ts_code), start_date=str(start_date), end_date=str(end_date))
    except Exception as exc:  # noqa: BLE001
        raise TushareKlineError(f"TuShare fund_daily 调用失败：{exc}") from exc

    if df is None or getattr(df, "empty", True):
        raise TushareKlineError(f"TuShare fund_daily 返回空：{ts_code}")

    dfx = df.copy()
    if "trade_date" not in dfx.columns:
        raise TushareKlineError("TuShare fund_daily 缺少 trade_date 列")

    if "vol" in dfx.columns and "volume" not in dfx.columns:
        dfx = dfx.rename(columns={"vol": "volume"})
    dfx = dfx.rename(columns={"trade_date": "date"})

    dfx["date"] = pd.to_datetime(dfx["date"], format="%Y%m%d", errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    if "amount" in dfx.columns:
        dfx["amount"] = _amount_to_yuan(dfx["amount"])

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in dfx.columns]
    out = dfx[keep].copy()
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out
