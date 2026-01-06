from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


class DataSourceError(RuntimeError):
    pass


AssetType = Literal["etf", "index", "stock"]


@dataclass(frozen=True)
class FetchParams:
    asset: AssetType
    symbol: str
    start_date: str | None = None  # YYYYMMDD or YYYY-MM-DD
    end_date: str | None = None  # YYYYMMDD or YYYY-MM-DD
    adjust: str | None = None  # only for stock; "", "qfq", "hfq"


def _require_akshare():
    try:
        import akshare  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DataSourceError(
            "没装 akshare？先把依赖装上：pip install -r \"requirements.txt\""
        ) from exc


def _parse_date_any(s: str) -> datetime:
    s2 = s.strip()
    if not s2:
        raise ValueError("空日期")
    if "-" in s2:
        return datetime.strptime(s2, "%Y-%m-%d")
    return datetime.strptime(s2, "%Y%m%d")


def _as_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _normalize_stock_name(name: str) -> str:
    s = str(name).strip().replace(" ", "")
    for ch in ["A", "a", "Ａ", "ａ"]:
        s = s.replace(ch, "")
    return s


def _normalize_symbol_with_prefix(symbol: str) -> str:
    sym = symbol.strip()
    if not sym:
        raise DataSourceError("symbol 为空，别闹。")
    return sym.lower()


def _try_fetch_etf(sym: str):
    import akshare as ak

    code = sym.strip().lower()
    if code.startswith(("sh", "sz")):
        code = code[2:]

    # 东财（日线）一般比 Sina 更新得快；Sina 当兜底
    df = None
    try:
        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date="19700101",
            end_date=_as_yyyymmdd(datetime.now()),
            adjust="",
        )
        # 统一列名（东财返回中文列名）
        if df is not None and (not getattr(df, "empty", True)) and "日期" in df.columns:
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
    except Exception:  # noqa: BLE001
        df = None

    if df is None or getattr(df, "empty", True):
        df = ak.fund_etf_hist_sina(symbol=sym)
    return df


def _try_fetch_index(sym: str):
    import akshare as ak

    return ak.stock_zh_index_daily(symbol=sym)


def _try_fetch_stock(sym: str, *, start_date: str, end_date: str, adjust: str):
    import akshare as ak

    code = sym.strip().lower()
    if code.startswith(("sh", "sz", "bj")):
        code = code[2:]

    # 东财接口更稳；Sina/腾讯这些经常抽风或被反爬
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
    except Exception:  # noqa: BLE001
        df = ak.stock_zh_a_daily(symbol=sym, start_date=start_date, end_date=end_date, adjust=adjust)

    # 统一列名（东财返回中文列名）
    if df is not None and (not getattr(df, "empty", True)) and "日期" in df.columns:
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
    return df


def resolve_symbol(asset: AssetType, symbol: str) -> str:
    sym = _normalize_symbol_with_prefix(symbol)
    if sym.startswith(("sh", "sz")):
        return sym
    if sym.startswith("bj"):
        return sym

    if asset == "stock":
        if sym.isdigit():
            if len(sym) != 6:
                raise DataSourceError(f"股票代码必须是 6 位数字：{symbol}")
            if sym.startswith("6"):
                return f"sh{sym}"
            if sym.startswith(("0", "3")):
                return f"sz{sym}"
            if sym.startswith("9"):
                return f"bj{sym}"
            raise DataSourceError(f"暂不支持的股票代码前缀：{symbol}")

        target = _normalize_stock_name(symbol)
        if not target:
            raise DataSourceError("股票名称为空，别闹。")

        try:
            import akshare as ak
        except ModuleNotFoundError as exc:
            raise DataSourceError("akshare 未安装，没法按名称解析股票。") from exc

        # 深市
        try:
            df_sz = ak.stock_info_sz_name_code()
            df_sz["_n"] = df_sz["A股简称"].map(_normalize_stock_name)
            hit_sz = df_sz[df_sz["_n"] == target]
            if len(hit_sz) == 1:
                code = str(hit_sz.iloc[0]["A股代码"]).zfill(6)
                return f"sz{code}"
        except Exception:  # noqa: BLE001
            pass

        # 沪市
        try:
            df_sh = ak.stock_info_sh_name_code()
            df_sh["_n"] = df_sh["公司简称"].map(_normalize_stock_name)
            hit_sh = df_sh[df_sh["_n"] == target]
            if len(hit_sh) == 1:
                code = str(hit_sh.iloc[0]["公司代码"]).zfill(6)
                return f"sh{code}"
        except Exception:  # noqa: BLE001
            pass

        # 北交所
        try:
            df_bj = ak.stock_info_bj_name_code()
            df_bj["_n"] = df_bj["证券简称"].map(_normalize_stock_name)
            hit_bj = df_bj[df_bj["_n"] == target]
            if len(hit_bj) == 1:
                code = str(hit_bj.iloc[0]["证券代码"]).zfill(6)
                return f"bj{code}"
        except Exception:  # noqa: BLE001
            pass

        raise DataSourceError(f"按名称找不到股票：{symbol}（建议直接传 000725 或 sz000725 这种代码）")

    if not sym.isdigit():
        raise DataSourceError(f"不认识的 symbol：{symbol}；ETF/指数请用 6 位数字或 sh/sz 前缀。")

    if len(sym) != 6:
        raise DataSourceError(f"代码必须是 6 位数字：{symbol}")

    if asset == "etf":
        # ETF 市场前缀按代码规则即可：5xxxx => 沪；其他常见 => 深
        return f"sh{sym}" if sym.startswith("5") else f"sz{sym}"

    candidates = [f"sh{sym}", f"sz{sym}"]
    for cand in candidates:
        try:
            if asset == "index":
                df = _try_fetch_index(cand)
            else:
                raise DataSourceError(f"不支持的 asset: {asset}")
        except Exception:  # noqa: BLE001
            continue
        if getattr(df, "empty", True):
            continue
        return cand

    raise DataSourceError(f"symbol 解析失败：{symbol}（尝试了 {candidates} 都拿不到数据）")


def fetch_daily(params: FetchParams):
    _require_akshare()
    asset = params.asset
    symbol = resolve_symbol(asset, params.symbol)

    if asset == "etf":
        df = _try_fetch_etf(symbol)
    elif asset == "index":
        df = _try_fetch_index(symbol)
    elif asset == "stock":
        start_dt = _parse_date_any(params.start_date) if params.start_date else None
        end_dt = _parse_date_any(params.end_date) if params.end_date else None
        start = _as_yyyymmdd(start_dt) if start_dt else "19900101"
        end = _as_yyyymmdd(end_dt) if end_dt else _as_yyyymmdd(datetime.now())
        adjust = params.adjust if params.adjust is not None else "qfq"
        df = _try_fetch_stock(symbol, start_date=start, end_date=end, adjust=adjust)
    else:
        raise DataSourceError(f"不支持的 asset: {asset}")

    if df.empty:
        raise DataSourceError(f"没抓到数据：{asset} {symbol}")

    import pandas as pd

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if params.start_date:
        start_dt2 = _parse_date_any(params.start_date)
        df = df[df["date"] >= start_dt2]
    if params.end_date:
        end_dt2 = _parse_date_any(params.end_date)
        df = df[df["date"] <= end_dt2]

    df = df.reset_index(drop=True)
    if df.empty:
        raise DataSourceError(f"时间区间过滤后无数据：{asset} {symbol}")

    return df
