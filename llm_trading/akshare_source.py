from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .logger import get_logger
from .utils_time import as_yyyymmdd, parse_date_any


_LOG = get_logger(__name__)
_AUTO_FALLBACK_WARNED: set[str] = set()


def _warn_auto_fallback_once(*, asset: str, symbol: str, err: str) -> None:
    """
    source=auto 时，TuShare 失败会回退到 AkShare。
    这不算致命错误，但如果不提示，你会以为自己一直在用 TuShare 口径（结果对不上券商/软件）。

    扫描场景下别刷屏：每个 asset 只警告一次（进程内）。
    """
    k = str(asset or "").strip().lower() or "unknown"
    if k in _AUTO_FALLBACK_WARNED:
        return
    _AUTO_FALLBACK_WARNED.add(k)
    try:
        _LOG.warning(
            "[source=auto] TuShare 抓取失败，已回退 AkShare（asset=%s symbol=%s；本进程仅提示一次）。err=%s",
            str(asset),
            str(symbol),
            str(err),
        )
    except (AttributeError):  # noqa: BLE001
        pass

class DataSourceError(RuntimeError):
    pass


AssetType = Literal["etf", "index", "stock", "crypto"]


@dataclass(frozen=True)
class FetchParams:
    asset: AssetType
    symbol: str
    start_date: str | None = None  # YYYYMMDD or YYYY-MM-DD
    end_date: str | None = None  # YYYYMMDD or YYYY-MM-DD
    adjust: str | None = None  # stock/etf: "", "qfq", "hfq"
    source: str | None = None  # "akshare" / "tushare" / "auto"


def _require_akshare():
    try:
        import akshare  # noqa: F401
    except ModuleNotFoundError as exc:
        raise DataSourceError(
            "没装 akshare？先把依赖装上：pip install -r \"requirements.txt\""
        ) from exc


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


def _try_fetch_etf(sym: str, *, start_date: str, end_date: str, adjust: str):
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
            start_date=str(start_date),
            end_date=str(end_date),
            adjust=str(adjust or "").strip(),
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
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        df = None

    # 某些标的/时段 qfq/hfq 可能抽风：再试一次“不复权”，能拿到数据先。
    if (df is None or getattr(df, "empty", True)) and str(adjust or "").strip() not in {"", "0", "none"}:
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=str(start_date),
                end_date=str(end_date),
                adjust="",
            )
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
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
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
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
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
    if asset == "crypto":
        s = sym.lower().strip()
        s = s.replace("/", "").replace("-", "").replace("_", "")
        # 先只支持 BTC（别tm一上来就全币圈，先把一件事做对）
        if s in {"btc", "xbt", "bitcoin", "btcusdt", "btcusd"}:
            return "btcusd"
        if s.startswith("btc") and s.endswith("usdt"):
            return "btcusd"
        if s.startswith("btc") and s.endswith("usd"):
            return "btcusd"
        raise DataSourceError(f"暂不支持的 crypto symbol：{symbol}（目前只支持 btc/btcusd/btcusdt）")
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
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass

        # 沪市
        try:
            df_sh = ak.stock_info_sh_name_code()
            df_sh["_n"] = df_sh["公司简称"].map(_normalize_stock_name)
            hit_sh = df_sh[df_sh["_n"] == target]
            if len(hit_sh) == 1:
                code = str(hit_sh.iloc[0]["公司代码"]).zfill(6)
                return f"sh{code}"
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass

        # 北交所
        try:
            df_bj = ak.stock_info_bj_name_code()
            df_bj["_n"] = df_bj["证券简称"].map(_normalize_stock_name)
            hit_bj = df_bj[df_bj["_n"] == target]
            if len(hit_bj) == 1:
                code = str(hit_bj.iloc[0]["证券代码"]).zfill(6)
                return f"bj{code}"
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
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
        except (DataSourceError, OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
            continue
        if getattr(df, "empty", True):
            continue
        return cand

    raise DataSourceError(f"symbol 解析失败：{symbol}（尝试了 {candidates} 都拿不到数据）")


def fetch_daily(params: FetchParams):
    asset = params.asset
    symbol = resolve_symbol(asset, params.symbol)
    # 默认走 AkShare（免费+ETF支持复权）。需要对齐券商/行情软件口径时，可显式传 source=auto/tushare。
    src = str(params.source or "").strip().lower() or "akshare"
    if src not in {"akshare", "tushare", "auto"}:
        src = "akshare"

    if asset == "etf":
        df = None
        auto_fallback_error: str | None = None
        # ETF 也可能发生折算/拆分；AkShare EM 接口支持 qfq/hfq，更适合做连续性分析。
        adjust = str(params.adjust).strip() if params.adjust is not None else "qfq"
        if adjust not in {"", "qfq", "hfq"}:
            adjust = "qfq"
        start_dt = parse_date_any(params.start_date) if params.start_date else None
        end_dt = parse_date_any(params.end_date) if params.end_date else None
        start = as_yyyymmdd(start_dt) if start_dt else "19700101"
        end = as_yyyymmdd(end_dt) if end_dt else as_yyyymmdd(datetime.now())

        if src in {"tushare", "auto"}:
            try:
                from .tushare_kline import TushareKlineError, fetch_etf_daily_tushare
                from .tushare_source import TushareSourceError, load_tushare_env, normalize_ts_code

                env = load_tushare_env()
                if env is not None:
                    ts_code = normalize_ts_code(symbol)
                    if ts_code is None:
                        raise DataSourceError(f"ETF symbol 解析失败：{symbol}")
                    df = fetch_etf_daily_tushare(ts_code=ts_code, start_date=start, end_date=end)
                    try:
                        df.attrs["data_source"] = "tushare"
                    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                        pass
                elif src == "tushare":
                    raise DataSourceError("缺少环境变量 TUSHARE_TOKEN（在 .env 里配，别往聊天里发）")
            except (TushareSourceError, TushareKlineError, DataSourceError) as exc:
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise
                df = None
            except Exception as exc:  # noqa: BLE001
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise DataSourceError(f"TuShare 抓取ETF失败：{exc}") from exc
                df = None

        if df is None:
            _require_akshare()
            df = _try_fetch_etf(symbol, start_date=start, end_date=end, adjust=adjust)
            try:
                df.attrs["data_source"] = "akshare"
            except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                pass
            if auto_fallback_error:
                _warn_auto_fallback_once(asset="etf", symbol=str(symbol), err=str(auto_fallback_error))
                try:
                    df.attrs["data_source_auto_fallback_error"] = str(auto_fallback_error)
                except (AttributeError):  # noqa: BLE001
                    pass
    elif asset == "index":
        df = None
        auto_fallback_error: str | None = None
        start_dt = parse_date_any(params.start_date) if params.start_date else None
        end_dt = parse_date_any(params.end_date) if params.end_date else None
        start = as_yyyymmdd(start_dt) if start_dt else "19900101"
        end = as_yyyymmdd(end_dt) if end_dt else as_yyyymmdd(datetime.now())

        if src in {"tushare", "auto"}:
            try:
                from .tushare_kline import TushareKlineError, fetch_index_daily_tushare
                from .tushare_source import TushareSourceError, load_tushare_env, prefixed_symbol_to_ts_code

                env = load_tushare_env()
                if env is not None:
                    ts_code = prefixed_symbol_to_ts_code(symbol)
                    if ts_code is None:
                        raise DataSourceError(f"指数 symbol 解析失败：{symbol}")
                    df = fetch_index_daily_tushare(ts_code=ts_code, start_date=start, end_date=end)
                    try:
                        df.attrs["data_source"] = "tushare"
                    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                        pass
                elif src == "tushare":
                    raise DataSourceError("缺少环境变量 TUSHARE_TOKEN（在 .env 里配，别往聊天里发）")
            except (TushareSourceError, TushareKlineError, DataSourceError) as exc:
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise
                df = None
            except Exception as exc:  # noqa: BLE001
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise DataSourceError(f"TuShare 抓取指数失败：{exc}") from exc
                df = None

        if df is None:
            _require_akshare()
            df = _try_fetch_index(symbol)
            try:
                df.attrs["data_source"] = "akshare"
            except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                pass
            if auto_fallback_error:
                _warn_auto_fallback_once(asset="index", symbol=str(symbol), err=str(auto_fallback_error))
                try:
                    df.attrs["data_source_auto_fallback_error"] = str(auto_fallback_error)
                except (AttributeError):  # noqa: BLE001
                    pass
    elif asset == "stock":
        start_dt = parse_date_any(params.start_date) if params.start_date else None
        end_dt = parse_date_any(params.end_date) if params.end_date else None
        start = as_yyyymmdd(start_dt) if start_dt else "19900101"
        end = as_yyyymmdd(end_dt) if end_dt else as_yyyymmdd(datetime.now())
        adjust = params.adjust if params.adjust is not None else "qfq"
        df = None
        auto_fallback_error: str | None = None

        if src in {"tushare", "auto"}:
            try:
                from .tushare_kline import TushareKlineError, fetch_stock_daily_tushare
                from .tushare_source import TushareSourceError, load_tushare_env, prefixed_symbol_to_ts_code

                env = load_tushare_env()
                if env is not None:
                    ts_code = prefixed_symbol_to_ts_code(symbol)
                    if ts_code is None:
                        raise DataSourceError(f"股票 symbol 解析失败：{symbol}")
                    df = fetch_stock_daily_tushare(ts_code=ts_code, start_date=start, end_date=end, adjust=adjust)
                    try:
                        df.attrs["data_source"] = "tushare"
                    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                        pass
                elif src == "tushare":
                    raise DataSourceError("缺少环境变量 TUSHARE_TOKEN（在 .env 里配，别往聊天里发）")
            except (TushareSourceError, TushareKlineError, DataSourceError) as exc:
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise
                df = None
            except Exception as exc:  # noqa: BLE001
                if src == "auto":
                    auto_fallback_error = str(exc)
                if src == "tushare":
                    raise DataSourceError(f"TuShare 抓取个股失败：{exc}") from exc
                df = None

        if df is None:
            _require_akshare()
            df = _try_fetch_stock(symbol, start_date=start, end_date=end, adjust=adjust)
            try:
                df.attrs["data_source"] = "akshare"
            except (TypeError, ValueError, AttributeError):  # noqa: BLE001
                pass
            if auto_fallback_error:
                _warn_auto_fallback_once(asset="stock", symbol=str(symbol), err=str(auto_fallback_error))
                try:
                    df.attrs["data_source_auto_fallback_error"] = str(auto_fallback_error)
                except (AttributeError):  # noqa: BLE001
                    pass
    elif asset == "crypto":
        try:
            from .crypto_source import fetch_crypto_daily_stooq
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:  # noqa: BLE001
            raise DataSourceError(f"crypto 数据源不可用：{exc}") from exc

        try:
            df = fetch_crypto_daily_stooq(
                symbol=str(symbol),
                start_date=str(params.start_date) if params.start_date else None,
                end_date=str(params.end_date) if params.end_date else None,
                timeout_sec=12.0,
            )
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
            raise DataSourceError(f"抓取 crypto 失败：{exc}") from exc
    else:
        raise DataSourceError(f"不支持的 asset: {asset}")

    if df.empty:
        raise DataSourceError(f"没抓到数据：{asset} {symbol}")

    import pandas as pd

    # attrs 在某些 pandas 版本/操作里会丢，这里尽量保留一下
    ds_name = None
    try:
        ds_name = getattr(df, "attrs", {}).get("data_source")
    except (AttributeError):  # noqa: BLE001
        ds_name = None

    df = df.copy()
    if ds_name:
        try:
            df.attrs["data_source"] = str(ds_name)
        except (AttributeError):  # noqa: BLE001
            pass
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # 交易时段里某些源会把“今日实时价”塞进日线里（date=今天，close=当前价），
    # 但你要的是“收盘价”（尤其是收盘后跑策略、次日执行），所以默认把未收盘的今天丢掉。
    # 只有当用户显式传了 end_date 才保留（他自己承担“实时K线不稳定”的后果）。
    if asset in {"etf", "stock", "index"} and (not params.end_date) and (not df.empty):
        try:
            now = datetime.now()
            last_dt = df.iloc[-1]["date"]
            last_date = last_dt.date() if hasattr(last_dt, "date") else None
            if last_date == now.date():
                # A股收盘 15:00；给数据源/网络点缓冲，15:05 前都当“未收盘”
                if (now.hour, now.minute) < (15, 5) and len(df) >= 2:
                    df = df.iloc[:-1].reset_index(drop=True)
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass

    if params.start_date:
        start_dt2 = parse_date_any(params.start_date)
        df = df[df["date"] >= start_dt2]
    if params.end_date:
        end_dt2 = parse_date_any(params.end_date)
        df = df[df["date"] <= end_dt2]

    df = df.reset_index(drop=True)
    if df.empty:
        raise DataSourceError(f"时间区间过滤后无数据：{asset} {symbol}")

    return df
