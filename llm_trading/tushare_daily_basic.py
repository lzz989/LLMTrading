from __future__ import annotations

"""
TuShare 日频基础数据（daily_basic）缓存封装（研究用途）。

用途：
- 给“换手率驱动的成本分布(筹码峰近似)”提供 turnover_rate（%）

注意：
- daily_basic 是“每交易日一条”，字段多；我们只取最小子集，避免薅接口。
- 这是研究工具，不保证每次都有数据；拿不到就降级为“无 cost 分布”。
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import get_logger
from .tushare_source import TushareSourceError, get_pro_api, load_tushare_env, prefixed_symbol_to_ts_code
from .utils_time import as_yyyymmdd, parse_date_any


_LOG = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DailyBasicPack:
    ok: bool
    as_of: str | None
    df: Any | None
    error: str | None = None


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        if not meta_path.exists():
            return None
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError, AttributeError):
        return


def fetch_daily_basic_turnover_rate_tushare(*, ts_code: str, start_date: str, end_date: str):
    """
    直接调用 TuShare daily_basic，返回 DataFrame（trade_date, turnover_rate）。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：pandas") from exc

    env = load_tushare_env()
    if env is None:
        raise TushareSourceError("缺少环境变量 TUSHARE_TOKEN（在 .env 里配，别往聊天里发）")
    pro = get_pro_api(env)

    # 只取最小字段集合
    try:
        df = pro.daily_basic(
            ts_code=str(ts_code),
            start_date=str(start_date),
            end_date=str(end_date),
            fields="ts_code,trade_date,turnover_rate",
        )
    except Exception as exc:  # noqa: BLE001
        raise TushareSourceError(f"daily_basic 拉取失败：{exc}") from exc

    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["date", "turnover_rate"])

    dfx = df.copy()
    # trade_date: YYYYMMDD
    if "trade_date" not in dfx.columns:
        return pd.DataFrame(columns=["date", "turnover_rate"])
    dfx["date"] = pd.to_datetime(dfx["trade_date"], format="%Y%m%d", errors="coerce")
    dfx["turnover_rate"] = pd.to_numeric(dfx.get("turnover_rate"), errors="coerce").astype(float)
    dfx = dfx.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return dfx[["date", "turnover_rate"]]


def fetch_turnover_rate_cached(
    *,
    symbol: str,
    cache_dir: Path,
    ttl_hours: float,
    start_date: str | None,
    end_date: str | None,
) -> DailyBasicPack:
    """
    获取 turnover_rate（%）序列（按 date）并做本地缓存。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError:  # pragma: no cover
        return DailyBasicPack(ok=False, as_of=None, df=None, error="缺依赖：pandas")

    sym = str(symbol or "").strip().lower()
    if not sym:
        return DailyBasicPack(ok=False, as_of=None, df=None, error="symbol 为空")

    ts_code = prefixed_symbol_to_ts_code(sym)
    if ts_code is None:
        return DailyBasicPack(ok=False, as_of=None, df=None, error=f"symbol 无法转 ts_code：{symbol}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"stock_basic_turnover_{sym}.csv".replace("/", "_").replace("\\", "_")
    path = cache_dir / key
    meta_path = path.with_suffix(path.suffix + ".meta.json")

    # 解析日期范围（尽量少拉）
    sd = parse_date_any(start_date) if start_date else None
    ed = parse_date_any(end_date) if end_date else None
    s = as_yyyymmdd(sd) if sd else "19900101"
    e = as_yyyymmdd(ed) if ed else as_yyyymmdd(datetime.now())

    # 缓存命中：ttl 未过期且 as_of 覆盖到 end_date
    if path.exists() and float(ttl_hours) > 0:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 9e18
        if age <= float(ttl_hours) * 3600.0:
            meta = _read_meta(meta_path) or {}
            as_of = str(meta.get("as_of") or "").strip() or None
            try:
                df0 = pd.read_csv(path, encoding="utf-8")
                if "date" in df0.columns:
                    df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
                df0 = df0.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                df0 = None
                _LOG.warning("[cache] stock_basic turnover read failed: %s", exc)
            if df0 is not None and (not getattr(df0, "empty", True)):
                try:
                    last_dt = df0["date"].iloc[-1]
                    last_s = as_yyyymmdd(last_dt)
                except Exception:  # noqa: BLE001
                    last_s = None
                if last_s is not None and last_s >= e:
                    return DailyBasicPack(ok=True, as_of=as_of or last_s, df=df0, error=None)

    # 缓存未命中：拉取
    try:
        df = fetch_daily_basic_turnover_rate_tushare(ts_code=ts_code, start_date=s, end_date=e)
        as_of = None
        try:
            if df is not None and (not getattr(df, "empty", True)):
                as_of = as_yyyymmdd(df["date"].iloc[-1])
        except Exception:  # noqa: BLE001
            as_of = None
        try:
            df.to_csv(path, index=False, encoding="utf-8")
            _write_meta(
                meta_path,
                {
                    "symbol": sym,
                    "ts_code": ts_code,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "as_of": as_of,
                    "fields": ["date", "turnover_rate"],
                },
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("[cache] stock_basic turnover write failed: %s", exc)
        return DailyBasicPack(ok=True, as_of=as_of, df=df, error=None)
    except (TushareSourceError, RuntimeError, ValueError, TypeError) as exc:
        return DailyBasicPack(ok=False, as_of=None, df=None, error=str(exc))

