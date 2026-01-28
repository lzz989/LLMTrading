from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .akshare_source import FetchParams, fetch_daily, resolve_price_source
from .logger import get_logger
from .utils_time import as_yyyymmdd, parse_date_any


_LOG = get_logger(__name__)


def _prefer_tushare(params: FetchParams) -> bool:
    """
    决定“是否应该优先使用 TuShare”：
    - 仅当 params.source 显式为 tushare/auto 时，才允许走 TuShare（否则默认按 AkShare 口径缓存）
    - 且本地确实配置了 TUSHARE_TOKEN
    """
    src = resolve_price_source(getattr(params, "source", None), asset=str(getattr(params, "asset", "") or ""))
    if src not in {"auto", "tushare"}:
        return False
    try:
        from .tushare_source import load_tushare_env

        return load_tushare_env() is not None
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError):  # noqa: BLE001
        return False


def _has_non_ascii_columns(df) -> bool:
    try:
        # pandas Index 的 truthiness 会抛 ValueError，别用 `or []` 这种写法
        cols = list(getattr(df, "columns", []))
    except (TypeError, ValueError, AttributeError):  # noqa: BLE001
        return False
    for c in cols:
        s = str(c or "")
        if any(ord(ch) > 127 for ch in s):
            return True
    return False


def _read_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        if not meta_path.exists():
            return None
        txt = meta_path.read_text(encoding="utf-8")
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError, TypeError, AttributeError):  # noqa: BLE001
        return None


def _write_cache_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError, AttributeError):  # noqa: BLE001
        return None


def _attach_meta_to_df(df, meta: dict[str, Any] | None) -> None:
    if df is None or meta is None:
        return
    try:
        for k in (
            "data_source",
            "data_source_warning",
            "source_requested",
            "adjust",
            "asset",
            "symbol",
            "updated_at",
            "as_of",
            "intraday_unclosed",
        ):
            if k in meta:
                df.attrs[k] = meta[k]  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError):  # noqa: BLE001
        return


def _capture_df_meta(df) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        attrs = getattr(df, "attrs", {}) or {}
        if isinstance(attrs, dict):
            if attrs.get("data_source"):
                out["data_source"] = attrs.get("data_source")
            if attrs.get("data_source_warning"):
                out["data_source_warning"] = attrs.get("data_source_warning")
            if attrs.get("data_source_auto_fallback_error"):
                out["data_source_warning"] = attrs.get("data_source_auto_fallback_error")
    except (AttributeError, TypeError, ValueError):  # noqa: BLE001
        return out
    return out


def _should_expect_new_bar(now: datetime, *, asset: str) -> bool:
    a = str(asset or "").strip().lower()
    if a == "crypto":
        # 币圈 24/7：只要 ttl 过期就允许向后补（是否“真有新K线”由数据源自己决定）。
        return True

    # A股收盘 15:00；给点缓冲，15:05 前都当“未收盘”（避免盘中刷爆源站）
    if (now.hour, now.minute) < (15, 5):
        return False
    # 周末别折腾
    if now.weekday() >= 5:
        return False
    return True


def _expected_latest_bar_dt(now: datetime, *, asset: str) -> datetime | None:
    """
    估算“应该至少有到哪一天的收盘K线”（研究工具用，别当交易所日历）。

    目的：避免数据源抽风/网络波动导致缓存被写成“缺了最近几天”，但因为周末/盘中不触发 forward
    更新而一直卡住（尤其影响牛熊判定/风控）。
    """
    a = str(asset or "").strip().lower()
    if a == "crypto":
        return now

    # 对 A股/ETF/指数：我们只关心“日期级别”的最后收盘K线。
    # 保持为 00:00:00，避免用带时分秒的 datetime 去比较时，出现“同一天也被认为缺口”，
    # 从而在周末/盘中触发无意义的 forward_fill 抓数（并产生一堆 suppressed_errors 警告）。
    def _day(dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # 周末：A股最后一个交易日≈上一个周五
    if now.weekday() >= 5:
        # Sat(5)->Fri: -1；Sun(6)->Fri: -2
        delta = int(now.weekday() - 4)
        return _day(now - timedelta(days=max(0, delta)))

    # 交易日盘后：今天收盘K线应该可用了（给点缓冲）
    if (now.hour, now.minute) >= (15, 5):
        return _day(now)

    # 盘中：最新应该是“上一个交易日”
    dt = now - timedelta(days=1)
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    return _day(dt)


def fetch_daily_cached(params: FetchParams, *, cache_dir: Path, ttl_hours: float) -> Any:
    """
    增量 CSV 缓存：把过去K线落盘，后续只补“新K线”，别每次都全量拉历史把源站薅秃。

    - cache key: {asset}_{symbol}_{adjust}.csv
    - ttl_hours: 0 表示不使用缓存（每次强制拉取）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    adjust = params.adjust if params.adjust is not None else "qfq"
    key = f"{params.asset}_{params.symbol}_{adjust}.csv".replace("/", "_").replace("\\", "_")
    path = cache_dir / key
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_cache = _read_cache_meta(meta_path)
    last_fetch_meta: dict[str, Any] = {}

    want_start = parse_date_any(params.start_date) if params.start_date else None
    want_end = parse_date_any(params.end_date) if params.end_date else None
    suppressed: list[dict[str, str]] = []  # 吞错也得留痕：否则结果默默不对你还以为自己很稳

    def _normalize(df):
        if df is None or getattr(df, "empty", True):
            return df
        df2 = df.copy()
        if "date" in df2.columns:
            df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
            df2 = df2.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df2

    def _slice(df):
        if df is None or getattr(df, "empty", True):
            return df
        df2 = df
        if want_start is not None and "date" in df2.columns:
            df2 = df2[df2["date"] >= want_start]
        if want_end is not None and "date" in df2.columns:
            df2 = df2[df2["date"] <= want_end]
        return df2.reset_index(drop=True)

    df_cache = None
    cache_ok = False
    cache_min = None
    cache_max = None
    if path.exists() and float(ttl_hours) != 0:
        try:
            df_cache = _normalize(pd.read_csv(path, encoding="utf-8"))
            if df_cache is not None and (not getattr(df_cache, "empty", True)) and "date" in df_cache.columns:
                cache_min = df_cache["date"].iloc[0]
                cache_max = df_cache["date"].iloc[-1]
                cache_ok = True
            _attach_meta_to_df(df_cache, meta_cache)
        except (KeyError, IndexError, AttributeError) as exc:  # noqa: BLE001
            df_cache = None
            cache_ok = False
            suppressed.append({"stage": "read_cache", "error": str(exc)})

    # 如果用户显式要求走 TuShare/auto，但缓存明显是 AkShare 的“中文列”口径，
    # 我们会做一次“尾部迁移”（只刷新最近一段，再把缓存裁成标准列）。
    # 原因：不同数据源收盘价可能有细小差异；持仓风控/对账更需要口径一致。
    migrate_to_tushare = bool(cache_ok and _prefer_tushare(params) and _has_non_ascii_columns(df_cache))

    # ttl 未过期且不缺区间：直接用缓存
    if cache_ok and float(ttl_hours) > 0:
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:  # noqa: BLE001
            age = 9e18

        need_backfill = bool(want_start is not None and cache_min is not None and pd.Timestamp(cache_min) > pd.Timestamp(want_start))
        now = datetime.now()
        need_forward = False
        if want_end is not None and cache_max is not None:
            need_forward = bool(pd.Timestamp(cache_max) < pd.Timestamp(want_end))
        elif want_end is None and cache_max is not None:
            # 没指定 end_date：默认要“最新收盘”（即便是周末，也至少要补到上一个交易日）
            try:
                exp = _expected_latest_bar_dt(now, asset=str(params.asset))
                exp_date = exp.date() if exp is not None else None
                if exp_date is not None and pd.Timestamp(cache_max).date() < exp_date:
                    need_forward = True
            except (AttributeError):  # noqa: BLE001
                need_forward = False

        # migrate_to_tushare：强制走后续“尾部迁移/刷新”，别直接返回旧口径。
        if (not migrate_to_tushare) and (not need_backfill) and (not need_forward) and age <= float(ttl_hours) * 3600.0:
            return _slice(df_cache)

    # ttl=0：完全不走缓存，直接拉取（保留旧语义）
    if float(ttl_hours) == 0:
        return fetch_daily(params)

    # 下面开始“增量补齐”：能用缓存就尽量只补缺口
    df_work = df_cache if cache_ok else None

    def _fetch_range(*, start_dt: datetime | None, end_dt: datetime | None):
        p2 = FetchParams(asset=params.asset, symbol=params.symbol, adjust=params.adjust, source=getattr(params, "source", None))
        if start_dt is not None:
            p2 = FetchParams(
                asset=p2.asset,
                symbol=p2.symbol,
                adjust=p2.adjust,
                source=getattr(p2, "source", None),
                start_date=as_yyyymmdd(start_dt),
                end_date=p2.end_date,
            )
        if end_dt is not None:
            p2 = FetchParams(
                asset=p2.asset,
                symbol=p2.symbol,
                adjust=p2.adjust,
                source=getattr(p2, "source", None),
                start_date=p2.start_date,
                end_date=as_yyyymmdd(end_dt),
            )
        df2 = fetch_daily(p2)
        last_fetch_meta.update(_capture_df_meta(df2))
        return df2

    # 额外：把 AkShare 的“中文列口径”缓存迁移到 TuShare/标准列（只刷尾部，避免全量重拉）
    if migrate_to_tushare and cache_ok and df_work is not None and cache_max is not None:
        try:
            cache_max_dt = pd.Timestamp(cache_max).to_pydatetime()
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            cache_max_dt = None

        if cache_max_dt is not None:
            try:
                # 刷新最近 ~6 个月（够覆盖常见的“价格微调/口径差异修正”）
                refresh_days = 180
                start_dt = cache_max_dt - timedelta(days=int(refresh_days))
                df_patch = _normalize(_fetch_range(start_dt=start_dt, end_dt=cache_max_dt))
                if df_patch is not None and (not getattr(df_patch, "empty", True)):
                    df_work = pd.concat([df_work, df_patch], ignore_index=True)

                # 统一成最小必要列，避免“同一文件里混中文列/英文列”导致 downstream 口径分裂
                keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df_work.columns]
                if keep_cols:
                    df_work = df_work[keep_cols].copy()
            except (AttributeError) as exc:  # noqa: BLE001
                suppressed.append({"stage": "migrate_tail_refresh", "error": str(exc)})

    # 1) 向前补：如果用户要更早的区间，而缓存起点更晚
    if cache_ok and want_start is not None and cache_min is not None:
        try:
            cache_min_dt = pd.Timestamp(cache_min).to_pydatetime()
        except (AttributeError):  # noqa: BLE001
            cache_min_dt = None

        if cache_min_dt is not None and cache_min_dt.date() > want_start.date():
            try:
                end_dt = cache_min_dt - timedelta(days=1)
                if end_dt.date() >= want_start.date():
                    df_pre = _normalize(_fetch_range(start_dt=want_start, end_dt=end_dt))
                    if df_pre is not None and (not getattr(df_pre, "empty", True)):
                        df_work = pd.concat([df_pre, df_work], ignore_index=True) if df_work is not None else df_pre
            except Exception as exc:  # noqa: BLE001
                suppressed.append({"stage": "backfill", "error": str(exc)})

    # 2) 向后补：如果缓存没有覆盖到“想要的 end”，或盘后默认想要最新收盘
    need_forward2 = False
    desired_end = want_end
    if desired_end is None:
        # 不管是不是周末/盘中，都给一个“应该至少到哪天”的目标，用来判断缓存是否缺口
        desired_end = _expected_latest_bar_dt(datetime.now(), asset=str(params.asset))
    if cache_ok and desired_end is not None and cache_max is not None:
        try:
            if pd.Timestamp(cache_max) < pd.Timestamp(desired_end):
                need_forward2 = True
        except (AttributeError):  # noqa: BLE001
            need_forward2 = False
    elif (not cache_ok) and desired_end is not None:
        need_forward2 = True

    if need_forward2:
        start_dt = None
        if cache_ok and cache_max is not None:
            try:
                start_dt = pd.Timestamp(cache_max).to_pydatetime() + timedelta(days=1)
            except (AttributeError):  # noqa: BLE001
                start_dt = None
        try:
            df_new = _normalize(_fetch_range(start_dt=start_dt, end_dt=desired_end if want_end is not None else None))
            if df_new is not None and (not getattr(df_new, "empty", True)):
                df_work = pd.concat([df_work, df_new], ignore_index=True) if df_work is not None else df_new
        except Exception as exc:  # noqa: BLE001
            # 拉新数据失败：能用旧缓存就先用旧的，别炸全流程
            suppressed.append({"stage": "forward_fill", "error": str(exc)})

    # 3) 如果最终还是没有，就全量拉一次（兜底）
    if df_work is None or getattr(df_work, "empty", True):
        df_full = fetch_daily(params)
        last_fetch_meta.update(_capture_df_meta(df_full))
        df_work = _normalize(df_full)

    # 4) 去重/排序并写回缓存
    try:
        if df_work is not None and (not getattr(df_work, "empty", True)) and "date" in df_work.columns:
            df_work = df_work.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        df_work.to_csv(path, index=False, encoding="utf-8")
    except (AttributeError) as exc:  # noqa: BLE001
        suppressed.append({"stage": "write_cache", "error": str(exc)})

    # 写 cache meta（不影响主流程）
    try:
        meta = meta_cache if isinstance(meta_cache, dict) else {}
        if last_fetch_meta:
            meta.update(last_fetch_meta)
        meta.update(
            {
                "asset": str(params.asset),
                "symbol": str(params.symbol),
                "adjust": str(adjust),
                "source_requested": resolve_price_source(getattr(params, "source", None), asset=str(params.asset)),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        try:
            if df_work is not None and (not getattr(df_work, "empty", True)) and "date" in df_work.columns:
                last_dt = df_work["date"].iloc[-1]
                meta["as_of"] = str(last_dt.date()) if hasattr(last_dt, "date") else str(last_dt)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass
        # 如果用户显式传 end_date，且当前仍未收盘，标记可能是盘中价
        intraday_unclosed = False
        try:
            if getattr(params, "end_date", None):
                now = datetime.now()
                if (now.hour, now.minute) < (15, 5):
                    last_date = None
                    if df_work is not None and "date" in df_work.columns:
                        last_dt = df_work["date"].iloc[-1]
                        last_date = last_dt.date() if hasattr(last_dt, "date") else None
                    if last_date == now.date():
                        intraday_unclosed = True
        except (AttributeError, TypeError, ValueError, KeyError, IndexError):  # noqa: BLE001
            intraday_unclosed = False
        meta["intraday_unclosed"] = bool(intraday_unclosed)
        _write_cache_meta(meta_path, meta)
        _attach_meta_to_df(df_work, meta)
    except (OSError, ValueError, TypeError, AttributeError):  # noqa: BLE001
        pass

    # 吞错也要让人看见：默认只打一条 warning，避免扫描时刷屏到死。
    if suppressed:
        try:
            _LOG.warning(
                "[cache] %s %s: suppressed_errors=%d last_stage=%s last_error=%s",
                str(params.asset),
                str(params.symbol),
                int(len(suppressed)),
                str(suppressed[-1].get("stage")),
                str(suppressed[-1].get("error")),
            )
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            # 记录日志失败也别影响主流程（但这属于“极端垃圾环境”）
            pass
        try:
            # attrs 仅用于审计/排查；不影响下游逻辑（也别写太多，顶多留前几条）
            df_work.attrs["cache_warnings"] = suppressed[:5]  # type: ignore[union-attr]
            df_work.attrs["cache_warnings_count"] = int(len(suppressed))  # type: ignore[union-attr]
        except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
            pass

    return _slice(df_work)
