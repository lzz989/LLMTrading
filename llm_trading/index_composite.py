from __future__ import annotations

from pathlib import Path
from typing import Any


def _normalize_index_symbol(sym: str) -> str:
    s = str(sym or "").strip().lower()
    return s


def parse_index_combo_spec(spec: str) -> list[str]:
    """
    解析指数“合成基准”：
    - off/none/0/"" => []
    - "sh000300" => ["sh000300"]
    - "sh000300+sh000905" => ["sh000300", "sh000905"]（去重但保序）

    说明：这里只处理 '+'，别跟 market_regime 的逗号混一起（逗号那套是“多指数投票+canary”）。
    """
    raw = str(spec or "").strip()
    if raw.lower() in {"", "off", "none", "0"}:
        return []
    parts = [p.strip() for p in raw.split("+") if p.strip()]
    out: list[str] = []
    for it in parts:
        s = _normalize_index_symbol(it)
        if not s:
            continue
        if s not in out:
            out.append(s)
    return out


def build_equal_weight_index(df_by_symbol: dict[str, Any]):
    """
    把多个指数日线 close 合成一条“等权指数”（用收益率等权，不是 close 等权）。

    - 输出只保证 date/close（open/high/low 缺失会被上游补成 close）
    - 这是研究工具：我们只需要一个“稳定的基准尺子”，别在这块过度设计。
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    if not df_by_symbol:
        return None

    # 1) 统一成 pivot：index=date, columns=symbol, values=close
    ser_list = []
    for sym, df in df_by_symbol.items():
        if df is None or getattr(df, "empty", True):
            continue
        d2 = df.copy()
        if "date" not in d2.columns or "close" not in d2.columns:
            continue
        d2["date"] = pd.to_datetime(d2["date"], errors="coerce")
        d2 = d2.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if d2.empty:
            continue
        close = pd.to_numeric(d2["close"], errors="coerce").astype(float)
        s = pd.Series(close.to_numpy(), index=d2["date"].to_numpy(), name=str(sym))
        ser_list.append(s)

    if not ser_list:
        return None

    px = pd.concat(ser_list, axis=1).sort_index()
    # 前向填充：缺一天就当“该指数当天不变”（比直接 inner join 更不容易因为源站缺一根K线就掉链子）
    px = px.ffill().dropna(how="all")
    if px.empty:
        return None

    # 2) 等权收益：每个指数日收益 -> 行均值（忽略 NaN）
    rets = (px / px.shift(1).replace({0.0: float("nan")})) - 1.0
    ew = rets.mean(axis=1, skipna=True)

    # 3) 还原成一个“价格序列”（基准=1.0；初值不影响动量/回撤等比例指标）
    ew = ew.fillna(0.0)
    close_combo = (1.0 + ew).cumprod()

    out = pd.DataFrame({"date": close_combo.index, "close": close_combo.to_numpy()})
    out = out.dropna(subset=["date", "close"]).reset_index(drop=True)
    return out


def fetch_index_daily_spec(
    spec: str,
    *,
    cache_dir: Path,
    ttl_hours: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Any, str | None]:
    """
    拉取指数日线（支持单指数或 '+' 合成基准）：
    - "sh000300" => df, "sh000300"
    - "sh000300+sh000905" => 合成 df, "sh000300+sh000905"
    """
    parts = parse_index_combo_spec(spec)
    if not parts:
        return None, None

    # 依赖放函数里，避免 import 链把 CLI 启动拖慢
    from .akshare_source import FetchParams
    from .data_cache import fetch_daily_cached

    if len(parts) == 1:
        sym = str(parts[0])
        df = fetch_daily_cached(
            FetchParams(asset="index", symbol=sym, start_date=start_date, end_date=end_date),
            cache_dir=cache_dir,
            ttl_hours=float(ttl_hours),
        )
        return df, sym

    dfs: dict[str, Any] = {}
    for sym in parts:
        df = fetch_daily_cached(
            FetchParams(asset="index", symbol=str(sym), start_date=start_date, end_date=end_date),
            cache_dir=cache_dir,
            ttl_hours=float(ttl_hours),
        )
        if df is None or getattr(df, "empty", True):
            continue
        dfs[str(sym)] = df

    df_combo = build_equal_weight_index(dfs)
    if df_combo is None or getattr(df_combo, "empty", True):
        return None, "+".join(parts)

    return df_combo, "+".join(parts)

