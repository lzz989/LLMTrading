from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .utils_time import parse_date_any

def _dt_to_ms(dt: datetime) -> int:
    # 统一按 UTC 处理（Binance kline 时间戳是 UTC）。
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _http_get_json(url: str, *, timeout_sec: float = 10.0) -> Any:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=float(timeout_sec)) as resp:  # noqa: S310
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def fetch_crypto_daily_binance(
    *,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 1000,
    timeout_sec: float = 10.0,
):
    """
    Binance 现货日线（1d）K线：
    - 默认取最近 1000 根（日线），足够覆盖 60 周以上的BBB最小样本要求。
    - 返回列：date/open/high/low/close/volume/amount
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol 为空")

    lim = int(limit or 0)
    lim = 1000 if lim <= 0 else max(1, min(lim, 1000))

    start_ms = None
    end_ms = None
    if start_date:
        sd = parse_date_any(start_date).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        start_ms = _dt_to_ms(sd)
    if end_date:
        # end_date 为“闭区间”：取到这一天的日K（把 endTime 设到这天的 23:59:59.999 UTC）
        ed = parse_date_any(end_date).replace(hour=23, minute=59, second=59, microsecond=999000, tzinfo=timezone.utc)
        end_ms = _dt_to_ms(ed)

    base = "https://api.binance.com/api/v3/klines"
    params: dict[str, Any] = {"symbol": sym, "interval": "1d", "limit": lim}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    # 仅 1 个标的，KISS：默认一把取完；如果用户强行给了很早的 start_date，才做分页补齐。
    out: list[list[Any]] = []
    loops = 0
    while True:
        loops += 1
        if loops > 10:  # 兜底：别无限薅源站
            break

        url = f"{base}?{urlencode(params)}"
        rows = _http_get_json(url, timeout_sec=timeout_sec)
        if not isinstance(rows, list) or not rows:
            break

        for r in rows:
            if isinstance(r, list) and len(r) >= 6:
                out.append(r)

        if len(rows) < lim:
            break

        # 下一页：从最后一根K线的开盘时间之后继续
        last_open_ms = int(rows[-1][0])
        params["startTime"] = int(last_open_ms + 1)

        # 如果用户没给 start_date：我们本来就只想要最近 lim 根，别分页
        if start_ms is None:
            break

    if not out:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    df = pd.DataFrame(
        out,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["amount"] = pd.to_numeric(df["quote_volume"], errors="coerce")

    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # 默认丢掉“当天未收盘”的日K（避免未来函数）
    if not end_date and (not df.empty):
        try:
            now_utc = datetime.now(timezone.utc)
            today_utc = now_utc.date()
            last_dt = df.iloc[-1]["date"]
            last_date = last_dt.date() if hasattr(last_dt, "date") else None
            if last_date == today_utc and len(df) >= 2:
                df = df.iloc[:-1].reset_index(drop=True)
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass

    return df


def fetch_crypto_daily_stooq(
    *,
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timeout_sec: float = 10.0,
):
    """
    Stooq 日线（CSV）：稳定、免Key，适合做“周线+日线”的中低频信号。

    示例：
    - BTC/USD: https://stooq.com/q/d/l/?s=btcusd&i=d
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("没装 pandas？先跑：pip install -r \"requirements.txt\"") from exc

    sym = str(symbol or "").strip().lower()
    if not sym:
        raise ValueError("symbol 为空")

    url = f"https://stooq.com/q/d/l/?{urlencode({'s': sym, 'i': 'd'})}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=float(timeout_sec)) as resp:  # noqa: S310
        raw = resp.read()
    txt = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(txt))
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    # Stooq: Date,Open,High,Low,Close,Volume
    rename = {}
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc == "date":
            rename[c] = "date"
        elif cc == "open":
            rename[c] = "open"
        elif cc == "high":
            rename[c] = "high"
        elif cc == "low":
            rename[c] = "low"
        elif cc == "close":
            rename[c] = "close"
        elif cc == "volume":
            rename[c] = "volume"
    df = df.rename(columns=rename)

    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    # amount：用 close*volume 粗算（Stooq volume 口径不一定一致，但 BBB 不依赖它）
    try:
        if "volume" in df.columns:
            df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
        else:
            df["amount"] = None
    except (AttributeError):  # noqa: BLE001
        df["amount"] = None

    # 默认丢掉“当天未收盘”的日K（避免未来函数）
    if not end_date and (not df.empty):
        try:
            today = datetime.now(timezone.utc).date()
            last_dt = df.iloc[-1]["date"]
            last_date = last_dt.date() if hasattr(last_dt, "date") else None
            if last_date == today and len(df) >= 2:
                df = df.iloc[:-1].reset_index(drop=True)
        except (KeyError, IndexError, AttributeError):  # noqa: BLE001
            pass

    if start_date:
        sd = parse_date_any(start_date)
        df = df[df["date"] >= sd].reset_index(drop=True)
    if end_date:
        ed = parse_date_any(end_date)
        df = df[df["date"] <= ed].reset_index(drop=True)

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    df = df[keep].reset_index(drop=True)
    return df
