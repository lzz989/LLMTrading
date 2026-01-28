from __future__ import annotations

import argparse
from pathlib import Path

from ..akshare_source import FetchParams
from ..data_cache import fetch_daily_cached
from ..price_utils import calc_pct_chg, extract_close_pair


def _norm_basis(basis: str | None) -> str:
    s = str(basis or "").strip().lower()
    if s in {"", "raw", "none", "0"}:
        return ""
    if s in {"qfq", "hfq"}:
        return s
    return ""


def _describe_df(df) -> dict:
    prev_close, close_last, as_of = extract_close_pair(df)
    pct = calc_pct_chg(prev_close, close_last)
    attrs = getattr(df, "attrs", {}) or {}
    data_source = None
    data_source_warning = None
    intraday_unclosed = None
    if isinstance(attrs, dict):
        data_source = attrs.get("data_source")
        data_source_warning = attrs.get("data_source_warning") or attrs.get("data_source_auto_fallback_error")
        intraday_unclosed = attrs.get("intraday_unclosed")
    return {
        "as_of": as_of,
        "close": close_last,
        "prev_close": prev_close,
        "pct_chg": pct,
        "data_source": data_source,
        "data_source_warning": data_source_warning,
        "intraday_unclosed": intraday_unclosed,
    }


def cmd_verify_prices(args: argparse.Namespace) -> int:
    asset = str(getattr(args, "asset", "") or "").strip().lower()
    symbol = str(getattr(args, "symbol", "") or "").strip()
    source = str(getattr(args, "source", "auto") or "auto").strip().lower()
    basis = _norm_basis(getattr(args, "basis", "raw"))
    compare = str(getattr(args, "compare", "") or "").strip()
    compare = _norm_basis(compare) if compare else ""
    threshold = float(getattr(args, "threshold", 0.015) or 0.015)

    cache_dir = Path(str(getattr(args, "cache_dir", "") or "").strip() or (Path("data") / "cache" / asset))
    ttl_hours = float(getattr(args, "cache_ttl_hours", 24.0) or 24.0)

    df_main = fetch_daily_cached(
        FetchParams(asset=asset, symbol=symbol, adjust=basis, source=source),
        cache_dir=cache_dir,
        ttl_hours=ttl_hours,
    )
    info_main = _describe_df(df_main)

    basis_label = "raw" if basis == "" else basis
    print(f"asset={asset} symbol={symbol} basis={basis_label}")
    print(
        f"as_of={info_main['as_of']} close={info_main['close']} prev={info_main['prev_close']} pct={info_main['pct_chg']}"
    )
    if info_main.get("data_source"):
        print(f"data_source={info_main['data_source']} warning={info_main.get('data_source_warning')}")
    if info_main.get("intraday_unclosed"):
        print("warning=intraday_unclosed")

    if compare:
        df_cmp = fetch_daily_cached(
            FetchParams(asset=asset, symbol=symbol, adjust=compare, source=source),
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
        )
        info_cmp = _describe_df(df_cmp)
        cmp_label = "raw" if compare == "" else compare
        print(f"compare_basis={cmp_label} as_of={info_cmp['as_of']} pct={info_cmp['pct_chg']}")
        if info_main.get("pct_chg") is not None and info_cmp.get("pct_chg") is not None:
            delta = float(info_main["pct_chg"]) - float(info_cmp["pct_chg"])
            print(f"delta={delta}")
            if abs(delta) > threshold:
                print(f"FAIL: |delta|>{threshold}")
                return 2
    return 0
