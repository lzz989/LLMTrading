from __future__ import annotations

import argparse
from pathlib import Path

from ..akshare_source import DataSourceError, FetchParams, fetch_daily
from ..resample import resample_to_weekly


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        df = fetch_daily(
            FetchParams(
                asset=args.asset,
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                adjust=args.adjust,
                source=getattr(args, "source", None),
            )
        )
    except DataSourceError as exc:
        raise SystemExit(str(exc)) from exc

    if args.freq == "weekly":
        df = resample_to_weekly(df)

    out_path = Path(args.out) if args.out else Path("data") / f"{args.asset}_{args.symbol}_{args.freq}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(str(out_path.resolve()))
    return 0

