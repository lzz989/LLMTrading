# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


def _make_margin_df(d0: date, n: int, *, base: float, drift: float, shock_last: float = 0.0) -> pd.DataFrame:
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(n)]
    vals = [base + drift * i for i in range(n)]
    if n > 0 and shock_last:
        vals[-1] = vals[-1] + float(shock_last)
    return pd.DataFrame({"日期": dates, "融资融券余额": vals})


class TestMarketMarginRisk(unittest.TestCase):
    def test_margin_overheat_true(self) -> None:
        from llm_trading.margin_risk import compute_market_margin_risk

        d0 = date(2025, 9, 1)
        # 有波动的历史 + 末日暴增 => score01 高 + overheat
        df_sh = _make_margin_df(d0, 120, base=1.0e12, drift=2.0e9, shock_last=3.0e11)
        df_sz = _make_margin_df(d0, 120, base=1.0e12, drift=1.0e9, shock_last=2.0e11)

        with TemporaryDirectory() as td:
            with patch("akshare.macro_china_market_margin_sh", return_value=df_sh), patch(
                "akshare.macro_china_market_margin_sz", return_value=df_sz
            ):
                out = compute_market_margin_risk(as_of=date(2026, 1, 23), cache_dir=Path(td), ttl_hours=0.0)  # ttl=0 => force fetch

        self.assertTrue(bool(out.get("ok")))
        self.assertIsNotNone(out.get("score01"))
        self.assertTrue(bool(out.get("overheat")))

    def test_margin_deleveraging_true(self) -> None:
        from llm_trading.margin_risk import compute_market_margin_risk

        d0 = date(2025, 9, 1)
        # 末段明显下滑 => deleveraging
        # 5d/20d 跌幅要够大（触发阈值：5d<=-2% 或 20d<=-4%）
        df_sh = _make_margin_df(d0, 120, base=1.2e12, drift=-3.0e9, shock_last=0.0)
        df_sz = _make_margin_df(d0, 120, base=1.1e12, drift=-2.5e9, shock_last=0.0)

        with TemporaryDirectory() as td:
            with patch("akshare.macro_china_market_margin_sh", return_value=df_sh), patch(
                "akshare.macro_china_market_margin_sz", return_value=df_sz
            ):
                out = compute_market_margin_risk(as_of=date(2026, 1, 23), cache_dir=Path(td), ttl_hours=0.0)

        self.assertTrue(bool(out.get("ok")))
        self.assertTrue(bool(out.get("deleveraging")))


if __name__ == "__main__":
    unittest.main()
