import unittest


class TestChip(unittest.TestCase):
    def _df(self, rows):
        import pandas as pd

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_vbp_overhead_zero_when_at_top(self):
        from llm_trading.chip import ChipVbpParams, compute_volume_by_price

        # 单调上行，最后一天是区间最高：上方应该几乎没有“筹码”
        df = self._df(
            [
                {"date": "2026-01-01", "open": 9, "high": 10, "low": 9, "close": 10, "volume": 100},
                {"date": "2026-01-02", "open": 10, "high": 11, "low": 10, "close": 11, "volume": 120},
                {"date": "2026-01-03", "open": 11, "high": 12, "low": 11, "close": 12, "volume": 130},
                {"date": "2026-01-04", "open": 12, "high": 13, "low": 12, "close": 13, "volume": 140},
            ]
        )
        out = compute_volume_by_price(df, params=ChipVbpParams(window_days=20, bins=12, method="typical"))
        self.assertIsNotNone(out)
        self.assertTrue(out.get("ok"))
        self.assertLessEqual(float(out.get("overhead_supply_pct") or 0.0), 0.20)
        # 新增：集中度/获利盘代理字段必须存在且范围合理
        self.assertIn("profit_proxy_pct", out)
        self.assertIn("loss_proxy_pct", out)
        self.assertIn("concentration_top1", out)
        self.assertIn("concentration_top3", out)
        self.assertGreaterEqual(float(out.get("concentration_top1") or 0.0), 0.0)
        self.assertLessEqual(float(out.get("concentration_top1") or 0.0), 1.0)
        self.assertGreaterEqual(float(out.get("concentration_top3") or 0.0), 0.0)
        self.assertLessEqual(float(out.get("concentration_top3") or 0.0), 1.0)

    def test_vbp_support_nonzero_when_at_bottom(self):
        from llm_trading.chip import ChipVbpParams, compute_volume_by_price

        # 单调下行，最后一天是最低：下方支撑几乎没有，上方套牢盘比例高
        df = self._df(
            [
                {"date": "2026-01-01", "open": 13, "high": 13, "low": 12, "close": 12, "volume": 100},
                {"date": "2026-01-02", "open": 12, "high": 12, "low": 11, "close": 11, "volume": 120},
                {"date": "2026-01-03", "open": 11, "high": 11, "low": 10, "close": 10, "volume": 130},
                {"date": "2026-01-04", "open": 10, "high": 10, "low": 9, "close": 9, "volume": 140},
            ]
        )
        out = compute_volume_by_price(df, params=ChipVbpParams(window_days=20, bins=12, method="typical"))
        self.assertIsNotNone(out)
        self.assertTrue(out.get("ok"))
        self.assertGreaterEqual(float(out.get("overhead_supply_pct") or 0.0), 0.40)
        self.assertGreaterEqual(float(out.get("loss_proxy_pct") or 0.0), 0.40)

    def test_cost_distribution_profit_ratio_increases(self):
        from llm_trading.chip import ChipCostParams, compute_turnover_cost_distribution

        # 换手率固定，价格逐步抬升：最终 profit_ratio 应该较高
        df = self._df(
            [
                {"date": "2026-01-01", "high": 10, "low": 9, "close": 10, "turnover_rate": 10.0},
                {"date": "2026-01-02", "high": 11, "low": 10, "close": 11, "turnover_rate": 10.0},
                {"date": "2026-01-03", "high": 12, "low": 11, "close": 12, "turnover_rate": 10.0},
                {"date": "2026-01-04", "high": 13, "low": 12, "close": 13, "turnover_rate": 10.0},
                {"date": "2026-01-05", "high": 14, "low": 13, "close": 14, "turnover_rate": 10.0},
            ]
        )
        out = compute_turnover_cost_distribution(df, params=ChipCostParams(window_days=120, bins=12, method="typical"))
        self.assertIsNotNone(out)
        self.assertTrue(out.get("ok"))
        pr = float(out.get("profit_ratio") or 0.0)
        self.assertGreaterEqual(pr, 0.50)


if __name__ == "__main__":
    unittest.main()
