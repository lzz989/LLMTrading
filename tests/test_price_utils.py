import unittest

import pandas as pd

from llm_trading.price_utils import calc_pct_chg, extract_close_pair, select_price_df


class TestPriceUtils(unittest.TestCase):
    def test_calc_pct_chg(self):
        self.assertAlmostEqual(calc_pct_chg(100, 105), 0.05)
        self.assertIsNone(calc_pct_chg(0, 105))
        self.assertIsNone(calc_pct_chg(None, 105))

    def test_extract_close_pair(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-27", "2026-01-28"],
                "close": [10.0, 11.0],
            }
        )
        prev_close, close_last, as_of = extract_close_pair(df)
        self.assertEqual(prev_close, 10.0)
        self.assertEqual(close_last, 11.0)
        self.assertTrue(str(as_of).startswith("2026-01-28"))

    def test_select_price_df(self):
        df_raw = pd.DataFrame()
        df_fallback = pd.DataFrame({"date": ["2026-01-28"], "close": [1.0]})
        df_sel, basis, warn = select_price_df(df_raw, df_fallback, asset="stock")
        self.assertEqual(basis, "qfq")
        self.assertEqual(warn, "raw_missing_fallback_qfq")
        self.assertIs(df_sel, df_fallback)


if __name__ == "__main__":
    unittest.main()
