import unittest
from datetime import date


class TestEtfHoldings(unittest.TestCase):
    def test_parse_quarter_label(self) -> None:
        from llm_trading.etf_holdings import _parse_quarter_label

        self.assertEqual(_parse_quarter_label("2025年4季度股票投资明细"), (2025, 4))
        self.assertEqual(_parse_quarter_label("2025年1季度股票投资明细"), (2025, 1))
        self.assertIsNone(_parse_quarter_label(""))
        self.assertIsNone(_parse_quarter_label("foo"))

    def test_quarter_end_date(self) -> None:
        from llm_trading.etf_holdings import _quarter_end_date

        self.assertEqual(_quarter_end_date(2025, 1), date(2025, 3, 31))
        self.assertEqual(_quarter_end_date(2025, 2), date(2025, 6, 30))
        self.assertEqual(_quarter_end_date(2025, 3), date(2025, 9, 30))
        self.assertEqual(_quarter_end_date(2025, 4), date(2025, 12, 31))

    def test_guess_cn_stock_symbol(self) -> None:
        from llm_trading.etf_holdings import _guess_cn_stock_symbol

        self.assertEqual(_guess_cn_stock_symbol("300058"), "sz300058")
        self.assertEqual(_guess_cn_stock_symbol("002027"), "sz002027")
        self.assertEqual(_guess_cn_stock_symbol("600000"), "sh600000")
        self.assertEqual(_guess_cn_stock_symbol("430047"), "bj430047")
        self.assertIsNone(_guess_cn_stock_symbol("ABC"))
        self.assertIsNone(_guess_cn_stock_symbol("12345"))

    def test_prefixed_stock_symbol_to_ts_code(self) -> None:
        from llm_trading.etf_holdings import _prefixed_stock_symbol_to_ts_code

        self.assertEqual(_prefixed_stock_symbol_to_ts_code("sz300058"), "300058.SZ")
        self.assertEqual(_prefixed_stock_symbol_to_ts_code("sh600000"), "600000.SH")
        self.assertEqual(_prefixed_stock_symbol_to_ts_code("bj430047"), "430047.BJ")
        self.assertIsNone(_prefixed_stock_symbol_to_ts_code("300058"))
        self.assertIsNone(_prefixed_stock_symbol_to_ts_code("sh123"))


if __name__ == "__main__":
    unittest.main()

