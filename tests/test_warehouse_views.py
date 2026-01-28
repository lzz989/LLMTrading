# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestWarehouseViews(unittest.TestCase):
    def test_sql_init_and_views_queryable(self) -> None:
        """
        覆盖 Phase3/4/5 里最容易“悄悄炸”的点：
        - outputs/ 为空时 DuckDB glob 不得报错（sentinel 生效）
        - 关键 wh.v_* 视图必须能 query（哪怕返回 0 行）
        """
        try:
            import duckdb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("缺 duckdb 依赖：先装 requirements.txt") from exc

        from llm_trading.warehouse import sql_init

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "root"
            root.mkdir(parents=True, exist_ok=True)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "outputs").mkdir(parents=True, exist_ok=True)

            db_path = root / "data" / "warehouse.duckdb"
            p = sql_init(db_path=str(db_path), root_dir=str(root))
            self.assertTrue(Path(p).exists())

            con = duckdb.connect(str(p))
            try:
                con.execute("select count(*) from wh.file_catalog").fetchone()

                views = [
                    "wh.v_bars",
                    "wh.v_analysis_meta",
                    "wh.v_signal_backtest",
                    "wh.v_tushare_factors_flat",
                    "wh.v_etf_holdings_top10_items",
                    "wh.v_signals_items",
                    "wh.v_top_bbb_items",
                    "wh.v_holdings_user_holdings",
                    "wh.v_orders_next_open_orders",
                    "wh.v_rebalance_user_orders_next_open",
                    "wh.v_game_theory_factors",
                    "wh.v_opportunity_score",
                    "wh.v_cash_signal",
                    "wh.v_position_sizing",
                    # Phase3/4
                    "wh.v_strategy_signal",
                    "wh.v_strategy_alignment",
                    "wh.v_strategy_alignment_mismatches",
                    "wh.v_dynamic_weights_summary",
                    "wh.v_dynamic_weights_factors",
                    "wh.v_dynamic_weights_ic",
                ]

                for v in views:
                    con.execute(f"select count(*) from {v}").fetchone()
            finally:
                con.close()


if __name__ == "__main__":
    unittest.main()

