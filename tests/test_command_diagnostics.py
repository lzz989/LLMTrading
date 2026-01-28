# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv_csv(path: Path, n: int = 260, *, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    steps = rng.normal(loc=0.001, scale=0.01, size=n)
    close = 10.0 + np.cumsum(steps)
    close = np.maximum(close, 0.1)

    open_ = close * (1.0 + rng.normal(loc=0.0, scale=0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(loc=0.0, scale=0.003, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(loc=0.0, scale=0.003, size=n)))
    volume = (1_000_000 + rng.integers(low=0, high=200_000, size=n)).astype(float)
    amount = close * volume

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8")


class TestCommandDiagnostics(unittest.TestCase):
    def test_analyze_writes_diagnostics_json(self) -> None:
        from llm_trading.cli import build_parser

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "ohlcv.csv"
            out_dir = td_path / "out_analyze"
            out_dir.mkdir(parents=True, exist_ok=True)

            _make_ohlcv_csv(csv_path, n=260)

            parser = build_parser()
            ns = parser.parse_args(["analyze", "--csv", str(csv_path), "--method", "institution", "--out-dir", str(out_dir)])
            rc = int(ns.func(ns))
            self.assertEqual(rc, 0)

            diag_path = out_dir / "diagnostics.json"
            self.assertTrue(diag_path.exists(), "analyze should create diagnostics.json")
            obj = json.loads(diag_path.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("schema"), "llm_trading.diagnostics.v1")
            self.assertEqual(obj.get("cmd"), "analyze")
            self.assertIn("warnings", obj)
            self.assertIn("errors", obj)

    def test_holdings_user_writes_diagnostics_json_on_suppressed_error(self) -> None:
        from llm_trading.cli import build_parser

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            holdings_path = td_path / "user_holdings.json"
            out_path = td_path / "holdings_user.json"

            holdings_path.write_text(
                json.dumps(
                    {
                        "positions": [
                            {"asset": "etf", "symbol": "sh510300", "shares": 100, "cost_basis": 1.0},
                        ],
                        "cash": {"amount": 1000.0},
                        "trade_rules": {"rebalance_schedule": "any_day"},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            def _fake_analyze_holdings(*_a, **_kw):
                return {"as_of": "2026-01-23", "market_regime": {"label": "neutral"}, "holdings": [], "warnings": ["dummy"]}

            with patch("llm_trading.holdings.analyze_holdings", side_effect=_fake_analyze_holdings):
                with patch("llm_trading.portfolio.build_portfolio_summary", side_effect=RuntimeError("boom")):
                    parser = build_parser()
                    ns = parser.parse_args(["holdings-user", "--path", str(holdings_path), "--out", str(out_path)])
                    rc = int(ns.func(ns))
                    self.assertEqual(rc, 0)

            diag_path = td_path / "diagnostics.json"
            self.assertTrue(diag_path.exists(), "holdings-user should create diagnostics.json next to --out")
            obj = json.loads(diag_path.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("schema"), "llm_trading.diagnostics.v1")
            self.assertEqual(obj.get("cmd"), "holdings-user")
            errs = obj.get("errors") or []
            self.assertTrue(any("build_portfolio_summary" in str(e.get("stage")) for e in errs))


if __name__ == "__main__":
    unittest.main()

