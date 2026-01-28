# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestDiagnostics(unittest.TestCase):
    def test_warn_and_record_dedupe_and_limits(self) -> None:
        from llm_trading.diagnostics import Diagnostics

        d = Diagnostics(max_items=2)
        with patch("llm_trading.diagnostics._LOG.warning", return_value=None):
            d.warn("")  # ignored
            d.warn("a")
            d.warn("a")  # dedupe
            d.warn("b")
            d.warn("c")  # over limit

        self.assertEqual(d.warnings, ["a", "b"])

        with patch("llm_trading.diagnostics._LOG.warning", return_value=None):
            d.record("s1", RuntimeError("x"))
            d.record("s1", RuntimeError("y"))  # dedupe by stage
            d.record("s2", RuntimeError("z"))
            d.record("s3", RuntimeError("k"))  # over limit

        self.assertEqual(len(d.errors), 2)
        self.assertEqual(d.errors[0].get("stage"), "s1")
        self.assertEqual(d.errors[1].get("stage"), "s2")

    def test_warn_logger_failure_is_suppressed(self) -> None:
        from llm_trading.diagnostics import Diagnostics

        d = Diagnostics()
        with patch("llm_trading.diagnostics._LOG.warning", side_effect=RuntimeError("logger boom")):
            d.warn("x")
        self.assertEqual(d.warnings, ["x"])

    def test_write_failure_is_suppressed(self) -> None:
        from llm_trading.diagnostics import Diagnostics

        d = Diagnostics()
        with patch("llm_trading.diagnostics._LOG.warning", return_value=None):
            d.warn("x")
            d.record("s1", RuntimeError("boom"))

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            # write_json fails -> should not raise
            with patch("llm_trading.diagnostics.write_json", side_effect=RuntimeError("write boom")):
                with patch("llm_trading.diagnostics._LOG.warning", side_effect=RuntimeError("logger boom")):
                    d.write(out_dir, cmd="t")

            # No file created
            self.assertFalse((out_dir / "diagnostics.json").exists())

        # Happy path sanity: it writes valid JSON
        with tempfile.TemporaryDirectory() as td2:
            out_dir2 = Path(td2)
            with patch("llm_trading.diagnostics._LOG.warning", return_value=None):
                d.write(out_dir2, cmd="t")
            p = out_dir2 / "diagnostics.json"
            self.assertTrue(p.exists())
            obj = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(obj.get("schema"), "llm_trading.diagnostics.v1")
            self.assertEqual(obj.get("cmd"), "t")


if __name__ == "__main__":
    unittest.main()
