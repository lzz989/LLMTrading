from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

from llm_trading.memory_store import (
    append_daily_memory,
    archive_daily_memory,
    ensure_memory_layout,
    keyword_search_memory,
    resolve_memory_paths,
)


class TestMemoryArchive(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_dir = Path(self._tmp.name)

        self._old_env = dict(os.environ)
        self.addCleanup(self._restore_env)
        os.environ["LLM_TRADING_MEMORY_DIR"] = str(tmp_dir / "memory")
        os.environ["LLM_TRADING_PROFILE_PATH"] = str(tmp_dir / "user_profile.json")

        self.paths = resolve_memory_paths(project_root=Path.cwd())
        ensure_memory_layout(self.paths)

        # Seed daily logs across different days.
        append_daily_memory(self.paths, d=date(2026, 1, 1), title="测试", text="旧1：右侧主升浪，不猜底。", source={"type": "test"})
        append_daily_memory(self.paths, d=date(2026, 1, 10), title="测试", text="旧2：止损先于一切。", source={"type": "test"})
        append_daily_memory(self.paths, d=date(2026, 1, 15), title="测试", text="新：只保留最近N天。", source={"type": "test"})

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_archive_dry_run_and_apply_is_idempotent(self) -> None:
        # today=2026-01-20, keep_days=7 => keep >= 2026-01-14 (i.e. 2026-01-15 kept)
        plan = archive_daily_memory(self.paths, keep_days=7, group="month", today=date(2026, 1, 20), apply=False)
        self.assertTrue(plan.get("ok"))
        self.assertEqual(int(plan.get("archived_count") or 0), 2)

        # apply
        out = archive_daily_memory(self.paths, keep_days=7, group="month", today=date(2026, 1, 20), apply=True)
        self.assertTrue(out.get("ok"))

        # old daily removed; recent kept
        self.assertFalse((self.paths.daily_dir / "2026-01-01.md").exists())
        self.assertFalse((self.paths.daily_dir / "2026-01-10.md").exists())
        self.assertTrue((self.paths.daily_dir / "2026-01-15.md").exists())

        # rollup file exists and contains archived content
        rollup = self.paths.base_dir / "archive" / "rollup" / "month" / "2026-01.md"
        self.assertTrue(rollup.exists())
        txt = rollup.read_text(encoding="utf-8")
        self.assertIn("旧1：右侧主升浪", txt)
        self.assertIn("旧2：止损先于一切", txt)
        self.assertIn("archived_from", txt)

        # keyword search can still find archived info
        hits = keyword_search_memory(self.paths, query="不猜底", max_results=10, context_lines=1)
        self.assertTrue(hits)
        self.assertTrue(any("archive" in str(h.get("path") or "") for h in hits))

        # idempotent: second run should do nothing
        plan2 = archive_daily_memory(self.paths, keep_days=7, group="month", today=date(2026, 1, 20), apply=False)
        self.assertEqual(int(plan2.get("archived_count") or 0), 0)


if __name__ == "__main__":
    unittest.main()

