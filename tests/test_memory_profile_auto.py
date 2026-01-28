from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_trading.memory_store import (
    ensure_memory_layout,
    resolve_memory_paths,
    rollback_user_profile,
    update_user_profile,
)


class TestMemoryProfileAuto(unittest.TestCase):
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

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_auto_mode_whitelist_and_rollback(self) -> None:
        # auto mode should only allow workflow/output/memory.*
        profile = update_user_profile(
            self.paths,
            updates={
                "workflow.preferred_interface": "chat",
                "risk.stop_loss_pct": 0.06,  # should be rejected in auto mode
            },
            source={"type": "test"},
            mode="auto",
        )
        prefs = profile.get("preferences") if isinstance(profile, dict) else None
        self.assertIsInstance(prefs, dict)
        self.assertEqual(prefs.get("workflow", {}).get("preferred_interface"), "chat")
        self.assertFalse("risk" in (prefs or {}))

        # ledger must contain rejected key
        lines = self.paths.ledger_jsonl.read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines)
        last = json.loads(lines[-1])
        self.assertEqual(last.get("type"), "profile_update")
        self.assertEqual(last.get("mode"), "auto")
        self.assertIn("risk.stop_loss_pct", (last.get("rejected") or {}))

        # rollback last auto update should remove workflow.preferred_interface (since it didn't exist before)
        rb = rollback_user_profile(self.paths, steps=1, mode="auto", apply=True)
        self.assertTrue(rb.get("ok"))

        profile2 = json.loads(self.paths.user_profile_json.read_text(encoding="utf-8"))
        prefs2 = profile2.get("preferences") or {}
        self.assertFalse("workflow" in prefs2)


if __name__ == "__main__":
    unittest.main()

