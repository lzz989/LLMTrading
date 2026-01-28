from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_trading.config import AppConfig
from llm_trading.memory_store import (
    append_daily_memory,
    ensure_memory_layout,
    resolve_memory_paths,
    update_user_profile,
)
from llm_trading.memory_vector import build_or_update_vector_index, vector_search


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """
    Deterministic tiny embedding for tests.
    vec = [count("主升浪"), count("止损"), count("偏好")]
    """

    out: list[list[float]] = []
    for t in texts:
        s = t or ""
        out.append(
            [
                float(s.count("主升浪")),
                float(s.count("止损")),
                float(s.count("偏好")),
            ]
        )
    return out


class TestMemoryVectorIndex(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_dir = Path(self._tmp.name)

        # Isolate memory store under /tmp to avoid polluting repo.
        self._old_env = dict(os.environ)
        self.addCleanup(self._restore_env)
        os.environ["LLM_TRADING_MEMORY_DIR"] = str(tmp_dir / "memory")
        os.environ["LLM_TRADING_PROFILE_PATH"] = str(tmp_dir / "user_profile.json")

        # Enable embeddings config (but we'll mock the HTTP call).
        os.environ["EMBEDDINGS_API_KEY"] = "dummy"
        os.environ["EMBEDDINGS_MODEL"] = "dummy-embed"
        os.environ["EMBEDDINGS_BASE_URL"] = "https://example.invalid"

        self.paths = resolve_memory_paths(project_root=Path.cwd())
        ensure_memory_layout(self.paths)
        self.cfg = AppConfig(project_root=Path.cwd())

        # Seed some memory.
        self.paths.long_term_md.write_text("长期：主升浪纪律，止损先于一切。\n", encoding="utf-8")
        append_daily_memory(self.paths, text="今天偏好：右侧主升浪，不猜底。", title="测试", source={"type": "test"})
        update_user_profile(
            self.paths,
            updates={"workflow.preferred_interface": "chat", "risk.stop_loss_pct": 0.06},
            source={"type": "test"},
        )

    def _restore_env(self) -> None:
        # Restore env variables to avoid leaking into other tests.
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_build_index_and_search(self) -> None:
        with mock.patch("llm_trading.memory_vector.openai_embeddings", side_effect=lambda _cfg, inputs: _fake_embed(inputs)):
            idx = build_or_update_vector_index(self.cfg, paths=self.paths, force=True)
            self.assertTrue(self.paths.vector_index_json.exists())
            self.assertIsInstance(idx, dict)

            res = vector_search(self.cfg, paths=self.paths, query="主升浪", mode="vector", max_results=5)
            self.assertTrue(res)
            self.assertIn("path", res[0])
            self.assertGreaterEqual(float(res[0]["score"]), 0.0)

    def test_incremental_rebuild_avoids_reembed_when_unchanged(self) -> None:
        # First build.
        with mock.patch("llm_trading.memory_vector.openai_embeddings", side_effect=lambda _cfg, inputs: _fake_embed(inputs)):
            build_or_update_vector_index(self.cfg, paths=self.paths, force=True)

        # Second build should not call embeddings if content unchanged.
        with mock.patch("llm_trading.memory_vector.openai_embeddings", side_effect=AssertionError("should not re-embed")):
            build_or_update_vector_index(self.cfg, paths=self.paths, force=False)


if __name__ == "__main__":
    unittest.main()
