# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_trading.signals_merge import merge_signals_files, parse_priority, parse_strategy_weights


def _write_signals_json(
    path: Path,
    *,
    strategy: str,
    items: list[dict],
    as_of: str = "2026-01-01",
    generated_at: str = "2026-01-01T00:00:00",
    config: dict | None = None,
) -> None:
    obj = {
        "schema_version": 1,
        "generated_at": generated_at,
        "as_of": as_of,
        "strategy": strategy,
        "source": {"type": "unit-test"},
        "config": (config or {}),
        "items": items,
    }
    path.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")


def _extract_merge_meta(item: dict) -> dict:
    meta = item.get("meta") or {}
    merged = meta.get("merged") or {}
    # Old/new compatibility: if primary_meta already had a dict merged, signals_merge nests under it.
    if isinstance(merged, dict) and "signals_merge" in merged and isinstance(merged.get("signals_merge"), dict):
        return merged["signals_merge"]
    if isinstance(merged, dict):
        return merged
    return {}


class TestSignalsMergeParsing(unittest.TestCase):
    def test_parse_strategy_weights_basic(self) -> None:
        self.assertEqual(
            parse_strategy_weights("bbb_weekly=1,trend_pullback_weekly=0.8"),
            {"bbb_weekly": 1.0, "trend_pullback_weekly": 0.8},
        )

    def test_parse_strategy_weights_invalid_ignored(self) -> None:
        got = parse_strategy_weights("a=0,b=-1,c=nan,d=inf,e=foo,=1,noeq,f=2")
        self.assertEqual(got, {"f": 2.0})

    def test_parse_priority_dedup(self) -> None:
        self.assertEqual(parse_priority("a,b,a, ,c"), ["a", "b", "c"])


class TestSignalsMergeCore(unittest.TestCase):
    def test_merge_risk_first_exit_wins(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            p2 = Path(td) / "s2.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[
                    {"asset": "stock", "symbol": "000001", "action": "entry", "score": 0.8, "confidence": 0.7, "entry": {"px": 10}, "meta": {"m": 1}, "tags": ["t1"]},
                ],
                config={"min_fee_yuan": 5},
            )
            _write_signals_json(
                p2,
                strategy="s2",
                items=[
                    {"asset": "stock", "symbol": "000001", "action": "exit", "score": 0.2, "confidence": 0.9, "entry": {"px": 9}, "meta": {"m": 2}, "tags": ["t2"]},
                ],
                config={"min_fee_yuan": 5},
            )
            out = merge_signals_files([p1, p2], conflict="risk_first")

            self.assertEqual(out["schema_version"], 1)
            self.assertEqual(out["counts"]["inputs"], 2)
            self.assertEqual(out["counts"]["items"], 1)

            it = out["items"][0]
            self.assertEqual(it["action"], "exit")
            self.assertEqual(it["asset"], "stock")
            self.assertEqual(it["symbol"], "000001")
            self.assertEqual(set(it.get("tags") or []), {"t1", "t2"})

            merged = _extract_merge_meta(it)
            self.assertEqual(merged.get("conflict_mode"), "risk_first")
            self.assertEqual(merged.get("primary_strategy"), "s2")  # exit contributor should become primary

            # Primary strategy entry/meta is forwarded
            self.assertEqual(it.get("entry"), {"px": 9})
            self.assertEqual(it.get("meta", {}).get("m"), 2)

            # Config is preserved when identical
            self.assertEqual(out.get("config", {}).get("min_fee_yuan"), 5)

    def test_merge_priority_can_override_risk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            p2 = Path(td) / "s2.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[{"asset": "stock", "symbol": "000001", "action": "entry", "score": 0.8, "confidence": 0.7, "entry": {"px": 10}}],
            )
            _write_signals_json(
                p2,
                strategy="s2",
                items=[{"asset": "stock", "symbol": "000001", "action": "exit", "score": 0.2, "confidence": 0.9, "entry": {"px": 9}}],
            )
            out = merge_signals_files([p1, p2], conflict="priority", priority=["s1"])
            it = out["items"][0]
            self.assertEqual(it["action"], "entry")

            merged = _extract_merge_meta(it)
            self.assertEqual(merged.get("primary_strategy"), "s1")
            self.assertEqual(it.get("entry"), {"px": 10})

    def test_merge_vote_weighted_majority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            p2 = Path(td) / "s2.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[{"asset": "stock", "symbol": "000001", "action": "entry", "score": 0.8, "confidence": 0.7, "entry": {"px": 10}}],
            )
            _write_signals_json(
                p2,
                strategy="s2",
                items=[{"asset": "stock", "symbol": "000001", "action": "exit", "score": 0.2, "confidence": 0.9, "entry": {"px": 9}}],
            )
            out = merge_signals_files([p1, p2], conflict="vote", weights={"s1": 2.0, "s2": 1.0})
            it = out["items"][0]
            self.assertEqual(it["action"], "entry")
            merged = _extract_merge_meta(it)
            self.assertEqual(merged.get("rule"), "vote.weighted_majority")

    def test_merge_vote_tie_falls_back_to_risk_first(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            p2 = Path(td) / "s2.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[{"asset": "stock", "symbol": "000001", "action": "entry", "score": 0.8, "confidence": 0.7}],
            )
            _write_signals_json(
                p2,
                strategy="s2",
                items=[{"asset": "stock", "symbol": "000001", "action": "exit", "score": 0.2, "confidence": 0.9}],
            )
            out = merge_signals_files([p1, p2], conflict="vote", weights={"s1": 1.0, "s2": 1.0})
            it = out["items"][0]
            self.assertEqual(it["action"], "exit")
            merged = _extract_merge_meta(it)
            self.assertEqual(merged.get("rule"), "vote.tie_exit")

    def test_merge_config_conflicts_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            p2 = Path(td) / "s2.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[{"asset": "stock", "symbol": "000001", "action": "watch"}],
                config={"min_fee_yuan": 5, "slippage_bps": 2.0},
            )
            _write_signals_json(
                p2,
                strategy="s2",
                items=[{"asset": "stock", "symbol": "000001", "action": "watch"}],
                config={"min_fee_yuan": 2, "slippage_bps": 2.0},
            )
            out = merge_signals_files([p1, p2], conflict="risk_first")

            cfg = out.get("config") or {}
            self.assertIsNone(cfg.get("min_fee_yuan"))
            self.assertEqual(cfg.get("slippage_bps"), 2.0)
            conflicts = (cfg.get("conflicts") or {}).get("min_fee_yuan") or []
            self.assertEqual(set(conflicts), {5, 2})

    def test_merge_sorting_and_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.json"
            _write_signals_json(
                p1,
                strategy="s1",
                items=[
                    {"asset": "stock", "symbol": "000001", "action": "watch", "score": 0.9, "confidence": 1.0},
                    {"asset": "stock", "symbol": "000002", "action": "entry", "score": 0.1, "confidence": 0.1},
                ],
            )
            out = merge_signals_files([p1], conflict="risk_first")
            items = out["items"]
            self.assertEqual([it["symbol"] for it in items], ["000002", "000001"])  # entry first

            out2 = merge_signals_files([p1], conflict="risk_first", top_k=1)
            self.assertEqual(out2["counts"]["items"], 1)
            self.assertEqual(out2["items"][0]["symbol"], "000002")


if __name__ == "__main__":
    unittest.main()
