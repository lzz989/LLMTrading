from __future__ import annotations

import unittest

from llm_trading.commands.chat import _rule_plan, parse_symbols_from_text


class TestChatRouter(unittest.TestCase):
    def test_parse_symbols_basic(self) -> None:
        xs = parse_symbols_from_text("分析 000725")
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].asset, "stock")
        self.assertEqual(xs[0].symbol, "000725")

        xs = parse_symbols_from_text("看看 510300")
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].asset, "etf")
        self.assertEqual(xs[0].symbol, "510300")

        xs = parse_symbols_from_text("分析 sh000300")
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].asset, "index")
        self.assertEqual(xs[0].symbol, "sh000300")

        xs = parse_symbols_from_text("分析 sz399006")
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].asset, "index")
        self.assertEqual(xs[0].symbol, "sz399006")

    def test_rule_plan_routes_to_run(self) -> None:
        plan = _rule_plan("跑一遍今天的计划", prefs={})
        acts = plan.get("actions") or []
        self.assertTrue(acts)
        self.assertEqual(acts[0].get("type"), "run")
        self.assertIn("out_dir", (acts[0].get("args") or {}))

    def test_rule_plan_routes_to_analyze(self) -> None:
        plan = _rule_plan("分析 000725", prefs={})
        acts = plan.get("actions") or []
        self.assertTrue(acts)
        self.assertEqual(acts[0].get("type"), "analyze")
        self.assertEqual((acts[0].get("args") or {}).get("asset"), "stock")
        self.assertEqual((acts[0].get("args") or {}).get("symbol"), "000725")

    def test_rule_plan_run_and_analyze(self) -> None:
        plan = _rule_plan("跑一遍并分析 510300", prefs={})
        acts = plan.get("actions") or []
        self.assertGreaterEqual(len(acts), 2)
        self.assertEqual(acts[0].get("type"), "run")
        self.assertEqual(acts[1].get("type"), "analyze")
        self.assertEqual((acts[1].get("args") or {}).get("asset"), "etf")
        self.assertEqual((acts[1].get("args") or {}).get("symbol"), "510300")

    def test_rule_plan_triggers_research_skill(self) -> None:
        plan = _rule_plan("舆情分析一下今天的计划", prefs={})
        acts = plan.get("actions") or []
        self.assertTrue(acts)
        # 先 run，再 skill
        self.assertEqual(acts[0].get("type"), "run")
        self.assertTrue(any(a.get("type") == "skill" and (a.get("args") or {}).get("name") == "research" for a in acts))

    def test_rule_plan_force_strategy_skill(self) -> None:
        plan = _rule_plan("#strategy 给我一份执行清单", prefs={})
        acts = plan.get("actions") or []
        self.assertTrue(acts)
        self.assertTrue(any(a.get("type") == "skill" and (a.get("args") or {}).get("name") == "strategy" for a in acts))


if __name__ == "__main__":
    unittest.main()
