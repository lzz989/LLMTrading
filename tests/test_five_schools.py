# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest


class TestFiveSchoolsQuickReview(unittest.TestCase):
    def test_render_school_quick_review_schema(self) -> None:
        from llm_trading.skills.five_schools import render_school_quick_review

        leaders_cfg = {}  # 用 fallback 名称/模板即可

        wyckoff = {
            "method": "wyckoff_features",
            "last": {"date": "2026-01-28", "close": 10.0, "ma50": 9.5, "ma200": 9.0, "ad_line": 123.0},
            "derived": {"close_vs_ma200": 1.0, "ma50_vs_ma200": 0.5, "ret_4": 0.02, "ret_12": 0.08, "ad_delta_20": 3.2},
        }
        chan = {
            "summary": {
                "last_date": "2026-01-28",
                "last_close": 10.0,
                "last_stroke_direction": "up",
                "position_vs_last_center": "above",
                "last_center": {"start_date": "2025-12-01", "end_date": "2026-01-10", "low": 8.8, "high": 9.6},
            }
        }
        ichimoku = {
            "method": "ichimoku",
            "last": {
                "date": "2026-01-28",
                "close": 10.0,
                "kijun": 9.4,
                "cloud_top": 9.3,
                "cloud_bottom": 8.9,
                "position": "above",
                "tk_cross": "bullish",
            },
        }
        turtle = {
            "method": "turtle",
            "last": {"date": "2026-01-28", "close": 10.0, "donchian_exit_lower": 9.1, "donchian_exit_upper": 10.2},
            "signals": {"long_entry_breakout": True, "long_exit_breakdown": False},
            "risk": {"long_stop": 9.3},
        }
        vsa = {"method": "vsa", "summary": {"vol_level": "high", "spread_level": "normal", "events": 5}, "last": {"date": "2026-01-28", "close": 10.0}}

        cases = [
            ("wyckoff", wyckoff),
            ("chan", chan),
            ("ichimoku", ichimoku),
            ("turtle", turtle),
            ("vsa", vsa),
            ("wyckoff", None),
        ]
        for school, evidence in cases:
            r = render_school_quick_review(symbol="sh600000", name="浦发银行", school=school, evidence=evidence, leaders_cfg=leaders_cfg)  # type: ignore[arg-type]
            self.assertIsInstance(r, dict)
            for k in ["school", "leader", "stance", "one_liner", "anchors", "invalidation"]:
                self.assertIn(k, r)
            self.assertIn(r.get("stance"), {"bull", "bear", "neutral"})
            inv = r.get("invalidation")
            self.assertIsInstance(inv, dict)
            self.assertIn("rule", inv)
            # note 必须可读（哪怕缺数据）
            self.assertTrue(str(inv.get("note") or "").strip())


if __name__ == "__main__":
    unittest.main()

