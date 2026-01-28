# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path


class TestStrategyConfigLoader(unittest.TestCase):
    def test_load_strategy_configs_yaml_ok(self) -> None:
        from llm_trading.strategy_config_loader import load_strategy_configs_yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strategy_configs.yaml"
            p.write_text(
                "\n".join(
                    [
                        "strategies:",
                        "  demo:",
                        "    factor_weights:",
                        "      ma_cross: 2.0",
                        "      macd: 1.0",
                        "    entry_threshold: 0.7",
                        "    exit_threshold: 0.4",
                        "    allowed_regimes: [bull, neutral]",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            cfgs = load_strategy_configs_yaml(p)
            self.assertIn("demo", cfgs)
            cfg = cfgs["demo"]
            # normalize_weights：2:1 -> 0.666.. / 0.333..
            self.assertAlmostEqual(float(cfg.factor_weights["ma_cross"]), 2.0 / 3.0, places=6)
            self.assertAlmostEqual(float(cfg.factor_weights["macd"]), 1.0 / 3.0, places=6)
            self.assertAlmostEqual(float(cfg.entry_threshold), 0.7, places=6)

    def test_load_strategy_configs_yaml_unknown_factor_raises(self) -> None:
        from llm_trading.strategy_config_loader import load_strategy_configs_yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "strategy_configs.yaml"
            p.write_text(
                "\n".join(
                    [
                        "strategies:",
                        "  bad:",
                        "    factor_weights:",
                        "      not_a_factor: 1.0",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_strategy_configs_yaml(p)

    def test_load_regime_weights_yaml_ok(self) -> None:
        from llm_trading.strategy_config_loader import load_regime_weights_yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "regime_weights.yaml"
            p.write_text(
                "\n".join(
                    [
                        "regime_weights:",
                        "  bull:",
                        "    ma_cross: 0.3",
                        "    macd: 0.7",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            m = load_regime_weights_yaml(p)
            self.assertIn("bull", m)
            self.assertAlmostEqual(float(m["bull"]["ma_cross"]), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()

