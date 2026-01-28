# -*- coding: utf-8 -*-
"""
Phase3：策略迁移到“因子配置”的配置加载器（YAML）。

目标：
- 把 config/strategy_configs.yaml 落成可用的 StrategyConfig（Factor 权重配置）
- 保持默认行为不变：只有在新命令/新开关启用时才生效
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("缺依赖：请先安装 requirements.txt（需要 pyyaml）") from exc


def _as_dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []


def _as_str(x: Any) -> str:
    return str(x) if x is not None else ""


def _as_float(x: Any) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    return v


def _as_bool(x: Any) -> bool:
    return bool(x)


@dataclass(frozen=True, slots=True)
class LoadedStrategyConfigs:
    path: Path
    strategies: dict[str, Any]


def load_strategy_configs_yaml(path: Path) -> dict[str, "StrategyConfig"]:
    """
    读取 config/strategy_configs.yaml -> {key: StrategyConfig}
    """
    from .factors.base import StrategyConfig

    # 确保因子已注册
    from . import factors as _  # noqa: F401
    from .factors.base import FACTOR_REGISTRY

    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _as_dict(obj)
    raw_strategies = _as_dict(root.get("strategies"))

    out: dict[str, StrategyConfig] = {}
    known_factors = set(FACTOR_REGISTRY.list_factors())

    for key, raw in raw_strategies.items():
        k = str(key or "").strip()
        if not k:
            continue
        d = _as_dict(raw)
        weights0 = _as_dict(d.get("factor_weights"))
        params0 = _as_dict(d.get("factor_params"))

        weights: dict[str, float] = {}
        for fac, w in weights0.items():
            fn = str(fac or "").strip()
            if not fn:
                continue
            if fn not in known_factors:
                # 配置写错了，直接跳过，避免 silent bug
                raise ValueError(f"未知因子：{fn}（策略={k}；可用因子={sorted(known_factors)}）")
            wf = _as_float(w)
            if wf is None:
                continue
            if wf < 0:
                raise ValueError(f"factor_weights 不能为负：{k}.{fn}={wf}")
            weights[fn] = float(wf)

        factor_params: dict[str, dict] = {}
        for fac, pp in params0.items():
            fn = str(fac or "").strip()
            if not fn:
                continue
            if fn not in known_factors:
                raise ValueError(f"未知因子参数目标：{fn}（策略={k}）")
            factor_params[fn] = _as_dict(pp)

        cfg = StrategyConfig(
            name=str(k),
            description=_as_str(d.get("description")),
            factor_weights=weights,
            factor_params=factor_params,
            entry_threshold=float(_as_float(d.get("entry_threshold")) or 0.6),
            exit_threshold=float(_as_float(d.get("exit_threshold")) or 0.4),
            require_factors=[str(x).strip() for x in _as_list(d.get("require_factors")) if str(x).strip()],
            exclude_factors=[str(x).strip() for x in _as_list(d.get("exclude_factors")) if str(x).strip()],
            allowed_regimes=[str(x).strip() for x in _as_list(d.get("allowed_regimes")) if str(x).strip()] or ["bull", "neutral"],
        )
        # validate 会要求权重和为1；这里容忍不精确，normalize 一下
        if not cfg.validate():
            cfg.normalize_weights()
        out[cfg.name] = cfg

    if not out:
        raise ValueError(f"策略配置为空：{p}")
    return out


def load_regime_weights_yaml(path: Path) -> dict[str, dict[str, float]]:
    """
    读取 config/regime_weights.yaml -> {regime: {factor: weight}}
    """
    # 确保因子已注册
    from . import factors as _  # noqa: F401
    from .factors.base import FACTOR_REGISTRY

    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    root = _as_dict(obj)
    raw = _as_dict(root.get("regime_weights"))

    known_factors = set(FACTOR_REGISTRY.list_factors())
    out: dict[str, dict[str, float]] = {}
    for regime, wmap in raw.items():
        rk = str(regime or "").strip().lower()
        if not rk:
            continue
        d = _as_dict(wmap)
        weights: dict[str, float] = {}
        for fac, w in d.items():
            fn = str(fac or "").strip()
            if not fn:
                continue
            if fn not in known_factors:
                raise ValueError(f"未知因子：{fn}（regime={rk}）")
            wf = _as_float(w)
            if wf is None:
                continue
            if wf < 0:
                raise ValueError(f"regime_weights 不能为负：{rk}.{fn}={wf}")
            weights[fn] = float(wf)
        out[rk] = weights

    if not out:
        raise ValueError(f"regime_weights 为空：{p}")
    return out

