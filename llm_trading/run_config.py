from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_CONFIG_SCHEMA_V1 = "llm_trading.run_config.v1"


def build_run_config(*, argv: list[str], note: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    统一配置文件（用于复跑/回放）：只存 argv（不存密钥）。
    """
    from .analysis_cache import compute_params_hash

    cmd = str(argv[0]) if argv else ""
    cfg: dict[str, Any] = {
        "schema": RUN_CONFIG_SCHEMA_V1,
        "generated_at": datetime.now().isoformat(),
        "cmd": cmd,
        "argv": list(argv),
        # 只基于 argv 计算的稳定 hash（同一套参数 => 同一 hash；用于报告/缓存键）。
        "params_hash": compute_params_hash({"cmd": cmd, "argv": list(argv)}),
    }
    if note:
        cfg["note"] = str(note)
    if extra:
        cfg["extra"] = extra
    return cfg


def write_run_config(path: Path, *, argv: list[str], note: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from .pipeline import write_json

    cfg = build_run_config(argv=argv, note=note, extra=extra)
    write_json(path, cfg)
    return cfg


def _load_json_text(text: str) -> Any:
    return json.loads(text)


def _load_yaml_text(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ValueError("要读 YAML 配置先装 PyYAML：pip install pyyaml") from exc
    return yaml.safe_load(text)


def load_any_config(path: Path) -> Any:
    """
    读 JSON/YAML。
    """
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in {".yaml", ".yml"}:
        return _load_yaml_text(text)
    return _load_json_text(text)


def extract_argv_from_any(payload: Any) -> tuple[list[str], str | None]:
    """
    支持从：
    - run_config.json（schema=run_config）
    - run_meta.json（tool=llm_trading）
    - report.json（run_meta / run_config 嵌套）
    抠出 argv + cmd。
    """
    if not isinstance(payload, dict):
        return [], None

    # report.json
    if isinstance(payload.get("run_config"), dict):
        argv = payload["run_config"].get("argv")
        if isinstance(argv, list) and argv:
            cmd = str(payload.get("cmd") or argv[0] or "")
            return [str(x) for x in argv], cmd or None

    if isinstance(payload.get("run_meta"), dict):
        argv = payload["run_meta"].get("argv")
        if isinstance(argv, list) and argv:
            cmd = str(payload.get("cmd") or payload["run_meta"].get("extra", {}).get("cmd") or argv[0] or "")
            return [str(x) for x in argv], cmd or None

    # run_config.json
    argv = payload.get("argv")
    if isinstance(argv, list) and argv:
        cmd = str(payload.get("cmd") or argv[0] or "")
        return [str(x) for x in argv], cmd or None

    return [], None
