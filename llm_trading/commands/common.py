from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _default_out_dir(csv_path: str) -> Path:
    stem = Path(csv_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{stem}_{ts}"


def _default_out_dir_for_symbol(asset: str, symbol: str, freq: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace(" ", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"{asset}_{safe_symbol}_{freq}_{ts}"


def _write_run_meta(out_dir: Path, args: argparse.Namespace, *, extra: dict | None = None) -> dict:
    """
    给每次运行写一个 run_meta.json，方便复现/排查。
    """
    from ..pipeline import write_json
    from ..run_meta import collect_run_meta

    argv = list(getattr(args, "_argv", []) or [])
    meta = collect_run_meta(argv=argv, extra=extra or None)
    try:
        write_json(out_dir / "run_meta.json", meta)
    except (AttributeError):  # noqa: BLE001
        pass
    return meta


def _write_run_config(out_dir: Path, args: argparse.Namespace, *, note: str | None = None, extra: dict | None = None) -> dict:
    """
    给每次运行写一个 run_config.json（统一配置文件：只存 argv，方便一键复跑）。
    """
    from ..run_config import write_run_config

    argv = list(getattr(args, "_argv", []) or [])
    try:
        return write_run_config(out_dir / "run_config.json", argv=argv, note=note, extra=extra or None)
    except (AttributeError):  # noqa: BLE001
        return {"argv": argv}


def _compute_market_regime_payload(
    regime_index: str,
    *,
    canary_downgrade: bool = True,
) -> tuple[dict | None, str | None, str | None]:
    """
    计算大盘牛熊/风险偏好（给 scan-* 用）。
    返回：
    - regime_dict: 可 JSON 化的 dict（或 None）
    - regime_error: 错误字符串（或 None）
    - regime_index_eff: 实际使用的指数代码；若关闭则为 None
    """
    idx_raw = str(regime_index or "").strip()
    try:
        from ..market_regime import compute_market_regime_payload as _cmp

        return _cmp(
            idx_raw,
            cache_dir=Path("data") / "cache" / "index",
            ttl_hours=6.0,  # 半天级别：小白工具默认别太“隔夜”感
            ensemble_mode="risk_first",  # 用户确认：风险优先（更保命）
            canary_downgrade=bool(canary_downgrade),
        )
    except (AttributeError) as exc:  # noqa: BLE001
        if idx_raw.lower() in {"", "off", "none", "0"}:
            return None, None, None
        return None, str(exc), idx_raw

