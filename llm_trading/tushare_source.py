from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class TushareSourceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TushareEnv:
    token: str
    # ⚠️ 非官方代理/转发有合规与安全风险（你自己确认后再用）
    http_url: str | None = None


def load_tushare_env() -> TushareEnv | None:
    """
    从环境变量读取 TuShare 配置：
    - TUSHARE_TOKEN: 必填
    - TUSHARE_HTTP_URL: 可选（⚠️ 非官方代理有合规/安全风险）
    """
    # CLI 里不是所有命令都会显式 load_config()，这里兜底加载 .env，免得用户配了却读不到。
    try:
        from .config import load_config

        load_config()
    except (AttributeError):  # noqa: BLE001
        pass

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        return None
    http_url = os.getenv("TUSHARE_HTTP_URL", "").strip() or None
    return TushareEnv(token=token, http_url=http_url)


def get_pro_api(env: TushareEnv | None = None) -> Any:
    """
    获取 TuShare Pro Api 客户端（tushare.pro_api 返回的对象）。

    兼容“教程里改私有字段”的写法：
    - pro._DataApi__token
    - pro._DataApi__http_url
    """
    env2 = env or load_tushare_env()
    if env2 is None or not env2.token:
        raise TushareSourceError("缺少环境变量 TUSHARE_TOKEN（在 .env 里配，别往聊天里发）")

    try:
        import tushare as ts
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise TushareSourceError("没装 tushare？先跑：pip install -r \"requirements.txt\"") from exc

    # 优先用 token 参数初始化，避免依赖全局状态。
    try:
        pro = ts.pro_api(env2.token)
    except (TypeError, ValueError, AttributeError, RuntimeError):  # noqa: BLE001
        try:
            ts.set_token(env2.token)
            pro = ts.pro_api()
        except (TypeError, ValueError, AttributeError) as exc:  # noqa: BLE001
            raise TushareSourceError(f"tushare pro_api 初始化失败：{exc}") from exc

    # 一些“代理教程”要求改私有字段；这里做成可选兼容，不然用户按教程配了也跑不通。
    try:
        setattr(pro, "_DataApi__token", env2.token)
    except (TypeError, ValueError, AttributeError, RuntimeError):  # noqa: BLE001
        pass
    if env2.http_url:
        u = str(env2.http_url).strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            raise TushareSourceError(f"TUSHARE_HTTP_URL 非法（必须 http(s)）：{u}")
        try:
            setattr(pro, "_DataApi__http_url", u)
        except (TypeError, ValueError, AttributeError, RuntimeError):  # noqa: BLE001
            pass

    return pro


def normalize_ts_code(sym: str) -> str | None:
    """
    把常见形式统一成 TuShare ts_code：
    - sh510150 / sz159915 -> 510150.SH / 159915.SZ
    - 510150 / 159915 -> 510150.SH (默认按 5xxxx=>SH；否则=>SZ)
    - 510150.SH / 159915.SZ -> 原样返回
    """
    s = str(sym or "").strip()
    if not s:
        return None

    u = s.upper()
    if "." in u and len(u) == 9:
        code, suf = u.split(".", 1)
        if len(code) == 6 and code.isdigit() and suf in {"SH", "SZ"}:
            return f"{code}.{suf}"

    sl = s.lower()
    if sl.startswith(("sh", "sz")) and len(sl) == 8 and sl[2:].isdigit():
        code = sl[2:]
        suf = "SH" if sl.startswith("sh") else "SZ"
        return f"{code}.{suf}"

    if s.isdigit() and len(s) == 6:
        # ETF 常见：5xxxx 沪；其余深（够用就行，别在这搞玄学）
        suf = "SH" if s.startswith("5") else "SZ"
        return f"{s}.{suf}"

    return None


def ts_code_to_symbol(ts_code: str) -> str | None:
    """
    510150.SH -> sh510150
    """
    s = str(ts_code or "").strip().upper()
    if "." not in s:
        return None
    code, suf = s.split(".", 1)
    if len(code) != 6 or (not code.isdigit()):
        return None
    if suf == "SH":
        return f"sh{code}"
    if suf == "SZ":
        return f"sz{code}"
    return None


def prefixed_symbol_to_ts_code(sym: str) -> str | None:
    """
    sh/sz/bj 前缀形式 -> TuShare ts_code：
    - sh600000 -> 600000.SH
    - sz000725 -> 000725.SZ
    - bj430047 -> 430047.BJ
    """
    s = str(sym or "").strip().lower()
    if len(s) != 8:
        return None
    if not s.startswith(("sh", "sz", "bj")):
        return None
    code = s[2:]
    if len(code) != 6 or (not code.isdigit()):
        return None
    suf = s[:2].upper()
    return f"{code}.{suf}"
