from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .json_utils import sanitize_for_json
from .llm_client import ChatMessage, gemini_generate_content, openai_chat_completion
from .prompting import build_messages, extract_first_json, load_prompt_text


def run_llm_analysis(
    cfg: AppConfig,
    *,
    df,
    prompt_path: str,
    max_rows: int = 300,
    extra_json: Any | None = None,
    system_text: str | None = None,
) -> dict[str, Any]:
    openai_cfg = cfg.openai()
    if not openai_cfg:
        raise RuntimeError("要用 --llm 就把 OPENAI_API_KEY 和 OPENAI_MODEL 配好，别让我猜你想干嘛。")

    prompt_text = load_prompt_text(prompt_path)
    messages = build_messages(prompt_text, df=df, max_rows=max_rows, extra_json=extra_json, system_text=system_text)
    raw = openai_chat_completion(openai_cfg, messages=messages, temperature=0.2)
    return extract_first_json(raw)


def run_llm_text(
    cfg: AppConfig,
    *,
    messages: list[ChatMessage],
    provider: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> str:
    p = (provider or "").strip().lower()
    if p == "openai":
        openai_cfg = cfg.openai()
        if not openai_cfg:
            raise RuntimeError("要用 OpenAI 就把 OPENAI_API_KEY 和 OPENAI_MODEL 配好，别让我猜你想干嘛。")
        return openai_chat_completion(
            openai_cfg,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_output_tokens) if max_output_tokens is not None else None,
        )
    if p == "gemini":
        gemini_cfg = cfg.gemini()
        if not gemini_cfg:
            raise RuntimeError(
                "要用 Gemini 就把 GEMINI_API_KEY 和 GEMINI_MODEL 配好（AI Studio API key），别让我猜你想干嘛。"
            )
        return gemini_generate_content(
            gemini_cfg,
            messages=messages,
            temperature=float(temperature),
            max_output_tokens=int(max_output_tokens),
        )
    raise ValueError(f"未知 LLM provider：{provider}")


def write_json(path: str | Path, obj: Any):
    p = Path(path)
    clean = sanitize_for_json(obj)
    p.write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
