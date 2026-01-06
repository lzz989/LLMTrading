from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import GeminiConfig, OpenAIConfig


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def openai_chat_completion(
    cfg: OpenAIConfig,
    *,
    messages: list[ChatMessage],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    base = cfg.base_url.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise LlmError(f"OpenAI HTTPError {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LlmError(f"OpenAI URLError: {exc}") from exc

    try:
        obj: dict[str, Any] = json.loads(raw)
        return obj["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise LlmError(f"OpenAI 返回解析失败：{raw[:500]}") from exc


def gemini_generate_content(
    cfg: GeminiConfig,
    *,
    messages: list[ChatMessage],
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> str:
    url = cfg.base_url.rstrip("/") + f"/models/{cfg.model}:generateContent"

    system_texts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = (m.role or "").strip().lower()
        if role == "system":
            system_texts.append(m.content)
            continue
        if role == "assistant":
            g_role = "model"
        else:
            g_role = "user"
        contents.append({"role": g_role, "parts": [{"text": m.content}]})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    system = "\n\n".join([t.strip() for t in system_texts if t and t.strip()]).strip()
    if system:
        payload["systemInstruction"] = {"role": "system", "parts": [{"text": system}]}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if cfg.api_key_mode == "authorization":
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    else:
        headers["x-goog-api-key"] = cfg.api_key

    req = urllib.request.Request(url=url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise LlmError(f"Gemini HTTPError {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LlmError(f"Gemini URLError: {exc}") from exc

    try:
        obj: dict[str, Any] = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise LlmError(f"Gemini 返回解析失败：{raw[:500]}") from exc

    try:
        cands = obj.get("candidates") or []
        if not cands:
            raise KeyError("missing candidates")
        content = (cands[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
        out = "".join(texts).strip()
        if out:
            return out
        raise KeyError("empty text parts")
    except Exception as exc:  # noqa: BLE001
        raise LlmError(f"Gemini 返回结构异常：{raw[:500]}") from exc
