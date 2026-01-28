from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except (AttributeError):  # noqa: BLE001
        return

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com"


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_mode: str = "x-goog-api-key"  # or "authorization"


@dataclass(frozen=True)
class EmbeddingsConfig:
    """
    OpenAI-compatible embeddings endpoint config.

    Notes:
    - Many proxy/compatible providers reuse the OpenAI /v1/embeddings schema.
    - This config is intentionally separate from chat/completions to avoid accidental
      "use chat key for embeddings" leaks; enable explicitly via env vars.
    """

    api_key: str
    model: str
    base_url: str = "https://api.openai.com"
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class AppConfig:
    project_root: Path

    def openai(self) -> OpenAIConfig | None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
        if not api_key or not model:
            return None
        return OpenAIConfig(api_key=api_key, model=model, base_url=base_url)

    def gemini(self) -> GeminiConfig | None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL", "").strip()
        base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()
        api_key_mode = os.getenv("GEMINI_API_KEY_MODE", "x-goog-api-key").strip().lower()
        if not api_key or not model:
            return None
        if api_key_mode not in {"x-goog-api-key", "authorization"}:
            api_key_mode = "x-goog-api-key"
        return GeminiConfig(api_key=api_key, model=model, base_url=base_url, api_key_mode=api_key_mode)

    def embeddings(self) -> EmbeddingsConfig | None:
        """
        Enable embeddings explicitly (privacy): set EMBEDDINGS_API_KEY (and optionally MODEL/BASE_URL).
        Fallbacks:
        - base_url: EMBEDDINGS_BASE_URL -> OPENAI_BASE_URL -> https://api.openai.com
        - model: EMBEDDINGS_MODEL -> text-embedding-3-small
        - headers: EMBEDDINGS_HEADERS_JSON (JSON object)
        """

        api_key = os.getenv("EMBEDDINGS_API_KEY", "").strip()
        if not api_key:
            return None

        base_url = os.getenv("EMBEDDINGS_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
        model = os.getenv("EMBEDDINGS_MODEL", "").strip() or "text-embedding-3-small"

        headers: dict[str, str] | None = None
        headers_raw = os.getenv("EMBEDDINGS_HEADERS_JSON", "").strip()
        if headers_raw:
            try:
                obj = json.loads(headers_raw)
                if isinstance(obj, dict):
                    headers = {str(k): str(v) for k, v in obj.items() if str(k).strip()}
            except Exception:  # noqa: BLE001
                headers = None

        return EmbeddingsConfig(api_key=api_key, model=model, base_url=base_url, headers=headers)


def load_config() -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    return AppConfig(project_root=root)
