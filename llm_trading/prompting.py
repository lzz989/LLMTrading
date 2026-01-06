from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_client import ChatMessage


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


def load_prompt_text(prompt_path: str | Path) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8")


def dataframe_to_llm_csv(df, *, max_rows: int = 300) -> str:
    df2 = df.copy()
    df2["date"] = df2["date"].dt.strftime("%Y-%m-%d")

    if len(df2) > max_rows:
        try:
            import numpy as np
        except ModuleNotFoundError:
            df2 = df2.tail(max_rows)
        else:
            idx = np.linspace(0, len(df2) - 1, max_rows).round().astype(int)
            df2 = df2.iloc[idx]

    cols = [c for c in ["date", "open", "high", "low", "close", "volume", "ma50", "ma200"] if c in df2.columns]
    return df2[cols].to_csv(index=False)


def build_messages(
    prompt_text: str,
    *,
    df,
    max_rows: int = 300,
    extra_json: Any | None = None,
    system_text: str | None = None,
) -> list[ChatMessage]:
    csv_text = dataframe_to_llm_csv(df, max_rows=max_rows)
    meta = {
        "rows": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
    }

    system = system_text or "你是一个严谨的技术分析助手。你必须按用户要求输出。"
    user = (
        prompt_text.strip()
        + "\n\n"
        + "数据元信息(JSON)：\n"
        + json.dumps(meta, ensure_ascii=False)
        + ("\n\n结构信息(JSON)：\n" + json.dumps(extra_json, ensure_ascii=False) if extra_json is not None else "")
        + "\n\n"
        + "数据如下(CSV)：\n"
        + csv_text
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def extract_first_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 输出里没找到 JSON 对象。")
    blob = text[start : end + 1]
    return json.loads(blob)
