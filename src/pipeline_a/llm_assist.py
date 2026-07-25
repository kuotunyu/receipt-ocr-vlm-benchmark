"""Ollama qwen3:4b 補漏：只在規則抽取「完全沒抓到品項」時才呼叫，
規則抽得到就不驚動 LLM——維持「傳統管線」的故事線乾淨、成本可預期。

本機若未安裝/啟動 Ollama，`refine_items` 會拋出 `OllamaUnavailable`，
呼叫端（assemble.py）需捕捉並優雅退回規則結果（等同 `--no-llm` 行為）。
"""

from __future__ import annotations

import json
import re


class OllamaUnavailable(Exception):
    pass


_PROMPT_TEMPLATE = """你是發票/收據品項抽取助手。以下是 OCR 逐行辨識出的文字（可能有斷行錯誤或雜訊）：

{lines}

請找出品項清單，只輸出一個 JSON 陣列，每個元素為 {{"name": "品項名稱", "amount": 金額數字}}。
不要輸出任何其他文字、不要用 markdown code block，找不到任何品項就輸出 []。"""


def _extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"回應內未找到 JSON 陣列：{text!r}")
    return json.loads(match.group(0))


def refine_items(raw_lines: list[str], model: str = "qwen3:4b") -> list[dict]:
    try:
        import ollama
    except ImportError as exc:
        raise OllamaUnavailable("未安裝 ollama 套件") from exc

    prompt = _PROMPT_TEMPLATE.format(lines="\n".join(raw_lines))
    try:
        response = ollama.generate(model=model, prompt=prompt)
    except Exception as exc:  # noqa: BLE001 — Ollama 服務未啟動/模型未 pull 等都算不可用
        raise OllamaUnavailable(str(exc)) from exc

    try:
        items = _extract_json_array(response["response"])
    except (ValueError, json.JSONDecodeError) as exc:
        raise OllamaUnavailable(f"LLM 回應無法解析為 JSON：{exc}") from exc

    return [
        {"name": item.get("name"), "amount": item.get("amount")}
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]
