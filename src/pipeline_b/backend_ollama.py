"""本地 Qwen3-VL-8B-Instruct，透過 Ollama 跑（GGUF 量化，12GB VRAM 可跑）。

需另外安裝 Ollama 應用程式（https://ollama.com）並 `ollama pull qwen3-vl:8b`，
本專案的 pip 依賴無法幫你裝好這個系統層級的服務。
"""

from __future__ import annotations

from src.pipeline_b.vlm_base import VLMBackend


class OllamaVLMBackend(VLMBackend):
    name = "qwen3-vl-8b-local"

    def __init__(self, model: str = "qwen3-vl:8b"):
        self.model = model

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        import ollama

        response = ollama.generate(model=self.model, prompt=prompt, images=[image_bytes])
        # eval_duration/prompt_eval_duration 是「純推論」奈秒數，不含模型載入 VRAM 的一次性成本
        # （那筆算在 latency_seconds 的 cold start 裡），拿來估 GPU 運算成本比較準。
        gpu_ns = (response.get("eval_duration") or 0) + (response.get("prompt_eval_duration") or 0)
        self._last_usage = {
            "input_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"),
            "gpu_seconds": gpu_ns / 1e9,
        }
        return response["response"]
