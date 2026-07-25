"""vlm_base.py 的重試/驗證邏輯用假 backend 測，不碰任何真實 API。"""

from src.pipeline_b.vlm_base import VLMBackend

VALID_JSON = """{"doc_type": "e_invoice", "seller_name": "全家便利商店", "date": "2024-05-12",
"invoice_number": "AB12345678", "seller_tax_id": "22555003", "buyer_tax_id": null,
"total_amount": 95, "items": [{"name": "拿鐵咖啡", "amount": 60}]}"""


class ScriptedBackend(VLMBackend):
    """依序回放預先寫好的回應腳本，模擬各種模型輸出情境。"""

    name = "scripted-test-backend"

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        self.call_count += 1
        return self._responses[self.call_count - 1]


class TestExtractSuccess:
    def test_valid_json_on_first_try(self):
        backend = ScriptedBackend([VALID_JSON])
        result = backend.extract(b"fake-image")
        assert result.is_valid_json
        assert result.attempts == 1
        assert result.record["seller_name"] == "全家便利商店"
        assert result.record["items"] == [{"name": "拿鐵咖啡", "amount": 60}]
        assert result.error is None

    def test_markdown_wrapped_json_still_parses(self):
        wrapped = f"這是結果：\n```json\n{VALID_JSON}\n```\n謝謝"
        backend = ScriptedBackend([wrapped])
        result = backend.extract(b"fake-image")
        assert result.is_valid_json
        assert result.record["total_amount"] == 95


class TestExtractRetry:
    def test_recovers_after_malformed_json(self):
        backend = ScriptedBackend(["這不是 JSON，我拒絕回答", VALID_JSON])
        result = backend.extract(b"fake-image", max_retries=2)
        assert result.is_valid_json
        assert result.attempts == 2
        assert backend.call_count == 2

    def test_recovers_after_schema_invalid_json(self):
        # 第一次回應是合法 JSON 但缺必要欄位（doc_type 用了 enum 外的值）
        bad = '{"doc_type": "invoice", "seller_name": null, "date": null, "invoice_number": null, "seller_tax_id": null, "buyer_tax_id": null, "total_amount": null, "items": []}'
        backend = ScriptedBackend([bad, VALID_JSON])
        result = backend.extract(b"fake-image", max_retries=2)
        assert result.is_valid_json
        assert result.attempts == 2

    def test_all_retries_exhausted_returns_none_record(self):
        backend = ScriptedBackend(["垃圾回應1", "垃圾回應2", "垃圾回應3"])
        result = backend.extract(b"fake-image", max_retries=2)
        assert result.record is None
        assert not result.is_valid_json
        assert result.attempts == 3
        assert backend.call_count == 3
        assert result.error is not None

    def test_max_retries_zero_means_single_attempt(self):
        backend = ScriptedBackend(["垃圾回應"])
        result = backend.extract(b"fake-image", max_retries=0)
        assert backend.call_count == 1
        assert result.record is None


class TestExtractWithOcrHint:
    def test_ocr_hint_reaches_prompt(self):
        captured_prompts = []

        class CapturingBackend(VLMBackend):
            name = "capturing"

            def _call(self, image_bytes, prompt):
                captured_prompts.append(prompt)
                return VALID_JSON

        CapturingBackend().extract(b"fake-image", ocr_hint="測試 OCR 文字內容")
        assert "測試 OCR 文字內容" in captured_prompts[0]

    def test_no_ocr_hint_not_in_prompt(self):
        captured_prompts = []

        class CapturingBackend(VLMBackend):
            name = "capturing"

            def _call(self, image_bytes, prompt):
                captured_prompts.append(prompt)
                return VALID_JSON

        CapturingBackend().extract(b"fake-image")
        assert "OCR" not in captured_prompts[0]
