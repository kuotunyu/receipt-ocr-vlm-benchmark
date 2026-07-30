import json

import pytest

from scripts import generate_chart_captions as generator


def test_json_retry_error_preserves_failed_attempts(monkeypatch):
    responses = iter(
        [
            (
                '{"answer":"truncated"',
                {
                    "latency_seconds": 1.0,
                    "gpu_seconds": 0.8,
                    "prompt_tokens": 10,
                    "output_tokens": 512,
                    "thinking_chars": 0,
                    "done_reason": "length",
                },
            ),
            (
                "",
                {
                    "latency_seconds": 2.0,
                    "gpu_seconds": 1.8,
                    "prompt_tokens": 11,
                    "output_tokens": 512,
                    "thinking_chars": 0,
                    "done_reason": "length",
                },
            ),
        ]
    )
    monkeypatch.setattr(
        generator, "_model_call", lambda *args, **kwargs: next(responses)
    )
    with pytest.raises(generator.JsonModelCallError) as captured:
        generator._json_model_call(
            "model",
            b"image",
            "prompt",
            output_format={"type": "object"},
            num_predict=512,
        )
    assert len(captured.value.usages) == 2
    assert captured.value.usages[1]["attempt"] == 2
    assert captured.value.raw_attempts == [
        '{"answer":"truncated"',
        "",
    ]


def test_caption_checkpoint_is_durable_and_accounts_discarded_run(tmp_path):
    target = tmp_path / "captions.partial.json"
    usage = {
        "latency_seconds": 2.5,
        "gpu_seconds": 2.0,
        "prompt_tokens": 100,
        "output_tokens": 50,
        "thinking_chars": 4,
    }
    generator._write_checkpoint(
        target,
        model="qwen3-vl:8b",
        records=[{"figure_id": "chart-1"}],
        raw_records=[{"figure_id": "chart-1", "status": "completed"}],
        usages=[usage],
        discarded_run_wall_seconds=7.5,
        status="partial-running",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["status"] == "partial-running"
    assert payload["captions"][0]["figure_id"] == "chart-1"
    assert payload["generation_summary"]["call_count"] == 1
    assert payload["generation_summary"][
        "discarded_run_wall_seconds"
    ] == 7.5
