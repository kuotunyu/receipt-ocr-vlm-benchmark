"""Generate chart captions from original pixels for retrieval experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.caption_index import (
    GENERIC_CAPTION_SCHEMA,
    PIXEL_ANSWER_SCHEMA,
)
from src.complex_document.ollama_compat import extract_ollama_text
from src.complex_document.parsers.base import ParserUnavailable
from src.complex_document.parsers.vlm_parser import QwenVLMParserAdapter

_STRUCTURED_PROMPT = """請直接分析圖表原始像素，輸出單一 JSON object，不要 code fence：
{"axis_names":["軸名"],"unit":"單位或 null","series":["系列"],
"values":["標籤:數值"],"trend":"主要趨勢","structured_caption":"繁體中文完整索引文字"}
structured_caption 必須包含軸名、單位、series、可讀的主要數值與趨勢。
看不清楚就明確寫無法辨識，不得臆測。"""
_STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "axis_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unit": {"type": ["string", "null"]},
        "series": {
            "type": "array",
            "items": {"type": "string"},
        },
        "values": {
            "type": "array",
            "items": {"type": "string"},
        },
        "trend": {"type": "string"},
        "structured_caption": {"type": "string"},
    },
    "required": [
        "axis_names",
        "unit",
        "series",
        "values",
        "trend",
        "structured_caption",
    ],
    "additionalProperties": False,
}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class JsonModelCallError(ValueError):
    def __init__(
        self,
        errors: list[str],
        usages: list[dict],
        raw_attempts: list[str],
    ):
        super().__init__(
            f"caption JSON failed after {len(usages)} attempts: "
            + " | ".join(errors)
        )
        self.usages = usages
        self.raw_attempts = raw_attempts
        self.errors = errors


def _model_call(
    model: str,
    image_bytes: bytes,
    prompt: str,
    *,
    output_format: dict | None = None,
    num_predict: int = 1024,
) -> tuple[str, dict]:
    import ollama

    started = time.perf_counter()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        images=[image_bytes],
        think=False,
        format=output_format,
        options={"temperature": 0, "num_predict": num_predict},
    )
    latency_seconds = time.perf_counter() - started
    text, output_channel = extract_ollama_text(response)
    if isinstance(response, dict):
        usage = {
            "latency_seconds": round(latency_seconds, 6),
            "prompt_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"),
            "gpu_seconds": round(
                (
                    (response.get("prompt_eval_duration") or 0)
                    + (response.get("eval_duration") or 0)
                )
                / 1e9,
                6,
            ),
            "thinking_chars": len(str(response.get("thinking") or "")),
            "output_channel": output_channel,
            "done_reason": response.get("done_reason"),
        }
    else:
        usage = {
            "latency_seconds": round(latency_seconds, 6),
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "output_tokens": getattr(response, "eval_count", None),
            "gpu_seconds": round(
                (
                    (getattr(response, "prompt_eval_duration", 0) or 0)
                    + (getattr(response, "eval_duration", 0) or 0)
                )
                / 1e9,
                6,
            ),
            "thinking_chars": len(
                str(getattr(response, "thinking", "") or "")
            ),
            "output_channel": output_channel,
            "done_reason": getattr(response, "done_reason", None),
        }
    return text, usage


def _json_object(text: str) -> dict:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("caption response does not contain JSON")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("caption response root is not an object")
    return value


def _json_model_call(
    model: str,
    image_bytes: bytes,
    prompt: str,
    *,
    output_format: dict,
    num_predict: int,
    max_attempts: int = 2,
) -> tuple[dict, str, list[dict], list[str]]:
    """Call a local VLM with a bounded, fully accounted JSON retry."""
    usages: list[dict] = []
    raw_attempts: list[str] = []
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n前一次輸出不是有效 JSON。請重新檢視原始像素，"
                "只輸出符合 schema 的單一 JSON object。"
            )
        raw, usage = _model_call(
            model,
            image_bytes,
            attempt_prompt,
            output_format=output_format,
            num_predict=num_predict,
        )
        usages.append({**usage, "attempt": attempt})
        raw_attempts.append(raw)
        try:
            return _json_object(raw), raw, usages, raw_attempts
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    raise JsonModelCallError(errors, usages, raw_attempts)


def _generation_summary(
    usages: list[dict], discarded_run_wall_seconds: float
) -> dict:
    return {
        "call_count": len(usages),
        "latency_seconds_total": round(
            sum(item["latency_seconds"] for item in usages), 6
        ),
        "gpu_seconds_total": round(
            sum(item["gpu_seconds"] for item in usages), 6
        ),
        "prompt_tokens_total": sum(
            int(item["prompt_tokens"] or 0) for item in usages
        ),
        "output_tokens_total": sum(
            int(item["output_tokens"] or 0) for item in usages
        ),
        "thinking_chars_total": sum(
            int(item["thinking_chars"] or 0) for item in usages
        ),
        "discarded_run_wall_seconds": round(
            discarded_run_wall_seconds, 6
        ),
    }


def _write_checkpoint(
    path: Path,
    *,
    model: str,
    records: list[dict],
    raw_records: list[dict],
    usages: list[dict],
    discarded_run_wall_seconds: float,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "model": model,
                "generation_summary": _generation_summary(
                    usages, discarded_run_wall_seconds
                ),
                "captions": records,
                "raw_records": raw_records,
                "usages": usages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--smoke", action="store_true", help="process one chart only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed targets from the durable partial artifact",
    )
    parser.add_argument(
        "--discarded-run-wall-seconds",
        type=float,
        default=0.0,
        help="Disclose overhead from an earlier run that predates checkpoints.",
    )
    parser.add_argument(
        "--structured-num-predict",
        type=int,
        default=1024,
        help="Structured JSON output cap; raise only after a recorded length stop.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/complex_document/manifest.json")
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("data/complex_document/gold/chart_targets.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/complex_document/questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/complex_document/chart_captions/qwen3-vl.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/complex_document/raw"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/complex_document"),
    )
    args = parser.parse_args()

    checker = QwenVLMParserAdapter(model=args.model)
    try:
        checker._ensure_model()
        checker.ensure_gpu_available()
    except ParserUnavailable as exc:
        print(f"SKIP: {exc}")
        raise SystemExit(0)

    import fitz

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = {item["document_id"]: item for item in manifest["documents"]}
    targets = json.loads(args.targets.read_text(encoding="utf-8"))["targets"]
    questions = {
        item["question_id"]: item
        for item in json.loads(
            args.questions.read_text(encoding="utf-8")
        )["questions"]
    }
    if args.smoke:
        targets = targets[:1]
    store = ArtifactStore(args.artifact_root)
    checkpoint_path = args.output.with_suffix(".partial.json")
    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )
        if checkpoint.get("model") != args.model:
            raise ValueError("checkpoint model does not match requested model")
        records = list(checkpoint.get("captions", []))
        raw_records = list(checkpoint.get("raw_records", []))
        all_usage = list(checkpoint.get("usages", []))
        for record in records:
            record.setdefault("generation_usage", {}).setdefault(
                "structured_caption", {}
            ).setdefault("num_predict", 1024)
    else:
        records = []
        raw_records = []
        all_usage = []
    completed_figure_ids = {
        record["figure_id"] for record in records
    }
    for target in targets:
        if target["figure_id"] in completed_figure_ids:
            continue
        checker.ensure_gpu_available()
        document = documents[target["document_id"]]
        pdf_path = args.raw_dir / document["filename"]
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"{pdf_path} is missing; run download_complex_documents.py"
            )
        with fitz.open(pdf_path) as pdf:
            page = pdf[target["page"] - 1]
            x0, y0, x1, y1 = target["bbox_normalized"]
            clip = fitz.Rect(
                x0 * page.rect.width,
                y0 * page.rect.height,
                x1 * page.rect.width,
                y1 * page.rect.height,
            )
            image_bytes = page.get_pixmap(clip=clip, dpi=180, alpha=False).tobytes(
                "png"
            )
        crop_path = (
            store.crop_dir(target["document_id"], "qwen3-vl-caption")
            / f"{target['figure_id']}.png"
        )
        crop_path.write_bytes(image_bytes)
        stage = "generic_caption"
        stage_num_predict = 512
        try:
            generic_value, generic_raw, generic_usages, generic_raw_attempts = (
                _json_model_call(
                    args.model,
                    image_bytes,
                    (
                        target["generic_prompt"]
                        + '\n只回傳 JSON object：{"caption":"一句繁體中文描述"}。'
                    ),
                    output_format=GENERIC_CAPTION_SCHEMA,
                    num_predict=512,
                )
            )
            generic = generic_value.get("caption", "")
            all_usage.extend(generic_usages)
            checker.ensure_gpu_available()
            stage = "structured_caption"
            stage_num_predict = args.structured_num_predict
            (
                structured,
                structured_raw,
                structured_usages,
                structured_raw_attempts,
            ) = _json_model_call(
                args.model,
                image_bytes,
                _STRUCTURED_PROMPT,
                output_format=_STRUCTURED_SCHEMA,
                num_predict=args.structured_num_predict,
            )
            all_usage.extend(structured_usages)
            pixel_answers = []
            for question_id in target["question_ids"]:
                checker.ensure_gpu_available()
                question = questions[question_id]
                stage = f"pixel_answer:{question_id}"
                stage_num_predict = 512
                (
                    answer_value,
                    answer_raw,
                    answer_usages,
                    answer_raw_attempts,
                ) = _json_model_call(
                    args.model,
                    image_bytes,
                    (
                        "請只根據這張原始圖表 crop 的像素回答問題。"
                        "若像素不足以回答，請回答「無法判讀」。"
                        "不得參考任何 caption。\n問題："
                        + question["question"]
                    ),
                    output_format=PIXEL_ANSWER_SCHEMA,
                    num_predict=512,
                )
                answer = answer_value.get("answer", "")
                pixel_answers.append(
                    {
                        "question_id": question_id,
                        "answer": answer.strip(),
                        "usage": {
                            "attempt_count": len(answer_usages),
                            "attempts": answer_usages,
                        },
                        "raw_attempts": answer_raw_attempts,
                    }
                )
                all_usage.extend(answer_usages)
        except JsonModelCallError as exc:
            all_usage.extend(exc.usages)
            raw_records.append(
                {
                    "status": "failed",
                    "figure_id": target["figure_id"],
                    "stage": stage,
                    "error": str(exc),
                    "num_predict": stage_num_predict,
                    "raw_attempts": exc.raw_attempts,
                    "usage": exc.usages,
                }
            )
            _write_checkpoint(
                checkpoint_path,
                model=args.model,
                records=records,
                raw_records=raw_records,
                usages=all_usage,
                discarded_run_wall_seconds=args.discarded_run_wall_seconds,
                status="partial-failed",
            )
            raise
        records.append(
            {
                **target,
                "model": args.model,
                "generic_caption": generic.strip(),
                "structured_caption": structured.get("structured_caption", ""),
                "axis_names": structured.get("axis_names", []),
                "unit": structured.get("unit"),
                "series": structured.get("series", []),
                "values": structured.get("values", []),
                "trend": structured.get("trend"),
                "crop_ref": str(crop_path.as_posix()),
                "pixel_answers": pixel_answers,
                "generation_usage": {
                    "generic_caption": {
                        "attempt_count": len(generic_usages),
                        "attempts": generic_usages,
                    },
                    "structured_caption": {
                        "attempt_count": len(structured_usages),
                        "attempts": structured_usages,
                        "num_predict": args.structured_num_predict,
                    },
                },
            }
        )
        raw_records.append(
            {
                "status": "completed",
                "figure_id": target["figure_id"],
                "generic_response": generic_raw,
                "structured_response": structured_raw,
                "generic_raw_attempts": generic_raw_attempts,
                "structured_raw_attempts": structured_raw_attempts,
                "generic_usage": generic_usages,
                "structured_usage": structured_usages,
                "pixel_answers": pixel_answers,
            }
        )
        _write_checkpoint(
            checkpoint_path,
            model=args.model,
            records=records,
            raw_records=raw_records,
            usages=all_usage,
            discarded_run_wall_seconds=args.discarded_run_wall_seconds,
            status="partial-running",
        )
    generation_summary = _generation_summary(
        all_usage, args.discarded_run_wall_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "config": {
                    "think": False,
                    "generic_num_predict": 512,
                    "structured_num_predict_values": sorted(
                        {
                            record["generation_usage"][
                                "structured_caption"
                            ].get("num_predict", 1024)
                            for record in records
                        }
                    ),
                    "pixel_answer_num_predict": 512,
                    "temperature": 0,
                    "json_max_attempts": 2,
                    "checkpoint_resume": True,
                },
                "generation_summary": generation_summary,
                "captions": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    store.write_parser_raw(
        "chart-caption-batch", "qwen3-vl-caption", {"responses": raw_records}
    )
    _write_checkpoint(
        checkpoint_path,
        model=args.model,
        records=records,
        raw_records=raw_records,
        usages=all_usage,
        discarded_run_wall_seconds=args.discarded_run_wall_seconds,
        status="completed",
    )
    print(f"wrote {len(records)} captions to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except ParserUnavailable as exc:
        print(f"PAUSE: {exc}")
