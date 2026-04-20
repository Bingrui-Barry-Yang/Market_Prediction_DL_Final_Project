from typing import Any

import orjson

from src.data.schemas import Confidence, Direction, ExtractionRecord, ParseStatus


def parse_extraction_response(
    *,
    article_id: str,
    model_name: str,
    prompt_version: str,
    raw_response: str,
) -> ExtractionRecord:
    try:
        payload: dict[str, Any] = orjson.loads(raw_response)
        return ExtractionRecord(
            article_id=article_id,
            model_name=model_name,
            prompt_version=prompt_version,
            pred_direction=Direction(payload["direction"]),
            pred_confidence=Confidence(payload["confidence"]),
            pred_reasoning=str(payload["reasoning"]),
            raw_response=raw_response,
            parse_status=ParseStatus.ok,
        )
    except Exception:
        return ExtractionRecord(
            article_id=article_id,
            model_name=model_name,
            prompt_version=prompt_version,
            raw_response=raw_response,
            parse_status=ParseStatus.failed,
        )
