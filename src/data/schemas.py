from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Direction(str, Enum):
    up = "up"
    down = "down"
    neutral = "neutral"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ParseStatus(str, Enum):
    ok = "ok"
    failed = "failed"


class ArticleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: HttpUrl
    source: str = Field(min_length=1)
    date: str = Field(min_length=4)
    gold_direction: Direction
    gold_confidence: Confidence
    gold_reasoning: str = Field(min_length=1)


class ExtractionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    pred_direction: Direction | None = None
    pred_confidence: Confidence | None = None
    pred_reasoning: str | None = None
    raw_response: str
    parse_status: ParseStatus
