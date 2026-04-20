from pathlib import Path

from src.common.jsonl import read_jsonl
from src.data.schemas import ArticleRecord, ExtractionRecord


def load_articles(path: Path) -> list[ArticleRecord]:
    return [ArticleRecord.model_validate(record) for record in read_jsonl(path)]


def load_extractions(path: Path) -> list[ExtractionRecord]:
    return [ExtractionRecord.model_validate(record) for record in read_jsonl(path)]
