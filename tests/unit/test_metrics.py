from src.data.schemas import ArticleRecord, ExtractionRecord
from src.evaluation.metrics import direction_accuracy, macro_f1_direction, parse_failure_rate


def test_direction_metrics() -> None:
    articles = [
        ArticleRecord(
            article_id="article-001",
            text="",
            title="Bitcoin should rise",
            url="https://example.com/a1",
            source="Example News",
            date="2024-03",
            gold_score=15,
            gold_reasoning="Demand is expected to increase.",
        ),
        ArticleRecord(
            article_id="article-002",
            text="",
            title="Bitcoin should fall",
            url="https://example.com/a2",
            source="Example News",
            date="2024-04",
            gold_score=3,
            gold_reasoning="The author expects risk-off selling.",
        ),
    ]
    extractions = [
        ExtractionRecord(
            article_id="article-001",
            model_name="qwen",
            prompt_version="prompt-v001",
            pred_direction="up",
            pred_confidence="high",
            pred_reasoning="Demand is expected to increase.",
            raw_response="{}",
            parse_status="ok",
        ),
        ExtractionRecord(
            article_id="article-002",
            model_name="qwen",
            prompt_version="prompt-v001",
            raw_response="not json",
            parse_status="failed",
        ),
    ]

    assert direction_accuracy(articles, extractions) == 1.0
    assert parse_failure_rate(extractions) == 0.5
    assert macro_f1_direction(articles, extractions) == 0.5
