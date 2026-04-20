from collections import Counter

from src.data.schemas import ArticleRecord, ExtractionRecord, ParseStatus


def direction_accuracy(articles: list[ArticleRecord], extractions: list[ExtractionRecord]) -> float:
    gold_by_id = {article.article_id: article.gold_direction for article in articles}
    comparable = [
        extraction
        for extraction in extractions
        if extraction.parse_status == ParseStatus.ok
        and extraction.article_id in gold_by_id
        and extraction.pred_direction is not None
    ]
    if not comparable:
        return 0.0
    correct = sum(1 for extraction in comparable if is_direction_match(extraction, gold_by_id))
    return correct / len(comparable)


def parse_failure_rate(extractions: list[ExtractionRecord]) -> float:
    if not extractions:
        return 0.0
    failed = sum(1 for extraction in extractions if extraction.parse_status == ParseStatus.failed)
    return failed / len(extractions)


def macro_f1_direction(articles: list[ArticleRecord], extractions: list[ExtractionRecord]) -> float:
    gold_by_id = {article.article_id: article.gold_direction for article in articles}
    labels = sorted({article.gold_direction for article in articles}, key=str)
    predictions = [
        (gold_by_id[extraction.article_id], extraction.pred_direction)
        for extraction in extractions
        if extraction.parse_status == ParseStatus.ok
        and extraction.article_id in gold_by_id
        and extraction.pred_direction is not None
    ]
    if not predictions:
        return 0.0

    scores: list[float] = []
    for label in labels:
        counts = Counter(
            "tp"
            if gold == label and pred == label
            else "fp"
            if gold != label and pred == label
            else "fn"
            if gold == label and pred != label
            else "tn"
            for gold, pred in predictions
        )
        precision_total = counts["tp"] + counts["fp"]
        recall_total = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_total if precision_total else 0
        recall = counts["tp"] / recall_total if recall_total else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def is_direction_match(
    extraction: ExtractionRecord,
    gold_by_id: dict[str, object],
) -> bool:
    return extraction.pred_direction == gold_by_id[extraction.article_id]
