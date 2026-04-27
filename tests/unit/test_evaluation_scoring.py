import pytest

from src.evaluation.metrics import aggregate_cell_metrics
from src.evaluation.scoring import (
    build_article_input,
    direction_band,
    gepa_partial_credit,
    parse_score_response,
)


def test_parse_plain_json() -> None:
    p = parse_score_response('{"score": 13}')
    assert p.pred_score == 13
    assert p.parse_status == "ok"
    assert p.parse_error is None


def test_parse_markdown_fence_with_json_tag() -> None:
    p = parse_score_response('```json\n{"score": 7}\n```')
    assert p.pred_score == 7
    assert p.parse_status == "ok"


def test_parse_markdown_fence_without_tag() -> None:
    p = parse_score_response('```\n{"score": 1}\n```')
    assert p.pred_score == 1
    assert p.parse_status == "ok"


def test_parse_invalid_json() -> None:
    p = parse_score_response("not json")
    assert p.pred_score is None
    assert p.parse_status == "failed"
    assert p.parse_error is not None


def test_parse_out_of_range() -> None:
    p = parse_score_response('{"score": 99}')
    assert p.pred_score is None
    assert p.parse_status == "failed"
    assert "out of range" in p.parse_error


def test_parse_missing_key() -> None:
    p = parse_score_response('{"foo": 1}')
    assert p.pred_score is None
    assert p.parse_status == "failed"


def test_parse_empty() -> None:
    p = parse_score_response("")
    assert p.parse_status == "failed"


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        (13, 13, 1.0),
        (12, 13, 0.75),
        (14, 13, 0.75),
        (11, 13, 0.5),
        (10, 13, 0.25),
        (9, 13, 0.0),
        (1, 13, 0.0),
        (None, 13, 0.0),
    ],
)
def test_gepa_partial_credit(pred, gold, expected) -> None:
    assert gepa_partial_credit(pred, gold) == expected


@pytest.mark.parametrize(
    "score,band",
    [
        (1, "bearish"),
        (5, "bearish"),
        (6, "neutral"),
        (10, "neutral"),
        (11, "bullish"),
        (15, "bullish"),
    ],
)
def test_direction_band(score, band) -> None:
    assert direction_band(score) == band


def test_build_article_input_title_only() -> None:
    out = build_article_input({"title": "BTC moves", "text": ""})
    assert out == "Title: BTC moves"


def test_build_article_input_both() -> None:
    out = build_article_input({"title": "BTC moves", "text": "Full body."})
    assert out == "Title: BTC moves\n\nArticle text:\nFull body."


def _row(*, gold: int, pred: int | None, status: str = "ok") -> dict:
    from src.evaluation.scoring import direction_band as db
    abs_err = None if pred is None else abs(pred - gold)
    signed = None if pred is None else (pred - gold)
    return {
        "article_id": f"a-{gold}-{pred}",
        "gold_score": gold,
        "pred_score": pred,
        "abs_error": abs_err,
        "signed_error": signed,
        "gepa_score": gepa_partial_credit(pred, gold),
        "exact_match": pred is not None and pred == gold,
        "direction_correct": pred is not None and db(pred) == db(gold),
        "gold_band": db(gold),
        "pred_band": None if pred is None else db(pred),
        "parse_status": status,
        "latency_ms": 100.0,
        "n_retries": 0,
    }


def test_aggregate_cell_metrics_basic() -> None:
    rows = [
        _row(gold=13, pred=13),
        _row(gold=13, pred=12),
        _row(gold=8, pred=8),
        _row(gold=3, pred=None, status="failed"),
    ]
    m = aggregate_cell_metrics(rows)
    assert m["n_articles"] == 4
    assert m["n_parsed"] == 3
    assert m["parse_rate"] == 0.75
    assert m["mean_gepa_score"] == pytest.approx((1.0 + 0.75 + 1.0 + 0.0) / 4)
    assert m["exact_match_accuracy"] == 0.5  # 2 / 4 (rows, not just parsed)
    assert m["mean_absolute_error"] == pytest.approx(1 / 3)
    # 3 of 4 rows were direction-correct (parse failure does not count).
    assert m["direction_accuracy"] == 0.75


def test_aggregate_cell_metrics_empty() -> None:
    m = aggregate_cell_metrics([])
    assert m["n_articles"] == 0
    assert m["parse_rate"] == 0.0
