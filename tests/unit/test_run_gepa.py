import json
import sys
from pathlib import Path

import pytest

import scripts.run_gepa as run_gepa


def article_record(index: int, score: int = 13) -> dict[str, object]:
    return {
        "article_id": f"article-{index:03d}",
        "text": f"Full article text {index}.",
        "title": f"Bitcoin outlook {index}",
        "url": f"https://example.com/{index}",
        "source": "Example News",
        "date": "2024-03",
        "gold_score": score,
        "gold_reasoning": "The article expects Bitcoin to move higher.",
    }


def write_articles(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, count + 1):
            handle.write(json.dumps(article_record(index)) + "\n")


def test_sentiment_score_evaluator_exact_match() -> None:
    evaluator = run_gepa.SentimentScoreEvaluator()
    result = evaluator(
        {"answer": "13", "additional_context": {"gold_reasoning": "Bullish demand."}},
        '{"score": 13}',
    )

    assert result.score == 1.0
    assert "Correct" in result.feedback


def test_sentiment_score_evaluator_partial_and_failed_parse() -> None:
    evaluator = run_gepa.SentimentScoreEvaluator()
    data = {"answer": "13", "additional_context": {"gold_reasoning": "Bullish demand."}}

    partial = evaluator(data, "```json\n{\"score\": 12}\n```")
    failed = evaluator(data, "not json")

    assert partial.score == 0.75
    assert "too bearish" in partial.feedback
    assert failed.score == 0.0
    assert "Failed to parse response" in failed.feedback


def test_build_article_input_prefers_title_and_text() -> None:
    text = run_gepa.build_article_input(
        {
            "title": "Bitcoin may rise",
            "text": "Analysts expect stronger demand.",
        }
    )

    assert text == "Title: Bitcoin may rise\n\nArticle text:\nAnalysts expect stronger demand."


def test_load_jsonl_uses_every_fifth_record_for_validation(tmp_path, capsys) -> None:
    data_path = tmp_path / "articles.jsonl"
    write_articles(data_path, 10)

    trainset, valset = run_gepa.load_jsonl(str(data_path))

    assert len(trainset) == 8
    assert len(valset) == 2
    assert valset[0]["input"].startswith("Title: Bitcoin outlook 1")
    assert valset[1]["input"].startswith("Title: Bitcoin outlook 6")
    assert "Loaded 8 train, 2 val examples" in capsys.readouterr().out


def test_main_dry_run_does_not_call_gepa(tmp_path, monkeypatch, capsys) -> None:
    data_path = tmp_path / "articles.jsonl"
    write_articles(data_path, 10)

    def fail_optimize(*args, **kwargs):
        raise AssertionError("GEPA should not be called during dry runs")

    monkeypatch.setattr(run_gepa.gepa, "optimize", fail_optimize)
    monkeypatch.setattr(sys, "argv", ["run_gepa.py", "--dry-run", str(data_path)])

    run_gepa.main()

    assert "Dry run complete" in capsys.readouterr().out


def test_main_requires_enough_training_and_validation_examples(tmp_path, monkeypatch) -> None:
    data_path = tmp_path / "articles.jsonl"
    write_articles(data_path, 6)
    monkeypatch.setattr(sys, "argv", ["run_gepa.py", "--dry-run", str(data_path)])

    with pytest.raises(ValueError, match="Need at least 5 training examples"):
        run_gepa.main()
