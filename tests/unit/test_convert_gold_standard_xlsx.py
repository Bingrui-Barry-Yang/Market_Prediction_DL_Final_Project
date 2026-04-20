import pytest

from scripts.convert_gold_standard_xlsx import convert_xlsx_to_jsonl, integrated_gold_score
from src.common.jsonl import read_jsonl


def test_integrated_gold_score_bands() -> None:
    assert integrated_gold_score("-1.0", "1.0") == 1
    assert integrated_gold_score("-1.0", "5.0") == 5
    assert integrated_gold_score("0.0", "1.0") == 6
    assert integrated_gold_score("0.0", "5.0") == 10
    assert integrated_gold_score("1.0", "1.0") == 11
    assert integrated_gold_score("1.0", "5.0") == 15


def test_convert_xlsx_to_jsonl_includes_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.convert_gold_standard_xlsx.read_first_sheet_rows",
        lambda path: [
            [
                "month",
                "source",
                "title",
                "url",
                "direction",
                "confidence",
                "notes",
                "text",
            ],
            [
                "45352.0",
                "Example News",
                "Bitcoin could move higher",
                "https://example.com/article",
                "1.0",
                "5.0",
                "The article is strongly bullish.",
                " Full worksheet article text. ",
            ],
        ],
    )

    output_path = tmp_path / "articles.jsonl"
    count = convert_xlsx_to_jsonl(tmp_path / "input.xlsx", output_path)
    records = read_jsonl(output_path)

    assert count == 1
    assert records[0]["article_id"] == "btc-gepa-train-001"
    assert records[0]["text"] == "Full worksheet article text."
    assert records[0]["gold_score"] == 15


def test_convert_xlsx_to_jsonl_requires_text_header(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.convert_gold_standard_xlsx.read_first_sheet_rows",
        lambda path: [
            ["month", "source", "title", "url", "direction", "confidence", "notes"],
            [
                "45352.0",
                "Example News",
                "Bitcoin could move higher",
                "https://example.com/article",
                "1.0",
                "5.0",
                "The article is strongly bullish.",
            ],
        ],
    )

    with pytest.raises(ValueError, match=r"Missing required columns: \['text'\]"):
        convert_xlsx_to_jsonl(tmp_path / "input.xlsx", tmp_path / "articles.jsonl")
