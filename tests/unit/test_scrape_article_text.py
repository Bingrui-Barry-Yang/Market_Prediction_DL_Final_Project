from scripts.scrape_article_text import clean_extracted_text, scrape_jsonl
from src.common.jsonl import read_jsonl, write_jsonl


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        raise AssertionError(f"Unexpected network call to {url}")


def test_clean_extracted_text_removes_blank_lines() -> None:
    assert clean_extracted_text(" First line \n\n Second line ") == "First line\nSecond line"


def test_scrape_jsonl_dry_run_does_not_write(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "articles.jsonl"
    output_path = tmp_path / "articles_with_text.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "article_id": "article-001",
                "text": "",
                "title": "Bitcoin rises",
                "url": "https://example.com/article",
                "source": "Example",
                "date": "2024-03",
                "gold_score": 15,
                "gold_reasoning": "Bullish title.",
            }
        ],
    )

    class Response:
        text = "<html><article>Long enough article body about Bitcoin.</article></html>"

        def raise_for_status(self) -> None:
            return None

    class Client:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            return Response()

    monkeypatch.setattr("scripts.scrape_article_text.httpx.Client", Client)
    monkeypatch.setattr(
        "scripts.scrape_article_text.extract_article_text",
        lambda html, url: "Long enough article body about Bitcoin.",
    )

    records, failures = scrape_jsonl(
        input_path,
        output_path,
        failure_path,
        dry_run=True,
        limit=None,
        overwrite_existing_text=False,
        min_chars=10,
        delay_seconds=0,
        timeout_seconds=1,
    )

    assert records[0]["text"] == "Long enough article body about Bitcoin."
    assert failures == []
    assert not output_path.exists()
    assert not failure_path.exists()
    assert read_jsonl(input_path)[0]["text"] == ""
