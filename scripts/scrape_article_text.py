import argparse
import time
from pathlib import Path
from typing import Any

import httpx
import trafilatura

from src.common.jsonl import read_jsonl, write_jsonl
from src.data.schemas import ArticleRecord

DEFAULT_INPUT = Path("data/train/articles.jsonl")
DEFAULT_OUTPUT = Path("data/train/articles_with_text.jsonl")
DEFAULT_FAILURE_OUTPUT = Path("outputs/artifacts/article_text_failures.jsonl")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def clean_extracted_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    kept_lines = [line for line in lines if line]
    return "\n".join(kept_lines).strip()


def extract_article_text(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    return clean_extracted_text(extracted or "")


def fetch_html(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def scrape_record(
    record: dict[str, Any],
    client: httpx.Client,
    min_chars: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    article = ArticleRecord.model_validate(record)
    try:
        html = fetch_html(client, str(article.url))
        text = extract_article_text(html, str(article.url))
        if len(text) < min_chars:
            return record, {
                "article_id": article.article_id,
                "url": str(article.url),
                "source": article.source,
                "reason": f"extracted text shorter than min_chars={min_chars}",
                "text_length": len(text),
            }
        updated = dict(record)
        updated["text"] = text
        return updated, None
    except Exception as exc:
        return record, {
            "article_id": article.article_id,
            "url": str(article.url),
            "source": article.source,
            "reason": str(exc),
        }


def scrape_jsonl(
    input_path: Path,
    output_path: Path,
    failure_output_path: Path,
    *,
    dry_run: bool,
    limit: int | None,
    overwrite_existing_text: bool,
    min_chars: int,
    delay_seconds: float,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(input_path)
    if limit is not None:
        records_to_process = records[:limit]
        untouched_records = records[limit:]
    else:
        records_to_process = records
        untouched_records = []

    updated_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_seconds) as client:
        for index, record in enumerate(records_to_process, start=1):
            if record.get("text") and not overwrite_existing_text:
                updated_records.append(record)
                continue

            updated, failure = scrape_record(record, client, min_chars)
            updated_records.append(updated)
            if failure is not None:
                failures.append(failure)

            print(
                f"[{index}/{len(records_to_process)}] "
                f"{record.get('article_id', '<missing-id>')} "
                f"{'failed' if failure else 'ok'}"
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    final_records = updated_records + untouched_records
    if not dry_run:
        write_jsonl(output_path, final_records)
        write_jsonl(failure_output_path, failures)

    return final_records, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape article body text into a new JSONL without modifying the source JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite-existing-text", action="store_true")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, failures = scrape_jsonl(
        args.input,
        args.output,
        args.failure_output,
        dry_run=args.dry_run,
        limit=args.limit,
        overwrite_existing_text=args.overwrite_existing_text,
        min_chars=args.min_chars,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    populated_count = sum(1 for record in records if record.get("text"))
    print(f"Records: {len(records)}")
    print(f"Records with text: {populated_count}")
    print(f"Failures: {len(failures)}")
    if args.dry_run:
        print("Dry run only; no files were written.")
    else:
        print(f"Wrote updated JSONL to {args.output}")
        print(f"Wrote failures to {args.failure_output}")


if __name__ == "__main__":
    main()
