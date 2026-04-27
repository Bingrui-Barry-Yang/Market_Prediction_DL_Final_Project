import argparse
import datetime as dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from src.common.jsonl import write_jsonl
from src.data.schemas import ArticleRecord

DEFAULT_INPUT = Path("data/train/BTC Project - Human Gold Standard Dataset (GEPA Input).xlsx")
DEFAULT_OUTPUT = Path("data/train/articles.jsonl")
DEFAULT_ARTICLE_ID_PREFIX = "btc-gepa-train"

SPREADSHEET_NS = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

SCORE_OFFSET_BY_DIRECTION = {
    "-1": 0,
    "-1.0": 0,
    "0": 5,
    "0.0": 5,
    "1": 10,
    "1.0": 10,
}


def column_name(cell_reference: str) -> str:
    return re.sub(r"\d+", "", cell_reference)


def column_index(name: str) -> int:
    index = 0
    for character in name:
        index = index * 26 + ord(character.upper()) - ord("A") + 1
    return index - 1


def excel_serial_date_to_month(value: str) -> str:
    parsed = float(value)
    date = dt.datetime(1899, 12, 30) + dt.timedelta(days=parsed)
    return date.strftime("%Y-%m")


def integrated_gold_score(direction_value: str, confidence_value: str) -> int:
    direction_key = direction_value.strip()
    confidence = int(float(confidence_value.strip()))
    if confidence < 1 or confidence > 5:
        raise ValueError(f"Confidence must be in 1-5, got {confidence_value!r}")
    return SCORE_OFFSET_BY_DIRECTION[direction_key] + confidence


def read_shared_strings(workbook: ZipFile) -> list[str]:
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("xlsx:si", SPREADSHEET_NS):
        text_nodes = item.findall(".//xlsx:t", SPREADSHEET_NS)
        strings.append("".join(text.text or "" for text in text_nodes))
    return strings


def read_first_sheet_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as workbook:
        shared_strings = read_shared_strings(workbook)
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.findall(".//xlsx:sheetData/xlsx:row", SPREADSHEET_NS):
        values: list[str] = []
        for cell in row.findall("xlsx:c", SPREADSHEET_NS):
            index = column_index(column_name(cell.attrib["r"]))
            while len(values) <= index:
                values.append("")

            value_node = cell.find("xlsx:v", SPREADSHEET_NS)
            value = "" if value_node is None else value_node.text or ""
            if cell.attrib.get("t") == "s" and value:
                value = shared_strings[int(value)]
            values[index] = value.strip() if isinstance(value, str) else str(value)
        rows.append(values)
    return rows


def row_to_article(
    record: dict[str, str],
    index: int,
    article_id_prefix: str = DEFAULT_ARTICLE_ID_PREFIX,
) -> ArticleRecord:
    title = record["title"].strip()
    gold_score = integrated_gold_score(record["direction"], record["confidence"])
    month = excel_serial_date_to_month(record["month"].strip())

    return ArticleRecord(
        article_id=f"{article_id_prefix}-{index:03d}",
        text=record["text"].strip(),
        title=title,
        url=record["url"].strip(),
        source=record["source"].strip(),
        date=month,
        gold_score=gold_score,
        gold_reasoning=record["notes"].strip(),
    )


def convert_xlsx_to_jsonl(
    input_path: Path,
    output_path: Path,
    article_id_prefix: str = DEFAULT_ARTICLE_ID_PREFIX,
) -> int:
    rows = read_first_sheet_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    headers = [header.strip() for header in rows[0]]
    required_headers = {
        "month",
        "source",
        "title",
        "url",
        "direction",
        "confidence",
        "notes",
        "text",
    }
    missing_headers = required_headers - set(headers)
    if missing_headers:
        raise ValueError(f"Missing required columns: {sorted(missing_headers)}")

    articles: list[ArticleRecord] = []
    for row in rows[1:]:
        padded_row = row + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded_row, strict=False))
        if not any(value.strip() for value in record.values()):
            continue
        articles.append(row_to_article(record, len(articles) + 1, article_id_prefix))

    write_jsonl(output_path, [article.model_dump(mode="json") for article in articles])
    return len(articles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the human gold-standard GEPA Excel worksheet to JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--article-id-prefix",
        default=DEFAULT_ARTICLE_ID_PREFIX,
        help="Prefix for generated article IDs before the numeric suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_xlsx_to_jsonl(args.input, args.output, args.article_id_prefix)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
