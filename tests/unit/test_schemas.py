from src.data.schemas import ArticleRecord, Direction


def test_article_schema_accepts_curated_record() -> None:
    article = ArticleRecord(
        article_id="article-001",
        text="Bitcoin could move higher.",
        url="https://example.com/article",
        source="Example News",
        date="2024-03",
        gold_direction="up",
        gold_confidence="high",
        gold_reasoning="The author expects demand to rise.",
    )

    assert article.gold_direction == Direction.up
