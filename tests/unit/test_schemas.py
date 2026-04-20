from src.data.schemas import ArticleRecord


def test_article_schema_accepts_curated_record() -> None:
    article = ArticleRecord(
        article_id="article-001",
        text="",
        title="Bitcoin could move higher",
        url="https://example.com/article",
        source="Example News",
        date="2024-03",
        gold_score=15,
        gold_reasoning="The author expects demand to rise.",
    )

    assert article.gold_score == 15
