from src.data.schemas import ArticleRecord

BASE_EXTRACTION_PROMPT = """Extract the author's Bitcoin price prediction from the article.

Return JSON with:
- direction: one of up, down, neutral
- confidence: one of low, medium, high
- reasoning: one short sentence explaining the predicted viewpoint

Article source: {source}
Article date: {date}
Article title:
{title}
Article text:
{text}
"""


def render_extraction_prompt(article: ArticleRecord, template: str = BASE_EXTRACTION_PROMPT) -> str:
    return template.format(
        source=article.source,
        date=article.date,
        title=article.title,
        text=article.text,
    )
