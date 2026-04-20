from src.data.schemas import ParseStatus
from src.extraction.parser import parse_extraction_response


def test_parse_extraction_response_ok() -> None:
    extraction = parse_extraction_response(
        article_id="article-001",
        model_name="qwen",
        prompt_version="prompt-v001",
        raw_response='{"direction":"up","confidence":"high","reasoning":"Demand should rise."}',
    )

    assert extraction.parse_status == ParseStatus.ok
    assert extraction.pred_reasoning == "Demand should rise."


def test_parse_extraction_response_failed() -> None:
    extraction = parse_extraction_response(
        article_id="article-001",
        model_name="qwen",
        prompt_version="prompt-v001",
        raw_response="not json",
    )

    assert extraction.parse_status == ParseStatus.failed
