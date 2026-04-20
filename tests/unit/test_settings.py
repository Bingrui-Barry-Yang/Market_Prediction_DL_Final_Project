from src.config.settings import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.default_prompt_version == "prompt-v001"
    assert settings.default_model_names == ["qwen", "kimi", "gpt-oss-120b"]
