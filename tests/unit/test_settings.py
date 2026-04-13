from src.config.settings import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.fastapi_port == 8000
    assert settings.default_trust_value == 0.5
