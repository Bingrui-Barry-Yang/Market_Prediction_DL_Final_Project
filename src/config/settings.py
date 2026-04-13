from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    fastapi_port: int = Field(default=8000, alias="FASTAPI_PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")
    google_news_api_key: str = Field(default="replace-me", alias="GOOGLE_NEWS_API_KEY")
    newsapi_api_key: str = Field(default="replace-me", alias="NEWSAPI_API_KEY")
    gemini_api_key: str = Field(default="replace-me", alias="GEMINI_API_KEY")
    btc_price_api_url: str = Field(
        default="https://api.coingecko.com/api/v3",
        alias="BTC_PRICE_API_URL",
    )
    mlflow_tracking_uri: str = Field(
        default="file:/app/outputs/mlruns",
        alias="MLFLOW_TRACKING_URI",
    )
    default_trust_value: float = Field(default=0.5, alias="DEFAULT_TRUST_VALUE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
