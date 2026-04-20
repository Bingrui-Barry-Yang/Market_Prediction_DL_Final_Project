from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    train_articles_path: Path = Field(
        default=Path("data/train/articles.jsonl"),
        alias="TRAIN_ARTICLES_PATH",
    )
    test_articles_path: Path = Field(
        default=Path("data/test/articles.jsonl"),
        alias="TEST_ARTICLES_PATH",
    )
    validation_articles_path: Path = Field(
        default=Path("data/validation/articles.jsonl"),
        alias="VALIDATION_ARTICLES_PATH",
    )
    outputs_dir: Path = Field(default=Path("outputs"), alias="OUTPUTS_DIR")
    default_prompt_version: str = Field(default="prompt-v001", alias="DEFAULT_PROMPT_VERSION")
    default_model_names: list[str] = Field(
        default_factory=lambda: ["qwen", "kimi", "gpt-oss-120b"],
        alias="DEFAULT_MODEL_NAMES",
    )

    qwen_api_base_url: str = Field(default="", alias="QWEN_API_BASE_URL")
    qwen_api_key: str = Field(default="", alias="QWEN_API_KEY")
    kimi_api_base_url: str = Field(default="", alias="KIMI_API_BASE_URL")
    kimi_api_key: str = Field(default="", alias="KIMI_API_KEY")
    gpt_oss_api_base_url: str = Field(default="", alias="GPT_OSS_API_BASE_URL")
    gpt_oss_api_key: str = Field(default="", alias="GPT_OSS_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("default_model_names", mode="before")
    @classmethod
    def split_model_names(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
