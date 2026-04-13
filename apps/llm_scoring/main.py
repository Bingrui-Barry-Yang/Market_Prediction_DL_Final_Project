from src.common.logging import configure_logging
from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    print("llm-scoring service scaffold is ready")
    print("Implement GEPA and Gemini batch scoring here.")


if __name__ == "__main__":
    main()
