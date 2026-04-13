from src.common.logging import configure_logging
from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    print("scraper-filter service scaffold is ready")
    print("Implement Stage 1 and Stage 2 job orchestration here.")


if __name__ == "__main__":
    main()
