from src.common.logging import configure_logging
from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    print("training service scaffold is ready")
    print("Implement Stage 4 to Stage 7 jobs here.")


if __name__ == "__main__":
    main()
