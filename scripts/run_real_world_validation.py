from src.config.settings import get_settings
from src.data.datasets import load_articles


def main() -> None:
    settings = get_settings()
    articles = load_articles(settings.validation_articles_path)
    print(f"Loaded {len(articles)} validation articles from {settings.validation_articles_path}")
    print("Optional real-world validation skeleton is ready.")


if __name__ == "__main__":
    main()
