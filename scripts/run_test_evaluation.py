from src.config.settings import get_settings
from src.data.datasets import load_articles


def main() -> None:
    settings = get_settings()
    articles = load_articles(settings.test_articles_path)
    print(f"Loaded {len(articles)} test articles from {settings.test_articles_path}")
    print("Test evaluation skeleton is ready for saved model predictions.")


if __name__ == "__main__":
    main()
