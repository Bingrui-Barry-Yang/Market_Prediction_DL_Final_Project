from src.config.settings import get_settings
from src.data.datasets import load_articles


def main() -> None:
    settings = get_settings()
    articles = load_articles(settings.train_articles_path)
    print(f"Loaded {len(articles)} training articles from {settings.train_articles_path}")
    print("GEPA optimization skeleton is ready for live model adapters.")


if __name__ == "__main__":
    main()
