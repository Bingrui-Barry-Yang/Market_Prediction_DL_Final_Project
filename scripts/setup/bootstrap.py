from pathlib import Path

PROJECT_DIRS = [
    "data/authordemo",
    "data/qwk",
    "data/train",
    "data/test",
    "outputs/gepa_runs/bitcoin_sentiment",
    "outputs/qwk/best_vs_seed",
    "outputs/qwk/per_model",
    "outputs/test_author",
    "outputs/test_eval",
    "outputs/gepa_runs/reports",
]


def touch_gitkeep(project_root: Path, relative_dir: str) -> None:
    directory = project_root / relative_dir
    directory.mkdir(parents=True, exist_ok=True)
    if not any(directory.iterdir()):
        (directory / ".gitkeep").touch()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for relative_dir in PROJECT_DIRS:
        touch_gitkeep(project_root, relative_dir)

    env_path = project_root / ".env"
    example_path = project_root / ".env.example"
    if not env_path.exists() and example_path.exists():
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {env_path}")
    else:
        print("Environment file already exists; left it unchanged.")

    print("Project directories are ready.")


if __name__ == "__main__":
    main()
