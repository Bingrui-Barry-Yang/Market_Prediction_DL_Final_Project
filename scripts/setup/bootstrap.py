from pathlib import Path

PROJECT_DIRS = [
    "data/train",
    "data/test",
    "data/validation",
    "data/external",
    "outputs/gepa_runs",
    "outputs/evaluations",
    "outputs/validation",
    "outputs/reports",
]


def touch_gitkeep(project_root: Path, relative_dir: str) -> None:
    directory = project_root / relative_dir
    directory.mkdir(parents=True, exist_ok=True)
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
