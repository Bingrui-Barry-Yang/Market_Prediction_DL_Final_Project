from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    example_path = project_root / "infra" / "env" / ".env.example"

    if not env_path.exists() and example_path.exists():
        env_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {env_path} from {example_path}")
    else:
        print(".env already exists or no example file found.")


if __name__ == "__main__":
    main()
