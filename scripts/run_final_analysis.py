from pathlib import Path

from src.analysis.reporting import write_summary
from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    report_path = Path(settings.outputs_dir) / "reports" / "summary.json"
    write_summary(
        report_path,
        {
            "project": "bitcoin-news-prompt-optimization",
            "status": "skeleton-ready",
            "stages": [
                "gepa_training",
                "test_evaluation",
                "optional_real_world_validation",
                "final_analysis",
            ],
        },
    )
    print(f"Wrote final analysis skeleton summary to {report_path}")


if __name__ == "__main__":
    main()
