from fastapi import FastAPI
import uvicorn

from src.config.settings import get_settings

app = FastAPI(title="Bitcoin Trend Prediction API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    settings = get_settings()
    return {
        "app_env": settings.app_env,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
    }


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "apps.inference_api.main:app",
        host=settings.host,
        port=settings.fastapi_port,
        reload=settings.app_env == "development",
    )


if __name__ == "__main__":
    main()
