from __future__ import annotations

import uvicorn

from utils.config import ensure_dirs, load_settings
from utils.logging import configure_logging


def run_api_main() -> None:
    settings = load_settings()
    ensure_dirs(settings)
    configure_logging()
    uvicorn.run(
        "api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=bool(settings.api.reload),
        log_level="info",
    )


if __name__ == "__main__":
    run_api_main()
