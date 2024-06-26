"""Development entrypoint."""

import logging

from rich.logging import RichHandler


def create_debug_app():
    """Workaround to use different logger for debug."""
    import fastapi
    import starlette
    import uvicorn

    logging.basicConfig(
        format=None,
        handlers=[
            RichHandler(
                rich_tracebacks=True, tracebacks_suppress=[fastapi, starlette, uvicorn]
            )
        ],
    )
    logging.getLogger("til24_asr").setLevel(logging.DEBUG)

    from til24_asr import create_app

    return create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(format=None, handlers=[RichHandler(rich_tracebacks=True)])

    uvicorn.run(
        "dev:create_debug_app",
        host="localhost",
        port=5001,
        log_level=logging.INFO,
        reload=True,
        factory=True,
        log_config=None,
    )
