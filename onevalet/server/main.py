"""CLI argument parsing and uvicorn entry point."""

import logging
import os


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OneValet API Server")
    parser.add_argument("--ui", action="store_true", help="Serve demo frontend (/ and /settings)")
    parser.add_argument("--host", default=os.getenv("ONEVALET_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ONEVALET_PORT", "8000")))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s - %(message)s",
    )

    from .app import api
    if args.ui:
        from .ui import register_ui_routes
        register_ui_routes(api)

    uvicorn.run(api, host=args.host, port=args.port)
