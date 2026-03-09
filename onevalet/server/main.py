"""CLI argument parsing and uvicorn entry point."""

import json
import logging
import os
import sys


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OneValet API Server")
    parser.add_argument("--ui", action="store_true", help="Serve demo frontend (/ and /settings)")
    parser.add_argument("--host", default=os.getenv("ONEVALET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ONEVALET_PORT", "8000")))
    args = parser.parse_args()

    if os.getenv("LOG_FORMAT", "json") == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")

    from .app import api
    if args.ui:
        from .ui import register_ui_routes
        register_ui_routes(api)

    uvicorn.run(api, host=args.host, port=args.port)
