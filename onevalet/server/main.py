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
    subparsers = parser.add_subparsers(dest="command")

    # Default: run server
    parser.add_argument("--ui", action="store_true", help="Serve demo frontend (/ and /settings)")
    parser.add_argument("--host", default=os.getenv("ONEVALET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ONEVALET_PORT", "8000")))

    # Subcommand: copilot-auth
    auth_parser = subparsers.add_parser(
        "copilot-auth",
        help="Authenticate with GitHub Copilot via device flow",
    )
    auth_parser.add_argument(
        "--save-to-env",
        metavar="FILE",
        help="Append tokens to a .env file",
    )

    args = parser.parse_args()

    if args.command == "copilot-auth":
        _run_copilot_auth(args)
        return

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


def _run_copilot_auth(args):
    """Run the GitHub Copilot device flow authentication."""
    import asyncio
    from onevalet.llm.copilot_auth import device_flow_authenticate

    logging.basicConfig(level=logging.WARNING)

    try:
        token_data = asyncio.run(device_flow_authenticate())
    except RuntimeError as e:
        print(f"\n  ✗ Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    access_token = token_data["access_token"]
    refresh_token = token_data.get("refresh_token", "")

    print("\n  Add these to your environment or config.yaml:\n")
    print(f"    GITHUB_TOKEN={access_token}")
    if refresh_token:
        print(f"    GITHUB_REFRESH_TOKEN={refresh_token}")

    print("\n  config.yaml example:\n")
    print("    llm:")
    print("      provider: copilot")
    print("      model: claude-sonnet-4.6")
    print(f"      github_token: {access_token}")
    if refresh_token:
        print(f"      github_refresh_token: {refresh_token}")

    if args.save_to_env:
        with open(args.save_to_env, "a", encoding="utf-8") as f:
            f.write(f"\nGITHUB_TOKEN={access_token}\n")
            if refresh_token:
                f.write(f"GITHUB_REFRESH_TOKEN={refresh_token}\n")
        print(f"\n  ✓ Tokens appended to {args.save_to_env}")
