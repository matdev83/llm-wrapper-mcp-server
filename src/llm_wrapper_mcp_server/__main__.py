"""
Main entry point for the MCP server.
"""

import argparse
import os
import sys
import logging
from typing import Optional, List  # Added List and Any for type hints

# Global logger for this module, configured in main or _configure_logging
logger = logging.getLogger(__name__)


def _handle_cwd_arg() -> List[str]:
    """Handles --cwd argument and changes directory if specified. Returns remaining args."""
    # Use parse_known_args to get --cwd before full parsing
    cwd_parser = argparse.ArgumentParser(add_help=False)
    cwd_parser.add_argument("--cwd", help="Change working directory to this path")
    cwd_args, remaining_args = cwd_parser.parse_known_args()

    if cwd_args.cwd:
        try:
            os.chdir(cwd_args.cwd)
            sys.stderr.flush()  # Keep flush for immediate feedback if needed
        except Exception as e:
            # Use module-level logger if available, otherwise print
            # logger might not be configured yet if CWD handling is very early
            print(
                f"Error changing working directory to {cwd_args.cwd}: {e}",
                file=sys.stderr,
            )
            sys.stderr.flush()
            # Exiting might be safer if CWD is critical, but original code continued.
            # For now, matching original behavior.
    return remaining_args


def _setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up and returns the main argument parser."""
    parser = argparse.ArgumentParser(description="Generic LLM API MCP server")
    parser.add_argument(
        "--server-name",
        default="llm-wrapper-mcp-server",
        help="Name of the MCP server (default: llm-wrapper-mcp-server)",
    )
    parser.add_argument(
        "--server-description",
        default="Generic LLM API MCP server",
        help="Description of the MCP server (default: Generic LLM API MCP server)",
    )
    parser.add_argument(
        "--system-prompt-file",
        default="config/prompts/system.txt",
        help="Path to system prompt file (default: config/prompts/system.txt)",
    )
    parser.add_argument(
        "--model",
        default="perplexity/llama-3.1-sonar-small-128k-online",
        help="Model to use for completions (default: perplexity/llama-3.1-sonar-small-128k-online)",
    )
    parser.add_argument(
        "--log-file",
        default="logs/server.log",
        help="Path to log file (default: logs/server.log)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--llm-api-base-url",
        default=os.getenv("LLM_API_BASE_URL"),
        help="LLM API base URL (default: from LLM_API_BASE_URL environment variable)",
    )
    parser.add_argument(
        "--allowed-models-file",
        help="Path to file with allowed model names (one per line). If specified, --model must be in this list.",
    )
    parser.add_argument(
        "--limit-user-prompt-length",
        type=int,
        default=100,
        help="Maximum allowed tokens in user prompt (default: 100)",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_false",
        dest="enable_logging",
        help="Disable LLM usage logging (会计).",
    )
    parser.add_argument(
        "--disable-rate-limiting",
        action="store_false",
        dest="enable_rate_limiting",
        help="Disable rate limiting (currently a placeholder).",
    )
    parser.add_argument(
        "--disable-audit-log",
        action="store_false",
        dest="enable_audit_log",
        help="Disable audit logging of prompts and replies.",
    )
    parser.add_argument(
        "--skip-outbound-key-leaks",
        action="store_true",
        help="Skip checks for outbound API key leaks in prompts (default: False)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate in the LLM response",
    )
    return parser


def _configure_logging(log_file: str, log_level_str: str) -> None:
    """Configures logging for the application."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level_val = getattr(logging, log_level_str.upper(), logging.INFO)

    # Use basicConfig with force=True if re-configuration is possible,
    # or ensure it's only called once.
    # For a script entry point, direct basicConfig is usually fine.
    logging.basicConfig(
        filename=log_file,
        level=log_level_val,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )
    # Global logger for this module is already defined at the top.
    # Re-fetch or ensure it uses the new basicConfig settings.
    # logging.getLogger(__name__) would return the same logger instance.
    logger.debug("Logging configured.")


def _validate_allowed_models(
    args_model: str, allowed_models_file_path: Optional[str]
) -> None:
    """Validates the selected model against an allowed models file if provided."""
    if not allowed_models_file_path:
        return  # No validation needed if file not specified

    allowed_models: List[str] = []
    if os.path.exists(allowed_models_file_path):
        with open(allowed_models_file_path, "r") as f:
            allowed_models = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"Allowed models file not found: {allowed_models_file_path}")
        sys.exit(1)

    if not allowed_models:
        logger.warning(
            "Allowed models file is empty - must contain at least one model name"
        )
        sys.exit(1)

    if args_model not in allowed_models:
        logger.warning(
            f"Model '{args_model}' is not in the allowed models list from {allowed_models_file_path}"
        )
        sys.exit(1)
    logger.debug(
        "Model '%s' validated against allowed models file '%s'.",
        args_model,
        allowed_models_file_path,
    )


def main() -> None:
    """Run the MCP server."""
    sys.stderr.flush()  # Initial flush

    remaining_args = _handle_cwd_arg()

    parser = _setup_arg_parser()
    args = parser.parse_args(remaining_args)  # Parse the remaining args

    _configure_logging(args.log_file, args.log_level)

    # Import here, after logging is configured, so LLMMCPWrapper can use the configured logger
    from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

    _validate_allowed_models(args.model, args.allowed_models_file)

    server = LLMMCPWrapper(
        system_prompt_path=args.system_prompt_file,
        model=args.model,
        llm_api_base_url=args.llm_api_base_url,
        enable_logging=args.enable_logging,
        enable_rate_limiting=args.enable_rate_limiting,
        enable_audit_log=args.enable_audit_log,
        skip_outbound_key_checks=args.skip_outbound_key_leaks,
        max_tokens=args.max_tokens,
        server_name=args.server_name,
        server_description=args.server_description,
        # Note: max_user_prompt_tokens is available on args if needed by LLMMCPWrapper constructor
        # For now, assuming LLMMCPWrapper uses its default or it's not passed from here.
        # If it should be passed: max_user_prompt_tokens=args.limit_user_prompt_length
    )
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Simplified error logging for unhandled exceptions from main()
        # Ensure logger is configured or fallback to print
        # If _configure_logging failed, this might also fail or log to wrong place.
        # A more robust approach might involve a try/except around _configure_logging
        # and a fallback basicConfig here if it failed.
        # For now, assume logger is available or basicConfig below is sufficient.
        logging.basicConfig(  # Fallback basicConfig
            filename="logs/server_unhandled_error.log",
            level="ERROR",
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="a",
            force=True,  # force=True to override previous config if any
        )
        logger.exception("Unhandled exception in MCP server main execution")
        raise
