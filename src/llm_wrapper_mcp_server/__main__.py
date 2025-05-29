"""
Main entry point for the MCP server.
"""
import argparse
import os
import sys
import logging
import warnings

def main() -> None:
    """Run the MCP server."""
    sys.stderr.flush()

    # --- Handle --cwd argument and change directory ---
    # Use parse_known_args to get --cwd before full parsing
    cwd_parser = argparse.ArgumentParser(add_help=False)
    cwd_parser.add_argument("--cwd", help="Change working directory to this path")
    cwd_args, _ = cwd_parser.parse_known_args()

    if cwd_args.cwd:
        try:
            os.chdir(cwd_args.cwd)
            #print(f"Changed working directory to: {os.getcwd()}", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"Error changing working directory to {cwd_args.cwd}: {e}", file=sys.stderr)
            sys.stderr.flush()
            # Continue execution, the trace log will show the final cwd

    # Full parser for all arguments
    parser = argparse.ArgumentParser(description="Generic LLM API MCP server")
    parser.add_argument(
        "--server-name",
        default="llm-wrapper-mcp-server",
        help="Name of the MCP server (default: llm-wrapper-mcp-server)"
    )
    parser.add_argument(
        "--server-description",
        default="Generic LLM API MCP server",
        help="Description of the MCP server (default: Generic LLM API MCP server)"
    )
    parser.add_argument(
        "--system-prompt-file",
        default="config/prompts/system.txt",
        help="Path to system prompt file (default: config/prompts/system.txt)"
    )
    parser.add_argument(
        "--model",
        default="perplexity/llama-3.1-sonar-small-128k-online",
        help="Model to use for completions (default: perplexity/llama-3.1-sonar-small-128k-online)"
    )
    parser.add_argument(
        "--log-file",
        default="logs/server.log",
        help="Path to log file (default: log/server.log)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], # Removed TRACE for simplicity
        help="Logging level (default: INFO)."
    )
    parser.add_argument(
        "--llm-api-base-url",
        default=os.getenv("LLM_API_BASE_URL"),
        help="LLM API base URL (default: from LLM_API_BASE_URL environment variable)"
    )
    parser.add_argument(
        "--allowed-models-file",
        help="Path to file with allowed model names (one per line). If specified, --model must be in this list."
    )
    parser.add_argument(
        "--limit-user-prompt-length",
        type=int,
        default=100,
        help="Maximum allowed tokens in user prompt (default: 100)"
    )
    parser.add_argument(
        "--skip-accounting",
        action="store_true",
        help="Skip accounting for LLM usage"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to generate in the LLM response"
    )

    # Parse all args again, passing the remaining args from the first parse
    args = parser.parse_args(_)

    # The os.chdir already happened based on cwd_args.cwd
    # We can potentially use args.cwd here if needed, but os.getcwd() is the source of truth

    # Ensure log directory exists (this will now use the new cwd if changed)
    log_dir = os.path.dirname(args.log_file) # args.log_file is "logs/debug.log"
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Convert log level and configure logging
    log_level = getattr(logging, args.log_level.upper()) # Use upper() for robustness

    logging.basicConfig(
        filename=args.log_file,
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured and this is a DEBUG test message.")

    from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

    # Validate allowed models if specified
    if args.allowed_models_file:
        if not os.path.exists(args.allowed_models_file):
            logger.warning(f"Allowed models file not found: {args.allowed_models_file}")
            sys.exit(1)

        with open(args.allowed_models_file, 'r') as f:
            allowed_models = [line.strip() for line in f if line.strip()]

        if not allowed_models:
            logger.warning("Allowed models file is empty - must contain at least one model name")
            sys.exit(1)

        if args.model not in allowed_models:
            logger.warning(f"Model '{args.model}' is not in the allowed models list")
            sys.exit(1)

    server = LLMMCPWrapper(
        system_prompt_path=args.system_prompt_file,
        model=args.model,
        llm_api_base_url=args.llm_api_base_url,
        skip_accounting=args.skip_accounting,
        max_tokens=args.max_tokens,
        server_name=args.server_name,
        server_description=args.server_description
    )
    server.run()

# Define logger outside main for global access in exception handler
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Simplified error logging for unhandled exceptions
        logging.basicConfig(
            filename="logs/server_unhandled_error.log", # Log unhandled errors to a separate file
            level="ERROR",
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="a"
        )
        logger.exception("Unhandled exception in MCP server")
        # No sys.exit(1) here, let the exception propagate if needed
        raise
