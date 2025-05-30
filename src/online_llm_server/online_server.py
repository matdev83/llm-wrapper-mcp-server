import argparse
import os
import sys
import logging
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper
from llm_wrapper_mcp_server.llm_client import LLMClient # Import LLMClient for default model configuration

def main() -> None:
    """Run the specialized online LLM MCP server."""
    sys.stderr.flush()

    # --- Handle --cwd argument and change directory ---
    cwd_parser = argparse.ArgumentParser(add_help=False)
    cwd_parser.add_argument("--cwd", help="Change working directory to this path")
    cwd_args, _ = cwd_parser.parse_known_args()

    if cwd_args.cwd:
        try:
            os.chdir(cwd_args.cwd)
            sys.stderr.flush()
        except Exception as e:
            print(f"Error changing working directory to {cwd_args.cwd}: {e}", file=sys.stderr)
            sys.stderr.flush()

    parser = argparse.ArgumentParser(description="Specialized MCP server for online LLM queries")
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
        default="logs/online_server.log",
        help="Path to log file (default: logs/online_server.log)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)."
    )
    parser.add_argument(
        "--llm-api-base-url",
        default=os.getenv("LLM_API_BASE_URL"),
        help="LLM API base URL (default: from LLM_API_BASE_URL environment variable)"
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

    args = parser.parse_args(_)

    log_dir = os.path.dirname(args.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        filename=args.log_file,
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging is configured for online_server.py.")

    # Define the specialized tool for online search
    specialized_tools = {
        "search_online": {
            "description": "Ask a natural language question for an online search assistant to retrieve information or perform fact checking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": f"The natural language question to ask. Maximum length is {args.limit_user_prompt_length} tokens."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name to use for this request. If not specified, uses the default model."
                    }
                },
                "required": ["prompt"]
            }
        }
    }

    server = LLMMCPWrapper(
        system_prompt_path=args.system_prompt_file,
        model=args.model,
        llm_api_base_url=args.llm_api_base_url,
        max_user_prompt_tokens=args.limit_user_prompt_length,
        skip_accounting=args.skip_accounting,
        max_tokens=args.max_tokens,
        server_name="online-llm-mcp-server",
        server_description="Specialized MCP server for online LLM queries",
        initial_tools=specialized_tools
    )
    server.run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.basicConfig(
            filename="logs/online_server_unhandled_error.log",
            level="ERROR",
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="a"
        )
        logger = logging.getLogger(__name__)
        logger.exception("Unhandled exception in online LLM MCP server")
