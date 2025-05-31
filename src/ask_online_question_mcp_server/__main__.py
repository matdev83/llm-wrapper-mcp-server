from .ask_online_question_server import AskOnlineQuestionServer
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Ask Online Question MCP Server")
    parser.add_argument(
        "--model",
        type=str,
        default="perplexity/llama-3.1-sonar-small-128k-online",
        help="The LLM model to use for answering questions (defaults to 'perplexity/llama-3.1-sonar-small-128k-online' if not specified)"
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default="config/prompts/system.txt",
        help="Path to the system prompt file."
    )
    parser.add_argument(
        "--llm-api-base-url",
        type=str,
        default=None,
        help="Base URL for the LLM API (e.g., 'https://api.openrouter.ai/api/v1')."
    )
    parser.add_argument(
        "--disable-logging",
        action='store_false',
        dest='enable_logging',
        help="Disable LLM usage logging (会计)."
    )
    parser.add_argument(
        "--disable-rate-limiting",
        action='store_false',
        dest='enable_rate_limiting',
        help="Disable rate limiting (currently a placeholder)."
    )
    parser.add_argument(
        "--disable-audit-log",
        action='store_false',
        dest='enable_audit_log',
        help="Disable audit logging of prompts and replies."
    )
    args = parser.parse_args()

    server = AskOnlineQuestionServer(
        model=args.model,
        system_prompt_path=args.system_prompt_path,
        llm_api_base_url=args.llm_api_base_url,
        enable_logging=args.enable_logging,
        enable_rate_limiting=args.enable_rate_limiting,
        enable_audit_log=args.enable_audit_log
    )
    server.run()

if __name__ == "__main__":
    main()
