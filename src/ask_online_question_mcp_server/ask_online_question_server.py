import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from llm_wrapper_mcp_server.llm_client_parts._llm_client_core import LLMClient # Updated import
from llm_wrapper_mcp_server.logger import get_logger

logger = get_logger(__name__)

class AskOnlineQuestionServer:
    """
    MCP server for asking online questions using a fixed LLM model.
    """

    def __init__(
        self,
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        system_prompt_path: str = "config/prompts/system.txt",
        llm_api_base_url: Optional[str] = None,
        server_name: str = "Ask Online Question",
        server_description: str = "MCP server for asking online questions using a fixed LLM model.",
        enable_logging: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_log: bool = True
    ) -> None:
        """
        Initialize the server with configuration options.
        The LLM model is fixed and passed during initialization.
        """
        self.llm_client = LLMClient(
            system_prompt_path=system_prompt_path,
            model=model,
            api_base_url=llm_api_base_url,
            enable_logging=enable_logging,
            enable_rate_limiting=enable_rate_limiting,
            enable_audit_log=enable_audit_log
        )
        self.server_name = server_name
        self.server_description = server_description
        self.tools = {
            "ask_online_question": {
                "description": "Asks an online question using the configured LLM.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The question to ask the online LLM."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
        logger.debug(f"{self.server_name} server initialized with model: {model}")

    def send_response(self, response: Dict[str, Any]) -> None:
        """Send a JSON-RPC response to stdout."""
        try:
            response_str = json.dumps(response) + "\n"
            sys.stdout.write(response_str)
            sys.stdout.flush()
        except Exception as e:
            logger.error("Error sending response to stdout: %s", str(e))
            raise

    def handle_request(self, request: Dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            logger.debug("Handling initialize request.")
            self.send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": self.server_name,
                        "version": "0.1.0",
                        "description": self.server_description
                    },
                    "capabilities": {
                        "tools": self.tools,
                        "resources": {},
                        "prompts": {},
                        "sampling": {}
                    }
                }
            })
        elif method == "tools/list":
            logger.debug("Handling tools/list request.")
            self.send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": self.tools
                }
            })
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})

            if name == "ask_online_question":
                prompt = args.get("prompt")
                if not prompt:
                    logger.warning("Missing required 'prompt' argument for 'ask_online_question'.")
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params",
                            "data": "Missing required 'prompt' argument"
                        }
                    })
                    return

                try:
                    response = self.llm_client.generate_response(prompt=prompt)
                    mcp_response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{
                                "type": "text",
                                "text": response["response"]
                            }],
                            "isError": False
                        }
                    }
                    self.send_response(mcp_response)
                except Exception as e:
                    logger.error(f"Error during 'ask_online_question' execution: {e}")
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32000, # Or MCP_ERROR_INTERNAL if constants were added
                            "message": "Internal error",
                            "data": "Internal server error. Check server logs for details."
                        },
                        "isError": True
                    })
            else:
                logger.warning(f"Tool not found: {name}")
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found",
                        "data": f"Tool '{name}' not found"
                    }
                })
        else:
            logger.warning(f"Method not found: {method}")
            self.send_response({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found",
                    "data": f"Method '{method}' not found"
                }
            })

    def run(self) -> None:
        """Run the server, reading from stdin and writing to stdout."""
        logger.debug("AskOnlineQuestionServer run method started. Sending initial capabilities.")
        self.send_response({
            "jsonrpc": "2.0",
            "id": None,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.server_name,
                    "version": "0.1.0",
                    "description": self.server_description
                },
                "capabilities": {
                    "tools": self.tools,
                    "resources": {},
                    "prompts": {},
                    "sampling": {}
                }
            }
        })
        logger.debug("Initial capabilities sent. Entering main request loop.")
        try: # Outer try for the finally block
            while True:
                line = sys.stdin.readline()
                if not line:
                    logger.info("Empty line or EOF received from stdin. Breaking loop.")
                    break
                try:
                    request = json.loads(line)
                    self.handle_request(request)
                except json.JSONDecodeError:
                    logger.error("Parse error: Invalid JSON received from stdin.")
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                            "data": "Invalid JSON"
                        }
                    })
                except Exception as e:
                    logger.error(f"Error in main request loop: {e}")
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32000, # Or MCP_ERROR_INTERNAL if constants were added
                            "message": "Internal error",
                            "data": "Internal server error. Check server logs for details."
                        }
                    })
        except Exception as e:
            logger.critical(f"Fatal error in AskOnlineQuestionServer run loop: {e}")
            raise
        finally:
            logger.debug("Ensuring LLMClient resources are closed.")
            if hasattr(self, 'llm_client') and self.llm_client:
                try:
                    self.llm_client.close()
                except Exception as e:
                    logger.warning(f"Error closing LLMClient: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Online Question MCP Server")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The LLM model to use for answering questions (e.g., 'perplexity/llama-3.1-sonar-small-128k-online')"
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
    args = parser.parse_args()

    server = AskOnlineQuestionServer(
        model=args.model,
        system_prompt_path=args.system_prompt_path,
        llm_api_base_url=args.llm_api_base_url
    )
    server.run()
