"""
STDIO-based MCP server implementation.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests.exceptions # Import requests.exceptions

from .logger import get_logger

logger = get_logger(__name__)

from llm_wrapper_mcp_server.llm_client import LLMClient


class LLMMCPWrapper:
    """LLM MCP Wrapper server implementation."""

    def __init__(
        self,
        system_prompt_path: str = "config/prompts/system.txt",
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        llm_api_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_user_prompt_tokens: int = 100,
        skip_outbound_key_checks: bool = False, # This is now passed to LLMClient as skip_redaction
        max_tokens: Optional[int] = None,
        server_name: str = "llm-wrapper-mcp-server",
        server_description: str = "Generic LLM API MCP server",
        initial_tools: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_log: bool = True
    ) -> None:
        """Initialize the server with configuration options."""
        logger.debug("StdioServer initialized")
        self.enable_logging = enable_logging
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_log = enable_audit_log
        self.llm_client = LLMClient(
            system_prompt_path=system_prompt_path,
            model=model,
            api_base_url=llm_api_base_url,
            api_key=llm_api_key,
            enable_logging=self.enable_logging,
            enable_rate_limiting=self.enable_rate_limiting,
            enable_audit_log=self.enable_audit_log,
            skip_outbound_key_checks=skip_outbound_key_checks # Pass this to LLMClient
        )
        self.system_prompt_path = system_prompt_path
        self.max_user_prompt_tokens = max_user_prompt_tokens
        self.skip_outbound_key_checks = skip_outbound_key_checks # Keep for internal use if needed
        self.max_tokens = max_tokens
        self.server_name = server_name
        self.server_description = server_description
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") # Still used for outbound key checks

        if initial_tools is None:
            self.tools = {
                "llm_call": {
                    "description": "Make a generic call to the configured LLM with a given prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The natural language prompt for the LLM. Maximum length is {max_tokens} tokens.".format(max_tokens=self.max_user_prompt_tokens)
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
        else:
            self.tools = initial_tools

    def send_response(self, response: Dict[str, Any]) -> None:
        """Send a JSON-RPC response to stdout."""
        try:
            response_str = json.dumps(response) + "\n"
            request_id = response.get('id', 'N/A')
            sys.stdout.write(response_str)
            sys.stdout.flush()
        except Exception as e:
            logger.error("Error sending response to stdout: %s", str(e), extra={'request_id': response.get('id', 'N/A')})
            raise

    def handle_request(self, request: Dict[str, Any]) -> None:
        """Handle an incoming JSON-RPC request."""
        try:
            method = request.get("method")
            request_id = request.get("id")

            if method == "initialize":
                logger.debug("Handling initialize request.", extra={'request_id': request_id})
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
                logger.debug("initialize response sent.", extra={'request_id': request_id})
            elif method == "tools/list":
                logger.debug("Handling tools/list request.", extra={'request_id': request_id})
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": self.tools
                    }
                })
                logger.debug("tools/list response sent.", extra={'request_id': request_id})
            elif method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})

                if name not in self.tools:
                    logger.warning("Tool not found: %s", name, extra={'request_id': request_id})
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": "Method not found",
                            "data": f"Tool '{name}' not found"
                        }
                    })
                    return

                prompt = args.get("prompt")
                if not prompt:
                    logger.warning("Missing required 'prompt' argument for tool '%s'.", name, extra={'request_id': request_id})
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

                if not self.skip_outbound_key_checks and self.openrouter_api_key and self.openrouter_api_key in prompt:
                    logger.warning("API key leak detected in prompt", extra={'request_id': request_id})
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Security violation",
                            "data": "Prompt contains sensitive API key - request rejected"
                        }
                    })
                    return

                prompt_tokens = len(self.llm_client.encoder.encode(prompt))
                logger.debug("Prompt token count: %d/%d", prompt_tokens, self.max_user_prompt_tokens, extra={'request_id': request_id})
                if prompt_tokens > self.max_user_prompt_tokens:
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params",
                            "data": f"Prompt exceeds maximum length of {self.max_user_prompt_tokens} tokens"
                        }
                    })
                    return

                model_arg = args.get("model")
                model_to_use = None

                if model_arg is not None:
                    stripped_model = model_arg.strip()
                    if len(stripped_model) < 2:
                        self.send_response({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must be at least 2 characters"}
                        })
                        return
                    if '/' not in stripped_model:
                        self.send_response({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must contain a '/' separator"}
                        })
                        return

                    parts = stripped_model.split('/')
                    if len(parts) != 2 or not all(parts):
                        self.send_response({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must contain a provider and a model separated by a single '/'" }
                        })
                        return
                    model_to_use = stripped_model

                try:
                    client_to_use = self.llm_client
                    if model_to_use:
                        temp_client = LLMClient(
                            system_prompt_path=self.system_prompt_path,
                            model=model_to_use,
                            api_base_url=self.llm_client.base_url,
                            api_key=self.llm_client.api_key,
                            enable_logging=self.enable_logging,
                            enable_rate_limiting=self.enable_rate_limiting,
                            enable_audit_log=self.enable_audit_log,
                            skip_outbound_key_checks=self.skip_outbound_key_checks # Pass this to temp client
                        )
                        client_to_use = temp_client

                    response_data = client_to_use.generate_response(prompt=prompt, max_tokens=self.max_tokens)

                    mcp_response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [{"type": "text", "text": response_data["response"]}],
                            "isError": False
                        }
                    }
                    logger.debug("Sending MCP response: %s", mcp_response, extra={'request_id': request_id})
                    self.send_response(mcp_response)
                    logger.debug("send_response completed.", extra={'request_id': request_id})

                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    error_message = f"Internal error: {str(e)}"
                    if isinstance(e, requests.Timeout):
                        error_message = "LLM call timed out."
                    elif isinstance(e, requests.HTTPError):
                        error_message = f"LLM API HTTP error: {e.response.status_code} {e.response.reason}"
                    elif isinstance(e, requests.RequestException):
                        error_message = f"LLM API network error: {str(e)}"
                    elif isinstance(e, RuntimeError):
                        if "API rate limit exceeded" in str(e) or \
                           "Invalid API response format" in str(e) or \
                           "Unexpected API response format" in str(e):
                            error_message = str(e)

                    logger.error("Error during tool '%s' execution: %s\n%s", name, str(e), tb, extra={'request_id': request_id})
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32000, "message": error_message, "data": "Internal server error. Check server logs for details."},
                        "isError": True
                    })
            elif method == "resources/list":
                logger.debug("Handling resources/list request.", extra={'request_id': request_id})
                self.send_response({"jsonrpc": "2.0", "id": request_id, "result": {"resources": {}}})
                logger.debug("resources/list response sent.", extra={'request_id': request_id})
            elif method == "resources/templates/list":
                logger.debug("Handling resources/templates/list request.", extra={'request_id': request_id})
                self.send_response({"jsonrpc": "2.0", "id": request_id, "result": {"templates": {}}})
                logger.debug("resources/templates/list response sent.", extra={'request_id': request_id})
            else:
                logger.warning("Method not found: %s", method, extra={'request_id': request_id})
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found", "data": f"Method '{method}' not found"}
                })
        except Exception as e:
            logger.error("Error handling request: %s", str(e), extra={'request_id': request.get('id', 'N/A')})
            self.send_response({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32000, "message": "Internal error", "data": str(e)}
            })

    def run(self) -> None:
        """Run the server, reading from stdin and writing to stdout."""
        # Parse command line arguments for skip_outbound_key_checks
        skip_outbound_key_checks_cli = "--skip-outbound-key-leaks" in sys.argv
        if skip_outbound_key_checks_cli:
            logger.info("Outbound key leak checks disabled by command line parameter")
            self.skip_outbound_key_checks = True # Update instance attribute if CLI flag is present

        logger.debug("StdioServer run method started. Sending initial capabilities.")
        self.send_response({
            "jsonrpc": "2.0",
            "id": None, # Per MCP spec for server notifications
            "method": "mcp/serverReady", # Using method instead of result for notification
            "params": { # params instead of result
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
        try:
            loop_count = 0
            while True:
                loop_count += 1
                logger.debug(f"Request loop iter {loop_count}. PRE sys.stdin.readline()", extra={'request_id': 'N/A'})
                line = sys.stdin.readline()

                if not line:
                    logger.info("Empty line or EOF received from stdin. Breaking loop.")
                    break

                request = None # Initialize request to None
                try:
                    request = json.loads(line)
                    request_id = request.get("id", "N/A")
                    logger.debug("Parsed MCP request: %s", request, extra={'request_id': request_id})
                    self.handle_request(request)
                except json.JSONDecodeError:
                    logger.error("Parse error: Invalid JSON received from stdin.", extra={'request_id': 'N/A'}) # request_id might not be available
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": None, # id is null for parse error
                        "error": {"code": -32700, "message": "Parse error", "data": "Invalid JSON"}
                    })
                except Exception as e: # Catch-all for other errors in the loop
                    current_request_id = None
                    if request and isinstance(request, dict): # Check if request is not None and is a dict
                        current_request_id = request.get("id")
                    logger.error("Error in main request loop: %s", str(e), extra={'request_id': current_request_id if current_request_id else 'N/A'})
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": current_request_id, # Use ID from current request if available
                        "error": {"code": -32000, "message": "Internal error", "data": "Internal server error. Check server logs for details."}
                    })
        except Exception as e: # Fatal error in the server run loop itself
            logger.critical("Fatal error in StdioServer run loop: %s", str(e))
            # Attempt to send a final error response if possible, though stdout might be compromised
            try:
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Fatal server error", "data": str(e)}
                })
            except Exception as final_e:
                logger.critical("Failed to send final error message: %s", str(final_e))
            raise # Re-raise the original fatal error
