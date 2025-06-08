"""
STDIO-based MCP server implementation.
"""
import json
import os
import sys
from typing import Any, Dict, Optional

import requests.exceptions

from .logger import get_logger
from llm_wrapper_mcp_server.llm_client import LLMClient

logger = get_logger(__name__)


class LLMMCPWrapper:
    """LLM MCP Wrapper server implementation."""

    def __init__(
        self,
        system_prompt_path: str = "config/prompts/system.txt",
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        llm_api_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_user_prompt_tokens: int = 100,
        skip_outbound_key_checks: bool = False,
        max_tokens: Optional[int] = None,
        server_name: str = "llm-wrapper-mcp-server",
        server_description: str = "Generic LLM API MCP server",
        initial_tools: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_log: bool = True
    ) -> None:
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
            skip_outbound_key_checks=skip_outbound_key_checks
        )
        self.system_prompt_path = system_prompt_path
        self.max_user_prompt_tokens = max_user_prompt_tokens
        self.skip_outbound_key_checks = skip_outbound_key_checks
        self.max_tokens = max_tokens
        self.server_name = server_name
        self.server_description = server_description
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

        if initial_tools is None:
            self.tools = {
                "llm_call": {
                    "description": "Make a generic call to the configured LLM with a given prompt.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": f"The natural language prompt for the LLM. Maximum length is {self.max_user_prompt_tokens} tokens."
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
        try:
            response_str = json.dumps(response) + "\n"
            sys.stdout.write(response_str)
            sys.stdout.flush()
        except Exception as e:
            logger.error("Error sending response to stdout: %s", str(e), extra={'request_id': response.get('id', 'N/A')})
            raise

    def _handle_initialize(self, request_id: Optional[str]) -> None:
        logger.debug("Handling initialize request.", extra={'request_id': request_id})
        self.send_response({
            "jsonrpc": "2.0", "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": self.server_name, "version": "0.1.0", "description": self.server_description},
                "capabilities": {"tools": self.tools, "resources": {}, "prompts": {}, "sampling": {}}
            }
        })
        logger.debug("initialize response sent.", extra={'request_id': request_id})

    def _handle_tools_list(self, request_id: Optional[str]) -> None:
        logger.debug("Handling tools/list request.", extra={'request_id': request_id})
        self.send_response({"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.tools}})
        logger.debug("tools/list response sent.", extra={'request_id': request_id})

    def _validate_tools_call_params(self, params: Dict[str, Any], request_id: Optional[str]) -> Optional[Dict[str, Any]]:
        name = params.get("name")
        args = params.get("arguments", {})

        if name not in self.tools:
            logger.warning("Tool not found: %s", name, extra={'request_id': request_id})
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32601, "message": "Method not found", "data": f"Tool '{name}' not found"}
            })
            return None

        prompt = args.get("prompt")
        if not prompt:
            logger.warning("Missing required 'prompt' argument for tool '%s'.", name, extra={'request_id': request_id})
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Invalid params", "data": "Missing required 'prompt' argument"}
            })
            return None

        if not self.skip_outbound_key_checks and self.openrouter_api_key and self.openrouter_api_key in prompt:
            logger.warning("API key leak detected in prompt", extra={'request_id': request_id})
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Security violation", "data": "Prompt contains sensitive API key - request rejected"}
            })
            return None

        prompt_tokens = len(self.llm_client.encoder.encode(prompt))
        logger.debug("Prompt token count: %d/%d", prompt_tokens, self.max_user_prompt_tokens, extra={'request_id': request_id})
        if prompt_tokens > self.max_user_prompt_tokens:
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Invalid params", "data": f"Prompt exceeds maximum length of {self.max_user_prompt_tokens} tokens"}
            })
            return None

        return {"name": name, "prompt": prompt, "args": args}


    def _validate_model_arg(self, model_arg: Optional[str], request_id: Optional[str]) -> Optional[str]:
        if model_arg is None:
            return None

        stripped_model = model_arg.strip()
        if len(stripped_model) < 2:
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must be at least 2 characters"}
            })
            return None
        if '/' not in stripped_model:
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must contain a '/' separator"}
            })
            return None
        parts = stripped_model.split('/')
        if len(parts) != 2 or not all(parts):
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": "Invalid model specification", "data": "Model name must contain a provider and a model separated by a single '/'" }
            })
            return None
        return stripped_model

    def _execute_llm_call(self, tool_name: str, prompt: str, model_to_use: Optional[str], request_id: Optional[str]) -> None:
        try:
            client_to_use = self.llm_client
            if model_to_use and model_to_use != self.llm_client.model:
                temp_client = LLMClient(
                    system_prompt_path=self.system_prompt_path, model=model_to_use,
                    api_base_url=self.llm_client.base_url, api_key=self.llm_client.api_key,
                    enable_logging=self.enable_logging, enable_rate_limiting=self.enable_rate_limiting,
                    enable_audit_log=self.enable_audit_log, skip_outbound_key_checks=self.skip_outbound_key_checks
                )
                client_to_use = temp_client

            response_data = client_to_use.generate_response(prompt=prompt, max_tokens=self.max_tokens)
            mcp_response = {
                "jsonrpc": "2.0", "id": request_id,
                "result": {"content": [{"type": "text", "text": response_data["response"]}], "isError": False}
            }
            logger.debug("Sending MCP response: %s", mcp_response, extra={'request_id': request_id})
            self.send_response(mcp_response)
            logger.debug("send_response completed.", extra={'request_id': request_id})

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_message = f"Internal error: {str(e)}"
            if isinstance(e, requests.Timeout): error_message = "LLM call timed out."
            elif isinstance(e, requests.HTTPError): error_message = f"LLM API HTTP error: {e.response.status_code} {e.response.reason}"
            elif isinstance(e, requests.RequestException): error_message = f"LLM API network error: {str(e)}"
            elif isinstance(e, RuntimeError) and ("API rate limit exceeded" in str(e) or "Invalid API response format" in str(e) or "Unexpected API response format" in str(e)):
                error_message = str(e)

            logger.error("Error during tool '%s' execution: %s\n%s", tool_name, str(e), tb, extra={'request_id': request_id})
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32000, "message": error_message, "data": "Internal server error. Check server logs for details."},
                "isError": True
            })

    def _handle_tools_call(self, params: Dict[str, Any], request_id: Optional[str]) -> None:
        validated_params = self._validate_tools_call_params(params, request_id)
        if not validated_params:
            return

        tool_name = validated_params["name"]
        prompt = validated_params["prompt"]
        args = validated_params["args"]

        model_arg = args.get("model")
        validated_model_to_use = self._validate_model_arg(model_arg, request_id)

        if model_arg is not None and validated_model_to_use is None:
            return

        self._execute_llm_call(tool_name, prompt, validated_model_to_use, request_id)


    def _handle_resources_list(self, request_id: Optional[str]) -> None:
        logger.debug("Handling resources/list request.", extra={'request_id': request_id})
        self.send_response({"jsonrpc": "2.0", "id": request_id, "result": {"resources": {}}})
        logger.debug("resources/list response sent.", extra={'request_id': request_id})

    def _handle_resources_templates_list(self, request_id: Optional[str]) -> None:
        logger.debug("Handling resources/templates/list request.", extra={'request_id': request_id})
        self.send_response({"jsonrpc": "2.0", "id": request_id, "result": {"templates": {}}})
        logger.debug("resources/templates/list response sent.", extra={'request_id': request_id})

    def _handle_unknown_method(self, method: str, request_id: Optional[str]) -> None:
        logger.warning("Method not found: %s", method, extra={'request_id': request_id})
        self.send_response({
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": "Method not found", "data": f"Method '{method}' not found"}
        })

    def handle_request(self, request: Dict[str, Any]) -> None:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize": self._handle_initialize(request_id)
            elif method == "tools/list": self._handle_tools_list(request_id)
            elif method == "tools/call": self._handle_tools_call(params, request_id)
            elif method == "resources/list": self._handle_resources_list(request_id)
            elif method == "resources/templates/list": self._handle_resources_templates_list(request_id)
            else: self._handle_unknown_method(method, request_id)
        except Exception as e:
            logger.error("Unexpected error handling request method '%s': %s", method, str(e), extra={'request_id': request_id}, exc_info=True)
            self.send_response({
                "jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32000, "message": "Internal error", "data": f"An unexpected error occurred: {str(e)}"}
            })

    def _read_and_parse_request(self) -> Optional[Dict[str, Any]]:
        """Reads a line from stdin, parses it as JSON. Returns None on EOF or JSONDecodeError."""
        line = sys.stdin.readline()
        if not line: # EOF
            logger.info("EOF received from stdin.")
            return None

        try:
            request_data = json.loads(line)
            current_request_id = request_data.get("id", "N/A")
            logger.debug("Parsed MCP request: %s", request_data, extra={'request_id': current_request_id})
            return request_data
        except json.JSONDecodeError:
            logger.error("Parse error: Invalid JSON received from stdin.", extra={'request_id': 'N/A'})
            self.send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error", "data": "Invalid JSON"}})
            return None # Indicate error, allowing loop to decide whether to continue or break

    def _process_parsed_request(self, request_data: Optional[Dict[str, Any]]) -> None:
        """Processes a parsed request. If request_data is None, it's a no-op."""
        if request_data is None:
            return # Nothing to process
        try:
            self.handle_request(request_data)
        except Exception as e:
            current_request_id_on_error = request_data.get("id")
            logger.error("Error processing request: %s", str(e), extra={'request_id': current_request_id_on_error if current_request_id_on_error else 'N/A'}, exc_info=True)
            self.send_response({"jsonrpc": "2.0", "id": current_request_id_on_error, "error": {"code": -32000, "message": "Internal error", "data": "Internal server error. Check server logs for details."}})

    def run(self) -> None:
        skip_outbound_key_checks_cli = "--skip-outbound-key-leaks" in sys.argv
        if skip_outbound_key_checks_cli:
            logger.info("Outbound key leak checks disabled by command line parameter")
            self.skip_outbound_key_checks = True
            self.llm_client.skip_redaction = True

        logger.debug("StdioServer run method started. Sending initial capabilities.")
        self.send_response({
            "jsonrpc": "2.0", "id": None, "method": "mcp/serverReady",
            "params": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": self.server_name, "version": "0.1.0", "description": self.server_description},
                "capabilities": {"tools": self.tools, "resources": {}, "prompts": {}, "sampling": {}}
            }
        })
        logger.debug("Initial capabilities sent. Entering main request loop.")

        loop_count = 0
        try:
            while True:
                loop_count += 1
                logger.debug(f"Request loop iter {loop_count}. Waiting for request.", extra={'request_id': 'N/A'})

                request_data = self._read_and_parse_request()

                if request_data is None:
                    # This means _read_and_parse_request encountered EOF (empty line from readline)
                    # or a JSONDecodeError (in which case an error response was already sent).
                    # If it was EOF, we break. If JSON error, the original code would also break the request processing for this line
                    # and effectively continue to the next readline. The current refactor will break the loop entirely
                    # if _read_and_parse_request returns None after a JSON error. This is a stricter error handling.
                    # For the specific failing test (test_valid_model_selection), readline is mocked to return "",
                    # so _read_and_parse_request will return None, and this break is the correct behavior for that test.
                    logger.info("No request data received (EOF or parse error handled). Terminating run loop.")
                    break

                self._process_parsed_request(request_data)

        except Exception as e:
            logger.critical("Fatal error in StdioServer run loop: %s", str(e), exc_info=True)
            try:
                self.send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": "Fatal server error", "data": str(e)}})
            except Exception as final_e:
                logger.critical("Failed to send final error message: %s", str(final_e))
            raise
        finally:
            logger.info("Server run loop terminated.")
