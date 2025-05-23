"""
STDIO-based MCP server implementation.
"""
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .logger import get_logger

logger = get_logger(__name__)

from llm_delegate_mcp_server.llm_client import LLMClient


class StdioServer:
    """STDIO-based MCP server implementation."""
    
    def __init__(
        self,
        system_prompt_path: str = "config/prompts/system.txt",
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        llm_api_base_url: Optional[str] = None,
        max_user_prompt_tokens: int = 100,
        skip_outbound_key_checks: bool = False,
        skip_api_key_redaction: bool = False,
        skip_accounting: bool = False,
        max_tokens: Optional[int] = None
    ) -> None:
        """Initialize the server with configuration options."""
        logger.debug("StdioServer initialized")
        self.llm_client = LLMClient(
            system_prompt_path=system_prompt_path,
            model=model,
            api_base_url=llm_api_base_url
        )
        self.max_user_prompt_tokens = max_user_prompt_tokens
        self.skip_outbound_key_checks = skip_outbound_key_checks
        self.skip_accounting = skip_accounting
        self.max_tokens = max_tokens
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.tools = {
            "ask_online": {
                "description": "Ask a natural language question for an online search assistant to retrieve information or perform fact checking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The natural language question to ask. Maximum length is {max_tokens} tokens.".format(max_tokens=self.max_user_prompt_tokens)
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
                            "name": "llm-delegate-mcp-server",
                            "version": "0.1.0",
                            "description": "Generic LLM API MCP server"
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
                
                if name == "ask_online":
                    prompt = args.get("prompt")
                    if not prompt:
                        logger.warning("Missing required 'prompt' argument for ask_online.", extra={'request_id': request_id})
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
                    
                    # Check for API key leak in prompt
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
                        return  # Make sure to return immediately after sending error response
                    
                    # Check prompt token length
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
                    
                    # Get the optional model parameter
                    model = args.get("model")
                    
                    # Variable to hold the model to use for generation
                    model_to_use = None

                    # Validate model name if specified
                    if model is not None: # Check if model argument was provided at all
                        stripped_model = model.strip() # Strip whitespace before validation
                        if len(stripped_model) < 2:
                            self.send_response({
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid model specification",
                                    "data": "Model name must be at least 2 characters"
                                }
                            })
                            return
                        if '/' not in stripped_model:
                            self.send_response({
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid model specification",
                                    "data": "Model name must contain a '/' separator"
                                }
                            })
                            return
                        
                        parts = stripped_model.split('/')
                        if len(parts) != 2 or not all(parts): # Check if there are exactly two parts and both are non-empty
                            self.send_response({
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32602,
                                    "message": "Invalid model specification",
                                    "data": "Model name must contain a provider and a model separated by a single '/'"
                                }
                            })
                            return
                        
                        # If model was provided and passed validation, set it for use
                        model_to_use = stripped_model
                    
                    try:
                        # Determine which LLMClient to use
                        client_to_use = self.llm_client
                        if model_to_use: # If a specific, validated model was determined
                            # Create a temporary LLM client with the specified model
                            temp_client = LLMClient(
                                system_prompt_path=self.llm_client.system_prompt,
                                model=model_to_use,
                                api_base_url=self.llm_client.base_url
                            )
                            client_to_use = temp_client
                        
                        response = client_to_use.generate_response(prompt=prompt, max_tokens=self.max_tokens)

                        # Construct the response in the format observed from fetch-mcp
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
                        logger.debug("Sending MCP response: %s", mcp_response, extra={'request_id': request_id})
                        self.send_response(mcp_response)
                        logger.debug("send_response completed.", extra={'request_id': request_id})

                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        logger.error("Error during ask_online tool execution: %s", str(e), extra={'request_id': request_id})
                        self.send_response({
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32000,
                                "message": f"Internal error: {str(e)}",
                            "data": str(tb) # Ensure data is always a string
                        }
                    })
                else:
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
            elif method == "resources/list":
                logger.debug("Handling resources/list request.", extra={'request_id': request_id})
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": {}
                    }
                })
                logger.debug("resources/list response sent.", extra={'request_id': request_id})
            elif method == "resources/templates/list":
                logger.debug("Handling resources/templates/list request.", extra={'request_id': request_id})
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "templates": {}
                    }
                })
                logger.debug("resources/templates/list response sent.", extra={'request_id': request_id})
            else:
                logger.warning("Method not found: %s", method, extra={'request_id': request_id})
                self.send_response({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": "Method not found",
                        "data": f"Method '{method}' not found"
                        }
                    })
        except Exception as e:
            logger.error("Error handling request: %s", str(e), extra={'request_id': request.get('id', 'N/A')})
            self.send_response({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32000,
                    "message": "Internal error",
                    "data": str(e) # Ensure data is always a string
                }
            })
    
    def run(self) -> None:
        """Run the server, reading from stdin and writing to stdout."""
        # Parse command line arguments
        skip_outbound_key_checks = False
        skip_accounting = False
        if "--skip-outbound-key-leaks" in sys.argv:
            skip_outbound_key_checks = True
            logger.info("Outbound key leak checks disabled by command line parameter")
        if "--skip-accounting" in sys.argv:
            skip_accounting = True
            logger.info("Accounting disabled by command line parameter")
            
        self.skip_outbound_key_checks = skip_outbound_key_checks
        self.skip_accounting = skip_accounting
        logger.debug("StdioServer run method started. Sending initial capabilities.")
        # Send initial handshake/capabilities message on startup
        self.send_response({
            "jsonrpc": "2.0",
            "id": None,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                            "name": "llm-delegate-mcp-server",
                    "version": "0.1.0",
                    "description": "Generic LLM API MCP server"
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
                logger.debug(f"Request loop iter {loop_count}. PRE sys.stdin.readline()", extra={'request_id': 'N/A'}) # Added request_id placeholder
                line = sys.stdin.readline()

                if not line:
                    logger.info("Empty line or EOF received from stdin. Breaking loop.") # Kept as info, as it's a specific event
                    break 
                
                try:
                    request = json.loads(line)
                    request_id = request.get("id", "N/A") # Get request_id for logging
                    logger.debug("Parsed MCP request: %s", request, extra={'request_id': request_id}) # Added request_id
                    self.handle_request(request)
                except json.JSONDecodeError:
                    logger.error("Parse error: Invalid JSON received from stdin.", extra={'request_id': 'N/A'})
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
                    logger.error("Error in main request loop: %s", str(e), extra={'request_id': 'N/A'})
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32000,
                            "message": "Internal error",
                            "data": str(e) # Ensure data is always a string
                        }
                    })
        except Exception as e:
            logger.critical("Fatal error in StdioServer run loop: %s", str(e))
            raise
