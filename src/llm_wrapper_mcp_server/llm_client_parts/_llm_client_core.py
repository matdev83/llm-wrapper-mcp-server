import os
import requests
import tiktoken
from typing import Dict, Any, Optional
from ..logger import get_logger
from ._api_key_filter import ApiKeyFilter
from ._accounting import LLMAccountingManager
from ._config import load_system_prompt, get_api_base_url

logger = get_logger(__name__)


class LLMClient:
    """Generic LLM API client with OpenRouter compatibility."""

    def __init__(
        self,
        system_prompt_path: str = "config/prompts/system.txt",
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,  # New parameter
        enable_logging: bool = True,
        enable_rate_limiting: bool = True,
        enable_audit_log: bool = True,
        skip_outbound_key_checks: bool = False,  # New parameter for outbound key checks
    ) -> None:
        """Initialize the client with API key from environment or direct parameter."""
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.skip_redaction = (
            skip_outbound_key_checks  # Use the new parameter for redaction control
        )
        logger.debug("LLMClient initialized")
        self.api_key = api_key or os.getenv(
            "OPENROUTER_API_KEY"
        )  # Use provided key or env var
        logger.debug(f"DEBUG: LLMClient initialized with API Key: {self.api_key}")

        self.enable_rate_limiting = enable_rate_limiting

        self.accounting_manager = LLMAccountingManager(
            enable_logging=enable_logging, enable_audit_log=enable_audit_log
        )
        self.llm_tracker = self.accounting_manager.get_tracker()
        self.audit_logger = self.accounting_manager.get_audit_logger()

        if self.enable_rate_limiting:
            # TODO: Check if llm-accounting supports rate limiting.
            # For now, assume it doesn't and log a warning.
            logger.warning(
                "Rate limiting is enabled but not yet implemented in LLMClient."
            )

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        elif not (self.api_key.startswith("sk-") and len(self.api_key) >= 32):
            raise ValueError(
                "Invalid OPENROUTER_API_KEY format - must start with 'sk-' and be at least 32 characters"
            )

        # Add API key redaction filter to logger
        logger.addFilter(ApiKeyFilter(self.api_key))
        logger.info("API key format validation passed")
        self.base_url = get_api_base_url(api_base_url)
        logger.debug(f"DEBUG: LLMClient using base URL: {self.base_url}")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/llm-wrapper-mcp-server",
            "X-Title": "Ask MCP Server",
            "Content-Type": "application/json",
            "X-API-Version": "1",
            "X-Response-Content": "usage",
        }
        self.model = model

        # Handle system prompt configuration
        self.system_prompt = load_system_prompt(system_prompt_path)

    def generate_response(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a response for the given prompt."""
        # Calculate token counts
        system_tokens = len(self.encoder.encode(self.system_prompt))
        user_tokens = len(self.encoder.encode(prompt))
        logger.debug(
            "Token counts - system: %d, user: %d, total: %d",
            system_tokens,
            user_tokens,
            system_tokens + user_tokens,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
        }

        try:
            # Log outbound prompt
            self.accounting_manager.log_prompt(
                app_name="LLMClient.generate_response",
                user_name=os.getenv("USERNAME", "unknown_user"),
                model=self.model,
                prompt_text=prompt,
            )

            response_content, response_headers, response_data = self._send_llm_request(payload)

            response_tokens = len(self.encoder.encode(response_content))
            logger.debug("Response token count: %d", response_tokens)

            # Log remote reply
            self.accounting_manager.log_response(
                app_name="LLMClient.generate_response",
                user_name=os.getenv("USERNAME", "unknown_user"),
                model=self.model,
                response_text=response_content,
                remote_completion_id=response_data.get(
                    "id"
                ),  # Assuming 'id' is the completion ID
            )

            # Log accounting information
            logger.debug(
                "API usage - Total: %s, Prompt: %s, Completion: %s, Cost: %s",
                response_headers.get("X-Total-Tokens"),
                response_headers.get("X-Prompt-Tokens"),
                response_headers.get("X-Completion-Tokens"),
                response_headers.get("X-Total-Cost"),
            )

            # Record usage with llm-accounting
            self.accounting_manager.track_usage(
                model=self.model,
                prompt_tokens=int(response_headers.get("X-Prompt-Tokens", 0)),
                completion_tokens=int(response_headers.get("X-Completion-Tokens", 0)),
                total_tokens=int(response_headers.get("X-Total-Tokens", 0)),
                cost=float(response_headers.get("X-Total-Cost", 0.0)),
                cached_tokens=int(response_headers.get("X-Cached-Tokens", 0)),
                reasoning_tokens=int(response_headers.get("X-Reasoning-Tokens", 0)),
                caller_name="LLMClient.generate_response",
                project="llm_wrapper_mcp_server",
                username=os.getenv(
                    "USERNAME", "unknown_user"
                ),  # Use USERNAME env var or fallback
            )

            return {
                "response": response_content,
                "input_tokens": system_tokens + user_tokens,
                "output_tokens": response_tokens,
                "api_usage": {
                    "total_tokens": response_headers.get("X-Total-Tokens"),
                    "prompt_tokens": response_headers.get("X-Prompt-Tokens"),
                    "completion_tokens": response_headers.get("X-Completion-Tokens"),
                    "total_cost": response_headers.get("X-Total-Cost"),
                    "cached_tokens": response_headers.get("X-Cached-Tokens"),
                    "reasoning_tokens": response_headers.get("X-Reasoning-Tokens"),
                },
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", 60)
                logger.error("Rate limited - retry after %s seconds", retry_after)
                raise RuntimeError(
                    f"API rate limit exceeded: Retry after {retry_after} seconds"
                ) from e
            logger.error(
                "API HTTP error: %d %s", e.response.status_code, e.response.reason
            )
            raise RuntimeError(
                f"API HTTP error: {e.response.status_code} {e.response.reason}"
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s", str(e))
            raise RuntimeError(f"Network error: {str(e)}") from e
        except KeyError as e:
            logger.error("Malformed API response: %s", str(e))
            raise RuntimeError(f"Unexpected API response format: {str(e)}") from e

    def _send_llm_request(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        logger.debug(
            "DEBUG: Sending LLM API request to %s",
            f"{self.base_url}/chat/completions",
        )
        logger.debug("DEBUG: Request payload: %s", payload)
        logger.debug("DEBUG: Request headers: %s", self.headers)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30,
        )

        logger.debug(
            "DEBUG: Received API response: %d %s",
            response.status_code,
            response.reason,
        )
        logger.debug("DEBUG: Response headers: %s", dict(response.headers))
        logger.debug(
            "DEBUG: Response content (first 200 chars): %.200s...", response.text
        )

        response.raise_for_status()
        data = response.json()

        if not isinstance(data.get("choices"), list) or len(data["choices"]) == 0:
            raise RuntimeError("Invalid API response format: Missing choices array")

        first_choice = data["choices"][0]
        if (
            "message" not in first_choice
            or "content" not in first_choice["message"]
        ):
            raise RuntimeError(
                "Invalid API response format: Missing message content"
            )

        response_content = first_choice["message"]["content"]
        response_content = self.redact_api_key(response_content)
        return response_content, response.headers, data

    def redact_api_key(self, content: str) -> str:
        """Redact actual API key value from content."""
        if self.skip_redaction:
            return content
        if self.api_key and self.api_key in content:
            logger.warning("Redacting API key from response content")
            return content.replace(
                self.api_key, "(API key redacted due to security reasons)"
            )
        return content
