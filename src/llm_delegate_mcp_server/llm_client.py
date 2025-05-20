"""
Generic LLM API client with OpenRouter compatibility.
"""
import os
import logging
import requests
import tiktoken
from typing import Dict, Any, Optional
from .logger import get_logger

logger = get_logger(__name__)
# Keep NOTSET to inherit level from root logger
logger.setLevel(logging.NOTSET)
logger.propagate = True

class ApiKeyFilter(logging.Filter):
    """Filter to redact API keys from log messages"""
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        
    def filter(self, record: logging.LogRecord) -> bool:
        if self.api_key:
            record.msg = str(record.msg).replace(self.api_key, "***REDACTED***")
        return True

class LLMClient:
    """Generic LLM API client with OpenRouter compatibility."""
    
    def __init__(
        self,
        system_prompt_path: str = "config/prompts/system.txt",
        model: str = "perplexity/llama-3.1-sonar-small-128k-online",
        api_base_url: Optional[str] = None
    ) -> None:
        """Initialize the client with API key from environment."""
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.skip_redaction = False  # Initialize redaction control flag
        logger.debug("LLMClient initialized")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        elif not (self.api_key.startswith('sk-') and len(self.api_key) >= 32):
            raise ValueError("Invalid OPENROUTER_API_KEY format - must start with 'sk-' and be at least 32 characters")
            
        # Add API key redaction filter to logger
        logger.addFilter(ApiKeyFilter(self.api_key))
        logger.info("API key format validation passed")
        self.base_url = api_base_url or os.getenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Ask MCP Server",
            "Content-Type": "application/json",
            "X-API-Version": "1",
            "X-Response-Content": "usage"
        }
        self.model = model
        
        # Handle system prompt configuration
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read()
        else:
            logger.warning(f"System prompt file {system_prompt_path} not found. Using empty system prompt.")
            self.system_prompt = ""
        
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Generate a response for the given prompt."""
        # Calculate token counts
        system_tokens = len(self.encoder.encode(self.system_prompt))
        user_tokens = len(self.encoder.encode(prompt))
        logger.debug("Token counts - system: %d, user: %d, total: %d", 
                   system_tokens, user_tokens, system_tokens + user_tokens)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens
        }
        
        try:
            logger.trace("Sending LLM API request to %s", f"{self.base_url}/chat/completions")
            logger.trace("Request payload: %s", payload)

            logger.trace("Request headers: %s", self.headers)

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            logger.trace("Received API response: %d %s", response.status_code, response.reason)
            logger.trace("Response headers: %s", dict(response.headers))
            logger.trace("Response content (first 200 chars): %.200s...", response.text)

            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data.get('choices'), list) or len(data['choices']) == 0:
                raise RuntimeError("Invalid API response format: Missing choices array")
                
            first_choice = data['choices'][0]
            if 'message' not in first_choice or 'content' not in first_choice['message']:
                raise RuntimeError("Invalid API response format: Missing message content")
                
            response_content = first_choice['message']['content']
            
            # Redact API key based on skip_redaction flag
            response_content = self.redact_api_key(response_content)
            
            response_tokens = len(self.encoder.encode(response_content))
            logger.debug("Response token count: %d", response_tokens)

            # Log accounting information
            logger.debug(
                "API usage - Total: %s, Prompt: %s, Completion: %s, Cost: %s",
                response.headers.get("X-Total-Tokens"),
                response.headers.get("X-Prompt-Tokens"),
                response.headers.get("X-Completion-Tokens"),
                response.headers.get("X-Total-Cost")
            )
            
            return {
                "response": response_content,
                "input_tokens": system_tokens + user_tokens,
                "output_tokens": response_tokens,
                "api_usage": {
                    "total_tokens": response.headers.get("X-Total-Tokens"),
                    "prompt_tokens": response.headers.get("X-Prompt-Tokens"),
                    "completion_tokens": response.headers.get("X-Completion-Tokens"),
                    "total_cost": response.headers.get("X-Total-Cost")
                }
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get('Retry-After', 60)
                logger.error("Rate limited - retry after %s seconds", retry_after)
                raise RuntimeError(f"API rate limit exceeded: Retry after {retry_after} seconds") from e
            logger.error("API HTTP error: %d %s", e.response.status_code, e.response.reason)
            raise RuntimeError(f"API HTTP error: {e.response.status_code} {e.response.reason}") from e
        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s", str(e))
            raise RuntimeError(f"Network error: {str(e)}") from e
        except KeyError as e:
            logger.error("Malformed API response: %s", str(e))
            raise RuntimeError(f"Unexpected API response format: {str(e)}") from e

    def redact_api_key(self, content: str) -> str:
        """Redact actual API key value from content."""
        if self.skip_redaction:
            return content
        if self.api_key and self.api_key in content:
            logger.warning("Redacting API key from response content")
            return content.replace(self.api_key, "(API key redacted due to security reasons)")
        return content
