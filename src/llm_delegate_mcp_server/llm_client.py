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
        logger.debug("LLMClient initialized")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        elif not (self.api_key.startswith('sk-') and len(self.api_key) >= 32):
            logger.warning("OPENROUTER_API_KEY appears invalid - should start with 'sk-' and be at least 32 characters")
            
        logger.info(f"OPENROUTER_API_KEY validation passed - starts with 'sk-' and has {len(self.api_key)} characters")
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
        
    def generate_response(self, prompt: str) -> Dict[str, Any]:
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
            ]
        }
        
        try:
            logger.trace("Sending LLM API request to %s", f"{self.base_url}/chat/completions")
            logger.trace("Request payload: %s", payload)

            # Log redacted headers
            redacted_headers = {k: "***" if k == "Authorization" else v for k,v in self.headers.items()}
            logger.trace("Request headers: %s", redacted_headers)

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

            response_content = data['choices'][0]['message']['content']
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

        except requests.exceptions.RequestException as e:
            logger.error("API request failed: %s", str(e))
            raise RuntimeError(f"LLM API error: {str(e)}") from e
