import os
import logging
from typing import Optional
from ..logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.NOTSET)
logger.propagate = True

def load_system_prompt(system_prompt_path: str) -> str:
    """Loads the system prompt from the specified path."""
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, 'r') as f:
            return f.read()
    else:
        logger.warning(f"System prompt file {system_prompt_path} not found. Using empty system prompt.")
        return ""

def get_api_base_url(api_base_url: Optional[str]) -> str:
    """Determines the API base URL."""
    return api_base_url or os.getenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1")
