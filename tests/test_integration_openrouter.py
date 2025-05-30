import pytest
import os
import json
import io
from unittest.mock import patch

# Attempt to import requests and skip if unavailable (e.g. no internet for tool)
try:
    import requests.exceptions
except ImportError:
    requests = None # type: ignore

from src.llm_wrapper_mcp_server.llm_client import LLMClient
from src.llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

# Provided API Key and Model for testing
TEST_API_KEY = "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"
TEST_MODEL = "meta-llama/llama-3.3-8b-instruct:free" # Adjusted to a known free model if previous was example
INVALID_API_KEY = "sk-invalid-dummy-key"

# Pytest marker for integration tests
integration_test_marker = pytest.mark.integration

# Skip condition for tests if requests is not available or connection errors occur
skip_if_no_requests = pytest.mark.skipif(requests is None, reason="requests library not found, skipping integration tests")

# Custom skip decorator for connection errors
def skip_on_connection_error(func):
    if not requests: # If requests itself is not imported
        return skip_if_no_requests(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            pytest.skip(f"Skipping due to ConnectionError: {e}")
        except requests.exceptions.Timeout as e: # Also skip on timeouts
            pytest.skip(f"Skipping due to Timeout: {e}")
    return wrapper


@integration_test_marker
@skip_if_no_requests
class TestOpenRouterIntegration:

    @pytest.fixture(autouse=True)
    def set_api_key(self, monkeypatch):
        """Set the OpenRouter API key for the duration of the test class."""
        monkeypatch.setenv("OPENROUTER_API_KEY", TEST_API_KEY)
        # Also ensure that if LLM_API_BASE_URL is set by other tests, it's OpenRouter for these
        monkeypatch.setenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1")


    @skip_on_connection_error
    def test_direct_llm_client_call(self):
        """Test Case 1: Basic LLM Call via LLMClient directly."""
        client = LLMClient(model=TEST_MODEL)
        prompt = "What is the capital of France?"
        
        response_data = client.generate_response(prompt)

        assert response_data is not None
        assert "response" in response_data
        assert isinstance(response_data["response"], str)
        assert len(response_data["response"].strip()) > 0
        # Check if "Paris" is in the response, a likely answer
        assert "Paris" in response_data["response"]

        assert "api_usage" in response_data
        api_usage = response_data["api_usage"]
        assert api_usage is not None
        # OpenRouter free models might not return all these headers or values might be 0
        # So, we check for presence or if they are positive if present and not None
        for token_key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if api_usage.get(token_key) is not None:
                 assert int(api_usage[token_key]) >= 0
        
        client.close() # Clean up accounting resources


    @skip_on_connection_error
    def test_llm_call_via_mcp_wrapper(self, monkeypatch):
        """Test Case 2: LLM Call via LLMMCPWrapper simulating MCP."""
        
        # Mock stdin and stdout for the wrapper
        mock_stdin = io.StringIO()
        mock_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdin", mock_stdin)
        monkeypatch.setattr("sys.stdout", mock_stdout)

        # Instantiate LLMMCPWrapper. It will pick up OPENROUTER_API_KEY from env.
        # Ensure other potentially interfering args are set to benign values.
        wrapper = LLMMCPWrapper(
            model=TEST_MODEL, # Default model for the wrapper if not specified in call
            skip_outbound_key_checks=True, # To simplify test, focus on API call
            skip_accounting=False # We want this to run through LLMClient normally
        )
        # Ensure the wrapper's client uses the correct API key if it re-initializes or has complex logic
        # For this test, LLMMCPWrapper's __init__ creates an LLMClient which reads from env.
        
        request_id = "mcp-integ-test-1"
        mcp_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "llm_call",
                "arguments": {
                    "prompt": "Translate 'hello' to Spanish.",
                    "model": TEST_MODEL # Explicitly use the test model
                }
            }
        }
        
        # Simulate handling the request (directly calling handle_request)
        wrapper.handle_request(mcp_request)
        
        response_str = mock_stdout.getvalue()
        assert response_str is not None and len(response_str.strip()) > 0
        
        mcp_response = json.loads(response_str)
        
        assert mcp_response["jsonrpc"] == "2.0"
        assert mcp_response["id"] == request_id
        assert "result" in mcp_response
        assert "error" not in mcp_response # Ensure no error field at top level

        result = mcp_response["result"]
        assert result["isError"] is False
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert isinstance(result["content"][0]["text"], str)
        assert len(result["content"][0]["text"].strip()) > 0
        # Check for a likely Spanish translation
        assert "Hola" in result["content"][0]["text"]

        # Clean up client resources if wrapper had a close method that calls client.close()
        if hasattr(wrapper, 'llm_client') and wrapper.llm_client:
            wrapper.llm_client.close()


    @skip_on_connection_error
    def test_invalid_api_key_llm_client(self, monkeypatch):
        """Test Case 3: Error Handling for Invalid API Key with LLMClient."""
        # Temporarily set an invalid API key
        monkeypatch.setenv("OPENROUTER_API_KEY", INVALID_API_KEY)
        
        # This client will fail on __init__ due to key format, or on API call if format check was less strict
        # The current LLMClient checks format on init.
        with pytest.raises(ValueError, match="Invalid OPENROUTER_API_KEY format"):
            LLMClient(model=TEST_MODEL)

        # If we wanted to test a syntactically valid but unauthorized key:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-thisisprobablynotavalidkey12345") 
        client_unauth = LLMClient(model=TEST_MODEL) # Init should pass
        
        with pytest.raises(RuntimeError) as excinfo:
            client_unauth.generate_response("This should fail.")
        
        assert "API HTTP error" in str(excinfo.value) # Expecting a 401 or similar
        # The specific status code might vary (401, 403), checking for generic message part.
        
        client_unauth.close()

```
