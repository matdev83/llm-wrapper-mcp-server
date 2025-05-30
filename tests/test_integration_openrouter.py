import pytest
import os
import json
import io
from unittest.mock import patch, MagicMock

# Attempt to import requests and skip if unavailable (e.g. no internet for tool)
try:
    import requests
    from requests import exceptions
except ImportError:
    requests = None
    exceptions = None # Ensure exceptions is None if requests is None

from src.llm_wrapper_mcp_server.llm_client import LLMClient
from src.llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

# Provided API Key and Model for testing
TEST_MODEL = "microsoft/phi-4-reasoning:free" # Adjusted to a known free model if previous was example
INVALID_API_KEY = "sk-invalid-dummy-key"

# Pytest marker for integration tests
integration_test_marker = pytest.mark.integration

# Skip condition for tests if requests is not available or connection errors occur
skip_if_no_requests = pytest.mark.skipif(requests is None, reason="requests library not found, skipping integration tests")

@integration_test_marker
@skip_if_no_requests
class TestOpenRouterIntegration:

    @pytest.fixture(autouse=True)
    def set_api_key(self, monkeypatch):
        """Set the OpenRouter API key for the duration of the test class."""
        # We no longer set TEST_API_KEY at module level.
        # Instead, we ensure OPENROUTER_API_KEY is available for LLMClient.
        # The user's environment should provide OPENROUTER_API_KEY.
        # If not, this test will fail due to LLMClient's ValueError.
        
        # Also ensure that if LLM_API_BASE_URL is set by other tests, it's OpenRouter for these
        monkeypatch.setenv("LLM_API_BASE_URL", "https://openrouter.ai/api/v1")


    def test_direct_llm_client_call(self, monkeypatch):
        """Test Case 1: Basic LLM Call via LLMClient directly."""
        # This test will now make actual API calls.
        # Ensure OPENROUTER_API_KEY is set to a valid key for this to pass.

        # Retrieve the API key from the environment within the test scope
        api_key_from_env = os.getenv("OPENROUTER_API_KEY")
        
        # Pass the API key directly to LLMClient to bypass module-level os.getenv issues
        client = LLMClient(model=TEST_MODEL, api_key=api_key_from_env)
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


    def test_llm_call_via_mcp_wrapper(self, monkeypatch):
        """Test Case 2: LLM Call via LLMMCPWrapper simulating MCP."""
        # This test will now make actual API calls.
        # Ensure TEST_API_KEY is set to a valid key for this to pass.
        
        # Mock stdin and stdout for the wrapper
        mock_stdin = io.StringIO()
        mock_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdin", mock_stdin)
        monkeypatch.setattr("sys.stdout", mock_stdout)

        # Instantiate LLMMCPWrapper. It will pick up OPENROUTER_API_KEY from env.
        # Ensure other potentially interfering args are set to benign values.
        # Pass the API key directly to LLMMCPWrapper's LLMClient to bypass module-level os.getenv issues
        api_key_from_env = os.getenv("OPENROUTER_API_KEY")
        wrapper = LLMMCPWrapper(
            model=TEST_MODEL, # Default model for the wrapper if not specified in call
            skip_outbound_key_checks=True, # To simplify test, focus on API call
            skip_accounting=False, # We want this to run through LLMClient normally
            llm_api_key=api_key_from_env # Pass API key to LLMMCPWrapper
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
        import re
        # Use regex to find "hola" or "Hola" within the potentially verbose response
        assert re.search(r"hola", result["content"][0]["text"], re.IGNORECASE) is not None

        # Clean up client resources if wrapper had a close method that calls client.close()
        if hasattr(wrapper, 'llm_client') and wrapper.llm_client:
            wrapper.llm_client.close()


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
        # Pass the invalid API key directly to LLMClient
        client_unauth = LLMClient(model=TEST_MODEL, api_key="sk-thisisprobablynotavalidkey12345") # Init should pass
        
        # Mock requests.post to simulate an unauthorized response by raising HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason = "Unauthorized"
        mock_response.url = "https://openrouter.ai/api/v1/chat/completions"
        
        def raise_401_for_status():
            raise exceptions.HTTPError(
                f"{mock_response.status_code} Client Error: {mock_response.reason} for url: {mock_response.url}",
                response=mock_response
            )
        mock_response.raise_for_status.side_effect = raise_401_for_status
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

        with pytest.raises(RuntimeError) as excinfo:
            client_unauth.generate_response("This should fail.")
        
        assert "API HTTP error: 401 Unauthorized" in str(excinfo.value)
        
        client_unauth.close()
