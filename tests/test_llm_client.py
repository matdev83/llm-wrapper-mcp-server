import os
import pytest
import requests
from unittest.mock import patch, Mock
from src.llm_wrapper_mcp_server.llm_client import LLMClient, logger, ApiKeyFilter

@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-valid-test-key-1234567890abcdef")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://mock.openrouter.ai/api/v1")

@pytest.fixture
def client(mock_env):
    return LLMClient(
        system_prompt_path="tests/fixtures/system_prompt.txt",
        model="test-model"
    )

def test_initialization_with_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable not set"):
        LLMClient()

def test_invalid_api_key_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "invalid-key")
    with pytest.raises(ValueError, match="Invalid OPENROUTER_API_KEY format"):
        LLMClient()

def test_system_prompt_loading(tmp_path):
    # Create temporary system prompt file
    prompt_file = tmp_path / "system.txt"
    prompt_file.write_text("Test system prompt")
    
    client = LLMClient(system_prompt_path=str(prompt_file))
    assert client.system_prompt == "Test system prompt"

def test_missing_system_prompt_file(caplog):
    client = LLMClient(system_prompt_path="non_existent.txt")
    assert "System prompt file non_existent.txt not found" in caplog.text
    assert client.system_prompt == ""

@patch("requests.post")
def test_successful_response(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": "Test response",
                "role": "assistant"
            }
        }]
    }
    mock_response.headers = {
        "X-Total-Tokens": "100",
        "X-Prompt-Tokens": "80",
        "X-Completion-Tokens": "20",
        "X-Total-Cost": "0.05"
    }
    mock_response.text = '{"choices":[{"message":{"content":"Test response"}}]}'
    mock_post.return_value = mock_response

    response = client.generate_response("Test prompt")
    
    assert response["response"] == "Test response"
    assert response["input_tokens"] == len(client.encoder.encode(client.system_prompt)) + len(client.encoder.encode("Test prompt"))
    assert response["api_usage"]["total_cost"] == "0.05"

@patch("requests.post")
def test_rate_limiting_handling(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "30"}
    mock_post.return_value = mock_response
    mock_post.side_effect = requests.exceptions.HTTPError(response=mock_response)

    with pytest.raises(RuntimeError, match="Retry after 30 seconds"):
        client.generate_response("Test prompt")

@patch("requests.post")
def test_network_error_handling(mock_post, client):
    mock_post.side_effect = requests.exceptions.ConnectionError("Network failure")
    
    with pytest.raises(RuntimeError, match="Network error"):
        client.generate_response("Test prompt")

@patch("requests.post")
def test_malformed_response_handling(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"invalid": "response"}
    mock_response.headers = {}
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError, match="Missing choices array"):
        client.generate_response("Test prompt")

def test_api_key_redaction(caplog):
    client = LLMClient()
    assert client.api_key is not None  # Type check guard
    test_content = f"Here is the key: {client.api_key}"
    redacted = client.redact_api_key(test_content)
    
    assert "(API key redacted due to security reasons)" in redacted
    assert client.api_key not in redacted
    assert "Redacting API key" in caplog.text

def test_response_redaction_disabled(client):
    assert client.api_key is not None  # Type check guard
    client.skip_redaction = True
    test_content = f"Here is the key: {client.api_key}"
    redacted = client.redact_api_key(test_content)
    
    assert client.api_key in redacted

def test_request_headers(client):
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "test"}}]
        }
        
        client.generate_response("test")
        
        headers = mock_post.call_args[1]["headers"]
        assert headers["X-Title"] == "Ask MCP Server"
        assert headers["X-API-Version"] == "1"
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

@patch("requests.post")
def test_token_counting_special_chars(mock_post, client):
    client.system_prompt = "Thïs häs spéciäl chäracters"
    test_prompt = "Âccéntéd téxt"
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_response.headers = {}
    mock_post.return_value = mock_response

    response = client.generate_response(test_prompt)
    system_tokens = len(client.encoder.encode(client.system_prompt))
    user_tokens = len(client.encoder.encode(test_prompt))
    
    assert response["input_tokens"] == system_tokens + user_tokens

def test_logger_filter_attachment(client):
    assert any(isinstance(f, ApiKeyFilter) 
              for f in logger.filters)

def test_timeout_handling():
    client = LLMClient()
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(RuntimeError, match="Network error"):
            client.generate_response("test")

def test_default_base_url(monkeypatch):
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    client = LLMClient()
    assert client.base_url == "https://openrouter.ai/api/v1"
