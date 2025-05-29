import pytest
import json
from unittest.mock import patch, MagicMock
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

import io

@pytest.fixture
def mock_stdout(mocker):
    # Patch the sys module within llm_mcp_wrapper
    mock_sys = mocker.patch('llm_wrapper_mcp_server.llm_mcp_wrapper.sys')
    # Set the stdout attribute of the mocked sys module to a StringIO object
    mock_sys.stdout = io.StringIO()
    yield mock_sys.stdout

@pytest.fixture
def mcp_wrapper(): # mock_stdout is now handled by redirecting sys.stdout
    # Initialize LLMMCPWrapper with minimal config for testing
    # Mock LLMClient to avoid actual API calls
    with patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMClient') as MockLLMClient:
        mock_llm_client_instance = MockLLMClient.return_value
        mock_llm_client_instance.encoder = MagicMock()
        # Explicitly mock encode to ensure it's a mock object
        mock_llm_client_instance.encoder.encode = MagicMock(return_value=[])
        mock_llm_client_instance.generate_response.return_value = {"response": "Mocked LLM response"}
        
        wrapper = LLMMCPWrapper(
            system_prompt_path="non_existent_path.txt", # Path doesn't matter as LLMClient is mocked
            model="test_model",
            max_user_prompt_tokens=100,
            skip_outbound_key_checks=True, # Simplify testing by skipping this check
            skip_accounting=True # Simplify testing by skipping this check
        )
        yield wrapper

def get_response_from_mock(mock_stdout):
    """Helper to parse JSON response from mock stdout."""
    # Read the content from the StringIO object
    content = mock_stdout.getvalue()
    if not content.strip(): # Check if there's any content
        raise AssertionError(f"sys.stdout was not written to. Content: '{content}'")
    return json.loads(content.strip())


def test_initialize_request(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "serverInfo" in response["result"]
    assert response["result"]["serverInfo"]["name"] == "llm-wrapper-mcp-server"
    assert "capabilities" in response["result"]

def test_tools_list_request(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    assert "result" in response
    assert "tools" in response["result"]
    assert "llm_call" in response["result"]["tools"]

def test_tools_call_llm_call_success(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {
                "prompt": "Hello, LLM!"
            }
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 3
    assert "result" in response
    assert response["result"]["content"][0]["text"] == "Mocked LLM response"
    assert response["result"]["isError"] is False

def test_tools_call_llm_call_missing_prompt(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {}
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 4
    assert "error" in response
    assert response["error"]["message"] == "Invalid params"
    assert response["error"]["data"] == "Missing required 'prompt' argument"

def test_tools_call_unknown_tool(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "unknown_tool",
            "arguments": {}
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 5
    assert "error" in response
    assert response["error"]["message"] == "Method not found"
    assert response["error"]["data"] == "Tool 'unknown_tool' not found"

def test_resources_list_request(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/list",
        "params": {}
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 6
    assert "result" in response
    assert "resources" in response["result"]
    assert response["result"]["resources"] == {}

def test_resources_templates_list_request(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/templates/list",
        "params": {}
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 7
    assert "result" in response
    assert "templates" in response["result"]
    assert response["result"]["templates"] == {}

def test_unknown_method(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "unknown_method",
        "params": {}
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 8
    assert "error" in response
    assert response["error"]["message"] == "Method not found"
    assert response["error"]["data"] == "Method 'unknown_method' not found"

def test_prompt_exceeds_max_tokens(mcp_wrapper, mock_stdout):
    with patch.object(mcp_wrapper.llm_client.encoder, 'encode') as mock_encode:
        mock_encode.return_value = [0] * (mcp_wrapper.max_user_prompt_tokens + 1)
        request = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "llm_call",
                "arguments": {
                    "prompt": "This is a very long prompt that will exceed the token limit."
                }
            }
        }
        mcp_wrapper.handle_request(request)
        # Add assertions to check the mock
        mock_encode.assert_called_once_with("This is a very long prompt that will exceed the token limit.")
        assert len(mock_encode.return_value) > mcp_wrapper.max_user_prompt_tokens

        response = get_response_from_mock(mock_stdout)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 9
        assert "error" in response
        assert response["error"]["message"] == "Invalid params"
        assert f"Prompt exceeds maximum length of {mcp_wrapper.max_user_prompt_tokens} tokens" in response["error"]["data"]

def test_model_validation_invalid_format(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {
                "prompt": "Test prompt",
                "model": "invalid_model" # Missing '/'
            }
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 10
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must contain a '/' separator" in response["error"]["data"]

def test_model_validation_empty_parts(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {
                "prompt": "Test prompt",
                "model": "provider/" # Empty second part
            }
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 11
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must contain a provider and a model separated by a single '/'" in response["error"]["data"]

def test_model_validation_too_short(mcp_wrapper, mock_stdout):
    request = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {
                "prompt": "Test prompt",
                "model": "a" # Too short
            }
        }
    }
    mcp_wrapper.handle_request(request)
    response = get_response_from_mock(mock_stdout)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 12
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must be at least 2 characters" in response["error"]["data"]
