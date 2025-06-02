import pytest
import json
import sys # For CLI tests
from unittest.mock import patch, MagicMock, call # Added call
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper
# For CLI tests
from src.llm_wrapper_mcp_server.__main__ import main as llm_wrapper_main

import io

# Path to LLMClient where it's imported in llm_mcp_wrapper.py
WRAPPER_LLMCLIENT_PATH = 'llm_wrapper_mcp_server.llm_mcp_wrapper.LLMClient'
# Path to LLMMCPWrapper where it's imported in __main__.py
MAIN_LLMMCPWRAPPER_PATH = "llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper"


@pytest.fixture
def mcp_wrapper_fixture(capsys): # Add capsys here
    # This fixture provides a basic LLMMCPWrapper with a mocked LLMClient
    # for tests that don't need to assert calls to LLMClient constructor.
    with patch(WRAPPER_LLMCLIENT_PATH) as MockLLMClient:
        mock_llm_client_instance = MockLLMClient.return_value
        mock_llm_client_instance.encoder = MagicMock()
        mock_llm_client_instance.encoder.encode = MagicMock(return_value=[]) # Simulate token calculation
        mock_llm_client_instance.generate_response.return_value = {"response": "Mocked LLM response"}
        # Set a dummy api_key on the mocked instance for the temp client creation
        mock_llm_client_instance.api_key = "sk-dummyfixturekey"
        mock_llm_client_instance.base_url = "https://dummyfixture.com/api/v1"


        wrapper = LLMMCPWrapper(
            system_prompt_path="non_existent_path.txt",
            model="test_model",
            max_user_prompt_tokens=100,
            skip_outbound_key_checks=True,
            # Default new flags to True for backward compatibility of existing tests using this fixture
            enable_logging=True,
            enable_rate_limiting=True,
            enable_audit_log=True
        )
        # Attach the mock for LLMClient constructor to the wrapper instance for potential inspection
        yield wrapper


def get_response_from_mock(capsys): # Change parameter to capsys
    captured = capsys.readouterr()
    content = captured.out
    if not content.strip():
        return None
    # Ensure we only parse once if multiple lines are present (take the last one)
    lines = content.strip().split('\n')
    return json.loads(lines[-1])


# --- Programmatic Control Tests for LLMMCPWrapper ---

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_programmatic_defaults(MockLLMClient_constructor, capsys): # Change parameter to capsys
    LLMMCPWrapper() # Rely on default params
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_programmatic_disable_logging(MockLLMClient_constructor, capsys): # Change parameter to capsys
    LLMMCPWrapper(enable_logging=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_programmatic_disable_audit_log(MockLLMClient_constructor, capsys): # Change parameter to capsys
    LLMMCPWrapper(enable_audit_log=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is True

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_programmatic_disable_rate_limiting(MockLLMClient_constructor, capsys): # Change parameter to capsys
    LLMMCPWrapper(enable_rate_limiting=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is False

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_programmatic_all_disabled(MockLLMClient_constructor, capsys): # Change parameter to capsys
    LLMMCPWrapper(enable_logging=False, enable_audit_log=False, enable_rate_limiting=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is False

@patch(WRAPPER_LLMCLIENT_PATH)
def test_wrapper_temp_client_inherits_flags(MockLLMClient_constructor, capsys): # Change parameter to capsys
    # Setup main client mock part
    main_client_instance = MockLLMClient_constructor.return_value
    main_client_instance.encoder.encode.return_value = [1,2,3] # for token counting
    main_client_instance.generate_response.return_value = {"response": "Mocked LLM response"}
    main_client_instance.api_key = "sk-mainclientkey"
    main_client_instance.base_url = "https://mainclient.com/api/v1"

    wrapper = LLMMCPWrapper(
        enable_logging=False,
        enable_audit_log=True, # Mix
        enable_rate_limiting=False,
        skip_outbound_key_checks=True
    )
    # Reset mock because LLMMCPWrapper instantiation already called it
    MockLLMClient_constructor.reset_mock()

    # Mock the generate_response of the *temporary* client instance
    # The temp client is a *new* instance, so its mock needs to be configured
    # when MockLLMClient_constructor is called the *second* time.
    temp_client_instance_mock = MagicMock()
    temp_client_instance_mock.generate_response.return_value = {"response": "Temporary client response"}

    # Side effect to return main_client_instance first, then temp_client_instance_mock
    MockLLMClient_constructor.side_effect = [main_client_instance, temp_client_instance_mock]


    request = {
        "jsonrpc": "2.0", "id": 13, "method": "tools/call",
        "params": {"name": "llm_call", "arguments": {"prompt": "Hello", "model": "custom/model"}}
    }
    wrapper.handle_request(request) # This should create a temporary LLMClient

    # After handle_request, MockLLMClient_constructor should have been called twice:
    # 1. Once during LLMMCPWrapper initialization (already happened, was reset)
    # 2. Once for the temporary client inside handle_request
    # We are interested in the arguments of the *second* call for the temporary client.
    # However, the setup above re-instantiates wrapper, so the first call in call_args_list is the one for main client
    # and second for temp. Let's adjust.

    # Re-patch for a clean check on the temporary client call
    with patch(WRAPPER_LLMCLIENT_PATH) as MockTempLLMClient:
        # Configure the main client on the wrapper instance to have necessary attributes
        wrapper.llm_client = MagicMock()
        wrapper.llm_client.encoder.encode.return_value = [1,2,3]
        wrapper.llm_client.api_key = "sk-mainclientkey"
        wrapper.llm_client.base_url = "https://mainclient.com/api/v1"

        # The wrapper itself was created with enable_logging=False, enable_audit_log=True, enable_rate_limiting=False
        # These are stored as self.enable_logging etc. by the wrapper.
        # These stored values should be passed to the temp client.

        wrapper.handle_request(request) # This call will use MockTempLLMClient for the temp client

        args, kwargs = MockTempLLMClient.call_args
        assert kwargs.get('model') == "custom/model"
        assert kwargs.get('enable_logging') is False # Inherited from wrapper's self.enable_logging
        assert kwargs.get('enable_audit_log') is True  # Inherited
        assert kwargs.get('enable_rate_limiting') is False # Inherited
        assert kwargs.get('api_key') == "sk-mainclientkey" # Inherited from main client instance


# --- CLI Control Tests ---

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper') # Mock LLMMCPWrapper directly
def test_main_cli_defaults(MockedMCPWrapperInMain, monkeypatch, capsys):
    capsys.readouterr() # Clear any previous output
    monkeypatch.setattr(sys, 'argv', ['__main__.py'])
    # Mock the .run() method to prevent the server from actually running
    MockedMCPWrapperInMain.return_value.run = MagicMock()
    llm_wrapper_main()

    args, kwargs = MockedMCPWrapperInMain.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
def test_main_cli_disable_logging(MockedMCPWrapperInMain, monkeypatch, capsys):
    capsys.readouterr() # Clear any previous output
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--disable-logging'])
    MockedMCPWrapperInMain.return_value.run = MagicMock()
    llm_wrapper_main()
    args, kwargs = MockedMCPWrapperInMain.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
def test_main_cli_disable_audit_log(MockedMCPWrapperInMain, monkeypatch, capsys):
    capsys.readouterr() # Clear any previous output
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--disable-audit-log'])
    MockedMCPWrapperInMain.return_value.run = MagicMock()
    llm_wrapper_main()
    args, kwargs = MockedMCPWrapperInMain.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is True

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
def test_main_cli_disable_rate_limiting(MockedMCPWrapperInMain, monkeypatch, capsys):
    capsys.readouterr() # Clear any previous output
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--disable-rate-limiting'])
    MockedMCPWrapperInMain.return_value.run = MagicMock()
    llm_wrapper_main()
    args, kwargs = MockedMCPWrapperInMain.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is False

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
def test_main_cli_all_disabled(MockedMCPWrapperInMain, monkeypatch, capsys):
    capsys.readouterr() # Clear any previous output
    monkeypatch.setattr(sys, 'argv', [
        '__main__.py',
        '--disable-logging',
        '--disable-audit-log',
        '--disable-rate-limiting'
    ])
    MockedMCPWrapperInMain.return_value.run = MagicMock()
    llm_wrapper_main()
    args, kwargs = MockedMCPWrapperInMain.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is False


# --- Existing tests (ensure they still pass or adapt them) ---
# The mcp_wrapper_fixture has been updated to use new flags with True defaults.

def test_initialize_request(mcp_wrapper_fixture, capsys):
    capsys.readouterr() # Clear any previous output
    mcp_wrapper_fixture.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    response = get_response_from_mock(capsys)
    assert response is not None # Add check for None
    assert response["id"] == 1
    assert "serverInfo" in response["result"]

def test_tools_list_request(mcp_wrapper_fixture, capsys):
    capsys.readouterr() # Clear any previous output
    mcp_wrapper_fixture.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    response = get_response_from_mock(capsys)
    assert response is not None # Add check for None
    assert response["id"] == 2
    assert "llm_call" in response["result"]["tools"]

def test_tools_call_llm_call_success(mcp_wrapper_fixture, capsys):
    capsys.readouterr() # Clear any previous output
    request = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "llm_call", "arguments": {"prompt": "Hello, LLM!"}}}
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None # Add check for None
    assert response["id"] == 3
    assert response["result"]["content"][0]["text"] == "Mocked LLM response"

# Add capsys.readouterr() and assert response is not None for other tests that use get_response_from_mock
def test_tools_call_llm_call_missing_prompt(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {}
        }
    }
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 4
    assert "error" in response
    assert response["error"]["message"] == "Invalid params"
    assert response["error"]["data"] == "Missing required 'prompt' argument"

def test_tools_call_unknown_tool(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "unknown_tool",
            "arguments": {}
        }
    }
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 5
    assert "error" in response
    assert response["error"]["message"] == "Method not found"
    assert response["error"]["data"] == "Tool 'unknown_tool' not found"

def test_resources_list_request(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/list",
        "params": {}
    }
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 6
    assert "result" in response
    assert "resources" in response["result"]
    assert response["result"]["resources"] == {}

def test_resources_templates_list_request(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/templates/list",
        "params": {}
    }
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 7
    assert "result" in response
    assert "templates" in response["result"]
    assert response["result"]["templates"] == {}

def test_unknown_method(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "unknown_method",
        "params": {}
    }
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 8
    assert "error" in response
    assert response["error"]["message"] == "Method not found"
    assert response["error"]["data"] == "Method 'unknown_method' not found"

def test_prompt_exceeds_max_tokens(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
    # Access the mocked LLMClient instance from the fixture
    mock_llm_client_instance = mcp_wrapper_fixture.llm_client

    with patch.object(mock_llm_client_instance.encoder, 'encode') as mock_encode:
        mock_encode.return_value = [0] * (mcp_wrapper_fixture.max_user_prompt_tokens + 1)
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
        mcp_wrapper_fixture.handle_request(request)
        mock_encode.assert_called_once_with("This is a very long prompt that will exceed the token limit.")

        response = get_response_from_mock(capsys)
        assert response is not None
        assert response["id"] == 9
        assert "error" in response
        assert f"Prompt exceeds maximum length of {mcp_wrapper_fixture.max_user_prompt_tokens} tokens" in response["error"]["data"]

def test_model_validation_invalid_format(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
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
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 10
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must contain a '/' separator" in response["error"]["data"]

def test_model_validation_empty_parts(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
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
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 11
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must contain a provider and a model separated by a single '/'" in response["error"]["data"]

def test_model_validation_too_short(mcp_wrapper_fixture, capsys):
    capsys.readouterr()
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
    mcp_wrapper_fixture.handle_request(request)
    response = get_response_from_mock(capsys)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 12
    assert "error" in response
    assert response["error"]["message"] == "Invalid model specification"
    assert "Model name must be at least 2 characters" in response["error"]["data"]
