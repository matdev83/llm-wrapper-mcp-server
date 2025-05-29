import json
import pytest
import tiktoken
import os # Added import for os
from unittest.mock import patch, mock_open, Mock
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

# Helper to create prompts of specific token lengths
def make_prompt(token_length: int) -> str:
    encoder = tiktoken.get_encoding("cl100k_base")
    return " ".join(["test"] * token_length)

def test_server_initialization():
    server = LLMMCPWrapper()
    assert len(server.tools) == 1
    assert "llm_call" in server.tools

@patch('llm_wrapper_mcp_server.llm_mcp_wrapper.logger') # Corrected patch path
@patch('sys.argv', ['', '--skip-accounting'])
@patch('sys.stdin.readline', return_value='')
def test_skip_accounting_flag(mock_readline, mock_logger):
    server = LLMMCPWrapper()
    server.run()
    mock_logger.info.assert_any_call("Accounting disabled by command line parameter")

def test_skip_accounting_initialization():
    server = LLMMCPWrapper(skip_accounting=True)
    assert hasattr(server, 'skip_accounting'), "LLMMCPWrapper should have skip_accounting attribute"
    assert server.skip_accounting is True

@patch('sys.stdout')
def test_handle_initialize_request(mock_stdout):
    server = LLMMCPWrapper()
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    })
    
    # Get the written response
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    
    assert "result" in response
    assert response["result"]["serverInfo"]["name"] == "llm-wrapper-mcp-server"

@patch('sys.stdout')
def test_handle_invalid_request(mock_stdout):
    server = LLMMCPWrapper()
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "invalid.method",
        "id": 1
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    
    assert "error" in response
    assert response["error"]["code"] == -32601

@patch('llm_wrapper_mcp_server.llm_client.LLMClient.generate_response') # Patch generate_response directly
@patch('sys.stdout')
def test_prompt_under_limit(mock_stdout, mock_generate_response):
    # Configure the mock generate_response
    mock_generate_response.return_value = {
        "response": "Mocked response",
        "input_tokens": 50,
        "output_tokens": 10,
        "api_usage": {
            "total_tokens": 60,
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_cost": 0.001
        }
    }
    os.environ["OPENROUTER_API_KEY"] = "sk-testkey1234567890abcdefghijklmnopqr" # Set a dummy API key for the test
    server = LLMMCPWrapper(max_user_prompt_tokens=100)
    # Mock the encoder attribute on the actual LLMClient instance created by LLMMCPWrapper
    server.llm_client.encoder = Mock()
    server.llm_client.encoder.encode.return_value = [] # Return empty list for token counting

    prompt = make_prompt(50)  # Well under limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "llm_call",
            "arguments": {"prompt": prompt}
        }
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    assert "error" not in response
    mock_generate_response.assert_called_once() # Assert on the patched generate_response

@patch('llm_wrapper_mcp_server.llm_client.LLMClient.generate_response') # Patch generate_response directly
@patch('sys.stdout')
def test_prompt_at_limit(mock_stdout, mock_generate_response):
    # Configure the mock generate_response
    mock_generate_response.return_value = {
        "response": "Mocked response",
        "input_tokens": 100,
        "output_tokens": 20,
        "api_usage": {
            "total_tokens": 120,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_cost": 0.002
        }
    }
    os.environ["OPENROUTER_API_KEY"] = "sk-testkey1234567890abcdefghijklmnopqr" # Set a dummy API key for the test
    server = LLMMCPWrapper(max_user_prompt_tokens=100)
    # Mock the encoder attribute on the actual LLMClient instance created by LLMMCPWrapper
    server.llm_client.encoder = Mock()
    server.llm_client.encoder.encode.return_value = [] # Return empty list for token counting

    prompt = make_prompt(100)  # Exactly at limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "llm_call",
            "arguments": {"prompt": prompt}
        }
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    assert "error" not in response
    mock_generate_response.assert_called_once() # Assert on the patched generate_response

@patch('sys.stdout')
def test_prompt_over_limit(mock_stdout):
    server = LLMMCPWrapper(max_user_prompt_tokens=100)
    prompt = make_prompt(101)  # Just over limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "llm_call",
            "arguments": {"prompt": prompt}
        }
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    assert "error" in response
    assert response["error"]["code"] == -32602
    assert "exceeds maximum length" in response["error"]["data"]

@patch('sys.stdout')
def test_send_response(mock_stdout):
    server = LLMMCPWrapper(max_user_prompt_tokens=100)
    test_response = {"test": "response"}
    server.send_response(test_response)
    mock_stdout.write.assert_called_with(json.dumps(test_response) + "\n")
