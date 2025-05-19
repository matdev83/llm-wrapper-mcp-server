import json
import pytest
import tiktoken
from unittest.mock import patch, mock_open
from llm_delegate_mcp_server.stdio_server import StdioServer

# Helper to create prompts of specific token lengths
def make_prompt(token_length: int) -> str:
    encoder = tiktoken.get_encoding("cl100k_base")
    return " ".join(["test"] * token_length)

def test_server_initialization():
    server = StdioServer()
    assert len(server.tools) == 1
    assert "ask_online" in server.tools

@patch('sys.stdout')
def test_handle_initialize_request(mock_stdout):
    server = StdioServer()
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1
    })
    
    # Get the written response
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    
    assert "result" in response
    assert response["result"]["serverInfo"]["name"] == "llm-delegate-mcp-server"

@patch('sys.stdout')
def test_handle_invalid_request(mock_stdout):
    server = StdioServer()
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "invalid.method",
        "id": 1
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    
    assert "error" in response
    assert response["error"]["code"] == -32601

@patch('sys.stdout')
def test_prompt_under_limit(mock_stdout):
    server = StdioServer(max_user_prompt_tokens=100)
    prompt = make_prompt(50)  # Well under limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "ask_online",
            "arguments": {"prompt": prompt}
        }
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    assert "error" not in response

@patch('sys.stdout')
def test_prompt_at_limit(mock_stdout):
    server = StdioServer(max_user_prompt_tokens=100)
    prompt = make_prompt(100)  # Exactly at limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "ask_online",
            "arguments": {"prompt": prompt}
        }
    })
    
    written = mock_stdout.write.call_args[0][0]
    response = json.loads(written)
    assert "error" not in response

@patch('sys.stdout')
def test_prompt_over_limit(mock_stdout):
    server = StdioServer(max_user_prompt_tokens=100)
    prompt = make_prompt(101)  # Just over limit
    server.handle_request({
        "jsonrpc": "2.0",
        "method": "tools/call",
        "id": 1,
        "params": {
            "name": "ask_online",
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
    server = StdioServer(max_user_prompt_tokens=100)
    test_response = {"test": "response"}
    server.send_response(test_response)
    mock_stdout.write.assert_called_with(json.dumps(test_response) + "\n")
