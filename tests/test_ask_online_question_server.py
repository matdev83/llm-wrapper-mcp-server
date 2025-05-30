# tests/test_ask_online_question_server.py
import pytest
import json
from unittest.mock import patch, MagicMock
from ask_online_question_mcp_server.ask_online_question_server import AskOnlineQuestionServer
import io

@pytest.fixture
def ask_server():
    with patch('ask_online_question_mcp_server.ask_online_question_server.LLMClient') as MockLLMClient:
        mock_llm_client_instance = MockLLMClient.return_value
        mock_llm_client_instance.generate_response.return_value = {"response": "Mocked online question LLM response"}

        server = AskOnlineQuestionServer(
            model="test_ask_model",
            system_prompt_path="dummy_prompt.txt"
        )
        server.llm_client = mock_llm_client_instance
        yield server

def get_response_from_ask_mock(capsys):
    # Read all captured stdout
    captured = capsys.readouterr()
    content = captured.out

    # Process each line, as there might be multiple JSON objects
    for line in content.strip().splitlines():
        if line.strip():  # Ensure the line is not empty
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # If a line is not valid JSON, skip it and try the next
                continue
    return None

def test_ask_server_initialize(ask_server, capsys):
    # The server sends an initial capabilities response during initialization
    # so we need to clear that before making our test request
    get_response_from_ask_mock(capsys)

    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    ask_server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "Ask Online Question"

def test_ask_server_tools_list(ask_server, capsys):
    get_response_from_ask_mock(capsys)  # Clear initial response

    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    ask_server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert "ask_online_question" in response["result"]["tools"]

def test_ask_server_tool_call_success(ask_server, capsys):
    get_response_from_ask_mock(capsys)  # Clear initial response

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "ask_online_question", "arguments": {"prompt": "What is pytest?"}}
    }
    ask_server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["result"]["content"][0]["text"] == "Mocked online question LLM response"
    assert response["result"]["isError"] is False
    ask_server.llm_client.generate_response.assert_called_once_with(prompt="What is pytest?")

def test_ask_server_tool_call_missing_prompt(ask_server, capsys):
    get_response_from_ask_mock(capsys)  # Clear initial response

    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "ask_online_question", "arguments": {}}
    }
    ask_server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["error"]["message"] == "Invalid params"
    assert "Missing required 'prompt' argument" in response["error"]["data"]

def test_ask_server_unknown_tool(ask_server, capsys):
    get_response_from_ask_mock(capsys)  # Clear initial response

    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "unknown_tool", "arguments": {"prompt": "test"}}
    }
    ask_server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["error"]["message"] == "Method not found"
    assert "Tool 'unknown_tool' not found" in response["error"]["data"]

@patch('ask_online_question_mcp_server.ask_online_question_server.sys.stdin')
def test_ask_server_run_loop_and_client_close(mock_stdin, ask_server, capsys):
    # Simulate initial capabilities response, then EOF to stop run loop
    mock_stdin.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 100, "method": "initialize", "params": {}}) + '\n', # initial request
        ""  # EOF
    ]

    ask_server.llm_client.close = MagicMock()

    ask_server.run()

    # Verify close was called once
    ask_server.llm_client.close.assert_called_once()
