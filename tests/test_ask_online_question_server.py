# tests/test_ask_online_question_server.py
import pytest
import json
import subprocess
import sys
import os
import time
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

@pytest.fixture
def run_server_process():
    """Fixture to run the MCP server as a subprocess and yield its process object and stderr buffer."""
    # Ensure LLM_ACCOUNTING_DB_URL is not set to trigger warnings if alembic.ini is missing
    env = os.environ.copy()
    if "LLM_ACCOUNTING_DB_URL" in env:
        del env["LLM_ACCOUNTING_DB_URL"]

    # Start the server as a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "src.ask_online_question_mcp_server", "--model", "test-model"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Use text mode for universal newlines and string I/O
        bufsize=1,  # Line-buffered
        env=env
    )
    
    stderr_buffer = []
    yield process, stderr_buffer
    
    # Clean up: terminate the process and read any remaining output
    process.terminate()
    try:
        process.wait(timeout=1)
        if process.stderr:
            stderr_buffer.append(process.stderr.read())
    except subprocess.TimeoutExpired:
        process.kill()
        if process.stderr:
            stderr_buffer.append(process.stderr.read())
    
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()


def test_server_initial_output_no_warnings(run_server_process):
    """
    Test that the server produces no unexpected output (like warnings) before the initial JSON handshake.
    """
    process, stderr_buffer = run_server_process
    
    # Read lines from stdout until the first JSON object is found
    pre_json_stdout = []
    json_found = False
    start_time = time.time()
    timeout = 10  # seconds

    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            # If no line is read, and process is still alive, wait a bit
            if process.poll() is None:
                time.sleep(0.1)
                continue
            else:
                # Process exited, no more output
                break
        
        stripped_line = line.strip()
        if stripped_line:
            try:
                # Attempt to parse as JSON
                json_data = json.loads(stripped_line)
                # If successful, this is the handshake. Stop reading pre-JSON output.
                json_found = True
                break
            except json.JSONDecodeError:
                # Not JSON, so it's unexpected output
                pre_json_stdout.append(stripped_line)
        
    assert json_found, "Server did not produce initial JSON handshake within timeout."
    
    # Assert that all collected pre-JSON output lines are empty or expected
    # Assert that all collected pre-JSON output lines are empty or expected
    print(f"\n--- Debugging Output ---")
    print(f"Pre-JSON stdout: {pre_json_stdout}")
    
    # Check stderr for any errors/warnings after the process has terminated (handled by fixture cleanup)
    final_stderr_output = "".join(stderr_buffer).strip()
    print(f"Final stderr: {final_stderr_output}")
    print(f"--- End Debugging Output ---")

    assert not pre_json_stdout, f"Unexpected output on stdout before JSON handshake: {pre_json_stdout}"
    assert not final_stderr_output, f"Unexpected output on stderr: {final_stderr_output}"

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
