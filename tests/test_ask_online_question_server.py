# tests/test_ask_online_question_server.py
import pytest
import json
import subprocess
import sys
import os
import time
from unittest.mock import patch, MagicMock, call
from ask_online_question_mcp_server.ask_online_question_server import AskOnlineQuestionServer
# For CLI tests
from src.ask_online_question_mcp_server.__main__ import main as ask_online_main


# Path to LLMClient where it's imported in ask_online_question_server.py
ASKSERVER_LLMCLIENT_PATH = 'ask_online_question_mcp_server.ask_online_question_server.LLMClient'
# Path to AskOnlineQuestionServer where it's imported in its __main__.py
MAIN_ASKSERVER_PATH = "src.ask_online_question_mcp_server.__main__.AskOnlineQuestionServer"


@pytest.fixture
def ask_server_fixture(): # Renamed to make it clear it's a fixture
    with patch(ASKSERVER_LLMCLIENT_PATH) as MockLLMClient:
        mock_llm_client_instance = MockLLMClient.return_value
        mock_llm_client_instance.generate_response.return_value = {"response": "Mocked online question LLM response"}
        # mock_llm_client_instance.close = MagicMock() # Add if close is called by server logic directly

        server = AskOnlineQuestionServer(
            model="test_ask_model",
            system_prompt_path="dummy_prompt.txt",
            # Add new flags with default True for backward compatibility of existing tests
            enable_logging=True,
            enable_rate_limiting=True,
            enable_audit_log=True
        )
        server.llm_client = mock_llm_client_instance # Ensure the instance on server is the mock
        yield server, MockLLMClient # Yield MockLLMClient for inspection if needed

def get_response_from_ask_mock(capsys):
    captured = capsys.readouterr()
    content = captured.out
    # Clear after reading by reading again (capsys behavior)
    # capsys.readouterr()

    # Process each line, as there might be multiple JSON objects (e.g. initial + actual)
    # Return the last valid JSON object found.
    last_json_response = None
    for line in content.strip().splitlines():
        if line.strip():
            try:
                last_json_response = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last_json_response

@pytest.fixture
def run_server_process():
    """Fixture to run the MCP server as a subprocess and yield its process object and stderr buffer."""
    # Ensure LLM_ACCOUNTING_DB_URL is not set to trigger warnings if alembic.ini is missing
    env = os.environ.copy()
    if "LLM_ACCOUNTING_DB_URL" in env:
        del env["LLM_ACCOUNTING_DB_URL"]
    # Set a valid dummy API key for the subprocess environment
    env["OPENROUTER_API_KEY"] = "sk-dummykeyforvalidationtests1234567890"

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

# --- Programmatic Control Tests ---

@patch(ASKSERVER_LLMCLIENT_PATH)
def test_askserver_programmatic_defaults(MockLLMClient_constructor, capsys):
    AskOnlineQuestionServer(model="test_model", system_prompt_path="dummy.txt") # Defaults for enable_*
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(ASKSERVER_LLMCLIENT_PATH)
def test_askserver_programmatic_disable_logging(MockLLMClient_constructor, capsys):
    AskOnlineQuestionServer(model="test_model", system_prompt_path="dummy.txt", enable_logging=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(ASKSERVER_LLMCLIENT_PATH)
def test_askserver_programmatic_disable_audit_log(MockLLMClient_constructor, capsys):
    AskOnlineQuestionServer(model="test_model", system_prompt_path="dummy.txt", enable_audit_log=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is True

@patch(ASKSERVER_LLMCLIENT_PATH)
def test_askserver_programmatic_disable_rate_limiting(MockLLMClient_constructor, capsys):
    AskOnlineQuestionServer(model="test_model", system_prompt_path="dummy.txt", enable_rate_limiting=False)
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is False

@patch(ASKSERVER_LLMCLIENT_PATH)
def test_askserver_programmatic_all_disabled(MockLLMClient_constructor, capsys):
    AskOnlineQuestionServer(
        model="test_model", system_prompt_path="dummy.txt",
        enable_logging=False, enable_audit_log=False, enable_rate_limiting=False
    )
    args, kwargs = MockLLMClient_constructor.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is False

# --- CLI Control Tests ---

@patch(MAIN_ASKSERVER_PATH) # Mock AskOnlineQuestionServer in __main__
def test_ask_cli_defaults(MockedAskServerInMain, monkeypatch, capsys):
    # Mandatory args for ask_online_main
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--model', 'cli/test'])
    MockedAskServerInMain.return_value.run = MagicMock() # Prevent actual run

    ask_online_main()

    args, kwargs = MockedAskServerInMain.call_args
    assert kwargs.get('model') == 'cli/test'
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(MAIN_ASKSERVER_PATH)
def test_ask_cli_disable_logging(MockedAskServerInMain, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--model', 'cli/test', '--disable-logging'])
    MockedAskServerInMain.return_value.run = MagicMock()
    ask_online_main()
    args, kwargs = MockedAskServerInMain.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is True

@patch(MAIN_ASKSERVER_PATH)
def test_ask_cli_disable_audit_log(MockedAskServerInMain, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--model', 'cli/test', '--disable-audit-log'])
    MockedAskServerInMain.return_value.run = MagicMock()
    ask_online_main()
    args, kwargs = MockedAskServerInMain.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is True

@patch(MAIN_ASKSERVER_PATH)
def test_ask_cli_disable_rate_limiting(MockedAskServerInMain, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['__main__.py', '--model', 'cli/test', '--disable-rate-limiting'])
    MockedAskServerInMain.return_value.run = MagicMock()
    ask_online_main()
    args, kwargs = MockedAskServerInMain.call_args
    assert kwargs.get('enable_logging') is True
    assert kwargs.get('enable_audit_log') is True
    assert kwargs.get('enable_rate_limiting') is False

@patch(MAIN_ASKSERVER_PATH)
def test_ask_cli_all_disabled(MockedAskServerInMain, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [
        '__main__.py', '--model', 'cli/test',
        '--disable-logging',
        '--disable-audit-log',
        '--disable-rate-limiting'
    ])
    MockedAskServerInMain.return_value.run = MagicMock()
    ask_online_main()
    args, kwargs = MockedAskServerInMain.call_args
    assert kwargs.get('enable_logging') is False
    assert kwargs.get('enable_audit_log') is False
    assert kwargs.get('enable_rate_limiting') is False


# --- Existing Tests (adapted to use ask_server_fixture and clearer capsys handling) ---

def test_ask_server_initialize(ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    get_response_from_ask_mock(capsys) # Clear initial server ready notification if any (depends on run)
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "Ask Online Question"

def test_ask_server_tools_list(ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    get_response_from_ask_mock(capsys)
    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert "ask_online_question" in response["result"]["tools"]

def test_ask_server_tool_call_success(ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    get_response_from_ask_mock(capsys)
    request = {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "ask_online_question", "arguments": {"prompt": "What is pytest?"}}
    }
    server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["result"]["content"][0]["text"] == "Mocked online question LLM response"
    assert response["result"]["isError"] is False
    server.llm_client.generate_response.assert_called_once_with(prompt="What is pytest?")

def test_ask_server_tool_call_missing_prompt(ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    get_response_from_ask_mock(capsys)
    request = {
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "ask_online_question", "arguments": {}}
    }
    server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["error"]["message"] == "Invalid params"
    assert "Missing required 'prompt' argument" in response["error"]["data"]

def test_ask_server_unknown_tool(ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    get_response_from_ask_mock(capsys)
    request = {
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "unknown_tool", "arguments": {"prompt": "test"}}
    }
    server.handle_request(request)
    response = get_response_from_ask_mock(capsys)
    assert response is not None
    assert response["error"]["message"] == "Method not found"
    assert "Tool 'unknown_tool' not found" in response["error"]["data"]

@patch('ask_online_question_mcp_server.ask_online_question_server.sys.stdin')
def test_ask_server_run_loop_and_client_close(mock_stdin, ask_server_fixture, capsys):
    server, _ = ask_server_fixture # Unpack the fixture
    # Server sends initial ready on run, then we send one request, then EOF.
    mock_stdin.readline.side_effect = [
        json.dumps({"jsonrpc": "2.0", "id": 100, "method": "initialize", "params": {}}) + '\n',
        ""  # EOF
    ]
    # Mock the close method on the llm_client *instance* from the fixture
    server.llm_client.close = MagicMock()

    server.run() # Call run on the instance from the fixture

    # Verify close was called.
    # The llm_client instance on ask_server_fixture is the one we want to check.
    server.llm_client.close.assert_called_once()

    # Clear the initial server ready message and the response to initialize
    get_response_from_ask_mock(capsys)
    get_response_from_ask_mock(capsys)

    # Final check that no other output is present
    final_output = capsys.readouterr().out
    assert not final_output.strip(), f"Expected no more output, but got: {final_output}"
