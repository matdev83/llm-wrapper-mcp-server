import pytest
import logging
import os
import json
import sys
from unittest.mock import patch, Mock
from src.llm_wrapper_mcp_server.__main__ import main
from src.llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

VALID_DUMMY_API_KEY = "sk-dummykeyforvalidationtests1234567890"

@patch.dict(os.environ, {"OPENROUTER_API_KEY": VALID_DUMMY_API_KEY})
@patch('src.llm_wrapper_mcp_server.__main__.logger.warning')
def test_valid_model_selection(mock_main_logger_warning, tmp_path):
    """Test valid model selection from allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\nanother/model")

    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'perplexity/llama-3.1-sonar-small-128k-online'
    ]), patch('src.llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper.run') as mock_run, \
         patch('sys.stdin.readline', return_value=''):
        mock_run.side_effect = lambda: None
        main()

    mock_main_logger_warning.assert_not_called()

@patch('src.llm_wrapper_mcp_server.__main__.logger.warning')
def test_missing_model_file(mock_main_logger_warning, tmp_path):
    """Test missing allowed models file handling"""
    missing_file = tmp_path / "missing.txt"

    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(missing_file)
    ]), patch('sys.stdin.readline', return_value=''), pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    mock_main_logger_warning.assert_called_with(f"Allowed models file not found: {missing_file}")

@patch('src.llm_wrapper_mcp_server.__main__.logger.warning')
def test_empty_model_file(mock_main_logger_warning, tmp_path):
    """Test empty allowed models file handling"""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\n\n  \n")

    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(empty_file)
    ]), patch('sys.stdin.readline', return_value=''), pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    mock_main_logger_warning.assert_called_with("Allowed models file is empty - must contain at least one model name")

@patch.dict(os.environ, {"OPENROUTER_API_KEY": VALID_DUMMY_API_KEY})
@patch('src.llm_wrapper_mcp_server.llm_client.LLMClient')
def test_invalid_model_formatting(MockLLMClient, mocker):
    """Test various invalid model name formats"""
    mock_llm_client_instance = MockLLMClient.return_value
    mock_llm_client_instance.system_prompt = ''
    mock_llm_client_instance.model = 'default/model'
    mock_llm_client_instance.base_url = "https://mocked.api"
    mock_llm_client_instance.encoder = mocker.Mock()
    mock_llm_client_instance.encoder.encode.return_value = []
    mock_llm_client_instance.generate_response.return_value = {"response": "mocked response content"}

    mock_llm_client_instance.skip_redaction = False

    server = LLMMCPWrapper()

    test_cases = [
        ("", "Model name must be at least 2 characters"),
        ("a", "Model name must be at least 2 characters"),
        ("noslash", "Model name must contain a '/' separator"),
        ("  ", "Model name must be at least 2 characters"),
        ("/missingprovider", "Model name must contain a provider and a model separated by a single '/'"),
        ("missingmodel/", "Model name must contain a provider and a model separated by a single '/'")
    ]

    for model, expected_error in test_cases:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "llm_call",
                "arguments": {
                    "prompt": "test prompt",
                    "model": model
                }
            }
        }

        from io import StringIO
        import sys
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        server.handle_request(request)

        response = sys.stdout.getvalue()
        sys.stdout = original_stdout
        response_data = json.loads(response)

        assert response_data["error"]["code"] == -32602
        assert response_data["error"]["message"] == "Invalid model specification"
        assert expected_error in response_data["error"]["data"]
        mock_llm_client_instance.generate_response.assert_not_called()
        mock_llm_client_instance.generate_response.reset_mock()

@patch('src.llm_wrapper_mcp_server.__main__.logger.warning')
def test_invalid_model_selection(mock_main_logger_warning, tmp_path):
    """Test invalid model not in allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("allowed/model-1\nallowed/model-2")

    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'invalid/model'
    ]), patch('sys.stdin.readline', return_value=''), pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    # Updated assertion to include the dynamic file path
    expected_log_message = f"Model 'invalid/model' is not in the allowed models list from {model_file}"
    mock_main_logger_warning.assert_called_with(expected_log_message)
