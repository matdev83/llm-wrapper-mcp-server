import pytest
import logging
import os
import json
from unittest.mock import patch, Mock
from llm_wrapper_mcp_server.__main__ import main

def test_valid_model_selection(tmp_path, caplog):
    """Test valid model selection from allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\nanother/model")
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'perplexity/llama-3.1-sonar-small-128k-online'
    ]), patch('llm_wrapper_mcp_server.stdio_server.StdioServer.run') as mock_run:
        mock_run.side_effect = lambda: None  # Prevent actual server startup
        main()
    
    # Should have no validation errors
    assert "Allowed models file not found" not in caplog.text
    assert "not in the allowed models list" not in caplog.text
    assert "Allowed models file is empty" not in caplog.text

def test_missing_model_file(tmp_path, caplog):
    """Test missing allowed models file handling"""
    missing_file = tmp_path / "missing.txt"
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(missing_file)
    ]), pytest.raises(SystemExit) as excinfo:
        main()
    
    assert excinfo.value.code == 1
    assert f"Allowed models file not found: {missing_file}" in caplog.text

def test_empty_model_file(tmp_path, caplog):
    """Test empty allowed models file handling"""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\n\n  \n")  # Only whitespace
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(empty_file)
    ]), pytest.raises(SystemExit) as excinfo:
        main()
    
    assert excinfo.value.code == 1
    assert "Allowed models file is empty" in caplog.text

def test_invalid_model_formatting(mocker, caplog):
    """Test various invalid model name formats"""
    test_cases = [
        ("", "Model name must be at least 2 characters"),
        ("a", "Model name must be at least 2 characters"),
        ("noslash", "Model name must contain a '/' separator"),
        ("  ", "Model name must be at least 2 characters"),
        ("/missingprovider", "Model name must contain a provider and a model separated by a single '/'"),
        ("missingmodel/", "Model name must contain a provider and a model separated by a single '/'")
    ]
    
    for model, expected_error in test_cases:
        # Patch LLMClient at the module level to control its instantiation
        with patch('llm_wrapper_mcp_server.stdio_server.LLMClient') as MockLLMClient:
            # Configure the side_effect for MockLLMClient to return a properly configured mock
            def side_effect_llm_client(*args, **kwargs):
                mock_instance = mocker.Mock()
                # Set attributes based on kwargs passed to LLMClient constructor
                mock_instance.system_prompt = kwargs.get('system_prompt_path', '')
                mock_instance.model = kwargs.get('model', 'default/model')
                mock_instance.base_url = kwargs.get('api_base_url', "https://mocked.api") # Crucial for this error
                
                mock_instance.encoder = mocker.Mock()
                mock_instance.encoder.encode.return_value = []
                mock_instance.generate_response.return_value = {"response": "mocked response content"}
                return mock_instance

            MockLLMClient.side_effect = side_effect_llm_client

            # Initialize server (this will now use the mocked LLMClient)
            from llm_wrapper_mcp_server.stdio_server import StdioServer
            server = StdioServer()

            # Get the initial mocked LLMClient instance that was created during server initialization
            # This is MockLLMClient.side_effect() called once.
            mock_llm_client_instance = MockLLMClient.call_args.return_value
        
        # Mock a tools/call request with invalid model
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "ask_online",
                "arguments": {
                    "prompt": "test prompt",
                    "model": model
                }
            }
        }
        
        # Capture stdout
        from io import StringIO
        import sys
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        
        server.handle_request(request)
        
        # Get and parse response
        response = sys.stdout.getvalue()
        sys.stdout = original_stdout
        response_data = json.loads(response)
        
        assert response_data["error"]["code"] == -32602
        assert response_data["error"]["message"] == "Invalid model specification"
        assert expected_error in response_data["error"]["data"]
        # Ensure generate_response was NOT called for invalid model formats
        mock_llm_client_instance.generate_response.assert_not_called()
        mock_llm_client_instance.generate_response.reset_mock() # Reset mock for next iteration

def test_invalid_model_selection(tmp_path, caplog):
    """Test invalid model not in allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("allowed/model-1\nallowed/model-2")
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'invalid/model'
    ]), pytest.raises(SystemExit) as excinfo:
        main()
    
    assert excinfo.value.code == 1
    assert "Model 'invalid/model' is not in the allowed models list" in caplog.text
