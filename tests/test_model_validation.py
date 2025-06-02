import pytest
import logging
import os
import json
import sys # Added import for sys
from unittest.mock import patch, Mock
from llm_wrapper_mcp_server.__main__ import main
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper # Moved import to top

def test_valid_model_selection(tmp_path, caplog):
    """Test valid model selection from allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\nanother/model")
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'perplexity/llama-3.1-sonar-small-128k-online'
    ]), patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper.run') as mock_run: # Corrected patch path
        mock_run.side_effect = lambda: None  # Prevent actual server startup
        main()
    
    # Should have no validation errors
    assert "Allowed models file not found" not in caplog.text
    assert "not in the allowed models list" not in caplog.text
    assert "Allowed models file is empty" not in caplog.text

def test_missing_model_file(tmp_path, capsys): # Use capsys again
    """Test missing allowed models file handling"""
    missing_file = tmp_path / "missing.txt"
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(missing_file)
    ]), patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper.run'), \
         patch('logging.getLogger') as mock_get_logger, pytest.raises(SystemExit) as excinfo:
        
        # Configure the mock logger to write to sys.stderr
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_logger.warning.side_effect = lambda msg, *args, **kwargs: print(msg % args, file=sys.stderr)
        
        main()
    
    assert excinfo.value.code == 1
    outerr = capsys.readouterr()
    # Check both stdout and stderr as logging might redirect
    assert f"Allowed models file not found: {missing_file}" in (outerr.err + outerr.out)

def test_empty_model_file(tmp_path, capsys): # Use capsys again
    """Test empty allowed models file handling"""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\n\n  \n")  # Only whitespace
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(empty_file)
    ]), patch('sys.stdin.readline', return_value=''), \
         patch('logging.getLogger') as mock_get_logger, pytest.raises(SystemExit) as excinfo:
        
        # Configure the mock logger to write to sys.stderr
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_logger.warning.side_effect = lambda msg, *args, **kwargs: print(msg % args, file=sys.stderr)
        
        main()
    
    assert excinfo.value.code == 1
    outerr = capsys.readouterr()
    # Check both stdout and stderr as logging might redirect
    assert "Allowed models file is empty" in (outerr.err + outerr.out)

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
        with patch('llm_wrapper_mcp_server.llm_client.LLMClient') as MockLLMClient:
            # Configure the mock LLMClient class
            mock_llm_client_instance = MockLLMClient.return_value # This is the instance that LLMMCPWrapper will get
            mock_llm_client_instance.system_prompt = ''
            mock_llm_client_instance.model = 'default/model'
            mock_llm_client_instance.base_url = "https://mocked.api"
            mock_llm_client_instance.encoder = mocker.Mock()
            mock_llm_client_instance.encoder.encode.return_value = []
            mock_llm_client_instance.generate_response.return_value = {"response": "mocked response content"}

            server = LLMMCPWrapper() # This will use the patched LLMClient

            # Now, mock_llm_client_instance is the same as server.llm_client
            # We can assert on mock_llm_client_instance directly
            
            # Mock a tools/call request with invalid model
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

def test_invalid_model_selection(tmp_path, capsys): # Use capsys again
    """Test invalid model not in allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("allowed/model-1\nallowed/model-2")
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'invalid/model'
    ]), patch('sys.stdin.readline', return_value=''), \
         patch('logging.getLogger') as mock_get_logger, pytest.raises(SystemExit) as excinfo:
        
        # Configure the mock logger to write to sys.stderr
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_logger.warning.side_effect = lambda msg, *args, **kwargs: print(msg % args, file=sys.stderr)
        
        main()
    
    assert excinfo.value.code == 1
    outerr = capsys.readouterr()
    # Check both stdout and stderr as logging might redirect
    assert "Model 'invalid/model' is not in the allowed models list" in (outerr.err + outerr.out)
