import pytest
import logging
import os
import json
from unittest.mock import patch
from llm_delegate_mcp_server.__main__ import main

def test_valid_model_selection(tmp_path, caplog):
    """Test valid model selection from allowed list"""
    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\nanother/model")
    
    with patch('sys.argv', [
        'server.py',
        '--allowed-models-file', str(model_file),
        '--model', 'perplexity/llama-3.1-sonar-small-128k-online'
    ]), patch('llm_delegate_mcp_server.__main__.StdioServer.run') as mock_run:
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

def test_invalid_model_formatting(caplog):
    """Test various invalid model name formats"""
    test_cases = [
        ("", "Model name must be at least 2 characters"),
        ("a", "Model name must be at least 2 characters"),
        ("noslash", "Model name must contain a '/' separator"),
        ("  ", "Model name must be at least 2 characters"),
        ("/missingprovider", "Model name must contain a '/' separator"),
        ("missingmodel/", "Model name must contain a '/' separator")
    ]
    
    for model, expected_error in test_cases:
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
        
        # Initialize server without allowed models list
        from llm_delegate_mcp_server.stdio_server import StdioServer
        server = StdioServer()
        
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
