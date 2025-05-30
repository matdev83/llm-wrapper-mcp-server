import os
import pytest
from unittest.mock import patch, MagicMock
from src.llm_wrapper_mcp_server.llm_client import LLMClient
from src.llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper
import logging # Import logging for caplog assertions

@pytest.fixture
def redaction_setup():
    test_api_key = "sk-testkey1234567890abcdefghijklmnopqr"
    os.environ["OPENROUTER_API_KEY"] = test_api_key
    
    mock_response_data = {
        "response": f"Here is your key: {test_api_key}",
        "input_tokens": 10,
        "output_tokens": 20,
        "api_usage": {}
    }
    
    yield test_api_key, mock_response_data
    
    # Teardown
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]

@patch('requests.post')
def test_api_key_redaction_enabled(mock_post, unique_db_paths, redaction_setup):
    """Test that API key is redacted when feature is enabled (default)"""
    test_api_key, mock_response_data = redaction_setup
    
    # Patch LLMClient's __init__ to inject unique DB paths and handle skip_redaction
    with patch('src.llm_wrapper_mcp_server.llm_client.LLMClient.__init__', autospec=True) as mock_llm_client_init:
        def mock_init_side_effect(self_client, *args, **kwargs):
            original_init = LLMClient.__init__
            # Extract skip_redaction from kwargs if present, default to False
            skip_redaction_arg = kwargs.pop('skip_redaction', False)
            # Pass the test_api_key directly to the LLMClient constructor
            original_init(self_client, api_key=test_api_key, db_path_accounting=unique_db_paths[0], db_path_audit=unique_db_paths[1], skip_redaction=skip_redaction_arg, *args, **kwargs)
        mock_llm_client_init.side_effect = mock_init_side_effect

        # Setup mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": f"Here is your key: {test_api_key}"}}]
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper(skip_api_key_redaction=False)
        response = server.llm_client.generate_response("test prompt")
        processed_response = response["response"]
        
        assert "(API key redacted due to security reasons)" in processed_response
        assert test_api_key not in processed_response

@patch('src.llm_wrapper_mcp_server.llm_client.requests.post') # Patch requests.post within llm_client
def test_api_key_redaction_disabled(mock_post, unique_db_paths, redaction_setup):
    """Test that API key remains when redaction is disabled"""
    test_api_key, mock_response_data = redaction_setup

    # Setup mock API response for requests.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": f"Here is your key: {test_api_key}"}}]
    }
    mock_response.headers = {}
    mock_post.return_value = mock_response

    # Patch LLMClient's __init__ to prevent API key validation/real calls AND inject unique DB paths
    with patch('src.llm_wrapper_mcp_server.llm_client.LLMClient.__init__', autospec=True) as mock_llm_client_init:
        # Configure the mock __init__ to call the original __init__ but bypass API key validation
        def mock_init_side_effect(self_client, *args, **kwargs):
            original_init = LLMClient.__init__
            # Extract skip_redaction from kwargs if present, default to False
            skip_redaction_arg = kwargs.pop('skip_redaction', False)
            # Temporarily disable API key validation by setting a dummy key
            original_init(self_client, api_key="sk-dummy-key-1234567890abcdefghijklmnopqr", db_path_accounting=unique_db_paths[0], db_path_audit=unique_db_paths[1], skip_redaction=skip_redaction_arg, *args, **kwargs)
            
        mock_llm_client_init.side_effect = mock_init_side_effect

        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper(skip_api_key_redaction=True)
        
        # Simulate API response containing the actual key
        response = server.llm_client.generate_response("test prompt")
        processed_response = response["response"]
        
        assert test_api_key in processed_response
        assert "(API key redacted due to security reasons)" not in processed_response
        mock_post.assert_called_once() # Assert that requests.post was called

@patch('src.llm_wrapper_mcp_server.llm_client.requests.post') # Patch requests.post within llm_client
@patch('src.llm_wrapper_mcp_server.llm_client.logger') # Patch the specific logger
def test_redaction_logging(mock_logger, mock_post, unique_db_paths, redaction_setup):
    """Test that redaction events are properly logged"""
    test_api_key, mock_response_data = redaction_setup

    # Setup mock API response for requests.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": f"Here is your key: {test_api_key}"}}]
    }
    mock_response.headers = {}
    mock_post.return_value = mock_response

    # Setup mock API response for requests.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": f"Here is your key: {test_api_key}"}}]
    }
    mock_response.headers = {}
    mock_post.return_value = mock_response

    # Directly instantiate LLMClient with the test API key and mocked logger
    # Ensure the logger used by LLMClient is the mocked one
    with patch('src.llm_wrapper_mcp_server.llm_client.logger', new=mock_logger):
        client = LLMClient(
            api_key=test_api_key,
            db_path_accounting=unique_db_paths[0],
            db_path_audit=unique_db_paths[1],
            skip_redaction=False # Ensure redaction is enabled
        )
        client.generate_response("test prompt")
        
        # Assert that the warning message was logged
        mock_logger.warning.assert_called_with("Redacting API key from response content")
