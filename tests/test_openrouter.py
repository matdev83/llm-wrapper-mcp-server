import os
import pytest
import logging
from unittest.mock import patch, MagicMock, call
from src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core import LLMClient # Updated import
from src.llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

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

@patch('requests.post') # Patched globally as _llm_client_core imports requests directly
def test_api_key_redaction_enabled(mock_post, unique_db_paths, redaction_setup):
    """Test that API key is redacted when feature is enabled (default)"""
    test_api_key, mock_response_data = redaction_setup

    # Patch LLMClient's __init__ to inject unique DB paths and handle skip_outbound_key_checks
    with patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.LLMClient.__init__', autospec=True) as mock_llm_client_init: # Updated import
        def mock_init_side_effect(self_client, *args, **kwargs):
            original_init = LLMClient.__init__
            # Extract skip_outbound_key_checks from kwargs if present, default to False
            skip_outbound_key_checks_arg = kwargs.pop('skip_outbound_key_checks', False)
            # Pass the test_api_key directly to the LLMClient constructor
            original_init(self_client, api_key=test_api_key, enable_logging=True, enable_audit_log=True, skip_outbound_key_checks=skip_outbound_key_checks_arg, *args, **kwargs)
        mock_llm_client_init.side_effect = mock_init_side_effect

        # Setup mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": f"Here is your key: {test_api_key}"}}]
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response

        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper(skip_outbound_key_checks=False) # Ensure redaction is enabled
        response = server.llm_client.generate_response("test prompt")
        processed_response = response["response"]

        assert "(API key redacted due to security reasons)" in processed_response
        assert test_api_key not in processed_response

@patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.requests.post') # Corrected patch target
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
    with patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.LLMClient.__init__', autospec=True) as mock_llm_client_init: # Updated import
        # Configure the mock __init__ to call the original __init__ but bypass API key validation
        def mock_init_side_effect(self_client, *args, **kwargs):
            original_init = LLMClient.__init__
            # Extract skip_outbound_key_checks from kwargs if present, default to False
            skip_outbound_key_checks_arg = kwargs.pop('skip_outbound_key_checks', False)
            # Temporarily disable API key validation by setting a dummy key
            original_init(self_client, api_key="sk-dummy-key-1234567890abcdefghijklmnopqr", enable_logging=True, enable_audit_log=True, skip_outbound_key_checks=skip_outbound_key_checks_arg, *args, **kwargs)

        mock_llm_client_init.side_effect = mock_init_side_effect

        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper(skip_outbound_key_checks=True) # Ensure redaction is disabled

        # Simulate API response containing the actual key
        response = server.llm_client.generate_response("test prompt")
        processed_response = response["response"]

        assert test_api_key in processed_response
        assert "(API key redacted due to security reasons)" not in processed_response
        mock_post.assert_called_once() # Assert that requests.post was called

@patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.requests.post') # Corrected patch target
@patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.logger') # Corrected patch target
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

    # Directly instantiate LLMClient with the test API key and mocked logger
    # Ensure the logger used by LLMClient is the mocked one
    # The outer patch for _llm_client_core.logger handles this now.
    # The 'with patch(... new=mock_logger)' is redundant if the decorator correctly patches the logger
    # that LLMClient internally uses. However, to be absolutely sure the instance of logger
    # within this test's scope (if any were directly used, which it isn't) and the one
    # in _llm_client_core are the same mock, this explicit context manager can be kept,
    # but it should also target the correct logger.
    with patch('src.llm_wrapper_mcp_server.llm_client_parts._llm_client_core.logger', new=mock_logger): # Corrected patch target
        client = LLMClient(
            api_key=test_api_key,
            enable_logging=True,
            enable_audit_log=True,
            skip_outbound_key_checks=False # Ensure redaction is enabled
        )
        client.generate_response("test prompt")

        # Assert that the warning message was logged
        expected_calls = [
            call("Rate limiting is enabled but not yet implemented in LLMClient."),
            call("Redacting API key from response content")
        ]
        mock_logger.warning.assert_has_calls(expected_calls, any_order=True)
