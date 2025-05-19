"""
Simple test script for OpenRouter API communication using chatlet.
"""
import os
from llm_delegate_mcp_server.llm_client import LLMClient
from unittest.mock import Mock
import pytest
import tiktoken

@pytest.fixture
def client():
    return LLMClient()

@pytest.fixture
def mock_encoder(mocker):
    """Fixture to mock tiktoken encoder"""
    mock = Mock()
    mocker.patch.object(tiktoken, 'get_encoding', return_value=mock)
    return mock

def test_openrouter_response(client, mock_encoder):
    """Test OpenRouter API response structure"""
    prompt = "What is 2+2?"
    mock_encoder.encode.side_effect = lambda x: len(x.split())  # Simple token approximation
    
    response = client.generate_response(prompt)
    
    assert isinstance(response, dict)
    assert "response" in response
    assert isinstance(response["response"], str)
    assert len(response["response"]) > 0
    assert "input_tokens" in response
    assert "output_tokens" in response
    assert isinstance(response["input_tokens"], int)
    assert isinstance(response["output_tokens"], int)

def test_token_counting(client, mock_encoder, mocker):
    """Test token counting functionality"""
    prompt = "Test prompt"
    system_prompt = "Test system prompt"
    mock_response = {"choices": [{"message": {"content": "Test response"}}]}
    
    # Mock the encoder and requests
    mock_encoder.encode.side_effect = lambda x: len(x.split())  # Simple token approximation
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.raise_for_status.return_value = None
    
    # Mock logger and system prompt
    mock_logger = mocker.patch('llm_delegate_mcp_server.llm_client.logger')
    mocker.patch('builtins.open', mocker.mock_open(read_data=system_prompt))
    client.system_prompt = system_prompt  # Set directly since we're testing token counting
    
    # Test
    response = client.generate_response(prompt)
    
    # Verify token counts
    expected_input = len(system_prompt.split()) + len(prompt.split())
    expected_output = len(mock_response["choices"][0]["message"]["content"].split())
    
    assert response["input_tokens"] == expected_input
    assert response["output_tokens"] == expected_output
    
    # Verify debug logging
    mock_logger.debug.assert_any_call(
        "Token counts - system: %d, user: %d, total: %d",
        len(system_prompt.split()),
        len(prompt.split()),
        expected_input
    )
    mock_logger.debug.assert_any_call(
        "Response token count: %d",
        expected_output
    )

def test_empty_prompt_token_count(client, mock_encoder, mocker):
    """Test token counting with empty prompts"""
    prompt = ""
    system_prompt = ""
    mock_response = {"choices": [{"message": {"content": ""}}]}
    
    # Mock the encoder, requests and file reading
    mock_encoder.encode.return_value = 0
    mock_post = mocker.patch('requests.post')
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.raise_for_status.return_value = None
    mocker.patch('builtins.open', mocker.mock_open(read_data=system_prompt))
    client.system_prompt = system_prompt  # Set directly since we're testing token counting
    
    # Test
    response = client.generate_response(prompt)
    
    # Verify token counts
    assert response["input_tokens"] == 0
    assert response["output_tokens"] == 0
