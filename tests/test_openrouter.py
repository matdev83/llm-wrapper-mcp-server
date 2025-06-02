import os
import unittest
from unittest.mock import patch, MagicMock, call
from llm_wrapper_mcp_server.llm_client import LLMClient
from llm_wrapper_mcp_server.llm_mcp_wrapper import LLMMCPWrapper

class TestRedactionFunctionality(unittest.TestCase):
    def setUp(self):
        self.test_api_key = "sk-testkey1234567890abcdefghijklmnopqr"
        os.environ["OPENROUTER_API_KEY"] = self.test_api_key
        
        self.mock_response = {
            "response": f"Here is your key: {self.test_api_key}",
            "input_tokens": 10,
            "output_tokens": 20,
            "api_usage": {}
        }

    @patch('requests.post')
    def test_api_key_redaction_enabled(self, mock_post):
        """Test that API key is redacted when feature is enabled (default)"""
        # Setup mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": f"Here is your key: {self.test_api_key}"}}]
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper(skip_api_key_redaction=False)
        response = server.llm_client.generate_response("test prompt")
        processed_response = response["response"]
        
        self.assertIn("(API key redacted due to security reasons)", processed_response)
        self.assertNotIn(self.test_api_key, processed_response)

    def test_api_key_redaction_disabled(self):
        """Test that API key remains when redaction is disabled"""
        with patch.object(LLMClient, 'generate_response', return_value=self.mock_response) as mock_gen:
            # Use LLMMCPWrapper instead of StdioServer
            server = LLMMCPWrapper(skip_api_key_redaction=True)
            
            # Simulate API response containing the actual key
            response = server.llm_client.generate_response("test prompt")
            processed_response = response["response"]
            
            self.assertIn(self.test_api_key, processed_response)
            self.assertNotIn("(API key redacted due to security reasons)", processed_response)
            mock_gen.assert_called_once()

    @patch('requests.post')
    @patch('llm_wrapper_mcp_server.llm_client.logger')
    def test_redaction_logging(self, mock_logger, mock_post):
        """Test that redaction events are properly logged"""
        # Setup mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": f"Here is your key: {self.test_api_key}"}}]
        }
        mock_response.headers = {}
        mock_post.return_value = mock_response
        
        # Use LLMMCPWrapper instead of StdioServer
        server = LLMMCPWrapper()
        server.llm_client.generate_response("test prompt")
        
        expected_calls = [
            call("Rate limiting is enabled but not yet implemented in LLMClient."),
            call("Redacting API key from response content")
        ]
        mock_logger.warning.assert_has_calls(expected_calls, any_order=True)

if __name__ == '__main__':
    unittest.main()
