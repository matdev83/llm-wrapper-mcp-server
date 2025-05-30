# tests/test_main_llm_wrapper.py
import pytest
from unittest.mock import patch, MagicMock, call
import os
import sys
import logging
import io
import gettext # Import gettext
# Ensure the main function from the script is importable
# If __main__.py is part of a package, this might need adjustment based on pythonpath
# For this structure, assuming src is in pythonpath or package is installed.
from llm_wrapper_mcp_server.__main__ import main as llm_wrapper_main

@pytest.fixture(autouse=True)
def manage_cwd():
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


@pytest.fixture
def mock_llm_mcp_wrapper_constructor(mocker):
    # Mocks the LLMMCPWrapper class constructor at its original definition
    mock_constructor = mocker.patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
    # Configure the mock constructor to return a mock instance
    mock_instance = mock_constructor.return_value
    mock_instance.run = MagicMock() # Mock the run method of the instance
    return mock_constructor, mock_instance

@pytest.fixture
def mock_dependencies(mocker, monkeypatch):
    # Mock os.makedirs to prevent actual directory creation
    monkeypatch.setattr(os, 'makedirs', MagicMock())
    # Mock logging.basicConfig
    mock_basic_config = mocker.patch('logging.basicConfig')
    # Mock getLogger, and its return value's methods like debug, warning, info, exception
    mock_get_logger = mocker.patch('logging.getLogger')
    mock_logger_instance = MagicMock()
    mock_get_logger.return_value = mock_logger_instance
    
    # Mock LLMClient within __main__ if it's imported there (it's not, wrapper is)
    # Mock os.path.exists for allowed_models_file if needed for a specific test
    mocker.patch('os.path.exists', return_value=True) # General mock for existence
    mocker.patch('builtins.open', mocker.mock_open(read_data="allowed/model")) # Mock open for allowed_models_file

    # Mock gettext.translation to prevent it from trying to read .mo files
    class MockTranslations:
        def gettext(self, message):
            return message
        def ngettext(self, singular, plural, n):
            return singular if n == 1 else plural

    mocker.patch('gettext.translation', return_value=MockTranslations())

    return {
        "basicConfig": mock_basic_config,
        "logger": mock_logger_instance,
        "exists": mocker.patch('os.path.exists'), # allow re-patching per test
        "open": mocker.patch('builtins.open') # allow re-patching per test
    }


def test_main_llm_wrapper_default_args(mock_llm_mcp_wrapper_constructor, mock_dependencies, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor
    
    # Mock sys.stdin to prevent OSError during server.run()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))

    with patch.object(sys, 'argv', ['__main__.py']):
        llm_wrapper_main()

    mock_constructor.assert_called_once()
    call_args = mock_constructor.call_args[1] # Get kwargs
    
    assert call_args['system_prompt_path'] == "config/prompts/system.txt"
    assert call_args['model'] == "perplexity/llama-3.1-sonar-small-128k-online"
    assert call_args['skip_accounting'] is False
    assert call_args['skip_outbound_key_checks'] is False # Default from argparse
    assert call_args['server_name'] == "llm-wrapper-mcp-server"
    # max_user_prompt_tokens is part of LLMMCPWrapper's __init__ defaults, not directly from __main__'s default args to constructor
    # Check __main__.py: limit_user_prompt_length is NOT passed to LLMMCPWrapper. This is a gap.
    # So, we expect LLMMCPWrapper to use its own default for max_user_prompt_tokens.
    # We can't directly check call_args['max_user_prompt_tokens'] unless __main__ passes it.
    mock_instance.run.assert_called_once()
    mock_dependencies["basicConfig"].assert_called_once()


def test_main_llm_wrapper_custom_args(mock_llm_mcp_wrapper_constructor, mock_dependencies, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor

    # Mock sys.stdin to prevent OSError during server.run()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))

    test_args = [
        '__main__.py',
        '--model', 'custom/model',
        '--system-prompt-file', 'custom_prompt.txt',
        '--skip-accounting',
        '--skip-outbound-key-leaks',
        '--server-name', 'MyTestServer',
        '--llm-api-base-url', 'https://custom.api',
        '--log-file', 'custom.log',
        '--log-level', 'DEBUG',
        '--max-tokens', '500'
        # Note: --limit-user-prompt-length is parsed by __main__ but not passed to LLMMCPWrapper constructor
    ]
    with patch.object(sys, 'argv', test_args):
        llm_wrapper_main()

    mock_constructor.assert_called_once()
    call_args = mock_constructor.call_args[1]
    assert call_args['model'] == 'custom/model'
    assert call_args['system_prompt_path'] == 'custom_prompt.txt'
    assert call_args['skip_accounting'] is True
    assert call_args['skip_outbound_key_checks'] is True
    assert call_args['server_name'] == 'MyTestServer'
    assert call_args['llm_api_base_url'] == 'https://custom.api'
    assert call_args['max_tokens'] == 500
    
    mock_instance.run.assert_called_once()
    # Check log configuration based on custom args
    mock_dependencies["basicConfig"].assert_called_once_with(
        filename='custom.log', # From --log-file
        level=logging.DEBUG,    # From --log-level
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

def test_main_llm_wrapper_cwd_change(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor
    
    new_cwd = tmp_path / "new_work_dir"
    new_cwd.mkdir()

    # Mock sys.stdin to prevent OSError during server.run()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))

    # Mock os.chdir to verify it's called and to see its effect
    with patch.object(os, 'chdir') as mock_chdir:
        with patch.object(sys, 'argv', ['__main__.py', '--cwd', str(new_cwd)]):
            llm_wrapper_main()
    
    mock_chdir.assert_called_once_with(str(new_cwd))
    # main() changes CWD, so subsequent tests might be affected if not reset.
    # The manage_cwd fixture handles resetting CWD.

def test_main_llm_wrapper_allowed_models_valid(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path, monkeypatch):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    
    # Mock sys.stdin to prevent OSError during server.run()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))

    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\ncustom/model")
    
    # Mock os.path.exists for the system prompt path to return False,
    # so LLMClient uses an empty system prompt and doesn't try to open the file.
    original_exists = os.path.exists
    def mock_os_path_exists(path):
        if path == "config/prompts/system.txt":
            return False
        return original_exists(path)
    monkeypatch.setattr(os.path, 'exists', mock_os_path_exists)

    mock_dependencies["exists"].return_value = True # Ensure file is seen as existing
    mock_dependencies["open"].return_value = io.StringIO(model_file.read_text())


    test_args = [
        '__main__.py',
        '--allowed-models-file', str(model_file),
        '--model', 'custom/model'
    ]
    with patch.object(sys, 'argv', test_args):
        llm_wrapper_main()
    
    mock_constructor.assert_called_once() # Should proceed to wrapper creation
    # Check that no warning about model not in list was logged
    assert not any(
        "not in the allowed models list" in call_args[0][0] 
        for call_args in mock_dependencies["logger"].warning.call_args_list
    )


def test_main_llm_wrapper_allowed_models_invalid_selection(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    
    model_file = tmp_path / "models.txt"
    model_file.write_text("allowed/model1\nallowed/model2")

    mock_dependencies["exists"].return_value = True
    mock_dependencies["open"].return_value = io.StringIO(model_file.read_text())

    test_args = [
        '__main__.py',
        '--allowed-models-file', str(model_file),
        '--model', 'forbidden/model'
    ]
    with patch.object(sys, 'argv', test_args), pytest.raises(SystemExit) as excinfo:
        llm_wrapper_main()
    
    assert excinfo.value.code == 1
    mock_dependencies["logger"].warning.assert_any_call(
        "Model 'forbidden/model' is not in the allowed models list"
    )
    mock_constructor.assert_not_called() # Server should not start

def test_main_llm_wrapper_allowed_models_file_not_found(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    
    missing_model_file = tmp_path / "nonexistent_models.txt"
    mock_dependencies["exists"].return_value = False # File does not exist

    test_args = [
        '__main__.py',
        '--allowed-models-file', str(missing_model_file)
    ]
    with patch.object(sys, 'argv', test_args), pytest.raises(SystemExit) as excinfo:
        llm_wrapper_main()
        
    assert excinfo.value.code == 1
    mock_dependencies["logger"].warning.assert_any_call(
        f"Allowed models file not found: {str(missing_model_file)}"
    )
    mock_constructor.assert_not_called()
