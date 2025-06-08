# tests/test_main_llm_wrapper.py
import pytest
from unittest.mock import patch, MagicMock, call
import os
import sys
import logging
import io
import gettext
from llm_wrapper_mcp_server.__main__ import main as llm_wrapper_main

@pytest.fixture(autouse=True)
def manage_cwd():
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


@pytest.fixture
def mock_llm_mcp_wrapper_constructor(mocker):
    mock_constructor = mocker.patch('llm_wrapper_mcp_server.llm_mcp_wrapper.LLMMCPWrapper')
    mock_instance = mock_constructor.return_value
    mock_instance.run = MagicMock()
    return mock_constructor, mock_instance

@pytest.fixture
def mock_dependencies(mocker, monkeypatch):
    monkeypatch.setattr(os, 'makedirs', MagicMock())
    mock_basic_config = mocker.patch('logging.basicConfig')
    
    # Patch the logger instance directly in the __main__ module
    mock_logger_instance = mocker.patch('llm_wrapper_mcp_server.__main__.logger')

    original_os_path_exists = os.path.exists
    def default_exists_side_effect(path):
        if "config/prompts/system.txt" in str(path):
            return False
        return original_os_path_exists(path)

    mock_path_exists = mocker.patch('os.path.exists', side_effect=default_exists_side_effect)
    mock_file_open = mocker.patch('builtins.open', mocker.mock_open(read_data="default_model_content"))

    class MockTranslations:
        def gettext(self, message): return message
        def ngettext(self, s, p, n): return s if n == 1 else p
    mocker.patch('gettext.translation', return_value=MockTranslations())

    return {
        "basicConfig": mock_basic_config,
        "logger": mock_logger_instance,
        "exists": mock_path_exists,
        "open": mock_file_open
    }


def test_main_llm_wrapper_default_args(mock_llm_mcp_wrapper_constructor, mock_dependencies, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    with patch.object(sys, 'argv', ['__main__.py']):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-dummykeyfortests12345678901234"}):
            llm_wrapper_main()

    mock_constructor.assert_called_once()
    call_args = mock_constructor.call_args[1]
    assert call_args['system_prompt_path'] == "config/prompts/system.txt"
    assert call_args['model'] == "perplexity/llama-3.1-sonar-small-128k-online"
    mock_instance.run.assert_called_once()
    # basicConfig is called by _configure_logging, which uses the global logger from __main__
    # but basicConfig itself is a global logging function.
    mock_dependencies["basicConfig"].assert_called_once()


def test_main_llm_wrapper_custom_args(mock_llm_mcp_wrapper_constructor, mock_dependencies, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    test_args = [
        '__main__.py', '--model', 'custom/model', '--system-prompt-file', 'custom_prompt.txt',
        '--skip-outbound-key-leaks', '--server-name', 'MyTestServer',
        '--llm-api-base-url', 'https://custom.api', '--log-file', 'custom.log',
        '--log-level', 'DEBUG', '--max-tokens', '500', '--disable-logging',
        '--disable-audit-log', '--disable-rate-limiting'
    ]
    with patch.object(sys, 'argv', test_args):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-dummykeyfortests12345678901234"}):
            llm_wrapper_main()

    mock_constructor.assert_called_once()
    call_args = mock_constructor.call_args[1]
    assert call_args['model'] == 'custom/model'
    assert call_args['skip_outbound_key_checks'] is True
    mock_instance.run.assert_called_once()
    mock_dependencies["basicConfig"].assert_called_once_with(
        filename='custom.log', level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s', filemode='a'
    )

def test_main_llm_wrapper_cwd_change(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path, monkeypatch):
    mock_constructor, mock_instance = mock_llm_mcp_wrapper_constructor
    new_cwd = tmp_path / "new_work_dir"
    new_cwd.mkdir()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    with patch.object(os, 'chdir') as mock_chdir, \
         patch.object(sys, 'argv', ['__main__.py', '--cwd', str(new_cwd)]):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-dummykeyfortests12345678901234"}):
            llm_wrapper_main()
    mock_chdir.assert_called_once_with(str(new_cwd))

def test_main_llm_wrapper_allowed_models_valid(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path, monkeypatch):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
    model_file = tmp_path / "models.txt"
    model_file.write_text("perplexity/llama-3.1-sonar-small-128k-online\ncustom/model")
    
    mock_dependencies["exists"].side_effect = lambda p: True if p == str(model_file) else (False if "config/prompts/system.txt" in str(p) else True)
    mock_dependencies["open"].return_value = io.StringIO(model_file.read_text())

    test_args = ['__main__.py', '--allowed-models-file', str(model_file), '--model', 'custom/model']
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-dummykeyfortests12345678901234"}), \
         patch.object(sys, 'argv', test_args):
        llm_wrapper_main()
    
    mock_constructor.assert_called_once()
    found_warning = any("not in the allowed models list" in str(call_item) for call_item in mock_dependencies["logger"].warning.call_args_list)
    assert not found_warning, "Warning about model not in list should not have been logged."


def test_main_llm_wrapper_allowed_models_invalid_selection(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    model_file = tmp_path / "models.txt"
    model_file.write_text("allowed/model1\nallowed/model2")

    # Configure the 'exists' mock for this specific test
    # The allowed_models_file should exist, system_prompt might not (handled by LLMClient)
    mock_dependencies["exists"].side_effect = lambda p: True if p == str(model_file) else (False if "config/prompts/system.txt" in str(p) else True)
    mock_dependencies["open"].return_value = io.StringIO(model_file.read_text())

    test_args = ['__main__.py', '--allowed-models-file', str(model_file), '--model', 'forbidden/model']
    with patch.object(sys, 'argv', test_args), pytest.raises(SystemExit) as excinfo:
        llm_wrapper_main() # OPENROUTER_API_KEY is not needed here as LLMMCPWrapper is not instantiated
    
    assert excinfo.value.code == 1
    expected_log_message = f"Model 'forbidden/model' is not in the allowed models list from {model_file}"
    # Using assert_any_call directly as the logger instance should now be correctly patched
    mock_dependencies["logger"].warning.assert_any_call(expected_log_message)
    mock_constructor.assert_not_called()

def test_main_llm_wrapper_allowed_models_file_not_found(mock_llm_mcp_wrapper_constructor, mock_dependencies, tmp_path):
    mock_constructor, _ = mock_llm_mcp_wrapper_constructor
    missing_model_file = tmp_path / "nonexistent_models.txt"

    # Configure the 'exists' mock for this specific test
    mock_dependencies["exists"].side_effect = lambda p: False if p == str(missing_model_file) else True

    test_args = ['__main__.py', '--allowed-models-file', str(missing_model_file)]
    with patch.object(sys, 'argv', test_args), pytest.raises(SystemExit) as excinfo:
        llm_wrapper_main() # OPENROUTER_API_KEY is not needed here
        
    assert excinfo.value.code == 1
    expected_log_message = f"Allowed models file not found: {str(missing_model_file)}"
    mock_dependencies["logger"].warning.assert_any_call(expected_log_message)
    mock_constructor.assert_not_called()
