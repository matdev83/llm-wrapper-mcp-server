# Project Structure

This document outlines the directory and file structure of the `llm-wrapper-mcp-server` project.

```
.
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── config/
│   └── prompts/
│       └── system.txt
├── data/
├── docs/
│   └── STRUCTURE.md
├── logs/
├── src/
│   ├── llm_wrapper_mcp_server/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── llm_client.py
│   │   ├── llm_mcp_server.py
│   │   ├── llm_mcp_wrapper.py
│   │   └── logger.py
│   └── ask_online_question_mcp_server/
│       ├── __init__.py
│       ├── __main__.py
│       └── ask_online_question_server.py
├── tests/
│   ├── test_llm_client.py
│   ├── test_llm_mcp_wrapper.py
│   ├── test_model_validation.py
│   └── test_openrouter.py
└── version_manager.py
└── release_orchestrator.py
└── build.bat
```

### Directory Descriptions:

*   `.`: The root directory of the project.
*   `config/`: Contains configuration files for the application.
    *   `prompts/`: Stores system prompts used by the LLM.
*   `data/`: Intended for storing any data files generated or used by the application.
*   `docs/`: Contains project documentation, including this structure description.
*   `logs/`: Stores application log files.
*   `src/`: Contains the source code of the application.
    *   `llm_wrapper_mcp_server/`: The main Python package for the LLM wrapper MCP server.
    *   `ask_online_question_mcp_server/`: A reference implementation of a custom MCP server, demonstrating how to extend the `llm-wrapper-mcp-server` foundation.
*   `tests/`: Contains unit and integration tests for the project.

### File Descriptions:

*   `.gitignore`: Specifies intentionally untracked files to ignore by Git.
*   `CHANGELOG.md`: Documents all notable changes to the project.
*   `LICENSE`: Contains the licensing information for the project.
*   `pyproject.toml`: Project configuration file, including build system and dependencies.
*   `README.md`: Provides a general overview of the project, setup instructions, and usage.
*   `requirements.txt`: Lists the Python dependencies required for the project.
*   `src/llm_wrapper_mcp_server/__init__.py`: Initializes the `llm_wrapper_mcp_server` Python package.
*   `src/llm_wrapper_mcp_server/__main__.py`: Entry point for running the package as a script.
*   `src/llm_wrapper_mcp_server/llm_client.py`: Handles interactions with LLM APIs.
*   `src/llm_wrapper_mcp_server/llm_mcp_server.py`: Implements the MCP server logic for the LLM wrapper.
*   `src/llm_wrapper_mcp_server/llm_mcp_wrapper.py`: Wraps LLM functionalities for MCP integration.
*   `src/llm_wrapper_mcp_server/logger.py`: Configures and provides logging utilities.
*   `src/ask_online_question_mcp_server/__init__.py`: Initializes the `ask_online_question_mcp_server` Python package.
*   `src/ask_online_question_mcp_server/__main__.py`: Entry point for running the `ask_online_question_mcp_server` as a script.
*   `src/ask_online_question_mcp_server/ask_online_question_server.py`: Implements the "Ask Online Question" MCP server, exposing the `ask_online_question` tool.
*   `tests/test_llm_client.py`: Tests for the `llm_client.py` module.
*   `tests/test_llm_mcp_wrapper.py`: Tests for the `llm_mcp_wrapper.py` module.
*   `tests/test_model_validation.py`: Tests for model validation logic.
*   `tests/test_openrouter.py`: Tests specific to the OpenRouter LLM integration.
