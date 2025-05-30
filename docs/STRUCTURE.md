# Project Structure

This document outlines the directory and file structure of the `llm-wrapper-mcp-server` project.

```
.
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── README.md
├── config/
│   └── prompts/
│       └── system.txt
├── data/
│   └── .gitkeep # (or actual db files if they exist and are not gitignored)
├── docs/
│   └── STRUCTURE.md
├── logs/
│   └── .gitkeep
├── src/
│   ├── llm_wrapper_mcp_server/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── llm_client.py
│   │   ├── llm_mcp_wrapper.py
│   │   └── logger.py
│   └── ask_online_question_mcp_server/  # Reference implementation
│       ├── __init__.py
│       ├── __main__.py
│       ├── README.md
│       └── ask_online_question_server.py
├── tests/
│   ├── fixtures/  # Recommended for test fixtures
│   │   └── __init__.py # Make it a package
│   │   └── system_prompt.txt # Example, based on test_llm_client.py
│   ├── test_llm_client.py
│   ├── test_llm_mcp_wrapper.py
│   ├── test_model_validation.py
│   └── test_openrouter.py
```

### Directory Descriptions:

*   `.`: The root directory of the project.
*   `config/`: Contains configuration files for the application.
    *   `prompts/`: Stores system prompts used by the LLM.
*   `data/`: Intended for storing any data files generated or used by the application, such as SQLite databases for `llm-accounting`.
*   `docs/`: Contains project documentation, including this structure description.
*   `logs/`: Stores application log files.
*   `src/`: Contains the source code of the application.
    *   `llm_wrapper_mcp_server/`: The main Python package for the LLM wrapper MCP server.
    *   `ask_online_question_mcp_server/`: A reference implementation of a custom MCP server.
*   `tests/`: Contains unit and integration tests for the project.
    *   `fixtures/`: Recommended location for test fixtures.

### Key File Descriptions:

*   `.gitignore`: Specifies intentionally untracked files to ignore by Git.
*   `CHANGELOG.md`: Documents all notable changes to the project.
*   `LICENSE`: Contains the licensing information for the project.
*   `pyproject.toml`: Project configuration file, including build system and dependencies.
*   `README.md`: Provides a general overview of the project, setup instructions, and usage.
*   `src/llm_wrapper_mcp_server/__init__.py`: Initializes the `llm_wrapper_mcp_server` Python package.
*   `src/llm_wrapper_mcp_server/__main__.py`: Entry point for running the package as a script, handles CLI argument parsing.
*   `src/llm_wrapper_mcp_server/llm_client.py`: Handles interactions with LLM APIs and includes accounting.
*   `src/llm_wrapper_mcp_server/llm_mcp_wrapper.py`: Implements the MCP server logic for the LLM wrapper.
*   `src/llm_wrapper_mcp_server/logger.py`: Configures and provides logging utilities.
*   `src/ask_online_question_mcp_server/...`: Files for the reference MCP server implementation.
*   `tests/...`: Test files corresponding to different modules.
```
