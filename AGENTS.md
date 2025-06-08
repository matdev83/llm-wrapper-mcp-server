# Agent Guidelines and Project Structure

This document serves as a comprehensive guide for software development agents working on the `llm-wrapper-mcp-server` project. It outlines the project's directory and file structure, and more importantly, establishes the core software development principles that all agents MUST adhere to.

## Software Development Principles for Agents

All agents contributing to this project are required to strictly follow these principles to ensure code quality, maintainability, and scalability:

1. **Layered, Modular Architecture:**
    * Design and implement solutions with clear separation of concerns.
    * Break down complex functionalities into smaller, independent, and reusable modules.
    * Each layer and module should have a well-defined responsibility and interface.
    * Minimize coupling between modules and maximize cohesion within modules.

2. **Pythonic Conventions and Standards:**
    * Adhere to PEP 8 style guide for Python code.
    * Write clean, readable, and idiomatic Python code.
    * Utilize Python's built-in features and standard library effectively.
    * Ensure consistent naming conventions for variables, functions, classes, and files.

3. **TDD - Test-Driven Development:**
    * **Every enhanced, changed, or added functionality MUST be covered by related tests.**
    * Write tests before writing the production code.
    * Ensure tests are comprehensive, covering various scenarios including edge cases.
    * Tests should be automated, repeatable, and fast.
    * Agents are strictly forbidden from introducing any changes that are not covered by extensive tests ensuring proper project maintenance.

4. **Software Architecture Principles:**
    * **TDD (Test-Driven Development):** As detailed above, tests drive development.
    * **SOLID Principles:**
        * **S**ingle Responsibility Principle: Each class or module should have only one reason to change.
        * **O**pen/Closed Principle: Software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification.
        * **L**iskov Substitution Principle: Objects in a program should be replaceable with instances of their subtypes without altering the correctness of that program.
        * **I**nterface Segregation Principle: Clients should not be forced to depend on interfaces they do not use.
        * **D**ependency Inversion Principle: Depend upon abstractions, not concretions.
    * **KISS (Keep It Simple, Stupid):** Strive for simplicity in design and implementation. Avoid unnecessary complexity.
    * **DRY (Don't Repeat Yourself):** Avoid duplication of code and knowledge. Abstract common functionalities into reusable components.

## Project Structure

```bash
.
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── AGENTS.md # This file
├── config/
│   └── prompts/
│       └── system.txt
├── data/
│   └── .gitkeep # (or actual db files if they exist and are not gitignored)
├── docs/
│   └── .gitkeep # (or other documentation files)
├── logs/
│   └── .gitkeep
├── src/
│   ├── llm_wrapper_mcp_server/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── llm_client.py
│   │   ├── llm_mcp_server.py
│   │   ├── llm_mcp_wrapper.py
│   │   ├── logger.py
│   │   └── llm_client_parts/
│   │       ├── _accounting.py
│   │       ├── _api_key_filter.py
│   │       ├── _config.py
│   │       └── _llm_client_core.py
│   ├── ask_online_question_mcp_server/  # Reference implementation
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── README.md
│   │   └── ask_online_question_server.py
│   └── online_llm_server/ # New specialized MCP server
│       └── online_server.py
├── tests/
│   ├── conftest.py
│   ├── test_ask_online_question_server.py
│   ├── test_integration_openrouter.py
│   ├── test_llm_client.py
│   ├── test_llm_mcp_wrapper.py
│   ├── test_main_llm_wrapper.py
│   ├── test_model_validation.py
│   └── test_openrouter.py
```

### Directory Descriptions

* `.`: The root directory of the project.
* `config/`: Contains configuration files for the application.
  * `prompts/`: Stores system prompts used by the LLM.
* `data/`: Intended for storing any data files generated or used by the application, such as SQLite databases for `llm-accounting`.
* `docs/`: Contains project documentation. This directory should now primarily contain user-facing documentation or other non-agent specific documents.
* `logs/`: Stores application log files.
* `src/`: Contains the source code of the application.
  * `llm_wrapper_mcp_server/`: The main Python package for the LLM wrapper MCP server.
  * `ask_online_question_mcp_server/`: A reference implementation of a custom MCP server.
  * `online_llm_server/`: A specialized MCP server for online LLM queries, providing a `search_online` tool.
* `tests/`: Contains unit and integration tests for the project.
  * `llm_client_parts/`: Contains modularized components of the `llm_client`.

### Key File Descriptions

* `.gitignore`: Specifies intentionally untracked files to ignore by Git.
* `LICENSE`: Contains the licensing information for the project.
* `pyproject.toml`: Project configuration file, including build system and dependencies.
* `README.md`: Provides a general overview of the project, setup instructions, and usage.
* `AGENTS.md`: This file, providing guidelines and project structure specifically for software development agents.
* `src/llm_wrapper_mcp_server/__init__.py`: Initializes the `llm_wrapper_mcp_server` Python package.
* `src/llm_wrapper_mcp_server/__main__.py`: Entry point for running the package as a script, handles CLI argument parsing.
* `src/llm_wrapper_mcp_server/llm_client.py`: Handles interactions with LLM APIs and includes accounting.
* `src/llm_wrapper_mcp_server/llm_mcp_server.py`: Implements the core MCP server functionality.
* `src/llm_wrapper_mcp_server/llm_mcp_wrapper.py`: Implements the MCP server logic for the LLM wrapper.
* `src/llm_wrapper_mcp_server/logger.py`: Configures and provides logging utilities.
* `src/llm_wrapper_mcp_server/llm_client_parts/_accounting.py`: Handles usage accounting for LLM calls.
* `src/llm_wrapper_mcp_server/llm_client_parts/_api_key_filter.py`: Filters and manages API keys.
* `src/llm_wrapper_mcp_server/llm_client_parts/_config.py`: Manages LLM client configuration.
* `src/llm_wrapper_mcp_server/llm_client_parts/_llm_client_core.py`: Core logic for interacting with LLM APIs.
* `src/ask_online_question_mcp_server/...`: Files for the reference MCP server implementation.
* `src/online_llm_server/online_server.py`: The main script for the specialized online LLM MCP server.
* `tests/conftest.py`: Pytest file for test configuration and fixtures.
* `tests/test_ask_online_question_server.py`: Tests for the `ask_online_question_mcp_server`.
* `tests/test_integration_openrouter.py`: Integration tests for OpenRouter API.
* `tests/test_llm_client.py`: Tests for the `llm_client` module.
* `tests/test_llm_mcp_wrapper.py`: Tests for the `llm_mcp_wrapper` module.
* `tests/test_main_llm_wrapper.py`: Tests for the main LLM wrapper entry point.
* `tests/test_model_validation.py`: Tests for model validation logic.
* `tests/test_openrouter.py`: Unit tests for OpenRouter specific functionalities.
