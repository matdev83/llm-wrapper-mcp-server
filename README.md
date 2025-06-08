# LLM Wrapper MCP Server

> "Allow any MCP-capable LLM agent to communicate with or delegate tasks to any other LLM available through the OpenRouter.ai API."

[GitHub Repository](https://github.com/matdev83/llm-wrapper-mcp-server)

A Model Context Protocol (MCP) server wrapper designed to facilitate seamless interaction with various Large Language Models (LLMs) through a standardized interface. This project enables developers to integrate LLM capabilities into their applications by providing a robust and flexible STDIO-based server that handles LLM calls, tool execution, and result processing.

## Features

- Implements the Model Context Protocol (MCP) specification for standardized LLM interactions.
- Provides an STDIO-based server for handling LLM requests and responses via standard input/output.
- Supports advanced features like tool calls and results through the MCP protocol.
- Configurable to use various LLM providers (e.g., OpenRouter, local models) via API base URL and model parameters.
- Designed for extensibility, allowing easy integration of new LLM backends.
- Integrates with `llm-accounting` for robust logging, rate limiting, and audit functionalities, enabling monitoring of remote LLM usage, inference costs, and inspection of queries/responses for debugging or legal purposes.

## Dependencies

This project relies on the following key dependencies:

### Core Dependencies

- `pydantic`: Data validation and settings management using Python type hints.

- `pydantic-settings`: Pydantic's settings management for environment variables and configuration.
- `python-dotenv`: Reads key-value pairs from a `.env` file and sets them as environment variables.
- `requests`: An elegant and simple HTTP library for Python.
- `tiktoken`: A fast BPE tokeniser for use with OpenAI's models.
- `llm-accounting`: For robust logging, rate limiting, and audit functionalities.

*(Note: `fastapi` and `uvicorn` have been removed as the primary server is STDIO-based. If these are used for other utilities within the project, they should be re-added with clarification.)*

### Development Dependencies

- `pytest`: A mature full-featured Python testing framework.

- `black`: An uncompromising Python code formatter.
- `isort`: A Python utility / library to sort imports alphabetically, and automatically separate into sections and by type.
- `mypy`: An optional static type checker for Python.
- `pytest-mock`: A pytest plugin that provides a `mocker` fixture for easier mocking.

## Installation

The `llm-wrapper-mcp-server` package is available on PyPI and can be installed via pip:

```bash
pip install llm-wrapper-mcp-server
```

Alternatively, for local development or to install from source:

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package:

```bash
pip install -e .
```

## Configuration

Create a `.env` file in the project root with the following variable:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The server is configured to use OpenRouter by default. The API key is loaded from the `OPENROUTER_API_KEY` environment variable. The specific LLM model and API base URL are primarily configured via command-line arguments when running the server (see below).

Default settings if not overridden by CLI arguments:

- API Base URL for LLMClient: <https://openrouter.ai/api/v1> (can be overridden by `LLM_API_BASE_URL` env var or `--llm-api-base-url` CLI arg)
- Default Model for LLMClient: perplexity/llama-3.1-sonar-small-128k-online (can be overridden by `--model` CLI arg)

## Usage

**Textual Overview:**

- Agent Software communicates with the LLM Wrapper MCP Server via the MCP Protocol (stdin/stdout).
- The LLM Wrapper MCP Server interacts with LLM providers (e.g., OpenRouter.ai) for LLM API calls and responses.
- The server also integrates with an LLM Accounting System for logging and auditing.
- Main components:
  - MCP Communication Handler
  - LLM Client
  - Tool Executor
  - LLM Accounting Integration

### Ask Online Question MCP Server (Reference Implementation)

This project includes a reference implementation of a fully functional MCP server named "Ask Online Question".

It can be directly integrated into your agentic workflows, providing cloud-based, LLM-powered online search capabilities via the MCP protocol.  

This server demonstrates how to build a specialized MCP server on top of the `llm-wrapper-mcp-server` foundation. For detailed information on its features, usage, and how to integrate it with your agent, please refer to its dedicated README: [src/ask_online_question_mcp_server/README.md](src/ask_online_question_mcp_server/README.md).

### Running the Server

To run the server, execute the following command:

```bash
python -m llm_wrapper_mcp_server [OPTIONS]
```

For example:

```bash
python -m llm_wrapper_mcp_server --model your-org/your-model-name --log-level DEBUG
```

Run `python -m llm_wrapper_mcp_server --help` to see all available command-line options for configuring the server.

This server operates as a Model Context Protocol (MCP) STDIO server, communicating via standard input and output. It does not open a network port for MCP communication.

### MCP Communication

The server communicates using JSON-RPC messages over `stdin` and `stdout`. It supports the following MCP methods:

- `initialize`: Handshake to establish protocol version and server capabilities.
- `tools/list`: Lists available tools. The main server provides an `llm_call` tool.
- `tools/call`: Executes a specified tool.
- `resources/list`: Lists available resources (currently none).
- `resources/templates/list`: Lists available resource templates (currently none).

The `llm_call` tool takes `prompt` (string, required) and optionally `model` (string) as arguments to allow per-call model overrides if the specified model is permitted.

### Client Interaction Example (Python)

You can interact with the STDIO MCP server using any language that supports standard input/output communication. Here's a Python example using the `subprocess` module:

```python
import subprocess
import json
import time

def send_request(process, request):
    """Sends a JSON-RPC request to the server's stdin."""
    request_str = json.dumps(request) + "\\n"
    process.stdin.write(request_str.encode('utf-8'))
    process.stdin.flush()

def read_response(process):
    """Reads a JSON-RPC response from the server's stdout."""
    line = process.stdout.readline().decode('utf-8').strip()
    if line:
        return json.loads(line)
    return None

if __name__ == "__main__":
    # Start the MCP server as a subprocess
    # Ensure you have the virtual environment activated or the package installed globally
    server_process = subprocess.Popen(
        ["python", "-m", "llm_wrapper_mcp_server"], # Add any CLI args here if needed
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, # Capture stderr for debugging
        text=False # Use bytes for stdin/stdout
    )

    print("Waiting for server to initialize...")
    # The server sends an initial capabilities message on startup (id: None)
    initial_response = read_response(server_process)
    print(f"Server Initial Response: {json.dumps(initial_response, indent=2)}")

    # 1. Send an 'initialize' request
    initialize_request = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "initialize",
        "params": {}
    }
    print("\\nSending initialize request...")
    send_request(server_process, initialize_request)
    initialize_response = read_response(server_process)
    print(f"Initialize Response: {json.dumps(initialize_response, indent=2)}")

    # 2. Send a 'tools/call' request to use the 'llm_call' tool
    llm_call_request = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "tools/call",
        "params": {
            "name": "llm_call",
            "arguments": {
                "prompt": "What is the capital of France?"
                # Optionally add: "model": "another-model/if-allowed"
            }
        }
    }
    print("\\nSending llm_call request...")
    send_request(server_process, llm_call_request)
    llm_call_response = read_response(server_process)
    print(f"LLM Call Response: {json.dumps(llm_call_response, indent=2)}")

    # You can also read stderr for any server logs/errors
    # Note: stderr might block if there's no output, consider using non-blocking reads or threads for real apps
    # stderr_output = server_process.stderr.read().decode('utf-8')
    # if stderr_output:
    #     print("\\nServer Stderr Output:\\n", stderr_output)

    # Terminate the server process
    server_process.terminate()
    server_process.wait(timeout=5) # Wait for process to terminate
    print("\\nServer process terminated.")
```

## Development

For a detailed overview of the project's directory and file structure, and crucial guidelines for software development agents, refer to [AGENTS.md](AGENTS.md). This document is essential for agents contributing to the codebase.

### Running Tests

This project uses `pytest` for testing.

To run all unit tests:

```bash
pytest
```

Integration tests are disabled by default to avoid making external API calls during normal test runs. To include and run integration tests, use the `integration` marker:

```bash
pytest -m integration
```

### Install Development Dependencies

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License
