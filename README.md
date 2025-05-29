# LLM Wrapper MCP Server

> "Allow any MCP-capable LLM agent to communicate with or delegate tasks to any other LLM available through the OpenRouter.ai API."

A Model Context Protocol (MCP) server wrapper designed to facilitate seamless interaction with various Large Language Models (LLMs) through a standardized interface. This project enables developers to integrate LLM capabilities into their applications by providing a robust and flexible server that handles LLM calls, tool execution, and result processing.

## Features

- Implements the Model Context Protocol (MCP) specification for standardized LLM interactions.
- Provides a FastAPI-based server for handling LLM requests and responses.
- Supports advanced features like tool calls and results through the MCP protocol.
- Configurable to use various LLM providers (e.g., OpenRouter, local models).
- Designed for extensibility, allowing easy integration of new LLM backends.
- Integrates with `llm-accounting` for robust logging, rate limiting, and audit functionalities, enabling monitoring of remote LLM usage, inference costs, and inspection of queries/responses for debugging or legal purposes.

## Installation

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

Create a `.env` file in the project root with the following variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# Optional: Override default model
OPENROUTER_MODEL=your_preferred_model
```

The server is configured to use OpenRouter by default with the following settings:
- API Base URL: https://openrouter.ai/api/v1
- Default Model: perplexity/llama-3.1-sonar-small-128k-online

## Usage

### Running the Server

To run the server, execute the following command:

```bash
python -m llm_wrapper_mcp_server
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

- `POST /ask`: Main endpoint for LLM requests
- `GET /health`: Health check endpoint

### Client Code Examples

The `llm-wrapper-mcp-server` package can be used by client applications to create their own MCP servers and interact with remote LLM models. Here's an example of how to set up a basic client:

```python
from llm_wrapper_mcp_server.llm_mcp_server import LLMMCPWrapperServer
from llm_wrapper_mcp_server.llm_client import LLMClient

# Initialize the LLM MCP Wrapper Server
# This server will handle communication with the actual LLM provider
llm_server = LLMMCPWrapperServer()

# Initialize the LLM Client
# This client can be used by your application to send requests to the LLM server
llm_client = LLMClient(server_url="http://localhost:8000") # Assuming your server is running locally

async def main():
    # Example: Ask the LLM a question
    response = await llm_client.ask("What is the capital of France?")
    print(f"LLM Response: {response}")

    # Example: Use a tool (if supported by the LLM and configured)
    # This is a simplified example, actual tool usage depends on your MCP server's capabilities
    tool_response = await llm_client.use_tool("calculator", {"expression": "2+2"})
    print(f"Tool Response: {tool_response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### CLI Mode Usage

The `llm-wrapper-mcp-server` can also be used directly from the command line for quick interactions or testing.

**Basic Query:**

```bash
python -m llm_wrapper_mcp_server --query "Tell me a short story about a robot."
```

**Query with Model Specification:**

```bash
python -m llm_wrapper_mcp_server --query "What is the square root of 144?" --model "perplexity/llama-3.1-sonar-small-128k-online"
```

**Query with Tool Call (if configured):**

```bash
python -m llm_wrapper_mcp_server --query "Calculate 15 * 3." --tool "calculator" --tool-args '{"expression": "15 * 3"}'
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License
