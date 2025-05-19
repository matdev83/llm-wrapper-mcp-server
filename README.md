# LLM Delegate MCP Server

A Model Context Protocol (MCP) server wrapper for remote LLM calls using OpenRouter. This server implements the MCP protocol to provide a standardized interface for LLM interactions.

## Features

- Implements the Model Context Protocol (MCP) specification
- Provides a FastAPI-based server for handling LLM requests
- Supports tool calls and results through the MCP protocol
- Uses OpenRouter as the default LLM provider
- Configurable through environment variables

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

Run the server:

```bash
python -m llm_delegate_mcp_server
```

The server will start on `http://localhost:8000` by default.

### API Endpoints

- `POST /ask`: Main endpoint for LLM requests
- `GET /health`: Health check endpoint

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

## License

MIT License
