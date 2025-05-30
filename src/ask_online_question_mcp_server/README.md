# Ask Online Question MCP Server (Reference Implementation)

This directory contains a reference implementation of a custom Model Context Protocol (MCP) server named "Ask Online Question". This server demonstrates how to build a specialized MCP server on top of the `llm-wrapper-mcp-server` foundation, exposing a single, focused tool. It operates as an STDIO MCP server, similar to the main `llm-wrapper-mcp-server`.

### Features:
*   **Dedicated Tool:** Exposes a single tool: `ask_online_question`.
*   **Simple Interface:** The `ask_online_question` tool takes one input parameter: `prompt` (string).
*   **Fixed Model Configuration:** The LLM model used for answering questions is configured when the server starts and is not exposed via the MCP interface.
*   **Customizable Model:** You can specify a custom LLM model via a CLI argument.
*   **Customizable System Prompt:** The system prompt can be overridden via a CLI argument, defaulting to `config/prompts/system.txt`.

### Usage:

The "Ask Online Question" server can be run in two ways:

1.  **From the project source (for development or local use):**
    Navigate to the root of the `llm-wrapper-mcp-server` project and run:
    ```bash
    python -m src.ask_online_question_mcp_server --llm-api-base-url https://api.openrouter.ai/api/v1
    ```

2.  **After installing the `llm-wrapper-mcp-server` package via `pip`:**
    If you have installed the main `llm-wrapper-mcp-server` package (e.g., `pip install llm-wrapper-mcp-server`), you can run the submodule directly as part of the installed package:
    ```bash
    python -m llm_wrapper_mcp_server.ask_online_question_mcp_server --llm-api-base-url https://api.openrouter.ai/api/v1
    ```

**Specifying a Custom Model:**

You can override the default model (`perplexity/llama-3.1-sonar-small-128k-online`) using the `--model` argument:

```bash
python -m src.ask_online_question_mcp_server --model "your_custom_model_name" --llm-api-base-url https://api.openrouter.ai/api/v1
```

### Recommended Models

This server is optimized for use with Perplexity's Sonar models, which offer excellent performance for factual queries.

*   **Default Model:** `perplexity/llama-3.1-sonar-small-128k-online`
    This model is suitable for most general questions and provides a good balance of speed and accuracy.

*   **For Complex Queries:** `perplexity/llama-3.1-sonar-large-128k-online`
    For more complex questions, queries that may generate longer output, or tasks requiring interpretation, rephrasing, or reformatting, it is highly recommended to use the `sonar-large` variant. The smaller `sonar-small` model may sometimes hallucinate or provide less accurate results in such scenarios.

    To run the MCP server configured with the larger Sonar model:

    ```bash
    python -m src.ask_online_question_mcp_server --model perplexity/llama-3.1-sonar-large-128k-online --llm-api-base-url https://api.openrouter.ai/api/v1
    ```

**Specifying a Custom System Prompt:**

The system prompt defaults to `config/prompts/system.txt`. You can provide a different system prompt file using the `--system-prompt-path` argument:

```bash
python -m src.ask_online_question_mcp_server --system-prompt-path "path/to/your/custom_prompt.txt" --llm-api-base-url https://api.openrouter.ai/api/v1
```

### Example MCP Client Interaction (Conceptual):

An MCP client would interact with this server by calling the `ask_online_question` tool:

```json
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "ask_online_question",
    "arguments": {
      "prompt": "What's the weather today in Berlin?"
    }
  }
}
```

### Agent Prompting for Optimal Use:

To optimally utilize the "Ask Online Question" MCP server, your agent software should include instructions to the executing LLM on when and how to use this server. This ensures the LLM intelligently delegates appropriate queries.

Here's a usable example prompt for an execution LLM:

```
You have access to a tool called `ask_online_question` which can answer factual questions by querying an online knowledge base.
Use this tool when you need to find up-to-date information, verify facts, perform online searches, or answer questions that require external knowledge beyond your training data.

**Tool Name:** `ask_online_question`
**Input:** `prompt` (string) - The question you want to ask.

**Examples of when to use `ask_online_question`:**
- "What is the current population of Tokyo?"
- "Who won the latest Nobel Prize in Physics?"
- "Explain the concept of quantum entanglement."
- "What are the main features of Python 3.10?"

**Example of tool usage:**
<tool_code>
print(ask_online_question(prompt="What's the current weather in the capital of Australia?"))
</tool_code>

**Avoid overusing** `ask_online_question` if you can fully address the original user's prompt without relying on an external knowledge base or live web search.

**Avoid using `ask_online_question` for:**
- Creative writing tasks.
- Personal opinions or subjective interpretations.
- Questions that can be answered directly from your internal knowledge base (if applicable).
- Tasks that require complex reasoning or multi-step problem-solving that cannot be resolved by a single factual query.

If you opt to use `ask_online_question` mase sure you always provide the full question or keywords for online search as the `prompt` argument.
```
