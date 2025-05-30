# CHANGELOG

### feat: Introduce Ask Online Question MCP Server reference implementation and update core server logic

- Add a new `ask_online_question_mcp_server` as a reference implementation for custom MCP servers.
- Update `README.md` and `docs/STRUCTURE.md` to document the new server.
- Refactor `llm_mcp_server.py` to use `LLMMCPWrapper` directly for server instantiation.
- Revamp `config/prompts/system.txt` with comprehensive assistant behavior guidelines.

### feat: Replace StdioServer with LLMMCPWrapper for generic LLM API server

## feat: Implement LLM audit logging and enhance usage tracking

This commit introduces comprehensive audit logging for LLM interactions and
enhances the existing usage tracking mechanism.

Key changes include:
- Integrated `AuditLogger` to log outbound prompts and inbound responses.
- Updated LLM usage tracking to include `cached_tokens`, `reasoning_tokens`,
  `project`, and `username` for more granular reporting.
- Ensured the `data` directory is created for SQLite databases.
- Added a `close` method to `LLMClient` for proper resource management.
- Updated project description and author email in `pyproject.toml`.
- Adjusted `.gitignore` to include the `data/` directory and remove `logs/`.
- Removed `data/accounting.sqlite` as it's no longer needed.
