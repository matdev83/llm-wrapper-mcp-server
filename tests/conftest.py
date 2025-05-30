import pytest

def pytest_collection_modifyitems(config, items):
    """
    Deselect integration tests by default unless -m integration is used.
    """
    if config.getoption("-m") is None or "integration" not in config.getoption("-m"):
        # If -m is not used, or if it's used but doesn't include "integration"
        # then deselect tests marked with 'integration'.
        skip_integration = pytest.mark.skip(reason="integration tests skipped by default. Use '-m integration' to run.")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
