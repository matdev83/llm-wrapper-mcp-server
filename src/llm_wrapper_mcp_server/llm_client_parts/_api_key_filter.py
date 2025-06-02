import logging

class ApiKeyFilter(logging.Filter):
    """Filter to redact API keys from log messages"""
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def filter(self, record: logging.LogRecord) -> bool:
        if self.api_key:
            record.msg = str(record.msg).replace(self.api_key, "***REDACTED***")
        return True
