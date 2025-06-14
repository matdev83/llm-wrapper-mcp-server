import os
import logging
from typing import Optional
from llm_accounting import LLMAccounting
from llm_accounting.backends.sqlite import SQLiteBackend
from llm_accounting.backends.mock_backend import MockBackend
from llm_accounting.audit_log import AuditLogger
from ..logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.NOTSET)
logger.propagate = True


class LLMAccountingManager:
    def __init__(self, enable_logging: bool, enable_audit_log: bool):
        os.makedirs("data", exist_ok=True)

        self.llm_tracker = None
        if enable_logging:
            try:
                db_url_env = os.getenv("LLM_ACCOUNTING_DB_URL")
                if db_url_env and "://" in db_url_env:
                    backend = SQLiteBackend(db_path=db_url_env)
                else:
                    backend = MockBackend()
                self.llm_tracker = LLMAccounting(backend=backend)
            except Exception as e:
                logger.error(f"Failed to initialize LLMAccounting: {e}")
                self.llm_tracker = None
        else:
            logger.info("LLM accounting is disabled.")

        self.audit_logger = None
        if enable_audit_log:
            try:
                db_url_env = os.getenv("LLM_ACCOUNTING_DB_URL")
                if db_url_env and "://" in db_url_env:
                    backend = SQLiteBackend(db_path=db_url_env)
                else:
                    backend = MockBackend()
                self.audit_logger = AuditLogger(backend=backend)
            except Exception as e:
                logger.error(f"Failed to initialize AuditLogger: {e}")
                self.audit_logger = None
        else:
            logger.info("Audit logging is disabled.")

    def get_tracker(self) -> Optional[LLMAccounting]:
        return self.llm_tracker

    def get_audit_logger(self) -> Optional[AuditLogger]:
        return self.audit_logger

    def track_usage(self, **kwargs) -> None:
        if self.llm_tracker:
            try:
                self.llm_tracker.track_usage(**kwargs)
            except Exception as e:
                logger.error(f"Failed to track LLM usage: {e}")

    def log_prompt(self, **kwargs) -> None:
        if self.audit_logger:
            self.audit_logger.log_prompt(**kwargs)

    def log_response(self, **kwargs) -> None:
        if self.audit_logger:
            self.audit_logger.log_response(**kwargs)

    def close(self) -> None:
        """Close any open database connections."""
        if self.llm_tracker and hasattr(self.llm_tracker.backend, "close"):
            try:
                self.llm_tracker.backend.close()
            except Exception as e:
                logger.warning(f"Failed to close accounting backend: {e}")
        if self.audit_logger and hasattr(self.audit_logger.backend, "close"):
            try:
                self.audit_logger.backend.close()
            except Exception as e:
                logger.warning(f"Failed to close audit backend: {e}")
