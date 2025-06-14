import os
import logging
from typing import Optional
from llm_accounting import LLMAccounting
from llm_accounting.backends.sqlite import SQLiteBackend
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
            self.llm_tracker = LLMAccounting(
                backend=SQLiteBackend(db_path="data/accounting.sqlite")
            )
        else:
            logger.info("LLM accounting is disabled.")

        self.audit_logger = None
        if enable_audit_log:
            self.audit_logger = AuditLogger(
                backend=SQLiteBackend(db_path="data/audit.sqlite")
            )
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
