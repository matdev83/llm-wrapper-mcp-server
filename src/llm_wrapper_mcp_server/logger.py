"""Custom logging configuration with TRACE level support."""
import logging
import typing

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

class TraceLogger(logging.Logger):
    """Custom logger with TRACE level support."""
    
    def trace(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'TRACE'."""
        if self.isEnabledFor(TRACE_LEVEL):
            self._log(TRACE_LEVEL, msg, args, **kwargs)

# Configure the root logger to use our custom class
logging.setLoggerClass(TraceLogger)

def get_logger(name: str) -> TraceLogger:
    """Get a logger instance with type hints for custom methods."""
    return typing.cast(TraceLogger, logging.getLogger(name))
