"""
Custom exception wrapper.

Every component catches low-level exceptions (KeyError, FileNotFoundError,
etc.) and re-raises them as `FraudDetectionError`, which records exactly
which file/line the failure happened in. This keeps tracebacks useful when
the pipeline is run end-to-end instead of cell-by-cell in a notebook.
"""

import sys


def _error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error: {error}"
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in script [{file_name}] "
        f"at line [{line_number}] "
        f"with message: [{error}]"
    )


class FraudDetectionError(Exception):
    """Raised by any pipeline component on failure, with full context."""

    def __init__(self, error_message: Exception, error_detail: sys = sys):
        super().__init__(str(error_message))
        self.error_message = _error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
