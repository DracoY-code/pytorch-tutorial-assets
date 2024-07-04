"""
Testing Utilities

This module defines various utility functions for the unit tests in the repository.
"""

from typing import Callable


def get_func_header(func: Callable) -> str:
    """
    Returns a function header of format: `[module_name].[func_name]`.

    Args:
        func (Callable): A callable function with __name__.

    Returns:
        str: The formatted function header.
    """
    return f'\n\nRunning `{__name__}.{func.__name__}`'
