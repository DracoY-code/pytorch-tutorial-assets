"""
Custom Error Handlers

This module defines custom error classes and errors used throughout the repository.

Classes:
    - ChannelMismatchError:
        Raised when the number of channels in an operation does not match.
"""

class ChannelMismatchError(Exception):
    """
    Exception raised when the number of channels in an operation does not match.

    Attributes:
        message (str): Error message describing the exception.
    """

    def __init__(
        self, message: str = 'The number of channels does not match.',
        *args: object,
    ) -> None:
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:
        return self.message
