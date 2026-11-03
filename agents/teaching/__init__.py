"""Explicit teaching application factories; importing this package creates no files."""

from .config import TeachingConfig
from .factory import create_agent, create_tools

__all__ = ["TeachingConfig", "create_agent", "create_tools"]