"""Label-free HotpotQA applications; importing this package performs no application I/O.

Use agents.hotpotqa.single.run_agent or agents.hotpotqa.multi.run_agents. Worker
commands are also available with python -m agents.hotpotqa.single (or .multi).
"""

from .common import HotpotQAConfig, create_agent, make_tools

__all__ = ["HotpotQAConfig", "create_agent", "make_tools"]