"""Build independent teaching applications on the public runtime API."""

from functools import partial

from harness import Agent, ToolRegistry

from . import tools
from .demos import SYSTEM_PROMPT


def create_tools(config):
    """Snapshot configuration into independent tool bindings, without filesystem I/O."""
    registry = ToolRegistry()
    for name in ("calculator", "current_time", "list_files", "read_file", "write_file",
                 "http_get", "search_documents"):
        fn = getattr(tools, name)
        if name not in ("calculator", "current_time"):
            bound = partial(fn, config.tool_context())
            # Keep the partial's real signature: update_wrapper would expose context.
            bound.__name__, bound.__doc__ = fn.__name__, fn.__doc__
            fn = bound
        risky = name in ("write_file", "http_get")
        registry.tool(risky=risky, read_only=not risky)(fn)
    return registry


def create_agent(model, config, system=SYSTEM_PROMPT, **options):
    """Create a workspace explicitly; no private runtime overrides or global settings."""
    config.workspace.mkdir(parents=True, exist_ok=True)
    return Agent(model, system, tools=create_tools(config), writable_roots=(config.workspace,),
                 compact_at=config.compact_at, max_tool_output=config.max_tool_output, **options)