"""Application-owned tool registries and the supported flat argument validator."""

import inspect
import math
from functools import partial
from typing import Annotated, Literal, get_args, get_origin

from .tool_execution import validate_tool_timeout


PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _parameter_schema(name, annotation):
    """Map str/int/float/bool, Literal choices and an Annotated description to a property schema."""
    description = None
    if get_origin(annotation) is Annotated:
        annotation, *metadata = get_args(annotation)
        if len(metadata) != 1 or type(metadata[0]) is not str or not metadata[0].strip():
            raise TypeError(f"tool parameter {name}: Annotated needs exactly one nonempty description string")
        description = metadata[0]
    if get_origin(annotation) is Literal:
        choices = list(get_args(annotation))
        kinds = {type(choice) for choice in choices}
        if len(kinds) != 1 or not kinds <= {str, int}:
            raise TypeError(f"tool parameter {name}: Literal choices must be all strings or all integers")
        schema = {"type": PY_TO_JSON[kinds.pop()], "enum": choices}
    elif annotation in (str, int, float, bool):
        schema = {"type": PY_TO_JSON[annotation]}
    else:
        raise TypeError(f"tool parameter {name} needs an explicit str, int, float or bool annotation, "
                        "optionally as Literal choices or wrapped in Annotated with a description")
    if description is not None:
        schema["description"] = description
    return schema


class ToolRegistry(dict):
    """An application-owned registry of functions, schemas and execution metadata."""

    def tool(self, risky=False, timeout_seconds=None, read_only=False):
        """Register a synchronous keyword-callable tool with explicit primitive types."""
        validate_tool_timeout(timeout_seconds)
        if type(read_only) is not bool or type(risky) is not bool:
            raise TypeError("risky and read_only must be booleans")

        def register(fn):
            tool_name = getattr(fn, "__name__", None)
            if not callable(fn) or type(tool_name) is not str or not tool_name:
                raise TypeError("tool must be a named callable")
            if tool_name in self:
                raise ValueError(f"tool already registered: {tool_name}; use a separate registry or remove it explicitly")
            implementation = fn
            while isinstance(implementation, partial):
                implementation = implementation.func
            if not inspect.isroutine(implementation):
                implementation = type(implementation).__call__
            if any(check(implementation) for check in (inspect.iscoroutinefunction,
                                                       inspect.isgeneratorfunction, inspect.isasyncgenfunction)):
                raise TypeError("tools must be synchronous functions, not coroutine or generator functions")
            props, required = {}, []
            for name, param in inspect.signature(fn).parameters.items():
                if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    raise TypeError(f"tool parameter {name} must accept a named argument; variadic arguments are unsupported")
                # Resolve only builtin spellings, without evaluating annotation expressions.
                annotation = param.annotation
                if type(annotation) is str:
                    annotation = {kind.__name__: kind for kind in PY_TO_JSON}.get(annotation)
                props[name] = _parameter_schema(name, annotation)
                if param.default is inspect.Parameter.empty:
                    required.append(name)
                else:
                    validate_tool_input({name: param.default}, {"properties": {name: props[name]}})
            self[tool_name] = {
                "fn": fn, "risky": risky, "read_only": read_only,
                "schema": {
                    "name": tool_name, "description": inspect.getdoc(fn) or "",
                    "input_schema": {"type": "object", "properties": props, "required": required,
                                     "additionalProperties": False},
                },
            }
            if timeout_seconds is not None:
                self[tool_name]["timeout_seconds"] = timeout_seconds
            return fn
        return register


def validate_tool_input(args, schema):
    """Validate flat primitive schemas and enum choices, not general JSON Schema or semantics."""
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            raise ValueError(f"missing required argument: {name}")
    types = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,)}
    for name, value in args.items():
        if name not in properties and schema.get("additionalProperties") is False:
            raise ValueError(f"unexpected argument: {name}")
        expected = properties.get(name, {}).get("type")
        if expected in types and type(value) not in types[expected]:
            raise ValueError(f"argument {name} must be {expected}")
        if expected == "number" and type(value) is float and not math.isfinite(value):
            raise ValueError(f"argument {name} must be finite")
        choices = properties.get(name, {}).get("enum")
        if choices is not None and not any(type(choice) is type(value) and choice == value for choice in choices):
            raise ValueError(f"argument {name} must be one of: {', '.join(str(choice) for choice in choices)}")