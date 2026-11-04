"""Strict registration is atomic; unsupported definitions never become schemas."""

import inspect
from functools import partial
from typing import Annotated, Literal

from harness import ToolRegistry, validate_tool_input
from tests.helpers import TemporaryDirectoryTestCase


class ToolContractTests(TemporaryDirectoryTestCase):
    def test_supported_types_defaults_keyword_only_and_bound_context(self):
        def lookup(context, sku: str, *, count: int = 1, ratio: float = 1, active: bool = False):
            """Look up application-owned data."""
            return context[sku] * count

        bound = partial(lookup, {"WIDGET": 7})
        bound.__name__ = "lookup"
        tools = ToolRegistry()
        self.assertIs(tools.tool(read_only=True)(bound), bound)
        schema = tools["lookup"]["schema"]["input_schema"]
        self.assertEqual(schema["required"], ["sku"])
        self.assertEqual(set(schema["properties"]), {"sku", "count", "ratio", "active"})
        self.assertEqual(tools["lookup"]["fn"](sku="WIDGET", count=2), 14)

    def test_named_synchronous_callable_objects_and_partials_remain_supported(self):
        class PrefixTool:
            __name__ = "prefix"

            def __call__(self, prefix: str, value: str):
                return prefix + value

        tools = ToolRegistry()
        tools.tool()(PrefixTool())
        bound = partial(PrefixTool(), "hello ")
        bound.__name__ = "greet"
        tools.tool()(bound)
        self.assertEqual(tools["prefix"]["fn"](prefix="hi ", value="reader"), "hi reader")
        self.assertEqual(tools["greet"]["fn"](value="reader"), "hello reader")
        self.assertEqual(tools["greet"]["schema"]["input_schema"]["properties"], {"value": {"type": "string"}})

    def test_async_and_generator_callable_objects_are_rejected_atomically(self):
        class CoroutineTool:
            __name__ = "coroutine_tool"

            async def __call__(self, value: str):
                return value

        class GeneratorTool:
            __name__ = "generator_tool"

            def __call__(self, value: str):
                yield value

        class AsyncGeneratorTool:
            __name__ = "async_generator_tool"

            async def __call__(self, value: str):
                yield value

        for kind in (CoroutineTool, GeneratorTool, AsyncGeneratorTool):
            for wrapped in (False, True):
                fn = partial(kind()) if wrapped else kind()
                fn.__name__ = kind.__name__
                tools = ToolRegistry()
                with self.subTest(kind=kind.__name__, partial=wrapped):
                    with self.assertRaisesRegex(TypeError, "synchronous"):
                        tools.tool()(fn)
                    self.assertEqual(tools, {})

    def test_annotated_descriptions_and_literal_choices_become_schema_and_are_enforced(self):
        tools = ToolRegistry()

        @tools.tool(read_only=True)
        def sort_items(path: Annotated[str, "Path relative to the workspace"],
                       order: Annotated[Literal["asc", "desc"], "Sort direction"] = "asc",
                       top: Literal[1, 5, 10] = 5):
            """Sort items."""
            return f"{path}:{order}:{top}"

        properties = tools["sort_items"]["schema"]["input_schema"]["properties"]
        self.assertEqual(properties, {
            "path": {"type": "string", "description": "Path relative to the workspace"},
            "order": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction"},
            "top": {"type": "integer", "enum": [1, 5, 10]}})
        schema = tools["sort_items"]["schema"]["input_schema"]
        validate_tool_input({"path": "a", "order": "desc", "top": 10}, schema)
        for args in ({"path": "a", "order": "up"}, {"path": "a", "top": 3}, {"path": "a", "top": True}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "must"):
                validate_tool_input(args, schema)

    def test_invalid_annotated_and_literal_forms_are_rejected_atomically(self):
        def sample(value: str):
            pass

        for annotation in (Annotated[str, 1], Annotated[str, ""], Annotated[str, "a", "b"],
                           Literal["a", 1], Literal[True, False], Literal[1.5], Annotated[list, "items"],
                           "Literal['a']"):
            sample.__annotations__["value"] = annotation
            tools = ToolRegistry()
            with self.subTest(annotation=str(annotation)), self.assertRaisesRegex(TypeError, "value"):
                tools.tool()(sample)
            self.assertEqual(tools, {})

        def defaulted(order="sideways"):
            pass
        defaulted.__annotations__["order"] = Literal["asc", "desc"]
        tools = ToolRegistry()
        with self.assertRaisesRegex(ValueError, "one of"):
            tools.tool()(defaulted)
        self.assertEqual(tools, {})

    def test_future_builtin_annotations_work_without_evaluating_expressions(self):
        namespace = {}
        exec("from __future__ import annotations\ndef echo(value: str, count: int = 1): pass", namespace)
        tools = ToolRegistry()
        tools.tool()(namespace["echo"])
        self.assertEqual(tools["echo"]["schema"]["input_schema"]["properties"]["count"], {"type": "integer"})
        namespace["echo"].__annotations__["value"] = "raise_if_evaluated()"
        with self.assertRaisesRegex(TypeError, "explicit"):
            ToolRegistry().tool()(namespace["echo"])

    def test_unsupported_annotations_and_parameter_forms_are_rejected_atomically(self):
        def sample(value: str):
            pass

        for annotation in (inspect.Parameter.empty, list, dict, object, None, "Optional[str]"):
            sample.__annotations__["value"] = annotation
            tools = ToolRegistry()
            with self.subTest(annotation=str(annotation)), self.assertRaisesRegex(TypeError, "value"):
                tools.tool()(sample)
            self.assertEqual(tools, {})
        for definition in ("def bad(value: str, /): pass", "def bad(*args: str): pass",
                           "def bad(**kwargs: str): pass", "async def bad(value: str): pass",
                           "def bad(value: str): yield value"):
            namespace = {}
            exec(definition, namespace)
            tools = ToolRegistry()
            with self.subTest(definition=definition), self.assertRaises(TypeError):
                tools.tool()(namespace["bad"])
            self.assertEqual(tools, {})

    def test_invalid_defaults_do_not_leak_values_or_partially_register(self):
        for annotation, default in ((str, None), (int, True), (bool, 1), (float, "private-default"),
                                    (float, float("inf")), (float, float("nan"))):
            def sample(value=default):
                pass
            sample.__annotations__["value"] = annotation
            tools = ToolRegistry()
            with self.subTest(annotation=annotation), self.assertRaises(ValueError) as error:
                tools.tool()(sample)
            self.assertNotIn("private-default", str(error.exception))
            self.assertEqual(tools, {})

    def test_duplicate_registration_preserves_original_callable_and_metadata(self):
        tools = ToolRegistry()

        @tools.tool(read_only=True)
        def echo(value: str):
            return value

        original = tools["echo"]
        with self.assertRaisesRegex(ValueError, "already registered"):
            tools.tool(risky=True)(echo)
        self.assertIs(tools["echo"], original)
        self.assertFalse(original["risky"])
        for options in ({"risky": 1}, {"read_only": "yes"}):
            with self.assertRaises(TypeError):
                ToolRegistry().tool(**options)

    def test_nonfinite_tool_arguments_are_rejected_before_dispatch(self):
        schema = {"properties": {"ratio": {"type": "number"}}}
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_tool_input({"ratio": value}, schema)
