"""Automatic JSON Schema generation from Python functions.

The model chooses tools from their schema, so the schema *is* the prompt. Windlass
derives it from what you already wrote — type hints and a docstring — so there is
no second source of truth to drift out of date.

Supported annotations:

* scalars — ``str``, ``int``, ``float``, ``bool``
* containers — ``list[T]``, ``dict[str, T]``, ``tuple[T, ...]``, ``set[T]``
* ``Literal[...]`` becomes an enum, ``Enum`` subclasses become enums
* ``T | None`` marks the parameter optional
* Pydantic models and dataclasses expand into nested object schemas
* ``Annotated[T, "description"]`` supplies a per-parameter description

Descriptions come from ``Annotated`` metadata, then from the docstring's
``Args:`` section, then from Pydantic ``Field`` descriptions.

Example:
    >>> from typing import Literal
    >>> def convert(amount: float, to: Literal["usd", "eur"] = "usd") -> float:
    ...     '''Convert an amount.
    ...
    ...     Args:
    ...         amount: The amount to convert.
    ...         to: Target currency.
    ...     '''
    ...     return amount
    >>> schema = function_schema(convert)
    >>> schema["properties"]["to"]["enum"]
    ['usd', 'eur']
    >>> schema["required"]
    ['amount']
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import re
import types
import typing
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin

__all__ = ["function_schema", "parse_docstring", "python_type_to_schema"]

_ARGS_SECTION_RE = re.compile(
    r"^\s*(?:Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE | re.MULTILINE
)
_SECTION_RE = re.compile(
    r"^\s*(?:Returns|Raises|Yields|Examples?|Note|Notes|Attributes|Performance)\s*:\s*$",
    re.IGNORECASE,
)
_ARG_RE = re.compile(r"^\s{0,8}(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")

#: Direct Python type to JSON Schema type mapping.
_SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    r"""Split a Google-style docstring into a summary and per-argument descriptions.

    Args:
        doc: The raw docstring, or ``None``.

    Returns:
        A ``(summary, {arg_name: description})`` pair. The summary is everything
        before the first section header.

    Example:
        >>> summary, args = parse_docstring(
        ...     "Add numbers.\n\nArgs:\n    a: First.\n    b: Second.\n"
        ... )
        >>> summary, args["a"]
        ('Add numbers.', 'First.')
    """
    if not doc:
        return "", {}

    text = inspect.cleandoc(doc)
    match = _ARGS_SECTION_RE.search(text)
    summary = (text[: match.start()] if match else text).strip()
    summary = " ".join(summary.split("\n\n")[0].split())

    if not match:
        return summary, {}

    descriptions: dict[str, str] = {}
    current: str | None = None
    for line in text[match.end() :].splitlines():
        if not line.strip():
            continue
        if _SECTION_RE.match(line):
            break
        arg = _ARG_RE.match(line)
        if arg and not line.startswith(" " * 9):
            current = arg.group(1).lstrip("*")
            descriptions[current] = arg.group(2).strip()
        elif current:
            descriptions[current] = f"{descriptions[current]} {line.strip()}".strip()
    return summary, descriptions


def python_type_to_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation into a JSON Schema fragment.

    Args:
        annotation: The annotation to convert. ``inspect.Parameter.empty`` and
            bare ``Any`` produce a permissive ``{}``.

    Returns:
        A JSON Schema fragment.

    Example:
        >>> python_type_to_schema(list[str])
        {'type': 'array', 'items': {'type': 'string'}}
        >>> python_type_to_schema(int | None)
        {'type': 'integer'}
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    # Annotated[T, "description", ...] -> unwrap, keeping any string metadata.
    if get_origin(annotation) is typing.Annotated:
        base, *metadata = get_args(annotation)
        schema: dict[str, Any] = python_type_to_schema(base)
        for item in metadata:
            if isinstance(item, str):
                schema.setdefault("description", item)
            elif isinstance(item, dict):
                schema.update(item)
        return schema

    if annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}

    origin = get_origin(annotation)

    if origin is Literal:
        options = list(get_args(annotation))
        kinds = {type(o) for o in options}
        schema = {"enum": options}
        if len(kinds) == 1:
            only = kinds.pop()
            if only in _SCALARS:
                schema["type"] = _SCALARS[only]
        return schema

    if origin in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return python_type_to_schema(members[0])
        variants = [python_type_to_schema(m) for m in members]
        return {"anyOf": [v for v in variants if v]}

    if origin in (list, set, frozenset):
        args = get_args(annotation)
        items = python_type_to_schema(args[0]) if args else {}
        schema = {"type": "array"}
        if items:
            schema["items"] = items
        if origin in (set, frozenset):
            schema["uniqueItems"] = True
        return schema

    if origin is tuple:
        args = get_args(annotation)
        if args and args[-1] is Ellipsis:
            return {"type": "array", "items": python_type_to_schema(args[0])}
        return {
            "type": "array",
            "prefixItems": [python_type_to_schema(a) for a in args],
            "minItems": len(args),
            "maxItems": len(args),
        }

    if origin is dict:
        args = get_args(annotation)
        schema = {"type": "object"}
        if len(args) == 2:
            values = python_type_to_schema(args[1])
            if values:
                schema["additionalProperties"] = values
        return schema

    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            members = [member.value for member in annotation]
            schema = {"enum": members}
            kinds = {type(v) for v in members}
            if len(kinds) == 1 and kinds.copy().pop() in _SCALARS:
                schema["type"] = _SCALARS[kinds.pop()]
            return schema

        if _is_pydantic_model(annotation):
            # Duck-typed rather than importing pydantic: this module must stay
            # usable for tools whose annotations are plain classes.
            return _clean_model_schema(annotation.model_json_schema())  # type: ignore[attr-defined]

        if dataclasses.is_dataclass(annotation):
            return _dataclass_schema(annotation)

        if annotation in (list, dict, set, tuple):
            return {
                "type": {"list": "array", "set": "array", "tuple": "array"}.get(
                    annotation.__name__, "object"
                )
            }

    # Unknown annotation: stay permissive rather than emitting a wrong schema.
    return {}


def function_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    skip: tuple[str, ...] = ("self", "cls"),
) -> dict[str, Any]:
    """Build a JSON Schema ``parameters`` object for a function.

    Args:
        fn: The function to inspect. Type hints are resolved against its module
            globals, so ``from __future__ import annotations`` works fine.
        name: Override for the tool name. Defaults to ``fn.__name__``.
        description: Override for the description. Defaults to the docstring
            summary.
        skip: Parameter names to omit.

    Returns:
        A JSON Schema object with ``type``, ``properties``, ``required`` and
        ``additionalProperties: false``.

    Raises:
        TypeError: If ``fn`` is not callable.

    Note:
        ``*args`` and ``**kwargs`` are skipped — providers cannot express them,
        and a tool that needs them almost always wants an explicit list or dict
        parameter instead.

    Example:
        >>> def greet(name: str, excited: bool = False) -> str:
        ...     '''Greet someone.
        ...
        ...     Args:
        ...         name: Who to greet.
        ...         excited: Add an exclamation mark.
        ...     '''
        ...     return name
        >>> schema = function_schema(greet)
        >>> schema["properties"]["name"]["description"]
        'Who to greet.'
        >>> schema["required"]
        ['name']
    """
    if not callable(fn):
        raise TypeError(f"Expected a callable, got {type(fn).__name__}.")

    signature = inspect.signature(fn)
    hints = _resolve_hints(fn)

    _, arg_docs = parse_docstring(inspect.getdoc(fn))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.name in skip:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(parameter.name, parameter.annotation)
        schema = python_type_to_schema(annotation)

        doc = arg_docs.get(parameter.name)
        if doc:
            schema["description"] = doc

        if parameter.default is not inspect.Parameter.empty:
            if _is_jsonable(parameter.default):
                schema["default"] = parameter.default
        else:
            required.append(parameter.name)

        properties[parameter.name] = schema or {}

    out: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        out["required"] = required
    return out


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolve a function's type hints, tolerating individual failures.

    ``from __future__ import annotations`` turns every annotation into a string,
    and :func:`typing.get_type_hints` resolves the whole set or none of it. One
    annotation naming a symbol that is not visible at module scope (a type
    imported inside a function, say) would therefore leave *every* parameter
    unresolved and produce an empty schema for all of them.

    This resolves the set, and on failure falls back to resolving each parameter
    on its own — so one awkward annotation costs that one parameter's schema
    rather than the whole tool's.

    Args:
        fn: The function to inspect.

    Returns:
        A mapping of parameter name to resolved annotation. Parameters whose
        annotation cannot be resolved are omitted, which makes them permissive
        rather than wrong.
    """
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        pass

    raw = getattr(fn, "__annotations__", {}) or {}
    namespace = getattr(fn, "__globals__", {})
    resolved: dict[str, Any] = {}
    for name, annotation in raw.items():
        if not isinstance(annotation, str):
            resolved[name] = annotation
            continue
        try:
            resolved[name] = eval(annotation, namespace, None)
        except Exception:
            continue
    return resolved


def _is_pydantic_model(candidate: type) -> bool:
    """Return whether ``candidate`` is a Pydantic v2 model class."""
    return hasattr(candidate, "model_json_schema") and hasattr(candidate, "model_fields")


def _clean_model_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` references so providers see a self-contained schema.

    Providers reject ``$ref``. Pydantic emits them for nested models, so they are
    resolved here rather than at each call site.

    Args:
        schema: A Pydantic-generated JSON Schema.

    Returns:
        The same schema with references inlined and ``$defs`` removed.
    """
    definitions = schema.pop("$defs", {})

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 10:  # cycles: stop before recursing forever
            return {"type": "object"}
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = definitions.get(ref.split("/")[-1], {})
                merged = {
                    **_resolve(target, depth + 1),
                    **{k: v for k, v in node.items() if k != "$ref"},
                }
                return merged
            return {k: _resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item, depth + 1) for item in node]
        return node

    resolved = _resolve(schema)
    resolved.pop("title", None)
    return resolved


def _dataclass_schema(cls: type) -> dict[str, Any]:
    """Build an object schema from a dataclass's fields."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
    except Exception:
        hints = {}

    for field in dataclasses.fields(cls):
        properties[field.name] = python_type_to_schema(hints.get(field.name, field.type))
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if not has_default:
            required.append(field.name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _is_jsonable(value: Any) -> bool:
    """Return whether ``value`` can appear as a JSON Schema default."""
    return isinstance(value, str | int | float | bool | type(None)) or (
        isinstance(value, list | dict) and not value
    )
