"""LLM-friendly formatting and display utilities for inputs and REPL environment.

Generates concise string representations of objects, tools, data structures,
and function signatures for the LLM prompt as described in the JAZ paper.
"""

import inspect
from typing import Any, Callable, Dict


def format_class_preview(cls: type) -> str:
    """Format a class preview with docstring and public members (Paper L1201-1202)."""
    name = cls.__name__
    doc = inspect.getdoc(cls) or ""
    first_doc = doc.strip().split("\n")[0] if doc else ""
    public_members = [m for m in dir(cls) if not m.startswith("_")][:10]
    members_str = f", members={public_members}" if public_members else ""
    doc_str = f": {first_doc}" if first_doc else ""
    return f"<Class `{name}`{members_str}{doc_str}>"


def format_value_preview(val: Any, max_str_len: int = 50000) -> str:
    """Format an arbitrary Python object into a concise LLM-friendly preview.
    
    Default max_str_len is 50,000 characters as specified in JAZ Paper L731.
    """
    if inspect.isclass(val):
        return format_class_preview(val)

    if callable(val):
        return format_callable_preview(val)

    # Check for pandas/polars DataFrame without requiring import
    type_name = type(val).__name__
    if "DataFrame" in type_name:
        shape = getattr(val, "shape", None)
        cols = list(getattr(val, "columns", []))
        cols_preview = f", columns={cols[:5]}" if cols else ""
        return f"<{type_name} shape={shape}{cols_preview}>"

    if isinstance(val, (list, tuple, set)):
        items_count = len(val)
        coll_type = type(val).__name__
        if items_count == 0:
            return f"{coll_type}()"
        if items_count <= 3:
            items_str = ", ".join(repr(x)[:60] for x in val)
            return f"[{items_str}]" if isinstance(val, list) else f"{coll_type}([{items_str}])"
        sample = repr(next(iter(val)))[:60]
        return f"<{coll_type} with {items_count} items, sample: {sample}...>"

    if isinstance(val, dict):
        keys = list(val.keys())
        if len(keys) <= 3:
            return repr({k: val[k] for k in keys})
        return f"<dict with {len(keys)} keys: {keys[:4]}...>"

    if isinstance(val, str):
        if len(val) <= max_str_len:
            return repr(val)
        return repr(val[:max_str_len] + f"... [truncated, total {len(val)} chars]")

    r = repr(val)
    if len(r) > max_str_len:
        return r[:max_str_len] + "...>"
    return r


def format_callable_preview(fn: Callable[..., Any]) -> str:
    """Generate a clean docstring and signature preview for tools passed to invoke."""
    name = getattr(fn, "__name__", str(fn))
    try:
        sig = str(inspect.signature(fn))
    except Exception:
        sig = "(...)"

    doc = inspect.getdoc(fn)
    if doc:
        first_line = doc.strip().split("\n")[0]
        return f"<Tool `{name}{sig}`: {first_line}>"
    return f"<Tool `{name}{sig}`>"


def format_invoke_signature(inputs: Dict[str, Any], max_str_len: int = 50000) -> str:
    """Format all inputs as the virtual function definition seen by the LLM.
    
    Matches Paper Figure 1 & 2:
    def invoke(
        task="...",
        data=<DataFrame ...>,
        tool=<Tool ...>,
    ):
    """
    lines = ["def invoke("]
    for key, val in inputs.items():
        preview = format_value_preview(val, max_str_len=max_str_len)
        lines.append(f"    {key}={preview},")
    lines.append("):")
    return "\n".join(lines)


def format_available_variables_doc(inputs: Dict[str, Any], max_str_len: int = 50000) -> str:
    """Generate documentation of variables available in the REPL."""
    lines = ["Available variables in this REPL:"]
    for key, val in inputs.items():
        if callable(val):
            doc = inspect.getdoc(val) or ""
            sig = ""
            try:
                sig = str(inspect.signature(val))
            except Exception:
                pass
            lines.append(f"- `{key}{sig}`: {doc.strip() or 'Callable tool'}")
        else:
            lines.append(f"- `{key}`: {format_value_preview(val, max_str_len=max_str_len)}")
    lines.append("- `__history__`: List of past steps in this REPL session.")
    lines.append("- `invoke`: Function to spawn sub-agents recursively.")
    return "\n".join(lines)

