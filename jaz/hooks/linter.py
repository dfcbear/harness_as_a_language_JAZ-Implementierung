"""AST-based Python Linter and Auto-Repair Hook for JAZ.

Provides both:
1. Auto-Correction: Strips conversational preambles/markdown, normalizes indentation,
   removes code after return.
2. Error Detection & Guidance: Detects syntax errors, hallucinated/misspelled tools,
   and disallowed module imports before REPL execution, feeding helpful hints back to the LLM.
"""

import ast
import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .dispatcher import Hook
from .events import REPLExecEnter


class PythonLinterHook(Hook):
    """Linter and auto-repair hook for agent code before REPL execution."""

    def __init__(
        self,
        auto_fix: bool = True,
        check_undefined_tools: bool = True,
        disallowed_modules: Optional[List[str]] = None,
        suggest_similar_names: bool = True,
    ):
        super().__init__()
        self.auto_fix = auto_fix
        self.check_undefined_tools = check_undefined_tools
        self.disallowed_modules = set(disallowed_modules or [])
        self.suggest_similar_names = suggest_similar_names
        self.active_repl_namespace: Dict[str, Any] = {}

    def set_namespace_context(self, namespace: Dict[str, Any]):
        """Update available names in the REPL."""
        self.active_repl_namespace = dict(namespace)

    def on_repl_exec_enter(self, event: REPLExecEnter):
        code = event.code

        # 1. AUTO-CORRECTION PHASE
        if self.auto_fix:
            code = self.auto_repair_code(code)
            event.code = code

        # 2. DIAGNOSTIC & LINTING PHASE
        lint_errors = self.lint_code(code)
        if lint_errors:
            # Intercept execution cleanly and provide diagnostic to LLM
            event.skip_execution = True
            event.lint_error_message = (
                "Linter Pre-Execution Diagnostics (Execution was blocked to protect REPL state):\n"
                + "\n".join(f"- {err}" for err in lint_errors)
                + "\nPlease fix the issues above and write corrected Python code."
            )

        return None

    def auto_repair_code(self, code: str) -> str:
        """Automatically repair common model code formatting anomalies."""
        from ..protocol import _clean_and_extract_code
        text = _clean_and_extract_code(code)

        # Strip conversational preamble (e.g. "Here is the code:", "I will run:")
        lines = text.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            l_strip = line.strip()
            if re.match(r"^(here is|sure,|certainly,|i will|let me|to solve|step \d+:|i'll)", l_strip, re.I):
                start_idx = i + 1
            else:
                break
        if start_idx < len(lines):
            lines = lines[start_idx:]

        # Strip conversational postamble (e.g. "This code will calculate ...", "Hope this helps!")
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            l_strip = lines[i].strip()
            if re.match(r"^(this (code|script)|hope this|let me know|explanation:)", l_strip, re.I):
                end_idx = i
            else:
                break
        lines = lines[:end_idx]

        repaired = "\n".join(lines).strip()
        return repaired or code

    def lint_code(self, code: str) -> List[str]:
        """Perform AST analysis for syntax, disallowed imports, and hallucinated tools."""
        import builtins
        errors: List[str] = []

        # Check Syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as se:
            errors.append(f"SyntaxError on line {se.lineno}: {se.msg} (Code: `{se.text and se.text.strip()}`)")
            return errors

        # Check Disallowed Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._is_module_disallowed(alias.name):
                        errors.append(f"Security: Import of module '{alias.name}' is disallowed.")
            elif isinstance(node, ast.ImportFrom):
                if node.module and self._is_module_disallowed(node.module):
                    errors.append(f"Security: Import from module '{node.module}' is disallowed.")

        # Check Undefined Tool Calls
        if self.check_undefined_tools and self.active_repl_namespace:
            # Collect local definitions inside the snippet itself
            local_defs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    local_defs.add(node.name)
                    for arg in node.args.args + getattr(node.args, "posonlyargs", []) + node.args.kwonlyargs:
                        local_defs.add(arg.arg)
                    if node.args.vararg:
                        local_defs.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        local_defs.add(node.args.kwarg.arg)
                elif isinstance(node, ast.ClassDef):
                    local_defs.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split('.')[0]
                        local_defs.add(name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        local_defs.add(name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        for n in ast.walk(target):
                            if isinstance(n, ast.Name):
                                local_defs.add(n.id)
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        local_defs.add(node.target.id)
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    for n in ast.walk(node.target):
                        if isinstance(n, ast.Name):
                            local_defs.add(n.id)
                elif isinstance(node, (ast.With, ast.AsyncWith)):
                    for item in node.items:
                        if item.optional_vars:
                            for n in ast.walk(item.optional_vars):
                                if isinstance(n, ast.Name):
                                    local_defs.add(n.id)
                elif isinstance(node, getattr(ast, "NamedExpr", ())):
                    if isinstance(node.target, ast.Name):
                        local_defs.add(node.target.id)

            builtin_names = set(dir(builtins))
            known_names = set(self.active_repl_namespace.keys()) | builtin_names | local_defs

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                    if fn_name not in known_names:
                        suggestion = ""
                        if self.suggest_similar_names:
                            matches = difflib.get_close_matches(fn_name, list(self.active_repl_namespace.keys()), n=1)
                            if matches:
                                suggestion = f" Did you mean `{matches[0]}`?"
                        errors.append(
                            f"Undefined Function Call: `{fn_name}()` is not defined in this REPL.{suggestion} "
                            f"Available tools: {[k for k, v in self.active_repl_namespace.items() if callable(v) and not k.startswith('_')]}"
                        )

        return errors

    def _is_module_disallowed(self, mod_name: str) -> bool:
        for disallowed in self.disallowed_modules:
            if mod_name == disallowed or mod_name.startswith(disallowed + "."):
                return True
        return False
