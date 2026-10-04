"""Direct in-process Python REPL implementation."""

import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Optional, Union

from ..exceptions import FatalError
from .base import BaseREPL
from .types import REPLResult


class _JazReturnSignal(BaseException):
    """Internal signal used to unwind execution when a top-level return is executed."""

    def __init__(self, value: Any):
        super().__init__()
        self.value = value


class _ReturnTransformer(ast.NodeTransformer):
    """Transforms top-level return statements into raising _JazReturnSignal."""

    def __init__(self):
        self.function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        return node

    def visit_Return(self, node: ast.Return):
        if self.function_depth == 0:
            val = node.value if node.value is not None else ast.Constant(value=None)
            call = ast.Call(
                func=ast.Name(id="__jaz_return_signal__", ctx=ast.Load()),
                args=[val],
                keywords=[],
            )
            return ast.copy_location(ast.Raise(exc=call, cause=None), node)
        return node


class LocalPythonREPL(BaseREPL):
    """A persistent in-process Python REPL supporting top-level return and state persistence."""

    def __init__(
        self,
        initial_namespace: Optional[Dict[str, Any]] = None,
        workspace_dir: Optional[Union[str, Path]] = None,
    ):
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        if self.workspace_dir:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            ws_str = str(self.workspace_dir)
            if ws_str not in sys.path:
                sys.path.insert(0, ws_str)

        self.namespace: Dict[str, Any] = {}
        # Prepopulate with standard builtins and return handler
        self.namespace["__builtins__"] = __builtins__
        self.namespace["__jaz_return_signal__"] = _JazReturnSignal
        self.namespace["ensure_packages"] = self._ensure_packages
        if self.workspace_dir:
            self.namespace["workspace_dir"] = str(self.workspace_dir)
            self.namespace["output_dir"] = str(self.workspace_dir)
        if initial_namespace:
            self.namespace.update(initial_namespace)

    @staticmethod
    def _ensure_packages(*packages: str) -> None:
        """Installs missing Python packages via pip dynamically."""
        import importlib.util
        to_install = []
        for pkg in packages:
            pkg_str = str(pkg).strip()
            if not pkg_str:
                continue
            base = pkg_str.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].replace("-", "_")
            if importlib.util.find_spec(base) is None:
                to_install.append(pkg_str)
        if to_install:
            print(f"[JAZ REPL] Installing required packages: {', '.join(to_install)}...")
            subprocess.run([sys.executable, "-m", "pip", "install", *to_install], check=True)
            print(f"[JAZ REPL] Successfully installed {', '.join(to_install)}.")

    def execute(self, code: str) -> REPLResult:
        start_time = time.time()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        has_returned = False
        return_value = None
        error_msg = None
        raised_exception: Optional[BaseException] = None

        orig_cwd = None
        if self.workspace_dir:
            try:
                orig_cwd = os.getcwd()
                os.chdir(self.workspace_dir)
            except Exception:
                orig_cwd = None

        attempted_installs = set()
        try:
            while True:
                try:
                    tree = ast.parse(code)
                    transformer = _ReturnTransformer()
                    tree = transformer.visit(tree)
                    ast.fix_missing_locations(tree)
                    compiled = compile(tree, filename="<jaz_repl>", mode="exec")

                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                        exec(compiled, self.namespace)
                    break

                except _JazReturnSignal as sig:
                    has_returned = True
                    return_value = sig.value
                    break
                except FatalError:
                    raise
                except ModuleNotFoundError as mne:
                    mod_name = getattr(mne, "name", None)
                    top_pkg = mod_name.split(".")[0] if mod_name else None
                    if top_pkg and top_pkg not in attempted_installs:
                        attempted_installs.add(top_pkg)
                        try:
                            res = subprocess.run(
                                [sys.executable, "-m", "pip", "install", top_pkg],
                                capture_output=True,
                                text=True,
                            )
                            if res.returncode == 0:
                                stdout_buf.write(f"[JAZ Auto-Install] Successfully installed package '{top_pkg}'\n")
                                continue
                        except Exception:
                            pass
                    raised_exception = mne
                    lines = traceback.format_exception(type(mne), mne, mne.__traceback__)
                    clean_lines = [l for l in lines if "_JazReturnSignal" not in l and "LocalPythonREPL" not in l]
                    error_msg = "".join(clean_lines).strip()
                    break
                except SyntaxError as se:
                    raised_exception = se
                    error_msg = f"SyntaxError on line {se.lineno}: {se.msg}\n{se.text}"
                    break
                except Exception as e:
                    raised_exception = e
                    # Format exception traceback cleanly without REPL internal frames
                    lines = traceback.format_exception(type(e), e, e.__traceback__)
                    clean_lines = [l for l in lines if "_JazReturnSignal" not in l and "LocalPythonREPL" not in l]
                    error_msg = "".join(clean_lines).strip()
                    break
                except BaseException as be:
                    if isinstance(be, KeyboardInterrupt):
                        raise
                    raised_exception = be
                    error_msg = f"Interrupted by {type(be).__name__}: {be}"
                    break
        finally:
            if orig_cwd is not None:
                try:
                    os.chdir(orig_cwd)
                except Exception:
                    pass

        exec_time = time.time() - start_time
        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            return_value=return_value,
            has_returned=has_returned,
            error=error_msg,
            exception=raised_exception,
            execution_time=exec_time,
        )

    def set_variable(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def get_variable(self, name: str) -> Any:
        return self.namespace.get(name)

    def has_variable(self, name: str) -> bool:
        return name in self.namespace

    def delete_variable(self, name: str) -> None:
        self.namespace.pop(name, None)

    def get_namespace(self) -> Dict[str, Any]:
        return dict(self.namespace)

    def close(self) -> None:
        self.namespace.clear()
