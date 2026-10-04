"""Subprocess-isolated Python REPL with host tool RPC bridging."""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from .base import BaseREPL
from .rpc import deserialize_payload, serialize_payload
from .types import REPLResult


class SubprocessREPL(BaseREPL):
    """Executes agent Python code in a separated child process.
    
    Provides workspace isolation, filtered environment variables (preventing API key theft),
    and proxies calls to host tools/sub-invokes via RPC.
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        timeout: float = 60.0,
        filter_env: bool = True,
    ):
        self.timeout = timeout
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir).resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="jaz_sandbox_")
            self.workspace_dir = Path(self._temp_dir.name).resolve()

        self.filter_env = filter_env
        self.host_tools: Dict[str, Callable[..., Any]] = {}
        self.local_namespace_cache: Dict[str, Any] = {}
        self.process: Optional[subprocess.Popen] = None
        self._start_worker()

    def _get_clean_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        if self.filter_env:
            # Scrub API keys and tokens from sandbox process
            sensitive_patterns = ["API_KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH"]
            keys_to_remove = [
                k for k in env if any(pat in k.upper() for pat in sensitive_patterns)
            ]
            for k in keys_to_remove:
                env.pop(k, None)
        # Ensure PYTHONPATH includes project root so jaz is importable
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{project_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root
        )
        return env

    def _start_worker(self):
        cmd = [sys.executable, "-u", "-m", "jaz.repl.rpc"]
        env = self._get_clean_env()
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.workspace_dir),
            env=env,
            text=True,
            bufsize=1,
        )

        # Wait for worker ready signal
        ready_line = self.process.stdout.readline().strip()
        if ready_line != "JAZ_WORKER_READY":
            err = self.process.stderr.read()
            raise RuntimeError(f"Sandbox worker failed to start: {err or ready_line}")

    def set_variable(self, name: str, value: Any) -> None:
        self.local_namespace_cache[name] = value
        if callable(value):
            self.host_tools[name] = value
            doc = inspect.getdoc(value) or ""
            cmd = {
                "type": "set_tool_proxies",
                "tools": {name: doc},
            }
            self._send_cmd(cmd)
        else:
            cmd = {
                "type": "set_variables",
                "variables": {name: serialize_payload(value)},
            }
            self._send_cmd(cmd)

    def get_variable(self, name: str) -> Any:
        return self.local_namespace_cache.get(name)

    def has_variable(self, name: str) -> bool:
        return name in self.local_namespace_cache

    def delete_variable(self, name: str) -> None:
        self.local_namespace_cache.pop(name, None)
        self.host_tools.pop(name, None)

    def get_namespace(self) -> Dict[str, Any]:
        return dict(self.local_namespace_cache)

    def _send_cmd(self, cmd_dict: dict) -> None:
        line = "JAZ_CMD:" + json.dumps(cmd_dict)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        # Read acknowledgment
        resp_line = self.process.stdout.readline().strip()
        try:
            resp = json.loads(resp_line)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Worker command error: {resp}")
        except Exception:
            pass

    def execute(self, code: str) -> REPLResult:
        start_time = time.time()
        exec_cmd = {"type": "execute", "code": code}
        line = "JAZ_CMD:" + json.dumps(exec_cmd)
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError, ValueError):
            stderr_text = ""
            try:
                stderr_text = self.process.stderr.read()
            except Exception:
                pass
            return REPLResult(
                stdout="",
                stderr=stderr_text,
                error=f"Sandbox process is not running: {stderr_text}",
                execution_time=time.time() - start_time,
            )

        # Listen for results or RPC callbacks
        while True:
            out_line = self.process.stdout.readline()
            if not out_line:
                # Worker died
                stderr_text = self.process.stderr.read()
                return REPLResult(
                    stdout="",
                    stderr=stderr_text,
                    error=f"Sandbox process terminated unexpectedly: {stderr_text}",
                    execution_time=time.time() - start_time,
                )

            out_line = out_line.strip()
            if not out_line:
                continue

            # Check if worker is making an RPC call to a host tool or invoke
            if out_line.startswith("JAZ_RPC_REQ:"):
                req = json.loads(out_line[len("JAZ_RPC_REQ:"):])
                tool_name = req["tool_name"]
                args = deserialize_payload(req["args"])
                kwargs = deserialize_payload(req["kwargs"])

                try:
                    tool_fn = self.host_tools.get(tool_name)
                    if tool_fn is None:
                        raise ValueError(f"Host tool `{tool_name}` not found")
                    res = tool_fn(*args, **kwargs)
                    resp_dict = {
                        "status": "ok",
                        "result": serialize_payload(res),
                    }
                except Exception as e:
                    resp_dict = {
                        "status": "error",
                        "error": str(e),
                    }

                resp_line = "JAZ_RPC_RESP:" + json.dumps(resp_dict)
                self.process.stdin.write(resp_line + "\n")
                self.process.stdin.flush()
                continue

            # Check if execution finished
            if out_line.startswith("JAZ_EXEC_RESULT:"):
                data = json.loads(out_line[len("JAZ_EXEC_RESULT:"):])
                has_ret = data["has_returned"]
                ret_val = deserialize_payload(data["return_value"]) if has_ret and data["return_value"] else None
                return REPLResult(
                    stdout=data.get("stdout", ""),
                    stderr=data.get("stderr", ""),
                    return_value=ret_val,
                    has_returned=has_ret,
                    error=data.get("error"),
                    execution_time=data.get("execution_time", time.time() - start_time),
                )

    def close(self) -> None:
        if self.process:
            try:
                self.process.stdin.write("JAZ_WORKER_EXIT\n")
                self.process.stdin.flush()
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.kill()
            self.process = None

        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
