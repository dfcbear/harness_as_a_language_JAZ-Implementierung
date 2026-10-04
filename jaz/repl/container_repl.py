"""Container-isolated Python REPL using Podman Desktop or Docker."""

import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Optional

from .base import BaseREPL
from .rpc import deserialize_payload, serialize_payload
from .types import REPLResult


class ContainerREPL(BaseREPL):
    """Executes agent code inside an isolated container (Docker or Podman Desktop)."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        runtime: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        timeout: float = 60.0,
        network_enabled: bool = False,
    ):
        self.image = image
        self.timeout = timeout
        self.network_enabled = network_enabled

        # Auto-detect podman or docker
        if runtime:
            self.runtime = runtime
        else:
            if shutil.which("podman"):
                self.runtime = "podman"
            elif shutil.which("docker"):
                self.runtime = "docker"
            else:
                raise RuntimeError(
                    "Neither 'podman' nor 'docker' was found on your system PATH. "
                    "Please install Podman Desktop or Docker, or use 'subprocess' or 'direct' sandbox."
                )

        if workspace_dir:
            self.workspace_dir = Path(workspace_dir).resolve()
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="jaz_container_")
            self.workspace_dir = Path(self._temp_dir.name).resolve()

        self.host_tools: Dict[str, Callable[..., Any]] = {}
        self.local_namespace_cache: Dict[str, Any] = {}
        self.process: Optional[subprocess.Popen] = None
        self._start_container()

    def _start_container(self):
        project_root = str(Path(__file__).resolve().parent.parent.parent)

        cmd = [
            self.runtime,
            "run",
            "-i",
            "--rm",
            "-v",
            f"{project_root}:/jaz_app:ro",
            "-v",
            f"{self.workspace_dir}:/workspace:rw",
            "-w",
            "/workspace",
            "-e",
            "PYTHONPATH=/jaz_app",
        ]

        if not self.network_enabled:
            cmd.extend(["--network", "none"])

        cmd.extend([self.image, "python", "-u", "-m", "jaz.repl.rpc"])

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Container runtime executable '{self.runtime}' could not be started.")

        ready_line = self.process.stdout.readline().strip()
        if ready_line != "JAZ_WORKER_READY":
            err = self.process.stderr.read()
            raise RuntimeError(
                f"Container worker ({self.runtime}) failed to start. Error:\n{err or ready_line}"
            )

    def set_variable(self, name: str, value: Any) -> None:
        self.local_namespace_cache[name] = value
        if callable(value):
            self.host_tools[name] = value
            doc = inspect.getdoc(value) or ""
            cmd = {"type": "set_tool_proxies", "tools": {name: doc}}
            self._send_cmd(cmd)
        else:
            cmd = {"type": "set_variables", "variables": {name: serialize_payload(value)}}
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
        self.process.stdout.readline()

    def execute(self, code: str) -> REPLResult:
        start_time = time.time()
        exec_cmd = {"type": "execute", "code": code}
        line = "JAZ_CMD:" + json.dumps(exec_cmd)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

        while True:
            out_line = self.process.stdout.readline()
            if not out_line:
                stderr_text = self.process.stderr.read()
                return REPLResult(
                    stdout="",
                    stderr=stderr_text,
                    error=f"Container process terminated: {stderr_text}",
                    execution_time=time.time() - start_time,
                )

            out_line = out_line.strip()
            if not out_line:
                continue

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
                    resp_dict = {"status": "ok", "result": serialize_payload(res)}
                except Exception as e:
                    resp_dict = {"status": "error", "error": str(e)}

                resp_line = "JAZ_RPC_RESP:" + json.dumps(resp_dict)
                self.process.stdin.write(resp_line + "\n")
                self.process.stdin.flush()
                continue

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
