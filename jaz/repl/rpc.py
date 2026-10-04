"""Remote Procedure Call (RPC) bridge for sandboxed REPLs (Subprocess and Container).

Enables the sandboxed code to execute in an isolated environment while transparently
calling host tools and recursive invoke() via bi-directional IPC.
"""

import base64
import json
import os
import pickle
import sys
import traceback
from typing import Any, Callable, Dict, Optional, Tuple


def serialize_payload(obj: Any) -> str:
    """Serialize an object to a base64 encoded pickle string."""
    try:
        raw = pickle.dumps(obj)
        return base64.b64encode(raw).decode("ascii")
    except Exception as e:
        # Fallback to string if not picklable
        return base64.b64encode(pickle.dumps(str(obj))).decode("ascii")


def deserialize_payload(encoded: str) -> Any:
    """Deserialize an object from a base64 encoded pickle string."""
    raw = base64.b64decode(encoded.encode("ascii"))
    return pickle.loads(raw)


class RemoteToolProxy:
    """Callable proxy running inside the sandbox that delegates calls back to the host."""

    def __init__(self, tool_name: str, docstring: Optional[str] = None):
        self._tool_name = tool_name
        self.__name__ = tool_name
        self.__doc__ = docstring

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Write RPC request to sys.__stdout__ for host process (bypassing REPL stdout capture)
        req = {
            "type": "rpc_tool_call",
            "tool_name": self._tool_name,
            "args": serialize_payload(args),
            "kwargs": serialize_payload(kwargs),
        }
        line = "JAZ_RPC_REQ:" + json.dumps(req)
        sys.__stdout__.write(line + "\n")
        sys.__stdout__.flush()

        # Read response from sys.__stdin__
        resp_line = sys.__stdin__.readline().strip()
        if not resp_line.startswith("JAZ_RPC_RESP:"):
            raise RuntimeError(f"Unexpected RPC response from host: {resp_line}")

        resp = json.loads(resp_line[len("JAZ_RPC_RESP:"):])
        if resp.get("status") == "error":
            raise RuntimeError(f"Host tool `{self._tool_name}` error: {resp.get('error')}")

        return deserialize_payload(resp["result"])

    def __repr__(self) -> str:
        return f"<RemoteToolProxy `{self._tool_name}`>"


def run_worker_loop():
    """Worker loop executed inside the child process or container."""
    from .local import LocalPythonREPL
    repl = LocalPythonREPL()

    # Signal ready to host
    sys.__stdout__.write("JAZ_WORKER_READY\n")
    sys.__stdout__.flush()

    while True:
        line = sys.__stdin__.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        if line == "JAZ_WORKER_PING":
            sys.__stdout__.write("JAZ_WORKER_PONG\n")
            sys.__stdout__.flush()
            continue

        if line == "JAZ_WORKER_EXIT":
            break

        if line.startswith("JAZ_CMD:"):
            cmd_data = json.loads(line[len("JAZ_CMD:"):])
            cmd_type = cmd_data.get("type")

            if cmd_type == "set_variables":
                serialized_vars = cmd_data.get("variables", {})
                for k, v_enc in serialized_vars.items():
                    repl.set_variable(k, deserialize_payload(v_enc))
                sys.__stdout__.write(json.dumps({"status": "ok"}) + "\n")
                sys.__stdout__.flush()

            elif cmd_type == "set_tool_proxies":
                tools = cmd_data.get("tools", {})
                for tool_name, doc in tools.items():
                    repl.set_variable(tool_name, RemoteToolProxy(tool_name, doc))
                sys.__stdout__.write(json.dumps({"status": "ok"}) + "\n")
                sys.__stdout__.flush()

            elif cmd_type == "execute":
                code = cmd_data.get("code", "")
                result = repl.execute(code)

                # Send execution result back to host
                res_dict = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "has_returned": result.has_returned,
                    "return_value": serialize_payload(result.return_value) if result.has_returned else None,
                    "error": result.error,
                    "execution_time": result.execution_time,
                }
                out_line = "JAZ_EXEC_RESULT:" + json.dumps(res_dict)
                sys.__stdout__.write(out_line + "\n")
                sys.__stdout__.flush()


if __name__ == "__main__":
    run_worker_loop()
