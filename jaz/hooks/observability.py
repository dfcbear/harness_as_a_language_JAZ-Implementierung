"""Observability hooks: logging, trajectory recording, and JSONL event streaming."""

import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from .dispatcher import Hook
from .events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
    REPLExecEnter,
    REPLExecExit,
)


class PrintLogger(Hook):
    """Prints agent execution steps and tool calls to stdout."""

    def __init__(self, verbose: bool = True, stream=sys.stdout):
        super().__init__()
        self.verbose = verbose
        self.stream = stream

    def on_invoke_enter(self, event: InvokeEnter):
        indent = "  " * event.depth
        print(f"{indent}▶ [invoke] id={event.invoke_id[:8]} depth={event.depth} inputs={list(event.inputs.keys())}", file=self.stream)

    def on_repl_exec_enter(self, event: REPLExecEnter):
        if self.verbose:
            code_line = event.code.strip().replace("\n", " ¶ ")
            if len(code_line) > 100:
                code_line = code_line[:97] + "..."
            print(f"  [REPL turn {event.turn}] {code_line}", file=self.stream)

    def on_repl_exec_exit(self, event: REPLExecExit):
        if self.verbose and event.stdout.strip():
            out_preview = event.stdout.strip().replace("\n", " ¶ ")
            if len(out_preview) > 120:
                out_preview = out_preview[:117] + "..."
            print(f"    ↳ Output: {out_preview}", file=self.stream)
        if event.error:
            print(f"    ❌ Error: {event.error}", file=self.stream)
        if event.has_returned:
            print(f"    ✔ Returned: {repr(event.return_value)[:100]}", file=self.stream)

    def on_invoke_exit(self, event: InvokeExit):
        if event.error:
            print(f"  ✖ [invoke exited with error] {event.error}", file=self.stream)
        else:
            print(f"  ■ [invoke completed in {event.duration:.2f}s]", file=self.stream)


class FileLogger(Hook):
    """Logs human-readable trace to a file."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "a", encoding="utf-8")

    def _log(self, text: str):
        self._file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        self._file.flush()

    def on_invoke_enter(self, event: InvokeEnter):
        self._log(f"INVOKE ENTER id={event.invoke_id} depth={event.depth} inputs={list(event.inputs.keys())}")

    def on_repl_exec_enter(self, event: REPLExecEnter):
        self._log(f"TURN {event.turn} CODE:\n{event.code}")

    def on_repl_exec_exit(self, event: REPLExecExit):
        self._log(f"TURN {event.turn} OUT:\n{event.stdout}")
        if event.error:
            self._log(f"TURN {event.turn} ERROR: {event.error}")
        if event.has_returned:
            self._log(f"TURN {event.turn} RETURN: {event.return_value}")

    def on_invoke_exit(self, event: InvokeExit):
        self._log(f"INVOKE EXIT id={event.invoke_id} duration={event.duration:.2f}s")
        self._file.flush()


class JsonlEventEmitter(Hook):
    """Streams structured JSON events to an events.jsonl file for live dashboards and analysis."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = open(self.filepath, "a", encoding="utf-8")

    def _emit(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "type": event_type,
            "timestamp": time.time(),
            **data,
        }
        with self._lock:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()

    def on_invoke_enter(self, event: InvokeEnter):
        inputs_summary = {k: str(type(v).__name__) for k, v in event.inputs.items()}
        self._emit("invoke_enter", {
            "invoke_id": event.invoke_id,
            "parent_id": event.parent_id,
            "depth": event.depth,
            "inputs": inputs_summary,
        })

    def on_llm_query_enter(self, event: LLMQueryEnter):
        self._emit("llm_query_enter", {
            "invoke_id": event.invoke_id,
            "model": event.model,
            "message_count": len(event.messages),
        })

    def on_llm_query_exit(self, event: LLMQueryExit):
        self._emit("llm_query_exit", {
            "invoke_id": event.invoke_id,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "cost": event.cost,
            "duration": event.duration,
            "error": str(event.error) if event.error else None,
        })

    def on_repl_exec_enter(self, event: REPLExecEnter):
        self._emit("repl_exec_enter", {
            "invoke_id": event.invoke_id,
            "turn": event.turn,
            "code": event.code,
        })

    def on_repl_exec_exit(self, event: REPLExecExit):
        self._emit("repl_exec_exit", {
            "invoke_id": event.invoke_id,
            "turn": event.turn,
            "stdout": event.stdout,
            "stderr": event.stderr,
            "has_returned": event.has_returned,
            "return_preview": repr(event.return_value)[:200] if event.has_returned else None,
            "error": event.error,
            "duration": event.duration,
        })

    def on_invoke_exit(self, event: InvokeExit):
        self._emit("invoke_exit", {
            "invoke_id": event.invoke_id,
            "duration": event.duration,
            "has_error": event.error is not None,
            "error": str(event.error) if event.error else None,
        })


class TrajectoryRecorder(Hook):
    """Records the full sequence of actions, outputs, and LLM calls."""

    def __init__(self, filepath: Optional[str] = None):
        super().__init__()
        self.filepath = Path(filepath) if filepath else None
        self.records: List[Dict[str, Any]] = []

    def on_repl_exec_exit(self, event: REPLExecExit):
        self.records.append({
            "turn": event.turn,
            "stdout": event.stdout,
            "has_returned": event.has_returned,
            "return_value": repr(event.return_value) if event.has_returned else None,
            "error": event.error,
        })

    def on_invoke_exit(self, event: InvokeExit):
        if self.filepath:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, default=str)


class TrajectoryDirectoryRecorder(Hook):
    """Records trajectories into a directory partitioned by invoke_id."""

    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.current_records: Dict[str, List[Dict[str, Any]]] = {}

    def on_invoke_enter(self, event: InvokeEnter):
        self.current_records[event.invoke_id] = []

    def on_repl_exec_exit(self, event: REPLExecExit):
        if event.invoke_id in self.current_records:
            self.current_records[event.invoke_id].append({
                "turn": event.turn,
                "stdout": event.stdout,
                "error": event.error,
                "has_returned": event.has_returned,
            })

    def on_invoke_exit(self, event: InvokeExit):
        records = self.current_records.pop(event.invoke_id, [])
        file_path = self.dirpath / f"trajectory_{event.invoke_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
