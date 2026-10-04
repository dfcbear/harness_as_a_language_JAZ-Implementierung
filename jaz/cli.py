"""Command-line interface for running and monitoring JAZ agent tasks."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

# Ensure UTF-8 output on Windows consoles to prevent cp1252 encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console

from .config import Config, load_config_from_env_or_file, set_default_config
from .core import invoke
from .hooks.observability import JsonlEventEmitter, TrajectoryRecorder
from .hooks.replay import TrajectoryReplay
from .hooks.resources import BudgetPool, IterationLimit
from .llm import OpenAICompatibleLLM
from .repl import ContainerREPL, LocalPythonREPL, SubprocessREPL
from .scope import scope
from .ui import TerminalLiveUI

console = Console()


def load_tools_from_file(filepath: str) -> dict:
    """Dynamically load functions from a python script to expose as tools."""
    p = Path(filepath).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Tools file not found: {filepath}")

    module_name = f"jaz_custom_tools_{p.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tools = {}
    for attr in dir(mod):
        if not attr.startswith("_"):
            val = getattr(mod, attr)
            if callable(val):
                tools[attr] = val
    return tools


def cmd_run(args: argparse.Namespace):
    # 1. Load configuration
    cfg = load_config_from_env_or_file(args.config)

    # CLI sandbox override
    if args.sandbox:
        mode = args.sandbox.lower()
        if mode == "subprocess":
            cfg.repl = SubprocessREPL()
        elif mode in ("container", "docker", "podman"):
            cfg.repl = ContainerREPL(runtime="podman" if mode == "podman" else "docker")
        elif mode == "direct":
            cfg.repl = LocalPythonREPL()

    set_default_config(cfg)

    # 2. Setup run directory and event emitter
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(args.run_dir or f"runs/{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Route REPL workspace and imports directly to run_dir so project root stays clean
    if hasattr(cfg.repl, "workspace_dir"):
        cfg.repl.workspace_dir = run_dir
    elif isinstance(cfg.repl, LocalPythonREPL):
        cfg.repl = LocalPythonREPL(workspace_dir=run_dir)
    set_default_config(cfg)

    events_file = run_dir / "events.jsonl"
    trajectory_file = run_dir / "trajectory.json"

    jsonl_emitter = JsonlEventEmitter(str(events_file))
    traj_recorder = TrajectoryRecorder(str(trajectory_file))

    # 3. Assemble inputs
    inputs = {"task": args.task}

    # Key=value inputs
    if args.input:
        for item in args.input:
            if "=" in item:
                k, v = item.split("=", 1)
                inputs[k.strip()] = v.strip()

    # File inputs
    if args.file_input:
        for item in args.file_input:
            if "=" in item:
                k, fpath = item.split("=", 1)
                p = Path(fpath.strip())
                if p.is_file():
                    inputs[k.strip()] = p.read_text(encoding="utf-8")

    # 4. Load tools
    custom_tools = {}
    if args.tools:
        for t_file in args.tools:
            loaded = load_tools_from_file(t_file)
            custom_tools.update(loaded)

    # 5. Build hooks
    hooks = [jsonl_emitter, traj_recorder]
    if args.linter:
        from .hooks.linter import PythonLinterHook
        hooks.append(PythonLinterHook(auto_fix=True, check_undefined_tools=True))
    if args.max_iterations:
        hooks.append(IterationLimit(args.max_iterations))
    if args.max_cost:
        hooks.append(BudgetPool(max_cost=args.max_cost))
    if cfg.llm.context_window:
        from .hooks.resources import ContextWindowWarning
        hooks.append(ContextWindowWarning(threshold_tokens=cfg.llm.context_window, threshold_ratio=0.70))

    # Terminal Status UI / Logging
    ui_hook = None
    if getattr(args, "plain", False):
        from .hooks.observability import PrintLogger
        hooks.append(PrintLogger(verbose=True))
    elif not args.quiet:
        ui_hook = TerminalLiveUI(console)
        hooks.append(ui_hook)

    console.print(f"[green]Starting JAZ run {run_id}[/green]")
    console.print(f"  Endpoint: {cfg.llm.base_url} | Model: {cfg.llm.model} | Sandbox: {type(cfg.repl).__name__}")
    console.print(f"  Run Directory: [cyan]{run_dir}[/cyan]")
    console.print(f"  Events streaming to: [cyan]{events_file}[/cyan]\n")

    if ui_hook:
        ui_hook.start()

    result = None
    try:
        with scope(**custom_tools):
            result = invoke(*hooks, **inputs)
    finally:
        if ui_hook:
            ui_hook.stop()

    # Save output artifacts to run_dir
    result_file = run_dir / "result.txt"
    result_file.write_text(str(result) if result is not None else "", encoding="utf-8")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(str(result), encoding="utf-8")
        if not out_path.is_absolute() and out_path.parent == Path("."):
            (run_dir / out_path.name).write_text(str(result), encoding="utf-8")

    console.print("\n[bold green]Run Completed Successfully![/bold green]")
    console.print(f"[bold cyan]Final Result:[/bold cyan] {result}\n")
    console.print(f"Trace saved to: {trajectory_file}")

    # Report generated output files in run folder
    artifacts = [f for f in run_dir.iterdir() if f.is_file() and f.name not in ("events.jsonl", "trajectory.json")]
    if artifacts:
        console.print(f"\n[bold green]Output files in run folder:[/bold green] [cyan]{run_dir}[/cyan]")
        for f in sorted(artifacts):
            console.print(f"  • [yellow]{f.name}[/yellow] ({f.stat().st_size / 1024.0:.1f} KB)")


def cmd_check(args: argparse.Namespace):
    cfg = load_config_from_env_or_file(args.config)
    console.print("[bold yellow]Checking LLM Endpoint...[/bold yellow]")
    console.print(f"  Base URL: {cfg.llm.base_url}")
    console.print(f"  Model:    {cfg.llm.model}")
    console.print(f"  Context:  {cfg.llm.context_window} tokens")

    start_time = time.time()
    try:
        resp = cfg.llm.query([
            {"role": "system", "content": "Respond with 'JAZ_OK'"},
            {"role": "user", "content": "Ping"},
        ])
        latency = (time.time() - start_time) * 1000
        console.print(f"[bold green][OK] Connection Successful![/bold green] (Latency: {latency:.1f}ms)")
        console.print(f"  Model responded: {resp.content.strip()[:100]}")
    except Exception as e:
        console.print(f"[bold red][FAIL] Connection Failed:[/bold red] {e}")
        sys.exit(1)


def cmd_replay(args: argparse.Namespace):
    p = Path(args.trajectory)
    if not p.is_file():
        console.print(f"[red]Trajectory file not found: {p}[/red]")
        sys.exit(1)

    console.print(f"[yellow]Replaying run from {p}...[/yellow]")
    replay_hook = TrajectoryReplay(p)
    res = invoke(replay_hook, task="Replay session")
    console.print(f"[bold green]Replay Result:[/bold green] {res}")


def main():
    parser = argparse.ArgumentParser(
        prog="jaz",
        description="JAZ: A Minimalist Agent Framework With Maximal Expressivity",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # RUN command
    p_run = subparsers.add_parser("run", help="Run an agent task")
    p_run.add_argument("task", help="Natural language task description")
    p_run.add_argument("-i", "--input", action="append", help="Pass key=value inputs")
    p_run.add_argument("-f", "--file-input", action="append", help="Pass key=filepath inputs")
    p_run.add_argument("-t", "--tools", action="append", help="Path to Python script containing tool functions")
    p_run.add_argument("-c", "--config", help="Path to jaz.toml config file")
    p_run.add_argument(
        "--sandbox",
        choices=["direct", "subprocess", "container", "docker", "podman"],
        help="Sandbox execution mode",
    )
    p_run.add_argument("--linter", action="store_true", help="Enable pre-execution AST linter and auto-repair")
    p_run.add_argument("--max-iterations", type=int, default=50, help="Max iterations per invoke")
    p_run.add_argument("--max-cost", type=float, help="Max dollar budget")
    p_run.add_argument("--run-dir", help="Directory to save run logs and events.jsonl")
    p_run.add_argument("-o", "--output", help="Write final return value to file")
    p_run.add_argument("--quiet", action="store_true", help="Disable live terminal dashboard")
    p_run.add_argument("--plain", "--log", dest="plain", action="store_true", help="Print progressive scrolling text instead of live dashboard buffer")
    p_run.set_defaults(func=cmd_run)

    # CHECK command
    p_check = subparsers.add_parser("check", help="Verify connection to LLM API endpoint")
    p_check.add_argument("-c", "--config", help="Path to jaz.toml config file")
    p_check.set_defaults(func=cmd_check)

    # REPLAY command
    p_replay = subparsers.add_parser("replay", help="Replay a prior trajectory")
    p_replay.add_argument("trajectory", help="Path to trajectory.json")
    p_replay.set_defaults(func=cmd_replay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
