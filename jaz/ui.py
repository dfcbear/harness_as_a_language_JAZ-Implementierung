"""Rich terminal UI and live dashboard for JAZ agent runs."""

import sys
import time
from typing import Any, Dict, List, Optional

# Ensure Windows terminal supports UTF-8 Unicode characters (spinners, tree glyphs, arrows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .hooks.dispatcher import Hook
from .hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
    REPLExecEnter,
    REPLExecExit,
)


class TerminalLiveUI(Hook):
    """Real-time rich status display with full subagent observability."""

    def __init__(
        self,
        console: Optional[Console] = None,
        screen: bool = True,
        refresh_per_second: int = 20,
    ):
        import collections
        import threading
        self.console = console or Console()
        self.screen = screen
        self.refresh_per_second = refresh_per_second
        self.invokes: Dict[str, Dict[str, Any]] = {}
        self.active_invoke_id: Optional[str] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.live_streamed_chunks = 0
        self._chunk_timestamps: collections.deque = collections.deque(maxlen=4000)
        self.start_time = time.time()
        self.live: Optional[Live] = None
        self.activity_log: List[str] = []
        self._lock = threading.Lock()

    def _log_activity(self, message: str) -> None:
        self.activity_log.append(message)
        if len(self.activity_log) > 50:
            self.activity_log = self.activity_log[-50:]

    def start(self):
        self.live = Live(
            get_renderable=self._render_view,
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
            screen=self.screen,
            auto_refresh=True,
        )
        self.live.start()

    def stop(self):
        if self.live:
            try:
                self.live.update(self._render_view(), refresh=True)
            except Exception:
                pass
            self.live.stop()
            self.live = None

    def on_invoke_enter(self, event: InvokeEnter):
        task_preview = ""
        if "task" in event.inputs:
            raw_task = str(event.inputs["task"]).replace("\r", " ").replace("\n", " ").strip()
            task_preview = raw_task[:67] + "..." if len(raw_task) > 70 else raw_task
        elif event.inputs:
            items = [f"{k}={repr(v)[:20]}" for k, v in list(event.inputs.items())[:2]]
            task_preview = ", ".join(items)

        role = "Root Agent" if event.depth == 1 else f"Subagent L{event.depth-1}"
        t_str = time.strftime("%H:%M:%S")

        with self._lock:
            self.active_invoke_id = event.invoke_id
            self.invokes[event.invoke_id] = {
                "depth": event.depth,
                "parent_id": event.parent_id,
                "inputs": list(event.inputs.keys()),
                "task_preview": task_preview or "(unnamed task)",
                "status": "running",
                "turns": 0,
                "last_code": None,
                "last_output": None,
                "last_error": None,
                "tokens": 0,
                "start_time": time.time(),
                "query_start": None,
                "repl_start": None,
                "duration": 0.0,
                "model": "default",
                "stream_chunks": 0,
                "stream_tail": "",
                "tok_per_sec": 0.0,
            }
            self._log_activity(f"[dim]{t_str}[/dim] ▶ [bold cyan]{role}[/bold cyan] [dim]({event.invoke_id[:6]})[/dim] started: [italic]\"{escape(task_preview)}\"[/italic]")
        self._refresh()

    def on_llm_query_enter(self, event: LLMQueryEnter):
        with self._lock:
            self.active_invoke_id = event.invoke_id
            if event.invoke_id in self.invokes:
                inv = self.invokes[event.invoke_id]
                inv["status"] = "querying_llm"
                inv["query_start"] = time.time()
                inv["model"] = event.model
                inv["stream_chunks"] = 0
                inv["stream_tail"] = ""
                inv["tok_per_sec"] = 0.0
        self._refresh()

    def on_llm_stream_chunk(self, invoke_id: str, chunk: str, total_chunks: int) -> None:
        now = time.time()
        with self._lock:
            self._chunk_timestamps.append(now)
            self.live_streamed_chunks += 1
            if invoke_id in self.invokes:
                inv = self.invokes[invoke_id]
                inv["stream_chunks"] = total_chunks
                inv["stream_tail"] = (inv.get("stream_tail", "") + chunk)[-400:]
                elapsed = max(0.05, now - (inv["query_start"] or now))
                inv["tok_per_sec"] = total_chunks / elapsed

    def on_llm_query_exit(self, event: LLMQueryExit):
        t_str = time.strftime("%H:%M:%S")
        with self._lock:
            self.total_input_tokens += event.input_tokens
            self.total_output_tokens += event.output_tokens
            self.total_cost += event.cost
            if event.invoke_id in self.invokes:
                inv = self.invokes[event.invoke_id]
                chunks_from_this_inv = inv.get("stream_chunks", 0)
                self.live_streamed_chunks = max(0, self.live_streamed_chunks - chunks_from_this_inv)
                inv["tokens"] = (event.input_tokens + event.output_tokens)
                inv["status"] = "executing_repl"
                inv["repl_start"] = time.time()
                self._log_activity(
                    f"[dim]{t_str}[/dim] ⚡ [dim]{event.invoke_id[:6]}[/dim] Generated {event.output_tokens} toks in {event.duration:.1f}s"
                )
        self._refresh()

    def on_repl_exec_enter(self, event: REPLExecEnter):
        t_str = time.strftime("%H:%M:%S")
        with self._lock:
            self.active_invoke_id = event.invoke_id
            if event.invoke_id in self.invokes:
                inv = self.invokes[event.invoke_id]
                inv["turns"] = event.turn
                inv["last_code"] = event.code
                inv["last_output"] = None
                inv["last_error"] = None
                inv["status"] = "executing_repl"
                inv["repl_start"] = time.time()

            first_line = event.code.strip().split("\n")[0]
            if len(first_line) > 65:
                first_line = first_line[:62] + "..."
            self._log_activity(
                f"[dim]{t_str}[/dim] ⚙ [dim]{event.invoke_id[:6]}[/dim] Turn {event.turn} REPL: [yellow]{escape(first_line)}[/yellow]"
            )
        self._refresh()

    def on_repl_exec_exit(self, event: REPLExecExit):
        t_str = time.strftime("%H:%M:%S")
        with self._lock:
            if event.invoke_id in self.invokes:
                inv = self.invokes[event.invoke_id]
                inv["last_output"] = event.stdout
                inv["last_error"] = event.error
                if event.has_returned:
                    inv["status"] = "returned"
                    ret_prev = repr(event.return_value)[:65]
                    self._log_activity(
                        f"[dim]{t_str}[/dim] ✔ [dim]{event.invoke_id[:6]}[/dim] [bold green]RETURNED[/bold green]: {escape(ret_prev)}"
                    )
                elif event.error:
                    err_prev = event.error.strip().split("\n")[-1][:65]
                    self._log_activity(
                        f"[dim]{t_str}[/dim] ❌ [dim]{event.invoke_id[:6]}[/dim] [bold red]ERROR[/bold red]: {escape(err_prev)}"
                    )
                elif event.stdout.strip():
                    out_prev = event.stdout.strip().split("\n")[0][:65]
                    self._log_activity(
                        f"[dim]{t_str}[/dim] ↳ [dim]{event.invoke_id[:6]}[/dim] Output: {escape(out_prev)}"
                    )
        self._refresh()

    def on_invoke_exit(self, event: InvokeExit):
        t_str = time.strftime("%H:%M:%S")
        with self._lock:
            if event.invoke_id in self.invokes:
                inv = self.invokes[event.invoke_id]
                inv["status"] = "failed" if event.error else "completed"
                inv["duration"] = event.duration
                parent = inv.get("parent_id")
                if self.active_invoke_id == event.invoke_id:
                    self.active_invoke_id = parent

            status_icon = "❌ Failed" if event.error else "🏁 Completed"
            self._log_activity(
                f"[dim]{t_str}[/dim] {status_icon} [dim]{event.invoke_id[:6]}[/dim] in {event.duration:.1f}s"
            )
        self._refresh()

    def _refresh(self):
        """No-op: auto_refresh at 15fps handles all display updates smoothly.
        
        Manual refresh calls were the primary cause of flickering — they triggered
        extra renders on top of the auto-refresh cycle, causing double-rendering.
        """
        pass

    def _get_breadcrumbs(self, invoke_id: str) -> List[str]:
        crumbs = []
        curr: Optional[str] = invoke_id
        while curr and curr in self.invokes:
            info = self.invokes[curr]
            name = "🟢 Root" if info["depth"] == 1 else f"🟣 Subagent L{info['depth']-1} ({curr[:6]})"
            crumbs.append(name)
            curr = info["parent_id"]
        return list(reversed(crumbs))

    def _render_view(self) -> Group:
        SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = SPINNER_FRAMES[int(time.time() * 4) % len(SPINNER_FRAMES)]

        # 1. Take lightweight state snapshot under lock
        with self._lock:
            elapsed = time.time() - self.start_time
            now = time.time()

            # Clean up timestamps older than 1.5s
            while self._chunk_timestamps and now - self._chunk_timestamps[0] > 1.5:
                self._chunk_timestamps.popleft()

            chunk_count = len(self._chunk_timestamps)
            if chunk_count > 0:
                window = max(0.1, min(1.5, now - self._chunk_timestamps[0]))
                current_rate = chunk_count / window
            else:
                current_rate = 0.0

            total_in = self.total_input_tokens
            total_out = self.total_output_tokens + self.live_streamed_chunks
            total_toks = total_in + total_out
            invokes_snapshot = [(k, dict(v)) for k, v in self.invokes.items()]
            active_id = self.active_invoke_id
            recent_logs = list(self.activity_log[-6:]) if self.activity_log else ["[dim]Waiting for first agent action...[/dim]"]
            # Pad to exactly 6 lines for stable layout height
            while len(recent_logs) < 6:
                recent_logs.append("[dim] [/dim]")

        # 2. Render all Rich widgets outside the lock to eliminate contention & frame drops
        root_invokes = [inv for _, inv in invokes_snapshot if inv["depth"] == 1]
        sub_invokes = [inv for _, inv in invokes_snapshot if inv["depth"] > 1]

        root_active = any(
            inv["status"] in ("running", "querying_llm", "executing_repl")
            for inv in root_invokes
        )
        sub_active_count = sum(
            1 for inv in sub_invokes
            if inv["status"] in ("running", "querying_llm", "executing_repl")
        )
        sub_total_count = len(sub_invokes)
        active_count = (1 if root_active else 0) + sub_active_count
        runtime_icon = spinner if active_count > 0 else "🏁"

        # 1. Header Status Table (Separates Root Agent from Subagents clearly)
        stats_table = Table.grid(expand=True, padding=(0, 2))
        stats_table.add_column("K1", style="bold cyan")
        stats_table.add_column("V1", style="bold white")
        stats_table.add_column("K2", style="bold cyan")
        stats_table.add_column("V2", style="bold white")
        stats_table.add_column("K3", style="bold cyan")
        stats_table.add_column("V3", style="bold white")
        stats_table.add_column("K4", style="bold cyan")
        stats_table.add_column("V4", style="bold white")
        stats_table.add_column("K5", style="bold cyan")
        stats_table.add_column("V5", style="bold white")

        root_status_str = f"[bold green]{spinner} aktiv[/bold green]" if root_active else "[dim]idle[/dim]"
        if sub_total_count == 0:
            sub_status_str = "[dim]0 gestartet[/dim]"
        else:
            sub_status_str = f"[bold magenta]{sub_active_count} aktiv[/bold magenta] [dim]({sub_total_count} total)[/dim]"

        rate_str = f"[bold green]⚡ {current_rate:,.1f} t/s[/bold green]" if current_rate > 0 else "[dim]0.0 t/s[/dim]"

        stats_table.add_row(
            "Runtime:", f"{runtime_icon} {elapsed:.1f}s",
            "Tokens:", f"{total_toks:,} (in: {total_in:,}, out: {total_out:,})",
            "Tokenrate:", rate_str,
            "Root Agent:", root_status_str,
            "Subagenten:", sub_status_str,
        )

        header_panel = Panel(
            stats_table,
            title="[bold green]JAZ Agent Session[/bold green]",
            border_style="green",
        )

        # 2. Invocation Hierarchy Tree (Auto-compacting to prevent terminal screen overflow)
        tree = Tree("[bold white]Agents & Sub-invokes Hierarchy[/bold white]")
        node_map: Dict[str, Any] = {}

        MAX_SUBAGENTS_DISPLAY = 6
        active_sub_ids = [
            inv_id for inv_id, info in invokes_snapshot
            if info["depth"] > 1 and info["status"] in ("running", "querying_llm", "executing_repl")
        ]
        completed_sub_count = sum(
            1 for _, info in invokes_snapshot
            if info["depth"] > 1 and info["status"] in ("completed", "returned")
        )
        failed_sub_count = sum(
            1 for _, info in invokes_snapshot
            if info["depth"] > 1 and info["status"] == "failed"
        )

        shown_sub_ids = set(active_sub_ids[:MAX_SUBAGENTS_DISPLAY])
        if len(shown_sub_ids) < MAX_SUBAGENTS_DISPLAY:
            for inv_id, info in reversed(invokes_snapshot):
                if info["depth"] > 1 and inv_id not in shown_sub_ids:
                    shown_sub_ids.add(inv_id)
                    if len(shown_sub_ids) >= MAX_SUBAGENTS_DISPLAY:
                        break

        hidden_sub_count = len(sub_invokes) - len(shown_sub_ids)

        for inv_id, info in invokes_snapshot:
            if info["depth"] > 1 and inv_id not in shown_sub_ids:
                continue

            is_active = info["status"] in ("running", "querying_llm", "executing_repl")
            active_badge = f"[bold green]{spinner} AKTIV [/bold green]" if is_active else ""

            depth = info["depth"]
            role_text = "[bold cyan]🟢 Root Agent[/bold cyan]" if depth == 1 else f"[bold magenta]🟣 Subagent L{depth-1}[/bold magenta]"

            status = info["status"]
            if status == "querying_llm":
                q_sec = time.time() - (info["query_start"] or time.time())
                tok_spd = f", {info['tok_per_sec']:.1f} t/s" if info.get("tok_per_sec", 0) > 0 else ""
                chunks = info.get("stream_chunks", 0)
                if chunks == 0:
                    status_badge = f"[bold yellow]{spinner} [Connecting / Waiting: {q_sec:.1f}s][/bold yellow]"
                else:
                    status_badge = f"[bold yellow]{spinner} [Streaming: {q_sec:.1f}s ({chunks} chunks{tok_spd})][/bold yellow]"
            elif status == "executing_repl":
                r_sec = time.time() - (info.get("repl_start") or time.time())
                status_badge = f"[bold cyan]{spinner} [REPL Turn {info['turns']} ({r_sec:.1f}s)][/bold cyan]"
            elif status == "returned":
                status_badge = "[bold green][Returned][/bold green]"
            elif status == "completed":
                status_badge = f"[bold green][Done in {info.get('duration', 0.0):.1f}s][/bold green]"
            elif status == "failed":
                status_badge = "[bold red][Failed][/bold red]"
            else:
                status_badge = f"[dim][{status}][/dim]"

            task_txt = f"[italic white]\"{escape(info['task_preview'])}\"[/italic white]"
            tok_txt = f"[dim]({info['tokens']:,} toks)[/dim]" if info["tokens"] > 0 else ""

            line = f"{active_badge}{role_text} [dim]{inv_id[:6]}[/dim] {status_badge} {task_txt} {tok_txt}"

            parent = info["parent_id"]
            if parent and parent in node_map:
                node = node_map[parent].add(line)
            else:
                node = tree.add(line)
            node_map[inv_id] = node

        if hidden_sub_count > 0:
            root_node = None
            for inv_id, info in invokes_snapshot:
                if info["depth"] == 1 and inv_id in node_map:
                    root_node = node_map[inv_id]
                    break
            target_node = root_node or tree
            fail_str = f", [red]{failed_sub_count} fehlgeschlagen[/red]" if failed_sub_count > 0 else ""
            target_node.add(
                f"[dim italic]↳ ... und {hidden_sub_count} weitere Subagenten "
                f"([bold magenta]{sub_active_count} aktiv[/bold magenta], "
                f"[green]{completed_sub_count} abgeschlossen[/green]{fail_str})[/dim italic]"
            )

        if sub_total_count == 0 and root_invokes:
            tree.add("[dim italic]↳ (Noch keine Subagenten gestartet – erscheinen hier live mit 🟣, sobald invoke(...) aufgerufen wird)[/dim italic]")

        tree_panel = Panel(tree, title="Hierarchy & Progress", border_style="blue")

        # 3. Live Chronological Activity Log (Last 6 entries)
        activity_table = Table.grid(expand=True)
        activity_table.add_column("event")
        for log_entry in recent_logs:
            activity_table.add_row(log_entry)
        activity_panel = Panel(activity_table, title="Real-Time Activity Feed", border_style="magenta")

        # 4. Active Execution Details Box
        invokes_dict = dict(invokes_snapshot)
        if not active_id or active_id not in invokes_dict:
            active_id = invokes_snapshot[-1][0] if invokes_snapshot else None

        items: List[Any] = []
        is_subagent_active = False
        active_role_str = "Agent"
        if active_id and active_id in invokes_dict:
            act_info = invokes_dict[active_id]
            is_subagent_active = act_info["depth"] > 1
            depth_str = f"L{act_info['depth']-1}" if is_subagent_active else "Root"
            role_badge = f"🟣 Subagent {depth_str}" if is_subagent_active else "🟢 Root Agent"
            active_role_str = f"{role_badge} ({active_id[:6]}) Turn {act_info.get('turns', 1)}"

            crumbs = self._get_breadcrumbs(active_id)
            breadcrumb_text = " ➔ ".join(crumbs)

            items.append(Text.assemble(
                ("Role: ", "bold magenta" if is_subagent_active else "bold cyan"),
                (f"{role_badge}  |  ", "bold white"),
                ("Path: ", "bold cyan"),
                (breadcrumb_text, "bold white"),
                ("  |  Task: ", "bold cyan"),
                (act_info["task_preview"], "italic white"),
            ))

            if act_info["status"] == "querying_llm":
                q_sec = time.time() - (act_info["query_start"] or time.time())
                chunks = act_info.get("stream_chunks", 0)
                tok_speed = act_info.get("tok_per_sec", 0.0)
                stream_tail = act_info.get("stream_tail", "").strip()

                speed_str = f" | {chunks} chunks | {tok_speed:.1f} tok/s" if chunks > 0 else " | waiting for first token..."
                gen_lines = [
                    f"[bold yellow]{spinner} Model ({act_info['model']}) {'streaming response...' if chunks > 0 else 'connecting to endpoint...'} ({q_sec:.1f}s{speed_str})[/bold yellow]"
                ]
                if stream_tail:
                    gen_lines.append(f"[dim white]… {escape(stream_tail)}[/dim white]")
                items.append(Panel("\n".join(gen_lines), border_style="yellow", title="Live Generation Stream"))

            elif act_info["status"] == "executing_repl":
                r_sec = time.time() - (act_info.get("repl_start") or time.time())
                items.append(Panel(
                    f"[bold cyan]{spinner} Executing Python code in REPL... ({r_sec:.1f}s)[/bold cyan]",
                    border_style="cyan",
                    title=f"Turn {act_info['turns']} REPL Execution",
                ))

            if act_info["last_code"]:
                raw_code = act_info["last_code"].strip().split("\n")
                if len(raw_code) > 14:
                    display_code = "\n".join(raw_code[:6] + [f"    # ... ({len(raw_code)-12} lines omitted) ..."] + raw_code[-6:])
                else:
                    display_code = "\n".join(raw_code)
                syntax = Syntax(display_code, "python", theme="monokai", line_numbers=True)
                items.append(Panel(
                    syntax,
                    title=f"{'Root' if act_info['depth'] == 1 else 'Subagent'} Turn {act_info['turns']} Executed Code ({len(raw_code)} lines)",
                    border_style="yellow",
                ))

            if act_info["last_output"]:
                out_text = act_info["last_output"].strip()
                if len(out_text) > 1000:
                    out_text = out_text[:997] + "... [truncated]"
                items.append(Panel(Text(out_text), title="Output", border_style="dim"))

            if act_info["last_error"]:
                items.append(Panel(Text(act_info["last_error"], style="bold red"), title="Error", border_style="red"))
        else:
            items.append(Text("Waiting for first step...", style="dim"))

        # Ensure exec_panel has minimum height to avoid layout shift
        for _ in range(3):
            items.append(Text(""))
        exec_group = Group(*items)
        panel_color = "magenta" if is_subagent_active else "cyan"
        exec_panel = Panel(exec_group, title=f"Active Execution: {active_role_str}", border_style=panel_color)

        return Group(header_panel, tree_panel, activity_panel, exec_panel)

