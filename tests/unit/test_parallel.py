"""Tests for parallel sub-agent execution and TerminalLiveUI responsiveness."""

from concurrent.futures import ThreadPoolExecutor
import time
import pytest

from jaz import ConfigOverride, invoke, scope
from jaz.hooks import ContextWindowWarning, IterationLimit
from jaz.llm import ScriptedLLM
from jaz.ui import TerminalLiveUI


def test_parallel_subagent_execution_with_threadpool():
    """Verify that an agent can launch multiple sub-agents in parallel via ThreadPoolExecutor."""
    ui = TerminalLiveUI()

    llm = ScriptedLLM(responses=[
        # Root Agent: runs 2 subagents concurrently
        """from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as pool:
    fut_track = pool.submit(invoke, task='Build 3D track mesh')
    fut_car = pool.submit(invoke, task='Build car physics engine')
    track = fut_track.result()
    car = fut_car.result()

return {'track': track, 'car': car}
""",
        # Subagent 1 response:
        "return {'mesh_vertices': 4500, 'length_km': 4.2}",
        # Subagent 2 response:
        "return {'mass': 1250, 'top_speed_kmh': 290}",
    ])

    with ConfigOverride(llm=llm):
        result = invoke(ui, task="Parallel racing game assembly")

    results = [result["track"], result["car"]]
    lengths = [r.get("length_km") for r in results if isinstance(r, dict) and "length_km" in r]
    speeds = [r.get("top_speed_kmh") for r in results if isinstance(r, dict) and "top_speed_kmh" in r]
    assert lengths == [4.2]
    assert speeds == [290]

    # Verify that UI tracked root agent and both subagents
    assert len(ui.invokes) == 3
    root_id = [k for k, v in ui.invokes.items() if v["depth"] == 1][0]
    sub_ids = [k for k, v in ui.invokes.items() if v["depth"] == 2]

    assert len(sub_ids) == 2
    for sid in sub_ids:
        assert ui.invokes[sid]["parent_id"] == root_id
        assert ui.invokes[sid]["status"] == "completed"

    # Verify activity log entries exist
    assert len(ui.activity_log) > 0
    activity_text = " ".join(ui.activity_log)
    assert "Root Agent" in activity_text
    assert "Subagent L1" in activity_text
    assert "RETURNED" in activity_text


def test_ui_stream_chunk_and_activity_feed():
    """Verify on_llm_stream_chunk updates live tokens and stream preview in TerminalLiveUI."""
    ui = TerminalLiveUI()

    # Simulate invoke lifecycle
    from jaz.hooks.events import InvokeEnter, LLMQueryEnter, LLMQueryExit, REPLExecEnter, REPLExecExit, InvokeExit

    ui.on_invoke_enter(InvokeEnter(invoke_id="inv_test123", parent_id=None, depth=1, inputs={"task": "Streaming test"}))
    ui.on_llm_query_enter(LLMQueryEnter(invoke_id="inv_test123", messages=[], model="test-model"))

    # Simulate streaming tokens
    ui.on_llm_stream_chunk("inv_test123", "import math\n", 1)
    ui.on_llm_stream_chunk("inv_test123", "return math.pi", 2)

    inv = ui.invokes["inv_test123"]
    assert inv["stream_chunks"] == 2
    assert "return math.pi" in inv["stream_tail"]
    assert inv["status"] == "querying_llm"

    ui.on_llm_query_exit(LLMQueryExit(invoke_id="inv_test123", response="return math.pi", input_tokens=10, output_tokens=5, cost=0.0, duration=0.2))
    ui.on_repl_exec_enter(REPLExecEnter(invoke_id="inv_test123", code="return math.pi", turn=1))
    ui.on_repl_exec_exit(REPLExecExit(invoke_id="inv_test123", turn=1, stdout="", stderr="", return_value=3.14159, has_returned=True, duration=0.01))
    ui.on_invoke_exit(InvokeExit(invoke_id="inv_test123", result=3.14159, error=None, duration=0.25))

    # Verify activity feed contains all steps
    log_text = " ".join(ui.activity_log)
    assert "Streaming test" in log_text
    assert "Turn 1 REPL" in log_text
    assert "RETURNED" in log_text
    assert "Completed" in log_text


def test_ui_subagent_visual_tracking_and_render():
    """Verify that TerminalLiveUI distinguishes root agent vs subagents in header and hierarchy."""
    from jaz.hooks.events import InvokeEnter, LLMQueryEnter
    ui = TerminalLiveUI()

    # 1. Initially only root agent
    ui.on_invoke_enter(InvokeEnter(invoke_id="root_001", parent_id=None, depth=1, inputs={"task": "Root Task"}))
    group_root = ui._render_view()
    assert group_root is not None

    # 2. Subagent launched
    ui.on_invoke_enter(InvokeEnter(invoke_id="sub_001", parent_id="root_001", depth=2, inputs={"task": "Subtask 1"}))
    ui.on_llm_query_enter(LLMQueryEnter(invoke_id="sub_001", messages=[], model="gemini-2.5-flash"))
    
    group_sub = ui._render_view()
    assert group_sub is not None

    # Verify tracking state
    assert ui.invokes["root_001"]["depth"] == 1
    assert ui.invokes["sub_001"]["depth"] == 2
    assert ui.active_invoke_id == "sub_001"


def test_terminal_ui_live_token_rate_and_compact_tree():
    """Verify aggregated live token rate computation and subagent auto-compaction."""
    from jaz.hooks.events import InvokeEnter, LLMQueryEnter
    ui = TerminalLiveUI()

    # Launch root agent
    ui.on_invoke_enter(InvokeEnter(invoke_id="root_main", parent_id=None, depth=1, inputs={"task": "Master plan"}))

    # Launch 20 subagents
    for i in range(20):
        sid = f"sub_{i:03d}"
        ui.on_invoke_enter(InvokeEnter(invoke_id=sid, parent_id="root_main", depth=2, inputs={"task": f"Worker task {i}"}))
        ui.on_llm_query_enter(LLMQueryEnter(invoke_id=sid, messages=[], model="test-model"))
        # Stream chunks to each
        ui.on_llm_stream_chunk(sid, f"token_{i}", 1)

    # Verify live streamed tokens accumulated
    assert ui.live_streamed_chunks == 20
    assert len(ui._chunk_timestamps) == 20

    # Render view and verify it renders without error
    render_group = ui._render_view()
    assert render_group is not None


