"""Core agent loop and the `invoke` primitive for JAZ."""

import time
from typing import Any, Dict, List, Optional
import uuid

from .config import Config, ConfigOverride, get_default_config
from .display import format_available_variables_doc, format_invoke_signature
from .exceptions import AbortError, FatalError, JazError
from .history import History
from .hooks.actions import Abort, AddMessages, DropMessages, DropVariables, HookAction
from .hooks.dispatcher import Hook, HookDispatcher
from .hooks.events import (
    InvokeEnter,
    InvokeExit,
    LLMQueryEnter,
    LLMQueryExit,
    REPLExecEnter,
    REPLExecExit,
)
from .hooks.replay import TrajectoryReplay
from .prompts import get_default_system_prompt
from .repl import BaseREPL, ContainerREPL, LocalPythonREPL, REPLResult, SubprocessREPL
from .scope import (
    _call_depth,
    _parent_invoke_id,
    _scoped_config,
    _scoped_hooks,
    get_current_depth,
    get_parent_invoke_id,
    get_scoped_config,
    get_scoped_hooks,
    get_scoped_variables,
    scope,
)


def _instantiate_repl_like(prototype_repl: BaseREPL) -> BaseREPL:
    """Create a fresh REPL session matching the prototype configuration."""
    ws_dir = getattr(prototype_repl, "workspace_dir", None)
    if isinstance(prototype_repl, SubprocessREPL):
        return SubprocessREPL(
            workspace_dir=str(ws_dir) if ws_dir else None,
            timeout=prototype_repl.timeout,
            filter_env=prototype_repl.filter_env,
        )
    if isinstance(prototype_repl, ContainerREPL):
        return ContainerREPL(
            image=prototype_repl.image,
            runtime=prototype_repl.runtime,
            timeout=prototype_repl.timeout,
            network_enabled=prototype_repl.network_enabled,
        )
    return LocalPythonREPL(workspace_dir=ws_dir)


def invoke(*hooks_or_overrides: Any, **inputs: Any) -> Any:
    """The central JAZ language primitive.
    
    Treats an LLM query as a function call whose body is written by the LLM
    at runtime in a Python REPL.
    
    Args:
        *hooks_or_overrides: Local Hook or ConfigOverride instances.
        **inputs: Named inputs passed to the agent. Treated as variables in the REPL.
        
    Returns:
        The value returned by the agent via top-level `return <expr>`.
    """
    invoke_id = f"inv_{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    # 1. Resolve hierarchy & depth
    parent_id = get_parent_invoke_id()
    depth = get_current_depth() + 1
    depth_token = _call_depth.set(depth)
    parent_token = _parent_invoke_id.set(invoke_id)

    # 2. Separate local hooks and local config overrides
    local_hooks: List[Hook] = []
    local_override: Optional[ConfigOverride] = None
    for arg in hooks_or_overrides:
        if isinstance(arg, Hook):
            local_hooks.append(arg)
        elif isinstance(arg, ConfigOverride):
            local_override = arg
        else:
            raise TypeError(f"Positional arguments to invoke must be Hook or ConfigOverride, got {type(arg)}")

    # 3. Combine with scoped hooks, variables, and config
    active_hooks = list(get_scoped_hooks())
    for h in local_hooks:
        if h not in active_hooks:
            active_hooks.append(h)
    dispatcher = HookDispatcher(active_hooks)

    all_inputs: Dict[str, Any] = {**get_scoped_variables(), **inputs}

    # Resolve configuration
    base_config = get_default_config()
    scoped_override = get_scoped_config()
    current_config = base_config
    if scoped_override:
        current_config = scoped_override.apply_to(current_config)
    if local_override:
        current_config = local_override.apply_to(current_config)

    llm = current_config.llm
    protocol = current_config.protocol
    repl = _instantiate_repl_like(current_config.repl)

    # Dynamically scope active hooks and current configuration so recursive sub-invokes inherit them
    hooks_token = _scoped_hooks.set(active_hooks)
    config_token = _scoped_config.set(
        ConfigOverride(llm=current_config.llm, repl=current_config.repl, protocol=current_config.protocol)
    )

    # 4. Initialize REPL state
    history = History()
    for name, val in all_inputs.items():
        repl.set_variable(name, val)
    repl.set_variable("__history__", history)
    repl.set_variable("scope", scope)
    ws_dir = getattr(repl, "workspace_dir", None)
    if ws_dir:
        repl.set_variable("workspace_dir", str(ws_dir))
        repl.set_variable("output_dir", str(ws_dir))

    # Subagent delegation function: recursive invoke call
    # Capture current context so if sub_invoke is submitted to worker threads (e.g. ThreadPoolExecutor),
    # the worker thread seamlessly inherits all dynamically scoped hooks, config, and depth!
    import contextvars
    parent_ctx = contextvars.copy_context()

    def sub_invoke(*sub_args: Any, **sub_inputs: Any) -> Any:
        return parent_ctx.copy().run(invoke, *sub_args, **sub_inputs)

    sub_invoke.__name__ = "invoke"
    sub_invoke.__doc__ = (
        "Recursively launch a sub-agent. Arguments passed are variables in the sub-agent's REPL."
    )
    repl.set_variable("invoke", sub_invoke)

    # 5. Dispatch InvokeEnter event
    enter_actions = dispatcher.dispatch_invoke_enter(
        InvokeEnter(invoke_id=invoke_id, parent_id=parent_id, depth=depth, inputs=all_inputs)
    )

    # 6. Build initial messages
    system_prompt = get_default_system_prompt()
    fn_sig = format_invoke_signature(all_inputs)
    vars_doc = format_available_variables_doc(all_inputs)

    user_start_msg = (
        f"You are invoked as:\n\n```python\n{fn_sig}\n```\n\n"
        f"{vars_doc}\n\n"
        f"Implement the body of this invoke call. Write Python code now:"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_start_msg},
    ]

    # Process any initial hook actions
    _apply_hook_actions(enter_actions, messages, repl)

    turn = 0
    final_result = None
    execution_error: Optional[Exception] = None

    try:
        while True:
            turn += 1

            # Check if TrajectoryReplay is active
            replay_code: Optional[str] = None
            for hook in active_hooks:
                if isinstance(hook, TrajectoryReplay):
                    code_cand = hook.get_next_response()
                    if code_cand:
                        replay_code = code_cand
                        break

            # LLM Query Phase
            q_enter_actions = dispatcher.dispatch_llm_query_enter(
                LLMQueryEnter(invoke_id=invoke_id, messages=messages, model=llm.model)
            )
            _apply_hook_actions(q_enter_actions, messages, repl)

            if replay_code is not None:
                raw_response = replay_code
                input_toks, output_toks, cost, duration = 0, len(raw_response) // 4, 0.0, 0.01
            else:
                def _stream_callback(chunk: str, total_chunks: int):
                    dispatcher.dispatch_llm_stream_chunk(invoke_id, chunk, total_chunks)

                resp = llm.query(messages, on_chunk=_stream_callback)
                raw_response = resp.content
                input_toks, output_toks, cost, duration = (
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.cost,
                    resp.duration,
                )

            q_exit_actions = dispatcher.dispatch_llm_query_exit(
                LLMQueryExit(
                    invoke_id=invoke_id,
                    response=raw_response,
                    input_tokens=input_toks,
                    output_tokens=output_toks,
                    cost=cost,
                    duration=duration,
                )
            )
            _apply_hook_actions(q_exit_actions, messages, repl)

            # Code extraction
            code = protocol.extract_code(raw_response)

            # Update hook namespace context (for linter and tools validation)
            for h in active_hooks:
                if hasattr(h, "set_namespace_context"):
                    try:
                        h.set_namespace_context(repl.get_namespace())
                    except Exception:
                        pass

            # REPL Execution Phase
            r_enter_event = REPLExecEnter(invoke_id=invoke_id, code=code, turn=turn)
            r_enter_actions = dispatcher.dispatch_repl_exec_enter(r_enter_event)
            _apply_hook_actions(r_enter_actions, messages, repl)

            code = r_enter_event.code
            if r_enter_event.skip_execution:
                repl_res = REPLResult(
                    stdout="",
                    stderr="",
                    error=r_enter_event.lint_error_message or "Execution blocked by pre-execution validator",
                    execution_time=0.0,
                )
            elif not code.strip():
                repl_res = REPLResult(
                    stdout="",
                    stderr="",
                    error="No executable Python code was detected in your response. Please write Python code directly to advance the task or use return <result> to finish.",
                    execution_time=0.0,
                )
            else:
                repl_res = repl.execute(code)

            exec_exit_event = REPLExecExit(
                invoke_id=invoke_id,
                turn=turn,
                stdout=repl_res.stdout,
                stderr=repl_res.stderr,
                return_value=repl_res.return_value,
                has_returned=repl_res.has_returned,
                error=repl_res.error,
                exception=repl_res.exception,
                duration=repl_res.execution_time,
            )
            r_exit_actions = dispatcher.dispatch_repl_exec_exit(exec_exit_event)
            _apply_hook_actions(r_exit_actions, messages, repl)

            # Formulate repl_output according to JAZ Paper L1184 (printed output + error traceback)
            repl_output_combined = repl_res.stdout
            if repl_res.error:
                repl_output_combined = (
                    f"{repl_res.stdout}\n{repl_res.error}".strip()
                    if repl_res.stdout.strip()
                    else repl_res.error
                )

            # Record step in REPL __history__
            history.append_step(
                turn=turn,
                code=code,
                repl_output=repl_output_combined,
                error=repl_res.error,
                return_value=exec_exit_event.return_value,
                has_returned=exec_exit_event.has_returned,
                execution_time=repl_res.execution_time,
                llm_response=raw_response,
                repl_exception=repl_res.exception,
            )

            # Check if return occurred and wasn't invalidated by hooks
            if exec_exit_event.has_returned:
                final_result = exec_exit_event.return_value
                break

            # Format observation for next turn
            observation = protocol.format_repl_output(repl_res.stdout, repl_res.error)
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": f"[REPL output from step {turn}]:\n{observation}\n\nContinue writing Python code:",
            })

    except Abort as abort_act:
        execution_error = AbortError(abort_act.reason)
        raise execution_error
    except Exception as exc:
        execution_error = exc
        raise
    finally:
        repl.close()
        _call_depth.reset(depth_token)
        _parent_invoke_id.reset(parent_token)
        _scoped_hooks.reset(hooks_token)
        _scoped_config.reset(config_token)
        dispatcher.dispatch_invoke_exit(
            InvokeExit(
                invoke_id=invoke_id,
                result=final_result,
                error=execution_error,
                duration=time.time() - start_time,
            )
        )

    return final_result


def _apply_hook_actions(
    actions: List[HookAction],
    messages: List[Dict[str, str]],
    repl: BaseREPL,
) -> None:
    """Apply actions returned by hooks (AddMessages, DropMessages, DropVariables, Abort)."""
    for action in actions:
        if isinstance(action, AddMessages):
            messages.extend(action.messages)
        elif isinstance(action, DropMessages):
            for idx in sorted(action.indices, reverse=True):
                if 0 <= idx < len(messages):
                    messages.pop(idx)
        elif isinstance(action, DropVariables):
            for var_name in action.var_names:
                repl.delete_variable(var_name)
        elif isinstance(action, Abort):
            raise action
