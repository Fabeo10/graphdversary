"""
Rule-based and optional local-LLM agent duel helpers.

Hybrid Ollama uses separate HTTP POST /api/generate calls per turn—no shared chat
session, context window, or coordinator passing one model the other's hidden chain-of-thought.
Each side only sees its role, allowed options, and published retrieval metrics (environment).
"""

import hashlib
import json
import random
import re
import urllib.error
import urllib.request
from pathlib import Path

from src.pipeline import run_scenario

# Default family split for adversarial realism: attacker vs defender use different weights.
DEFAULT_OLLAMA_RED = "llama3.2:3b"
DEFAULT_OLLAMA_BLUE = "qwen2.5:3b"

_ISOLATION_NOTE = (
    "Stateless single-turn: there is no shared chat history with the opposing agent's model; "
    "do not assume access to the other side's private rationale."
)


def _enabled_prefix(items, count):
    configured = []
    for item in items[:count]:
        copy = dict(item)
        copy["enabled"] = True
        configured.append(copy)
    return configured


def _fallback_rationale(role, action):
    label = action.get("label", action.get("type", "action"))
    if role == "red":
        return f"Selected '{label}' from the allowed red-team options to vary the attack path."
    return f"Selected '{label}' from the allowed blue-team controls to respond to the current state."


def _ollama_rationale(role, action, result, model, timeout=8):
    prompt = {
        "role": role,
        "simulation_policy": _ISOLATION_NOTE,
        "action": action,
        "metrics": {
            "attack_recall": result["adversarial"]["metrics"]["recall"],
            "defended_recall": result["defended"]["metrics"]["recall"],
            "attack_poison_exposure": result["poison_exposure"]["score"],
            "defended_poison_exposure": result["defended_poison_exposure"]["score"],
        },
        "instruction": "Explain in one concise sentence why this action was selected. Do not propose new actions.",
    }
    payload = json.dumps({
        "model": model,
        "prompt": json.dumps(prompt),
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    rationale = data.get("response", "").strip()
    return rationale or None


def _extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _ollama_choice(role, options, result, model, timeout=10):
    """Call Ollama for JSON option pick; return trace dict (always) with ok flag and offsets when parsed."""
    prompt = {
        "role": role,
        "simulation_policy": _ISOLATION_NOTE,
        "allowed_options": [
            {
                "option": index + 1,
                "type": option.get("type"),
                "label": option.get("label", option.get("type", "action")),
            }
            for index, option in enumerate(options)
        ],
        "metrics": {
            "baseline_recall": result["baseline"]["metrics"]["recall"],
            "attack_recall": result["adversarial"]["metrics"]["recall"],
            "defended_recall": result["defended"]["metrics"]["recall"],
            "attack_poison_exposure": result["poison_exposure"]["score"],
            "defended_poison_exposure": result["defended_poison_exposure"]["score"],
        },
        "instruction": (
            "Choose exactly one allowed option. Return only JSON with keys "
            "'option' and 'rationale'. The option value must be the option number."
        ),
    }
    trace = {
        "endpoint": "http://localhost:11434/api/generate",
        "model": model,
        "temperature": 0.7,
        "prompt": prompt,
        "prompt_payload_preview": json.dumps(prompt),
        "raw_response": None,
        "http_ok": False,
        "parse_ok": False,
        "selected_offset": None,
        "rationale_from_model": "",
        "error": None,
    }
    payload = json.dumps({
        "model": model,
        "prompt": json.dumps(prompt),
        "stream": False,
        "options": {"temperature": 0.7},
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        trace["error"] = str(exc)
        return trace

    trace["http_ok"] = True
    raw_text = data.get("response", "") if isinstance(data, dict) else ""
    trace["raw_response"] = raw_text

    parsed = _extract_json(raw_text)
    if not parsed:
        trace["error"] = "response_not_json_object"
        return trace

    try:
        choice = int(parsed.get("option")) - 1
    except (TypeError, ValueError):
        trace["error"] = "invalid_option_field"
        return trace

    trace["rationale_from_model"] = (parsed.get("rationale") or "").strip()
    if 0 <= choice < len(options):
        trace["parse_ok"] = True
        trace["selected_offset"] = choice
        return trace

    trace["error"] = "option_out_of_range"
    return trace


def _model_for_role(role, ollama_model_red, ollama_model_blue):
    return ollama_model_red if role == "red" else ollama_model_blue


def choose_action(
    role,
    options,
    result,
    mode="Agent-selected",
    ollama_model_red=DEFAULT_OLLAMA_RED,
    ollama_model_blue=DEFAULT_OLLAMA_BLUE,
    seed="default",
    step=0,
):
    if not options:
        return None, None, None

    if mode == "Hybrid Ollama":
        model = _model_for_role(role, ollama_model_red, ollama_model_blue)
        trace = _ollama_choice(role, options, result, model)
        summary = {
            "role": role,
            "model": trace.get("model"),
            "endpoint": trace.get("endpoint"),
            "http_ok": trace.get("http_ok"),
            "parse_ok": trace.get("parse_ok"),
            "ollama_raw_response": trace.get("raw_response"),
            "ollama_prompt_json": trace.get("prompt_payload_preview"),
            "ollama_error": trace.get("error"),
        }
        if trace.get("parse_ok"):
            selected_index = trace["selected_offset"]
            action = options[selected_index]
            rationale = trace.get("rationale_from_model") or _fallback_rationale(role, action)
            summary["selection"] = "ollama_json"
            return selected_index, rationale, summary

        summary["selection"] = "fallback_rng_after_llm"
        rng = random.Random(f"{seed}:{role}:{step}:{len(options)}")
        selected_index = rng.randrange(len(options))
        action = options[selected_index]
        return selected_index, _fallback_rationale(role, action), summary

    rng = random.Random(f"{seed}:{role}:{step}:{len(options)}")
    selected_index = rng.randrange(len(options))
    action = options[selected_index]
    return selected_index, _fallback_rationale(role, action), None


def explain_action(
    role,
    action,
    result,
    mode="Agent-selected",
    ollama_model_red=DEFAULT_OLLAMA_RED,
    ollama_model_blue=DEFAULT_OLLAMA_BLUE,
):
    if mode == "Hybrid Ollama":
        model = _model_for_role(role, ollama_model_red, ollama_model_blue)
        rationale = _ollama_rationale(role, action, result, model)
        if rationale:
            return rationale
    return _fallback_rationale(role, action)


def _enabled_indices(items, indices):
    configured = []
    for index in indices:
        copy = dict(items[index])
        copy["enabled"] = True
        configured.append(copy)
    return configured


def _run_duel_state(scenario_path, query, top_k, hop_depth, attacks, defenses, mock_answer, attack_indices, defense_indices):
    return run_scenario(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=_enabled_indices(attacks, attack_indices),
        defenses=_enabled_indices(defenses, defense_indices),
        mock_answer=mock_answer,
        verbose=False,
    )


def _duel_step_snapshot(step, agent, display_kind, result, action_label=""):
    """
    display_kind: baseline | attacked | defended — which graph/metrics slice to show for this moment.
    """
    return {
        "step": step,
        "agent": agent,
        "display_kind": display_kind,
        "action_label": action_label,
        "result": result,
    }


def _action_pool_fingerprint(attacks, defenses):
    """Stable hash so checkpoint signatures match across Streamlit reruns (dict order, etc.)."""
    parts_a = sorted(json.dumps(a, sort_keys=True, default=str) for a in attacks)
    parts_d = sorted(json.dumps(d, sort_keys=True, default=str) for d in defenses)
    blob = ("A:" + "\n".join(parts_a) + "\nD:" + "\n".join(parts_d)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def duel_input_signature(
    scenario_path,
    query,
    top_k,
    hop_depth,
    attacks,
    defenses,
    mock_answer,
    mode,
    ollama_model_red,
    ollama_model_blue,
    seed,
):
    path_obj = Path(scenario_path)
    try:
        scenario_norm = str(path_obj.resolve())
    except OSError:
        scenario_norm = str(path_obj)
    return json.dumps(
        {
            "scenario_path": scenario_norm,
            "query": (query or "").strip(),
            "top_k": int(top_k),
            "hop_depth": int(hop_depth),
            "pool_fp": _action_pool_fingerprint(attacks, defenses),
            "mock_answer": (mock_answer or "").strip(),
            "mode": mode,
            "ollama_model_red": str(ollama_model_red),
            "ollama_model_blue": str(ollama_model_blue),
            "seed": str(seed),
        },
        sort_keys=True,
    )


def _attach_llm_columns(log_row, llm_trace):
    if not llm_trace:
        return log_row
    log_row["llm_model"] = llm_trace.get("model")
    log_row["llm_endpoint"] = llm_trace.get("endpoint")
    log_row["llm_selection"] = llm_trace.get("selection")
    log_row["llm_http_ok"] = llm_trace.get("http_ok")
    log_row["llm_parse_ok"] = llm_trace.get("parse_ok")
    log_row["llm_error"] = llm_trace.get("ollama_error")
    log_row["llm_prompt_json"] = llm_trace.get("ollama_prompt_json")
    log_row["llm_raw_response"] = llm_trace.get("ollama_raw_response")
    return log_row


def _execute_one_duel_iteration(
    step,
    scenario_path,
    query,
    top_k,
    hop_depth,
    attacks,
    defenses,
    mock_answer,
    selected_attack_indices,
    selected_defense_indices,
    final_result,
    log,
    snapshots,
    mode,
    ollama_model_red,
    ollama_model_blue,
    seed,
):
    """
    Perform a single agent step. Mutates lists and returns
    (selected_attack_indices, selected_defense_indices, final_result, current_agent, early_stop).
    """
    red_remaining = [index for index in range(len(attacks)) if index not in selected_attack_indices]
    blue_remaining = [index for index in range(len(defenses)) if index not in selected_defense_indices]
    preferred_agent = "red" if step % 2 == 1 else "blue"
    if preferred_agent == "red" and not red_remaining:
        preferred_agent = "blue"
    elif preferred_agent == "blue" and not blue_remaining:
        preferred_agent = "red"

    current_agent = "baseline"

    if preferred_agent == "red" and red_remaining:
        options = [attacks[index] for index in red_remaining]
        selected_offset, rationale, llm_trace = choose_action(
            "red",
            options,
            final_result,
            mode,
            ollama_model_red,
            ollama_model_blue,
            seed,
            step,
        )
        selected_index = red_remaining[selected_offset]
        selected_attack_indices.append(selected_index)
        final_result = _run_duel_state(
            scenario_path,
            query,
            top_k,
            hop_depth,
            attacks,
            defenses,
            mock_answer,
            selected_attack_indices,
            selected_defense_indices,
        )
        action = attacks[selected_index]
        label = action.get("label", action.get("type", "attack"))
        row = {
            "step": step,
            "turn": (step + 1) // 2,
            "agent": "red",
            "action": label,
            "rationale": rationale,
            "recall": final_result["adversarial"]["metrics"]["recall"],
            "poison_exposure": final_result["poison_exposure"]["score"],
            "available_options": ", ".join(option.get("label", option.get("type", "attack")) for option in options),
        }
        log.append(_attach_llm_columns(row, llm_trace))
        snapshots.append(_duel_step_snapshot(step, "red", "attacked", final_result, label))
        current_agent = "red"
        return selected_attack_indices, selected_defense_indices, final_result, current_agent, False

    if blue_remaining:
        options = [defenses[index] for index in blue_remaining]
        selected_offset, rationale, llm_trace = choose_action(
            "blue",
            options,
            final_result,
            mode,
            ollama_model_red,
            ollama_model_blue,
            seed,
            step,
        )
        selected_index = blue_remaining[selected_offset]
        selected_defense_indices.append(selected_index)
        final_result = _run_duel_state(
            scenario_path,
            query,
            top_k,
            hop_depth,
            attacks,
            defenses,
            mock_answer,
            selected_attack_indices,
            selected_defense_indices,
        )
        action = defenses[selected_index]
        label = action.get("label", action.get("type", "defense"))
        row = {
            "step": step,
            "turn": step // 2,
            "agent": "blue",
            "action": label,
            "rationale": rationale,
            "recall": final_result["defended"]["metrics"]["recall"],
            "poison_exposure": final_result["defended_poison_exposure"]["score"],
            "available_options": ", ".join(option.get("label", option.get("type", "defense")) for option in options),
        }
        log.append(_attach_llm_columns(row, llm_trace))
        snapshots.append(_duel_step_snapshot(step, "blue", "defended", final_result, label))
        current_agent = "blue"
        return selected_attack_indices, selected_defense_indices, final_result, current_agent, False

    return selected_attack_indices, selected_defense_indices, final_result, "baseline", True


def _package_duel_return(
    steps,
    max_steps,
    attacks,
    defenses,
    selected_attack_indices,
    selected_defense_indices,
    current_agent,
    log,
    final_result,
    snapshots,
    sig,
):
    checkpoint = {
        "signature": sig,
        "completed_steps": steps,
        "selected_attack_indices": list(selected_attack_indices),
        "selected_defense_indices": list(selected_defense_indices),
        "final_result": final_result,
        "log": log,
        "snapshots": snapshots,
        "current_agent": current_agent,
    }
    out = {
        "steps": steps,
        "max_steps": max_steps,
        "turns": (steps + 1) // 2,
        "max_turns": max(len(attacks), len(defenses)),
        "current_agent": current_agent,
        "selected_attack_indices": selected_attack_indices,
        "selected_defense_indices": selected_defense_indices,
        "log": log,
        "result": final_result,
        "snapshots": snapshots,
        "checkpoint": checkpoint,
    }
    return out


def run_agent_duel_steps(
    scenario_path,
    query,
    top_k,
    hop_depth,
    attacks,
    defenses,
    mock_answer,
    steps,
    mode="Agent-selected",
    ollama_model_red=DEFAULT_OLLAMA_RED,
    ollama_model_blue=DEFAULT_OLLAMA_BLUE,
    seed="default",
    resume_checkpoint=None,
):
    """Run cumulative one-agent-at-a-time duel steps with agent-selected actions."""
    max_steps = len(attacks) + len(defenses)
    steps = max(0, min(steps, max_steps))
    sig = duel_input_signature(
        scenario_path,
        query,
        top_k,
        hop_depth,
        attacks,
        defenses,
        mock_answer,
        mode,
        ollama_model_red,
        ollama_model_blue,
        seed,
    )

    # Incremental path: add exactly one agent step without replaying prior LLM calls.
    if (
        resume_checkpoint
        and resume_checkpoint.get("signature") == sig
        and resume_checkpoint["completed_steps"] == steps - 1
        and steps >= 1
    ):
        selected_attack_indices = list(resume_checkpoint["selected_attack_indices"])
        selected_defense_indices = list(resume_checkpoint["selected_defense_indices"])
        log = list(resume_checkpoint["log"])
        snapshots = list(resume_checkpoint["snapshots"])
        final_result = resume_checkpoint["final_result"]
        step_to_run = steps
        selected_attack_indices, selected_defense_indices, final_result, current_agent, early_stop = _execute_one_duel_iteration(
            step_to_run,
            scenario_path,
            query,
            top_k,
            hop_depth,
            attacks,
            defenses,
            mock_answer,
            selected_attack_indices,
            selected_defense_indices,
            final_result,
            log,
            snapshots,
            mode,
            ollama_model_red,
            ollama_model_blue,
            seed,
        )
        _ = early_stop
        return _package_duel_return(
            steps,
            max_steps,
            attacks,
            defenses,
            selected_attack_indices,
            selected_defense_indices,
            current_agent,
            log,
            final_result,
            snapshots,
            sig,
        )

    selected_attack_indices = []
    selected_defense_indices = []
    log = []
    final_result = _run_duel_state(
        scenario_path,
        query,
        top_k,
        hop_depth,
        attacks,
        defenses,
        mock_answer,
        selected_attack_indices,
        selected_defense_indices,
    )
    log.append({
        "step": 0,
        "turn": 0,
        "agent": "system",
        "action": "Capture baseline state",
        "rationale": "Both agents observe the baseline graph, query, retrieval metrics, and available action options before acting.",
        "recall": final_result["baseline"]["metrics"]["recall"],
        "poison_exposure": 0.0,
        "available_options": f"{len(attacks)} attacks / {len(defenses)} defenses",
    })

    snapshots = [
        _duel_step_snapshot(0, "system", "baseline", final_result, "Baseline (no strikes yet)"),
    ]

    current_agent = "baseline"
    for step in range(1, steps + 1):
        selected_attack_indices, selected_defense_indices, final_result, current_agent, early_stop = _execute_one_duel_iteration(
            step,
            scenario_path,
            query,
            top_k,
            hop_depth,
            attacks,
            defenses,
            mock_answer,
            selected_attack_indices,
            selected_defense_indices,
            final_result,
            log,
            snapshots,
            mode,
            ollama_model_red,
            ollama_model_blue,
            seed,
        )
        if early_stop:
            break

    return _package_duel_return(
        steps,
        max_steps,
        attacks,
        defenses,
        selected_attack_indices,
        selected_defense_indices,
        current_agent,
        log,
        final_result,
        snapshots,
        sig,
    )


def run_agent_duel(
    scenario_path,
    query,
    top_k,
    hop_depth,
    attacks,
    defenses,
    mock_answer,
    turns,
    mode="Agent-selected",
    ollama_model_red=DEFAULT_OLLAMA_RED,
    ollama_model_blue=DEFAULT_OLLAMA_BLUE,
):
    """Run cumulative red/blue turns and return the final result plus a log."""
    return run_agent_duel_steps(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=attacks,
        defenses=defenses,
        mock_answer=mock_answer,
        steps=turns * 2,
        mode=mode,
        ollama_model_red=ollama_model_red,
        ollama_model_blue=ollama_model_blue,
    )
