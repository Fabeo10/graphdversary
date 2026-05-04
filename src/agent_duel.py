"""
Rule-based and optional local-LLM agent duel helpers.
"""

import json
import random
import re
import urllib.error
import urllib.request

from src.pipeline import run_scenario


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
    prompt = {
        "role": role,
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
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    parsed = _extract_json(data.get("response", ""))
    if not parsed:
        return None

    try:
        choice = int(parsed.get("option")) - 1
    except (TypeError, ValueError):
        return None

    if 0 <= choice < len(options):
        return choice, parsed.get("rationale", "").strip()
    return None


def choose_action(role, options, result, mode="Agent-selected", ollama_model="qwen2.5:3b", seed="default", step=0):
    if not options:
        return None, None

    if mode == "Hybrid Ollama":
        llm_choice = _ollama_choice(role, options, result, ollama_model)
        if llm_choice:
            selected_index, rationale = llm_choice
            action = options[selected_index]
            return selected_index, rationale or _fallback_rationale(role, action)

    rng = random.Random(f"{seed}:{role}:{step}:{len(options)}")
    selected_index = rng.randrange(len(options))
    action = options[selected_index]
    return selected_index, _fallback_rationale(role, action)


def explain_action(role, action, result, mode="Agent-selected", ollama_model="qwen2.5:3b"):
    if mode == "Hybrid Ollama":
        rationale = _ollama_rationale(role, action, result, ollama_model)
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
    ollama_model="qwen2.5:3b",
    seed="default",
):
    """Run cumulative one-agent-at-a-time duel steps with agent-selected actions."""
    max_steps = len(attacks) + len(defenses)
    steps = max(0, min(steps, max_steps))
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

    current_agent = "baseline"
    for step in range(1, steps + 1):
        red_remaining = [index for index in range(len(attacks)) if index not in selected_attack_indices]
        blue_remaining = [index for index in range(len(defenses)) if index not in selected_defense_indices]
        preferred_agent = "red" if step % 2 == 1 else "blue"
        if preferred_agent == "red" and not red_remaining:
            preferred_agent = "blue"
        elif preferred_agent == "blue" and not blue_remaining:
            preferred_agent = "red"

        if preferred_agent == "red" and red_remaining:
            options = [attacks[index] for index in red_remaining]
            selected_offset, rationale = choose_action(
                "red",
                options,
                final_result,
                mode,
                ollama_model,
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
            log.append({
                "step": step,
                "turn": (step + 1) // 2,
                "agent": "red",
                "action": action.get("label", action.get("type", "attack")),
                "rationale": rationale,
                "recall": final_result["adversarial"]["metrics"]["recall"],
                "poison_exposure": final_result["poison_exposure"]["score"],
                "available_options": ", ".join(option.get("label", option.get("type", "attack")) for option in options),
            })
            current_agent = "red"
        elif blue_remaining:
            options = [defenses[index] for index in blue_remaining]
            selected_offset, rationale = choose_action(
                "blue",
                options,
                final_result,
                mode,
                ollama_model,
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
            log.append({
                "step": step,
                "turn": step // 2,
                "agent": "blue",
                "action": action.get("label", action.get("type", "defense")),
                "rationale": rationale,
                "recall": final_result["defended"]["metrics"]["recall"],
                "poison_exposure": final_result["defended_poison_exposure"]["score"],
                "available_options": ", ".join(option.get("label", option.get("type", "defense")) for option in options),
            })
            current_agent = "blue"
        else:
            break

    return {
        "steps": steps,
        "max_steps": max_steps,
        "turns": (steps + 1) // 2,
        "max_turns": max(len(attacks), len(defenses)),
        "current_agent": current_agent,
        "selected_attack_indices": selected_attack_indices,
        "selected_defense_indices": selected_defense_indices,
        "log": log,
        "result": final_result,
    }


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
    ollama_model="qwen2.5:3b",
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
        ollama_model=ollama_model,
    )
