"""
Rule-based and optional local-LLM agent duel helpers.
"""

import json
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
        return f"Selected '{label}' because it is the next available pressure point in the scenario."
    return f"Selected '{label}' because it is the paired control for the latest red-team action."


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


def explain_action(role, action, result, mode="Rule-based", ollama_model="qwen2.5:3b"):
    if mode == "Hybrid Ollama":
        rationale = _ollama_rationale(role, action, result, ollama_model)
        if rationale:
            return rationale
    return _fallback_rationale(role, action)


def duel_events(attacks, defenses):
    events = []
    max_turns = max(len(attacks), len(defenses))
    for index in range(max_turns):
        if index < len(attacks):
            events.append({"turn": index + 1, "agent": "red", "index": index})
        if index < len(defenses):
            events.append({"turn": index + 1, "agent": "blue", "index": index})
    return events


def _counts_for_events(events):
    red_count = sum(1 for event in events if event["agent"] == "red")
    blue_count = sum(1 for event in events if event["agent"] == "blue")
    return red_count, blue_count


def _run_duel_state(scenario_path, query, top_k, hop_depth, attacks, defenses, mock_answer, events):
    red_count, blue_count = _counts_for_events(events)
    return run_scenario(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=_enabled_prefix(attacks, red_count),
        defenses=_enabled_prefix(defenses, blue_count),
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
    mode="Rule-based",
    ollama_model="qwen2.5:3b",
):
    """Run cumulative one-agent-at-a-time duel steps and return current state."""
    events = duel_events(attacks, defenses)
    steps = max(0, min(steps, len(events)))
    active_events = events[:steps]
    final_result = _run_duel_state(
        scenario_path,
        query,
        top_k,
        hop_depth,
        attacks,
        defenses,
        mock_answer,
        active_events,
    )

    log = []
    for index, event in enumerate(active_events):
        step_events = active_events[:index + 1]
        step_result = _run_duel_state(
            scenario_path,
            query,
            top_k,
            hop_depth,
            attacks,
            defenses,
            mock_answer,
            step_events,
        )
        if event["agent"] == "red":
            action = attacks[event["index"]]
            log.append({
                "step": index + 1,
                "turn": event["turn"],
                "agent": "red",
                "action": action.get("label", action.get("type", "attack")),
                "rationale": explain_action("red", action, step_result, mode, ollama_model),
                "recall": step_result["adversarial"]["metrics"]["recall"],
                "poison_exposure": step_result["poison_exposure"]["score"],
            })
        else:
            action = defenses[event["index"]]
            log.append({
                "step": index + 1,
                "turn": event["turn"],
                "agent": "blue",
                "action": action.get("label", action.get("type", "defense")),
                "rationale": explain_action("blue", action, step_result, mode, ollama_model),
                "recall": step_result["defended"]["metrics"]["recall"],
                "poison_exposure": step_result["defended_poison_exposure"]["score"],
            })

    current_event = active_events[-1] if active_events else None
    return {
        "steps": steps,
        "max_steps": len(events),
        "turns": (steps + 1) // 2,
        "max_turns": max(len(attacks), len(defenses)),
        "current_agent": current_event["agent"] if current_event else "baseline",
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
    mode="Rule-based",
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
