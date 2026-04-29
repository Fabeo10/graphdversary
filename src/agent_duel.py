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
    max_turns = max(len(attacks), len(defenses))
    turns = max(0, min(turns, max_turns))
    log = []
    final_result = run_scenario(
        scenario_path=scenario_path,
        query=query,
        top_k=top_k,
        hop_depth=hop_depth,
        attacks=[],
        defenses=[],
        mock_answer=mock_answer,
        verbose=False,
    )

    for turn in range(1, turns + 1):
        active_attacks = _enabled_prefix(attacks, turn)
        active_defenses = _enabled_prefix(defenses, turn)
        final_result = run_scenario(
            scenario_path=scenario_path,
            query=query,
            top_k=top_k,
            hop_depth=hop_depth,
            attacks=active_attacks,
            defenses=active_defenses,
            mock_answer=mock_answer,
            verbose=False,
        )

        red_action = attacks[turn - 1] if turn <= len(attacks) else None
        blue_action = defenses[turn - 1] if turn <= len(defenses) else None

        if red_action:
            log.append({
                "turn": turn,
                "agent": "red",
                "action": red_action.get("label", red_action.get("type", "attack")),
                "rationale": explain_action("red", red_action, final_result, mode, ollama_model),
                "recall": final_result["adversarial"]["metrics"]["recall"],
                "poison_exposure": final_result["poison_exposure"]["score"],
            })

        if blue_action:
            log.append({
                "turn": turn,
                "agent": "blue",
                "action": blue_action.get("label", blue_action.get("type", "defense")),
                "rationale": explain_action("blue", blue_action, final_result, mode, ollama_model),
                "recall": final_result["defended"]["metrics"]["recall"],
                "poison_exposure": final_result["defended_poison_exposure"]["score"],
            })

    return {
        "turns": turns,
        "max_turns": max_turns,
        "log": log,
        "result": final_result,
    }
