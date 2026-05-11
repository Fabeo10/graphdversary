"""
Pipeline Module
---------------
Coordinates ingestion, retrieval, adversarial mutation, and evaluation.
"""

import argparse
import json
from pathlib import Path

from src.adversarial_module import AdversarialModule
from src.build_graph import KnowledgeBase
from src.evaluator import Evaluator
from src.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_PATH = PROJECT_ROOT / "data" / "scenarios" / "auth_token_poison.json"


def print_section(title):
    print(f"\n{'-'*50}\n{title}\n{'-'*50}")


def load_scenario(scenario_path=DEFAULT_SCENARIO_PATH):
    """Load a demo scenario and resolve paths relative to the scenario file."""
    path = Path(scenario_path)
    with path.open("r") as f:
        scenario = json.load(f)

    corpus_path = Path(scenario.get("corpus_path", "../mock_corpus.json"))
    if not corpus_path.is_absolute():
        corpus_path = (path.parent / corpus_path).resolve()

    scenario["scenario_path"] = str(path.resolve())
    scenario["corpus_path"] = str(corpus_path)
    return scenario


def graph_snapshot(kb):
    """Return graph state in a UI-friendly shape."""
    return {
        "nodes": [
            {
                "id": node_id,
                "content": attrs.get("content", ""),
                "type": attrs.get("type", "generic"),
            }
            for node_id, attrs in kb.graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "relation": attrs.get("relation", ""),
            }
            for source, target, attrs in kb.graph.edges(data=True)
        ],
    }


def unrestored_remove_edge_attacks(result):
    """Return ``{(source, target)}`` for ``remove_edge`` attacks that the blue
    team has NOT (yet) restored — i.e. they still don't appear in
    ``result["defended_graph"]``.

    Used by the duel visualization to highlight which red-team edge removals are
    still in effect, so the audience can see at a glance how much damage has
    survived blue's defenses.

    Returns an empty set when ``result`` has no attacks or no defended graph.
    """
    if not result:
        return set()
    defended_graph = result.get("defended_graph") or {}
    defended_edges = {
        (edge["source"], edge["target"])
        for edge in defended_graph.get("edges", [])
    }
    attempted_removals = {
        (attack["source"], attack["target"])
        for attack in (result.get("attacks") or [])
        if attack.get("type") == "remove_edge"
        and attack.get("success")
        and "source" in attack
        and "target" in attack
    }
    return attempted_removals - defended_edges


def evaluate_retrieval(retrieved_nodes, ground_truth_nodes):
    return {
        "precision": Evaluator.calculate_precision(retrieved_nodes, ground_truth_nodes),
        "recall": Evaluator.calculate_recall(retrieved_nodes, ground_truth_nodes),
    }


def run_retrieval(retriever, query, top_k, hop_depth, ground_truth_nodes):
    context, nodes, trace = retriever.retrieve(
        query,
        top_k=top_k,
        hop_depth=hop_depth,
        include_trace=True,
    )
    metrics = evaluate_retrieval(nodes, ground_truth_nodes)

    return {
        "query": query,
        "context": context,
        "nodes": nodes,
        "trace": trace,
        "metrics": metrics,
    }


def apply_attacks(kb, query, attacks):
    """Apply enabled attacks and return the mutated query plus an audit log."""
    adv = AdversarialModule()
    attack_log = []
    mutated_query = query

    for attack in attacks:
        if not attack.get("enabled", True):
            continue

        attack_type = attack["type"]
        if attack_type == "remove_edge":
            removed = adv.topological_edge_removal(kb, attack["source"], attack["target"])
            attack_log.append({
                "type": attack_type,
                "label": attack.get("label", "Remove edge"),
                "source": attack["source"],
                "target": attack["target"],
                "success": removed,
            })
        elif attack_type == "inject_poison":
            poison_id = adv.inject_poison_node(
                kb,
                attack["target"],
                attack["content"],
                poison_id=attack.get("poison_id"),
            )
            attack_log.append({
                "type": attack_type,
                "label": attack.get("label", "Inject poison"),
                "target": attack["target"],
                "poison_id": poison_id,
                "content": attack["content"],
                "success": True,
            })
        elif attack_type == "perturb_query":
            mutated_query = adv.perturb_query(
                mutated_query,
                attack.get("perturbation_type", "contradiction"),
            )
            attack_log.append({
                "type": attack_type,
                "label": attack.get("label", "Perturb query"),
                "perturbation_type": attack.get("perturbation_type", "contradiction"),
                "success": True,
            })
        elif attack_type == "benign_query":
            # Non-malicious "attack": red sends a normal user query. No graph
            # mutation, no query mutation — exercises blue's ability to recognize
            # there is nothing to defend against.
            attack_log.append({
                "type": attack_type,
                "label": attack.get("label", "Send normal user query"),
                "success": True,
            })
        else:
            attack_log.append({
                "type": attack_type,
                "label": attack.get("label", attack_type),
                "success": False,
                "error": f"Unsupported attack type: {attack_type}",
            })

    return mutated_query, attack_log


def sanitize_query(query, blocked_phrases):
    """Remove known adversarial instructions from the test query."""
    sanitized = query
    removed = []
    for phrase in blocked_phrases:
        if phrase.lower() in sanitized.lower():
            start = sanitized.lower().find(phrase.lower())
            sanitized = sanitized[:start] + sanitized[start + len(phrase):]
            removed.append(phrase)

    return " ".join(sanitized.split()), removed


def apply_defenses(kb, query, defenses, scenario):
    """Apply blue-team controls before adversarial retrieval."""
    defense_log = []
    defended_query = query
    index_dirty = False

    for defense in defenses:
        if not defense.get("enabled", True):
            continue

        defense_type = defense["type"]
        if defense_type == "remove_untrusted_nodes":
            node_types = set(defense.get("node_types", ["adversarial"]))
            removed_nodes = [
                node_id
                for node_id, attrs in kb.graph.nodes(data=True)
                if attrs.get("type") in node_types
            ]
            kb.graph.remove_nodes_from(removed_nodes)
            index_dirty = index_dirty or bool(removed_nodes)
            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", "Remove untrusted nodes"),
                "success": True,
                "removed_nodes": removed_nodes,
            })
        elif defense_type == "restore_protected_edges":
            restored_edges = []
            corpus_edges = {
                (edge["source"], edge["target"]): edge["relation"]
                for edge in scenario.get("corpus_edges", [])
            }
            for edge in defense.get("edges", []):
                source = edge["source"]
                target = edge["target"]
                relation = edge.get("relation") or corpus_edges.get((source, target), "RESTORED")
                if source in kb.graph and target in kb.graph and not kb.graph.has_edge(source, target):
                    kb.graph.add_edge(source, target, relation=relation)
                    restored_edges.append({"source": source, "target": target, "relation": relation})

            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", "Restore protected edges"),
                "success": True,
                "restored_edges": restored_edges,
            })
        elif defense_type == "sanitize_query":
            defended_query, removed_phrases = sanitize_query(
                defended_query,
                defense.get("blocked_phrases", []),
            )
            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", "Sanitize query"),
                "success": True,
                "removed_phrases": removed_phrases,
            })
        elif defense_type == "block_forbidden_claim_nodes":
            forbidden_claims = [claim.lower() for claim in scenario.get("forbidden_claims", [])]
            removed_nodes = []
            for node_id, attrs in list(kb.graph.nodes(data=True)):
                content = attrs.get("content", "").lower()
                if any(claim in content for claim in forbidden_claims):
                    removed_nodes.append(node_id)

            kb.graph.remove_nodes_from(removed_nodes)
            index_dirty = index_dirty or bool(removed_nodes)
            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", "Block forbidden claim nodes"),
                "success": True,
                "removed_nodes": removed_nodes,
            })
        elif defense_type == "no_op":
            # Monitor-only stance: blue observes without intervening. The correct
            # response to a benign red action; a costly false-negative against a
            # real attack.
            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", "Monitor (no action)"),
                "success": True,
            })
        else:
            defense_log.append({
                "type": defense_type,
                "label": defense.get("label", defense_type),
                "success": False,
                "error": f"Unsupported defense type: {defense_type}",
            })

    if index_dirty:
        kb.rebuild_index()

    return defended_query, defense_log


def calculate_poison_exposure(context, forbidden_claims, query=""):
    """Estimate whether retrieved context **or the active query** exposes
    forbidden or poisoned claims.

    Query-perturbation attacks never write to the corpus, so a context-only
    check returns 0 for them even when the malicious clause is sitting in the
    prompt. Scanning the (adversarial or defended) query string alongside the
    retrieved context lets the poison metric react to those attacks too, and
    drop back to 0 after the blue-team sanitizer strips the clause.
    """
    haystack = (" ".join(context) + " " + (query or "")).lower()
    matches = [claim for claim in forbidden_claims if claim.lower() in haystack]
    return {
        "matches": matches,
        "score": len(matches) / len(forbidden_claims) if forbidden_claims else 0.0,
    }


def run_scenario(
    scenario_path=DEFAULT_SCENARIO_PATH,
    query=None,
    top_k=None,
    hop_depth=None,
    attacks=None,
    defenses=None,
    mock_answer=None,
    verbose=True,
):
    """Run a full baseline -> attack -> adversarial evaluation cycle."""
    scenario = load_scenario(scenario_path)
    with Path(scenario["corpus_path"]).open("r") as f:
        corpus = json.load(f)
    scenario["corpus_edges"] = corpus.get("edges", [])
    kb = KnowledgeBase(verbose=verbose)
    metadata = kb.build_from_json(scenario["corpus_path"])

    query = query or scenario.get("query") or metadata.get("test_query", "")
    top_k = top_k if top_k is not None else scenario.get("top_k", 1)
    hop_depth = hop_depth if hop_depth is not None else scenario.get("hop_depth", 1)
    ground_truth_nodes = scenario.get("ground_truth_nodes") or metadata.get("ground_truth_nodes", [])
    attacks = attacks if attacks is not None else scenario.get("attacks", [])
    defenses = defenses if defenses is not None else scenario.get("defenses", [])

    retriever = HybridRetriever(kb)
    baseline = run_retrieval(retriever, query, top_k, hop_depth, ground_truth_nodes)
    baseline_graph = graph_snapshot(kb)

    adversarial_query, attack_log = apply_attacks(kb, query, attacks)
    attacked_graph = graph_snapshot(kb)
    adversarial = run_retrieval(retriever, adversarial_query, top_k, hop_depth, ground_truth_nodes)

    defended_query, defense_log = apply_defenses(kb, adversarial_query, defenses, scenario)
    defended_graph = graph_snapshot(kb)
    defended = run_retrieval(retriever, defended_query, top_k, hop_depth, ground_truth_nodes)

    answer = mock_answer or scenario.get("mock_answer", "")
    forbidden_claims = scenario.get("forbidden_claims", [])
    faithfulness = Evaluator.calculate_faithfulness(
        answer, adversarial["context"], forbidden_terms=forbidden_claims
    )
    poison_exposure = calculate_poison_exposure(
        adversarial["context"],
        forbidden_claims,
        query=adversarial_query,
    )
    defended_faithfulness = Evaluator.calculate_faithfulness(
        answer, defended["context"], forbidden_terms=forbidden_claims
    )
    defended_poison_exposure = calculate_poison_exposure(
        defended["context"],
        forbidden_claims,
        query=defended_query,
    )

    return {
        "scenario": scenario,
        "settings": {
            "query": query,
            "top_k": top_k,
            "hop_depth": hop_depth,
        },
        "baseline_graph": baseline_graph,
        "attacked_graph": attacked_graph,
        "defended_graph": defended_graph,
        "baseline": baseline,
        "attacks": attack_log,
        "adversarial": adversarial,
        "defenses": defense_log,
        "defended": defended,
        "answer": answer,
        "faithfulness": faithfulness,
        "poison_exposure": poison_exposure,
        "defended_faithfulness": defended_faithfulness,
        "defended_poison_exposure": defended_poison_exposure,
    }


def run_pipeline(scenario_path=DEFAULT_SCENARIO_PATH):
    result = run_scenario(scenario_path)

    print("="*60)
    print(" ADVERSARIAL GRAPHRAG EVALUATION PIPELINE ".center(60))
    print("="*60)

    print_section("[STAGE 1] BASELINE EXECUTION")
    for ctx in result["baseline"]["context"]:
        print(f"  [OK] {ctx}")
    print(
        "\nMetrics -> "
        f"Precision: {result['baseline']['metrics']['precision']:.2f} | "
        f"Recall: {result['baseline']['metrics']['recall']:.2f}"
    )

    print_section("[STAGE 2] ADVERSARIAL PERTURBATION")
    for attack in result["attacks"]:
        status = "applied" if attack["success"] else "skipped"
        print(f"  [!] {attack['label']} ({status})")

    print_section("[STAGE 3] ADVERSARIAL EXECUTION")
    for ctx in result["adversarial"]["context"]:
        marker = "POISON" if "bypass" in ctx.lower() else "TRACE"
        print(f"  [{marker}] {ctx}")

    print_section("[STAGE 4] BLUE-TEAM DEFENSE")
    for defense in result["defenses"]:
        status = "applied" if defense["success"] else "skipped"
        print(f"  [D] {defense['label']} ({status})")

    for ctx in result["defended"]["context"]:
        marker = "SAFE" if "bypass" not in ctx.lower() else "REVIEW"
        print(f"  [{marker}] {ctx}")

    print_section("[STAGE 5] FINAL EVALUATION SUMMARY")
    print(f"Baseline Recall:       {result['baseline']['metrics']['recall']:.2f}")
    print(f"Attack Recall:         {result['adversarial']['metrics']['recall']:.2f}")
    print(f"Defended Recall:       {result['defended']['metrics']['recall']:.2f}")
    print(f"Attack Poison:         {result['poison_exposure']['score']:.2f}")
    print(f"Defended Poison:       {result['defended_poison_exposure']['score']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run an adversarial GraphRAG scenario.")
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO_PATH),
        help="Path to a scenario JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured result as JSON.",
    )
    args = parser.parse_args()
    if args.json:
        result = run_scenario(args.scenario)
        print(json.dumps(result, indent=2))
    else:
        run_pipeline(args.scenario)


if __name__ == "__main__":
    main()