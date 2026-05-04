# graphdversary — project brief (course / stakeholder summary)

This document is **hand-written** and describes what the system is for, what it does *not* claim, and how to interpret results. For an inventory of the current **toy dataset and scenarios** that updates when you add JSON under `data/`, run `python scripts/generate_project_docs.py` and read [DATASET_AND_SCENARIOS.md](DATASET_AND_SCENARIOS.md).

## Objective

**graphdversary** is a small **adversarial GraphRAG** demonstration: a knowledge graph (from JSON) is embedded and searched with a **hybrid retriever** (dense anchors + graph expansion). The pipeline applies **scenario-defined** red-team mutations (e.g. edge removal, poison nodes, query perturbation) and blue-team defenses, then reports **retrieval metrics** and simple **exposure** checks. An optional **Agent Duel** lets rule-based or local **Ollama** models choose actions from the scenario’s allowed lists—illustrative, not autonomous penetration testing.

## Threat model (what is being simulated)

- **Assets:** A static **documented graph** and retrieval pipeline. There is no live production API, IAM, or network boundary in scope.
- **Adversary capabilities (as modeled):** Only what each scenario enables—typically structured edits to the graph or query text that affect **what gets retrieved**, not arbitrary code execution or OS compromise.
- **Defender capabilities:** Only the defense types implemented in code (e.g. restoring edges, removing untrusted nodes, sanitizing queries, blocking nodes matching forbidden claims).

Anything outside that (real attacker persistence, model theft, supply-chain attacks on PyTorch, etc.) is **out of scope**.

## In scope vs out of scope

| In scope | Out of scope |
|----------|----------------|
| Showing how **retrieval** changes under graph/query tampering | Proving security of a **production** RAG or GraphRAG product |
| Repeatable **scenarios** + **preflight tests** | Formal verification or certification |
| **Heuristic** metrics (precision/recall on node ids, forbidden-claim exposure, faithfulness-style checks) | Ground-truth answers for open-ended LLM safety in the wild |
| Optional **local LLM** choice among **fixed** scenario actions | Unconstrained agent hacking or tool use against real systems |

## Metrics (how to read the numbers)

- **Precision / recall** are computed against **scenario `ground_truth_nodes`** (and corpus metadata where applicable)—they measure overlap between **retrieved node ids** and a **curated** relevant set, not general correctness of natural-language answers.
- **Poison / forbidden-claim exposure** checks whether configured **strings** appear in the **retrieved context**—a coarse signal, not semantic entailment.
- **Faithfulness** in this codebase is a **lightweight** comparison between a **mock answer** and retrieved context, suitable for pedagogy, not production guardrails.

## Limitations (say these in a final presentation)

1. **Toy data:** One primary corpus and a small scenario set; conclusions do not generalize to arbitrary corpora without new scenarios and validation.
2. **Deterministic attacks:** Red-team steps are **authored**, not discovered by an adaptive attacker.
3. **LLM duel:** Models pick from **enumerated** actions; failures often reflect **JSON formatting** or **menu size**, not only “security judgment.”

## Related directions (for a bibliography)

Useful keywords for a short related-work paragraph: **retrieval poisoning**, **knowledge-graph augmentation for RAG**, **adversarial examples in retrieval**, **GraphRAG** survey papers. Cite peer-reviewed or standard sources your instructor prefers; this repo does not endorse a single citation list.

## Maintaining documentation when the dataset grows

1. Add or edit JSON under `data/` (corpus and/or `data/scenarios/*.json`).
2. Run preflight: `python3 -m tests.scenario_tests`.
3. Regenerate the auto inventory: `python scripts/generate_project_docs.py`.
4. Commit both the **data** and the **updated** `docs/DATASET_AND_SCENARIOS.md`.

See [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md) for a reproducible demo path before submission.
