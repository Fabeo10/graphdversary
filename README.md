# graphdversary

A small **adversarial GraphRAG** research demo: a JSON-backed knowledge graph is ingested into **NetworkX** (topology) and **FAISS** (dense retrieval over node text). A **hybrid retriever** picks **semantic anchor** nodes, then **expands along edges** (BFS). An **adversarial module** perturbs the query, removes edges, and injects poison nodes; an **evaluator** scores precision, recall, poison exposure, and a simple faithfulness-style check.

The repo now supports both:

- a scenario-driven terminal pipeline for repeatable runs
- an interactive **Streamlit** demo for presentation mode with query controls, attack toggles, graph comparison, retrieval traces, and metrics

**Scope:** This is a teaching / baseline implementation, not production RAG or a formal security proof.

## Documentation (final project / demo package)

| Document | Purpose |
|----------|---------|
| [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) | Objective, threat model, in/out scope, metrics, limitations, how to keep docs in sync when data grows. |
| [docs/DEMO_CHECKLIST.md](docs/DEMO_CHECKLIST.md) | Reproducible setup, preflight, and a suggested talk track for the Streamlit demo. |
| [docs/DATASET_AND_SCENARIOS.md](docs/DATASET_AND_SCENARIOS.md) | **Auto-generated** inventory of `data/mock_corpus.json` and every file in `data/scenarios/`. |

After you add scenarios or change the toy corpus, regenerate the inventory:

```bash
python scripts/generate_project_docs.py
```

Commit the updated `docs/DATASET_AND_SCENARIOS.md` with your data changes.

## Requirements

- **Python 3.10+** (3.11 recommended; tested with 3.11 in development)
- Internet access on **first run** to download the default embedding model from the Hugging Face Hub (`sentence-transformers/all-MiniLM-L6-v2`). Later runs use the local cache.

## Setup

1. **Clone the repository** (or unpack the project) and go to the project root:

   ```bash
   cd graphdversary
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   | Package | Role |
   |--------|------|
   | `networkx` | Directed graph for entities and relations |
   | `faiss-cpu` | L2 vector index over node embeddings |
   | `sentence-transformers` | Encode query and node text into embeddings |
   | `numpy` | Numeric arrays for FAISS |
   | `streamlit` | Interactive demo app |
   | `plotly` | Interactive graph rendering |

4. **Optional: Hugging Face token**  
   The first model download may show a warning about unauthenticated Hub requests. For higher rate limits, set a read token and export it (see [Hugging Face: User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens)):

   ```bash
   export HF_TOKEN=your_token_here
   ```

## Run the terminal pipeline

From the **repository root** (so `data/` and `src/` resolve correctly), run:

```bash
python3 -m src.pipeline
```

**Do not** rely on `python3 src/pipeline.py` from the root unless you set `PYTHONPATH=.`; imports use the `src` package. Using `python3 -m src.pipeline` is the supported entry point.

You should see:

1. Model load / ingestion of `data/mock_corpus.json`  
2. **Stage 1:** Baseline hybrid retrieval and precision/recall vs. scenario ground-truth nodes  
3. **Stage 2:** Scenario-defined adversarial steps (edge removal, poison node, query perturbation)  
4. **Stage 3:** Retrieval under attack  
5. **Stage 4:** Blue-team defense controls and defended retrieval
6. **Stage 5:** Summary including attack vs. defended recall and poison exposure

To run a specific scenario:

```bash
python3 -m src.pipeline --scenario data/scenarios/auth_token_poison.json
```

To inspect the full structured output:

```bash
python3 -m src.pipeline --json
```

## Run scenario preflight checks

Each scenario has `expected_outcomes` that compare ground truth against actual baseline, red-team, and blue-team outcomes. Before a demo, run:

```bash
python3 -m tests.scenario_tests
```

The preflight checks validate things like:

- baseline recall meets the scenario threshold
- red-team attacks expose the expected vulnerable outcome
- blue-team defenses remove forbidden claims or poisoned nodes
- attack and defense logs report successful execution

To check one scenario:

```bash
python3 -m tests.scenario_tests --scenario data/scenarios/auth_token_poison.json
```

## Run the interactive demo

From the repository root:

```bash
streamlit run src/demo_app.py
```

The app opens in your browser. Use the sidebar to:

1. Select a scenario.
2. Edit the query.
3. Tune semantic anchors (`top_k`) and graph expansion depth.
4. Toggle individual attacks.
5. Toggle blue-team defenses.
6. Switch between baseline, red-team, and blue-team graph states.
7. Run the Agent Duel in Agent-selected or Hybrid Ollama mode.
8. Compare retrieval context, traces, metrics, and poison exposure.

## Optional local LLM duel setup

The Agent Duel works without an LLM in **Agent-selected** mode. In that mode, local agent logic chooses from the remaining scenario-approved red/blue options.

For **Hybrid Ollama**, install [Ollama](https://ollama.com/) and pull **both** small models so red (attacker) and blue (defender) can run as separate weights with no shared chat session—each agent turn is a single stateless `/api/generate` request:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

- **Red team (attacker)** defaults to **Llama**: `llama3.2:3b`
- **Blue team (defender)** defaults to **Qwen**: `qwen2.5:3b`

The models do not share a conversation: the red model never sees the blue model’s prior hidden chain-of-thought, and vice versa. They only see the same published retrieval metrics and their own role’s allowed actions (the environment), matching a simple “two independent agents” simulation.

Then start or verify the Ollama service:

```bash
ollama serve
```

In the Streamlit app:

1. Open **Agent Duel**.
2. Choose **Hybrid Ollama**.
3. Adjust **Red team** and **Blue team** model names if you use different Ollama tags.
4. Click **Run next agent**, **Run next full turn**, or **Auto-play duel**.

Use **Duel scope -> All scenarios** to run a full gauntlet across every built-in scenario in one pass.

If Ollama is not running, the duel still works and falls back to rule-based rationales.

Built-in scenarios are summarized by **test type** in each JSON file; see the auto-generated table in [docs/DATASET_AND_SCENARIOS.md](docs/DATASET_AND_SCENARIOS.md) for titles, attack/defense counts, and corpus links (refresh with `python scripts/generate_project_docs.py` after you extend the toy dataset).

## Data format (mock corpus)

`data/mock_corpus.json` is a JSON object with:

- **`entities`:** `{ id, content, type? }` — become graph nodes.  
- **`edges`:** `{ source, target, relation }` — become directed edges.  
- **`metadata`:** e.g. `ground_truth_nodes` (list of node ids) and `test_query` (string used as the default question).

Scenario files live in `data/scenarios/`. A scenario points at a corpus and defines:

- **`query`:** the demo question
- **`ground_truth_nodes`:** relevant node ids for precision/recall
- **`top_k` / `hop_depth`:** retrieval settings
- **`attacks`:** edge removals, poison injections, and query perturbations
- **`defenses`:** query sanitization, protected edge restoration, untrusted node removal, and forbidden-claim blocking
- **`test_type`:** white-box, black-box, or gray-box classification
- **`mock_answer`:** answer text used for the faithfulness check
- **`forbidden_claims`:** claims that should not appear in retrieved context
- **`expected_outcomes`:** pass/fail thresholds for demo preflight checks
- **`presenter_notes`:** talking points for the demo

To experiment, edit the corpus or add a new scenario file and re-run the pipeline or Streamlit app. Then run `python scripts/generate_project_docs.py` so [docs/DATASET_AND_SCENARIOS.md](docs/DATASET_AND_SCENARIOS.md) stays current.

## Project layout

| Path | Purpose |
|------|--------|
| `docs/` | Project brief, demo checklist, auto-generated [DATASET_AND_SCENARIOS.md](docs/DATASET_AND_SCENARIOS.md) |
| `scripts/generate_project_docs.py` | Regenerates dataset/scenario inventory from `data/` |
| `data/mock_corpus.json` | Example graph + metadata for evaluation |
| `data/scenarios/*.json` | Repeatable red-team / blue-team presentation scenarios |
| `src/build_graph.py` | `KnowledgeBase`: load JSON, build `DiGraph` + FAISS index |
| `src/hybrid_retriever.py` | FAISS top-k anchor + BFS hop expansion with trace output |
| `src/adversarial_module.py` | Query perturbation, edge cut, poison node + index update |
| `src/evaluator.py` | Precision, recall, faithfulness heuristic |
| `src/pipeline.py` | Scenario runner + terminal script (`python3 -m src.pipeline`) |
| `src/agent_duel.py` | Agent-selected and optional Ollama red/blue agent duel |
| `src/scenario_assertions.py` | Ground truth vs. outcome expectation checks |
| `src/demo_app.py` | Interactive Streamlit demo |
| `tests/scenario_tests.py` | Preflight runner for all scenarios |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — Run from repo root: `python3 -m src.pipeline`.  
- **`No module named 'faiss'`** — Activate the same venv where you ran `pip install -r requirements.txt`. On Apple Silicon, `faiss-cpu` from PyPI is usually what you want; for GPU or special builds, see [facebookresearch/faiss](https://github.com/facebookresearch/faiss).  
- **Slow or blocked first run** — The embedding model is downloaded once; ensure network access or use an offline cache path per Hugging Face / `sentence-transformers` docs.
- **Streamlit command not found** — Re-run `pip install -r requirements.txt` inside the active virtual environment.
