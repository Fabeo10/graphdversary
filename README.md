# graphdversary

A small **adversarial GraphRAG** research demo: a JSON-backed knowledge graph is ingested into **NetworkX** (topology) and **FAISS** (dense retrieval over node text). A **hybrid retriever** picks **semantic anchor** nodes, then **expands along edges** (BFS). An **adversarial module** perturbs the query, removes edges, and injects poison nodes; an **evaluator** scores precision, recall, and a simple faithfulness-style check. The **`pipeline`** script runs a baseline pass, then attacks, and prints a short report.

**Scope:** This is a teaching / baseline implementation, not production RAG or a formal security proof.

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

4. **Optional: Hugging Face token**  
   The first model download may show a warning about unauthenticated Hub requests. For higher rate limits, set a read token and export it (see [Hugging Face: User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens)):

   ```bash
   export HF_TOKEN=your_token_here
   ```

## Run the pipeline

From the **repository root** (so `data/` and `src/` resolve correctly), run:

```bash
python3 -m src.pipeline
```

**Do not** rely on `python3 src/pipeline.py` from the root unless you set `PYTHONPATH=.`; imports use the `src` package. Using `python3 -m src.pipeline` is the supported entry point.

You should see:

1. Model load / ingestion of `data/mock_corpus.json`  
2. **Stage 1:** Baseline hybrid retrieval and precision/recall vs. metadata ground-truth nodes  
3. **Stage 2:** Adversarial steps (edge removal, poison node, query perturbation)  
4. **Stage 3:** Retrieval under attack  
5. **Stage 4:** Summary including a mock “LLM” answer and faithfulness-style score  

## Data format (mock corpus)

`data/mock_corpus.json` is a JSON object with:

- **`entities`:** `{ id, content, type? }` — become graph nodes.  
- **`edges`:** `{ source, target, relation }` — become directed edges.  
- **`metadata`:** e.g. `ground_truth_nodes` (list of node ids) and `test_query` (string used as the default question).

To experiment, edit this file and re-run the pipeline.

## Project layout

| Path | Purpose |
|------|--------|
| `data/mock_corpus.json` | Example graph + metadata for evaluation |
| `src/build_graph.py` | `KnowledgeBase`: load JSON, build `DiGraph` + FAISS index |
| `src/hybrid_retriever.py` | FAISS top-k anchor + BFS hop expansion |
| `src/adversarial_module.py` | Query perturbation, edge cut, poison node + index update |
| `src/evaluator.py` | Precision, recall, faithfulness heuristic |
| `src/pipeline.py` | End-to-end script (`python3 -m src.pipeline`) |
| `requirements.txt` | Python dependencies |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — Run from repo root: `python3 -m src.pipeline`.  
- **`No module named 'faiss'`** — Activate the same venv where you ran `pip install -r requirements.txt`. On Apple Silicon, `faiss-cpu` from PyPI is usually what you want; for GPU or special builds, see [facebookresearch/faiss](https://github.com/facebookresearch/faiss).  
- **Slow or blocked first run** — The embedding model is downloaded once; ensure network access or use an offline cache path per Hugging Face / `sentence-transformers` docs.
