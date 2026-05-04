# Demo checklist (final presentation)

Use this path **from the repository root** so imports and `data/` paths resolve. Adjust paths if your clone lives elsewhere.

## Before the session

1. **Environment**

   ```bash
   cd graphdversary
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Optional:** set `HF_TOKEN` if Hub rate limits are an issue (first embedding download).

3. **Regenerate docs if you changed `data/`** (scenarios or corpus):

   ```bash
   python scripts/generate_project_docs.py
   ```

4. **Preflight** (must pass for a clean “tests green” story):

   ```bash
   python3 -m tests.scenario_tests
   ```

5. **Optional — Agent Duel with Ollama:** install Ollama, `ollama pull llama3.2:3b` and `ollama pull qwen2.5:3b`, run `ollama serve`. If skipped, use **Agent-selected** mode in the UI.

## Launch the demo

```bash
streamlit run src/demo_app.py
```

## Suggested 8–10 minute flow

| Step | What to do | What to say |
|------|------------|-------------|
| 1 | Sidebar: pick **Authentication Token Poisoning** (or walk all three). | “Hybrid GraphRAG: dense anchors + hop expansion on a small services graph.” |
| 2 | **Overview** tab: read metrics row; mention **white-box** vs other scenarios. | “Ground truth nodes are defined in the scenario—not universal truth.” |
| 3 | Sidebar: toggle **one red** attack; watch **attack recall** / poison. | “Structured tampering changes what retrieval surfaces.” |
| 4 | Toggle **blue** defenses; compare **defended poison** and recall. | “Defenses are scenario-approved controls, not magic.” |
| 5 | **Realtime Graph** or **Scenario Graphs:** switch **Baseline → Red → Blue**. | “Visual matches the same pipeline state as the metrics.” |
| 6 | **Traces & Logs:** expand **Retrieval trace** for baseline vs attacked. | “Anchors and BFS hops explain *why* nodes appeared.” |
| 7 | **Tests** tab: show preflight expectations for current toggles. | “Scenario expectations encode the teaching point.” |
| 8 | **Agent Duel** (optional): **Agent-selected** first; then **Hybrid Ollama** if available. | “Agents only choose among scenario actions—no arbitrary graph writes.” |
| 9 | Closing | Point to `docs/PROJECT_BRIEF.md` for threat model and limitations. |

## If something breaks

- **`ModuleNotFoundError: src`** — Run Streamlit and tests from repo root; use `python3 -m …` as in the README.
- **Preflight fails** — Restore scenario toggles to defaults or fix `expected_outcomes` / data.
- **Ollama errors** — Duel falls back to deterministic selection; mention that in the talk.

## Commands summary

```bash
python3 -m tests.scenario_tests
streamlit run src/demo_app.py
python scripts/generate_project_docs.py   # after editing data/
```
