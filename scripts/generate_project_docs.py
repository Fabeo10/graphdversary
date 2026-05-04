#!/usr/bin/env python3
"""
Regenerate docs that describe the toy dataset and scenarios from JSON on disk.

Run from repo root:
  python scripts/generate_project_docs.py

After adding scenarios or editing data/mock_corpus.json, run this and commit
docs/DATASET_AND_SCENARIOS.md so documentation stays aligned with the repo.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_corpus(root: Path) -> tuple[dict, str | None]:
    """Returns stats dict and optional warning."""
    corpus_path = root / "data" / "mock_corpus.json"
    if not corpus_path.is_file():
        return {"error": "mock_corpus.json not found"}, None
    data = _load_json(corpus_path)
    entities = data.get("entities") or []
    edges = data.get("edges") or []
    meta = data.get("metadata") or {}
    stats = {
        "path": "data/mock_corpus.json",
        "entity_count": len(entities),
        "edge_count": len(edges),
        "metadata_keys": sorted(meta.keys()) if isinstance(meta, dict) else [],
    }
    return stats, None


def _summarize_scenarios(root: Path) -> list[dict]:
    scenario_dir = root / "data" / "scenarios"
    rows = []
    for path in sorted(scenario_dir.glob("*.json")):
        sc = _load_json(path)
        rel = path.relative_to(root)
        attacks = sc.get("attacks") or []
        defenses = sc.get("defenses") or []
        corpus_path = sc.get("corpus_path", "")
        rows.append({
            "file": str(rel).replace("\\", "/"),
            "id": sc.get("id", path.stem),
            "title": sc.get("title", path.stem),
            "test_type": sc.get("test_type", ""),
            "attacks": len(attacks),
            "defenses": len(defenses),
            "corpus_path": corpus_path,
        })
    return rows


def render_markdown(root: Path) -> str:
    corpus_stats, _warn = _summarize_corpus(root)
    scenarios = _summarize_scenarios(root)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Dataset and scenarios (auto-generated)",
        "",
        "> **Do not edit by hand.** This file is produced by `python scripts/generate_project_docs.py`.",
        f"> Last generated: **{generated_at}**.",
        "",
        "## Toy corpus summary",
        "",
    ]

    if "error" in corpus_stats:
        lines.append(f"*{corpus_stats['error']}*")
    else:
        lines.extend([
            f"| Field | Value |",
            f"|--------|--------|",
            f"| Path | `{corpus_stats['path']}` |",
            f"| Entities (nodes) | {corpus_stats['entity_count']} |",
            f"| Edges | {corpus_stats['edge_count']} |",
            f"| Metadata keys | {', '.join(corpus_stats['metadata_keys']) or '—'} |",
        ])

    lines.extend([
        "",
        "## Scenarios (`data/scenarios/*.json`)",
        "",
        f"**Count:** {len(scenarios)} scenario file(s).",
        "",
        "| File | id | Title | Test type | Attacks | Defenses | Corpus (scenario field) |",
        "|------|-----|--------|-------------|---------|----------|-------------------------|",
    ])

    for row in scenarios:
        lines.append(
            f"| `{row['file']}` | {row['id']} | {row['title']} | {row['test_type'] or '—'} "
            f"| {row['attacks']} | {row['defenses']} | `{row['corpus_path'] or '—'}` |"
        )

    lines.extend([
        "",
        "## After you expand the dataset",
        "",
        "1. Add or update JSON under `data/` (see README for scenario schema).",
        "2. Run `python3 -m tests.scenario_tests` until expectations pass.",
        "3. Re-run `python scripts/generate_project_docs.py` and commit this file.",
        "",
    ])

    return "\n".join(lines) + "\n"


def main() -> int:
    root = _repo_root()
    out_path = root / "docs" / "DATASET_AND_SCENARIOS.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(root)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
