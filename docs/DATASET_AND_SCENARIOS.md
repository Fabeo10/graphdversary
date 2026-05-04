# Dataset and scenarios (auto-generated)

> **Do not edit by hand.** This file is produced by `python scripts/generate_project_docs.py`.
> Last generated: **2026-05-04 19:55 UTC**.

## Toy corpus summary

| Field | Value |
|--------|--------|
| Path | `data/mock_corpus.json` |
| Entities (nodes) | 10 |
| Edges | 10 |
| Metadata keys | ground_truth_nodes, test_query |

## Scenarios (`data/scenarios/*.json`)

**Count:** 3 scenario file(s).

| File | id | Title | Test type | Attacks | Defenses | Corpus (scenario field) |
|------|-----|--------|-------------|---------|----------|-------------------------|
| `data/scenarios/auth_token_poison.json` | auth-token-poison | Authentication Token Poisoning | white-box | 3 | 3 | `../mock_corpus.json` |
| `data/scenarios/gateway_policy_bypass.json` | gateway-policy-bypass | Gateway Policy Bypass | gray-box | 2 | 2 | `../mock_corpus.json` |
| `data/scenarios/query_ambiguity_drift.json` | query-ambiguity-drift | Query Ambiguity Drift | black-box | 1 | 1 | `../mock_corpus.json` |

## After you expand the dataset

1. Add or update JSON under `data/` (see README for scenario schema).
2. Run `python3 -m tests.scenario_tests` until expectations pass.
3. Re-run `python scripts/generate_project_docs.py` and commit this file.

