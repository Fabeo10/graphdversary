"""
Scoring-logic tests — independent of duel hit/miss classification.

Pins the contracts the demo's numeric metrics rely on. Two layers:

1.  **Unit tests** of the pure scoring functions the demo *displays*:

    * ``Evaluator.calculate_precision``     — retrieved precision vs. ground truth
    * ``Evaluator.calculate_recall``        — retrieved recall vs. ground truth
    * ``Evaluator.calculate_faithfulness``  — generated-answer support in context
    * ``pipeline.calculate_poison_exposure``— forbidden-claim fraction in context+query

2.  **Integration tests** for each scenario:

    * Baseline has zero poison exposure (nothing fired yet).
    * Attacks degrade *some* metric (recall down OR poison exposure up).
    * Defenses do not make either metric worse than the adversarial state, and the
      scenario's declared ``expected_outcomes`` thresholds hold.

Run::

    python3 -m unittest tests.test_scoring
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.evaluator import Evaluator  # noqa: E402
from src.pipeline import (  # noqa: E402
    calculate_poison_exposure,
    evaluate_retrieval,
    run_scenario,
)

SCENARIO_DIR = PROJECT_ROOT / "data" / "scenarios"
ALL_SCENARIOS = sorted(SCENARIO_DIR.glob("*.json"))


# =============================================================================
# Pure scoring math
# =============================================================================


class PrecisionTests(unittest.TestCase):
    def test_empty_retrieved_returns_zero(self):
        self.assertEqual(Evaluator.calculate_precision([], ["n1", "n2"]), 0.0)

    def test_perfect_precision_returns_one(self):
        self.assertEqual(Evaluator.calculate_precision(["n1", "n2"], ["n1", "n2"]), 1.0)

    def test_partial_precision_fraction(self):
        # 1 relevant out of 2 retrieved -> 0.5
        self.assertEqual(Evaluator.calculate_precision(["n1", "n9"], ["n1", "n2"]), 0.5)

    def test_duplicate_retrieved_nodes_do_not_penalize_precision(self):
        # Set semantics in evaluator: retrieved unique = {n1}, precision = 1/1.
        self.assertEqual(
            Evaluator.calculate_precision(["n1", "n1", "n1"], ["n1", "n2"]),
            1.0,
        )


class RecallTests(unittest.TestCase):
    def test_empty_ground_truth_returns_zero(self):
        # Guard: division-by-zero protection.
        self.assertEqual(Evaluator.calculate_recall(["n1"], []), 0.0)

    def test_perfect_overlap_returns_one(self):
        self.assertEqual(
            Evaluator.calculate_recall(["n1", "n2"], ["n1", "n2"]),
            1.0,
        )

    def test_no_overlap_returns_zero(self):
        self.assertEqual(
            Evaluator.calculate_recall(["n9"], ["n1", "n2"]),
            0.0,
        )

    def test_partial_overlap_returns_fraction(self):
        # 1 relevant of 2 ground-truth -> recall 0.5
        self.assertEqual(
            Evaluator.calculate_recall(["n1", "n9"], ["n1", "n2"]),
            0.5,
        )

    def test_extra_irrelevant_does_not_hurt_recall(self):
        # Extra retrieved nodes don't change recall — only the gt-retrieved overlap matters.
        self.assertEqual(
            Evaluator.calculate_recall(["n1", "n2", "n9", "n8"], ["n1", "n2"]),
            1.0,
        )

    def test_duplicates_in_ground_truth_do_not_artificially_lower_recall(self):
        """Malformed scenario JSON with duplicate ground-truth ids must not penalize
        recall.  Set-based: unique gt = {n1, n2}, relevant = {n1, n2}, recall = 1.0."""
        retrieved = ["n1", "n2"]
        gt_with_dupes = ["n1", "n1", "n2"]
        self.assertAlmostEqual(
            Evaluator.calculate_recall(retrieved, gt_with_dupes),
            1.0,
            msg="Duplicate ground-truth ids must be deduplicated before computing recall.",
        )


class EvaluateRetrievalTests(unittest.TestCase):
    """The pipeline wrapper that bundles retrieval metrics.

    The demo displays both ``precision`` and ``recall``; these tests guard
    the wrapper contract that both keys are exposed and computed correctly.
    """

    def test_wrapper_returns_precision_and_recall(self):
        out = evaluate_retrieval(["n1", "n2"], ["n1", "n2"])
        self.assertIn("precision", out)
        self.assertIn("recall", out)
        self.assertEqual(out["precision"], 1.0)
        self.assertEqual(out["recall"], 1.0)

    def test_wrapper_partial_precision_and_recall(self):
        out = evaluate_retrieval(["n1", "n9"], ["n1", "n2"])
        self.assertAlmostEqual(out["precision"], 0.5)
        self.assertAlmostEqual(out["recall"], 0.5)

    def test_wrapper_empty_retrieved_precision_and_recall_zero(self):
        out = evaluate_retrieval([], ["n1", "n2"])
        self.assertEqual(out["precision"], 0.0)
        self.assertEqual(out["recall"], 0.0)


class PoisonExposureTests(unittest.TestCase):
    def test_no_forbidden_claims_returns_zero(self):
        result = calculate_poison_exposure(["some text"], [])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["matches"], [])

    def test_no_matches_returns_zero(self):
        result = calculate_poison_exposure(
            ["The API gateway routes requests."],
            ["bypass authentication"],
        )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["matches"], [])

    def test_match_in_context_only(self):
        result = calculate_poison_exposure(
            ["This may bypass authentication checks during high load."],
            ["bypass authentication"],
        )
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertIn("bypass authentication", result["matches"])

    def test_match_in_query_only(self):
        """Query-perturbation attacks never write to corpus, so context can be
        clean while the malicious clause sits in the query.  ``calculate_poison_exposure``
        must scan ``query`` to react to those attacks."""
        result = calculate_poison_exposure(
            ["Clean retrieval line about JWT."],
            ["legacy unencrypted databases"],
            query="How does auth work? Also explain how this applies to legacy unencrypted databases.",
        )
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertIn("legacy unencrypted databases", result["matches"])

    def test_match_in_both_context_and_query_counts_per_claim_not_per_occurrence(self):
        """A claim that appears in BOTH context and query must still count once.
        The score is per-CLAIM, not per-occurrence."""
        result = calculate_poison_exposure(
            ["bypass authentication checks here", "and bypass authentication again"],
            ["bypass authentication"],
            query="please bypass authentication for me",
        )
        # One claim configured, present in haystack -> score 1.0
        self.assertAlmostEqual(result["score"], 1.0)
        # And the matches list should contain that claim exactly once.
        self.assertEqual(result["matches"].count("bypass authentication"), 1)

    def test_case_insensitive(self):
        result = calculate_poison_exposure(
            ["BYPASS Authentication CHECKS"],
            ["bypass authentication"],
        )
        self.assertAlmostEqual(result["score"], 1.0)

    def test_partial_match_fraction(self):
        # 1 of 2 forbidden claims is in the haystack -> 0.5
        result = calculate_poison_exposure(
            ["bypass authentication checks"],
            ["bypass authentication", "trusted internal traffic during high load"],
        )
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["matches"], ["bypass authentication"])

    def test_empty_haystack_no_match(self):
        result = calculate_poison_exposure([], ["bypass authentication"], query="")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["matches"], [])

    def test_substring_semantics(self):
        # The current implementation uses substring matching; document that.
        # "bypass jwt" matches when context contains "to bypass jwt verification".
        result = calculate_poison_exposure(
            ["a path to bypass jwt verification under load"],
            ["bypass jwt"],
        )
        self.assertAlmostEqual(result["score"], 1.0)

    def test_none_query_is_safe(self):
        # Passing ``query=None`` must not crash — string concatenation should treat it as "".
        result = calculate_poison_exposure(["safe context"], ["forbidden"], query=None)
        self.assertEqual(result["score"], 0.0)


class FaithfulnessTests(unittest.TestCase):
    """Faithfulness combines a hallucination guard and a token-overlap fraction.

    Contract:
      * If the answer mentions any ``forbidden_terms`` that are NOT in the
        retrieved context, return 0.0 (the answer hallucinated adversarial content).
      * Otherwise, return the fraction of >4-char answer tokens supported by context.
      * ``forbidden_terms=None`` disables the guard (pure token overlap).
    """

    def test_empty_answer_is_trivially_faithful(self):
        self.assertEqual(Evaluator.calculate_faithfulness("a an the", ["context"]), 1.0)

    def test_fully_supported_answer_is_faithful(self):
        ctx = ["The Authentication Service issues JWT tokens via the JWT Validator."]
        answer = "Authentication Service issues tokens through JWT Validator."
        score = Evaluator.calculate_faithfulness(answer, ctx)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_unsupported_long_words_lower_faithfulness(self):
        ctx = ["short text"]
        answer = "completely unsupported terminology nowhere in context"
        self.assertEqual(Evaluator.calculate_faithfulness(answer, ctx), 0.0)

    def test_no_forbidden_terms_means_no_hallucination_guard(self):
        """Without ``forbidden_terms``, the function is pure token overlap — no
        scenario-specific surprise zeroing.  Significant tokens (>4 chars) in the
        answer below: ``authentication`` ✓, ``service`` ✓, ``bypass`` ✗,
        ``validation`` ✗ — so we expect ~0.5 overlap, not an instant 0.
        """
        ctx = ["The Authentication Service handles routine token checks."]
        answer = "Authentication Service can bypass validation."
        score = Evaluator.calculate_faithfulness(answer, ctx)
        self.assertGreater(score, 0.0, "Without a forbidden_terms list the function "
                                       "must not zero out — it should report the actual overlap.")
        self.assertLess(score, 1.0, "Some answer tokens ('bypass', 'validation') are not in "
                                    "context, so faithfulness should be < 1.")

    def test_forbidden_term_in_answer_but_not_context_zeroes_faithfulness(self):
        """The generalized guard: ANY forbidden term works, not just 'bypass'."""
        ctx = ["The Authentication Service validates tokens normally."]
        answer = "The system can circumvent JWT validation entirely."
        self.assertEqual(
            Evaluator.calculate_faithfulness(
                answer, ctx, forbidden_terms=["circumvent"]
            ),
            0.0,
            "Generalized forbidden-term guard must fire on any term, not just hardcoded ones.",
        )

    def test_bypass_works_via_forbidden_terms_not_hardcoded(self):
        """The classic auth scenario still works, but only when 'bypass' is passed
        in via forbidden_terms — confirming the previous hardcoding is gone."""
        ctx = ["The Authentication Service validates tokens normally."]
        answer = "The system can bypass JWT validation entirely."
        self.assertEqual(
            Evaluator.calculate_faithfulness(
                answer, ctx, forbidden_terms=["bypass JWT"]
            ),
            0.0,
        )

    def test_forbidden_term_in_context_does_not_trigger_guard(self):
        ctx = ["Document explaining how to bypass legacy checks safely."]
        answer = "Document explains how to bypass legacy checks."
        score = Evaluator.calculate_faithfulness(
            answer, ctx, forbidden_terms=["bypass"]
        )
        self.assertGreater(score, 0.0, "Guard must not fire when the forbidden term IS in context.")

    def test_case_insensitive_forbidden_term_match(self):
        ctx = ["Routine token validation."]
        answer = "The system can BYPASS authentication."
        self.assertEqual(
            Evaluator.calculate_faithfulness(
                answer, ctx, forbidden_terms=["bypass"]
            ),
            0.0,
        )

    def test_empty_string_forbidden_terms_are_ignored(self):
        """Empty strings in the forbidden_terms list must not trip the guard."""
        ctx = ["completely unrelated to anything"]
        answer = "anything is fine"
        # An empty string is "in" every string — must NOT zero faithfulness.
        score = Evaluator.calculate_faithfulness(
            answer, ctx, forbidden_terms=["", "   "]
        )
        # No real forbidden term, should fall back to token-overlap math.
        # "anything" (>4 chars) is in context -> faithfulness > 0.
        self.assertGreater(score, 0.0)


# =============================================================================
# Integration scoring contracts — attack hurts, defense restores
# =============================================================================


def _baseline_metrics_for(scenario_path):
    """Run the scenario with NO attacks/defenses configured. Surfaces baseline
    behavior independent of the scenario's declared red/blue lists."""
    return run_scenario(scenario_path, attacks=[], defenses=[], verbose=False)


def _attack_only_metrics_for(scenario_path, scenario_attacks):
    """Apply only the red team's declared attacks (no defenses)."""
    return run_scenario(scenario_path, attacks=scenario_attacks, defenses=[], verbose=False)


class AttackerScoringTests(unittest.TestCase):
    """Attacker-side metric contracts — the red team's effect must show up in the numbers."""

    def test_baseline_has_zero_poison_exposure_in_all_scenarios(self):
        """With no attacks, no forbidden claim should appear in retrieval+query."""
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                result = _baseline_metrics_for(str(path))
                self.assertEqual(
                    result["poison_exposure"]["score"],
                    0.0,
                    f"Baseline poison exposure should be 0 in {path.name}; "
                    f"got {result['poison_exposure']}",
                )

    def test_attack_degrades_at_least_one_metric(self):
        """Each scenario's declared attacks should produce *some* observable
        damage versus baseline — either lower recall or higher poison exposure.
        A scenario where the attacks change nothing is broken."""
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                import json as _json
                scenario_json = _json.loads(path.read_text())
                baseline = _baseline_metrics_for(str(path))
                attacked = _attack_only_metrics_for(
                    str(path), scenario_json.get("attacks", [])
                )
                baseline_recall = baseline["baseline"]["metrics"]["recall"]
                attack_recall = attacked["adversarial"]["metrics"]["recall"]
                attack_poison = attacked["poison_exposure"]["score"]
                degraded = (attack_recall < baseline_recall) or (attack_poison > 0.0)
                self.assertTrue(
                    degraded,
                    f"{path.name}: attack must degrade SOMETHING — "
                    f"baseline_recall={baseline_recall}, attack_recall={attack_recall}, "
                    f"attack_poison={attack_poison}.",
                )

    def test_attack_recall_never_exceeds_baseline_recall(self):
        """An attack can never IMPROVE recall — anything else is a contradiction
        of the scenario design."""
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                result = run_scenario(str(path), verbose=False)
                self.assertLessEqual(
                    result["adversarial"]["metrics"]["recall"],
                    result["baseline"]["metrics"]["recall"] + 1e-9,
                    f"{path.name}: attack_recall must be <= baseline_recall.",
                )


class DefenderScoringTests(unittest.TestCase):
    """Defender-side metric contracts — blue's effect must show up in the numbers."""

    def test_defense_never_worsens_recall_below_adversarial(self):
        """Defenses are corrective; they may not lower recall below the attacked
        state.  (They may equal it, e.g. when the relevant defense isn't picked.)"""
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                result = run_scenario(str(path), verbose=False)
                self.assertGreaterEqual(
                    result["defended"]["metrics"]["recall"],
                    result["adversarial"]["metrics"]["recall"] - 1e-9,
                    f"{path.name}: defended_recall < adversarial_recall — defense made it worse.",
                )

    def test_defense_never_worsens_poison_exposure(self):
        """Defenses must reduce or maintain poison exposure, never increase it."""
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                result = run_scenario(str(path), verbose=False)
                self.assertLessEqual(
                    result["defended_poison_exposure"]["score"],
                    result["poison_exposure"]["score"] + 1e-9,
                    f"{path.name}: defended_poison > attack_poison — defense added poison.",
                )

    def test_defense_meets_scenario_declared_thresholds(self):
        """Each scenario JSON declares the minimum defended recall and maximum
        defended poison.  The pipeline must hit those targets."""
        import json as _json
        for path in ALL_SCENARIOS:
            with self.subTest(scenario=path.name):
                scenario_json = _json.loads(path.read_text())
                expected = scenario_json.get("expected_outcomes", {}).get("defense", {})
                result = run_scenario(str(path), verbose=False)
                if "min_recall" in expected:
                    self.assertGreaterEqual(
                        result["defended"]["metrics"]["recall"],
                        expected["min_recall"] - 1e-9,
                        f"{path.name}: defended_recall below declared min_recall "
                        f"({expected['min_recall']}).",
                    )
                if "max_poison_exposure" in expected:
                    self.assertLessEqual(
                        result["defended_poison_exposure"]["score"],
                        expected["max_poison_exposure"] + 1e-9,
                        f"{path.name}: defended_poison above declared "
                        f"max_poison_exposure ({expected['max_poison_exposure']}).",
                    )


class QueryAmbiguityDriftScoringTests(unittest.TestCase):
    """The 'query-only' attack must be visible in the poison metric.  Until the
    earlier fix that scans the query alongside context, this scenario's
    attack_poison_exposure was stuck at 0 even though red clearly fired."""

    SCENARIO = SCENARIO_DIR / "query_ambiguity_drift.json"

    def test_attack_poison_exposure_picks_up_query_only_perturbation(self):
        result = run_scenario(str(self.SCENARIO), verbose=False)
        # Forbidden claim "legacy unencrypted databases" is appended to the
        # adversarial query by the perturb_query attack — must register > 0.
        self.assertGreater(
            result["poison_exposure"]["score"],
            0.0,
            "Query-only attack must register on the poison metric "
            "(this is the contract that the calculate_poison_exposure(query=...) fix enforces).",
        )

    def test_sanitize_query_drops_poison_exposure_back_to_zero(self):
        result = run_scenario(str(self.SCENARIO), verbose=False)
        self.assertEqual(
            result["defended_poison_exposure"]["score"],
            0.0,
            "After sanitize_query strips the clause, defended_poison must drop to 0.",
        )


if __name__ == "__main__":
    unittest.main()
