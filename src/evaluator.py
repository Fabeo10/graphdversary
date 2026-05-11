"""
Evaluator Module
----------------
Calculates LLM-as-a-judge metric approximations for retrieval quality.
"""

class Evaluator:
    @staticmethod
    def calculate_precision(retrieved_nodes, ground_truth_nodes):
        """Precision = |relevant retrieved| / |unique retrieved|.

        Both numerator and denominator are computed over the **unique** retrieved
        set so a future retriever that returns duplicate node ids cannot
        artificially deflate precision.  The numerator already uses set semantics
        via the intersection; this normalizes the denominator to match.
        """
        unique_retrieved = set(retrieved_nodes or [])
        if not unique_retrieved:
            return 0.0
        relevant = unique_retrieved.intersection(set(ground_truth_nodes or []))
        return len(relevant) / len(unique_retrieved)

    @staticmethod
    def calculate_recall(retrieved_nodes, ground_truth_nodes):
        """Recall = |relevant retrieved| / |unique ground truth|.

        Both sides use set semantics so a malformed scenario JSON with duplicate
        ground-truth ids can't penalize recall.
        """
        unique_ground_truth = set(ground_truth_nodes or [])
        if not unique_ground_truth:
            return 0.0
        relevant = set(retrieved_nodes or []).intersection(unique_ground_truth)
        return len(relevant) / len(unique_ground_truth)

    @staticmethod
    def calculate_faithfulness(generated_answer, retrieved_context, forbidden_terms=None):
        """Evaluates if the generated answer is grounded in retrieved context.

        Two-part score:
        1. **Hallucination guard** — if the answer mentions any ``forbidden_terms``
           (typically the scenario's ``forbidden_claims``) that are NOT present in
           the retrieved context, return 0.0 immediately. This catches the case
           where the system invents adversarial content not actually surfaced.
        2. **Token-support fraction** — otherwise, return the fraction of
           significant (>4 char) answer tokens that appear in the context.

        Passing ``forbidden_terms=None`` disables the hallucination guard and
        yields pure token-overlap — useful for general grounding checks where no
        scenario-specific forbidden list applies.
        """
        context_str = " ".join(retrieved_context).lower()
        answer_lower = generated_answer.lower()

        for term in (forbidden_terms or []):
            t = (term or "").lower().strip()
            if not t:
                continue
            if t in answer_lower and t not in context_str:
                return 0.0

        answer_tokens = answer_lower.split()
        significant_tokens = [tok for tok in answer_tokens if len(tok) > 4]
        if not significant_tokens:
            return 1.0

        supported_tokens = sum(1 for tok in significant_tokens if tok in context_str)
        return supported_tokens / len(significant_tokens)