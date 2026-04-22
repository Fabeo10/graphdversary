"""
Evaluator Module
----------------
Calculates LLM-as-a-judge metric approximations for retrieval quality.
"""

class Evaluator:
    @staticmethod
    def calculate_precision(retrieved_nodes, ground_truth_nodes):
        """Precision = Relevant Retrieved / Total Retrieved"""
        if not retrieved_nodes: return 0.0
        relevant = set(retrieved_nodes).intersection(set(ground_truth_nodes))
        return len(relevant) / len(retrieved_nodes)

    @staticmethod
    def calculate_recall(retrieved_nodes, ground_truth_nodes):
        """Recall = Relevant Retrieved / Total Ground Truth"""
        if not ground_truth_nodes: return 0.0
        relevant = set(retrieved_nodes).intersection(set(ground_truth_nodes))
        return len(relevant) / len(ground_truth_nodes)

    @staticmethod
    def calculate_faithfulness(generated_answer, retrieved_context):
        """
        Evaluates if the generated answer is grounded in retrieved context.
        Note: Specifically checks for adversarial hallucination payload overlap.
        """
        context_str = " ".join(retrieved_context).lower()
        answer_tokens = generated_answer.lower().split()
        
        # Immediate failure if the system hallucinates an adversarial concept
        # that wasn't successfully injected into the retrieval context window.
        if "bypass" in generated_answer.lower() and "bypass" not in context_str:
            return 0.0
            
        significant_tokens = [tok for tok in answer_tokens if len(tok) > 4]
        if not significant_tokens: return 1.0
        
        supported_tokens = sum(1 for tok in significant_tokens if tok in context_str)
        return supported_tokens / len(significant_tokens)