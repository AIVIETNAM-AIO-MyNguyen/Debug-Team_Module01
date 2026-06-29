import re
import logging
from typing import List, Dict, Any
import time

# Graceful import of datasets to avoid crashes if datasets package is not installed
try:
    from datasets import Dataset
except ImportError:
    class Dataset:
        """Fallback Dataset class to avoid import failure when 'datasets' is missing."""
        def __init__(self, data: Dict[str, List[Any]]):
            self.data = data
            
        @classmethod
        def from_dict(cls, data: Dict[str, List[Any]]):
            return cls(data)
            
        def __getitem__(self, key):
            return self.data[key]
            
        def __iter__(self):
            # Iterate as a list of dicts
            keys = list(self.data.keys())
            num_rows = len(self.data[keys[0]]) if keys else 0
            for i in range(num_rows):
                yield {k: self.data[k][i] for k in keys}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Stage2GenerativeEvaluator:
    """Manages text generation testing and RAGAS semantic quality audits."""
    def __init__(self, top_5_configs: List[Dict[str, str]], judge_llm: Any):
        self.configs = top_5_configs
        self.judge = judge_llm

    def compile_rag_response(self, question: str, contexts: List[str], generator_llm: Any) -> str:
        """Merges text context and prompts to synthesize the final system response."""
        context_block = "\n".join([f"- {c}" for c in contexts])
        prompt = (
            f"You are a technical assistant. Answer the question using ONLY the retrieved contexts below. "
            f"If the contexts do not contain enough information, state that clearly.\n\n"
            f"Retrieved Contexts:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        if generator_llm is None:
            raise ValueError("Generator LLM must be configured for real-only compilation.")
            
        return generator_llm(prompt).strip()

    def _call_llm_as_judge(self, question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """Uses LLM-as-a-judge prompting to evaluate RAGAS-style scores."""
        scores = {}
        context_str = "\n".join([f"- {c}" for c in contexts])
        
        def parse_score(text: str) -> float:
            # Remove think tags
            cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            # Try to find a decimal number or integer in the remaining text
            # Matches: 0.85, 1.0, 1, 0, .5, etc.
            match = re.search(r'\b(0?\.\d+|1\.0+|1|0)\b', cleaned)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            # Fallback to any digit/float
            match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            return 0.5

        # Faithfulness Evaluation
        prompt_f = (
            f"Rate the FAITHFULNESS of the following Answer based on the Contexts. "
            f"Faithfulness measures if all statements in the Answer can be directly inferred from the Contexts. "
            f"Output ONLY a single float between 0.0 and 1.0 (e.g. 0.85). Do not output other text.\n\n"
            f"Contexts:\n{context_str}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Faithfulness Score:"
        )
        # Add time to check runtime of judge calls
        start_f = time.time()
        resp = self.judge(prompt_f)
        logger.info(
            f"Faithfulness took {time.time() - start_f:.2f}s"
        )
        try:
            scores["faithfulness"] = parse_score(resp)
        except Exception as e:
            logger.error(f"Error parsing faithfulness score from: {resp}. Error: {e}")
            scores["faithfulness"] = 0.5
            
        # Relevancy Evaluation
        prompt_r = (
            f"Rate the ANSWER RELEVANCY of the following Answer to the Question. "
            f"Relevancy measures how directly the answer addresses the question without containing redundant/fluffy info. "
            f"Output ONLY a single float between 0.0 and 1.0 (e.g. 0.90). Do not output other text.\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Answer Relevancy Score:"
        )
        # Add time to check runtime of judge calls
        start_r= time.time()
        resp = self.judge(prompt_r)
        logger.info(
            f"Relevancy took {time.time() - start_r:.2f}s"
        )
        try:
            scores["answer_relevancy"] = parse_score(resp)
        except Exception as e:
            logger.error(f"Error parsing relevancy score from: {resp}. Error: {e}")
            scores["answer_relevancy"] = 0.5
            
        # Correctness Evaluation
        prompt_c = (
            f"Rate the ANSWER CORRECTNESS of the following Generated Answer compared to the Ground Truth. "
            f"Correctness measures both semantic matching and factual similarity to the target ground truth. "
            f"Output ONLY a single float between 0.0 and 1.0 (e.g. 0.75). Do not output other text.\n\n"
            f"Ground Truth Answer:\n{ground_truth}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Answer Correctness Score:"
        )
        # Add time to check runtime of judge calls
        start_c =time.time()
        resp = self.judge(prompt_c)
        logger.info(
            f"Correctness took {time.time() - start_c:.2f}s"
        )
        try:
            scores["answer_correctness"] = parse_score(resp)
        except Exception as e:
            logger.error(f"Error parsing correctness score from: {resp}. Error: {e}")
            scores["answer_correctness"] = 0.5
            
        # Bound all scores between 0.0 and 1.0
        for k in scores:
            scores[k] = max(0.0, min(1.0, scores[k]))
            
        return scores

    def execute_ragas_audit(self, evaluation_payload: Any) -> Dict[str, float]:
        """Calculates Faithfulness, Answer Relevancy, and Answer Correctness metrics."""
        if self.judge is None:
            raise ValueError("Judge LLM must be configured for real-only RAGAS evaluations.")

        # Unpack payload items robustly supporting both datasets.Dataset and dictionary wrappers
        try:
            questions = evaluation_payload["question"]
            answers = evaluation_payload["answer"]
            contexts_list = evaluation_payload["contexts"]
            ground_truths = evaluation_payload["ground_truth"]
        except Exception:
            # Fallback for other data structures
            if isinstance(evaluation_payload, dict):
                questions = evaluation_payload.get("question", [])
                answers = evaluation_payload.get("answer", [])
                contexts_list = evaluation_payload.get("contexts", [])
                ground_truths = evaluation_payload.get("ground_truth", [])
            else:
                questions = [row["question"] for row in evaluation_payload]
                answers = [row["answer"] for row in evaluation_payload]
                contexts_list = [row["contexts"] for row in evaluation_payload]
                ground_truths = [row["ground_truth"] for row in evaluation_payload]

        num_records = len(questions)
        if num_records == 0:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0}

        accum_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": 0.0}
        
        for i in range(num_records):
            q = questions[i]
            a = answers[i]
            ctx = contexts_list[i]
            gt = ground_truths[i]
            
            item_scores = self._call_llm_as_judge(q, a, ctx, gt)
                
            for k in accum_scores:
                accum_scores[k] += item_scores.get(k, 0.0)

        # Average the scores
        avg_scores = {}
        for k in accum_scores:
            avg_scores[k] = max(0.0, min(1.0, accum_scores[k] / num_records))
            
        return avg_scores
