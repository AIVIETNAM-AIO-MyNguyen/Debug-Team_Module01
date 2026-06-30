import os, sys, math
import time
import json
import pandas as pd
import re
import logging
from typing import List, Dict, Any

from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage2_evaluator")

# Check if NLTK is installed and download 'punkt_tab' tokenizer if not present
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    nltk = None

# Evaluation with Local Ollama + Ollama Embeddings
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# Get current directory and src directory for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # path to src/evaluation/
src_dir = os.path.dirname(current_dir)  # path to src/

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import core modules for retrieval, query transformation, post-processing, caching, index manaagement
from core.query_transforms import QueryTransformer
from core.retrievers import ModularRetriever
from core.post_processors import PostProcessor
from core.cache_manager import LocalCacheManager
from core.index_manager import IndexManager

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
        start_r = time.time()
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
        start_c = time.time()
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

        try:
            questions = evaluation_payload["question"]
            answers = evaluation_payload["answer"]
            contexts_list = evaluation_payload["contexts"]
            ground_truths = evaluation_payload["ground_truth"]
        except Exception:
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

class RagEvaluator:
    def __init__(
        self, 
        jsonl_path: str = "data/processed/questions/questions.jsonl",
        metrics_log_file: str = "reports/stage1_screening_logs.csv",
        result_log_file: str = "reports/ragas_evaluation_checkpoint_local_90.csv",
        delay_requests: float = 0, # 0 for local
        max_questions: int = None
    ):
        """
        Initialize RAG evaluator using Ragas + Local Ollama.
        """

        # Set up project root
        project_root = os.path.dirname(src_dir)

        # Set up paths for input and output files
        self.jsonl_path = os.path.join(project_root, jsonl_path)
        self.metrics_log_file = os.path.join(project_root, metrics_log_file)
        self.result_log_file = os.path.join(project_root, result_log_file)
        self.delay_requests = delay_requests
        self.max_questions = max_questions

        # Initialize models
        self._init_models()

        # Create cache and Query Transformer
        cache_file_path = os.path.join(project_root, "data/pre_retrieval_cache.json")
        self.cache_manager = LocalCacheManager(cache_path=cache_file_path)
        self.query_tf = QueryTransformer(cache_manager=self.cache_manager)

        # Initialize index manager and retriever
        self.index_manager = IndexManager()
        chroma_path = os.path.join(project_root, "data/processed/embeddings")
        print("=== Connecting and Initializing ChromaDB ===")

        if self.index_manager.init_chroma(chroma_path):
            print("=== Loading local index overlays from ChromaDB ===")
            questions_pool = self._load_questions_from_jsonl()
            questions_list = []
            for q_id, q_data in questions_pool.items():
                questions_list.append({
                    "id": q_id,
                    "question": q_data.get("question"),
                    "ground_truth_answer": q_data.get("ground_truth_answer")
                })
            collection_map = {
                "fixed_512": "c_512",
                "fixed_1024": "c_1024",
                "recursive": "c_rec",
                "semantic": "c_sem",
            }

            for strategy, collection in collection_map.items():
                self.index_manager.load_indices_from_chroma(
                    strategy_name=strategy,
                    collection_name=collection,
                    dataset=questions_list
                )
        else:
            print(f"=== Warning: Unable to initialize ChromaDB at {chroma_path}. Please check data directory. ===")

        self.retriever = ModularRetriever(index_manager=self.index_manager)

        # Initialize post processor
        self.post_proc = PostProcessor()

    def _init_models(self):
        """Initialize local LLM models and Embeddings."""
        print("--- Initializing Local Qwen 2.5:1.5b as answer generator ---")
        self.qwen_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)

        print("--- Initializing Local Qwen 3:8b as Judge model for Ragas ---")

        self.eval_llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
            timeout=60
        )       

        """IMPORTANT: Use the same embedding model as ChromaDB to ensure consistency in vector space."""
        self.emb_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _load_questions_from_jsonl(self) -> dict:
        """Read JSONL file and convert to Dictionary to lookup by question_id."""
        questions_dict = {}
        if not os.path.exists(self.jsonl_path):
            print(f"Error: File not found at {self.jsonl_path}")
            return questions_dict

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    q_id = data.get("id")
                    if q_id:
                        questions_dict[q_id] = {
                            "question": data.get("question"),
                            "ground_truth_answer": data.get("ground_truth_answer")
                        }
                        if self.max_questions is not None and len(questions_dict) >= self.max_questions:
                            break
        return questions_dict

    def _get_top_5_pipelines_with_configs(self) -> list:
        """
        Get the top 5 optimal pipelines along with their full configuration parameters
        from stage1_screening_logs.csv.
        """
        df = pd.read_csv(self.metrics_log_file)
        df['combination_score'] = (df['hit_rate_at_5'] + df['recall_at_5'] + df['mrr_at_5'] + df['ndcg_at_5']) / 4
        
        # Group by configuration columns to calculate average score
        config_cols = ['pipeline_id', 'pre_retrieval', 'retrieval', 'chunking', 'index_structure', 'post_retrieval']
        grouped = df.groupby(config_cols)['combination_score'].mean().reset_index()
        
        # Sort and get top 5 configurations
        top_5_df = grouped.sort_values(by='combination_score', ascending=False).head(5)
        return top_5_df.to_dict(orient='records')

    def run_rag_pipeline(self, q_id: str, question_text: str, config: dict) -> tuple:
        """
        MAIN METHOD: Extract context based on auto-extracted configuration from log.
        """
        # 1. Pre-processing: Query Transformation
        queries = self.query_tf.execute_transform(q_id, question_text, config["pre_retrieval"])
        
        # 2. Retrieval
        if config["retrieval"] == "dense_cosine":
            retrieved = self.retriever.search_dense(queries, config["chunking"], config["index_structure"], 5)
        elif config["retrieval"] == "sparse_bm25":
            retriever_res = self.retriever.search_sparse(queries, config["chunking"], config["index_structure"], 5)
            retrieved = retriever_res if retriever_res is not None else []
        elif config["retrieval"] == "hybrid_rrf":
            dense_res = self.retriever.search_dense(queries, config["chunking"], config["index_structure"], 5)
            sparse_res = self.retriever.search_sparse(queries, config["chunking"], config["index_structure"], 5)
            dense_res = dense_res if dense_res is not None else []
            sparse_res = sparse_res if sparse_res is not None else []
            retrieved = self.retriever.fuse_hybrid_rrf(dense_res, sparse_res)[:5]
        else:
            retrieved = []
            
        # 3. Post-Retrieval
        if config["post_retrieval"] == "cross_encoder_rerank":
            processed = self.post_proc.rerank_cross_encoder(question_text, retrieved, top_n=5)
        elif config["post_retrieval"] == "contextual_compression":
            processed = self.post_proc.compress_contextual_noise(question_text, retrieved, threshold=0.15)[:5]
        else:
            processed = retrieved[:5]
            
        # 4. Extract context texts for answer generation
        context_texts = [res["metadata"]["text"] for res in processed if "metadata" in res and "text" in res["metadata"]]
        if not context_texts:
            context_texts = ["Cannot find relevant context."]

        return context_texts


    def judge_llm(self, prompt: str):
        """
        Invoke the judge LLM (Qwen 3:8b) for evaluation.
        Used in Ragas evaluation.
        """
        response = self.eval_llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content

        return str(response)

    def generator_llm(self,prompt: str) -> str:
        response = self.qwen_llm.invoke(prompt)

        if hasattr(response, "content"):
            return response.content

        return str(response)

    def run_evaluation(self):
        """Main evaluation process."""
        try:
            # Automatically get top 5 combinations with configs from the CSV file
            top_5_pipelines = self._get_top_5_pipelines_with_configs()
            print(f"Top 5 Combination selected: {[p['pipeline_id'] for p in top_5_pipelines]}")
        except Exception as e:
            print(f"Error reading log file at {self.metrics_log_file}: {e}")
            return

        questions_pool = self._load_questions_from_jsonl()
        if not questions_pool:
            print(f"Error: No questions found in {self.jsonl_path}. Exiting..")
            return

        if os.path.exists(self.result_log_file):
            df_checkpoint = pd.read_csv(self.result_log_file)
        else:
            df_checkpoint = pd.DataFrame(columns=[
                "pipeline_id", "question_id", "question", "contexts", "answer", "ground_truth", 
                "faithfulness", "answer_relevance", "answer_correctness"
            ])
            df_checkpoint.to_csv(self.result_log_file, index=False)

        # Initialize generator model for Ragas using custom class
        self.generative_evaluator = Stage2GenerativeEvaluator(
            top_5_configs=top_5_pipelines,
            judge_llm=self.judge_llm
        )

        # Iterate through each pipelines in top 5
        for config in top_5_pipelines:
            pipeline_id = config['pipeline_id']
            print(f"\n>>> RUNNING EVALUATION FOR PIPELINE: {pipeline_id}", flush= True)
            print(f"    [Config] Pre: {config['pre_retrieval']} | Retrieval: {config['retrieval']} | Chunking: {config['chunking']} | Post: {config['post_retrieval']}", flush=True)

            for q_id, question_data in questions_pool.items():
                is_evaluated = not df_checkpoint[
                    (df_checkpoint['pipeline_id'] == pipeline_id) & 
                    (df_checkpoint['question_id'] == q_id)
                ].empty

                if is_evaluated:
                    continue

                question_text = question_data["question"]
                ground_truth_answer = question_data["ground_truth_answer"]
                
                print(f" -> Processing Question ID: {q_id}", flush=True)

                while True:
                    try:
                        contexts = self.run_rag_pipeline(q_id, question_text, config)

                        answer = self.generative_evaluator.compile_rag_response(
                            question=question_text,
                            contexts=contexts,
                            generator_llm=self.generator_llm
                        )

                        sample_data = {
                            "question": [question_text],
                            "contexts": [contexts],
                            "answer": [answer],
                            "ground_truth": [ground_truth_answer]
                        }

                        score = self.generative_evaluator.execute_ragas_audit(
                            sample_data
                        )

                        f_score = score["faithfulness"]
                        ar_score = score["answer_relevancy"]
                        ac_score = score["answer_correctness"]

                        new_row = pd.DataFrame([{
                            "pipeline_id": pipeline_id,
                            "question_id": q_id,
                            "question": question_text,
                            "contexts": json.dumps(contexts, ensure_ascii=False),
                            "answer": answer,
                            "ground_truth": ground_truth_answer,
                            "faithfulness": f_score,
                            "answer_relevance": ar_score,
                            "answer_correctness": ac_score
                        }])
                        
                        new_row.to_csv(self.result_log_file, mode='a', header=False, index=False)
                        print(f"    => Success! F: {f_score:.2f} | AR: {ar_score:.2f} | AC: {ac_score:.2f}")
                        
                        time.sleep(self.delay_requests)
                        break
                        
                    except Exception as error:
                        print(f" [Connection Error / Stuck]: {error}")
                        print("Restarting in 30 seconds...")
                        time.sleep(30)

def main():
    print("=========================================================")
    print("     EXECUTING RAG EVALUATION (RAGAS + LOCAL QWEN)       ")
    print("=========================================================")

    # Declare paths for input and output files
    INPUT_QUESTIONS_JSONL = "data/processed/questions/questions.jsonl"
    STAGE1_SCREENING_LOGS = "reports/stage1_screening_logs.csv"
    FINAL_RAGAS_REPORT    = "reports/ragas_evaluation_checkpoint_local_quick_test.csv"

    # Initialize the evaluator with said paths
    evaluator = RagEvaluator(
        jsonl_path=INPUT_QUESTIONS_JSONL,
        metrics_log_file=STAGE1_SCREENING_LOGS,
        result_log_file=FINAL_RAGAS_REPORT,
        delay_requests=0 # 0 for local
    )

    # Execute the evalation process
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()