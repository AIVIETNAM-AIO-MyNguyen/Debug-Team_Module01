import os, sys, math
import time
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate, RunConfig
from ragas.llms import _LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness


"""
IMPORTANT: TO AVOID RAGAS CRASHHING DUE TO MISSING MODULES,
WE CREATE MOCK MODULES FOR IT. ONLY RUN IF YOU CANNOT SETUP RAGAS PROPERLY.
"""
'''
import sys
from types import ModuleType

mock_vertex_module = ModuleType("langchain_community.chat_models.vertexai")
mock_vertex_module.ChatVertexAI = None  # Gán bằng None để Ragas không bị crash khi import

mock_llms_module = ModuleType("langchain_community.llms")
mock_llms_module.VertexAI = None  # Gán bằng None dự phòng dòng tiếp theo của Ragas

sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_module
sys.modules["langchain_community.llms"] = mock_llms_module
'''

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


class RagEvaluator:
    def __init__(
        self, 
        jsonl_path: str = "data/processed/questions/test_questions.jsonl",
        metrics_log_file: str = "reports/stage1_screening_logs.csv",
        result_log_file: str = "reports/ragas_evaluation_checkpoint_local.csv",
        delay_requests: float = 0 # 0 for local
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

        # Initialize ragas models and metrics
        self._init_models()
        self._setup_metrics()

        # Create cache and Query Transformer
        cache_file_path = os.path.join(project_root, "data/pre_retrieval_cache.json")
        self.cache_manager = LocalCacheManager(cache_path=cache_file_path)
        self.query_tf = QueryTransformer(cache_manager=self.cache_manager)

        # Initialize index manager and retriever
        self.index_manager = IndexManager()
        chroma_path = os.path.join(project_root, "data/processed/embeddings")
        print("=== Đang kết nối và khởi tạo cấu trúc ChromaDB ===")
        success = self.index_manager.init_chroma(path=chroma_path)
        if not success:
            print(f"=== Cảnh báo: Không thể khởi tạo ChromaDB tại {chroma_path}. Kiểm tra lại thư mục dữ liệu. ===")

        self.retriever = ModularRetriever(index_manager=self.index_manager)

        # Initialize post processor
        self.post_proc = PostProcessor()

    def _init_models(self):
        """Khởi tạo nội bộ các kết nối LLM và Embeddings."""
        print("--- Khởi tạo mô hình Qwen chạy Local ---")
        self.qwen_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)

        print("--- Khởi tạo mô hình qwen3:8b cho Ragas ---")

        self.eval_llm = ChatOllama(
            model="qwen3:8b",
            temperature=0
        )       

        self.emb_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.ragas_eval_llm = _LangchainLLMWrapper(self.eval_llm)
        self.ragas_eval_emb = LangchainEmbeddingsWrapper(self.emb_model)

    def _setup_metrics(self):
        """Cấu hình các metric."""
        self.faithfulness = Faithfulness()
        self.faithfulness.llm = self.ragas_eval_llm

        self.answer_relevancy = AnswerRelevancy()
        self.answer_relevancy.llm = self.ragas_eval_llm
        self.answer_relevancy.embeddings = self.ragas_eval_emb

        self.answer_correctness = AnswerCorrectness(
            llm=self.ragas_eval_llm,
            embeddings=self.ragas_eval_emb
        )

        self.metrics = [
            self.faithfulness,
            self.answer_relevancy,
            self.answer_correctness
        ]

    def _load_questions_from_jsonl(self) -> dict:
        """Đọc file JSONL thành Dictionary để tra cứu bằng question_id."""
        questions_dict = {}
        if not os.path.exists(self.jsonl_path):
            print(f"Lỗi: Không tìm thấy file JSONL tại {self.jsonl_path}")
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
        return questions_dict

    def _get_top_5_pipelines_with_configs(self) -> list:
        """
        Lấy top 5 pipeline tối ưu nhất kèm theo toàn bộ tham số cấu hình của chúng
        từ file log sàng lọc (screening log).
        """
        df = pd.read_csv(self.metrics_log_file)
        df['combination_score'] = (df['hit_rate_at_5'] + df['recall_at_5'] + df['mrr_at_5'] + df['ndcg_at_5']) / 4
        
        # Nhóm theo các cột cấu hình cấu trúc hệ thống để tính điểm trung bình
        config_cols = ['pipeline_id', 'pre_retrieval', 'retrieval', 'chunking', 'index_structure', 'post_retrieval']
        grouped = df.groupby(config_cols)['combination_score'].mean().reset_index()
        
        # Sắp xếp lấy Top 5 và chuyển đổi thành danh sách từ điển (list of dicts)
        top_5_df = grouped.sort_values(by='combination_score', ascending=False).head(5)
        return top_5_df.to_dict(orient='records')

    def run_rag_pipeline(self, q_id: str, question_text: str, config: dict) -> tuple:
        """
        PHƯƠNG THỨC CHÍNH THỨC: Trích xuất ngữ cảnh dựa trên cấu hình tự động trích xuất từ log.
        """
        # 1. Tiền xử lý Query (Query Transformation)
        queries = self.query_tf.execute_transform(q_id, question_text, config["pre_retrieval"])
        
        # 2. Định tuyến chiến lược Tìm kiếm dữ liệu (Retrieval)
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
            
        # 3. Hậu xử lý kết quả (Post-Retrieval)
        if config["post_retrieval"] == "cross_encoder_rerank":
            processed = self.post_proc.rerank_cross_encoder(question_text, retrieved, top_n=5)
        elif config["post_retrieval"] == "contextual_compression":
            processed = self.post_proc.compress_contextual_noise(question_text, retrieved, threshold=0.15)[:5]
        else:
            processed = retrieved[:5]
            
        # 4. Trích xuất nội dung văn bản làm ngữ cảnh nền
        context_texts = [res["metadata"]["text"] for res in processed if "metadata" in res and "text" in res["metadata"]]
        if not context_texts:
            context_texts = ["Không tìm thấy ngữ cảnh phù hợp cho câu hỏi này."]

        # 5. Sinh câu trả lời bằng Qwen Local
        context_str = "\n".join(context_texts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question_text}\n\nAnswer:"

        # prompt = prompt = f"""
        #     You are a retrieval-grounded assistant.

        #     Answer ONLY using the provided Context.

        #     Rules:
        #     - Every claim must be supported by Context.
        #     - Do not use external knowledge.
        #     - Do not speculate.
        #     - If Context is insufficient, say so.
        #     - Be concise.
        #     - Answer the Question directly.

        #     Context:
        #     {context_str}

        #     Question:
        #     {question_text}

        #     Answer:
        #     """
        
        response = self.qwen_llm.invoke(prompt)
        generated_answer = response.content
        
        return generated_answer, context_texts

    def run_evaluation(self):
        """Tiến trình chạy đánh giá chính."""
        try:
            # Tự động lấy danh sách Top 5 kèm theo config đi kèm trong file CSV sàng lọc
            top_5_pipelines = self._get_top_5_pipelines_with_configs()
            print(f"Top 5 Combination được chọn: {[p['pipeline_id'] for p in top_5_pipelines]}")
        except Exception as e:
            print(f"Lỗi đọc file log tại {self.metrics_log_file}: {e}")
            return

        questions_pool = self._load_questions_from_jsonl()
        if not questions_pool:
            print(f"Không có câu hỏi nào được tải từ {self.jsonl_path}. Kết thúc.")
            return

        if os.path.exists(self.result_log_file):
            df_checkpoint = pd.read_csv(self.result_log_file)
        else:
            df_checkpoint = pd.DataFrame(columns=[
                "pipeline_id", "question_id", "question", "contexts", "answer", "ground_truth", 
                "faithfulness", "answer_relevance", "answer_correctness"
            ])
            df_checkpoint.to_csv(self.result_log_file, index=False)


        # Duyệt qua từng cấu hình pipeline trong nhóm Top 5
        for config in top_5_pipelines:
            pipeline_id = config['pipeline_id']
            print(f"\n>>> ĐANG CHẠY EVALUATION CHO PIPELINE: {pipeline_id}")
            print(f"    [Cấu hình] Pre: {config['pre_retrieval']} | Retrieval: {config['retrieval']} | Chunking: {config['chunking']} | Post: {config['post_retrieval']}")

            for q_id, question_data in questions_pool.items():
                is_evaluated = not df_checkpoint[
                    (df_checkpoint['pipeline_id'] == pipeline_id) & 
                    (df_checkpoint['question_id'] == q_id)
                ].empty

                if is_evaluated:
                    continue

                question_text = question_data["question"]
                ground_truth_answer = question_data["ground_truth_answer"]
                
                print(f" -> Đang xử lý Câu hỏi ID: {q_id}")
                
                while True:
                    try:
                        # Truyền trực tiếp từ điển 'config' chứa đầy đủ thông tin hàng vào hàm pipeline
                        answer, contexts = self.run_rag_pipeline(q_id, question_text, config)
                        
                        sample_data = {
                            "question": [question_text],
                            "contexts": [contexts],
                            "answer": [answer],
                            "ground_truth": [ground_truth_answer]
                        }
                        dataset = Dataset.from_dict(sample_data)
                        
                        config_ragas = RunConfig(max_workers=1, timeout=180)
                        score = evaluate(
                            dataset,
                            metrics=self.metrics,
                            llm=self.ragas_eval_llm,
                            embeddings=self.ragas_eval_emb,
                            run_config=config_ragas
                            )

                        df = score.to_pandas()
                        f_score = float(df["faithfulness"].iloc[0]) if not math.isnan(df["faithfulness"].iloc[0]) else 0.0
                        ar_score = float(df["answer_relevancy"].iloc[0]) if not math.isnan(df["answer_relevancy"].iloc[0]) else 0.0
                        ac_score = float(df["answer_correctness"].iloc[0]) if not math.isnan(df["answer_correctness"].iloc[0]) else 0.0

                        # # Nếu tất cả các metric chính đều bị Ragas gán nhãn `nan` (do sập API ẩn bên trong)
                        # if pd.isna(f_score) or pd.isna(ar_score) or pd.isna(ac_score):
                        #     raise Exception("ragas_all_nan_error: Toàn bộ điểm số trả về bị NAN do nghẽn token hệ thống.")

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
                        print(f"    => Thành công! F: {f_score:.2f} | AR: {ar_score:.2f} | AC: {ac_score:.2f}")
                        
                        time.sleep(self.delay_requests)
                        break
                        
                    except Exception as error:
                        error_msg = str(error).lower()
                        
                        if "quota" in error_msg and "day" in error_msg:
                            print("\n[CẢNH BÁO NGHẼN HỆ THỐNG]: Bạn đã dùng hết 1,500 requests ngày của tài khoản Google Free!")
                            print("Hệ thống tiến hành kích hoạt chế độ NGỦ ĐÔNG TRONG 24 GIỜ để chờ reset quota...")
                            print("⚠️ Xin vui lòng KHÔNG tắt chương trình. Tiến trình sẽ tự động chạy tiếp vào ngày mai.")
                            time.sleep(86400)
                            print("--- Hệ thống đã thức dậy! Đang thử lại câu hỏi vừa rồi... ---")
                        else:
                            print(f" [Lỗi kết nối / Nghẽn phút]: {error}")
                            print("Tạm nghỉ 60 giây để hệ thống hồi phục...")
                            time.sleep(60)

def main():
    print("=========================================================")
    print("     EXECUTING RAG EVALUATION (RAGAS + LOCAL QWEN)       ")
    print("=========================================================")

    # Declare paths for input and output files
    INPUT_QUESTIONS_JSONL = "data/processed/questions/test_questions.jsonl"
    STAGE1_SCREENING_LOGS = "reports/stage1_screening_logs.csv"
    FINAL_RAGAS_REPORT    = "reports/ragas_evaluation_checkpoint_local.csv"

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