import os, sys
import time
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate, RunConfig
from ragas.llms import _LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from core.query_transforms import QueryTransformer
from core.retrievers import ModularRetriever
from core.post_processors import PostProcessor
from core.cache_manager import LocalCacheManager
from core.index_manager import IndexManager

'''
Evaluation with GOOGLE GEMINI
# THƯ VIỆN KẾT NỐI VỚI LOCAL OLLAMA VÀ GOOGLE GEMINI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
'''

'''
Evaluation with GLM + Ollama Embeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
'''

'''
Evaluation with Local Ollama + Ollama Embeddings
'''
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class RagEvaluator:
    def __init__(
        self, 
        jsonl_path: str = "data/processed/questions/questions.jsonl",
        metrics_log_file: str = "reports/stage1_screening_logs.csv",
        result_log_file: str = "reports/ragas_evaluation_checkpoint_local.csv",
        delay_requests: float = 12.0
    ):
        """
        Khởi tạo bộ đánh giá RAG sử dụng Ragas và Google Gemini.
        Tự động nạp cấu hình bí mật từ file .env
        """
        # Tải các biến môi trường từ file .env bí mật của bạn
        load_dotenv()
        
        # Lấy trực tiếp từ file .env và gán vào biến hệ thống mà Ragas/Google yêu cầu
        gemini_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        glm_key = os.getenv("GLM_API_KEY")
        if not gemini_key:
            raise ValueError("Lỗi: Không tìm thấy 'GOOGLE_GEMINI_API_KEY' trong file .env của bạn!")
            
        os.environ["GOOGLE_API_KEY"] = gemini_key
        
        # Cấu hình đường dẫn hệ thống
        self.jsonl_path = jsonl_path
        self.metrics_log_file = metrics_log_file
        self.result_log_file = result_log_file
        self.delay_requests = delay_requests
        
        # Khởi tạo các mô hình và cấu hình Ragas
        self._init_models()
        self._setup_metrics()

        # Khời tạo các phương thức retrieval
        self.cache_manager = LocalCacheManager()
        self.query_tf = QueryTransformer(self.cache_manager)

        self.index_manager = IndexManager()
        self.retriever = ModularRetriever(self.index_manager) 
        self.post_proc = PostProcessor()

    def _init_models(self):
        """Khởi tạo nội bộ các kết nối LLM và Embeddings."""
        print("--- Khởi tạo mô hình Qwen chạy Local ---")
        self.qwen_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)

        print("--- Khởi tạo mô hình Google Gemini cho Ragas ---")
        # self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        # self.gemini_emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        # self.ragas_gemini_llm = _LangchainLLMWrapper(self.gemini_llm)
        # self.ragas_gemini_emb = LangchainEmbeddingsWrapper(self.gemini_emb)

        self.gemini_llm = ChatOllama(
            model="qwen3:8b",
            temperature=0
        )       

        self.gemini_emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.ragas_gemini_llm = _LangchainLLMWrapper(self.gemini_llm)
        self.ragas_gemini_emb = LangchainEmbeddingsWrapper(self.gemini_emb)

    def _setup_metrics(self):
        """Cấu hình các metric."""
        self.faithfulness = Faithfulness()
        self.faithfulness.llm = self.ragas_gemini_llm

        self.answer_relevancy = AnswerRelevancy()
        self.answer_relevancy.llm = self.ragas_gemini_llm
        self.answer_relevancy.embeddings = self.ragas_gemini_emb

        self.answer_correctness = AnswerCorrectness(
            llm=self.ragas_gemini_llm,
            embeddings=self.ragas_gemini_emb
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

        df_metrics = pd.read_csv(self.metrics_log_file)
        unique_question_ids = df_metrics['question_id'].unique().tolist()

        # Duyệt qua từng cấu hình pipeline trong nhóm Top 5
        for config in top_5_pipelines:
            pipeline_id = config['pipeline_id']
            print(f"\n>>> ĐANG CHẠY EVALUATION CHO PIPELINE: {pipeline_id}")
            print(f"    [Cấu hình] Pre: {config['pre_retrieval']} | Retrieval: {config['retrieval']} | Chunking: {config['chunking']} | Post: {config['post_retrieval']}")
            
            for q_id in unique_question_ids:
                is_evaluated = not df_checkpoint[
                    (df_checkpoint['pipeline_id'] == pipeline_id) & 
                    (df_checkpoint['question_id'] == q_id)
                ].empty
                
                if is_evaluated:
                    continue
                    
                question_data = questions_pool.get(q_id)
                if not question_data:
                    print(f" -> Cảnh báo: Không tìm thấy ID {q_id} trong file JSONL.")
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
                            llm=self.ragas_gemini_llm,
                            embeddings=self.ragas_gemini_emb,
                            run_config=config_ragas
                            )
                        
                        df = score.to_pandas()
                        f_score = float(df["faithfulness"].iloc[0]) if df["faithfulness"].iloc[0] != 'nan' else 0.0
                        ar_score = float(df["answer_relevancy"].iloc[0]) if df["answer_relevancy"].iloc[0] != 'nan' else 0.0
                        ac_score = float(df["answer_correctness"].iloc[0]) if df["answer_correctness"].iloc[0] != 'nan' else 0.0

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


if __name__ == "__main__":
    evaluator = RagEvaluator(
        jsonl_path="data/processed/questions/questions.jsonl",
        metrics_log_file="reports/stage1_screening_logs.csv",
        result_log_file="reports/ragas_evaluation_checkpoint_local.csv",
        delay_requests=12.0
    )

    evaluator.run_evaluation()