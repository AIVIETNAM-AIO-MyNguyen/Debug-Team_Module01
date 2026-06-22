# =====================================================================
# GỈA LẬP MODULE ĐỂ KHÔNG BỊ LỖI RAGAS
# =====================================================================
import sys
from types import ModuleType

# 1. Tự tạo cấu trúc module giả lập trong bộ nhớ Python độc lập
mock_vertex_module = ModuleType("langchain_community.chat_models.vertexai")
mock_vertex_module.ChatVertexAI = None  # Gán bằng None để Ragas không bị crash khi import

mock_llms_module = ModuleType("langchain_community.llms")
mock_llms_module.VertexAI = None  # Gán bằng None dự phòng dòng tiếp theo của Ragas

# 2. Ép hệ thống đăng ký các đường dẫn giả này vào danh sách quản lý module toàn cục
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex_module
sys.modules["langchain_community.llms"] = mock_llms_module
# =====================================================================


# =====================================================================
# KIỂM TRA VÀ TẢI GÓI PUNKT_TAB ĐỂ TRÁNH LỖI HUGGINGFACE
# =====================================================================
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    nltk = None
# =====================================================================


import os
import time
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate, RunConfig
from langchain_core.outputs import LLMResult, Generation
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, AnswerCorrectness

# THƯ VIỆN KẾT NỐI VỚI LOCAL OLLAMA, GROQ VÀ HUGGINGFACEEMBEDDINGS
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# =====================================================================
# LẤY ĐƯỜNG DẪN TUYỆT ĐỐI CỦA 'src' ĐỂ IMPORT PACKAGE 'core'
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # Đường dẫn đến src/evaluation/
src_dir = os.path.dirname(current_dir)  # Đường dẫn đến src/

# Chỉ định Python nhìn vào bên trong thư mục 'src' này để tìm các module con
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# =====================================================================


from core.cache_manager import LocalCacheManager
from core.index_manager import IndexManager
from core.query_transforms import QueryTransformer
from core.retrievers import ModularRetriever
from core.post_processors import PostProcessor

class RagEvaluator:
    def __init__(
        self, 
        jsonl_path: str = "data/processed/questions/questions.jsonl",
        metrics_log_file: str = "reports/stage1_screening_logs.csv",
        result_log_file: str = "reports/ragas_evaluation_checkpoint.csv",
        delay_requests: float = 12.0
    ):
        """Khởi tạo bộ đánh giá RAG sử dụng Ragas, Groq và HuggingfaceEmbedding."""

        # Tải các biến môi trường từ file .env bí mật của bạn
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("=== Lỗi: Không tìm thấy 'GROQ_API_KEY' trong file .env của bạn! ===")
        
        # Tìm đường dẫn gốc của dự án
        project_root = os.path.dirname(src_dir)  # Thư mục gốc project

        # Cấu hình đường dẫn và delay
        self.jsonl_path = os.path.join(project_root, jsonl_path)
        self.metrics_log_file = os.path.join(project_root, metrics_log_file)
        self.result_log_file = os.path.join(project_root, result_log_file)
        self.delay_requests = delay_requests
        
        # Khởi tạo các mô hình và cấu hình Ragas
        self._init_models()
        self._setup_metrics()

        # Tạo cache và QueryTransformer
        cache_file_path = os.path.join(project_root, "data/pre_retrieval_cache.json")
        self.cache_manager = LocalCacheManager(cache_path=cache_file_path)
        self.query_tf = QueryTransformer(cache_manager=self.cache_manager)

        # Tạo index_manager và ModularRetriever
        self.index_manager = IndexManager()
        chroma_path = os.path.join(project_root, "data/processed/embeddings")
        print("=== Đang kết nối và khởi tạo cấu trúc ChromaDB ===")
        success = self.index_manager.init_chroma(path=chroma_path)
        if not success:
            print(f"=== Cảnh báo: Không thể khởi tạo ChromaDB tại {chroma_path}. Kiểm tra lại thư mục dữ liệu. ===")
        self.retriever = ModularRetriever(index_manager=self.index_manager)

        # Tạo PostProcessor
        self.post_proc = PostProcessor()

    def _init_models(self):
        """Khởi tạo nội bộ các kết nối LLM và Embeddings."""

        print("=== Khởi tạo mô hình Qwen chạy Local ===")
        self.qwen_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)
        
        print("=== Khởi tạo siêu mô hình Llama-3.3-70B trên Groq Cloud qua LangChain ===")
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=groq_key, 
            temperature=0
        )

        print("=== Khởi tạo mô hình Embedding Local từ HuggingFace ===")
        # Sử dụng mô hình nhúng local để không tốn hạn ngạch mạng của Groq
        self.local_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Tự tạo Wrapper cho Groq, vì không được support sẵn
        class CustomGroqRagasLLM(BaseRagasLLM):
            """Lớp bọc chuẩn hóa kết nối LangChain LLM vào Ragas, ép tắt strict=False và hỗ trợ Async."""
            def __init__(self, langchain_llm):
                super().__init__()
                self.langchain_llm = langchain_llm
                
            def generate_text(self, prompt, n=1, temperature=0, stop=None, callbacks=None):
                """Xử lý sinh văn bản đồng bộ."""
                response = self.langchain_llm.invoke(prompt.to_string(), config={"callbacks": callbacks})
                return LLMResult(generations=[[Generation(text=response.content)]])
                
            async def agenerate_text(self, prompt, n=1, temperature=0, stop=None, callbacks=None):
                """Xử lý sinh văn bản bất đồng bộ."""
                response = await self.langchain_llm.ainvoke(prompt.to_string(), config={"callbacks": callbacks})
                return LLMResult(generations=[[Generation(text=response.content)]])
                
            def is_finished(self, response: LLMResult) -> bool:
                """Kiểm tra trạng thái kết thúc phản hồi."""
                return True

            def with_structured_output(self, schema, **kwargs):
                """Can thiệp tầng ép cấu hình JSON để tắt cờ strict=True của Ragas sang strict=False."""
                kwargs["strict"] = False
                structured_llm = self.langchain_llm.with_structured_output(schema, **kwargs)
                
                # Tạo một wrapper nhỏ để đồng bộ định dạng kết quả trả về cho Ragas
                class StructuredOutputWrapper:
                    def __init__(self, llm):
                        self.llm = llm
                    def invoke(self, prompt, callbacks=None):
                        # Chuyển đối tựng prompt sang kiểu chuỗi nếu cần
                        p_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
                        return self.llm.invoke(p_str, config={"callbacks": callbacks})
                    async def ainvoke(self, prompt, callbacks=None):
                        p_str = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
                        return await self.llm.ainvoke(p_str, config={"callbacks": callbacks})
                        
                return StructuredOutputWrapper(structured_llm)

        self.ragas_llm = CustomGroqRagasLLM(self.groq_llm)
        self.ragas_emb = LangchainEmbeddingsWrapper(self.local_emb)

    def _setup_metrics(self):
        """Cấu hình các metric gán cho Gemini."""
        self.answer_correctness = AnswerCorrectness()
        self.metrics = [faithfulness, answer_relevancy, self.answer_correctness]

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
        Trích xuất ngữ cảnh dựa trên cấu hình tự động trích xuất từ stage1_screening_logs.csv,
        và tạo câu trả lời.
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
            print(f"=== Top 5 Combination được chọn: {[p['pipeline_id'] for p in top_5_pipelines]} ===")
        except Exception as e:
            print(f"=== Lỗi đọc file log tại {self.metrics_log_file}: {e} ===")
            return

        # Lấy bộ câu hỏi để test ở stage 2
        questions_pool = self._load_questions_from_jsonl()
        if not questions_pool:
            print(f"=== Không có câu hỏi nào được tải từ {self.jsonl_path}. Kết thúc. ===")
            return

        # Lấy file checkpoint
        if os.path.exists(self.result_log_file):
            df_checkpoint = pd.read_csv(self.result_log_file)
        else:
            df_checkpoint = pd.DataFrame(columns=[
                "pipeline_id", "question_id", "question", "contexts", "answer", "ground_truth", 
                "faithfulness", "answer_relevancy", "answer_correctness"
            ])
            df_checkpoint.to_csv(self.result_log_file, index=False)

        # Tìm toàn bộ các câu hỏi được đánh giá ở stage 1
        # df_metrics = pd.read_csv(self.metrics_log_file)
        # unique_question_ids = df_metrics['question_id'].unique().tolist()

        # Duyệt qua từng cấu hình pipeline trong nhóm Top 5
        for config in top_5_pipelines:
            pipeline_id = config['pipeline_id']
            print(f"\n>>> ĐANG CHẠY EVALUATION CHO PIPELINE: {pipeline_id} <<<")
            print(f">>> [Cấu hình] Pre: {config['pre_retrieval']} | Retrieval: {config['retrieval']} | Chunking: {config['chunking']} | Post: {config['post_retrieval']} <<<")
            
            # Duyệt qua toàn bộ các câu hỏi trong stage 1
            for q_id, question_data in questions_pool.items():
                is_evaluated = not df_checkpoint[
                    (df_checkpoint['pipeline_id'] == pipeline_id) & 
                    (df_checkpoint['question_id'] == q_id)
                ].empty
                
                if is_evaluated:
                    continue
                    
                # Kiểm tra xem câu hỏi đó có trong bộ câu hỏi stage 2 không
                # question_data = questions_pool.get(q_id)
                # if not question_data:
                #     continue
    
                question_text = question_data["question"]
                ground_truth_answer = question_data["ground_truth_answer"]
                
                print(f"--- Đang xử lý Câu hỏi ID: {q_id} ---")
                
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
                        
                        config_ragas = RunConfig(max_workers=1, max_retries=2, max_wait=10, timeout=60)
                        score = evaluate(
                            dataset, 
                            metrics=self.metrics, 
                            llm=self.ragas_llm,
                            embeddings=self.ragas_emb,
                            run_config=config_ragas
                        )
                        
                        f_score = score.scores[0].get("faithfulness", 0.0) if score.scores else 0.0
                        ar_score = score.scores[0].get("answer_relevancy", 0.0) if score.scores else 0.0
                        ac_score = score.scores[0].get("answer_correctness", 0.0) if score.scores else 0.0
                        
                        # Nếu tất cả các metric chính đều bị Ragas gán nhãn `nan` (do sập API ẩn bên trong)
                        if pd.isna(f_score) or pd.isna(ar_score) or pd.isna(ac_score):
                            raise Exception("ragas_all_nan_error: Toàn bộ điểm số trả về bị NAN do nghẽn token hệ thống.")

                        new_row = pd.DataFrame([{
                            "pipeline_id": pipeline_id,
                            "question_id": q_id,
                            "question": question_text,
                            "contexts": json.dumps(contexts, ensure_ascii=False),
                            "answer": answer,
                            "ground_truth": ground_truth_answer,
                            "faithfulness": f_score,
                            "answer_relevancy": ar_score,
                            "answer_correctness": ac_score
                        }])
                        
                        new_row.to_csv(self.result_log_file, mode='a', header=False, index=False)
                        print(f"--> Thành công! F: {f_score:.2f} | AR: {ar_score:.2f} | AC: {ac_score:.2f} <--")
                        
                        time.sleep(self.delay_requests)
                        break
                        
                    except Exception as error:
                        error_msg = str(error).lower()
                        
                        if "tokens per day" in error_msg or "tpd" in error_msg or "rate_limit_exceeded" in error_msg:
                            print("\n=== [CẢNH BÁO NGHẼN HỆ THỐNG]: Bạn đã dùng hết requests ngày của tài khoản Groq Free!")
                            print("=== Hệ thống tiến hành kích hoạt chế độ NGỦ ĐÔNG TRONG 24 GIỜ để chờ reset quota... ===")
                            print("=== ⚠️ Xin vui lòng KHÔNG tắt chương trình. Tiến trình sẽ tự động chạy tiếp vào ngày mai. ===")
                            time.sleep(86400)
                            print("=== Hệ thống đã thức dậy! Đang thử lại câu hỏi vừa rồi... ===")
                        else:
                            print(f"=== [Lỗi hệ thống / Ragas trả về NAN]: {error} ===")
                            print("=== Tạm nghỉ 60 giây để làm sạch hàng đợi trước khi thử lại... ===")
                            time.sleep(60)

def main():
    print("=========================================================")
    print("   BẮT ĐẦU HỆ THỐNG ĐÁNH GIÁ RAG (RAGAS + LOCAL QWEN)    ")
    print("=========================================================")

    # 1. Cấu hình các đường dẫn tương ứng với cấu trúc dự án của bạn
    INPUT_QUESTIONS_JSONL = "data/processed/questions/test_questions.jsonl"
    STAGE1_SCREENING_LOGS = "reports/stage1_screening_logs.csv"
    FINAL_RAGAS_REPORT    = "reports/ragas_evaluation_checkpoint_test.csv"

    # 2. Khởi tạo đối tượng đánh giá
    evaluator = RagEvaluator(
        jsonl_path=INPUT_QUESTIONS_JSONL,
        metrics_log_file=STAGE1_SCREENING_LOGS,
        result_log_file=FINAL_RAGAS_REPORT,
        delay_requests=10.0)
    
    # 3. Kích hoạt tiến trình chạy tự động
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
