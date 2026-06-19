import os
import time
import json
import pandas as pd
from datasets import Dataset

# Import các thư viện ChromaDB và Ragas
import chromadb
# from google import genai
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevance, AnswerCorrectness

# THƯ VIỆN ĐỂ KẾT NỐI VỚI OLLAMA LOCAL
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ==========================================
# 1. CẤU HÌNH BIẾN MÔI TRƯỜNG & THAM SỐ
# ==========================================
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_FREE_API_KEY_HERE"

METRICS_LOG_FILE = "pipeline_metrics_log.csv"
CHROMA_DB_PATH = "./chroma_db_store"
RESULT_LOG_FILE = "ragas_evaluation_checkpoint.csv"

# Khoảng nghỉ RPM cho Gemini (Giảm xuống 5.5s vì giờ Gemini chỉ gánh 2 metrics)
DELAY_BETWEEN_REQUESTS = 5.5 

# ==========================================
# 2. KHỞI TẠO OLLAMA LOCAL (QWEN & LLAMA)
# ==========================================
print("--- Khởi tạo các mô hình Ollama chạy Local ---")
# Khởi tạo mô hình Qwen local của bạn (Đảm bảo bạn đã chạy 'ollama run qwen2.5:1.5b')
# Lưu ý: Thay đúng tên tag mô hình Qwen bạn đang cài trong máy
qwen_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)

# Khởi tạo mô hình Llama local dùng để chấm điểm Answer Relevancy
llama_evaluator = ChatOllama(model="llama3:8b", temperature=0) # Thay bằng tên model của bạn (ví dụ: llama3:6b)
ragas_llama_llm = LangchainLLMWrapper(llama_evaluator)

# ==========================================
# 3. KHỞI TẠO GOOGLE GEMINI (ĐÁNH GIÁ LAI)
# ==========================================
print("--- Khởi tạo mô hình Google Gemini cho Ragas ---")
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
gemini_emb = GoogleGenerativeAIEmbeddings(model="text-embedding-04s")

ragas_gemini_llm = LangchainLLMWrapper(gemini_llm)
ragas_gemini_emb = LangchainEmbeddingsWrapper(gemini_emb)

# ==========================================
# 4. PHÂN CHIA NHIỆM VỤ CHO CÁC METRICS (HYBRID)
# ==========================================
# Metric 1: Faithfulness -> Giao cho Google Gemini gánh
faithfulness.llm = ragas_gemini_llm

# Metric 2: AnswerCorrectness -> Giao cho Google Gemini gánh
answer_correctness = AnswerCorrectness(llm=ragas_gemini_llm, embeddings=ragas_gemini_emb)

# Metric 3: AnswerRelevancy -> ĐẨY SANG CHO LLAMA LOCAL CHẤM ĐIỂM (Bypass giới hạn Google)
answer_relevance.llm = ragas_llama_llm
answer_relevance.embeddings = ragas_gemini_emb  # Embedding vẫn dùng Gemini để có độ chính xác vector cao

metrics = [faithfulness, answer_relevance, answer_correctness]

# ==========================================
# 5. KẾT NỐI CHROMADB VÀ FILE LOG TOP 5
# ==========================================
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
question_collection = chroma_client.get_or_create_collection(name="questions_collection")

def get_top_5_combinations(csv_path):
    df = pd.read_csv(csv_path)
    df['combination_score'] = (df['hit_rate_at_5'] + df['recall_at_5'] + df['mrr_at_5'] + df['ndcg_at_5']) / 4
    grouped = df.groupby('pipeline_id')['combination_score'].mean().reset_index()
    return grouped.sort_values(by='combination_score', ascending=False).head(5)['pipeline_id'].tolist()

# ==========================================
# 6. PIPELINE CHẠY QWEN TRÊN OLLAMA THỰC TẾ
# ==========================================
def run_qwen_rag_pipeline_ollama(pipeline_id, question_text):
    """
    Hàm gọi mô hình Qwen thực tế thông qua Ollama để sinh câu trả lời.
    """
    # Bước giả lập: Giả sử đây là đoạn code bạn dùng pipeline_id để đi retrieve tài liệu
    # (Bạn tự tích hợp hàm retrieve ứng với pipeline_id của bạn vào đây nhé)
    retrieved_contexts = [
        "Ngữ cảnh mẫu 1 được trích xuất từ database ứng với cấu hình hệ thống.",
        "Ngữ cảnh mẫu 2 đã qua xử lý lọc nhiễu văn bản."
    ]
    
    # Tạo Prompt hoàn chỉnh gửi cho Qwen Local
    context_str = "\n".join(retrieved_contexts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {question_text}\n\nAnswer:"
    
    # Gọi Qwen sinh câu trả lời local
    response = qwen_llm.invoke(prompt)
    generated_answer = response.content
    
    return generated_answer, retrieved_contexts

# ==========================================
# 7. TIẾN TRÌNH CHẠY CHÍNH VÀ XỬ LÝ QUOTA DAY
# ==========================================
def main():
    try:
        top_5_pipelines = get_top_5_combinations(METRICS_LOG_FILE)
        print(f"Top 5 Combination được chọn: {top_5_pipelines}")
    except Exception as e:
        print(f"Lỗi đọc file log: {e}")
        return

    if os.path.exists(RESULT_LOG_FILE):
        df_checkpoint = pd.read_csv(RESULT_LOG_FILE)
    else:
        df_checkpoint = pd.DataFrame(columns=[
            "pipeline_id", "question_id", "question", "contexts", "answer", "ground_truth", 
            "faithfulness", "answer_relevance", "answer_correctness"
        ])
        df_checkpoint.to_csv(RESULT_LOG_FILE, index=False)

    df_metrics = pd.read_csv(METRICS_LOG_FILE)
    unique_question_ids = df_metrics['question_id'].unique().tolist()

    for pipeline_id in top_5_pipelines:
        print(f"\n>>> ĐANG CHẠY EVALUATION CHO PIPELINE: {pipeline_id}")
        
        for q_id in unique_question_ids:
            is_evaluated = not df_checkpoint[
                (df_checkpoint['pipeline_id'] == pipeline_id) & 
                (df_checkpoint['question_id'] == q_id)
            ].empty
            
            if is_evaluated:
                continue
                
            # Đọc dữ liệu từ ChromaDB
            db_result = question_collection.get(ids=[q_id], include=["metadatas", "documents"])
            if not db_result["ids"]:
                continue
                
            metadata = db_result["metadatas"][0]
            question_text = db_result["documents"][0]
            ground_truth_answer = metadata.get("ground_truth_answer", "")
            
            print(f" -> Đang xử lý Câu hỏi ID: {q_id}")
            
            # Khởi tạo một vòng lặp vô hạn cho đến khi câu hỏi này được xử lý thành công
            # Mục đích: Nếu dính lỗi hết hạn ngạch ngày (1500 RPD), code sẽ ngủ đông rồi thử lại câu này
            while True:
                try:
                    # 1. Chạy sinh câu trả lời bằng mô hình Qwen local qua Ollama
                    answer, contexts = run_qwen_rag_pipeline_ollama(pipeline_id, question_text)
                    
                    # 2. Chuẩn bị cấu trúc dữ liệu cho Ragas
                    sample_data = {
                        "question": [question_text],
                        "contexts": [contexts],
                        "answer": [answer],
                        "ground_truth": [ground_truth_answer] 
                    }
                    dataset = Dataset.from_dict(sample_data)
                    
                    # 3. Nghỉ an toàn tránh lỗi RPM (Hạn ngạch phút)
                    time.sleep(DELAY_BETWEEN_REQUESTS)
                    
                    # 4. Chạy hàm đánh giá kết hợp (Gemini + Llama Local)
                    result = evaluate(dataset, metrics=metrics)
                    
                    score_faithfulness = result.get("faithfulness", 0.0)
                    score_relevance = result.get("answer_relevance", 0.0)
                    score_correctness = result.get("answer_correctness", 0.0)
                    
                    # 5. Ghi dữ liệu vào CSV Checkpoint ngay lập tức
                    new_row = pd.DataFrame([{
                        "pipeline_id": pipeline_id,
                        "question_id": q_id,
                        "question": question_text,
                        "contexts": str(contexts),
                        "answer": answer,
                        "ground_truth": ground_truth_answer,
                        "faithfulness": score_faithfulness,
                        "answer_relevance": score_relevance,
                        "answer_correctness": score_correctness
                    }])
                    
                    new_row.to_csv(RESULT_LOG_FILE, mode='a', header=False, index=False)
                    df_checkpoint = pd.concat([df_checkpoint, new_row], ignore_index=True)
                    
                    # Thoát khỏi vòng lặp 'while True' của câu hỏi hiện tại để chuyển sang câu tiếp theo
                    break
                    
                except Exception as error:
                    error_msg = str(error).lower()
                    
                    # ĐOẠN KIỂM TRA CHẶN LỖI 1500 RPD (HẠN NGẠCH NGÀY CỦA GOOGLE)
                    if "quota" in error_msg and "day" in error_msg:
                        print("\n[CẢNH BÁO NGHẼN HỆ THỐNG]: Bạn đã dùng hết 1,500 requests ngày của tài khoản Google Free!")
                        print("Hệ thống tiến hành kích hoạt chế độ NGỦ ĐÔNG TRONG 24 GIỜ để chờ reset quota...")
                        print("⚠️ Xin vui lòng KHÔNG tắt chương trình. Tiến trình sẽ tự động chạy tiếp vào ngày mai.")
                        time.sleep(86400) # Sleep đúng 24 tiếng đồng hồ
                        print("--- Hệ thống đã thức dậy! Đang thử lại câu hỏi vừa rồi... ---")
                        # Không có lệnh break ở đây, vòng lặp 'while True' sẽ chạy lại câu hỏi này với Quota mới
                    
                    # Nếu là lỗi nghẽn phút (RPM) thông thường hoặc mất mạng tạm thời
                    else:
                        print(f" [Lỗi kết nối / Nghẽn phút]: {error}")
                        print("Tạm nghỉ 60 giây để hệ thống hồi phục...")
                        time.sleep(60)
                        # Tiếp tục vòng lặp để thử lại

    print("\n=== [HOÀN THÀNH] QUÁ TRÌNH TRÍCH XUẤT VÀ ĐÁNH GIÁ RAGAS ĐÃ HOÀN TẤT ===")

