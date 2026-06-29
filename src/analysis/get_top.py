import pandas as pd

def analyze_rag_pipelines(file_path):
    # 1. Đọc dữ liệu từ file CSV
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return
    
    # Định nghĩa các cột cấu hình và các chỉ số metric
    config_cols = ['chunking', 'pre_retrieval', 'retrieval', 'index_structure', 'post_retrieval']
    metric_cols = ['hit_rate_at_5', 'recall_at_5', 'mrr_at_5', 'ndcg_at_5']
    
    # 2. Gom nhóm theo pipeline_id và các trường cấu hình để tính trung bình cộng (Mean) trên toàn bộ câu hỏi
    # Việc giữ lại các trường cấu hình giúp bạn dễ đọc kết quả hơn thay vì chỉ thấy mỗi pipeline_id
    pipeline_grouped = df.groupby(['pipeline_id'] + config_cols)[metric_cols].mean().reset_index()
    
    # 3. Tính toán 2 tiêu chí đánh giá
    # Tiêu chí 1: Trung bình cộng của cả 4 yếu tố (Hit Rate, Recall, MRR, NDCG)
    pipeline_grouped['score_criterion_1'] = pipeline_grouped[metric_cols].mean(axis=1)
    
    # Tiêu chí 2: công thức Recall@5 * 0.7 + MRR@5 * 0.3
    pipeline_grouped['score_criterion_2'] = (pipeline_grouped['recall_at_5'] * 0.7) + (pipeline_grouped['mrr_at_5'] * 0.3)
    
    # 4. Trích xuất Top 10 cho Tiêu chí 1
    top_10_criterion_1 = pipeline_grouped.sort_values(by='score_criterion_1', ascending=False).head(10)
    
    # 5. Trích xuất Top 10 cho Tiêu chí 2
    top_10_criterion_2 = pipeline_grouped.sort_values(by='score_criterion_2', ascending=False).head(10)
    
    # 6. Xuất kết quả ra màn hình và lưu thành file CSV mới để bạn tiện làm báo cáo
    print("="*40)
    print("TOP 10 PIPELINES - TIÊU CHÍ 1 (TRUNG BÌNH 4 CHỈ SỐ)")
    print("="*40)
    print(top_10_criterion_1[['pipeline_id'] + config_cols + ['score_criterion_1']].to_string(index=False))
    top_10_criterion_1.to_csv('reports/top_10_pipeline_tieu_chi_1.csv', index=False)
    print("\n[Đã lưu kết quả Tiêu chí 1 vào file: top_10_pipeline_tieu_chi_1.csv]\n")
    
    print("="*40)
    print("TOP 10 PIPELINES - TIÊU CHÍ 2 (RECALL@5 * 0.7 + MRR@5 * 0.3)")
    print("="*40)
    print(top_10_criterion_2[['pipeline_id'] + config_cols + ['score_criterion_2']].to_string(index=False))
    top_10_criterion_2.to_csv('reports/top_10_pipeline_tieu_chi_2.csv', index=False)
    print("\n[Đã lưu kết quả Tiêu chí 2 vào file: top_10_pipeline_tieu_chi_2.csv]\n")

if __name__ == "__main__":
    file_input = 'reports/stage1_screening_logs.csv' 
    analyze_rag_pipelines(file_input)
