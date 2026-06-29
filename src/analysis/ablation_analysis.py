import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dunnett  # Đảm bảo tính tương thích cao

def run_multifactor_dunnett(csv_path, metric='recall_at_5'):
    print(f"--- Đang tải dữ liệu từ {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{csv_path}'. Vui lòng kiểm tra lại!")
        return

    factors = ['chunking', 'pre_retrieval', 'retrieval', 'index_structure', 'post_retrieval']
    summary_frames = []
    
    for factor in factors:
        print(f"\n[Xử lý Nhân tố]: {factor}")
        
        # Mức đầu tiên xuất hiện tự động được coi là Baseline/Control
        unique_levels = df[factor].unique().tolist()
        baseline_level = unique_levels[0]
        print(f"  -> Mức đối chứng (Baseline/Control): '{baseline_level}'")
        
        # Tách mảng điểm số của nhóm Baseline
        baseline_data = df[df[factor] == baseline_level][metric].values
        
        # Gom mảng điểm số của các nhóm nâng cao (Treatment groups)
        treatment_groups = unique_levels[1:]
        treatment_data_list = [df[df[factor] == level][metric].values for level in treatment_groups]
        
        if len(baseline_data) == 0 or any(len(g) == 0 for g in treatment_data_list):
            print(f"  [Bỏ quan]: Nhân tố '{factor}' có nhóm trống dữ liệu.")
            continue
            
        # Chạy kiểm định Dunnett qua Scipy
        res = dunnett(*treatment_data_list, control=baseline_data)
        
        # SỬA LỖI TẠI ĐÂY: Gọi hàm confidence_interval() để lấy đối tượng chứa mảng biên
        ci_object = res.confidence_interval()
        
        rows = []
        for i, level in enumerate(treatment_groups):
            # Tính toán chênh lệch trung bình thực tế
            mean_treatment = np.mean(treatment_data_list[i])
            mean_baseline = np.mean(baseline_data)
            mean_diff = mean_treatment - mean_baseline
            
            # Trích xuất biên lower và upper từ mảng kết quả của Scipy theo chỉ mục i
            low_bound = ci_object.low[i]
            high_bound = ci_object.high[i]
            
            rows.append({
                'factor': factor,
                'comparison_level': level,
                'baseline_level': baseline_level,
                'diff': mean_diff,
                'pvalue': res.pvalue[i],
                'lower': low_bound,
                'upper': high_bound
            })
            
        res_df = pd.DataFrame(rows)
        summary_frames.append(res_df)
        
        # Vẽ biểu đồ khoảng tin cậy Dunnett trực quan
        plt.figure(figsize=(10, 5))
        for idx, row in res_df.iterrows():
            plt.errorbar(x=row['diff'], y=row['comparison_level'], 
                         xerr=[[row['diff'] - row['lower']], [row['upper'] - row['diff']]],
                         fmt='o', color='blue', ecolor='red', capsize=5, elinewidth=2)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
        plt.title(f"Dunnett's Test - '{factor}' Analysis (vs Baseline: '{baseline_level}')")
        plt.xlabel(f"Chênh lệch điểm số trung bình của {metric}")
        plt.ylabel("Cấu hình thử nghiệm")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/dunnett_{metric}_{factor}_ci.png")
        plt.close()
        print(f"  -> Đã lưu biểu đồ khoảng tin cậy: dunnett_{metric}_{factor}_ci.png")

    # Gom toàn bộ báo cáo và xuất file CSV phục vụ cho việc làm Ablation Study
    if summary_frames:
        final_report = pd.concat(summary_frames, ignore_index=True)
        
        def determine_impact(row):
            if row['pvalue'] >= 0.05:
                return "Không khác biệt (Không có ý nghĩa thống kê)"
            elif row['diff'] > 0:
                return f"TỐT HƠN Baseline (Tăng {row['diff']:.4f} điểm)"
            else:
                return f"TỆ HƠN Baseline (Giảm {abs(row['diff']):.4f} điểm)"
                
        final_report['ablation_conclusion'] = final_report.apply(determine_impact, axis=1)
        
        output_csv = f"reports/ablation_dunnett_{metric}_report.csv"
        final_report.to_csv(output_csv, index=False)
        print(f"\n[HOÀN THÀNH XUẤT SẮC]: Báo cáo Dunnett đã xuất ra tại: {output_csv}")
        return final_report
    else:
        print("\n[LỖI]: Không có dữ liệu kiểm định nào được tạo ra.")
        return None

if __name__ == "__main__":
    # Tự động trỏ thẳng tới file log sàng lọc của bạn
    run_multifactor_dunnett('reports/stage1_screening_logs.csv', metric='hit_rate_at_5')
    run_multifactor_dunnett('reports/stage1_screening_logs.csv', metric='recall_at_5')
    run_multifactor_dunnett('reports/stage1_screening_logs.csv', metric='mrr_at_5')
    run_multifactor_dunnett('reports/stage1_screening_logs.csv', metric='ndcg_at_5')
