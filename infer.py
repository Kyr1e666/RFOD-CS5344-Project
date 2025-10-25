"""
RFOD 模型推理脚本。
"""
import pickle
import pandas as pd
import numpy as np
from rfod import RFOD, _safe_clean_csv, _select_and_align_features, REQ_FEATURES

def main(model_path: str, test_csv: str, process_args: bool = False):
    # 加载模型
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    rfod = data['model']
    
    print(f"--> 加载模型完成")

    # 清洗 test
    print(f"--> 清洗测试集: {test_csv}")
    df_test = _safe_clean_csv(test_csv, process_args=process_args)
    if 'Id' not in df_test.columns:
        raise ValueError("测试集必须包含 'Id' 列。")

    X_test = _select_and_align_features(df_test, REQ_FEATURES)

    # 预测
    test_scores = rfod.predict(X_test)
    
    # 归一化到0-1范围
    min_score = np.min(test_scores)
    max_score = np.max(test_scores)
    normalized_scores = (test_scores - min_score) / (max_score - min_score)
    
    # 输出 CSV
    out_df = pd.DataFrame({
        'Id': df_test['Id'],
        'target': normalized_scores  # 输出归一化后的分数
    })
    
    out_df.to_csv('result/kaggle_submission.csv', index=False)
    print("--> 已保存预测结果到 'result/kaggle_submission.csv' (包含归一化后的异常分数)")

if __name__ == "__main__":
    model_path = "model/best_model.pkl"
    test_csv = "data/processes_test.csv"
    main(model_path, test_csv, process_args=False)