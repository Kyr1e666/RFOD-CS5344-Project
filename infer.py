"""
RFOD 模型推理脚本。
"""
import os
import pickle
import pandas as pd
import numpy as np
from rfod import RFOD, _safe_clean_csv, _select_and_align_features, REQ_FEATURES

def main(model_path: str, test_csv: str, output_path: str = "result/kaggle_submission.csv",
         process_args: bool = False, use_threshold: bool = False,
         normalize_method: str = "minmax"):
    """
    运行RFOD模型推理

    参数:
        model_path: 模型文件路径
        test_csv: 测试集CSV路径
        output_path: 输出文件路径
        process_args: 是否处理args列
        use_threshold: 是否使用阈值进行二分类（True=输出0/1，False=输出分数）
        normalize_method: 归一化方法
            - "minmax": Min-Max归一化到[0,1]（推荐，保留所有相对关系）
            - "robust": 使用分位数的鲁棒归一化（对极端值不敏感）
            - "clip": 简单裁剪到[0,1]（会丢失>1的信息）
            - "none": 不归一化（保留原始分数）
    """

    # 加载模型
    print(f"--> 加载模型: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path, 'rb') as f:

        data = pickle.load(f)

    rfod = data['model']
    threshold = data.get('threshold', None)
    params = data.get('params', {})

    print(f"    模型参数: {params}")
    if threshold is not None:
        print(f"    保存的阈值: {threshold:.6f}")
    print(f"    模型后端: {rfod.backend}")

    # 清洗测试集
    print(f"--> 清洗测试集: {test_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"测试集文件不存在: {test_csv}")

    df_test = _safe_clean_csv(test_csv, process_args=process_args)

    if 'Id' not in df_test.columns:
        raise ValueError("测试集必须包含 'Id' 列。")

    print(f"    测试集样本数: {len(df_test)}")

    X_test = _select_and_align_features(df_test, REQ_FEATURES)

    if X_test.empty:
        raise ValueError("测试数据为空或所有特征缺失")

    # 预测（获取原始分数，不在模型内裁剪）
    print("--> 开始预测...")
    test_scores = rfod.predict(X_test, clip_scores=False)  # 获取原始分数

    original_min = test_scores.min()
    original_max = test_scores.max()
    print(f"    原始分数范围: [{original_min:.6f}, {original_max:.6f}]")
    print(f"    原始分数均值: {test_scores.mean():.6f}")
    print(f"    原始分数标准差: {test_scores.std():.6f}")

    # 归一化处理
    if normalize_method == "minmax":
        # Min-Max归一化：将分数线性映射到[0,1]
        score_range = original_max - original_min
        if score_range > 1e-10:
            normalized_scores = (test_scores - original_min) / score_range
        else:
            normalized_scores = np.zeros_like(test_scores)
        print(f"--> Min-Max归一化: [{original_min:.6f}, {original_max:.6f}] -> [0.0, 1.0]")
        print(f"    归一化后范围: [{normalized_scores.min():.6f}, {normalized_scores.max():.6f}]")

    elif normalize_method == "robust":
        # 鲁棒归一化：使用中位数和IQR，对极端值不敏感
        q25, q50, q75 = np.percentile(test_scores, [25, 50, 75])
        iqr = q75 - q25
        if iqr > 1e-10:
            normalized_scores = (test_scores - q50) / iqr
            # 再映射到[0,1]
            normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
        else:
            normalized_scores = np.zeros_like(test_scores)
        print(f"--> 鲁棒归一化 (中位数={q50:.4f}, IQR={iqr:.4f})")
        print(f"    归一化后范围: [{normalized_scores.min():.6f}, {normalized_scores.max():.6f}]")

    elif normalize_method == "clip":
        # 简单裁剪到[0,1]
        normalized_scores = np.clip(test_scores, 0.0, 1.0)
        print(f"--> 裁剪到 [0, 1]: [{original_min:.6f}, {original_max:.6f}] -> [{normalized_scores.min():.6f}, {normalized_scores.max():.6f}]")
        if original_max > 1.0:
            clipped_count = (test_scores > 1.0).sum()
            print(f"    警告: {clipped_count} 个样本被裁剪到1.0 ({clipped_count/len(test_scores)*100:.2f}%)")

    elif normalize_method == "none":
        # 保留原始分数
        normalized_scores = test_scores
        print(f"--> 不归一化，保留原始分数")

    else:
        raise ValueError(f"未知的归一化方法: {normalize_method}. 请使用 'minmax', 'robust', 'clip', 或 'none'")

    # 根据参数决定输出格式
    if use_threshold and threshold is not None:
        # 使用阈值进行二分类
        predictions = (normalized_scores > threshold).astype(int)
        print(f"\n--> 使用阈值 {threshold:.6f} 进行二分类")
        print(f"    预测为异常的样本数: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")

        out_df = pd.DataFrame({
            'Id': df_test['Id'],
            'target': predictions
        })
        output_type = "二分类 (0/1)"
    else:
        # 输出归一化后的异常分数
        print(f"\n--> 输出归一化后的异常分数（作为异常概率）")

        out_df = pd.DataFrame({
            'Id': df_test['Id'],
            'target': normalized_scores
        })
        output_type = f"异常概率 (归一化方法: {normalize_method})"

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"    创建输出目录: {output_dir}")

    # 保存结果
    out_df.to_csv(output_path, index=False)
    print(f"--> 已保存预测结果到: {output_path}")
    print(f"    输出格式: {output_type}")

    # 输出前几行预览
    print("\n预测结果预览:")
    print(out_df.head(10))

    return out_df

if __name__ == "__main__":
    model_path = "model/all/rfod_alpha=0p005_beta=0p7_max_depth=15_n_estimators=60__3168cd65.pkl"  # 使用最佳模型
    test_csv = "data/processes_test.csv"

    # 归一化方法选择：
    # - "minmax": Min-Max归一化（推荐，保留所有信息）
    # - "robust": 鲁棒归一化（对极端值不敏感）
    # - "clip": 简单裁剪（会丢失>1的信息）
    # - "none": 不归一化（保留原始分数）

    main(
        model_path=model_path,
        test_csv=test_csv,
        output_path="result/kaggle_submission1.csv",
        process_args=False,
        use_threshold=False,         # False=输出分数, True=输出0/1
        normalize_method="clip"    # 使用Min-Max归一化（推荐）
    )