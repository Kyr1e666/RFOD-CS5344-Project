#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估所有模型在验证集上的性能
"""
import sys
import io
import os
import glob
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rfod import (
    RFOD, _safe_clean_csv, _select_and_align_features, REQ_FEATURES,
    _compute_binary_metrics, _scan_thresholds_from_scores
)
from sklearn.metrics import f1_score, confusion_matrix


def evaluate_single_model(
    model_path: str,
    valid_csv: str,
    process_args: bool = False,
    threshold_strategy: str = "train_quantile",
    threshold_percentile: float = 99.7
) -> Dict[str, Union[str, float, int, None]]:
    """
    评估单个模型在验证集上的性能

    参数:
        model_path: 模型文件路径
        valid_csv: 验证集CSV路径
        process_args: 是否处理args列
        threshold_strategy: 阈值策略
        threshold_percentile: 阈值分位数

    返回:
        包含所有评估指标的字典
    """

    # 加载模型
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    rfod = data['model']
    saved_threshold = data.get('threshold', None)
    params = data.get('params', {})

    # 提取模型签名
    basename = os.path.basename(model_path)
    signature = basename.replace("rfod_", "").replace(".pkl", "")

    # 清洗验证集
    df_valid = _safe_clean_csv(valid_csv, process_args=process_args)

    if "target" not in df_valid.columns:
        raise ValueError("验证集必须包含 'target' 列")

    X_valid = _select_and_align_features(df_valid, REQ_FEATURES)
    y_true = df_valid["target"].astype(int).values

    # 预测
    valid_scores = rfod.predict(X_valid, clip_scores=False)

    # 确定阈值
    if saved_threshold is not None:
        # 使用保存的阈值
        threshold = saved_threshold
        threshold_source = "saved_in_model"
    elif threshold_strategy == "train_quantile":
        # 使用分位数（基于验证集分数）
        threshold = float(np.percentile(valid_scores, threshold_percentile))
        threshold_source = f"valid_quantile_{threshold_percentile}"
    elif threshold_strategy == "valid_best_f1":
        # 在验证集上搜索最优F1阈值
        candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
        best_f1, threshold = -1.0, candidates[0]
        for t in candidates:
            yp = (valid_scores > t).astype(int)
            f1v = f1_score(y_true, yp, zero_division=0)
            if f1v > best_f1:
                best_f1, threshold = f1v, float(t)
        threshold_source = "valid_best_f1"
    elif threshold_strategy == "valid_best_j":
        # 在验证集上搜索最优Youden's J阈值
        candidates = _scan_thresholds_from_scores(valid_scores, n_points=256)
        best_j, threshold = -1.0, candidates[0]
        for t in candidates:
            yp = (valid_scores > t).astype(int)
            cm = confusion_matrix(y_true, yp, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                j = tpr - fpr
            else:
                j = 0.0
            if j > best_j:
                best_j, threshold = j, float(t)
        threshold_source = "valid_best_j"
    else:
        threshold = float(np.percentile(valid_scores, threshold_percentile))
        threshold_source = f"valid_quantile_{threshold_percentile}"

    # 二分类预测
    y_pred = (valid_scores > threshold).astype(int)

    # 计算指标
    metrics = _compute_binary_metrics(y_true, valid_scores, y_pred)

    # 构建结果
    result = {
        'signature': signature,
        'model_file': basename,
        'threshold': threshold,
        'threshold_source': threshold_source,
        'n_valid': len(y_true),
        'score_min': float(valid_scores.min()),
        'score_max': float(valid_scores.max()),
        'score_mean': float(valid_scores.mean()),
        'score_std': float(valid_scores.std()),
    }

    # 添加模型参数
    result.update(params)

    # 添加评估指标
    result.update(metrics)

    return result


def evaluate_all_models(
    model_dir: str = "model/all",
    valid_csv: str = "data/processes_valid.csv",
    output_csv: str = "model/evaluation_results.csv",
    output_jsonl: str = "model/evaluation_results.jsonl",
    process_args: bool = False,
    threshold_strategy: str = "saved_in_model",  # 使用保存的阈值
    threshold_percentile: float = 99.7,
    pattern: str = "*.pkl"
):
    """
    评估所有模型

    参数:
        model_dir: 模型目录
        valid_csv: 验证集路径
        output_csv: 输出CSV路径
        output_jsonl: 输出JSONL路径
        process_args: 是否处理args列
        threshold_strategy: 阈值策略
        threshold_percentile: 阈值分位数
        pattern: 模型文件匹配模式
    """

    # 查找所有模型
    search_pattern = os.path.join(model_dir, pattern)
    model_files = glob.glob(search_pattern)

    if not model_files:
        print(f"错误: 在 {model_dir} 中未找到模型文件")
        return

    print("=" * 80)
    print(f"找到 {len(model_files)} 个模型文件")
    print("=" * 80)

    results = []

    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] 评估: {os.path.basename(model_path)}")

        try:
            result = evaluate_single_model(
                model_path=model_path,
                valid_csv=valid_csv,
                process_args=process_args,
                threshold_strategy=threshold_strategy,
                threshold_percentile=threshold_percentile
            )

            results.append(result)

            # 打印关键指标
            print(f"  ✓ F1={result.get('f1', 0):.4f}, "
                  f"ROC-AUC={result.get('roc_auc', 0):.4f}, "
                  f"PR-AUC={result.get('pr_auc', 0):.4f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n没有成功评估的模型")
        return

    # 保存结果
    df_results = pd.DataFrame(results)

    # 按F1降序排序
    if 'f1' in df_results.columns:
        df_results = df_results.sort_values('f1', ascending=False)

    # 保存CSV
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ 已保存评估结果到: {output_csv}")

    # 保存JSONL
    import json
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"✓ 已保存评估明细到: {output_jsonl}")

    # 打印Top-5
    print("\n" + "=" * 80)
    print("Top-5 模型（按F1排序）")
    print("=" * 80)

    cols_to_show = ['signature', 'f1', 'roc_auc', 'pr_auc', 'precision', 'recall',
                    'balanced_accuracy', 'threshold', 'alpha', 'beta', 'n_estimators', 'max_depth']

    # 只显示存在的列
    cols_to_show = [c for c in cols_to_show if c in df_results.columns]

    top5 = df_results.head(5)[cols_to_show]
    print(top5.to_string(index=False))

    # 统计摘要
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)

    if 'f1' in df_results.columns:
        print(f"F1 分数:")
        print(f"  最佳: {df_results['f1'].max():.4f}")
        print(f"  平均: {df_results['f1'].mean():.4f}")
        print(f"  最差: {df_results['f1'].min():.4f}")

    if 'roc_auc' in df_results.columns:
        print(f"\nROC-AUC:")
        print(f"  最佳: {df_results['roc_auc'].max():.4f}")
        print(f"  平均: {df_results['roc_auc'].mean():.4f}")
        print(f"  最差: {df_results['roc_auc'].min():.4f}")

    if 'pr_auc' in df_results.columns:
        print(f"\nPR-AUC:")
        print(f"  最佳: {df_results['pr_auc'].max():.4f}")
        print(f"  平均: {df_results['pr_auc'].mean():.4f}")
        print(f"  最差: {df_results['pr_auc'].min():.4f}")

    # 最佳模型
    if 'f1' in df_results.columns:
        best_idx = df_results['f1'].idxmax()
        best_model = df_results.loc[best_idx]

        print("\n" + "=" * 80)
        print("最佳模型（按F1）")
        print("=" * 80)
        print(f"文件: {best_model['model_file']}")
        print(f"参数: alpha={best_model.get('alpha', 'N/A')}, "
              f"beta={best_model.get('beta', 'N/A')}, "
              f"n_estimators={best_model.get('n_estimators', 'N/A')}, "
              f"max_depth={best_model.get('max_depth', 'N/A')}")
        print(f"F1: {best_model['f1']:.4f}")
        print(f"ROC-AUC: {best_model.get('roc_auc', 0):.4f}")
        print(f"PR-AUC: {best_model.get('pr_auc', 0):.4f}")
        print(f"阈值: {best_model['threshold']:.6f}")

    return df_results


if __name__ == "__main__":
    # 评估所有模型
    df_results = evaluate_all_models(
        model_dir="model/all",
        valid_csv="data/processes_valid.csv",
        output_csv="model/evaluation_results.csv",
        output_jsonl="model/evaluation_results.jsonl",
        process_args=False,
        threshold_strategy="saved_in_model",  # 使用训练时保存的阈值
        pattern="*.pkl"  # 评估所有模型
    )

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    print("\n查看完整结果:")
    print("  CSV: model/evaluation_results.csv")
    print("  JSONL: model/evaluation_results.jsonl")
