#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理脚本 - 对model/all中的多个模型运行推理
"""
import sys
import io
import os
import glob
import pickle
import pandas as pd
import numpy as np

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入RFOD类（确保pickle可以正确加载）
from rfod import RFOD, _safe_clean_csv, _select_and_align_features, REQ_FEATURES

def run_single_inference(model_path, test_csv, output_path, normalize_method="minmax", process_args=False):
    """运行单个模型的推理"""

    # 加载模型
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    rfod = data['model']

    # 清洗测试集
    df_test = _safe_clean_csv(test_csv, process_args=process_args)
    X_test = _select_and_align_features(df_test, REQ_FEATURES)

    # 预测
    test_scores = rfod.predict(X_test, clip_scores=False)

    # 归一化
    if normalize_method == "minmax":
        score_range = test_scores.max() - test_scores.min()
        if score_range > 1e-10:
            normalized_scores = (test_scores - test_scores.min()) / score_range
        else:
            normalized_scores = np.zeros_like(test_scores)
    elif normalize_method == "clip":
        normalized_scores = np.clip(test_scores, 0.0, 1.0)
    elif normalize_method == "none":
        normalized_scores = test_scores
    else:
        normalized_scores = test_scores

    # 保存结果
    out_df = pd.DataFrame({
        'Id': df_test['Id'],
        'target': normalized_scores
    })

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    out_df.to_csv(output_path, index=False)

    return out_df


def batch_infer(
    model_dir: str = "model/all",
    test_csv: str = "data/processes_test.csv",
    output_dir: str = "result",
    pattern: str = "*alpha=0p003*",  # 筛选 alpha=0.003 的模型
    normalize_method: str = "minmax",
    process_args: bool = False
):
    """
    批量推理

    参数:
        model_dir: 模型目录
        test_csv: 测试集路径
        output_dir: 输出目录
        pattern: 模型文件名模式
        normalize_method: 归一化方法
        process_args: 是否处理args列
    """

    # 查找所有匹配的模型
    search_pattern = os.path.join(model_dir, f"rfod_{pattern}.pkl")
    model_files = glob.glob(search_pattern)

    if not model_files:
        print(f"错误: 在 {model_dir} 中未找到匹配 {pattern} 的模型")
        print(f"搜索模式: {search_pattern}")
        return

    print("=" * 80)
    print(f"找到 {len(model_files)} 个模型文件:")
    print("=" * 80)
    for i, mf in enumerate(model_files, 1):
        print(f"  {i}. {os.path.basename(mf)}")
    print("")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 对每个模型运行推理
    results = []
    for i, model_path in enumerate(model_files, 1):
        print("\n" + "=" * 80)
        print(f"推理 [{i}/{len(model_files)}]: {os.path.basename(model_path)}")
        print("=" * 80)

        # 从文件名提取参数信息
        basename = os.path.basename(model_path)
        # 例如: rfod_alpha=0p003_beta=0p7_max_depth=15_n_estimators=60__xxxxx.pkl

        # 提取关键参数用于输出文件名
        parts = basename.replace("rfod_", "").replace(".pkl", "").split("__")[0]

        # 生成输出文件名
        output_filename = f"kaggle_submission_{parts}.csv"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # 运行推理
            out_df = run_single_inference(
                model_path=model_path,
                test_csv=test_csv,
                output_path=output_path,
                normalize_method=normalize_method,
                process_args=process_args
            )

            results.append({
                'model': basename,
                'output': output_filename,
                'status': 'success',
                'samples': len(out_df)
            })

            print(f"\nOK 成功: {output_filename}")

        except Exception as e:
            print(f"\nERROR 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': basename,
                'output': output_filename,
                'status': 'failed',
                'error': str(e)
            })

    # 打印总结
    print("\n" + "=" * 80)
    print("批量推理完成")
    print("=" * 80)

    success_count = sum(1 for r in results if r['status'] == 'success')
    fail_count = len(results) - success_count

    print(f"\n总计: {len(results)} 个模型")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")

    if success_count > 0:
        print(f"\n成功生成的提交文件:")
        for r in results:
            if r['status'] == 'success':
                print(f"  ✓ {r['output']} ({r['samples']} 样本)")

    if fail_count > 0:
        print(f"\n失败的模型:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  ✗ {r['model']}: {r['error']}")

    return results


if __name__ == "__main__":
    # 批量推理所有 alpha=0.003 的模型
    results = batch_infer(
        model_dir="model/all",
        test_csv="data/processes_test.csv",
        output_dir="result",
        pattern="*alpha=0p003*",  # 筛选 alpha=0.003
        normalize_method="minmax",  # Min-Max归一化
        process_args=False
    )