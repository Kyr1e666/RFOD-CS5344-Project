import pandas as pd
import ast
import json
from typing import List, Dict, Any, Set


def parse_list_field(value: Any) -> List:
    """安全地解析字符串形式的list字段（如stackAddresses、args）"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def extract_arg_features(df: pd.DataFrame, args_col: str = "args") -> pd.DataFrame:
    """展平args字段"""
    if args_col not in df.columns:
        print(f"⚠️ 警告：CSV中不存在 '{args_col}' 列，跳过 args 特征提取。")
        return df

    all_feature_names: Set[str] = set()
    feature_types: Dict[str, Set[str]] = {}

    for args_str in df[args_col]:
        for arg in parse_list_field(args_str):
            if isinstance(arg, dict) and "name" in arg:
                name = arg["name"]
                all_feature_names.add(name)
                t = arg.get("type", "unknown")
                feature_types.setdefault(name, set()).add(t)

    all_feature_names = sorted(list(all_feature_names))

    flattened_features = []
    for args_str in df[args_col]:
        feature_map = {name: None for name in all_feature_names}
        for arg in parse_list_field(args_str):
            if isinstance(arg, dict) and "name" in arg and "value" in arg:
                feature_map[arg["name"]] = arg["value"]
        flattened_features.append(feature_map)

    args_df = pd.DataFrame(flattened_features)
    df = pd.concat([df.reset_index(drop=True), args_df.reset_index(drop=True)], axis=1)
    
    print(f"✅ args 列已展平，新增 {len(all_feature_names)} 个特征。")
    return df

def convert_dtypes_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """将 timestamp、argsNum、stack_depth 转为数值型，其他全部为分类型。"""
    numeric_cols = ["timestamp", "argsNum", "stack_depth"]
    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col != "args":
            df[col] = df[col].astype(str)
    print(f"✅ 类型转换完成：{len(numeric_cols)} 数值型特征，其余为分类型。")
    return df


def clean_csv(input_path: str, output_path: str, process_args: bool = True, save: bool = True):
    """
    主清洗函数：增加了 processId 分组时间归一化
    """
    print(f"🔧 开始处理文件：{input_path}")
    df = pd.read_csv(input_path)

    # stack_depth 特征
    if "stackAddresses" in df.columns:
        df["stackAddresses"] = df["stackAddresses"].apply(parse_list_field)
        df["stack_depth"] = df["stackAddresses"].apply(len)
        print("✅ stack_depth 特征已计算。")

    # 删除指定列
    drop_cols = ["threadId", "eventId", "stackAddresses"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    print(f"✅ 删除了列：{drop_cols}")

    # 展平 args
    args_col_present = "args" in df.columns
    if process_args and args_col_present:
        df = extract_arg_features(df, args_col="args")
        df = df.drop(columns=["args"], errors="ignore")
    elif not process_args and args_col_present:
        df = df.drop(columns=["args"], errors="ignore")
        print("⏭️ 已跳过 'args' 列的处理并将其删除。")
    elif process_args and not args_col_present:
        print("⚠️ 警告：设置了处理 'args'，但 CSV 中不存在该列。")

    # ✅ 新增：按 processId 分组归一化 timestamp
    if "processId" in df.columns and "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["timestamp"] = df.groupby("processId")["timestamp"].transform(lambda x: x - x.min())
        print("✅ 已完成按 processId 分组的 timestamp 归一化。")
    else:
        print("⚠️ 缺少 processId 或 timestamp 列，跳过时间归一化。")

    # 类型转换
    df = convert_dtypes_for_training(df)

    if save:
        df.to_csv(output_path, index=False)
        print(f"\n🎉 处理完毕，Cleaned data saved to: {output_path}")
    return df


if __name__ == "__main__":
    input_file = "data/processes_train.csv"
    output_file = "data/cleaned_train.csv"
   
    SHOULD_PROCESS_ARGS = False
    save = True

    clean_csv(
        input_path=input_file, 
        output_path=output_file, 
        process_args=SHOULD_PROCESS_ARGS,
        save=True
    )
