"""
RFOD (Random Forest-based Outlier Detection) 
"""
import os
import tempfile
import json
import itertools
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from data_process import clean_csv



class RFOD:
    def __init__(
        self,
        alpha: float = 0.02,
        beta: float = 0.7,
        n_estimators: int = 30,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        self.alpha = alpha
        self.beta = beta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.forests_ = {}
        self.feature_types_ = {}
        self.quantiles_ = {}
        self.feature_names_ = []
        self.n_features_ = 0
        self.encoders_: Dict[str, LabelEncoder] = {}

    def _identify_feature_types(self, X: pd.DataFrame) -> Dict[int, str]:
        """识别数值型和类别型特征"""
        feature_types = {}
        for idx, col in enumerate(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[idx] = 'numeric'
            else:
                feature_types[idx] = 'categorical'
        return feature_types

    def _compute_quantiles(self, X: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        """计算数值型特征的 alpha 和 1-alpha 分位数"""
        quantiles = {}
        for idx, col in enumerate(X.columns):
            if self.feature_types_[idx] == 'numeric':
                q_low = X[col].quantile(self.alpha)
                q_high = X[col].quantile(1 - self.alpha)
                if q_high - q_low < 1e-10:
                    q_high = q_low + 1.0
                quantiles[idx] = (q_low, q_high)
        return quantiles

    def _fit_encoders(self, X: pd.DataFrame):
        """拟合类别特征的 LabelEncoders"""
        self.encoders_ = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                series = X[col].astype(str).fillna("NaN_TOKEN")
                le.fit(series)
                self.encoders_[col] = le

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用存储的 LabelEncoders 转换数据"""
        X_transformed = X.copy()
        for col, le in self.encoders_.items():
            if col in X_transformed.columns:
                series = X_transformed[col].astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~series.isin(le.classes_)
                series.loc[unseen_mask] = le.classes_[0]
                transformed_series = le.transform(series)
                transformed_series[unseen_mask] = -1
                X_transformed[col] = transformed_series
        return X_transformed

    def _train_feature_forest(self, X: pd.DataFrame, feature_idx: int) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """训练单个特征的预测森林"""
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]

        X_train_encoded = self._transform_data(X_train_df)
        
        target_col_name = X.columns[feature_idx]
        if self.feature_types_[feature_idx] == 'categorical':
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = y_train_encoded
            
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )
        else:
            y_train = y_train.fillna(y_train.mean())
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )
            
        forest.fit(X_train_encoded, y_train)
        return forest

    def _prune_forest(self, forest, X: pd.DataFrame, feature_idx: int):
        """使用真实的 OOB 样本修剪森林"""
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]

        X_train_encoded = self._transform_data(X_train_df)
        
        target_col_name = X.columns[feature_idx]
        is_classifier = isinstance(forest, RandomForestClassifier)
        
        if is_classifier:
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = pd.Series(y_train_encoded, index=X_train_df.index)
        else:
            y_train = y_train.fillna(y_train.mean())

        n_samples = X_train_encoded.shape[0]
        if n_samples == 0:
            return forest
        
        # 修复: 直接计算 n_samples_bootstrap
        if forest.max_samples is None:
            n_samples_bootstrap = n_samples
        elif isinstance(forest.max_samples, int):
            n_samples_bootstrap = forest.max_samples
        else:  # float
            n_samples_bootstrap = int(forest.max_samples * n_samples)
        
        tree_scores = []
        for tree in forest.estimators_:
            try:
                oob_indices = _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)
                
                if len(oob_indices) == 0:
                    tree_scores.append(0.0)
                    continue

                X_oob = X_train_encoded.iloc[oob_indices]
                y_oob = y_train.iloc[oob_indices]

                if len(y_oob) == 0:
                    tree_scores.append(0.0)
                    continue
                
                if is_classifier:
                    if len(np.unique(y_oob)) <= 1:
                        tree_scores.append(0.0)
                        continue
                    y_pred_proba = tree.predict_proba(X_oob)
                    score = roc_auc_score(y_oob, y_pred_proba, multi_class='ovr', average='macro', labels=forest.classes_)
                else:
                    y_pred = tree.predict(X_oob)
                    score = r2_score(y_oob, y_pred)
                    score = max(0, score)
            except Exception as e:
                score = 0.0
            tree_scores.append(score)

        n_trees_keep = max(1, int(self.beta * len(forest.estimators_)))
        top_indices = np.argsort(tree_scores)[-n_trees_keep:]

        if is_classifier:
            pruned = RandomForestClassifier(n_estimators=n_trees_keep, random_state=self.random_state)
        else:
            pruned = RandomForestRegressor(n_estimators=n_trees_keep, random_state=self.random_state)

        pruned.estimators_ = [forest.estimators_[i] for i in top_indices]
        pruned.n_estimators = n_trees_keep
        for attr in ["classes_", "n_classes_", "n_features_in_", "feature_names_in_"]:
            if hasattr(forest, attr):
                setattr(pruned, attr, getattr(forest, attr))
        
        if hasattr(pruned, 'oob_score_'):
            pruned.oob_score_ = None
            
        return pruned

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'RFOD':
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
        self.feature_names_ = list(X.columns)
        self.n_features_ = len(self.feature_names_)
        
        if self.verbose:
            print(f"[RFOD] 开始训练, 样本数: {len(X)}, 特征数: {self.n_features_}")
            
        self.feature_types_ = self._identify_feature_types(X)
        self._fit_encoders(X)
        
        if self.verbose:
            n_numeric = sum(1 for t in self.feature_types_.values() if t == 'numeric')
            n_categorical = self.n_features_ - n_numeric
            print(f"[RFOD] 特征类型: {n_numeric} 个数值型, {n_categorical} 个类别型")
            
        self.quantiles_ = self._compute_quantiles(X)
        
        if self.verbose:
            print("[RFOD] 训练特征专属随机森林...")
            
        for feature_idx in range(self.n_features_):
            if self.verbose:
                print(f"  训练特征 {feature_idx+1}/{self.n_features_}: {self.feature_names_[feature_idx]} ({self.feature_types_[feature_idx]})")
            
            forest = self._train_feature_forest(X, feature_idx)
            
            if self.beta < 1.0:
                forest = self._prune_forest(forest, X, feature_idx)
                
            self.forests_[feature_idx] = forest
            
        if self.verbose:
            print("[RFOD] 训练完成。")
        return self

    def _predict_feature(self, X: pd.DataFrame, feature_idx: int, batch_size: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        """预测单个特征的值及其不确定性"""
        forest = self.forests_[feature_idx]
        X_input_df = X.drop(X.columns[feature_idx], axis=1)
        X_input_encoded = self._transform_data(X_input_df)
        
        n_samples = X_input_encoded.shape[0]
        
        if isinstance(forest, RandomForestClassifier):
            n_classes = len(forest.classes_)
            sum_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
            sum_sq_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                all_tree_probs = np.array([tree.predict_proba(X_batch) for tree in forest.estimators_])
                mean_probs_batch = all_tree_probs.mean(axis=0)
                std_probs_batch = all_tree_probs.std(axis=0)
                sum_probs[start:end] = mean_probs_batch
                sum_sq_probs[start:end] = std_probs_batch
            
            uncertainties = sum_sq_probs.max(axis=1)
            return sum_probs, uncertainties
            
        else:
            predictions = np.zeros(n_samples, dtype=np.float64)
            std_devs = np.zeros(n_samples, dtype=np.float64)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                preds = np.array([tree.predict(X_batch) for tree in forest.estimators_])
                mean_batch = preds.mean(axis=0)
                std_batch = preds.std(axis=0)
                predictions[start:end] = mean_batch
                std_devs[start:end] = std_batch
                
            return predictions, std_devs

    def _compute_cell_scores(self, X: pd.DataFrame, predictions: Dict[int, np.ndarray]) -> np.ndarray:
        """计算 AGD 作为单元格异常分数"""
        n_samples = len(X)
        cell_scores = np.zeros((n_samples, self.n_features_))
        
        for feature_idx in range(self.n_features_):
            true_values_series = X.iloc[:, feature_idx]
            pred_values = predictions[feature_idx]
            
            if self.feature_types_[feature_idx] == 'numeric':
                q_low, q_high = self.quantiles_.get(feature_idx, (0.0, 1.0))
                denom = (q_high - q_low) if (q_high - q_low) != 0 else 1.0
                true_values_filled = true_values_series.fillna(np.mean(pred_values)).values.astype(float)
                pred_values_filled = np.nan_to_num(pred_values, nan=np.mean(pred_values)).astype(float)
                diff = np.abs(true_values_filled - pred_values_filled)
                cell_scores[:, feature_idx] = diff / denom
                
            else:
                forest = self.forests_[feature_idx]
                classes = getattr(forest, "classes_", None)
                if classes is None:
                    continue
                    
                target_col_name = self.feature_names_[feature_idx]
                le = self.encoders_.get(target_col_name)
                if le is None:
                    continue
                
                true_values_str = true_values_series.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~true_values_str.isin(le.classes_)
                true_values_str.loc[unseen_mask] = le.classes_[0]
                true_values_encoded = le.transform(true_values_str)
                true_values_encoded[unseen_mask] = -1
                
                for i in range(n_samples):
                    true_class_encoded = true_values_encoded[i]
                    prob = 0.0
                    try:
                        idx = np.where(classes == true_class_encoded)[0]
                        if len(idx) > 0:
                            prob = pred_values[i, idx[0]]
                        else:
                            prob = 0.0
                    except Exception:
                        prob = 0.0
                    cell_scores[i, feature_idx] = 1.0 - prob
                    
        return cell_scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测异常分数"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
            
        n_samples = len(X)
        if self.verbose:
            print(f"[RFOD] 开始预测 {n_samples} 个样本...")
            
        predictions = {}
        uncertainties = {}
        
        for feature_idx in range(self.n_features_):
            pred, uncert = self._predict_feature(X, feature_idx)
            predictions[feature_idx] = pred
            uncertainties[feature_idx] = uncert
            
        cell_scores = self._compute_cell_scores(X, predictions)
        uncertainty_matrix = np.column_stack([uncertainties[i] for i in range(self.n_features_)])
        row_sums = uncertainty_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        uncertainty_norm = uncertainty_matrix / row_sums
        weights = 1.0 - uncertainty_norm
        weighted_scores = weights * cell_scores
        row_scores = weighted_scores.mean(axis=1)
        
        if self.verbose:
            print(f"[RFOD] 预测完成，row_scores 范围: [{row_scores.min():.6f}, {row_scores.max():.6f}]")
            
        if return_cell_scores:
            return row_scores, cell_scores
        else:
            return row_scores

    def fit_predict(self, X_train: Union[pd.DataFrame, np.ndarray], X_test: Union[pd.DataFrame, np.ndarray], return_cell_scores: bool = False):
        self.fit(X_train)
        return self.predict(X_test, return_cell_scores=return_cell_scores)


# Pipeline 相关代码
REQ_FEATURES = [
    "timestamp", "processId", "parentProcessId", "userId", "mountNamespace",
    "processName", "hostName", "eventName", "argsNum", "returnValue", "stack_depth"
]

def _safe_clean_csv(input_path: str, process_args: bool = True) -> pd.DataFrame:
    """调用 clean_csv 返回 cleaned DataFrame"""
    tmp_out = os.path.join(tempfile.gettempdir(), f"cleaned_{os.path.basename(input_path)}")
    df = clean_csv(input_path, tmp_out, process_args=process_args, save=False)
    return df

def _select_and_align_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """选取特征，缺失则补 NaN"""
    out = pd.DataFrame()
    for f in feature_names:
        if f in df.columns:
            out[f] = df[f]
        else:
            out[f] = np.nan
            print(f"⚠️ 特征缺失，已用 NaN 填充: {f}")
    return out

def train_validate_pipeline(train_csv: str, valid_csv: str, process_args: bool = True,
                            threshold: Optional[float] = None, threshold_percentile: int = 95,
                            verbose: bool = True, drop_labelled_anomalies: bool = False,
                            param_grid: Optional[Dict[str, List]] = None) -> Dict[str, float]:
    
    print(f"--> 清洗训练集: {train_csv}")
    df_train = _safe_clean_csv(train_csv, process_args=process_args)
    if df_train.empty:
        print(f"错误: 无法加载或清洗 {train_csv}")
        return {}
        
    if drop_labelled_anomalies and "target" in df_train.columns:
        before = len(df_train)
        df_train = df_train[df_train["target"].astype(str) != "1"]
        print(f"  已移除 {before - len(df_train)} 个标注为异常的训练样本")
    X_train = _select_and_align_features(df_train, REQ_FEATURES)
    if X_train.empty:
        print("错误: 训练数据为空或所有特征缺失。")
        return {}

    print(f"--> 清洗验证集: {valid_csv}")
    df_valid = _safe_clean_csv(valid_csv, process_args=process_args)
    if df_valid.empty:
        print(f"错误: 无法加载或清洗 {valid_csv}")
        return {}
        
    if "target" not in df_valid.columns:
        print("警告: 验证集必须包含 'target' 列。")
        df_valid["target"] = 0
        
    X_valid = _select_and_align_features(df_valid, REQ_FEATURES)
    y_true = df_valid["target"].astype(int).values
    if X_valid.empty:
        print("错误: 验证数据为空。")
        return {}

    best_acc = -1.0
    best_params = None
    best_model = None
    best_thr = None
    best_preds = None
    best_report = None
    best_cm = None

    if param_grid is None:
        print("--> 使用默认参数训练 RFOD")
        rfod = RFOD(verbose=verbose)
        rfod.fit(X_train)
        train_scores = rfod.predict(X_train)
        if threshold is None:
            thr = float(np.percentile(train_scores, threshold_percentile))
            print(f"--> 使用训练集 {threshold_percentile}th 百分位作为阈值: {thr:.6f}")
        else:
            thr = float(threshold)
            print(f"--> 使用用户指定阈值: {thr:.6f}")
        valid_scores = rfod.predict(X_valid)
        preds = (valid_scores > thr).astype(int)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        report = classification_report(y_true, preds, digits=4, zero_division=0)
        best_acc = acc
        best_params = rfod.__dict__
        best_model = rfod
        best_thr = thr
        best_preds = preds
        best_report = report
        best_cm = cm
    else:
        print("--> 开始网格搜索...")
        keys, values = zip(*param_grid.items())
        combinations = list(itertools.product(*values))
        n_combos = len(combinations)
        print(f"  总参数组合数: {n_combos}")
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            print(f"  [{idx}/{n_combos}] 测试参数: {params}")
            rfod = RFOD(**params, verbose=verbose)
            rfod.fit(X_train)
            train_scores = rfod.predict(X_train)
            if threshold is None:
                thr = float(np.percentile(train_scores, threshold_percentile))
            else:
                thr = float(threshold)
            valid_scores = rfod.predict(X_valid)
            preds = (valid_scores > thr).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_params = params
                best_model = rfod
                best_thr = thr
                best_preds = preds
                best_cm = confusion_matrix(y_true, preds)
                best_report = classification_report(y_true, preds, digits=4, zero_division=0)
        print(f"--> 最佳参数: {best_params}")
        print(f"--> 最佳 accuracy: {best_acc:.6f}")

    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    
    save_data = {'model': best_model, 'threshold': best_thr}
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"--> 已保存最佳模型和阈值到 '{model_path}'")

    print("\n=== 验证结果 ===")
    print(f"验证集样本数: {len(y_true)}")
    print(f"Accuracy: {best_acc:.6f}")
    print("Confusion matrix:\n", best_cm)
    print("Classification report:\n", best_report)

    return {"accuracy": best_acc, "threshold": best_thr, "n_valid": len(y_true)}

if __name__ == "__main__":
    
    train_csv = "data/processes_train.csv"
    valid_csv = "data/processes_valid.csv"
    
    param_grid = {
        "alpha": [0.01],
        "beta": [0.7],
        "n_estimators": [30],
        "max_depth": [10]
    }
    
    res = train_validate_pipeline(
        train_csv=train_csv,
        valid_csv=valid_csv,
        process_args=False,
        threshold=None,
        threshold_percentile=99,
        verbose=True,
        drop_labelled_anomalies=False,
        param_grid=param_grid
    )
    print("\nDone. Summary:", res)