"""
模型训练模块
训练LightGBM、Random Forest、XGBoost模型并进行评估
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, List
import time
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')


def create_time_series_split(X: pd.DataFrame, n_splits: int = 3):
    """
    创建时间序列交叉验证器
    
    Args:
        X: 特征数据
        n_splits: 折数
    
    Returns:
        TimeSeriesSplit对象
    """
    return TimeSeriesSplit(n_splits=n_splits)


def train_and_evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             param_grid: dict, n_iter: int = 10, 
                             cv: int = 3, model_name: str = 'Model') -> Tuple:
    """
    训练模型并评估
    
    Args:
        model: 未初始化的模型对象
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        param_grid: 超参数网格
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
        model_name: 模型名称
    
    Returns:
        best_model: 最优模型
        metrics: 评估指标字典
        best_params: 最优参数
    """
    print(f"\n{'='*60}")
    print(f"训练 {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 时间序列交叉验证
    tscv = create_time_series_split(X_train, n_splits=cv)
    
    # 随机搜索调参
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='roc_auc',
        random_state=42,
        verbose=1,
        n_jobs=-1,  # 使用所有CPU核心
        error_score='raise'
    )
    
    print("开始超参数调优...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"最优参数: {best_params}")
    
    # 预测
    print("进行预测...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_name} 评估结果:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision): {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1']:.4f}")
    print(f"  AUC-ROC:            {metrics['auc']:.4f}")
    print(f"  训练用时:           {elapsed_time:.2f} 秒")
    print(f"{'='*60}\n")
    
    return best_model, metrics, best_params


def get_lightgbm_model():
    """获取LightGBM分类器"""
    import lightgbm as lgb
    
    return lgb.LGBMClassifier(
        objective='binary',
        metric=['auc', 'binary_logloss'],
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )


def get_random_forest_model():
    """获取Random Forest分类器"""
    from sklearn.ensemble import RandomForestClassifier
    
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


def get_xgboost_model():
    """获取XGBoost分类器"""
    import xgboost as xgb
    
    return xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['auc', 'logloss'],
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    )


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series,
                     param_grids: dict, n_iter: int = 10,
                     cv: int = 3) -> Dict:
    """
    训练所有模型并返回结果
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        param_grids: 各模型的超参数网格
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
    
    Returns:
        results: 包含模型、指标、参数的字典
    """
    results = {}
    
    # 定义模型配置
    models_config = [
        ('lightgbm', get_lightgbm_model(), param_grids.get('lightgbm', {})),
        ('random_forest', get_random_forest_model(), param_grids.get('random_forest', {})),
        ('xgboost', get_xgboost_model(), param_grids.get('xgboost', {}))
    ]
    
    for model_name, model, param_grid in models_config:
        if not param_grid:
            print(f"跳过 {model_name}: 没有提供超参数网格")
            continue
        
        try:
            best_model, metrics, best_params = train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test,
                param_grid, n_iter=n_iter, cv=cv,
                model_name=model_name.upper()
            )
            
            results[model_name] = {
                'model': best_model,
                'metrics': metrics,
                'params': best_params
            }
        except Exception as e:
            print(f"错误: {model_name} 训练失败 - {str(e)}")
            results[model_name] = None
    
    return results


def save_model(model, filepath: str):
    """保存模型到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(filepath: str):
    """从文件加载模型"""
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"模型文件不存在: {filepath}")


def train_stock_models(stock_code: str, stock_name: str,
                       features_dict: dict, labels_dict: dict,
                       param_grids: dict, model_save_dir: str = 'models',
                       n_iter: int = 10, cv: int = 3) -> Dict:
    """
    为单只股票训练所有时间窗口的模型
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        features_dict: {horizon: features_df}
        labels_dict: {horizon: labels_series}
        param_grids: 超参数网格
        model_save_dir: 模型保存目录
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
    
    Returns:
        all_results: 所有模型的训练结果
    """
    print(f"\n{'#'*70}")
    print(f"# 开始训练 {stock_name} ({stock_code}) 的模型")
    print(f"{'#'*70}\n")
    
    all_results = {}
    
    for horizon in [1, 5, 10]:
        print(f"\n{'='*70}")
        print(f"训练 {horizon}天预测模型")
        print(f"{'='*70}")
        
        features = features_dict[horizon]
        labels = labels_dict[horizon]
        
        # 按时间划分训练集和测试集（80%训练，20%测试）
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"标签分布 - 训练集: 涨={y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
        print(f"标签分布 - 测试集: 涨={y_test.sum()}/{len(y_test)} ({y_test.mean():.2%})")
        
        # 训练所有模型
        results = train_all_models(
            X_train, y_train, X_test, y_test,
            param_grids, n_iter=n_iter, cv=cv
        )
        
        # 保存模型
        for model_name, result in results.items():
            if result is not None:
                model_filename = f"stock_{stock_code.replace('.', '_')}_{model_name}_{horizon}day.pkl"
                model_path = os.path.join(model_save_dir, model_filename)
                save_model(result['model'], model_path)
                
                # 添加到结果
                if horizon not in all_results:
                    all_results[horizon] = {}
                all_results[horizon][model_name] = {
                    'metrics': result['metrics'],
                    'params': result['params'],
                    'model_path': model_path
                }
    
    return all_results


if __name__ == '__main__':
    # 测试代码
    from config import STOCK_CODES, MODEL_CONFIG, MODEL_PARAMS
    from data_fetch import fetch_stock_data, preprocess_data, DATA_CONFIG
    from feature_engineering import prepare_features_and_labels, FEATURE_CONFIG
    
    print("="*70)
    print("测试模型训练模块")
    print("="*70)
    
    # 获取一只股票的数据
    stock_code = list(STOCK_CODES.keys())[0]
    stock_name = STOCK_CODES[stock_code]
    
    print(f"\n处理 {stock_name} ({stock_code})")
    df = fetch_stock_data(stock_code, DATA_CONFIG['start_date'], 
                         DATA_CONFIG['end_date'])
    df = preprocess_data(df)
    
    # 准备特征和标签
    features_dict = {}
    labels_dict = {}
    
    for horizon in [1, 5, 10]:
        print(f"\n准备 {horizon}天预测的特征和标签...")
        features, labels = prepare_features_and_labels(df, FEATURE_CONFIG, horizon)
        features_dict[horizon] = features
        labels_dict[horizon] = labels
    
    # 训练模型
    print("\n开始训练模型...")
    results = train_stock_models(
        stock_code, stock_name,
        features_dict, labels_dict,
        MODEL_PARAMS,
        n_iter=MODEL_CONFIG['n_iter_search'],
        cv=MODEL_CONFIG['cv_folds']
    )
    
    # 显示结果汇总
    print("\n" + "="*70)
    print("训练结果汇总")
    print("="*70)
    
    for horizon, horizon_results in results.items():
        print(f"\n{horizon}天预测:")
        for model_name, model_result in horizon_results.items():
            metrics = model_result['metrics']
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}, "
                  f"F1: {metrics['f1']:.4f}, "
                  f"AUC: {metrics['auc']:.4f}")
