"""
模型训练模块（回归版本）
根据任务书要求，预测未来具体价格（回归任务）
使用MSE、MAE、R²作为评估指标
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, List
import time
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, mean_absolute_percentage_error)
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


def train_and_evaluate_model_regression(model, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_test: pd.DataFrame, y_test: pd.Series,
                                       param_grid: dict, n_iter: int = 10, 
                                       cv: int = 3, model_name: str = 'Model') -> Tuple:
    """
    训练回归模型并评估
    
    根据任务书要求，使用以下回归评估指标：
    - MSE (均方误差)
    - MAE (平均绝对误差)
    - R² (决定系数)
    - MAPE (平均绝对百分比误差)
    
    Args:
        model: 未初始化的回归模型对象
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
    print(f"训练 {model_name} (回归)...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # 时间序列交叉验证
    tscv = create_time_series_split(X_train, n_splits=cv)
    
    # 随机搜索调参（回归任务使用neg_mean_squared_error）
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_squared_error',  # 回归任务使用MSE
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
    
    # 计算回归评估指标（任务书要求）
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)  # RMSE更直观
    
    # MAPE（避免除零错误）
    mask = y_test != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    else:
        mape = float('inf')
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{model_name} 回归评估结果:")
    print(f"  MSE  (均方误差):           {metrics['mse']:.4f}")
    print(f"  RMSE (均方根误差):         {metrics['rmse']:.4f}")
    print(f"  MAE  (平均绝对误差):       {metrics['mae']:.4f}")
    print(f"  R²   (决定系数):           {metrics['r2']:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {metrics['mape']:.4%}")
    print(f"  训练耗时: {elapsed_time:.2f}秒")
    
    return best_model, metrics, best_params


def get_lightgbm_regressor_params():
    """获取LightGBM回归器的参数搜索空间"""
    from lightgbm import LGBMRegressor
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    model = LGBMRegressor(random_state=42, verbose=-1)
    return model, param_grid


def get_random_forest_regressor_params():
    """获取Random Forest回归器的参数搜索空间"""
    from sklearn.ensemble import RandomForestRegressor
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    model = RandomForestRegressor(random_state=42)
    return model, param_grid


def get_xgboost_regressor_params():
    """获取XGBoost回归器的参数搜索空间"""
    try:
        from xgboost import XGBRegressor
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        model = XGBRegressor(random_state=42, verbosity=0)
        return model, param_grid
    except ImportError:
        print("警告: XGBoost未安装，跳过该模型")
        return None, None


def train_all_models_regression(X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                stock_code: str = '', horizon: int = 1) -> Dict:
    """
    训练所有回归模型并返回结果
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        stock_code: 股票代码
        horizon: 预测时间窗口
    
    Returns:
        results: 包含所有模型结果的字典
    """
    results = {}
    
    # 定义模型配置
    models_config = [
        ('LightGBM', get_lightgbm_regressor_params),
        ('RandomForest', get_random_forest_regressor_params),
        ('XGBoost', get_xgboost_regressor_params)
    ]
    
    for model_name, get_params_func in models_config:
        try:
            model, param_grid = get_params_func()
            if model is None:
                continue
            
            print(f"\n{'#'*60}")
            print(f"# 训练 {model_name} - {stock_code} - {horizon}天")
            print(f"{'#'*60}")
            
            best_model, metrics, best_params = train_and_evaluate_model_regression(
                model, X_train, y_train, X_test, y_test,
                param_grid, n_iter=10, cv=3, model_name=model_name
            )
            
            results[model_name] = {
                'model': best_model,
                'metrics': metrics,
                'best_params': best_params
            }
            
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def save_model(model, filepath: str):
    """保存模型到文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ 模型已保存: {filepath}")


def load_model(filepath: str):
    """从文件加载模型"""
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"✓ 模型已加载: {filepath}")
        return model
    else:
        raise FileNotFoundError(f"模型文件不存在: {filepath}")


def predict_with_model(model, features: pd.DataFrame) -> np.ndarray:
    """
    使用模型进行预测
    
    Args:
        model: 训练好的回归模型
        features: 特征数据
    
    Returns:
        predictions: 预测的价格值
    """
    return model.predict(features)


if __name__ == '__main__':
    # 测试代码
    print("回归模型训练模块测试")
    print("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    y = pd.Series(100 + X['feature1'] * 10 + np.random.randn(n_samples) * 5)
    
    # 划分数据集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 测试LightGBM回归器
    model, param_grid = get_lightgbm_regressor_params()
    best_model, metrics, best_params = train_and_evaluate_model_regression(
        model, X_train, y_train, X_test, y_test,
        param_grid, n_iter=5, cv=2, model_name='LightGBM_Test'
    )
    
    print("\n✅ 回归模型训练模块测试通过！")
