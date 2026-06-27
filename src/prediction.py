"""
预测接口模块
提供股票涨跌预测功能
"""
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional
from feature_engineering import calculate_technical_indicators
from config import FEATURE_CONFIG, STOCK_CODES


def load_model(stock_code: str, model_name: str, horizon: int,
               model_dir: str = 'models'):
    """
    加载训练好的模型
    
    Args:
        stock_code: 股票代码
        model_name: 模型名称（lightgbm, random_forest, xgboost）
        horizon: 预测时间窗口
        model_dir: 模型目录
    
    Returns:
        加载的模型对象
    """
    model_filename = f"stock_{stock_code.replace('.', '_')}_{model_name}_{horizon}day.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return joblib.load(model_path)


def calculate_features_for_prediction(recent_data: pd.DataFrame, 
                                      config: dict = None) -> pd.DataFrame:
    """
    为预测计算特征
    
    Args:
        recent_data: 最近的历史数据（包含Open, High, Low, Close, Volume）
        config: 特征配置
    
    Returns:
        特征DataFrame（最后一行）
    """
    if config is None:
        config = FEATURE_CONFIG
    
    # 计算技术指标
    df_features = calculate_technical_indicators(recent_data, config)
    
    # 选择特征列
    feature_columns = [
        'MA5', 'MA10', 'MA20', 'MA60',
        'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'K', 'D', 'J',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'Volume', 'Volume_MA5', 'OBV',
        'Return_1d', 'Return_5d', 'Return_10d',
        'Volatility',
        'HL_Distance', 'Close_MA20_Ratio'
    ]
    
    # 确保所有特征列都存在
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 处理可能的无穷值和缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    # 返回最后一行（最新数据）
    return features.iloc[[-1]]


def predict_stock_direction(stock_code: str, horizon: int,
                            recent_data: pd.DataFrame,
                            model_name: str = 'lightgbm',
                            model_dir: str = 'models') -> Dict:
    """
    预测股票未来涨跌
    
    Args:
        stock_code: 股票代码
        horizon: 预测时间窗口（1, 5, 10）
        recent_data: 最近的历史数据（用于计算特征）
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {
            'direction': '涨' 或 '跌',
            'probability': 上涨概率（0-1）,
            'confidence': 置信度描述（高/中/低）
        }
    """
    try:
        # 加载模型
        model = load_model(stock_code, model_name, horizon, model_dir)
        
        # 计算特征
        features = calculate_features_for_prediction(recent_data)
        
        # 预测
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # 上涨概率
        
        # 确定置信度
        if probability > 0.7 or probability < 0.3:
            confidence = '高'
        elif probability > 0.6 or probability < 0.4:
            confidence = '中'
        else:
            confidence = '低'
        
        return {
            'direction': '涨' if prediction == 1 else '跌',
            'probability': round(float(probability), 4),
            'confidence': confidence
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'direction': None,
            'probability': None,
            'confidence': None
        }


def predict_all_horizons(stock_code: str, recent_data: pd.DataFrame,
                         model_name: str = 'lightgbm',
                         model_dir: str = 'models') -> Dict:
    """
    同时预测1天、5天、10天的涨跌
    
    Args:
        stock_code: 股票代码
        recent_data: 最近的历史数据
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {
            '1day': {'direction': '涨', 'probability': 0.65, 'confidence': '中'},
            '5day': {...},
            '10day': {...}
        }
    """
    results = {}
    
    for horizon in [1, 5, 10]:
        result = predict_stock_direction(
            stock_code, horizon, recent_data,
            model_name=model_name,
            model_dir=model_dir
        )
        results[f'{horizon}day'] = result
    
    return results


def batch_predict(stocks_data: Dict, model_name: str = 'lightgbm',
                  model_dir: str = 'models') -> Dict:
    """
    批量预测多只股票
    
    Args:
        stocks_data: {stock_code: recent_data_df}
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {stock_code: {horizon: prediction_result}}
    """
    all_predictions = {}
    
    for stock_code, recent_data in stocks_data.items():
        predictions = predict_all_horizons(
            stock_code, recent_data,
            model_name=model_name,
            model_dir=model_dir
        )
        all_predictions[stock_code] = predictions
    
    return all_predictions


if __name__ == '__main__':
    # 测试代码
    from data_fetch import fetch_stock_data, DATA_CONFIG
    from config import STOCK_CODES
    
    print("测试预测接口...")
    
    # 获取一只股票的最近数据
    stock_code = list(STOCK_CODES.keys())[0]
    stock_name = STOCK_CODES[stock_code]
    
    print(f"\n测试 {stock_name} ({stock_code})")
    
    # 获取最近3个月的数据用于预测
    ticker_data = fetch_stock_data(stock_code, '2024-10-01', '2025-01-01')
    
    # 预测所有时间窗口
    print("\n进行预测...")
    results = predict_all_horizons(stock_code, ticker_data)
    
    # 显示结果
    print(f"\n{stock_name} 预测结果:")
    print("-" * 60)
    for horizon, result in results.items():
        if 'error' in result:
            print(f"{horizon}: 错误 - {result['error']}")
        else:
            print(f"{horizon}: {result['direction']} "
                  f"(概率: {result['probability']:.2%}, "
                  f"置信度: {result['confidence']})")
    print("-" * 60)
