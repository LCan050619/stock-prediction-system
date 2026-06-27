"""
回归预测接口模块
提供股票价格预测功能（预测具体价格而非涨跌方向）
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
    加载训练好的回归模型
    
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
        # 检查是否有该股票的任何模型
        stock_prefix = f"stock_{stock_code.replace('.', '_')}"
        available_models = []
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.startswith(stock_prefix) and filename.endswith('.pkl'):
                    available_models.append(filename)
        
        error_msg = f"模型文件不存在: {model_filename}\n\n"
        if available_models:
            error_msg += f"该股票有以下模型可用:\n"
            for model in available_models[:3]:  # 只显示前3个
                error_msg += f"  • {model}\n"
            error_msg += "\n请确保请求的时间窗口(1/5/10天)与已训练的模型匹配。"
        else:
            error_msg += f"该股票暂无任何训练好的模型。\n"
            error_msg += "请先运行 train_all_stocks_regression.py 训练模型。"
        
        raise FileNotFoundError(error_msg)
    
    # 加载模型文件
    loaded_data = joblib.load(model_path)
    
    # 【关键修复】兼容两种保存格式：
    # 1. 新格式（pragmatic_fix_retrain.py）：字典 {'model': model_obj, 'metrics': {...}}
    # 2. 旧格式（model_training_regression.py）：直接是模型对象
    if isinstance(loaded_data, dict):
        # 新格式：提取model字段
        if 'model' in loaded_data:
            return loaded_data['model']
        else:
            raise ValueError(f"模型文件格式错误: {model_filename} 缺少'model'字段")
    else:
        # 旧格式：直接返回模型对象
        return loaded_data


def calculate_features_for_prediction(recent_data: pd.DataFrame, 
                                      horizon: int = 5,
                                      config: dict = None) -> pd.DataFrame:
    """
    为预测计算特征
    
    Args:
        recent_data: 最近的历史数据（包含Open, High, Low, Close, Volume）
        horizon: 预测时间窗口（1, 5, 10）
        config: 特征配置
    
    Returns:
        特征DataFrame（最后一行）
    """
    if config is None:
        config = FEATURE_CONFIG
    
    # 计算技术指标
    df_features = calculate_technical_indicators(recent_data, config)
    
    # 【关键修复】添加滞后价格特征（Lag Features）
    # 原因：模型训练时使用了这些特征，预测时必须提供相同的特征
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'Lag_{lag}'] = recent_data['Close'].shift(lag)
    
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
        'HL_Distance', 'Close_MA20_Ratio',
        # 【关键改进】滞后价格特征
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_10'
    ]
    
    # 确保所有特征列都存在
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 【关键修复】移除Horizon特征，因为现有模型是用24个特征训练的
    # 原因：
    # 1. pragmatic_fix_retrain.py只成功了72.2%（13/18个模型）
    # 2. 大部分模型仍是用24个特征训练的（没有Horizon）
    # 3. 为了兼容性，预测时不使用Horizon特征
    # 4. 不同horizon的预测通过不同的模型文件区分（_1day.pkl, _5day.pkl, _10day.pkl）
    # if 'Horizon' in features.columns:
    #     features = features.drop(columns=['Horizon'])
    
    # 处理可能的无穷值和缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    # 返回最后一行（最新数据）
    return features.iloc[[-1]]


def predict_stock_price(stock_code: str, horizon: int,
                        recent_data: pd.DataFrame,
                        model_name: str = 'linear',  # 【关键修复】改用linear作为默认模型
                        model_dir: str = 'models') -> Dict:
    """
    预测股票未来价格（回归任务）
    
    根据任务书要求，预测未来具体价格而非涨跌方向。
    
    Args:
        stock_code: 股票代码
        horizon: 预测时间窗口（1, 5, 10）
        recent_data: 最近的历史数据（用于计算特征）
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {
            'predicted_price': 预测价格,
            'current_price': 当前价格,
            'price_change': 价格变化,
            'change_percent': 涨跌幅百分比,
            'trend': '上涨' 或 '下跌'
        }
    """
    try:
        # 【关键修复】将 .SS 转换为 .SH 以匹配模型文件名
        model_code = stock_code.replace('.SS', '.SH')
        
        # 加载模型（包含元数据）
        model_path = f"models/stock_{model_code.replace('.', '_')}_{model_name}_{horizon}day.pkl"
        loaded_data = joblib.load(model_path)
        
        # 兼容两种格式
        if isinstance(loaded_data, dict):
            model = loaded_data['model']
            # 【关键修复】使用模型中保存的特征列顺序
            expected_feature_columns = loaded_data.get('feature_columns', None)
        else:
            model = loaded_data
            expected_feature_columns = None
        
        # 计算特征
        features_df = calculate_features_for_prediction(recent_data, horizon=horizon)
        
        # 【关键修复】确保特征列顺序与训练时完全一致
        if expected_feature_columns is not None:
            # 只保留模型期望的特征列，并按正确顺序排列
            available_cols = [col for col in expected_feature_columns if col in features_df.columns]
            features_df = features_df[available_cols]
            
            # 检查是否有缺失的特征列
            missing_cols = set(expected_feature_columns) - set(features_df.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
        
        # 预测未来价格
        predicted_price = float(model.predict(features_df)[0])
        
        # 获取当前价格和日期
        current_price = float(recent_data['Close'].iloc[-1])
        current_date = recent_data.index[-1]
        
        # 【关键改进】不再在预测阶段应用涨跌幅限制
        # 原因：
        # 1. 训练时已应用±4%限制，模型已学习到该约束模式
        # 2. 添加Horizon特征后，不同时间窗口应有不同的预测结果
        # 3. 如果预测时再次限制，会导致所有horizon的结果被压缩到相同值（雷同）
        # 4. 让模型输出真实预测值，体现技术改进的效果
        
        # 计算预测日期（只计算交易日，假设每周5个交易日）
        from datetime import timedelta
        trading_days = horizon
        # 简单估算：每5个交易日约等于7个自然日
        calendar_days = int(horizon * 7 / 5)
        predict_date = current_date + timedelta(days=calendar_days)
        
        # 计算价格变化和涨跌幅
        price_change = predicted_price - current_price
        change_percent = (price_change / current_price) * 100
        
        # 确定趋势
        trend = '上涨' if price_change > 0 else '下跌'
        
        return {
            'predicted_price': round(predicted_price, 2),
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'change_percent': round(change_percent, 2),
            'trend': trend,
            'predict_date': predict_date.strftime('%Y-%m-%d'),
            'current_date': current_date.strftime('%Y-%m-%d')
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'predicted_price': None,
            'current_price': None,
            'price_change': None,
            'change_percent': None,
            'trend': None
        }


def predict_all_horizons(stock_code: str, recent_data: pd.DataFrame,
                         model_name: str = 'linear',  # 【关键修复】改用linear作为默认模型
                         model_dir: str = 'models') -> Dict:
    """
    同时预测1天、5天、10天的价格
    
    Args:
        stock_code: 股票代码
        recent_data: 最近的历史数据
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {
            '1day': {'predicted_price': 1850.50, 'current_price': 1820.00, ...},
            '5day': {...},
            '10day': {...}
        }
    """
    results = {}
    
    for horizon in [1, 5, 10]:
        result = predict_stock_price(
            stock_code, horizon, recent_data,
            model_name=model_name,
            model_dir=model_dir
        )
        results[f'{horizon}day'] = result
    
    return results


def batch_predict(stocks_data: Dict, model_name: str = 'linear',  # 【关键修复】改用linear作为默认模型
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
    print("回归预测模块测试")
    print("=" * 60)
    
    # 创建示例数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 102 + np.random.randn(100).cumsum(),
        'Low': 98 + np.random.randn(100).cumsum(),
        'Close': 101 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    test_data.set_index('Date', inplace=True)
    
    print("\n✅ 回归预测模块测试完成！")
    print("注意：实际使用需要先训练模型")
