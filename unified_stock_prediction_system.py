"""
股票价格预测系统 - 完整实验代码
=====================================

本文件整合了股票预测系统的所有核心代码，包括：
1. 配置模块 - 股票代码、参数设置
2. 数据获取模块 - AkShare/Yahoo Finance双数据源
3. 特征工程模块 - 28种技术指标计算
4. 模型训练模块 - Linear/Ridge/Lasso回归
5. 预测接口模块 - 单股/批量预测
6. 可视化模块 - 预测对比、残差分析、模型对比
7. Web应用模块 - Flask RESTful API
8. 数据库模块 - SQLite预测历史记录

作者：课程设计项目组
日期：2026-06-25
版本：v2.0 (Unified)
"""

# ============================================================================
# 第一部分：导入依赖库
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import joblib
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Web框架
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# 数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare未安装")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("警告: yfinance未安装")

# 数据库
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第二部分：配置模块
# ============================================================================

# 股票代码映射（6只已训练的股票）
STOCK_CODES = {
    '600519.SS': '贵州茅台',      # 消费/白酒
    '000001.SZ': '平安银行',      # 金融/银行
    '601318.SS': '中国平安',      # 金融/保险
    '600036.SS': '招商银行',      # 金融/银行
    '300750.SZ': '宁德时代',      # 新能源/电池
    '600887.SS': '伊利股份',      # 消费/乳业
}

# 数据获取配置
DATA_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2025-12-31',
    'cache_dir': 'data/raw'
}

# 特征工程配置
FEATURE_CONFIG = {
    'ma_periods': [5, 10, 20, 60],           # 移动平均线周期
    'rsi_period': 14,                        # RSI周期
    'macd_fast': 12,                         # MACD快线
    'macd_slow': 26,                         # MACD慢线
    'macd_signal': 9,                        # MACD信号线
    'kdj_k': 9,                              # KDJ K值周期
    'kdj_d': 3,                              # KDJ D值周期
    'kdj_j': 3,                              # KDJ J值周期
    'bollinger_window': 20,                  # 布林带窗口
    'bollinger_std': 2.0,                    # 布林带标准差倍数
    'volatility_window': 20                  # 波动率计算窗口
}

# 预测时间窗口（天）
PREDICTION_HORIZONS = [1, 5, 10]

# 模型配置
MODEL_CONFIG = {
    'train_ratio': 0.8,           # 训练集比例
    'random_state': 42            # 随机种子
}


# ============================================================================
# 第三部分：数据获取模块
# ============================================================================

def fetch_from_akshare(stock_code: str, start_date: str, end_date: str, cache_file: str) -> pd.DataFrame:
    """
    从AkShare获取A股历史数据（国内数据源，访问更稳定）
    
    Args:
        stock_code: 股票代码，格式如 '600519.SS'（上海）或 '000001.SZ'（深圳）
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        cache_file: CSV缓存文件路径
    
    Returns:
        DataFrame包含Open, High, Low, Close, Volume列，索引为DatetimeIndex
    """
    if not AKSHARE_AVAILABLE:
        raise Exception("AkShare未安装")
    
    # 提取纯数字代码
    code = stock_code.split('.')[0]
    
    # 限制最大日期，防止系统时间错误
    MAX_DATE = '20251231'
    if end_date.replace('-', '') > MAX_DATE:
        print(f"WARNING: Limiting end_date from {end_date} to 2025-12-31")
        end_date = '2025-12-31'
    
    print(f"使用AkShare下载 {code} 的数据...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            })
            
            # 设置日期索引
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 保存到缓存
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_csv(cache_file)
            print(f"AkShare数据已保存到: {cache_file}")
            
            return df
            
        except Exception as e:
            print(f"AkShare尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise


def fetch_stock_data(stock_code: str, start_date: str, end_date: str, 
                     cache_dir: str = 'data/raw') -> pd.DataFrame:
    """
    获取单只股票的历史数据（优先使用AkShare，失败则使用Yahoo Finance）
    
    Args:
        stock_code: 股票代码（如 '600519.SS'）
        start_date: 开始日期（如 '2022-01-01'）
        end_date: 结束日期（如 '2025-12-31'）
        cache_dir: 缓存目录
    
    Returns:
        DataFrame包含Open, High, Low, Close, Volume
    """
    # 强制限制日期范围
    MAX_REASONABLE_DATE = '2025-12-31'
    try:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        max_dt = datetime.strptime(MAX_REASONABLE_DATE, '%Y-%m-%d')
        if end_dt > max_dt:
            print(f"WARNING: end_date ({end_date}) exceeds reasonable limit!")
            end_date = MAX_REASONABLE_DATE
    except:
        pass
    
    # 构建缓存文件路径
    cache_file = os.path.join(cache_dir, f'{stock_code.replace(".", "_")}.csv')
    
    # 如果缓存存在，直接读取
    if os.path.exists(cache_file):
        print(f"从缓存加载数据: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    # 首先尝试AkShare
    if AKSHARE_AVAILABLE:
        try:
            df = fetch_from_akshare(stock_code, start_date, end_date, cache_file)
            if not df.empty:
                return df
        except Exception as e:
            print(f"AkShare获取失败: {str(e)}，尝试使用Yahoo Finance...")
    
    # 备用方案：Yahoo Finance
    if not YFINANCE_AVAILABLE:
        raise Exception("yfinance未安装且AkShare失败")
    
    print(f"正在下载 {stock_code} 的数据...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index = pd.to_datetime(df.index)
            
            # 保存到缓存
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(cache_file)
            print(f"数据已保存到: {cache_file}")
            
            return df
            
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理：处理缺失值和异常值
    
    Args:
        df: 原始数据DataFrame
    
    Returns:
        预处理后的DataFrame
    """
    df_processed = df.copy()
    
    # 转换为float64
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_processed[column] = df_processed[column].astype(float)
    
    # 处理缺失值
    df_processed = df_processed.ffill().bfill()
    
    # 处理异常值（IQR方法）
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        Q1 = df_processed[column].quantile(0.25)
        Q3 = df_processed[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df_processed[column].median()
        
        mask = (df_processed[column] < lower_bound) | (df_processed[column] > upper_bound)
        df_processed.loc[mask, column] = median
    
    return df_processed


# ============================================================================
# 第四部分：特征工程模块
# ============================================================================

def calculate_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    计算所有技术指标（28种）
    
    包括：移动平均线、RSI、MACD、KDJ、布林带、成交量指标、价格变化率等
    
    Args:
        df: 包含Open, High, Low, Close, Volume的DataFrame
        config: 特征配置字典
    
    Returns:
        DataFrame包含原始数据+28个技术指标
    """
    df_features = df.copy()
    
    # 1. 移动平均线
    for period in config['ma_periods']:
        df_features[f'MA{period}'] = df_features['Close'].rolling(window=period).mean()
    
    # 2. RSI相对强弱指标
    rsi_period = config['rsi_period']
    delta = df_features['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD平滑异同移动平均线
    macd_fast = config['macd_fast']
    macd_slow = config['macd_slow']
    macd_signal = config['macd_signal']
    
    ema_fast = df_features['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df_features['Close'].ewm(span=macd_slow, adjust=False).mean()
    df_features['MACD'] = ema_fast - ema_slow
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df_features['MACD_Hist'] = df_features['MACD'] - df_features['MACD_Signal']
    
    # 4. KDJ随机指标
    kdj_k = config['kdj_k']
    kdj_d = config['kdj_d']
    kdj_j = config['kdj_j']
    
    low_min = df_features['Low'].rolling(window=kdj_k).min()
    high_max = df_features['High'].rolling(window=kdj_k).max()
    rsv = (df_features['Close'] - low_min) / (high_max - low_min) * 100
    df_features['K'] = rsv.ewm(com=kdj_d-1, adjust=False).mean()
    df_features['D'] = df_features['K'].ewm(com=kdj_j-1, adjust=False).mean()
    df_features['J'] = 3 * df_features['K'] - 2 * df_features['D']
    
    # 5. 布林带
    boll_window = config['bollinger_window']
    boll_std = config['bollinger_std']
    ma = df_features['Close'].rolling(window=boll_window).mean()
    std = df_features['Close'].rolling(window=boll_window).std()
    df_features['BB_Upper'] = ma + (std * boll_std)
    df_features['BB_Middle'] = ma
    df_features['BB_Lower'] = ma - (std * boll_std)
    df_features['BB_Width'] = (df_features['BB_Upper'] - df_features['BB_Lower']) / ma
    
    # 6. 成交量指标
    df_features['Volume_MA5'] = df_features['Volume'].rolling(window=5).mean()
    df_features['OBV'] = (np.sign(df_features['Close'].diff()) * df_features['Volume']).fillna(0).cumsum()
    
    # 7. 价格变化率
    df_features['Return_1d'] = df_features['Close'].pct_change(periods=1)
    df_features['Return_5d'] = df_features['Close'].pct_change(periods=5)
    df_features['Return_10d'] = df_features['Close'].pct_change(periods=10)
    
    # 8. 波动率
    vol_window = config['volatility_window']
    df_features['Volatility'] = df_features['Return_1d'].rolling(window=vol_window).std()
    
    # 9. 高低点距离
    df_features['HL_Distance'] = (df_features['High'] - df_features['Low']) / df_features['Close']
    
    # 10. 收盘价相对于MA的位置
    df_features['Close_MA20_Ratio'] = df_features['Close'] / df_features['MA20']
    
    return df_features


def prepare_features_and_labels_regression(df: pd.DataFrame, config: dict, 
                                            horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备回归任务的特征和标签
    
    Args:
        df: 原始数据DataFrame
        config: 特征配置
        horizon: 预测时间窗口（1/5/10天）
    
    Returns:
        features: 特征DataFrame
        labels: 标签Series（未来horizon天的收盘价）
    """
    # 计算技术指标
    df_features = calculate_technical_indicators(df, config)
    
    # 添加滞后价格特征（Lag Features）
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    # 创建回归标签（未来价格）
    labels = df_features['Close'].shift(-horizon)
    labels = labels.iloc[:-horizon]
    
    # 删除包含NaN的行
    valid_idx = labels.dropna().index
    df_features = df_features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
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
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_10'
    ]
    
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 处理无穷值和缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    return features, labels


# ============================================================================
# 第五部分：模型训练模块
# ============================================================================

def train_all_models():
    """
    训练所有股票的所有模型
    
    训练Linear、Ridge、Lasso三种线性回归模型，针对1天、5天、10天三个时间窗口
    """
    print("="*80)
    print("开始训练所有模型")
    print("="*80)
    
    models_config = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1)
    }
    
    horizons = [1, 5, 10]
    success_count = 0
    total_count = 0
    
    for stock_code, stock_name in STOCK_CODES.items():
        print(f"\n{'='*80}")
        print(f"处理: {stock_name} ({stock_code})")
        print(f"{'='*80}")
        
        # 加载数据
        cache_file = f'data/raw/{stock_code.replace(".", "_").replace(".SS", "_SH")}.csv'
        if not os.path.exists(cache_file):
            print(f"[ERROR] 缓存文件不存在: {cache_file}")
            continue
        
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"加载 {len(df)} 条记录")
        
        for horizon in horizons:
            print(f"\n--- {horizon}天预测 ---")
            
            # 准备特征和标签
            features, labels = prepare_features_and_labels_regression(df, FEATURE_CONFIG, horizon)
            print(f"生成 {len(features)} 个样本，{features.shape[1]} 个特征")
            
            # 划分训练集和测试集（80/20）
            train_size = int(len(features) * MODEL_CONFIG['train_ratio'])
            X_train = features.iloc[:train_size]
            y_train = labels.iloc[:train_size]
            X_test = features.iloc[train_size:]
            y_test = labels.iloc[train_size:]
            
            # 不使用特征缩放（简化预测流程）
            X_train_scaled = X_train
            X_test_scaled = X_test
            
            # 训练每个模型
            for model_name, model in models_config.items():
                total_count += 1
                try:
                    model.fit(X_train_scaled, y_train)
                    
                    # 评估
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # 保存模型（包含元数据）
                    model_path = f'models/stock_{stock_code.replace(".", "_").replace(".SS", "_SH")}_{model_name}_{horizon}day.pkl'
                    save_data = {
                        'model': model,
                        'metrics': {'R2': r2, 'RMSE': rmse, 'MAE': mae},
                        'feature_columns': features.columns.tolist(),
                        'config': FEATURE_CONFIG
                    }
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(save_data, model_path)
                    
                    success_count += 1
                    print(f"  ✓ {model_name:10s}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ {model_name:10s}: 训练失败 - {str(e)}")
    
    print("\n" + "="*80)
    print(f"训练完成！")
    print(f"总模型数:     {total_count}")
    print(f"成功训练:     {success_count}")
    print(f"成功率:       {success_count/total_count*100:.1f}%")
    print("="*80)


# ============================================================================
# 第六部分：预测接口模块
# ============================================================================

def calculate_features_for_prediction(recent_data: pd.DataFrame, 
                                      horizon: int = 5,
                                      config: dict = None) -> pd.DataFrame:
    """
    为预测计算特征
    
    Args:
        recent_data: 最近的历史数据
        horizon: 预测时间窗口
        config: 特征配置
    
    Returns:
        特征DataFrame（最后一行）
    """
    if config is None:
        config = FEATURE_CONFIG
    
    # 计算技术指标
    df_features = calculate_technical_indicators(recent_data, config)
    
    # 添加滞后价格特征
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
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_10'
    ]
    
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 处理无穷值和缺失值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    # 返回最后一行
    return features.iloc[[-1]]


def predict_stock_price(stock_code: str, horizon: int,
                        recent_data: pd.DataFrame,
                        model_name: str = 'linear',
                        model_dir: str = 'models') -> Dict:
    """
    预测股票未来价格（回归任务）
    
    Args:
        stock_code: 股票代码
        horizon: 预测时间窗口（1, 5, 10）
        recent_data: 最近的历史数据
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        包含predicted_price, current_price, price_change, change_percent, trend的字典
    """
    try:
        # 将 .SS 转换为 .SH 以匹配模型文件名
        model_code = stock_code.replace('.SS', '.SH')
        
        # 加载模型
        model_path = f"{model_dir}/stock_{model_code.replace('.', '_')}_{model_name}_{horizon}day.pkl"
        loaded_data = joblib.load(model_path)
        
        # 兼容两种格式
        if isinstance(loaded_data, dict):
            model = loaded_data['model']
            expected_feature_columns = loaded_data.get('feature_columns', None)
        else:
            model = loaded_data
            expected_feature_columns = None
        
        # 计算特征
        features_df = calculate_features_for_prediction(recent_data, horizon=horizon)
        
        # 确保特征列顺序与训练时一致
        if expected_feature_columns is not None:
            available_cols = [col for col in expected_feature_columns if col in features_df.columns]
            features_df = features_df[available_cols]
            
            missing_cols = set(expected_feature_columns) - set(features_df.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
        
        # 预测
        predicted_price = float(model.predict(features_df)[0])
        
        # 获取当前价格
        current_price = float(recent_data['Close'].iloc[-1])
        current_date = recent_data.index[-1]
        
        # 计算预测日期
        calendar_days = int(horizon * 7 / 5)
        predict_date = current_date + timedelta(days=calendar_days)
        
        # 计算价格变化
        price_change = predicted_price - current_price
        change_percent = (price_change / current_price) * 100
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
            'current_price': None
        }


def predict_all_horizons(stock_code: str, recent_data: pd.DataFrame,
                         model_name: str = 'linear',
                         model_dir: str = 'models') -> Dict:
    """
    同时预测1天、5天、10天的价格
    
    Args:
        stock_code: 股票代码
        recent_data: 最近的历史数据
        model_name: 使用的模型名称
        model_dir: 模型目录
    
    Returns:
        {'1day': {...}, '5day': {...}, '10day': {...}}
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


# ============================================================================
# 第七部分：可视化模块
# ============================================================================

def plot_prediction_vs_actual(y_true, y_pred, stock_name, model_name, horizon, save_path):
    """绘制预测vs实际价格对比图"""
    plt.figure(figsize=(12, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.xlabel('实际价格', fontsize=14, fontweight='bold')
    plt.ylabel('预测价格', fontsize=14, fontweight='bold')
    plt.title(f'{stock_name} - 预测vs实际对比图\n{model_name.upper()}模型 - {horizon}天预测 (R²={r2:.4f}, RMSE={rmse:.2f})',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_distribution(y_true, y_pred, stock_name, model_name, horizon, save_path):
    """绘制残差分布直方图"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 8))
    
    plt.hist(residuals, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
    
    mu, std = stats.norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    plt.plot(x_range, stats.norm.pdf(x_range, mu, std), 'r-', linewidth=2, label=f'正态拟合 (μ={mu:.2f}, σ={std:.2f})')
    
    plt.xlabel('残差 (预测值-实际值)', fontsize=14, fontweight='bold')
    plt.ylabel('密度', fontsize=14, fontweight='bold')
    plt.title(f'{stock_name} - 预测误差分布图\n{model_name.upper()}模型 - {horizon}天预测',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(stock_name, horizon, metrics_dict, save_path):
    """绘制模型性能对比图"""
    models = list(metrics_dict.keys())
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    r2_values = [metrics_dict[m]['R2'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    
    # RMSE对比
    bars1 = axes[0].bar(models, rmse_values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[0].set_title('RMSE 对比（越小越好）', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # MAE对比
    bars2 = axes[1].bar(models, mae_values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[1].set_title('MAE 对比（越小越好）', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R²对比
    bars3 = axes[2].bar(models, r2_values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('R²', fontsize=12, fontweight='bold')
    axes[2].set_title('R² 对比（越接近1越好）', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{stock_name} - 模型性能对比 ({horizon}天预测)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations_for_stock(stock_code, stock_name, df, features, labels, horizons=[1, 5, 10]):
    """为单只股票生成所有可视化图表"""
    print(f"\n{'='*80}")
    print(f"生成 {stock_name} 的可视化图表")
    print(f"{'='*80}")
    
    viz_dir = 'static/figures'
    os.makedirs(viz_dir, exist_ok=True)
    
    stock_prefix = stock_code.replace('.', '_').replace('.SS', '_SH')
    
    for horizon in horizons:
        print(f"\n--- {horizon}天预测窗口 ---")
        
        # 加载所有模型
        models_data = {}
        for model_type in ['linear', 'ridge', 'lasso']:
            model_path = f'models/stock_{stock_prefix}_{model_type}_{horizon}day.pkl'
            if os.path.exists(model_path):
                loaded_data = joblib.load(model_path)
                
                if isinstance(loaded_data, dict):
                    model = loaded_data['model']
                    metrics = loaded_data.get('metrics', {})
                else:
                    model = loaded_data
                    metrics = {}
                
                models_data[model_type] = {'model': model, 'metrics': metrics}
        
        # 划分测试集
        train_size = int(len(features) * 0.8)
        X_test = features.iloc[train_size:]
        y_test = labels.iloc[train_size:]
        
        # 为每个模型生成图表
        for model_name, data in models_data.items():
            model = data['model']
            y_pred = model.predict(X_test)
            
            # 预测vs实际图
            save_path = f'{viz_dir}/pred_vs_actual_{stock_prefix}_{model_name}_{horizon}day.png'
            plot_prediction_vs_actual(y_test, y_pred, stock_name, model_name, horizon, save_path)
            
            # 残差分布图
            save_path = f'{viz_dir}/residual_{stock_prefix}_{model_name}_{horizon}day.png'
            plot_residual_distribution(y_test, y_pred, stock_name, model_name, horizon, save_path)
        
        # 模型对比图
        if len(models_data) > 0:
            metrics_dict = {name: data['metrics'] for name, data in models_data.items() if 'metrics' in data}
            if metrics_dict:
                save_path = f'{viz_dir}/model_comparison_{stock_prefix}_{horizon}day.png'
                plot_model_comparison(stock_name, horizon, metrics_dict, save_path)
    
    print(f"\n[OK] {stock_name} 的所有可视化图表已生成")


# ============================================================================
# 第八部分：数据库模块
# ============================================================================

class StockPredictionDB:
    """SQLite数据库管理类"""
    
    def __init__(self, db_path='data/predictions.db'):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def connect(self):
        """连接数据库并创建表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
    
    def create_tables(self):
        """创建预测历史表"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                stock_name TEXT,
                horizon INTEGER,
                predicted_price REAL,
                current_price REAL,
                price_change REAL,
                change_percent REAL,
                trend TEXT,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_prediction(self, prediction_record):
        """保存单条预测记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (stock_code, stock_name, horizon, predicted_price, current_price, 
             price_change, change_percent, trend, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_record['stock_code'],
            prediction_record['stock_name'],
            prediction_record['horizon'],
            prediction_record['predicted_price'],
            prediction_record['current_price'],
            prediction_record['price_change'],
            prediction_record['change_percent'],
            prediction_record['trend'],
            prediction_record['model_used']
        ))
        self.conn.commit()
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


# ============================================================================
# 第九部分：Web应用模块（Flask）
# ============================================================================

if FLASK_AVAILABLE:
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """首页"""
        return render_template('index.html', stocks=STOCK_CODES)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """处理预测请求"""
        if request.is_json:
            data = request.get_json()
            stock_code = data.get('stock_code', '').strip()
        else:
            stock_code = request.form.get('stock_code', '').strip()
        
        # 验证股票代码
        valid_codes = list(STOCK_CODES.keys())
        if stock_code not in valid_codes:
            return jsonify({
                'success': False,
                'message': '代码编号错误，请重新输入'
            }), 400
        
        try:
            # 从缓存加载数据
            cache_code = stock_code.replace('.SS', '.SH')
            cache_file = f'data/raw/{cache_code.replace(".", "_")}.csv'
            
            if not os.path.exists(cache_file):
                return jsonify({
                    'success': False,
                    'message': f'暂无{stock_code}的缓存数据'
                }), 404
            
            recent_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(recent_data) >= 100:
                recent_data = recent_data.tail(100)
            else:
                recent_data = recent_data.tail(len(recent_data))
            
            # 预测
            results = predict_all_horizons(stock_code, recent_data)
            
            # 获取股票名称
            stock_name = STOCK_CODES[stock_code]
            
            # 检查是否有错误
            has_error = any('error' in result for result in results.values())
            if has_error:
                error_msgs = [results[k]['error'] for k in results if 'error' in results[k]]
                return jsonify({
                    'success': False,
                    'message': f'预测失败：{"; ".join(error_msgs)}'
                }), 500
            
            # 生成可视化图表URL
            stock_prefix = stock_code.replace('.', '_').replace('.SS', '_SH')
            visualizations = {
                'model_comparison': f'/static/figures/model_comparison_{stock_prefix}_5day.png',
                'prediction_vs_actual': f'/static/figures/pred_vs_actual_{stock_prefix}_linear_5day.png',
                'residual_distribution': f'/static/figures/residual_{stock_prefix}_linear_5day.png'
            }
            
            # 保存预测结果到数据库
            try:
                with StockPredictionDB() as db:
                    predictions_to_save = []
                    for horizon_key, result in results.items():
                        horizon_int = int(horizon_key.replace('day', ''))
                        prediction_record = {
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'horizon': horizon_int,
                            'predicted_price': float(result.get('predicted_price', 0)),
                            'current_price': float(result.get('current_price', 0)),
                            'price_change': float(result.get('price_change', 0)),
                            'change_percent': float(result.get('change_percent', 0)),
                            'trend': result.get('trend', '未知'),
                            'model_used': 'ensemble'
                        }
                        predictions_to_save.append(prediction_record)
                        db.save_prediction(prediction_record)
            except Exception as e:
                print(f"警告: 保存预测结果到数据库失败: {e}")
            
            return jsonify({
                'success': True,
                'stock_code': stock_code,
                'stock_name': stock_name,
                'predictions': results,
                'visualizations': visualizations
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'服务器错误: {str(e)}'
            }), 500
    
    @app.route('/api/stocks')
    def get_stocks():
        """获取股票列表API"""
        return jsonify(STOCK_CODES)
    
    @app.route('/api/statistics')
    def get_statistics():
        """获取系统统计信息"""
        try:
            with StockPredictionDB() as db:
                cursor = db.conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM predictions')
                total_predictions = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT stock_code) FROM predictions')
                unique_stocks = cursor.fetchone()[0]
                
                return jsonify({
                    'total_predictions': total_predictions,
                    'unique_stocks': unique_stocks,
                    'supported_stocks': len(STOCK_CODES)
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# ============================================================================
# 第十部分：主程序入口
# ============================================================================

def main():
    """主程序入口"""
    print("="*80)
    print("股票价格预测系统 v2.0 (Unified)")
    print("="*80)
    print("\n可用功能：")
    print("1. 训练所有模型 - 运行 train_all_models()")
    print("2. 生成可视化图表 - 运行 generate_visualizations()")
    print("3. 启动Web服务 - 运行 start_web_server()")
    print("4. 单次预测测试 - 运行 test_prediction()")
    print("="*80)


def generate_visualizations():
    """生成所有股票的可视化图表"""
    print("="*80)
    print("生成所有可视化图表")
    print("="*80)
    
    for stock_code, stock_name in STOCK_CODES.items():
        cache_file = f'data/raw/{stock_code.replace(".", "_").replace(".SS", "_SH")}.csv'
        if not os.path.exists(cache_file):
            print(f"[SKIP] {stock_name} 缓存文件不存在")
            continue
        
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        features, labels = prepare_features_and_labels_regression(df, FEATURE_CONFIG, horizon=5)
        generate_visualizations_for_stock(stock_code, stock_name, df, features, labels)


def test_prediction():
    """测试预测功能"""
    print("="*80)
    print("测试预测功能")
    print("="*80)
    
    stock_code = '600519.SS'
    cache_file = 'data/raw/600519_SH.csv'
    
    if not os.path.exists(cache_file):
        print(f"[ERROR] 缓存文件不存在: {cache_file}")
        return
    
    recent_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    if len(recent_data) >= 100:
        recent_data = recent_data.tail(100)
    
    results = predict_all_horizons(stock_code, recent_data)
    
    print(f"\n{STOCK_CODES[stock_code]} ({stock_code}) 预测结果:")
    print("-"*80)
    for horizon, result in results.items():
        if 'error' not in result:
            print(f"{horizon:6s}: ¥{result['current_price']:.2f} → ¥{result['predicted_price']:.2f} "
                  f"({result['change_percent']:+.2f}%) [{result['trend']}]")
        else:
            print(f"{horizon:6s}: ERROR - {result['error']}")


def start_web_server(host='0.0.0.0', port=5000, debug=True):
    """启动Web服务器"""
    if not FLASK_AVAILABLE:
        print("[ERROR] Flask未安装")
        return
    
    print("="*80)
    print("股票价格预测系统 Web服务启动")
    print("="*80)
    print(f"支持 {len(STOCK_CODES)} 只股票")
    print(f"访问地址: http://localhost:{port}")
    print("="*80)
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
    
    # 示例：执行训练
    # train_all_models()
    
    # 示例：生成可视化
    # generate_visualizations()
    
    # 示例：测试预测
    # test_prediction()
    
    # 示例：启动Web服务
    # start_web_server()
