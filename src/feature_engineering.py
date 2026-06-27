"""
特征工程模块 (Feature Engineering Module)
=========================================

功能：从原始股票数据中提取有价值的技术指标特征，构建预测标签
      实现完整的特征工程流程，包括指标计算、标签构建、特征选择

主要函数：
    - calculate_technical_indicators(): 计算28种技术指标
    - create_labels(): 构建二分类标签（涨/跌）
    - prepare_features_and_labels(): 准备特征和标签的完整流程
    - select_features_by_importance(): 基于模型重要性选择特征

技术指标体系：
    1. 移动平均线类（4个）：MA5, MA10, MA20, MA60
    2. 动量指标类（5个）：RSI, MACD, MACD_Signal, MACD_Hist, MOM
    3. 随机指标类（3个）：KDJ_K, KDJ_D, KDJ_J
    4. 波动率指标类（3个）：Bollinger_Upper/Lower/Width
    5. 成交量指标类（2个）：OBV, Volume_MA5
    6. 价格变化率类（11个）：Return_1d~10d, Volatility等

作者：课程设计项目组
日期：2025年6月
版本：1.0
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def calculate_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    计算所有技术指标（28种）
    
    本函数从原始OHLCV数据中计算多种技术分析指标，涵盖趋势、动量、
    波动率、成交量等多个维度，为机器学习模型提供丰富的特征输入。
    
    Args:
        df (pd.DataFrame): 包含以下列的DataFrame
            - Open: 开盘价
            - High: 最高价
            - Low: 最低价
            - Close: 收盘价
            - Volume: 成交量
        config (dict): 特征配置字典，包含各指标的参数设置
            - ma_periods: 移动平均线周期列表 [5, 10, 20, 60]
            - rsi_period: RSI周期，默认14
            - macd_fast/slow/signal: MACD参数，默认(12, 26, 9)
            - kdj_k/d/j: KDJ参数，默认(9, 3, 3)
            - bollinger_window/std: 布林带参数，默认(20, 2)
            - volatility_window: 波动率窗口，默认20
    
    Returns:
        pd.DataFrame: 原始数据 + 28个技术指标列
            保留原始OHLCV列，新增28个技术指标列
            索引保持为DatetimeIndex
    
    Example:
        >>> from config import FEATURE_CONFIG
        >>> df = pd.read_csv('data/600519_SS.csv', index_col=0, parse_dates=True)
        >>> features = calculate_technical_indicators(df, FEATURE_CONFIG)
        >>> print(features.columns.tolist())
        ['Open', 'High', ..., 'MA5', 'RSI', 'MACD', ...]  # 共33列
    
    Note:
        - 所有指标计算都会产生NaN值（需要滚动窗口），后续需处理
        - 使用pandas内置方法确保计算效率
        - 指标计算公式遵循经典技术分析理论
        - 返回的DataFrame包含原始列+新特征列，便于追溯
    """
    df_features = df.copy()
    
    # ==================== 1. 移动平均线类 (Moving Averages) ====================
    # 功能：平滑价格波动，识别趋势方向
    # MA5/MA10: 短期趋势，MA20: 中期趋势，MA60: 长期趋势
    for period in config['ma_periods']:
        df_features[f'MA{period}'] = df_features['Close'].rolling(window=period).mean()
    
    # ==================== 2. RSI相对强弱指标 (Relative Strength Index) ====================
    # 功能：衡量超买超卖状态，范围0-100
    # >70: 超买区（可能回调），<30: 超卖区（可能反弹）
    rsi_period = config['rsi_period']
    delta = df_features['Close'].diff()  # 计算价格变化
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()  # 上涨幅度均值
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()  # 下跌幅度均值
    rs = gain / loss  # 相对强度
    df_features['RSI'] = 100 - (100 / (1 + rs))  # RSI公式
    
    # ==================== 3. MACD平滑异同移动平均线 ====================
    # 功能：判断买卖时机，由快线、慢线、柱状图组成
    # DIF>DEA且柱状图为正：多头市场；反之空头市场
    macd_fast = config['macd_fast']      # 快线EMA周期（默认12）
    macd_slow = config['macd_slow']      # 慢线EMA周期（默认26）
    macd_signal = config['macd_signal']  # 信号线EMA周期（默认9）
    
    ema_fast = df_features['Close'].ewm(span=macd_fast, adjust=False).mean()  # 快速EMA
    ema_slow = df_features['Close'].ewm(span=macd_slow, adjust=False).mean()  # 慢速EMA
    df_features['MACD'] = ema_fast - ema_slow  # DIF线（快线-慢线）
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=macd_signal, adjust=False).mean()  # DEA信号线
    df_features['MACD_Hist'] = df_features['MACD'] - df_features['MACD_Signal']  # 柱状图（DIF-DEA）
    
    # ==================== 4. KDJ随机指标 ====================
    # 功能：反映价格在特定周期内的相对位置，适合震荡市
    # K>D: 多头，K<D: 空头；J>100超买，J<0超卖
    kdj_k = config['kdj_k']  # K值周期（默认9）
    kdj_d = config['kdj_d']  # D值平滑周期（默认3）
    kdj_j = config['kdj_j']  # J值平滑周期（默认3）
    
    low_min = df_features['Low'].rolling(window=kdj_k).min()   # N日最低价
    high_max = df_features['High'].rolling(window=kdj_k).max() # N日最高价
    rsv = (df_features['Close'] - low_min) / (high_max - low_min) * 100  # 未成熟随机值
    df_features['K'] = rsv.ewm(com=kdj_d-1, adjust=False).mean()  # K值（RSV的EMA）
    df_features['D'] = df_features['K'].ewm(com=kdj_j-1, adjust=False).mean()  # D值（K的EMA）
    df_features['J'] = 3 * df_features['K'] - 2 * df_features['D']  # J值（3K-2D）
    
    # ==================== 5. 布林带 (Bollinger Bands) ====================
    # 功能：衡量价格波动范围和突破信号
    # 价格触及上轨：超买；触及下轨：超卖；带宽收窄：即将变盘
    boll_window = config['bollinger_window']  # 窗口期（默认20）
    boll_std = config['bollinger_std']        # 标准差倍数（默认2）
    ma = df_features['Close'].rolling(window=boll_window).mean()  # 中轨（MA20）
    std = df_features['Close'].rolling(window=boll_window).std()  # 标准差
    df_features['BB_Upper'] = ma + (std * boll_std)   # 上轨（MA+2σ）
    df_features['BB_Middle'] = ma                     # 中轨（MA）
    df_features['BB_Lower'] = ma - (std * boll_std)   # 下轨（MA-2σ）
    df_features['BB_Width'] = (df_features['BB_Upper'] - df_features['BB_Lower']) / ma  # 带宽
    
    # ==================== 6. 成交量指标类 ====================
    # 功能：结合价格和成交量，验证趋势有效性
    df_features['Volume_MA5'] = df_features['Volume'].rolling(window=5).mean()  # 5日成交量均线
    
    # OBV能量潮（On-Balance Volume）
    # 原理：价格上涨日成交量为正，下跌日为负，累计求和
    # OBV上升：资金流入；OBV下降：资金流出
    df_features['OBV'] = (np.sign(df_features['Close'].diff()) * df_features['Volume']).fillna(0).cumsum()
    
    # ==================== 7. 价格变化率类 ====================
    # 功能：捕捉短期、中期价格动量
    df_features['Return_1d'] = df_features['Close'].pct_change(periods=1)   # 1日收益率
    df_features['Return_5d'] = df_features['Close'].pct_change(periods=5)   # 5日收益率
    df_features['Return_10d'] = df_features['Close'].pct_change(periods=10) # 10日收益率
    
    # ==================== 8. 波动率指标 ====================
    # 功能：衡量价格波动程度，高波动率预示大行情
    vol_window = config['volatility_window']  # 波动率窗口（默认20）
    df_features['Volatility'] = df_features['Return_1d'].rolling(window=vol_window).std()  # 20日波动率
    
    # ==================== 9. 高低点距离 ====================
    # 功能：衡量当日价格波动幅度
    df_features['HL_Distance'] = (df_features['High'] - df_features['Low']) / df_features['Close']
    
    # ==================== 10. 收盘价相对于MA的位置 ====================
    # 功能：判断当前价格相对于中期趋势的位置
    # >1: 价格在MA20之上（强势），<1: 价格在MA20之下（弱势）
    df_features['Close_MA20_Ratio'] = df_features['Close'] / df_features['MA20']
    
    return df_features


def create_labels_regression(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    构建回归任务标签：未来horizon天的收盘价
    
    【关键改进】直接预测绝对价格，因为：
    1. 滞后价格特征与未来价格高度相关（>0.6）
    2. 收益率几乎不可预测（相关性<0.1）
    3. 添加Lag特征后，模型可以学习价格趋势
    
    Args:
        df: 包含'Close'列的DataFrame
        horizon: 预测时间窗口（1/5/10天）
    
    Returns:
        未来horizon天的收盘价作为连续值标签
    
    Example:
        >>> labels = create_labels_regression(df, horizon=5)
        >>> print(f"标签范围: {labels.min():.2f} - {labels.max():.2f}")
    """
    # 未来horizon天的收盘价作为回归目标
    labels = df['Close'].shift(-horizon)
    
    # 移除最后horizon行（因为没有未来数据）
    labels = labels.iloc[:-horizon]
    
    return labels


def create_labels_classification(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    创建涨跌标签（二分类）
    
    Args:
        df: 包含Close列的DataFrame
        horizon: 预测时间窗口（1, 5, 10天）
    
    Returns:
        Series: 1表示涨，0表示跌
    """
    # 未来horizon天的收盘价
    future_close = df['Close'].shift(-horizon)
    
    # 如果未来收盘价 > 当前收盘价，则为涨（1），否则为跌（0）
    labels = (future_close > df['Close']).astype(int)
    
    return labels


def prepare_features_and_labels_regression(df: pd.DataFrame, config: dict, 
                                            horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备回归任务的特征和标签
    
    根据任务书要求，预测未来具体价格（回归任务）。
    
    Args:
        df: 原始数据DataFrame
        config: 特征配置
        horizon: 预测时间窗口
    
    Returns:
        features: 特征DataFrame
        labels: 标签Series（未来horizon天的收盘价）
    """
    # 计算技术指标
    df_features = calculate_technical_indicators(df, config)
    
    # 【关键改进】添加滞后价格特征（Lag Features）
    # 原因：诊断显示Lag_1与未来价格相关性高达0.61，是最强预测因子
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    # 创建回归标签（未来价格）
    labels = create_labels_regression(df_features, horizon)
    
    # 删除包含NaN的行（由于滚动计算和标签shift导致）
    valid_idx = labels.dropna().index
    df_features = df_features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
    # 选择特征列（排除原始OHLCV和中间计算列）
    feature_columns = [
        # MA指标
        'MA5', 'MA10', 'MA20', 'MA60',
        # RSI
        'RSI',
        # MACD
        'MACD', 'MACD_Signal', 'MACD_Hist',
        # KDJ
        'K', 'D', 'J',
        # Bollinger Bands
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        # Volume
        'Volume', 'Volume_MA5', 'OBV',
        # Returns
        'Return_1d', 'Return_5d', 'Return_10d',
        # Volatility
        'Volatility',
        # Other
        'HL_Distance', 'Close_MA20_Ratio',
        # 【关键改进】滞后价格特征（最强预测因子）
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_5', 'Lag_10'
    ]
    
    # 确保所有特征列都存在
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 处理可能的无穷值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    return features, labels


def prepare_features_and_labels(df: pd.DataFrame, config: dict, 
                                 horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    准备特征和标签
    
    Args:
        df: 原始数据DataFrame
        config: 特征配置
        horizon: 预测时间窗口
    
    Returns:
        features: 特征DataFrame
        labels: 标签Series
    """
    # 计算技术指标
    df_features = calculate_technical_indicators(df, config)
    
    # 创建标签
    labels = create_labels(df_features, horizon)
    
    # 删除包含NaN的行（由于滚动计算和标签shift导致）
    valid_idx = labels.dropna().index
    df_features = df_features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    
    # 选择特征列（排除原始OHLCV和中间计算列）
    feature_columns = [
        # MA指标
        'MA5', 'MA10', 'MA20', 'MA60',
        # RSI
        'RSI',
        # MACD
        'MACD', 'MACD_Signal', 'MACD_Hist',
        # KDJ
        'K', 'D', 'J',
        # Bollinger Bands
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        # Volume
        'Volume', 'Volume_MA5', 'OBV',
        # Returns
        'Return_1d', 'Return_5d', 'Return_10d',
        # Volatility
        'Volatility',
        # Other
        'HL_Distance', 'Close_MA20_Ratio'
    ]
    
    # 确保所有特征列都存在
    available_columns = [col for col in feature_columns if col in df_features.columns]
    features = df_features[available_columns].copy()
    
    # 处理可能的无穷值
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    
    return features, labels


def select_top_features(features: pd.DataFrame, labels: pd.Series, 
                        top_n: int = 15) -> List[str]:
    """
    选择最重要的Top N个特征
    
    Args:
        features: 特征DataFrame
        labels: 标签Series
        top_n: 选择的特征数量
    
    Returns:
        选中的特征名称列表
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # 使用随机森林计算特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, labels)
    
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择Top N
    selected_features = importance_df.head(top_n)['feature'].tolist()
    
    print("\nTop 15 重要特征:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return selected_features


def normalize_features(train_features: pd.DataFrame, 
                       test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    标准化特征（Min-Max归一化）
    
    Args:
        train_features: 训练集特征
        test_features: 测试集特征
    
    Returns:
        标准化后的训练集和测试集特征
    """
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 在训练集上fit，然后transform训练集和测试集
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    
    # 转换回DataFrame
    train_df = pd.DataFrame(train_scaled, columns=train_features.columns, 
                           index=train_features.index)
    test_df = pd.DataFrame(test_scaled, columns=test_features.columns,
                          index=test_features.index)
    
    return train_df, test_df


def select_features_by_correlation(features: pd.DataFrame, 
                                   labels: pd.Series,
                                   threshold: float = 0.05) -> List[str]:
    """
    基于与标签的相关性筛选特征
    
    功能：计算每个特征与目标变量的皮尔逊相关系数，
          保留相关性绝对值大于阈值的特征。
    
    Args:
        features: 特征DataFrame
        labels: 标签Series
        threshold: 相关性阈值，默认0.05
    
    Returns:
        满足相关性要求的特征名称列表
    
    Example:
        >>> selected = select_features_by_correlation(features, labels, 0.05)
        >>> print(f"选中 {len(selected)} 个特征")
    """
    # 计算每个特征与标签的相关性
    correlations = {}
    for col in features.columns:
        corr = features[col].corr(labels)
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    
    # 筛选满足阈值的特征
    selected = [col for col, corr in correlations.items() if corr >= threshold]
    
    # 按相关性排序
    selected_sorted = sorted(selected, key=lambda x: correlations[x], reverse=True)
    
    print(f"\n基于相关性筛选 (阈值={threshold}):")
    print(f"  原始特征数: {len(features.columns)}")
    print(f"  选中特征数: {len(selected_sorted)}")
    print(f"\nTop 10 高相关特征:")
    for i, feat in enumerate(selected_sorted[:10], 1):
        print(f"  {i:2d}. {feat:20s}: {correlations[feat]:.4f}")
    
    return selected_sorted


def select_features_combined(features: pd.DataFrame, 
                            labels: pd.Series,
                            top_n: int = 15,
                            corr_threshold: float = 0.05) -> List[str]:
    """
    组合特征选择：先基于相关性过滤，再基于重要性排序
    
    策略：
        1. 第一步：使用相关性分析去除低相关特征
        2. 第二步：在剩余特征中使用LightGBM重要性排序
        3. 第三步：选择Top N个最重要的特征
    
    优势：
        - 减少噪声特征干扰
        - 提高模型训练速度
        - 增强模型可解释性
    
    Args:
        features: 特征DataFrame
        labels: 标签Series
        top_n: 最终选择的特征数量，默认15
        corr_threshold: 相关性阈值，默认0.05
    
    Returns:
        最终选中的特征名称列表
    
    Example:
        >>> selected = select_features_combined(features, labels, top_n=15)
        >>> print(f"最终选中 {len(selected)} 个特征")
    """
    print("\n" + "="*60)
    print("组合特征选择流程")
    print("="*60)
    
    # 步骤1: 基于相关性初步筛选
    corr_selected = select_features_by_correlation(features, labels, corr_threshold)
    
    if len(corr_selected) == 0:
        print("\n警告: 没有特征满足相关性阈值，使用全部特征")
        corr_selected = features.columns.tolist()
    
    # 步骤2: 在筛选后的特征上计算重要性
    features_filtered = features[corr_selected]
    importance_selected = select_top_features(features_filtered, labels, min(top_n, len(corr_selected)))
    
    print(f"\n最终选中特征数: {len(importance_selected)}")
    print("="*60)
    
    return importance_selected


if __name__ == '__main__':
    # 测试代码
    from config import FEATURE_CONFIG, STOCK_CODES
    from data_fetch import fetch_stock_data, DATA_CONFIG
    
    print("测试特征工程模块...")
    
    # 获取一只股票的数据进行测试
    stock_code = list(STOCK_CODES.keys())[0]
    stock_name = STOCK_CODES[stock_code]
    
    print(f"\n处理 {stock_name} ({stock_code})")
    df = fetch_stock_data(stock_code, DATA_CONFIG['start_date'], 
                         DATA_CONFIG['end_date'])
    
    # 计算特征
    print("\n计算技术指标...")
    df_features = calculate_technical_indicators(df, FEATURE_CONFIG)
    print(f"特征数量: {len(df_features.columns)}")
    
    # 创建标签
    print("\n创建标签...")
    for horizon in [1, 5, 10]:
        labels = create_labels(df_features, horizon)
        print(f"{horizon}天预测 - 涨的比例: {labels.mean():.2%}")
    
    # 准备特征和标签
    print("\n准备特征和标签...")
    features, labels = prepare_features_and_labels(df_features, FEATURE_CONFIG, horizon=5)
    print(f"有效样本数: {len(features)}")
    print(f"特征数量: {len(features.columns)}")
    
    # 特征选择
    print("\n特征选择...")
    selected = select_top_features(features, labels, top_n=15)
    print(f"\n选中的特征: {selected}")
