# 股票代码映射（6只已训练的股票）
STOCK_CODES = {
    '600519.SS': '贵州茅台',      # ✅ 消费/白酒 - 已下载+已训练
    '000001.SZ': '平安银行',      # ✅ 金融/银行 - 已下载+已训练
    '601318.SS': '中国平安',      # ✅ 金融/保险 - 已下载+已训练
    '600036.SS': '招商银行',      # ✅ 金融/银行 - 已下载+已训练
    '300750.SZ': '宁德时代',      # ✅ 新能源/电池 - 已下载+已训练
    '600887.SS': '伊利股份',      # ✅ 消费/乳业 - 已下载+已训练
}

# 数据获取配置
DATA_CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2025-01-01',
    'cache_dir': 'data/raw'
}

# 模型训练配置
MODEL_CONFIG = {
    'train_ratio': 0.8,           # 训练集比例
    'cv_folds': 3,                # 交叉验证折数
    'n_iter_search': 10,          # 随机搜索迭代次数
    'random_state': 42            # 随机种子
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
    'bollinger_std': 2,                      # 布林带标准差倍数
    'volatility_window': 20                  # 波动率计算窗口
}

# 预测时间窗口（天）
PREDICTION_HORIZONS = [1, 5, 10]

# 模型超参数搜索范围
MODEL_PARAMS = {
    'lightgbm': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1]
    }
}
