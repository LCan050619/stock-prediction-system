"""
数据获取模块 (Data Fetch Module)
===============================

功能：从AkShare（优先）或Yahoo Finance（备选）获取A股股票历史数据
      实现双数据源容错机制，提高系统鲁棒性

主要函数：
    - fetch_from_akshare(): 从AkShare获取A股数据
    - fetch_stock_data(): 获取单只股票数据（含缓存和容错）
    - fetch_all_stocks(): 批量获取多只股票数据
    - preprocess_data(): 数据预处理（缺失值、异常值处理）

数据源对比：
    AkShare: 国内数据源，访问稳定，支持前复权，但偶尔连接中断
    Yahoo Finance: 国际通用，覆盖面广，但容易被限流

作者：课程设计项目组
日期：2025年6月
版本：1.0
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import Optional
import time

# 尝试导入akshare作为备选数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: akshare未安装，将仅使用yfinance")


def fetch_from_akshare(stock_code: str, start_date: str, end_date: str, cache_file: str) -> pd.DataFrame:
    """
    从AkShare获取A股历史数据（国内数据源，访问更稳定）
    
    AkShare是中国开源金融数据接口库，提供稳定的A股数据访问。
    使用ak.stock_zh_a_hist()接口获取日线级别的历史行情数据。
    
    Args:
        stock_code (str): 股票代码，格式如 '600519.SS'（上海）或 '000001.SZ'（深圳）
                         函数会自动提取纯数字部分（如 '600519'）
        start_date (str): 开始日期，格式 'YYYY-MM-DD'，如 '2022-01-01'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'，如 '2025-01-01'
        cache_file (str): CSV缓存文件路径，用于保存下载的数据避免重复请求
    
    Returns:
        pd.DataFrame: 包含以下列的DataFrame
            - Open: 开盘价（元）
            - High: 最高价（元）
            - Low: 最低价（元）
            - Close: 收盘价（元）
            - Volume: 成交量（手）
            索引为DatetimeIndex（交易日期）
    
    Raises:
        Exception: 当3次重试都失败时抛出异常
    
    Example:
        >>> df = fetch_from_akshare('600519.SS', '2022-01-01', '2025-01-01', 'data/600519_SS.csv')
        >>> print(df.head())
        
    Note:
        - 使用前复权数据（adjust="qfq"），消除除权除息影响
        - 自动重命名中文列为英文，与Yahoo Finance格式保持一致
        - 内置3次重试机制，应对网络波动
        - 成功后自动保存到CSV缓存文件
    """
    # 提取纯数字代码（去掉 .SS 或 .SZ）
    code = stock_code.split('.')[0]
    
    # 【关键修复】硬编码限制最大日期，防止系统时间错误
    MAX_DATE = '20251231'
    if end_date.replace('-', '') > MAX_DATE:
        print(f"WARNING: Limiting end_date from {end_date} to 2025-12-31")
        end_date = '2025-12-31'
    
    print(f"使用AkShare下载 {code} 的数据...")
    
    # 增加重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用ak.stock_zh_a_hist获取历史数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                print(f"警告: AkShare未获取到 {code} 的数据")
                return pd.DataFrame()
            
            # 重命名列以匹配Yahoo Finance格式
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
            
            # 选择需要的列
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 保存到缓存
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            df.to_csv(cache_file)
            print(f"AkShare数据已保存到: {cache_file}")
            
            return df
            
        except Exception as e:
            print(f"AkShare尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)  # 等待3秒后重试
            else:
                raise


def fetch_stock_data(stock_code: str, start_date: str, end_date: str, 
                     cache_dir: str = 'data/raw') -> pd.DataFrame:
    """
    获取单只股票的历史数据（优先使用AkShare，失败则使用Yahoo Finance）
    
    Args:
        stock_code: 股票代码（如 '600519.SS' 或 '600519'）
        start_date: 开始日期（如 '2022-01-01'）
        end_date: 结束日期（如 '2025-01-01'）
        cache_dir: 缓存目录
    
    Returns:
        DataFrame包含：Open, High, Low, Close, Volume
    """
    # 【关键修复】强制限制日期范围，防止系统时间错误
    # 硬编码最大合理日期（假设当前真实时间是2025年中）
    MAX_REASONABLE_DATE = '2025-12-31'
    
    from datetime import datetime as dt
    try:
        end_dt = dt.strptime(end_date, '%Y-%m-%d')
        max_dt = dt.strptime(MAX_REASONABLE_DATE, '%Y-%m-%d')
        
        # 如果end_date超过最大合理日期，强制限制
        if end_dt > max_dt:
            print(f"WARNING: end_date ({end_date}) exceeds reasonable limit!")
            print(f"   Limiting to: {MAX_REASONABLE_DATE}")
            end_date = MAX_REASONABLE_DATE
    except:
        pass  # 如果解析失败，继续使用原始日期
    
    # 构建缓存文件路径
    cache_file = os.path.join(cache_dir, f'{stock_code.replace(".", "_")}.csv')
    
    # 如果缓存存在，直接读取
    if os.path.exists(cache_file):
        print(f"从缓存加载数据: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    # 首先尝试使用AkShare（国内数据源，更稳定）
    if AKSHARE_AVAILABLE:
        try:
            df = fetch_from_akshare(stock_code, start_date, end_date, cache_file)
            if not df.empty:
                return df
        except Exception as e:
            print(f"AkShare获取失败: {str(e)}，尝试使用Yahoo Finance...")
    
    # 备用方案：从Yahoo Finance获取数据
    print(f"正在下载 {stock_code} 的数据...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"警告: {stock_code} 没有获取到数据")
                return pd.DataFrame()
            
            # 选择需要的列
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 确保索引是datetime类型
            df.index = pd.to_datetime(df.index)
            
            # 保存到缓存
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(cache_file)
            print(f"数据已保存到: {cache_file}")
            
            return df
            
        except Exception as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"错误: 无法获取 {stock_code} 的数据")
                raise


def fetch_all_stocks(stock_codes: dict, start_date: str, end_date: str,
                     cache_dir: str = 'data/raw') -> dict:
    """
    获取所有股票的数据
    
    Args:
        stock_codes: 股票代码字典 {code: name}
        start_date: 开始日期
        end_date: 结束日期
        cache_dir: 缓存目录
    
    Returns:
        字典 {stock_code: DataFrame}
    """
    all_data = {}
    
    for stock_code, stock_name in stock_codes.items():
        print(f"\n处理 {stock_name} ({stock_code})...")
        try:
            df = fetch_stock_data(stock_code, start_date, end_date, cache_dir)
            if not df.empty:
                all_data[stock_code] = df
                print(f"[OK] {stock_name} 数据获取成功，共 {len(df)} 条记录")
            else:
                print(f"[FAIL] {stock_name} 数据获取失败")
        except Exception as e:
            print(f"[ERROR] {stock_name} 数据获取异常: {str(e)}")
        
        # 避免请求过快
        time.sleep(1)
    
    print(f"\n总计成功获取 {len(all_data)} 只股票的数据")
    return all_data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理：处理缺失值和异常值
    
    Args:
        df: 原始数据DataFrame
    
    Returns:
        预处理后的DataFrame
    """
    df_processed = df.copy()
    
    # 将所有数值列转换为float64，避免pandas 3.0的类型限制
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_processed[column] = df_processed[column].astype(float)
    
    # 处理缺失值：前向填充 + 后向填充
    df_processed = df_processed.ffill().bfill()
    
    # 处理异常值（IQR方法）
    for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
        Q1 = df_processed[column].quantile(0.25)
        Q3 = df_processed[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median = df_processed[column].median()
        
        # 将异常值替换为中位数
        mask = (df_processed[column] < lower_bound) | (df_processed[column] > upper_bound)
        df_processed.loc[mask, column] = median
    
    return df_processed


if __name__ == '__main__':
    # 测试代码
    from config import STOCK_CODES, DATA_CONFIG
    
    print("开始获取股票数据...")
    print(f"时间范围: {DATA_CONFIG['start_date']} 至 {DATA_CONFIG['end_date']}")
    print(f"股票数量: {len(STOCK_CODES)}")
    
    # 获取所有股票数据
    all_data = fetch_all_stocks(
        STOCK_CODES,
        DATA_CONFIG['start_date'],
        DATA_CONFIG['end_date'],
        DATA_CONFIG['cache_dir']
    )
    
    # 显示数据统计信息
    print("\n数据统计:")
    for stock_code, df in all_data.items():
        stock_name = STOCK_CODES[stock_code]
        print(f"{stock_name} ({stock_code}):")
        print(f"  数据量: {len(df)} 条")
        print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
        print(f"  收盘价范围: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
        print()
