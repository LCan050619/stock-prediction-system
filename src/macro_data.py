"""
宏观经济数据模块 (Macro Economic Data Module)
==========================================

功能：从AkShare获取宏观经济数据（利率、汇率、经济指数等）
      作为股票预测的辅助特征，提高预测准确性

数据来源：AkShare开源金融数据接口库

主要数据类型：
    1. 利率数据：SHIBOR（上海银行间同业拆放利率）、LPR（贷款市场报价利率）
    2. 汇率数据：USD/CNY（美元兑人民币）、EUR/CNY（欧元兑人民币）
    3. 经济指数：CPI（消费者物价指数）、PPI（生产者物价指数）、PMI（采购经理指数）

作者：课程设计项目组
日期：2025年6月
版本：1.0
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from typing import Optional


def fetch_shibor_data(start_date: str = '2022-01-01', 
                      end_date: str = '2025-01-01') -> pd.DataFrame:
    """
    获取SHIBOR（上海银行间同业拆放利率）数据
    
    SHIBOR是由信用等级较高的银行组成报价团自主报出的人民币同业拆出利率
    计算确定的算术平均利率，是单利、无担保、批发性利率。
    
    Args:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: 包含以下列
            - date: 日期
            - ON: 隔夜利率
            - 1W: 1周利率
            - 2W: 2周利率
            - 1M: 1个月利率
            - 3M: 3个月利率
            - 6M: 6个月利率
            - 9M: 9个月利率
            - 1Y: 1年利率
    
    Example:
        >>> df = fetch_shibor_data('2022-01-01', '2025-01-01')
        >>> print(df.head())
        
    Note:
        - SHIBOR反映银行间资金供求状况
        - 利率上升通常对股市利空（资金成本增加）
        - 利率下降通常对股市利好（流动性充裕）
    """
    try:
        print("正在获取SHIBOR利率数据...")
        
        # 使用ak.shibor获取SHIBOR历史数据（注意：不同版本API可能不同）
        # 尝试多种可能的接口名称
        try:
            df = ak.shibor()
        except AttributeError:
            try:
                df = ak.rate_shibor_quote()
            except AttributeError:
                print("警告: AkShare当前版本不支持SHIBOR接口")
                return pd.DataFrame()
        
        if df.empty:
            print("警告: 未获取到SHIBOR数据")
            return pd.DataFrame()
        
        # 查看列名并处理
        print(f"SHIBOR数据列名: {df.columns.tolist()}")
        
        # 转换日期格式（根据实际列名调整）
        if '日期' in df.columns:
            df['date'] = pd.to_datetime(df['日期'])
            df.set_index('date', inplace=True)
        elif 'report_date' in df.columns:
            df['date'] = pd.to_datetime(df['report_date'])
            df.set_index('date', inplace=True)
        else:
            # 如果第一列是日期
            df.set_index(df.columns[0], inplace=True)
            df.index = pd.to_datetime(df.index)
        
        # 选择需要的期限品种（常用的是ON、1W、1M、3M、1Y）
        selected_columns = ['ON', '1W', '1M', '3M', '1Y']
        available_columns = [col for col in selected_columns if col in df.columns]
        
        if available_columns:
            df = df[available_columns]
        
        # 按日期范围筛选
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(df) > 0:
            print(f"✓ SHIBOR数据获取成功，共 {len(df)} 条记录")
            print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
            print("警告: 筛选后无数据")
        
        return df
        
    except Exception as e:
        print(f"✗ 获取SHIBOR数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_exchange_rate(currency: str = 'USD',
                       start_date: str = '2022-01-01',
                       end_date: str = '2025-01-01') -> pd.DataFrame:
    """
    获取汇率数据
    
    Args:
        currency (str): 货币类型，可选值：
            - 'USD': 美元兑人民币
            - 'EUR': 欧元兑人民币
            - 'JPY': 日元兑人民币
            - 'HKD': 港币兑人民币
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: 包含以下列
            - date: 日期
            - exchange_rate: 汇率（1外币兑多少人民币）
    
    Example:
        >>> df = fetch_exchange_rate('USD', '2022-01-01', '2025-01-01')
        >>> print(df.head())
        
    Note:
        - 人民币汇率影响进出口企业盈利
        - 人民币贬值（汇率上升）利好出口企业，利空进口企业
        - 人民币升值（汇率下降）利好进口企业，利空出口企业
    """
    try:
        print(f"正在获取{currency}/CNY汇率数据...")
        
        # 根据货币类型选择对应的AkShare接口
        # 注意：不同版本的AkShare API可能不同，需要适配
        try:
            if currency == 'USD':
                # 尝试多种可能的接口
                try:
                    df = ak.currency_boc_sina(symbol="美元")
                except TypeError:
                    # 如果不需要start/end参数
                    df = ak.currency_boc_sina(symbol="美元")
            elif currency == 'EUR':
                df = ak.currency_boc_sina(symbol="欧元")
            elif currency == 'JPY':
                df = ak.currency_boc_sina(symbol="日元")
            elif currency == 'HKD':
                df = ak.currency_boc_sina(symbol="港币")
            else:
                print(f"不支持的货币类型: {currency}")
                return pd.DataFrame()
        except AttributeError:
            print("警告: AkShare当前版本不支持汇率接口")
            return pd.DataFrame()
        
        if df.empty:
            print(f"警告: 未获取到{currency}/CNY汇率数据")
            return pd.DataFrame()
        
        # 查看列名
        print(f"汇率数据列名: {df.columns.tolist()}")
        
        # 重命名列（根据实际列名调整）
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'date'})
        elif 'report_date' in df.columns:
            df = df.rename(columns={'report_date': 'date'})
        
        # 寻找汇率列
        rate_columns = [col for col in df.columns if '收盘' in col or 'price' in col.lower() or 'rate' in col.lower()]
        if rate_columns:
            df = df.rename(columns={rate_columns[0]: 'exchange_rate'})
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 只保留汇率列
        if 'exchange_rate' in df.columns:
            df = df[['exchange_rate']]
        
        # 按日期范围筛选
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(df) > 0:
            print(f"✓ {currency}/CNY汇率数据获取成功，共 {len(df)} 条记录")
            print(f"  汇率范围: {df['exchange_rate'].min():.4f} - {df['exchange_rate'].max():.4f}")
        else:
            print("警告: 筛选后无数据")
        
        return df
        
    except Exception as e:
        print(f"✗ 获取{currency}/CNY汇率数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_cpi_data(start_date: str = '2022-01-01',
                   end_date: str = '2025-01-01') -> pd.DataFrame:
    """
    获取CPI（消费者物价指数）数据
    
    CPI是反映居民家庭购买消费商品和服务的价格变动情况的商品劳务价格统计，
    也是衡量通货膨胀的重要指标。
    
    Args:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: 包含以下列
            - date: 日期（月度数据）
            - cpi_yoy: CPI同比涨跌幅（%）
            - cpi_mom: CPI环比涨跌幅（%）
    
    Example:
        >>> df = fetch_cpi_data('2022-01-01', '2025-01-01')
        >>> print(df.head())
        
    Note:
        - CPI上涨表示通货膨胀，下跌表示通货紧缩
        - 适度通胀（2-3%）有利于经济增长
        - 高通胀对股市利空（央行可能加息）
        - 通缩对股市利空（经济衰退信号）
    """
    try:
        print("正在获取CPI数据...")
        
        # 使用ak.macro_china_cpi获取CPI数据
        df = ak.macro_china_cpi()
        
        if df.empty:
            print("警告: 未获取到CPI数据")
            return pd.DataFrame()
        
        # 查看列名（不同版本的AkShare可能列名不同）
        print(f"CPI数据列名: {df.columns.tolist()}")
        
        # 尝试提取需要的列
        # 注意：实际列名可能因AkShare版本而异，需要根据实际情况调整
        if '年份' in df.columns and '月份' in df.columns:
            df['date'] = pd.to_datetime(df['年份'].astype(str) + '-' + df['月份'].astype(str).str.zfill(2))
            df.set_index('date', inplace=True)
            
            # 寻找CPI相关列
            cpi_columns = [col for col in df.columns if 'cpi' in col.lower() or '居民消费价格指数' in col]
            if cpi_columns:
                df = df[cpi_columns[:2]]  # 取前两个CPI相关列
                df.columns = ['cpi_yoy', 'cpi_mom']
        
        print(f"✓ CPI数据获取成功，共 {len(df)} 条记录")
        
        return df
        
    except Exception as e:
        print(f" 获取CPI数据失败: {str(e)}")
        return pd.DataFrame()


def fetch_all_macro_data(start_date: str = '2022-01-01',
                         end_date: str = '2025-01-01') -> dict:
    """
    获取所有宏观经济数据
    
    Args:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'
        end_date (str): 结束日期，格式 'YYYY-MM-DD'
    
    Returns:
        dict: 包含以下键值对
            - 'shibor': SHIBOR利率数据 DataFrame
            - 'usd_cny': 美元兑人民币汇率数据 DataFrame
            - 'cpi': CPI数据 DataFrame
    """
    print("="*70)
    print("获取宏观经济数据")
    print("="*70)
    
    macro_data = {}
    
    # 获取SHIBOR利率
    macro_data['shibor'] = fetch_shibor_data(start_date, end_date)
    
    # 获取美元兑人民币汇率
    macro_data['usd_cny'] = fetch_exchange_rate('USD', start_date, end_date)
    
    # 获取CPI数据
    macro_data['cpi'] = fetch_cpi_data(start_date, end_date)
    
    # 统计信息
    print("\n" + "="*70)
    print("宏观经济数据获取完成")
    print("="*70)
    for key, df in macro_data.items():
        if not df.empty:
            print(f"✓ {key}: {len(df)} 条记录")
        else:
            print(f"✗ {key}: 获取失败")
    
    return macro_data


def merge_macro_with_stock(stock_df: pd.DataFrame, 
                          macro_data: dict) -> pd.DataFrame:
    """
    将宏观经济数据合并到股票数据中
    
    Args:
        stock_df (pd.DataFrame): 股票价格数据，索引为DatetimeIndex
        macro_data (dict): 宏观经济数据字典
    
    Returns:
        pd.DataFrame: 合并后的DataFrame，包含股票数据和宏观数据
    
    Example:
        >>> stock_df = pd.read_csv('data/600519_SS.csv', index_col=0, parse_dates=True)
        >>> macro_data = fetch_all_macro_data()
        >>> merged_df = merge_macro_with_stock(stock_df, macro_data)
        >>> print(merged_df.head())
        
    Note:
        - 使用pd.merge_asof进行向前填充，确保每个交易日都有对应的宏观数据
        - 宏观数据频率可能与股票数据不同（如CPI是月度，股票是日度）
        - 缺失的宏观数据使用前一个可用值填充
    """
    merged_df = stock_df.copy()
    
    # 依次合并每种宏观数据
    for macro_name, macro_df in macro_data.items():
        if macro_df.empty:
            print(f"跳过 {macro_name}：数据为空")
            continue
        
        try:
            # 确保索引是datetime类型
            if not isinstance(macro_df.index, pd.DatetimeIndex):
                macro_df.index = pd.to_datetime(macro_df.index)
            
            # 按日期对齐合并（向前填充）
            merged_df = pd.merge_asof(
                merged_df.sort_index(),
                macro_df.sort_index().reset_index().rename(columns={'index': 'date'}),
                left_index=True,
                right_on='date',
                direction='backward'
            )
            
            # 重新设置索引
            merged_df.set_index(pd.to_datetime(merged_df['date']), inplace=True)
            merged_df.drop(columns=['date'], inplace=True)
            
            print(f"✓ 已合并 {macro_name} 数据")
            
        except Exception as e:
            print(f"✗ 合并 {macro_name} 数据失败: {str(e)}")
    
    return merged_df


if __name__ == '__main__':
    # 测试代码
    print("测试宏观经济数据模块\n")
    
    # 获取所有宏观数据
    macro_data = fetch_all_macro_data('2022-01-01', '2025-01-01')
    
    # 显示数据统计
    print("\n" + "="*70)
    print("数据统计详情")
    print("="*70)
    
    for name, df in macro_data.items():
        if not df.empty:
            print(f"\n{name.upper()}:")
            print(f"  数据量: {len(df)} 条")
            print(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
            print(f"  列名: {df.columns.tolist()}")
            print(f"  样本数据:")
            print(df.head(3))
            print()
    
    # 测试合并功能
    print("\n" + "="*70)
    print("测试合并功能")
    print("="*70)
    
    try:
        # 加载一只股票的数据
        stock_df = pd.read_csv('data/raw/600519_SS.csv', index_col=0, parse_dates=True)
        print(f"\n原始股票数据: {len(stock_df)} 条记录")
        print(f"列名: {stock_df.columns.tolist()}")
        
        # 合并宏观数据
        merged_df = merge_macro_with_stock(stock_df, macro_data)
        print(f"\n合并后数据: {len(merged_df)} 条记录")
        print(f"列名: {merged_df.columns.tolist()}")
        print(f"样本数据:")
        print(merged_df.head(3))
        
    except Exception as e:
        print(f"合并测试失败: {str(e)}")
