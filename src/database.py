"""
数据库模块 (Database Module)
============================

功能：使用SQLite存储预测结果、模型信息和用户查询历史
      实现数据持久化，支持历史记录查询和统计分析

主要功能：
    - 创建数据库和表结构
    - 存储预测结果（股票代码、预测时间、预测方向、概率等）
    - 记录用户查询历史
    - 存储模型元信息（训练时间、性能指标等）
    - 提供统计查询接口

数据库结构：
    1. predictions表：预测结果记录
    2. query_history表：用户查询历史
    3. model_metadata表：模型元信息

作者：课程设计项目组
日期：2025年6月
版本：1.0
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json


class StockPredictionDB:
    """股票预测系统数据库管理类"""
    
    def __init__(self, db_path: str = 'data/predictions.db'):
        """
        初始化数据库连接
        
        Args:
            db_path: SQLite数据库文件路径，默认'data/predictions.db'
        """
        self.db_path = db_path
        # 只在有目录路径时才创建目录
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化数据库，创建必要的表"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # 创建预测结果表（回归版本）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT,
                    prediction_date TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    predicted_price REAL,
                    current_price REAL,
                    price_change REAL,
                    change_percent REAL,
                    trend TEXT,
                    model_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建用户查询历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT,
                    query_time TEXT DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # 创建模型元信息表（回归版本）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    stock_name TEXT,
                    horizon INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    training_date TEXT,
                    mse REAL,
                    rmse REAL,
                    mae REAL,
                    r2 REAL,
                    mape REAL,
                    model_path TEXT,
                    feature_count INTEGER,
                    sample_count INTEGER
                )
            ''')
            
            # 创建索引以提高查询效率
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_stock ON predictions(stock_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_stock ON query_history(stock_code)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_stock ON model_metadata(stock_code)')
            
            self.conn.commit()
            print(f"[OK] 数据库初始化成功: {self.db_path}")
            
        except Exception as e:
            print(f"[ERROR] 数据库初始化失败: {e}")
            raise
    
    def save_prediction(self, stock_code: str, stock_name: str,
                       horizon: int, predicted_price: float = None,
                       current_price: float = None, price_change: float = None,
                       change_percent: float = None, trend: str = None,
                       model_used: str = 'ensemble') -> bool:
        """
        保存预测结果到数据库（回归版本）
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            horizon: 预测时间窗口（1/5/10天）
            predicted_price: 预测价格
            current_price: 当前价格
            price_change: 价格变化
            change_percent: 涨跌幅百分比
            trend: 趋势（上涨/下跌）
            model_used: 使用的模型名称
        
        Returns:
            是否保存成功
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (stock_code, stock_name, prediction_date, horizon, 
                 predicted_price, current_price, price_change, change_percent, trend, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stock_code,
                stock_name,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                horizon,
                predicted_price,
                current_price,
                price_change,
                change_percent,
                trend,
                model_used
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] 保存预测结果失败: {e}")
            return False
    
    def save_predictions_batch(self, predictions: List[Dict]) -> int:
        """
        批量保存预测结果（回归版本）
        
        Args:
            predictions: 预测结果列表，每个元素是包含以下键的字典：
                - stock_code, stock_name, horizon, predicted_price,
                - current_price, price_change, change_percent, trend, model_used
        
        Returns:
            成功保存的记录数
        """
        count = 0
        for pred in predictions:
            if self.save_prediction(**pred):
                count += 1
        return count
    
    def record_query(self, stock_code: str, stock_name: str = None,
                    ip_address: str = None, success: bool = True) -> bool:
        """
        记录用户查询历史
        
        Args:
            stock_code: 查询的股票代码
            stock_name: 股票名称
            ip_address: 用户IP地址
            success: 查询是否成功
        
        Returns:
            是否记录成功
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO query_history 
                (stock_code, stock_name, ip_address, success)
                VALUES (?, ?, ?, ?)
            ''', (stock_code, stock_name, ip_address, success))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] 记录查询历史失败: {e}")
            return False
    
    def save_model_metadata(self, metadata: Dict) -> bool:
        """
        保存模型元信息
        
        Args:
            metadata: 模型元信息字典，包含：
                - stock_code, stock_name, horizon, model_name,
                - accuracy, precision_score, recall, f1_score, auc_roc,
                - model_path, feature_count, sample_count
        
        Returns:
            是否保存成功
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO model_metadata 
                (stock_code, stock_name, horizon, model_name, training_date,
                 accuracy, precision_score, recall, f1_score, auc_roc,
                 model_path, feature_count, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.get('stock_code'),
                metadata.get('stock_name'),
                metadata.get('horizon'),
                metadata.get('model_name'),
                metadata.get('training_date', datetime.now().strftime('%Y-%m-%d')),
                metadata.get('accuracy'),
                metadata.get('precision_score'),
                metadata.get('recall'),
                metadata.get('f1_score'),
                metadata.get('auc_roc'),
                metadata.get('model_path'),
                metadata.get('feature_count'),
                metadata.get('sample_count')
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ERROR] 保存模型元信息失败: {e}")
            return False
    
    def get_prediction_history(self, stock_code: str, 
                              limit: int = 10) -> List[Dict]:
        """
        获取某只股票的预测历史
        
        Args:
            stock_code: 股票代码
            limit: 返回记录数量限制，默认10
        
        Returns:
            预测历史记录列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE stock_code = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (stock_code, limit))
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            print(f"✗ 查询预测历史失败: {e}")
            return []
    
    def get_query_statistics(self, days: int = 7) -> Dict:
        """
        获取查询统计数据
        
        Args:
            days: 统计天数，默认7天
        
        Returns:
            统计信息字典
        """
        try:
            cursor = self.conn.cursor()
            
            # 总查询次数
            cursor.execute('''
                SELECT COUNT(*) FROM query_history
                WHERE query_time >= datetime('now', '-{} days')
            '''.format(days))
            total_queries = cursor.fetchone()[0]
            
            # 成功查询次数
            cursor.execute('''
                SELECT COUNT(*) FROM query_history
                WHERE query_time >= datetime('now', '-{} days') AND success = 1
            '''.format(days))
            success_queries = cursor.fetchone()[0]
            
            # 热门股票排行
            cursor.execute('''
                SELECT stock_code, stock_name, COUNT(*) as count
                FROM query_history
                WHERE query_time >= datetime('now', '-{} days')
                GROUP BY stock_code
                ORDER BY count DESC
                LIMIT 5
            '''.format(days))
            popular_stocks = cursor.fetchall()
            
            return {
                'total_queries': total_queries,
                'success_queries': success_queries,
                'success_rate': success_queries / total_queries if total_queries > 0 else 0,
                'popular_stocks': popular_stocks
            }
        except Exception as e:
            print(f"✗ 获取统计数据失败: {e}")
            return {}
    
    def get_model_performance_summary(self) -> List[Dict]:
        """
        获取所有模型的性能汇总
        
        Returns:
            模型性能列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT stock_name, horizon, model_name, accuracy, auc_roc, training_date
                FROM model_metadata
                ORDER BY stock_code, horizon, model_name
            ''')
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            print(f"✗ 获取模型性能汇总失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("✓ 数据库连接已关闭")
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭连接"""
        self.close()


# 测试代码
if __name__ == '__main__':
    print("="*60)
    print("数据库模块测试")
    print("="*60)
    
    # 使用上下文管理器
    with StockPredictionDB('data/test_predictions.db') as db:
        # 测试保存预测结果
        print("\n[测试1] 保存预测结果...")
        test_prediction = {
            'stock_code': '600519.SS',
            'stock_name': '贵州茅台',
            'horizon': 5,
            'predicted_direction': '涨',
            'up_probability': 0.75,
            'confidence': '高',
            'model_used': 'lightgbm'
        }
        
        if db.save_prediction(**test_prediction):
            print("✓ 预测结果保存成功")
        
        # 测试记录查询历史
        print("\n[测试2] 记录查询历史...")
        if db.record_query('600519.SS', '贵州茅台', '127.0.0.1', True):
            print("✓ 查询历史记录成功")
        
        # 测试保存模型元信息
        print("\n[测试3] 保存模型元信息...")
        test_metadata = {
            'stock_code': '600519.SS',
            'stock_name': '贵州茅台',
            'horizon': 5,
            'model_name': 'lightgbm',
            'accuracy': 0.72,
            'precision_score': 0.68,
            'recall': 0.75,
            'f1_score': 0.71,
            'auc_roc': 0.78,
            'model_path': 'models/600519_SS_horizon5_lightgbm.pkl',
            'feature_count': 28,
            'sample_count': 726
        }
        
        if db.save_model_metadata(test_metadata):
            print("✓ 模型元信息保存成功")
        
        # 测试查询预测历史
        print("\n[测试4] 查询预测历史...")
        history = db.get_prediction_history('600519.SS', limit=5)
        print(f"找到 {len(history)} 条预测记录")
        for record in history:
            print(f"  - {record['prediction_date']}: {record['predicted_direction']} "
                  f"(概率: {record['up_probability']:.2%})")
        
        # 测试查询统计数据
        print("\n[测试5] 查询统计数据...")
        stats = db.get_query_statistics(days=7)
        print(f"总查询次数: {stats.get('total_queries', 0)}")
        print(f"成功次数: {stats.get('success_queries', 0)}")
        print(f"成功率: {stats.get('success_rate', 0):.2%}")
        
        # 测试模型性能汇总
        print("\n[测试6] 模型性能汇总...")
        performance = db.get_model_performance_summary()
        print(f"共 {len(performance)} 个模型记录")
        for model in performance[:3]:
            print(f"  - {model['stock_name']} ({model['horizon']}天): "
                  f"{model['model_name']} AUC={model['auc_roc']:.4f}")
    
    print("\n" + "="*60)
    print("✓ 所有测试完成！")
    print("="*60)
