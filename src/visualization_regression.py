"""
回归任务可视化模块
生成回归评估图表：预测vs实际、残差分析、误差分布等
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from typing import Dict, List
import joblib


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              stock_name: str, model_name: str,
                              horizon: int, save_path: str):
    """
    绘制预测价格 vs 实际价格散点图
    
    Args:
        y_true: 真实价格
        y_pred: 预测价格
        stock_name: 股票名称
        model_name: 模型名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # 理想对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Actual Price (实际价格)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Price (预测价格)', fontsize=14, fontweight='bold')
    plt.title(f'Prediction vs Actual - {stock_name}\n{model_name} - {horizon}天预测 (R²={r2:.4f})',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 增大刻度字体
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               stock_name: str, model_name: str,
                               horizon: int, save_path: str):
    """
    绘制残差分布直方图
    
    Args:
        y_true: 真实价格
        y_pred: 预测价格
        stock_name: 股票名称
        model_name: 模型名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # 直方图 + KDE
    sns.histplot(residuals, bins=40, kde=True, color='steelblue', 
                edgecolor='navy', linewidth=0.5)
    
    # 均值线
    mean_res = residuals.mean()
    plt.axvline(x=mean_res, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_res:.2f}')
    plt.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)
    
    plt.xlabel('Residual (残差: 实际-预测)', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (频数)', fontsize=14, fontweight='bold')
    plt.title(f'Residual Distribution - {stock_name}\n{model_name} - {horizon}天预测',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 增大刻度字体
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")


def plot_residual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                               stock_name: str, model_name: str,
                               horizon: int, save_path: str):
    """
    绘制残差 vs 预测值散点图（检查异方差性）
    
    Args:
        y_true: 真实价格
        y_pred: 预测价格
        stock_name: 股票名称
        model_name: 模型名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_pred, residuals, alpha=0.5, s=30, color='steelblue', 
               edgecolors='navy', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # 添加±2标准差线
    std_res = residuals.std()
    plt.axhline(y=2*std_res, color='orange', linestyle=':', linewidth=1.5, 
               label=f'±2σ ({2*std_res:.2f})')
    plt.axhline(y=-2*std_res, color='orange', linestyle=':', linewidth=1.5)
    
    plt.xlabel('Predicted Price (预测价格)', fontsize=14, fontweight='bold')
    plt.ylabel('Residual (残差)', fontsize=14, fontweight='bold')
    plt.title(f'Residual vs Predicted - {stock_name}\n{model_name} - {horizon}天预测',
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 增大刻度字体
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")


def plot_error_metrics_comparison(results: Dict, stock_name: str,
                                  horizons: List[int], save_dir: str):
    """
    绘制不同模型和时间窗口的误差指标对比
    
    Args:
        results: 包含所有模型结果的字典
        stock_name: 股票名称
        horizons: 预测时间窗口列表
        save_dir: 保存目录
    """
    metrics_to_plot = ['rmse', 'mae', 'r2']
    metric_names = {'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²'}
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_positions = []
        bar_width = 0.25
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for idx, horizon in enumerate(horizons):
            if horizon not in results:
                continue
            
            values = []
            labels = []
            
            for model_name, model_data in results[horizon].items():
                if 'metrics' in model_data and metric in model_data['metrics']:
                    values.append(model_data['metrics'][metric])
                    labels.append(model_name)
            
            if values:
                x_pos = [i + idx * bar_width for i in range(len(values))]
                x_positions.extend(x_pos)
                ax.bar(x_pos, values, bar_width, label=f'{horizon}天', 
                      color=colors[idx % len(colors)], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model (模型)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_names.get(metric, metric), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_names.get(metric, metric)} Comparison - {stock_name}',
                    fontsize=16, fontweight='bold', pad=20)
        
        # 设置x轴标签
        if labels:
            ax.set_xticks([i + bar_width for i in range(len(labels))])
            ax.set_xticklabels(labels, fontsize=12)
        
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{stock_name}_{metric}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存: {save_path}")


def generate_all_visualizations_regression(stock_code: str, stock_name: str,
                                           results: Dict,
                                           features: pd.DataFrame,
                                           labels: pd.Series,
                                           horizons: List[int]):
    """
    生成所有回归可视化图表
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        results: 包含所有模型结果的字典
        features: 特征数据
        labels: 标签数据
        horizons: 预测时间窗口列表
    """
    print(f"\n{'='*80}")
    print(f" 生成 {stock_name} 的回归可视化图表")
    print(f"{'='*80}")
    
    # 创建保存目录
    viz_dir = f'results/visualizations/{stock_code}'
    os.makedirs(viz_dir, exist_ok=True)
    
    # 为每个时间窗口和模型生成图表
    for horizon in horizons:
        if horizon not in results:
            continue
        
        print(f"\n--- {horizon}天预测窗口 ---")
        
        for model_name, model_data in results[horizon].items():
            if 'model' not in model_data or 'metrics' not in model_data:
                continue
            
            model = model_data['model']
            metrics = model_data['metrics']
            
            print(f"\n生成 {model_name} 图表...")
            
            # 获取测试集预测
            split_idx = int(len(features) * 0.8)
            X_test = features.iloc[split_idx:]
            y_test = labels.iloc[split_idx:]
            
            if len(X_test) == 0:
                print(f"  ⚠️ 测试集为空，跳过")
                continue
            
            y_pred = model.predict(X_test)
            
            # 1. 预测vs实际散点图
            save_path = os.path.join(viz_dir, 
                                    f'{stock_name}_{model_name}_{horizon}d_pred_vs_actual.png')
            plot_prediction_vs_actual(y_test.values, y_pred, stock_name, model_name, 
                                     horizon, save_path)
            
            # 2. 残差分布
            save_path = os.path.join(viz_dir,
                                    f'{stock_name}_{model_name}_{horizon}d_residual_dist.png')
            plot_residual_distribution(y_test.values, y_pred, stock_name, model_name,
                                      horizon, save_path)
            
            # 3. 残差vs预测值
            save_path = os.path.join(viz_dir,
                                    f'{stock_name}_{model_name}_{horizon}d_residual_vs_pred.png')
            plot_residual_vs_predicted(y_test.values, y_pred, stock_name, model_name,
                                      horizon, save_path)
    
    # 4. 误差指标对比图
    print(f"\n生成误差指标对比图...")
    plot_error_metrics_comparison(results, stock_name, horizons, viz_dir)
    
    print(f"\n✓ {stock_name} 回归可视化图表生成完成！")


def save_metrics_to_csv(all_results: Dict, stock_codes: Dict,
                        output_path: str = 'results/evaluation_metrics.csv'):
    """
    保存回归评估指标到CSV文件
    
    Args:
        all_results: 所有股票的训练结果
        stock_codes: 股票代码字典
        output_path: 输出路径
    """
    records = []
    
    for stock_code, stock_name in stock_codes.items():
        if stock_code not in all_results:
            continue
        
        results = all_results[stock_code]
        
        for horizon in results.keys():
            for model_name, model_data in results[horizon].items():
                if 'metrics' not in model_data:
                    continue
                
                metrics = model_data['metrics']
                
                record = {
                    'Stock_Code': stock_code,
                    'Stock_Name': stock_name,
                    'Horizon': horizon,
                    'Model': model_name,
                    'MSE': metrics.get('mse', None),
                    'RMSE': metrics.get('rmse', None),
                    'MAE': metrics.get('mae', None),
                    'R2': metrics.get('r2', None),
                    'MAPE': metrics.get('mape', None)
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 评估指标已保存到: {output_path}")
    
    return df


if __name__ == '__main__':
    # 测试代码
    print("回归可视化模块测试")
    print("=" * 60)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randn(n_samples) * 10 + 100
    y_pred = y_true + np.random.randn(n_samples) * 2
    
    # 测试绘图函数
    plot_prediction_vs_actual(y_true, y_pred, 'Test Stock', 'LightGBM', 5, 
                             'test_pred_vs_actual.png')
    plot_residual_distribution(y_true, y_pred, 'Test Stock', 'LightGBM', 5,
                              'test_residual_dist.png')
    
    print("\n✅ 回归可视化模块测试通过！")
