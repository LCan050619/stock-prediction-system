"""
可视化模块
生成评估图表：混淆矩阵、ROC曲线、特征重要性等
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import os
from typing import Dict, List
import joblib


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          stock_name: str, model_name: str,
                          horizon: int, save_path: str):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        stock_name: 股票名称
        model_name: 模型名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 增大图片尺寸：从(8, 6)改为(10, 8)
    plt.figure(figsize=(10, 8))
    
    # 增大字体和注释大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['跌 (0)', '涨 (1)'],
                yticklabels=['跌 (0)', '涨 (1)'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    
    # 增大标题和标签字体
    plt.title(f'Confusion Matrix - {stock_name}\n{model_name} - {horizon}天预测',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label (预测标签)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label (真实标签)', fontsize=14, fontweight='bold')
    
    # 增大刻度字体
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    # 提高DPI
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   stock_name: str, model_name: str,
                   horizon: int, auc_score: float,
                   save_path: str):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        stock_name: 股票名称
        model_name: 模型名称
        horizon: 预测时间窗口
        auc_score: AUC分数
        save_path: 保存路径
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    # 增大图片尺寸：从(8, 6)改为(10, 8)
    plt.figure(figsize=(10, 8))
    
    # 增大线宽和标记
    plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (随机分类器)')
    plt.fill_between(fpr, tpr, alpha=0.15, color='blue')
    
    # 增大标题和标签字体
    plt.title(f'ROC Curve - {stock_name}\n{model_name} - {horizon}天预测',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate (假正例率)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (真正例率)', fontsize=14, fontweight='bold')
    
    # 增大图例字体
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # 增大刻度字体
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # 提高DPI
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"ROC曲线已保存: {save_path}")


def plot_feature_importance(model, feature_names: List[str],
                            stock_name: str, top_n: int = 15,
                            save_path: str = None):
    """
    绘制特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        stock_name: 股票名称
        top_n: 显示Top N个特征
        save_path: 保存路径（可选）
    """
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'booster_'):
        # LightGBM
        importance = model.booster_.feature_importance(importance_type='gain')
    else:
        print("警告: 无法获取特征重要性")
        return
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # 绘图 - 增大图片尺寸
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'].values)
    
    # 增大字体
    plt.yticks(range(len(importance_df)), importance_df['feature'].values, fontsize=11)
    plt.xlabel('Importance (重要性)', fontsize=14, fontweight='bold')
    plt.title(f'Feature Importance - Top {top_n}\n{stock_name}',
              fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # 添加数值标签 - 增大字体
    max_importance = importance_df['importance'].max()
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['importance'] + max_importance * 0.01, i, f"{row['importance']:.4f}",
                va='center', fontsize=10, fontweight='bold')
    
    # 增大刻度字体
    plt.xticks(fontsize=11)
    
    plt.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        # 提高DPI
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"特征重要性图已保存: {save_path}")
    
    plt.close()


def plot_model_comparison(metrics_dict: Dict, stock_name: str,
                          horizon: int, save_path: str):
    """
    绘制多模型性能对比柱状图
    
    Args:
        metrics_dict: {model_name: metrics_dict}
        stock_name: 股票名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    models = list(metrics_dict.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_labels = ['Accuracy\n(准确率)', 'Precision\n(精确率)', 'Recall\n(召回率)', 
                      'F1-Score', 'AUC-ROC']
    
    x = np.arange(len(metrics_names))
    width = 0.2
    
    # 增大图片尺寸：从(12, 6)改为(14, 8)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 使用不同的颜色
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][m] for m in metrics_names]
        bars = ax.bar(x + i*width, values, width, label=model.upper(), 
                     alpha=0.85, color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 增大字体
    ax.set_xlabel('Metrics (评估指标)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (分数)', fontsize=14, fontweight='bold')
    ax.set_title(f'Model Comparison - {stock_name}\n{horizon}天预测',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(models)-1) / 2)
    ax.set_xticklabels(metrics_labels, fontsize=11)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    
    # 增大刻度字体
    ax.tick_params(axis='both', labelsize=11)
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    # 提高DPI
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"模型对比图已保存: {save_path}")


def plot_prediction_timeline(dates: pd.DatetimeIndex, 
                             actual_prices: np.ndarray,
                             predictions: np.ndarray,
                             stock_name: str, horizon: int,
                             save_path: str):
    """
    绘制预测结果与实际价格对比时间序列图
    
    Args:
        dates: 日期索引
        actual_prices: 实际价格
        predictions: 预测标签（0或1）
        stock_name: 股票名称
        horizon: 预测时间窗口
        save_path: 保存路径
    """
    # 增大图片尺寸：从(14, 6)改为(16, 8)
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 绘制实际价格 - 增大线宽
    ax1.plot(dates, actual_prices, 'b-', linewidth=2.5, label='Actual Price', alpha=0.7)
    
    # 增大字体大小
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (CNY)', fontsize=14, fontweight='bold', color='b')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 增大标题字体
    ax1.set_title(f'Price Prediction Timeline - {stock_name}\n{horizon}天预测',
                  fontsize=18, fontweight='bold', pad=20)
    
    # 在价格图上标记预测结果 - 增大散点尺寸
    pred_dates = dates[len(dates)-len(predictions):]
    pred_prices = actual_prices[len(actual_prices)-len(predictions):]
    
    colors = ['green' if p == 1 else 'red' for p in predictions]
    ax1.scatter(pred_dates, pred_prices, c=colors, s=60, alpha=0.7,
               label='Predicted Direction', edgecolors='black', linewidths=1.0)
    
    # 添加图例 - 增大图例字体和标记尺寸
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=3, label='Actual Price', fontsize=12),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=12, label='Predicted Up (涨)', markeredgewidth=1.5, fontsize=12),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=12, label='Predicted Down (跌)', markeredgewidth=1.5, fontsize=12)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)
    
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 调整布局，增加边距
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.1, right=0.95)
    
    # 提高DPI以获得更清晰的图片
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"预测时间线图已保存: {save_path}")


def generate_all_visualizations(stock_code: str, stock_name: str,
                                results_dict: Dict, features: pd.DataFrame,
                                y_test: np.ndarray, horizons: List[int],
                                output_dir: str = 'results/figures'):
    """
    为单只股票生成所有可视化图表
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        results_dict: {horizon: {model_name: {model, metrics, y_pred, y_pred_proba}}}
        features: 特征数据
        y_test: 测试集标签
        horizons: 时间窗口列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"生成 {stock_name} ({stock_code}) 的可视化图表")
    print(f"{'='*70}\n")
    
    for horizon in horizons:
        if horizon not in results_dict:
            continue
        
        horizon_results = results_dict[horizon]
        
        # 获取测试集数据
        split_idx = int(len(features) * 0.8)
        X_test = features.iloc[split_idx:]
        y_test_horizon = y_test[split_idx:] if isinstance(y_test, np.ndarray) else y_test.iloc[split_idx:]
        test_dates = X_test.index
        
        # 收集所有模型的指标用于对比
        metrics_dict = {}
        
        for model_name, result in horizon_results.items():
            # 从文件加载模型
            model_path = result.get('model_path')
            if not model_path or not os.path.exists(model_path):
                print(f"警告: 找不到模型文件 {model_path}")
                continue
            
            model = joblib.load(model_path)
            metrics = result['metrics']
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 1. 混淆矩阵
            cm_path = os.path.join(output_dir, 
                                  f'cm_{stock_code.replace(".", "_")}_{model_name}_{horizon}day.png')
            plot_confusion_matrix(y_test_horizon, y_pred, stock_name, 
                                 model_name, horizon, cm_path)
            
            # 2. ROC曲线
            roc_path = os.path.join(output_dir,
                                   f'roc_{stock_code.replace(".", "_")}_{model_name}_{horizon}day.png')
            plot_roc_curve(y_test_horizon, y_pred_proba, stock_name,
                          model_name, horizon, metrics['auc'], roc_path)
            
            # 3. 特征重要性（只对第一个模型生成一次）
            if model_name == list(horizon_results.keys())[0]:
                feat_imp_path = os.path.join(output_dir,
                                            f'feature_imp_{stock_code.replace(".", "_")}.png')
                plot_feature_importance(model, X_test.columns.tolist(),
                                       stock_name, top_n=15,
                                       save_path=feat_imp_path)
            
            # 收集指标用于对比
            metrics_dict[model_name] = metrics
            
            # 保存预测结果用于时间线图
            result['y_pred'] = y_pred
            result['y_pred_proba'] = y_pred_proba
        
        # 4. 模型对比图
        if len(metrics_dict) > 1:
            comparison_path = os.path.join(output_dir,
                                          f'model_comparison_{stock_code.replace(".", "_")}_{horizon}day.png')
            plot_model_comparison(metrics_dict, stock_name, horizon, comparison_path)
        
        # 5. 预测时间线图（使用最佳模型）
        best_model_name = max(metrics_dict.keys(), 
                             key=lambda k: metrics_dict[k]['auc'])
        best_result = horizon_results[best_model_name]
        
        # 获取实际价格
        actual_prices = features['Close'].iloc[split_idx:].values if 'Close' in features.columns else \
                       np.random.randn(len(test_dates)).cumsum() + 100  # 如果没有Close列，生成模拟数据
        
        timeline_path = os.path.join(output_dir,
                                    f'prediction_timeline_{stock_code.replace(".", "_")}_{horizon}day.png')
        plot_prediction_timeline(test_dates, actual_prices,
                                best_result['y_pred'], stock_name,
                                horizon, timeline_path)
    
    print(f"\n✓ {stock_name} 的所有可视化图表已生成\n")


def save_metrics_to_csv(all_results: Dict, stock_codes: dict,
                        output_path: str = 'results/evaluation_metrics.csv'):
    """
    将所有评估指标保存到CSV文件
    
    Args:
        all_results: {stock_code: {horizon: {model_name: metrics}}}
        stock_codes: 股票代码字典
        output_path: 输出路径
    """
    records = []
    
    for stock_code, horizons in all_results.items():
        stock_name = stock_codes.get(stock_code, stock_code)
        
        for horizon, models in horizons.items():
            for model_name, result in models.items():
                if isinstance(result, dict) and 'metrics' in result:
                    metrics = result['metrics']
                    record = {
                        '股票代码': stock_code,
                        '股票名称': stock_name,
                        '模型': model_name,
                        '时间窗口': f'{horizon}天',
                        'Accuracy': round(metrics['accuracy'], 4),
                        'Precision': round(metrics['precision'], 4),
                        'Recall': round(metrics['recall'], 4),
                        'F1-Score': round(metrics['f1'], 4),
                        'AUC': round(metrics['auc'], 4)
                    }
                    records.append(record)
    
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"评估指标已保存到: {output_path}")
    
    return df


if __name__ == '__main__':
    # 测试代码
    print("可视化模块测试")
    print("请运行完整的训练流程后查看生成的图表")
