"""
Flask Web应用主程序
股票价格预测系统Web界面

功能：提供RESTful API接口，处理用户预测请求
      集成数据库记录查询历史和预测结果
      支持双数据源（AkShare优先，Yahoo Finance备用）

主要路由：
    - /: 首页，显示股票代码输入界面
    - /predict: POST接口，处理预测请求并返回结果
    - /history: GET接口，查询预测历史记录
    - /statistics: GET接口，获取系统统计信息

作者：课程设计项目组
日期：2025年6月
版本：1.0
"""
from flask import Flask, render_template, request, jsonify
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from prediction_regression import predict_all_horizons
from config import STOCK_CODES
from database import StockPredictionDB
import pandas as pd

# 尝试导入akshare作为主要数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

# 始终导入yfinance作为备用
import yfinance as yf

app = Flask(__name__)


@app.route('/')
def index():
    """首页：显示输入框"""
    return render_template('index.html', stocks=STOCK_CODES)


@app.route('/test-images')
def test_images():
    """图片加载测试页面"""
    return render_template('test_images.html')


@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    # 支持JSON和表单两种格式
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
        # 获取最近数据用于特征计算（最近3个月）
        recent_data = None
        last_error = None
        data_source_used = None
        
        # 策略1: 优先尝试从缓存加载已下载的数据（最可靠）
        # 【关键修复】将 .SS 转换为 .SH 以匹配缓存文件名
        cache_code = stock_code.replace('.SS', '.SH')
        cache_file = f'data/raw/{cache_code.replace(".", "_")}.csv'
        if os.path.exists(cache_file):
            try:
                print(f"尝试从缓存加载 {stock_code} 的数据...")
                recent_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # 【关键修复】加载至少100条数据，以确保能计算Lag_10等滞后特征
                # Lag_10需要往前推10天，所以需要足够的历史数据
                if len(recent_data) >= 100:
                    recent_data = recent_data.tail(100)
                else:
                    recent_data = recent_data.tail(len(recent_data))
                if not recent_data.empty:
                    data_source_used = "缓存数据"
                    print(f"✓ 从缓存成功加载 {len(recent_data)} 条数据")
            except Exception as e:
                print(f"缓存加载失败: {e}")
                recent_data = None
        
        # 策略2: 如果缓存失败，尝试AkShare
        if recent_data is None or recent_data.empty:
            if AKSHARE_AVAILABLE:
                try:
                    print(f"尝试使用AkShare获取 {stock_code} 的数据...")
                    code = stock_code.split('.')[0]
                    
                    for attempt in range(3):
                        try:
                            recent_data = ak.stock_zh_a_hist(
                                symbol=code,
                                period="daily",
                                adjust="qfq"
                            )
                            if not recent_data.empty:
                                break
                        except Exception as e:
                            if attempt < 2:
                                import time
                                time.sleep(2)
                            else:
                                raise
                    
                    if not recent_data.empty:
                        recent_data = recent_data.rename(columns={
                            '开盘': 'Open',
                            '最高': 'High',
                            '最低': 'Low',
                            '收盘': 'Close',
                            '成交量': 'Volume'
                        })
                        recent_data['日期'] = pd.to_datetime(recent_data['日期'])
                        recent_data.set_index('日期', inplace=True)
                        recent_data = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']]
                        # 【关键修复】保留至少100条数据用于Lag特征计算
                        if len(recent_data) >= 100:
                            recent_data = recent_data.tail(100)
                        else:
                            recent_data = recent_data.tail(len(recent_data))
                        data_source_used = "AkShare"
                        print(f"✓ AkShare成功获取 {len(recent_data)} 条数据")
                    else:
                        raise Exception("AkShare未返回数据")
                        
                except Exception as e:
                    last_error = f"AkShare失败: {str(e)}"
                    print(f"{last_error}")
        
        # 策略3: 如果AkShare失败，尝试Yahoo Finance
        if recent_data is None or recent_data.empty:
            try:
                print(f"尝试使用Yahoo Finance获取 {stock_code} 的数据...")
                ticker = yf.Ticker(stock_code)
                
                for attempt in range(3):
                    try:
                        recent_data = ticker.history(period='3mo')
                        if not recent_data.empty:
                            break
                    except Exception as e:
                        if attempt < 2:
                            import time
                            time.sleep(3)
                        else:
                            raise
                
                if not recent_data.empty:
                    data_source_used = "Yahoo Finance"
                    print(f"✓ Yahoo Finance成功获取 {len(recent_data)} 条数据")
                else:
                    raise Exception("Yahoo Finance未返回数据")
                    
            except Exception as e:
                last_error = f"Yahoo Finance失败: {str(e)}"
                print(f"{last_error}")
        
        # 如果所有方法都失败
        if recent_data is None or recent_data.empty:
            error_msg = f'无法获取股票数据。\n'
            if last_error:
                error_msg += f'{last_error}\n'
            
            # 检查是否有缓存文件
            cache_code = stock_code.replace('.SS', '.SH')
            cache_file = f'data/raw/{cache_code.replace(".", "_")}.csv'
            if os.path.exists(cache_file):
                # 有缓存文件，尝试使用
                try:
                    print("使用缓存文件进行预测...")
                    recent_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    # 【关键修复】保留至少100条数据用于Lag特征计算
                    if len(recent_data) >= 100:
                        recent_data = recent_data.tail(100)
                    else:
                        recent_data = recent_data.tail(len(recent_data))
                    data_source_used = "缓存数据（备用）"
                    print(f"✓ 使用缓存数据 {len(recent_data)} 条")
                except Exception as e:
                    error_msg += '\n建议：请稍后重试或检查网络连接。'
                    return jsonify({
                        'success': False,
                        'message': error_msg
                    }), 500
            else:
                # 没有缓存文件，提示用户使用有数据的股票
                error_msg += '\n\n⚠️ 该股票暂无历史数据缓存。\n'
                error_msg += '请尝试以下已有数据的股票：\n'
                error_msg += '• 600519.SS - 贵州茅台\n'
                error_msg += '• 000001.SZ - 平安银行\n'
                error_msg += '• 601318.SS - 中国平安\n'
                error_msg += '\n或者稍后再试，系统会自动下载并缓存数据。'
                
                return jsonify({
                    'success': False,
                    'message': error_msg
                }), 500
        
        # 预测
        results = predict_all_horizons(stock_code, recent_data)
        
        # 获取股票名称
        stock_name = STOCK_CODES[stock_code]
        
        # 检查是否有错误
        has_error = any('error' in result for result in results.values())
        if has_error:
            error_msgs = [results[k]['error'] for k in results if 'error' in results[k]]
            
            # 记录失败的查询
            try:
                with StockPredictionDB() as db:
                    db.record_query(stock_code, stock_name, request.remote_addr, False)
            except Exception as e:
                print(f"警告: 记录查询历史失败: {e}")
            
            return jsonify({
                'success': False,
                'message': f'预测失败：{"; ".join(error_msgs)}'
            }), 500
        
        # 生成可视化图表URL
        stock_prefix = stock_code.replace('.', '_')
        visualizations = {
            'model_comparison': f'/static/figures/model_comparison_{stock_prefix}_5day.png',
            # 【关键修复】使用Linear模型而非LightGBM，因为LightGBM已因过拟合被移除
            'prediction_vs_actual': f'/static/figures/pred_vs_actual_{stock_prefix}_linear_5day.png',
            'residual_distribution': f'/static/figures/residual_{stock_prefix}_linear_5day.png'
        }
        
        # 保存预测结果到数据库
        try:
            with StockPredictionDB() as db:
                # 记录成功的查询
                db.record_query(stock_code, stock_name, request.remote_addr, True)
                
                # 保存每个时间窗口的预测结果（回归版本）
                predictions_to_save = []
                for horizon_key, result in results.items():
                    # 【关键修复】从 '1day', '5day', '10day' 提取整数
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
                
                db.save_predictions_batch(predictions_to_save)
                print(f"✓ 已保存 {len(predictions_to_save)} 条预测记录到数据库")
        except Exception as e:
            print(f"警告: 保存预测结果到数据库失败: {e}")
            # 不影响主流程，继续返回结果
        
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
            'message': f'预测失败：{str(e)}'
        }), 500


@app.route('/visualize/<stock_code>')
def visualize(stock_code):
    """展示可视化图表页面"""
    if stock_code not in STOCK_CODES:
        return render_template('error.html', message='无效的股票代码')
    
    stock_name = STOCK_CODES[stock_code]
    return render_template('result.html', 
                          stock_code=stock_code, 
                          stock_name=stock_name)


@app.route('/api/stocks')
def get_stocks():
    """获取所有支持的股票列表API"""
    return jsonify({
        'success': True,
        'stocks': [{'code': code, 'name': name} for code, name in STOCK_CODES.items()]
    })


@app.route('/api/history/<stock_code>')
def get_history(stock_code):
    """获取某只股票的预测历史API"""
    if stock_code not in STOCK_CODES:
        return jsonify({
            'success': False,
            'message': '无效的股票代码'
        }), 400
    
    try:
        limit = request.args.get('limit', 10, type=int)
        
        with StockPredictionDB() as db:
            history = db.get_prediction_history(stock_code, limit=limit)
        
        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'stock_name': STOCK_CODES[stock_code],
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'查询失败: {str(e)}'
        }), 500


@app.route('/api/statistics')
def get_statistics():
    """获取系统统计信息API"""
    try:
        days = request.args.get('days', 7, type=int)
        
        with StockPredictionDB() as db:
            stats = db.get_query_statistics(days=days)
            model_performance = db.get_model_performance_summary()
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'model_performance': model_performance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取统计信息失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("="*60)
    print("股票价格预测系统 Web服务启动")
    print("="*60)
    print(f"支持 {len(STOCK_CODES)} 只股票")
    print("✓ 数据库功能已启用 (SQLite)")
    print("✓ 预测历史自动记录")
    print("✓ 查询统计实时可用")
    print("访问地址: http://localhost:5000")
    print("API文档:")
    print("  - GET  /api/stocks          : 获取股票列表")
    print("  - POST /predict             : 提交预测请求")
    print("  - GET  /api/history/<code>  : 查询预测历史")
    print("  - GET  /api/statistics      : 获取系统统计")
    print("="*60)
    
    # 生产环境用 gunicorn 运行，本地开发用 app.run
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
