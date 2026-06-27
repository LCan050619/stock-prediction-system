/**
 * 股票预测系统前端交互逻辑
 */

/**
 * 执行预测请求
 */
async function predict() {
    const stockCodeInput = document.getElementById('stock-code');
    const stockCode = stockCodeInput.value.trim();
    const errorDiv = document.getElementById('error-message');
    const resultSection = document.getElementById('result-section');
    
    // 验证输入
    if (!stockCode) {
        showError('请输入股票代码');
        return;
    }
    
    // 隐藏之前的结果和错误
    hideError();
    hideResult();
    
    // 显示加载状态
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `stock_code=${encodeURIComponent(stockCode)}`
        });
        
        const data = await response.json();
        
        // 隐藏加载状态
        hideLoading();
        
        if (!data.success) {
            showError(data.message || '预测失败，请稍后重试');
            return;
        }
        
        // 显示结果
        displayResults(data);
        
    } catch (error) {
        hideLoading();
        showError('网络请求失败，请检查网络连接后重试');
        console.error('Prediction error:', error);
    }
}

/**
 * 显示预测结果
 */
function displayResults(data) {
    const resultSection = document.getElementById('result-section');
    const stockTitle = document.getElementById('stock-title');
    
    // 设置标题
    stockTitle.textContent = `${data.stock_name} (${data.stock_code})`;
    
    // 显示三个时间窗口的预测结果
    displayPrediction('pred-1day', data.predictions['1day']);
    displayPrediction('pred-5day', data.predictions['5day']);
    displayPrediction('pred-10day', data.predictions['10day']);
    
    // 显示可视化图表（如果有）
    if (data.visualizations) {
        displayVisualizations(data.visualizations, data.stock_code);
    }
    
    // 显示结果区域
    resultSection.classList.remove('hidden');
    
    // 滚动到结果区域
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * 显示可视化图表
 */
function displayVisualizations(viz, stockCode) {
    const vizSection = document.getElementById('visualization-section');
    
    if (!viz || !vizSection) {
        console.log('No visualizations to display');
        return;
    }
    
    // 设置图表URL（默认使用5天预测的Linear模型图表 - 表现最优）
    const modelComparisonChart = document.getElementById('model-comparison-chart');
    const predVsActualChart = document.getElementById('pred-vs-actual-chart');
    const residualChart = document.getElementById('residual-chart');
    
    const stockPrefix = stockCode.replace('.', '_');
    
    if (viz.model_comparison) {
        modelComparisonChart.src = viz.model_comparison;
    } else {
        modelComparisonChart.src = `/static/figures/model_comparison_${stockPrefix}_5day.png`;
    }
    
    if (viz.prediction_vs_actual) {
        predVsActualChart.src = viz.prediction_vs_actual;
    } else {
        // 【修改】使用Linear模型而非LightGBM，因为Linear表现更好
        // 【关键修复】添加时间戳参数防止浏览器缓存
        const timestamp = new Date().getTime();
        predVsActualChart.src = `/static/figures/pred_vs_actual_${stockPrefix}_linear_5day.png?t=${timestamp}`;
    }
    
    if (viz.residual_distribution) {
        residualChart.src = viz.residual_distribution;
    } else {
        // 【修改】使用Linear模型的残差图
        // 【关键修复】添加时间戳参数防止浏览器缓存
        const timestamp = new Date().getTime();
        residualChart.src = `/static/figures/residual_${stockPrefix}_linear_5day.png?t=${timestamp}`;
    }
    
    // 显示可视化区域
    vizSection.style.display = 'block';
    
    console.log('Visualizations loaded successfully');
}

/**
 * 显示单个预测结果（回归版本）
 */
function displayPrediction(elementId, pred) {
    const element = document.getElementById(elementId);
    
    if (!element) {
        console.error(`Element not found: ${elementId}`);
        return;
    }
    
    if (pred.error) {
        element.innerHTML = `
            <div style="color: #c62828;">
                <strong>❌ 错误</strong><br>
                ${pred.error}
            </div>
        `;
    } else {
        // 回归版本：显示预测价格、当前价格、涨跌幅
        const predictedPrice = pred.predicted_price;
        const currentPrice = pred.current_price;
        const priceChange = pred.price_change;
        const changePercent = pred.change_percent;
        const trend = pred.trend;
        const predictDate = pred.predict_date || '未知';
        
        // 根据趋势设置颜色
        const color = trend === '上涨' ? '#4CAF50' : '#f44336';
        const icon = trend === '上涨' ? '📈' : '';
        const arrow = trend === '上涨' ? '↑' : '↓';
        
        element.innerHTML = `
            <div style="text-align: center;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-size: 13px;">
                    📅 预测日期：<strong>${predictDate}</strong>
                </div>
                <div style="font-size: 14px; color: #666; margin-bottom: 8px;">预测价格</div>
                <strong style="color: ${color}; font-size: 28px;">
                    ${icon} ¥${predictedPrice.toFixed(2)}
                </strong>
                <div style="margin-top: 15px; font-size: 14px; border-top: 1px solid #eee; padding-top: 10px;">
                    <div style="margin-bottom: 6px;">
                        <span style="color: #999;">当前价格：</span>
                        <span style="font-weight: bold;">¥${currentPrice.toFixed(2)}</span>
                    </div>
                    <div style="margin-bottom: 6px;">
                        <span style="color: #999;">价格变化：</span>
                        <span style="color: ${color}; font-weight: bold;">
                            ${arrow} ¥${Math.abs(priceChange).toFixed(2)}
                        </span>
                    </div>
                    <div>
                        <span style="color: #999;">涨跌幅：</span>
                        <span style="color: ${color}; font-weight: bold; font-size: 16px;">
                            ${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%
                        </span>
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * 显示错误信息
 */
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

/**
 * 隐藏错误信息
 */
function hideError() {
    const errorDiv = document.getElementById('error-message');
    errorDiv.classList.add('hidden');
}

/**
 * 隐藏结果区域
 */
function hideResult() {
    const resultSection = document.getElementById('result-section');
    resultSection.classList.add('hidden');
}

/**
 * 显示加载状态
 */
function showLoading() {
    const btn = document.querySelector('.btn-primary');
    const originalText = btn.textContent;
    btn.textContent = '查询中...';
    btn.disabled = true;
    btn.style.opacity = '0.6';
    btn.dataset.originalText = originalText;
}

/**
 * 隐藏加载状态
 */
function hideLoading() {
    const btn = document.querySelector('.btn-primary');
    btn.textContent = btn.dataset.originalText || '查询';
    btn.disabled = false;
    btn.style.opacity = '1';
}

/**
 * 监听回车键
 */
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('stock-code');
    
    if (input) {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                predict();
            }
        });
        
        // 自动聚焦输入框
        input.focus();
    }
});

/**
 * 获取所有支持的股票列表（可选功能）
 */
async function loadStockList() {
    try {
        const response = await fetch('/api/stocks');
        const data = await response.json();
        
        if (data.success) {
            console.log('支持的股票:', data.stocks);
            // 可以在这里动态更新股票列表显示
        }
    } catch (error) {
        console.error('Failed to load stock list:', error);
    }
}

// 页面加载时获取股票列表
if (window.location.pathname === '/') {
    loadStockList();
}
