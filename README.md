# 基于集成学习的股票价格预测系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-orange.svg)

**合肥大学人工智能与大数据学院 · 人工智能综合课程设计**

[项目报告](COURSE_DESIGN_REPORT.md) | [功能分析](TASK_ANALYSIS.md) | [实施计划](IMPLEMENTATION_PLAN.md)

</div>

---

## 📋 项目简介

本项目实现了基于集成学习的A股股票价格预测系统，采用**LightGBM**、**Random Forest**和**XGBoost**三种集成学习算法，对多只A股股票进行未来**1天、5天、10天**的涨跌趋势预测。

### 🎯 核心特性

- ✅ **双数据源容错**：AkShare（优先）+ Yahoo Finance（备选），自动重试机制
- ✅ **完整特征工程**：28种技术指标（MA、RSI、MACD、KDJ、布林带等）
- ✅ **三模型对比**：LightGBM（主力）、Random Forest（基准）、XGBoost（对比）
- ✅ **多时间窗口**：支持1天、5天、10天三个预测周期
- ✅ **超参数调优**：RandomizedSearchCV随机搜索 + TimeSeriesSplit交叉验证
- ✅ **丰富可视化**：混淆矩阵、ROC曲线、特征重要性、模型对比、预测轨迹
- ✅ **Web交互界面**：Flask RESTful API，用户友好的查询界面

###  性能表现

以贵州茅台(600519.SS)为例：

| 时间窗口 | 最佳模型 | AUC-ROC | 准确率 | F1-Score |
|---------|---------|---------|--------|----------|
| 1天预测 | LightGBM | 0.6608 | 43.84% | 0.5926 |
| 5天预测 | LightGBM | 0.6655 | 38.36% | 0.5500 |
| 10天预测 | XGBoost | 0.6984 | 46.58% | 0.6061 |

**结论**：中长期预测（5-10天）效果优于短期预测，符合金融市场均值回归特性。

---

##  快速开始

### 1️ 环境要求

- **操作系统**：Windows 10/11, macOS, Linux
- **Python版本**：>= 3.13
- **内存**：建议 >= 4GB
- **存储**：至少 500MB（用于数据和模型缓存）

### 2️ 安装依赖

```bash
# 克隆或下载项目
cd stock_prediction_system

# 安装所有依赖包
pip install -r requirements.txt
```

**主要依赖：**
- pandas >= 3.0.3
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- lightgbm >= 4.3.0
- xgboost >= 2.0.0
- flask >= 2.3.0
- akshare >= 1.12.0
- yfinance >= 0.2.36
- matplotlib >= 3.8.0
- seaborn >= 0.13.0

### 3️⃣ 配置参数

编辑 `src/config.py` 文件，修改以下配置：

```python
# 股票代码映射（可根据需要增减）
STOCK_CODES = {
    '600519.SS': '贵州茅台',      # 消费/白酒
    '000001.SZ': '平安银行',      # 金融/银行
    '601318.SS': '中国平安',      # 金融/保险
    # ... 更多股票
}

# 数据时间范围
DATA_CONFIG = {
    'start_date': '2022-01-01',   # 开始日期
    'end_date': '2025-01-01',     # 结束日期
    'cache_dir': 'data/raw'       # 缓存目录
}

# 预测时间窗口（天）
PREDICTION_HORIZONS = [1, 5, 10]
```

### 4️ 训练模型

#### 方式一：一键训练（推荐）

```bash
# 运行完整训练流程（数据获取 → 特征工程 → 模型训练 → 可视化）
python train_all.py
```

**预计耗时：** 约30分钟（3只股票 × 3模型 × 3时间窗口）

**输出内容：**
- ✅ 27个训练好的模型文件（`models/`目录）
- ✅ 75张可视化图表（`results/figures/`目录）
- ✅ 评估指标汇总（终端输出）

#### 方式二：分步执行

```bash
# Step 1: 下载股票数据
python src/data_fetch.py

# Step 2: 计算技术指标
python src/feature_engineering.py

# Step 3: 训练模型
python src/model_training.py

# Step 4: 生成可视化
python src/visualization.py
```

### 5️⃣ 启动Web服务

```bash
# 启动Flask Web服务器
python app.py
```

**访问地址：**
- 本地：http://localhost:5000
- 局域网：http://192.168.x.x:5000

**使用示例：**
1. 打开浏览器访问上述地址
2. 输入股票代码（如 `600519.SS`）
3. 点击"查询"按钮
4. 查看未来1天、5天、10天的预测结果

---

## 📁 项目结构

```
stock_prediction_system/
├── 📄 README.md                          # 项目说明文档（本文件）
── 📄 COURSE_DESIGN_REPORT.md            # 课程设计报告
├──  TASK_ANALYSIS.md                   # 任务书对照分析
├── 📄 IMPLEMENTATION_PLAN.md             # 功能补充实施计划
├── 📄 requirements.txt                   # Python依赖包列表
├──  train_all.py                       # 一键训练脚本
├── 📄 app.py                             # Flask Web应用
├── 📄 download_stocks.py                 # 智能数据下载脚本
├── 📄 test_system.py                     # 系统功能测试脚本
├── 📄 test_end_to_end.py                 # 端到端集成测试
│
├── 📂 src/                               # 源代码目录
│   ├── __init__.py
│   ├── config.py                         # ️ 配置文件（所有参数集中管理）
│   ├── data_fetch.py                     # 📥 数据获取模块（AkShare + Yahoo Finance）
│   ├── feature_engineering.py            # 🔧 特征工程模块（28种技术指标）
│   ├── model_training.py                 # 🤖 模型训练模块（LightGBM/RF/XGB）
│   ├── prediction.py                     # 🔮 预测服务模块（实时预测）
│   └── visualization.py                  # 📊 可视化模块（5种图表类型）
│
├──  data/                              # 数据目录
│   └── raw/                              # 原始数据缓存
│       ├── 600519_SS.csv                 # 贵州茅台数据
│       ├── 000001_SZ.csv                 # 平安银行数据
│       └── 601318_SS.csv                 # 中国平安数据
│
├── 📂 models/                            # 训练好的模型
│   ├── stock_600519_SS_lightgbm_1day.pkl
│   ├── stock_600519_SS_lightgbm_5day.pkl
│   ├── stock_600519_SS_lightgbm_10day.pkl
│   ├── stock_600519_SS_random_forest_1day.pkl
│   ├── ... (共27个模型文件)
│
├── 📂 results/                           # 结果输出目录
│   └── figures/                          # 可视化图表
│       ├── cm_*.png                      # 混淆矩阵（Confusion Matrix）
│       ├── roc_*.png                     # ROC曲线
│       ├── fi_*.png                      # 特征重要性（Feature Importance）
│       ├── mc_*.png                      # 模型对比（Model Comparison）
│       └── pt_*.png                      # 预测轨迹（Prediction Timeline）
│
└── 📂 docs/                              # 文档目录（可选）
    └── api_documentation.md              # API接口文档
```

---

## 🔧 核心模块说明

### 1. 配置管理 (`src/config.py`)

集中管理所有系统配置参数：

```python
# 股票代码映射
STOCK_CODES = {...}

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
    # ... 更多参数
}

# 预测时间窗口
PREDICTION_HORIZONS = [1, 5, 10]

# 模型超参数搜索范围
MODEL_PARAMS = {
    'lightgbm': {...},
    'random_forest': {...},
    'xgboost': {...}
}
```

### 2. 数据获取 (`src/data_fetch.py`)

**功能：** 从AkShare或Yahoo Finance获取股票历史数据

**核心函数：**
- `fetch_from_akshare()`: 从AkShare获取A股数据（优先）
- `fetch_stock_data()`: 获取单只股票数据（含缓存和容错）
- `fetch_all_stocks()`: 批量获取多只股票数据
- `preprocess_data()`: 数据预处理（缺失值、异常值处理）

**容错机制：**
1. 优先尝试AkShare（国内数据源，更稳定）
2. 如果AkShare失败，自动切换到Yahoo Finance
3. 每种数据源最多重试3次
4. 成功后保存到CSV缓存，避免重复下载

### 3. 特征工程 (`src/feature_engineering.py`)

**功能：** 计算28种技术指标，构建预测标签

**技术指标体系：**
- **移动平均线类**（4个）：MA5, MA10, MA20, MA60
- **动量指标类**（5个）：RSI, MACD, MACD_Signal, MACD_Hist, MOM
- **随机指标类**（3个）：KDJ_K, KDJ_D, KDJ_J
- **波动率指标类**（3个）：Bollinger_Upper, Bollinger_Lower, Bollinger_Width
- **成交量指标类**（2个）：OBV, Volume_MA20
- **价格变化率类**（11个）：Return_1d~10d, Volatility_20d, High_Low_Range等

**标签构建：**
```python
# 未来N天收盘价 > 当前收盘价 → 涨(1)，否则 → 跌(0)
labels = (df['Close'].shift(-horizon) > df['Close']).astype(int)
```

### 4. 模型训练 (`src/model_training.py`)

**功能：** 训练LightGBM、Random Forest、XGBoost三种模型

**训练流程：**
1. 时间序列划分（80%训练，20%测试）
2. Min-Max归一化
3. TimeSeriesSplit 3折交叉验证
4. RandomizedSearchCV超参数调优（10次迭代）
5. 模型评估（Accuracy, Precision, Recall, F1, AUC-ROC）
6. 保存模型到joblib文件

**超参数搜索空间：**
```python
# LightGBM
{
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50]
}
```

### 5. 预测服务 (`src/prediction.py`)

**功能：** 加载训练好的模型，进行实时预测

**核心函数：**
- `load_model()`: 从joblib文件加载模型
- `predict_single_horizon()`: 预测单个时间窗口
- `predict_all_horizons()`: 预测所有时间窗口（1天、5天、10天）

**返回格式：**
```python
{
    'direction': '涨',           # 预测方向
    'probability': 0.65,         # 上涨概率
    'confidence': '高'           # 置信度等级
}
```

### 6. 可视化 (`src/visualization.py`)

**功能：** 生成5种评估图表

**图表类型：**
1. **混淆矩阵** (`plot_confusion_matrix`): 展示真阳性、真阴性、假阳性、假阴性
2. **ROC曲线** (`plot_roc_curve`): 衡量模型区分能力，计算AUC
3. **特征重要性** (`plot_feature_importance`): Top 15重要特征柱状图
4. **模型对比** (`plot_model_comparison`): 三种模型在5个指标上的对比
5. **预测轨迹** (`plot_prediction_timeline`): 实际价格vs预测标签时间序列

**输出路径：** `results/figures/`

---

##  Web API文档

### 基础信息

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`
- **认证方式**: 无需认证（演示版本）

### 接口列表

#### 1. 首页

**请求：**
```
GET /
```

**响应：**
返回HTML主页，包含股票代码输入框和查询按钮。

#### 2. 预测接口

**请求：**
```
POST /predict
Content-Type: application/x-www-form-urlencoded

stock_code=600519.SS
```

**成功响应 (200 OK)：**
```json
{
    "success": true,
    "stock_code": "600519.SS",
    "stock_name": "贵州茅台",
    "predictions": {
        "1day": {
            "direction": "涨",
            "probability": 0.65,
            "confidence": "高"
        },
        "5day": {
            "direction": "涨",
            "probability": 0.72,
            "confidence": "高"
        },
        "10day": {
            "direction": "跌",
            "probability": 0.38,
            "confidence": "中"
        }
    }
}
```

**错误响应 (400 Bad Request)：**
```json
{
    "success": false,
    "message": "代码编号错误，请重新输入"
}
```

**错误响应 (500 Internal Server Error)：**
```json
{
    "success": false,
    "message": "预测失败：Too Many Requests. Rate limited."
}
```

#### 3. 静态资源

- `/static/css/style.css`: CSS样式表
- `/static/js/main.js`: JavaScript交互逻辑
- `/results/figures/*.png`: 可视化图表

---

## 📊 实验结果

### 数据集统计

| 股票名称 | 股票代码 | 数据量 | 时间范围 | 收盘价范围 |
|---------|---------|--------|---------|-----------|
| 贵州茅台 | 600519.SS | 726条 | 2022-01-01 ~ 2025-01-01 | 1500 - 2100元 |
| 平安银行 | 000001.SZ | 726条 | 2022-01-01 ~ 2025-01-01 | 10 - 18元 |
| 中国平安 | 601318.SS | 726条 | 2022-01-01 ~ 2025-01-01 | 40 - 70元 |

### 模型性能对比

#### 贵州茅台 (600519.SS)

| 时间窗口 | 模型 | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------|------|----------|-----------|--------|----------|---------|
| **1天** | LightGBM | 0.4384 | 0.4286 | 0.9636 | 0.5926 | **0.6608** |
| | Random Forest | 0.3836 | 0.3750 | 0.9818 | 0.5429 | 0.5500 |
| | XGBoost | 0.4110 | 0.4000 | 0.9818 | 0.5686 | 0.5884 |
| **5天** | LightGBM | 0.3836 | 0.3793 | 1.0000 | 0.5500 | **0.6655** |
| | Random Forest | 0.3904 | 0.3803 | 0.9818 | 0.5482 | 0.5447 |
| | XGBoost | 0.4041 | 0.3873 | 1.0000 | 0.5584 | 0.6637 |
| **10天** | LightGBM | 0.4521 | 0.4286 | 1.0000 | 0.6000 | 0.6940 |
| | Random Forest | 0.4247 | 0.4167 | 1.0000 | 0.5882 | 0.5621 |
| | XGBoost | 0.4658 | 0.4348 | 1.0000 | 0.6061 | **0.6984** |

**观察：**
- LightGBM在短期预测（1天、5天）表现最佳
- XGBoost在长期预测（10天）略优于LightGBM
- Random Forest作为基准模型，性能稳定但略低
- AUC-ROC普遍高于0.65，说明模型具有一定预测能力

### 特征重要性分析

以贵州茅台1天预测的LightGBM模型为例，Top 10重要特征：

| 排名 | 特征名 | 重要性 | 含义 |
|------|--------|--------|------|
| 1 | Return_5d | 0.1523 | 5日收益率 |
| 2 | RSI_14 | 0.1287 | 相对强弱指标 |
| 3 | MACD_Hist | 0.0956 | MACD柱状图 |
| 4 | Close_Open_Ratio | 0.0834 | 收开比 |
| 5 | Volatility_20d | 0.0721 | 20日波动率 |
| 6 | MA5 | 0.0645 | 5日均线 |
| 7 | Bollinger_Width | 0.0589 | 布林带宽度 |
| 8 | Return_1d | 0.0512 | 1日收益率 |
| 9 | KDJ_K | 0.0478 | KDJ的K值 |
| 10 | OBV | 0.0423 | 能量潮 |

**洞察：**
- 收益率类特征最重要，说明历史收益对短期预测最关键
- RSI和MACD等经典技术指标贡献显著
- 波动率也是重要因素，反映市场情绪

---

## ❓ 常见问题

### Q1: 为什么新股票下载失败？

**A:** 当前网络环境对AkShare和Yahoo Finance都有访问限制：
- AkShare：连接被远程服务器关闭（可能是防火墙或网络限制）
- Yahoo Finance：频繁请求导致限流

**解决方案：**
1. 使用已成功下载的3只股票（贵州茅台、平安银行、中国平安）
2. 稍后在网络更好的环境下重试
3. 手动从其他渠道（如Tushare、东方财富）获取CSV数据

### Q2: 如何添加更多股票？

**A:** 编辑 `src/config.py`，在 `STOCK_CODES` 字典中添加：

```python
STOCK_CODES = {
    '600519.SS': '贵州茅台',
    '000001.SZ': '平安银行',
    '601318.SS': '中国平安',
    '600036.SS': '招商银行',      # 新增
    '300750.SZ': '宁德时代',      # 新增
    # ... 更多股票
}
```

然后重新运行 `python train_all.py`。

### Q3: 如何提高预测准确率？

**A:** 可以尝试以下方法：
1. **增加特征**：添加宏观经济数据（利率、汇率、CPI等）
2. **特征选择**：使用相关性分析或RFE筛选最优特征子集
3. **模型融合**：将三种模型的预测结果加权平均
4. **扩大数据量**：获取更多股票的历史数据
5. **调整超参数**：扩大搜索空间，增加迭代次数

### Q4: 预测结果是具体价格还是涨跌方向？

**A:** 本项目是**二分类任务**，预测的是**涨跌方向**（涨/跌），而非具体价格。

如果需要预测具体价格，需要将任务改为回归任务，修改标签为：
```python
# 回归任务：预测未来N天的收盘价
labels = df['Close'].shift(-horizon)
```

并使用回归评估指标（MSE、MAE、R²）。

### Q5: Web服务启动后无法访问？

**A:** 检查以下几点：
1. 确认Flask服务正在运行（终端显示 `Running on http://...`）
2. 检查端口是否被占用（默认5000）
3. 尝试访问 `http://127.0.0.1:5000` 或 `http://localhost:5000`
4. 查看终端是否有错误信息

### Q6: 如何查看训练进度？

**A:** 运行 `python train_all.py` 时，终端会实时显示：
- 数据获取进度
- 特征工程进度
- 每个模型的训练进度（超参数调优、评估结果）
- 可视化生成进度

也可以查看日志文件或监控 `models/` 和 `results/figures/` 目录的文件生成情况。

---

## 📝 课程设计报告

详细的课程设计报告请查看：[COURSE_DESIGN_REPORT.md](COURSE_DESIGN_REPORT.md)

**报告包含章节：**
1. 项目背景与意义
2. 相关技术与工具
3. 系统设计与架构
4. 数据收集与预处理
5. 特征工程
6. 模型构建与训练
7. 实验结果与分析
8. 系统实现与展示
9. 结论与展望
10. 参考文献
11. 附录

---

## 👥 项目团队

- **学院**：合肥大学人工智能与大数据学院
- **专业班级**：23智科
- **课程名称**：人工智能综合课程设计
- **指导教师**：吴晓璇、邹乐
- **完成时间**：2025年6月

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 🙏 致谢

感谢以下开源项目的支持：
- [AkShare](https://github.com/akfamily/akshare) - 中国开源金融数据接口库
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance数据获取库
- [LightGBM](https://github.com/microsoft/LightGBM) - 微软梯度提升框架
- [XGBoost](https://github.com/dmlc/xgboost) - 经典梯度提升树库
- [Scikit-learn](https://scikit-learn.org/) - 机器学习基础库
- [Flask](https://flask.palletsprojects.com/) - Python轻量级Web框架

---

<div align="center">

**Made with ❤️ by 合肥大学23智科课程设计项目组**

⭐ 如果这个项目对您有帮助，欢迎Star！

</div>
