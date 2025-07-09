# 新闻情感量化因子分析项目

基于qlib框架的BTC/USDT新闻情感量化因子分析与策略开发项目。

## 项目概述

本项目通过分析新闻情感对加密货币价格的影响，构建有效的量化因子，用于BTC/USDT的交易策略开发。项目集成了多种情感分析模型（VADER、FinBERT、TextBlob），并提供了完整的数据采集、因子构建、评估和回测流程。

## 主要功能

- **多源新闻数据采集**: 支持RSS源、API接口、网页爬取
- **智能情感分析**: 集成多种NLP模型的情感分析引擎
- **因子工程**: 构建多时间维度的情感量化因子
- **因子评估**: 完整的IC分析和因子表现评估
- **策略回测**: 基于qlib的专业回测框架
- **实时监控**: 支持实时数据更新和因子监控

## 项目结构

```
examples/news_factor/
├── README.md                           # 项目说明文档
├── requirements.txt                    # 依赖包列表
├── 新闻量化因子分析需求文档.md          # 详细需求文档
├── 技术实现方案示例.py                 # 核心技术实现
├── configs/                           # 配置文件目录
│   ├── config.yaml                    # 主配置文件
│   ├── news_sources.yaml             # 新闻源配置
│   └── workflow_config.yaml          # 工作流配置
├── scripts/                          # 脚本目录
│   ├── data_collection.py            # 数据采集脚本
│   ├── sentiment_analysis.py         # 情感分析脚本
│   ├── factor_construction.py        # 因子构建脚本
│   ├── factor_evaluation.py          # 因子评估脚本
│   └── run_backtest.py               # 回测执行脚本
├── models/                           # 模型目录
│   ├── sentiment_analyzer.py         # 情感分析器
│   ├── factor_builder.py             # 因子构建器
│   └── custom_handler.py             # 自定义数据处理器
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后数据
│   └── factors/                      # 因子数据
└── notebooks/                        # Jupyter笔记本
    ├── data_exploration.ipynb        # 数据探索
    ├── sentiment_analysis_demo.ipynb # 情感分析演示
    └── factor_analysis.ipynb         # 因子分析
```

## 快速开始

### 1. 环境准备

```bash
# 确保您已经在qlib项目根目录下
# 如果还没有克隆qlib项目，请执行：
# git clone https://github.com/microsoft/qlib.git
# cd qlib

# 进入新闻因子项目目录
cd examples/news_factor

# 安装项目依赖（包括本地qlib开发版本）
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置模板并修改
cp examples/news_factor/configs/config.yaml.template examples/news_factor/configs/config.yaml

# 编辑配置文件，填入API密钥等信息
vim examples/news_factor/configs/config.yaml
```

### 3. 数据准备

```bash
# 采集BTC/USDT价格数据
python examples/news_factor/scripts/data_collection.py --data-type price

# 采集新闻数据
python examples/news_factor/scripts/data_collection.py --data-type news

# 处理数据并构建因子
python examples/news_factor/scripts/factor_construction.py
```

### 4. 运行分析

```bash
# 因子评估
python examples/news_factor/scripts/factor_evaluation.py

# 策略回测
python examples/news_factor/scripts/run_backtest.py
```

## 核心模块说明

### 数据采集模块 (`scripts/data_collection.py`)
- 支持Binance API获取BTC/USDT 1分钟数据
- 多源新闻采集（CoinDesk、CoinTelegraph等）
- 实时数据更新和增量采集

### 情感分析模块 (`models/sentiment_analyzer.py`)
- VADER词典法情感分析
- FinBERT金融领域预训练模型
- TextBlob基础情感分析
- 多模型集成策略

### 因子构建模块 (`models/factor_builder.py`)
- 多时间维度聚合（1m、5m、15m、1h）
- 情感得分、波动率、新闻量等基础因子
- 动量、相关性等复合因子
- 与价格数据的时间对齐

### 因子评估模块 (`scripts/factor_evaluation.py`)
- IC（信息系数）分析
- Rank IC计算
- 因子稳定性测试
- 收益率分解

## 配置说明

### 主配置文件 (`configs/config.yaml`)

```yaml
# 数据配置
data:
  symbol: "BTC/USDT"
  timeframe: "1m"
  lookback_days: 30
  
# 新闻源配置
news_sources:
  coindesk:
    url: "https://www.coindesk.com/arc/outboundfeeds/rss/"
    weight: 0.4
  cointelegraph:
    url: "https://cointelegraph.com/rss"
    weight: 0.3
    
# 情感分析配置
sentiment:
  models:
    vader: {enabled: true, weight: 0.3}
    finbert: {enabled: true, weight: 0.5}
    textblob: {enabled: true, weight: 0.2}
    
# 因子配置
factors:
  timeframes: ["1m", "5m", "15m", "1h"]
  rolling_windows: [5, 10, 20, 60]
```

## 使用示例

### 1. 基础情感分析

```python
from models.sentiment_analyzer import SentimentAnalyzer

# 初始化分析器
analyzer = SentimentAnalyzer()

# 分析单条新闻
text = "Bitcoin reaches new all-time high as institutional adoption grows"
result = analyzer.ensemble_sentiment(text)
print(f"情感得分: {result['ensemble_score']:.4f}")
```

### 2. 因子构建

```python
from models.factor_builder import NewsSentimentFactorBuilder

# 构建因子
builder = NewsSentimentFactorBuilder(price_data, news_data)
factors = builder.build_factors()

# 查看因子
print(f"构建了 {len(factors.columns)} 个因子")
print(factors.head())
```

### 3. 因子评估

```python
from scripts.factor_evaluation import FactorEvaluator

# 评估因子
evaluator = FactorEvaluator(factors, returns)
performance = evaluator.factor_performance_summary('sentiment_score_1m')

print(f"IC均值: {performance['ic_mean']:.4f}")
print(f"IC_IR: {performance['ic_ir']:.4f}")
```

## 结果解读

### 因子表现指标
- **IC均值**: 信息系数的平均值，反映因子预测能力
- **IC_IR**: IC信息比率，衡量因子稳定性
- **正IC比例**: 正向预测准确率
- **因子收益**: 基于因子的收益表现

### 策略表现指标
- **年化收益率**: 策略的年化投资回报
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大亏损幅度
- **胜率**: 盈利交易比例

## 进阶功能

### 1. 自定义新闻源
在 `configs/news_sources.yaml` 中添加新的新闻源：

```yaml
custom_source:
  name: "Custom News"
  url: "https://example.com/rss"
  weight: 0.2
  keywords: ["bitcoin", "crypto"]
```

### 2. 自定义情感模型
继承 `SentimentAnalyzer` 类添加新的分析模型：

```python
class CustomSentimentAnalyzer(SentimentAnalyzer):
    def analyze_custom_model(self, text: str) -> Dict[str, float]:
        # 自定义情感分析逻辑
        return {"score": 0.0}
```

### 3. 因子优化
使用因子选择和组合优化：

```python
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor

# 因子选择
selector = SelectKBest(k=10)
selected_factors = selector.fit_transform(factors, returns)
```

## 常见问题

### Q1: 如何获取更多历史数据？
A: 修改 `config.yaml` 中的 `lookback_days` 参数，或使用 `--start-date` 参数指定起始时间。

### Q2: 如何添加新的技术指标？
A: 在 `factor_builder.py` 中的 `build_factors()` 方法里添加新的计算逻辑。

### Q3: 如何调整情感分析模型权重？
A: 修改 `config.yaml` 中 `sentiment.models` 部分的权重配置。

### Q4: 如何处理数据缺失？
A: 项目提供了多种数据处理方法，包括前向填充、插值等，可在配置文件中调整。

## 性能优化

### 1. 数据缓存
- 启用Redis缓存以提高数据访问速度
- 使用增量更新减少重复计算

### 2. 并行处理
- 多线程新闻采集
- 批量情感分析处理
- 分布式因子计算

### 3. 模型优化
- 预编译正则表达式
- 缓存模型推理结果
- 使用GPU加速深度学习模型

## 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。详情请参阅qlib项目的LICENSE文件。

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 问题反馈: 请在GitHub Issues中提交

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 基础数据采集和情感分析功能
- 因子构建和评估框架
- 策略回测系统

---

## 相关资源

- [Qlib官方文档](https://qlib.readthedocs.io/)
- [FinBERT模型](https://huggingface.co/ProsusAI/finbert)
- [VADER情感分析](https://github.com/cjhutto/vaderSentiment)
- [加密货币API文档](https://github.com/ccxt/ccxt)

## 致谢

感谢Microsoft Qlib团队提供优秀的量化投资框架，以及开源社区提供的各种NLP工具和模型。 