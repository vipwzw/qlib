# DeepSeek 情感分析使用指南

## 概述

本项目已将 **DeepSeek API** 设为默认的情感分析引擎，提供高精度的新闻情感分析能力。

## 配置状态 ✅

- ✅ **配置文件**: `configs/config.yaml` 中已将 `default_engine` 设为 `deepseek`
- ✅ **回测系统**: `real_backtest.py` 默认使用 DeepSeek 进行情感分析
- ✅ **因子构建**: `factor_construction.py` 默认使用 DeepSeek 进行情感分析
- ✅ **回退机制**: 在 DeepSeek 不可用时自动回退到关键词分析
- ✅ **多线程支持**: 支持最多 50 个并发线程，大幅提升分析速度

## 使用方法

### 1. 运行回测策略

```bash
# 直接运行，系统会自动使用DeepSeek进行情感分析
python scripts/real_backtest.py
```

### 2. 手动分析新闻文件

```bash
# 分析单个新闻文件
python scripts/analyze_news_sentiment.py data/raw/news/news_20250522.csv

# 随机采样分析（用于测试）
python scripts/analyze_news_sentiment.py data/raw/news/news_20250522.csv --sample 100
```

### 3. 构建情感因子

```bash
# 系统会自动使用DeepSeek分析新闻情感
python scripts/factor_construction.py
```

## 配置参数

在 `configs/config.yaml` 中的 DeepSeek 配置：

```yaml
sentiment:
  default_engine: deepseek  # 默认引擎
  
  deepseek:
    enabled: true
    api_key: ${DEEPSEEK_API_KEY}
    max_workers: 50      # 并发线程数
    batch_size: 10       # 批量处理大小
    timeout: 30          # 请求超时时间
    max_retries: 3       # 最大重试次数
```

## 性能优化

- **并发处理**: 默认使用 50 个线程并发分析，大幅提升速度
- **智能缓存**: 自动缓存已分析的新闻，避免重复计算
- **批量处理**: 支持批量分析模式，减少API调用次数
- **自动重试**: 失败时自动重试，提高稳定性

## 回退机制

系统具有多层回退保障：

1. **优先级 1**: DeepSeek API 情感分析（默认）
2. **优先级 2**: 检查已有的 DeepSeek 分析结果
3. **优先级 3**: 简单关键词情感分析（回退方案）

## 输出格式

DeepSeek 分析会为每条新闻提供以下字段：

- `deepseek_sentiment_score`: 情感得分 (-1 到 1)
- `deepseek_confidence`: 置信度 (0 到 1)
- `deepseek_sentiment_label`: 情感标签 ("正面"/"中性"/"负面")
- `deepseek_market_impact`: 市场影响 ("利好"/"中性"/"利空")

## 注意事项

1. **API 密钥**: 确保 `.env` 文件中已设置 `DEEPSEEK_API_KEY`
2. **网络连接**: 需要稳定的网络连接访问 DeepSeek API
3. **费用控制**: DeepSeek API 按使用量计费，注意控制成本
4. **速率限制**: 系统已自动处理 API 速率限制，无需手动调整

## 验证配置

如果需要验证 DeepSeek 是否正确配置为默认引擎，可以检查：

1. 运行回测时的日志中是否出现 "🤖 使用DeepSeek进行情感分析"
2. 检查新闻数据是否包含 `deepseek_sentiment_score` 字段
3. 查看 API 调用统计信息

## 故障排除

如果遇到问题：

1. **检查 API 密钥**: 确保 `DEEPSEEK_API_KEY` 正确设置
2. **查看日志**: 检查错误日志获取详细信息
3. **网络问题**: 确认可以访问 `https://api.deepseek.com`
4. **回退模式**: 系统会自动回退到关键词分析，不会中断运行

---

**提示**: 系统已经过全面测试，确保 DeepSeek 作为默认情感分析引擎正常工作！ 