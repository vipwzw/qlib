# DeepSeek API 配置指南

## 🔑 API Key 配置

### 方法1: 使用 .env 文件 (推荐)

1. 在 `examples/news_factor/` 目录下创建 `.env` 文件：

```bash
# 在项目目录创建 .env 文件
touch examples/news_factor/.env
```

2. 在 `.env` 文件中添加以下配置：

```bash
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# 可选配置 (使用默认值即可)
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT=30
DEEPSEEK_MAX_RETRIES=3
DEEPSEEK_RATE_LIMIT=0.1
DEEPSEEK_MAX_WORKERS=100
DEEPSEEK_FAST_MODE=false
```

### 方法2: 使用环境变量

```bash
# 临时设置 (当前会话有效)
export DEEPSEEK_API_KEY='sk-your-deepseek-api-key-here'

# 永久设置 (添加到 ~/.bashrc 或 ~/.zshrc)
echo 'export DEEPSEEK_API_KEY="sk-your-deepseek-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 获取 DeepSeek API Key

1. 访问 [DeepSeek 官网](https://platform.deepseek.com/)
2. 注册账号并登录
3. 在控制台中创建 API Key
4. 将 API Key 复制到配置文件中

## ✅ 验证配置

### 方法1: 使用验证脚本 (推荐)

```bash
cd examples/news_factor
python scripts/verify_api_config.py
```

### 方法2: 手动验证

```bash
cd examples/news_factor
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('DEEPSEEK_API_KEY')
if api_key:
    print('✅ API Key 配置成功!')
    print(f'   Key: {api_key[:10]}...')
else:
    print('❌ API Key 未配置')
"
```

## 📁 配置文件位置

- `.env` 文件应放在: `examples/news_factor/.env`
- 项目会自动加载此位置的环境变量

## ⚠️ 安全提示

1. 不要将 `.env` 文件提交到 Git 仓库
2. 已在 `.gitignore` 中忽略 `.env` 文件
3. 不要在代码中硬编码 API Key
4. 定期更换 API Key 以确保安全

## 🛠️ 支持的脚本

以下脚本已支持从 `.env` 文件读取配置：

- `scripts/deepseek_sentiment_analyzer.py` ✅ (高级情感分析器)
- `scripts/analyze_news_sentiment.py` ✅ (新闻情感分析)
- `scripts/verify_api_config.py` ✅ (配置验证工具)

## 💡 使用示例

```bash
# 确保配置了 API Key
cd examples/news_factor

# 运行情感分析
python scripts/analyze_news_sentiment.py data/news_data.csv

# 或使用高级分析器
python scripts/deepseek_sentiment_analyzer.py
``` 