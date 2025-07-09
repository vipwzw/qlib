#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC/USDT 新闻情感量化因子分析 - 技术实现方案示例
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.model.trainer import task_train
from qlib.workflow import R
from qlib.utils import exists_qlib_data, init_instance_by_config
import ccxt
import feedparser
import requests
from transformers import pipeline
from textblob import TextBlob
import logging

# ================================
# 1. 数据采集模块
# ================================

class BTCUSDTCollector:
    """BTC/USDT 价格数据采集器"""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',  # 填入你的API密钥
            'secret': '',  # 填入你的秘钥
            'sandbox': True,  # 测试环境
        })
        self.symbol = 'BTC/USDT'
        
    def fetch_ohlcv(self, timeframe: str = '1m', since: int = None, limit: int = 1000) -> pd.DataFrame:
        """获取OHLCV数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logging.error(f"获取价格数据失败: {e}")
            return pd.DataFrame()

class NewsCollector:
    """新闻数据采集器"""
    
    def __init__(self):
        self.sources = {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'bitcoinmagazine': 'https://bitcoinmagazine.com/.rss/full/'
        }
        self.keywords = ['bitcoin', 'btc', 'cryptocurrency', 'crypto']
        
    def fetch_rss_news(self, source_url: str) -> List[Dict]:
        """从RSS源获取新闻"""
        try:
            feed = feedparser.parse(source_url)
            news_list = []
            
            for entry in feed.entries:
                news_item = {
                    'title': entry.title,
                    'description': entry.description,
                    'published_time': pd.to_datetime(entry.published),
                    'source': feed.feed.title,
                    'link': entry.link,
                    'content': getattr(entry, 'content', [{'value': ''}])[0]['value']
                }
                
                # 过滤包含关键词的新闻
                text_content = f"{news_item['title']} {news_item['description']}".lower()
                if any(keyword in text_content for keyword in self.keywords):
                    news_list.append(news_item)
                    
            return news_list
            
        except Exception as e:
            logging.error(f"获取RSS新闻失败: {e}")
            return []
    
    def collect_all_news(self) -> pd.DataFrame:
        """收集所有来源的新闻"""
        all_news = []
        
        for source_name, source_url in self.sources.items():
            news_list = self.fetch_rss_news(source_url)
            for news in news_list:
                news['source_name'] = source_name
                all_news.append(news)
        
        if all_news:
            df = pd.DataFrame(all_news)
            df.set_index('published_time', inplace=True)
            return df.sort_index()
        else:
            return pd.DataFrame()

# ================================
# 2. 情感分析模块
# ================================

class SentimentAnalyzer:
    """情感分析引擎"""
    
    def __init__(self):
        # 初始化多个情感分析模型
        self.vader_analyzer = self._init_vader()
        self.finbert_analyzer = self._init_finbert()
        
    def _init_vader(self):
        """初始化VADER分析器"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            logging.warning("VADER分析器未安装")
            return None
            
    def _init_finbert(self):
        """初始化FinBERT分析器"""
        try:
            return pipeline("sentiment-analysis", 
                          model="ProsusAI/finbert",
                          tokenizer="ProsusAI/finbert")
        except Exception as e:
            logging.warning(f"FinBERT分析器初始化失败: {e}")
            return None
    
    def analyze_vader(self, text: str) -> Dict[str, float]:
        """VADER情感分析"""
        if self.vader_analyzer is None:
            return {'compound': 0.0}
            
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
    def analyze_finbert(self, text: str) -> Dict[str, float]:
        """FinBERT情感分析"""
        if self.finbert_analyzer is None:
            return {'score': 0.0, 'label': 'neutral'}
            
        try:
            result = self.finbert_analyzer(text[:512])  # FinBERT限制长度
            label_mapping = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            
            return {
                'score': result[0]['score'] * label_mapping.get(result[0]['label'], 0.0),
                'label': result[0]['label'],
                'confidence': result[0]['score']
            }
        except Exception as e:
            logging.error(f"FinBERT分析失败: {e}")
            return {'score': 0.0, 'label': 'neutral'}
    
    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """TextBlob情感分析"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logging.error(f"TextBlob分析失败: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def ensemble_sentiment(self, text: str) -> Dict[str, float]:
        """集成多个模型的情感分析结果"""
        vader_result = self.analyze_vader(text)
        finbert_result = self.analyze_finbert(text)
        textblob_result = self.analyze_textblob(text)
        
        # 加权平均（可以根据模型表现调整权重）
        weights = {'vader': 0.3, 'finbert': 0.5, 'textblob': 0.2}
        
        ensemble_score = (
            vader_result['compound'] * weights['vader'] +
            finbert_result['score'] * weights['finbert'] +
            textblob_result['polarity'] * weights['textblob']
        )
        
        return {
            'ensemble_score': ensemble_score,
            'vader_compound': vader_result['compound'],
            'finbert_score': finbert_result['score'],
            'textblob_polarity': textblob_result['polarity'],
            'confidence': finbert_result.get('confidence', 0.5)
        }

# ================================
# 3. 因子构建模块
# ================================

class NewsSentimentFactorBuilder:
    """新闻情感因子构建器"""
    
    def __init__(self, price_data: pd.DataFrame, news_data: pd.DataFrame):
        self.price_data = price_data
        self.news_data = news_data
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def calculate_news_sentiment(self) -> pd.DataFrame:
        """计算新闻情感得分"""
        sentiment_scores = []
        
        for idx, row in self.news_data.iterrows():
            text_content = f"{row['title']} {row['description']}"
            sentiment = self.sentiment_analyzer.ensemble_sentiment(text_content)
            sentiment['timestamp'] = idx
            sentiment['source'] = row['source_name']
            sentiment_scores.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiment_scores)
        sentiment_df.set_index('timestamp', inplace=True)
        return sentiment_df
    
    def aggregate_sentiment_by_timeframe(self, sentiment_df: pd.DataFrame, 
                                       timeframe: str = '1T') -> pd.DataFrame:
        """按时间窗口聚合情感得分"""
        agg_functions = {
            'ensemble_score': ['mean', 'std', 'count'],
            'confidence': 'mean'
        }
        
        aggregated = sentiment_df.resample(timeframe).agg(agg_functions)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        
        # 计算额外特征
        aggregated['sentiment_volatility'] = aggregated['ensemble_score_std'].fillna(0)
        aggregated['news_volume'] = aggregated['ensemble_score_count'].fillna(0)
        aggregated['sentiment_momentum'] = aggregated['ensemble_score_mean'].diff()
        
        return aggregated
    
    def build_factors(self) -> pd.DataFrame:
        """构建完整的情感因子"""
        # 1. 计算基础情感得分
        sentiment_df = self.calculate_news_sentiment()
        
        # 2. 多时间维度聚合
        factors_1m = self.aggregate_sentiment_by_timeframe(sentiment_df, '1T')
        factors_5m = self.aggregate_sentiment_by_timeframe(sentiment_df, '5T')
        factors_15m = self.aggregate_sentiment_by_timeframe(sentiment_df, '15T')
        factors_1h = self.aggregate_sentiment_by_timeframe(sentiment_df, '1H')
        
        # 3. 重命名列名以区分时间框架
        factors_1m = factors_1m.add_suffix('_1m')
        factors_5m = factors_5m.add_suffix('_5m')
        factors_15m = factors_15m.add_suffix('_15m')
        factors_1h = factors_1h.add_suffix('_1h')
        
        # 4. 合并所有因子
        all_factors = pd.concat([factors_1m, factors_5m, factors_15m, factors_1h], 
                               axis=1, join='outer')
        
        # 5. 与价格数据对齐
        aligned_factors = all_factors.reindex(self.price_data.index, method='ffill')
        
        return aligned_factors

# ================================
# 4. 自定义数据处理器
# ================================

class CryptoNewsSentimentHandler(Alpha158):
    """加密货币新闻情感数据处理器"""
    
    def __init__(self, instruments="btcusdt", start_time=None, end_time=None,
                 news_factor_path: str = None, **kwargs):
        self.news_factor_path = news_factor_path
        super().__init__(instruments=instruments, start_time=start_time, 
                        end_time=end_time, **kwargs)
    
    def get_feature_config(self):
        """获取特征配置，包含新闻情感因子"""
        # 获取基础技术因子配置
        base_config = super().get_feature_config()
        
        # 添加新闻情感因子
        if self.news_factor_path and Path(self.news_factor_path).exists():
            sentiment_factors = self._load_sentiment_factors()
            base_config.extend(sentiment_factors)
        
        return base_config
    
    def _load_sentiment_factors(self) -> List[str]:
        """加载新闻情感因子定义"""
        # 这里可以定义情感因子的表达式
        sentiment_factors = [
            "$sentiment_score_1m",
            "$sentiment_volatility_1m", 
            "$news_volume_1m",
            "$sentiment_momentum_1m",
            "$sentiment_score_5m",
            "$sentiment_score_15m",
            "$sentiment_score_1h",
            # 可以添加更多复合因子
            "($sentiment_score_1m * $volume) / ($volume + 1e-12)",  # 成交量加权情感
            "($sentiment_score_1m - Ref($sentiment_score_1m, 1))",   # 情感变化
        ]
        return sentiment_factors

# ================================
# 5. 因子评估模块
# ================================

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, factor_data: pd.DataFrame, return_data: pd.DataFrame):
        self.factor_data = factor_data
        self.return_data = return_data
    
    def calculate_ic(self, factor_name: str, period: int = 1) -> pd.Series:
        """计算信息系数(IC)"""
        factor_values = self.factor_data[factor_name]
        future_returns = self.return_data.shift(-period)
        
        # 计算滚动IC
        ic_series = pd.Series(index=factor_values.index, dtype=float)
        
        for i in range(period, len(factor_values)):
            end_idx = i
            start_idx = max(0, i - 30)  # 30期滚动窗口
            
            factor_window = factor_values.iloc[start_idx:end_idx]
            return_window = future_returns.iloc[start_idx:end_idx]
            
            # 计算相关系数
            ic_series.iloc[i] = factor_window.corr(return_window)
        
        return ic_series
    
    def calculate_rank_ic(self, factor_name: str, period: int = 1) -> pd.Series:
        """计算Rank IC"""
        factor_values = self.factor_data[factor_name]
        future_returns = self.return_data.shift(-period)
        
        # 转换为排名
        factor_ranks = factor_values.rank()
        return_ranks = future_returns.rank()
        
        return self.calculate_ic_from_series(factor_ranks, return_ranks)
    
    def calculate_ic_ir(self, ic_series: pd.Series) -> float:
        """计算IC信息比率"""
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        return ic_mean / ic_std if ic_std != 0 else 0
    
    def factor_performance_summary(self, factor_name: str) -> Dict:
        """因子表现总结"""
        ic_series = self.calculate_ic(factor_name)
        rank_ic_series = self.calculate_rank_ic(factor_name)
        
        return {
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ic_ir': self.calculate_ic_ir(ic_series),
            'rank_ic_mean': rank_ic_series.mean(),
            'rank_ic_std': rank_ic_series.std(),
            'rank_ic_ir': self.calculate_ic_ir(rank_ic_series),
            'positive_ic_ratio': (ic_series > 0).mean(),
            'factor_name': factor_name
        }

# ================================
# 6. 主流程示例
# ================================

def main():
    """主要执行流程"""
    
    # 1. 初始化qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/crypto_data")
    
    # 2. 数据采集
    print("开始数据采集...")
    
    # 价格数据采集
    price_collector = BTCUSDTCollector()
    price_data = price_collector.fetch_ohlcv(timeframe='1m', limit=10000)
    
    # 新闻数据采集
    news_collector = NewsCollector()
    news_data = news_collector.collect_all_news()
    
    print(f"获取到 {len(price_data)} 条价格数据")
    print(f"获取到 {len(news_data)} 条新闻数据")
    
    # 3. 情感因子构建
    print("开始构建情感因子...")
    factor_builder = NewsSentimentFactorBuilder(price_data, news_data)
    sentiment_factors = factor_builder.build_factors()
    
    print(f"构建了 {len(sentiment_factors.columns)} 个情感因子")
    
    # 4. 因子评估
    print("开始因子评估...")
    returns = price_data['close'].pct_change()
    evaluator = FactorEvaluator(sentiment_factors, returns)
    
    # 评估主要因子
    main_factors = ['ensemble_score_mean_1m', 'sentiment_volatility_1m', 'news_volume_1m']
    
    for factor_name in main_factors:
        if factor_name in sentiment_factors.columns:
            performance = evaluator.factor_performance_summary(factor_name)
            print(f"\n因子 {factor_name} 的表现:")
            print(f"  IC均值: {performance['ic_mean']:.4f}")
            print(f"  IC标准差: {performance['ic_std']:.4f}")
            print(f"  IC信息比率: {performance['ic_ir']:.4f}")
            print(f"  正IC比例: {performance['positive_ic_ratio']:.4f}")
    
    # 5. 保存结果
    print("\n保存分析结果...")
    sentiment_factors.to_csv('sentiment_factors.csv')
    price_data.to_csv('btc_price_data.csv')
    news_data.to_csv('news_data.csv')
    
    print("分析完成！")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 运行主流程
    main()

# ================================
# 7. 配置文件示例
# ================================

# config.yaml 示例配置
CONFIG_EXAMPLE = """
data:
  symbol: "BTC/USDT"
  timeframe: "1m"
  lookback_days: 30
  
news_sources:
  - name: "coindesk"
    url: "https://www.coindesk.com/arc/outboundfeeds/rss/"
    weight: 0.4
  - name: "cointelegraph" 
    url: "https://cointelegraph.com/rss"
    weight: 0.3
  - name: "bitcoinmagazine"
    url: "https://bitcoinmagazine.com/.rss/full/"
    weight: 0.3

sentiment_models:
  vader:
    enabled: true
    weight: 0.3
  finbert:
    enabled: true
    weight: 0.5
    model_name: "ProsusAI/finbert"
  textblob:
    enabled: true
    weight: 0.2

factor_config:
  timeframes: ["1m", "5m", "15m", "1h"]
  aggregation_methods: ["mean", "std", "count"]
  rolling_windows: [5, 10, 20, 60]

evaluation:
  ic_window: 30
  test_periods: [1, 5, 15, 60]  # 分钟
  
backtest:
  start_date: "2023-01-01"
  end_date: "2024-01-01"
  initial_capital: 100000
  transaction_cost: 0.001
""" 