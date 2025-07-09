#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# 加载因子数据
df = pd.read_parquet('data/factors/factors_20250708_231857.parquet')
print('🔍 交易信号分析')
print('='*60)

# 主要情感因子
sentiment_col = 'sentiment_1h_sentiment_score_mean'
sentiment_data = df[sentiment_col].dropna()

print(f'📈 {sentiment_col} 统计信息:')
print(f'  数据点数: {len(sentiment_data)}')
print(f'  有效数据: {sentiment_data.count()}')
print(f'  最小值: {sentiment_data.min():.4f}')
print(f'  最大值: {sentiment_data.max():.4f}')
print(f'  平均值: {sentiment_data.mean():.4f}')
print(f'  标准差: {sentiment_data.std():.4f}')

# 检查买入卖出阈值
buy_threshold = 0.10
sell_threshold = -0.05

buy_signals = sentiment_data >= buy_threshold
sell_signals = sentiment_data <= sell_threshold

print(f'\n🎯 信号分析 (阈值: 买入>={buy_threshold}, 卖出<={sell_threshold}):')
print(f'  达到买入阈值的次数: {buy_signals.sum()}')
print(f'  达到卖出阈值的次数: {sell_signals.sum()}')
print(f'  买入信号比例: {buy_signals.mean():.2%}')
print(f'  卖出信号比例: {sell_signals.mean():.2%}')

# 分析不同阈值下的信号频率
print(f'\n📊 不同买入阈值下的信号频率:')
print(f"{'阈值':<8} {'信号次数':<10} {'信号比例':<10}")
print('-' * 30)

for threshold in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
    signals = sentiment_data >= threshold
    print(f"{threshold:<8} {signals.sum():<10} {signals.mean():<10.2%}")

# 检查新闻数量
if 'sentiment_1h_sentiment_score_count' in df.columns:
    news_count = df['sentiment_1h_sentiment_score_count']
    print(f'\n📰 新闻数量统计:')
    print(f'  有新闻的时间点: {(news_count > 0).sum()}')
    print(f'  无新闻的时间点: {(news_count == 0).sum()}')
    print(f'  平均新闻数量: {news_count.mean():.2f}')
    print(f'  最大新闻数量: {news_count.max()}')

# 分位数分析
print(f'\n📊 情感分布分析:')
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"{'分位数':<8} {'值':<10}")
print('-' * 20)
for p in percentiles:
    value = np.percentile(sentiment_data, p)
    print(f"{p}%{'':<5} {value:<10.4f}")

# 建议的阈值
print(f'\n💡 阈值建议:')
for target_signals in [10, 20, 50, 100]:
    sorted_sentiment = sentiment_data.sort_values(ascending=False)
    if len(sorted_sentiment) >= target_signals:
        threshold = sorted_sentiment.iloc[target_signals-1]
        print(f'  {target_signals}次信号 → 阈值: {threshold:.4f}') 