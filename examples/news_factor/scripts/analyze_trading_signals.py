#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易信号分析脚本
分析为什么交易次数这么少
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def analyze_trading_signals():
    """分析交易信号稀少的原因"""
    
    print("🔍 交易信号分析")
    print("="*60)
    
    # 1. 加载因子数据
    factor_file = project_root / "data" / "factors" / "factors_20250708_231857.parquet"
    if not factor_file.exists():
        print(f"❌ 因子文件不存在: {factor_file}")
        return
    
    factor_data = pd.read_parquet(factor_file)
    print(f"✅ 因子数据: {len(factor_data)} 条记录，{len(factor_data.columns)} 个因子")
    
    # 2. 检查情感相关因子
    sentiment_cols = [col for col in factor_data.columns if 'sentiment' in col.lower()]
    print(f"📊 情感相关因子: {len(sentiment_cols)} 个")
    for col in sentiment_cols[:10]:  # 显示前10个
        print(f"  - {col}")
    
    # 3. 分析主要情感因子
    if 'sentiment_1h_sentiment_score_mean' in factor_data.columns:
        sentiment_col = 'sentiment_1h_sentiment_score_mean'
    elif 'sentiment_mean' in factor_data.columns:
        sentiment_col = 'sentiment_mean'
    elif 'sentiment_score' in factor_data.columns:
        sentiment_col = 'sentiment_score'
    else:
        print("❌ 未找到主要情感因子")
        return
        
    sentiment_data = factor_data[sentiment_col].dropna()
    print(f"\n📈 {sentiment_col} 统计信息:")
    print(f"  数据点数: {len(sentiment_data)}")
    print(f"  有效数据: {sentiment_data.count()}")
    print(f"  最小值: {sentiment_data.min():.4f}")
    print(f"  最大值: {sentiment_data.max():.4f}")
    print(f"  平均值: {sentiment_data.mean():.4f}")
    print(f"  标准差: {sentiment_data.std():.4f}")
    
    # 4. 检查买入卖出阈值覆盖情况
    buy_threshold = 0.10
    sell_threshold = -0.05
    
    buy_signals = sentiment_data >= buy_threshold
    sell_signals = sentiment_data <= sell_threshold
    
    print(f"\n🎯 信号分析 (阈值: 买入>={buy_threshold}, 卖出<={sell_threshold}):")
    print(f"  达到买入阈值的次数: {buy_signals.sum()}")
    print(f"  达到卖出阈值的次数: {sell_signals.sum()}")
    print(f"  买入信号比例: {buy_signals.mean():.2%}")
    print(f"  卖出信号比例: {sell_signals.mean():.2%}")
    
    # 5. 分析不同阈值下的信号频率
    print(f"\n📊 不同买入阈值下的信号频率:")
    print(f"{'阈值':<8} {'信号次数':<10} {'信号比例':<10}")
    print("-" * 30)
    
    for threshold in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        signals = sentiment_data >= threshold
        print(f"{threshold:<8} {signals.sum():<10} {signals.mean():<10.2%}")
    
    # 6. 检查新闻数量因子
    if 'news_count' in factor_data.columns:
        news_count = factor_data['news_count']
        print(f"\n📰 新闻数量统计:")
        print(f"  有新闻的时间点: {(news_count > 0).sum()}")
        print(f"  无新闻的时间点: {(news_count == 0).sum()}")
        print(f"  平均新闻数量: {news_count.mean():.2f}")
        print(f"  最大新闻数量: {news_count.max()}")
        
        # 同时有新闻和买入信号的情况
        has_news = news_count > 0
        buy_with_news = (sentiment_data >= buy_threshold) & has_news
        print(f"  有新闻且达到买入阈值: {buy_with_news.sum()}")
    
    # 7. 时间分布分析
    print(f"\n📅 信号时间分布:")
    buy_signal_times = sentiment_data[sentiment_data >= buy_threshold].index
    if len(buy_signal_times) > 0:
        print(f"  首次买入信号: {buy_signal_times[0]}")
        print(f"  最后买入信号: {buy_signal_times[-1]}")
        
        # 按日期统计
        daily_signals = pd.Series(buy_signal_times).dt.date.value_counts().sort_index()
        print(f"  买入信号天数: {len(daily_signals)}")
        print(f"  平均每天信号: {len(buy_signal_times) / len(daily_signals):.1f}")
    
    # 8. 建议的阈值
    print(f"\n💡 阈值建议:")
    for target_signals in [10, 20, 50, 100]:
        # 找到能产生目标信号数的阈值
        sorted_sentiment = sentiment_data.sort_values(ascending=False)
        if len(sorted_sentiment) >= target_signals:
            threshold = sorted_sentiment.iloc[target_signals-1]
            print(f"  {target_signals}次信号 → 阈值: {threshold:.4f}")
    
    return sentiment_data

def analyze_sentiment_distribution(sentiment_data):
    """分析情感分布"""
    print(f"\n📊 情感分布分析:")
    
    # 分位数分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"{'分位数':<8} {'值':<10}")
    print("-" * 20)
    for p in percentiles:
        value = np.percentile(sentiment_data, p)
        print(f"{p}%{'':<5} {value:<10.4f}")

if __name__ == "__main__":
    sentiment_data = analyze_trading_signals()
    if sentiment_data is not None:
        analyze_sentiment_distribution(sentiment_data) 