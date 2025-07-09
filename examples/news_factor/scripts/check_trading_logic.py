#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# 加载因子数据
df = pd.read_parquet('data/factors/factors_20250708_231857.parquet')
print('🔍 交易逻辑分析')
print('='*60)

# 主要情感因子
sentiment_col = 'sentiment_1h_sentiment_score_mean'
sentiment_data = df[sentiment_col].dropna()

buy_threshold = 0.10
sell_threshold = -0.05

# 模拟交易逻辑
positions = []
current_position = None
trades = []

for i, (timestamp, sentiment) in enumerate(sentiment_data.items()):
    
    # 买入信号
    if sentiment >= buy_threshold and current_position is None:
        current_position = {
            'entry_time': timestamp,
            'entry_price': 1.0,  # 假设价格
            'entry_sentiment': sentiment,
            'holding_minutes': 0
        }
        positions.append(current_position)
        print(f"🟢 买入信号 {len(positions)}: {timestamp} (情感: {sentiment:.4f})")
        
    # 检查持仓
    elif current_position is not None:
        current_position['holding_minutes'] = (timestamp - current_position['entry_time']).total_seconds() / 60
        
        # 卖出条件
        sell_reason = None
        if sentiment <= sell_threshold:
            sell_reason = "卖出信号"
        elif current_position['holding_minutes'] >= 1440:  # 24小时超时
            sell_reason = "超时"
            
        if sell_reason:
            trade = {
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'entry_sentiment': current_position['entry_sentiment'],
                'exit_sentiment': sentiment,
                'holding_minutes': current_position['holding_minutes'],
                'exit_reason': sell_reason
            }
            trades.append(trade)
            print(f"🔴 卖出 {len(trades)}: {timestamp} ({sell_reason}, 持仓{current_position['holding_minutes']:.0f}分钟)")
            current_position = None
    
    # 只显示前20个买入信号
    if len(positions) >= 20:
        break

print(f"\n📊 交易统计:")
print(f"  买入信号总数: {len(positions)}")
print(f"  完成交易数: {len(trades)}")
print(f"  当前持仓: {'是' if current_position else '否'}")

if len(trades) > 0:
    print(f"\n📈 交易详情:")
    for i, trade in enumerate(trades):
        print(f"  交易{i+1}: {trade['entry_time']} → {trade['exit_time']}")
        print(f"    持仓时间: {trade['holding_minutes']:.0f}分钟")
        print(f"    退出原因: {trade['exit_reason']}")
        print(f"    情感变化: {trade['entry_sentiment']:.4f} → {trade['exit_sentiment']:.4f}") 