#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def debug_backtest():
    """调试回测逻辑"""
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    df = pd.read_parquet('data/factors/factors_20250708_231857.parquet')
    price_data = pd.read_csv('data/raw/price/btc_usdt_1m_real_20250708_231836.csv')
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    price_data.set_index('timestamp', inplace=True)
    
    # 参数
    buy_threshold = config['strategy']['signal_generation']['sentiment_threshold_buy']
    sell_threshold = config['strategy']['signal_generation']['sentiment_threshold_sell']
    max_holding_minutes = config['strategy']['risk_management']['max_holding_minutes']
    
    print(f"📊 参数:")
    print(f"  买入阈值: {buy_threshold}")
    print(f"  卖出阈值: {sell_threshold}")
    print(f"  最大持仓时间: {max_holding_minutes}分钟")
    
    # 情感因子
    sentiment_col = 'sentiment_1h_sentiment_score_mean'
    
    # 确保数据对齐
    common_index = df.index.intersection(price_data.index)
    
    print(f"\n📈 数据对齐:")
    print(f"  因子数据点: {len(df)}")
    print(f"  价格数据点: {len(price_data)}")
    print(f"  共同数据点: {len(common_index)}")
    
    # 模拟交易
    trades = []
    signals = []
    position = 0
    entry_time = None
    entry_price = 0
    
    # 调试计数器
    buy_signals = 0
    sell_signals = 0
    
    for i, timestamp in enumerate(common_index[:1000]):  # 只处理前1000个数据点进行调试
        
        # 获取数据
        current_price = price_data.loc[timestamp, 'close']
        sentiment_signal = df.loc[timestamp, sentiment_col]
        
        # 检查持仓
        if position != 0:
            # 计算持仓时间
            holding_minutes = (timestamp - entry_time).total_seconds() / 60
            
            # 检查退出条件
            should_exit = False
            exit_reason = ""
            
            # 超时退出
            if holding_minutes >= max_holding_minutes:
                should_exit = True
                exit_reason = "超时"
            
            # 反向信号退出
            if position == 1 and sentiment_signal <= sell_threshold:
                should_exit = True
                exit_reason = "反向信号"
                sell_signals += 1
            
            if should_exit:
                pnl_pct = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
                trade = {
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'holding_minutes': holding_minutes,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct
                }
                trades.append(trade)
                
                # 输出交易信息
                print(f"🔴 交易{len(trades)}: {entry_time} -> {timestamp} ({holding_minutes:.0f}分钟, {exit_reason}, PnL: {pnl_pct:.2%})")
                
                # 重置持仓
                position = 0
                entry_time = None
                entry_price = 0
        
        # 开仓检查
        if position == 0:
            if sentiment_signal >= buy_threshold:
                position = 1
                entry_time = timestamp
                entry_price = current_price
                buy_signals += 1
                print(f"🟢 买入信号{buy_signals}: {timestamp} (情感: {sentiment_signal:.4f}, 价格: {current_price:.2f})")
        
        # 记录信号
        if position != 0:
            signals.append(position)
        else:
            signals.append(0)
    
    print(f"\n📊 调试结果:")
    print(f"  买入信号数: {buy_signals}")
    print(f"  卖出信号数: {sell_signals}")
    print(f"  完成交易数: {len(trades)}")
    print(f"  当前持仓: {position}")
    
    return trades

if __name__ == "__main__":
    debug_backtest() 