#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 修复real_backtest.py中的列名错误
import re

def fix_backtest_file():
    """修复回测文件中的列名错误"""
    
    file_path = "scripts/real_backtest.py"
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换错误的列名
    fixes = [
        ("self.news_data['sentiment_score']", "self.news_data['deepseek_sentiment_score']"),
        ("'sentiment_score'", "'deepseek_sentiment_score'"),
        ("'sentiment_confidence'", "'deepseek_confidence'"),
        ("sentiment_score", "deepseek_sentiment_score"),
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 回测文件修复完成！")

if __name__ == "__main__":
    fix_backtest_file() 