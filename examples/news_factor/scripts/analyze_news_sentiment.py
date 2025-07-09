#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻情感分析脚本
使用DeepSeek API对新闻数据进行情感分析
"""

import pandas as pd
import sys
import time
import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 抑制HTTP请求日志，只显示进度条
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# DeepSeek API配置 - 从.env文件读取
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', "https://api.deepseek.com/v1")
MODEL = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")

class NewsAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
    def analyze_sentiment(self, text: str) -> dict:
        """分析单条新闻的情感"""
        prompt = f"""请对以下加密货币新闻进行情感分析：

新闻内容: {text}

请返回JSON格式的分析结果，包含：
- sentiment_score: -1到1的数值（-1极度负面，0中性，1极度正面）  
- confidence: 0到1的置信度
- sentiment_label: "负面"/"中性"/"正面"
- market_impact: "利空"/"中性"/"利好"

示例格式：
{{"sentiment_score": 0.5, "confidence": 0.8, "sentiment_label": "正面", "market_impact": "利好"}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # 解析JSON结果
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return {
                    'deepseek_sentiment_score': result.get('sentiment_score', 0.0),
                    'deepseek_confidence': result.get('confidence', 0.5),
                    'deepseek_sentiment_label': result.get('sentiment_label', '中性'),
                    'deepseek_market_impact': result.get('market_impact', '中性')
                }
            else:
                raise ValueError("无法解析JSON响应")
                
        except Exception as e:
            print(f"分析失败: {e}")
            return {
                'deepseek_sentiment_score': 0.0,
                'deepseek_confidence': 0.5,
                'deepseek_sentiment_label': '中性',
                'deepseek_market_impact': '中性'
            }
    
    def analyze_news_file(self, input_file: str, output_file: str = None, sample_size: int = None):
        """分析新闻文件"""
        print(f"📰 加载新闻文件: {input_file}")
        
        # 读取新闻数据
        df = pd.read_csv(input_file)
        print(f"📊 总共 {len(df)} 条新闻")
        
        # 如果指定了样本大小，随机采样
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"🎯 随机采样 {len(df)} 条新闻进行分析")
        
        # 为每条新闻进行情感分析
        print("🤖 开始DeepSeek情感分析...")
        
        results = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="分析进度"):
            # 构建分析文本
            text = row['title']
            if 'description' in row and pd.notna(row['description']):
                text += " " + str(row['description'])[:500]  # 限制长度
            
            # 分析情感
            sentiment_result = self.analyze_sentiment(text)
            
            # 合并结果
            result_row = dict(row)
            result_row.update(sentiment_result)
            results.append(result_row)
            
            # 添加延迟避免API限制
            time.sleep(0.1)
        
        # 保存结果
        results_df = pd.DataFrame(results)
        
        if output_file is None:
            # 生成输出文件名
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_with_sentiment{input_path.suffix}"
        
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 情感分析完成！结果保存至: {output_file}")
        
        # 统计信息
        sentiment_stats = results_df['deepseek_sentiment_score'].describe()
        print(f"\n📊 情感分析统计:")
        print(f"  平均情感得分: {sentiment_stats['mean']:.3f}")
        print(f"  情感得分范围: {sentiment_stats['min']:.3f} - {sentiment_stats['max']:.3f}")
        print(f"  标准差: {sentiment_stats['std']:.3f}")
        
        return str(output_file)

def main():
    parser = argparse.ArgumentParser(description="新闻情感分析")
    parser.add_argument("input_file", help="输入新闻CSV文件")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--sample", "-s", type=int, help="随机采样数量（用于测试）")
    
    args = parser.parse_args()
    
    # 检查API Key
    if not DEEPSEEK_API_KEY:
        print("❌ 请设置DEEPSEEK_API_KEY环境变量")
        print("   方法1: 在项目根目录创建.env文件，添加: DEEPSEEK_API_KEY=your-api-key")
        print("   方法2: 直接设置环境变量: export DEEPSEEK_API_KEY='your-api-key'")
        return 1
    
    try:
        analyzer = NewsAnalyzer()
        analyzer.analyze_news_file(args.input_file, args.output, args.sample)
        return 0
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 