#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek大模型情感分析器
基于DeepSeek API进行加密货币新闻的专业情感分析
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 抑制OpenAI客户端的HTTP请求日志
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# DeepSeek API 配置
DEEPSEEK_LIMITS = {
    'max_context_length': 32000,
    'safe_input_tokens': 20000,
    'max_output_tokens': 4000,
    'rate_limit_per_minute': 60,
    'concurrent_requests': 5,
    'timeout_seconds': 60,
    'retry_delay': 1.0,
    'avg_tokens_per_news': 400,
}

class APICallStats:
    """简化的API调用统计类"""
    
    def __init__(self):
        self.reset_stats()
        self.stats_lock = threading.Lock()
    
    def reset_stats(self):
        """重置统计数据"""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_response_time = 0.0
    
    def record_call(self, response_time: float, input_tokens: int, output_tokens: int, success: bool):
        """记录一次API调用"""
        with self.stats_lock:
            self.total_calls += 1
            self.total_response_time += response_time
            
            if success:
                self.successful_calls += 1
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            else:
                self.failed_calls += 1
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        avg_response_time = self.total_response_time / max(1, self.total_calls)
        success_rate = self.successful_calls / max(1, self.total_calls) * 100
        
        # 估算成本 (DeepSeek定价：输入 $0.14/1M tokens, 输出 $0.28/1M tokens)
        cost = (self.total_input_tokens * 0.14 + self.total_output_tokens * 0.28) / 1000000
        
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': f"{success_rate:.1f}%",
            'total_tokens': total_tokens,
            'avg_response_time': f"{avg_response_time:.2f}s",
            'estimated_cost_usd': f"${cost:.4f}",
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n📊 API调用统计: {stats['successful_calls']}/{stats['total_calls']} 成功率{stats['success_rate']}")
        print(f"💰 总计成本: {stats['estimated_cost_usd']} | ⏱️ 平均响应时间: {stats['avg_response_time']}")

class DeepSeekSentimentAnalyzer:
    """DeepSeek情感分析器"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat",
                 timeout: int = 30,
                 max_retries: int = 3,
                 rate_limit: float = 0.1,
                 fast_mode: bool = False):
        
        # API配置
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置或未提供API密钥")
        
        self.base_url = base_url
        self.model = model
        
        # 高并发模式配置（100线程优化）
        if fast_mode:
            self.timeout = min(timeout, 15)
            self.max_retries = min(max_retries, 2)
            self.rate_limit = max(rate_limit, 0.05)
            self.max_batch_size = 50
            self.max_tokens_per_batch = 15000
        else:
            self.timeout = min(timeout, 30)
            self.max_retries = max_retries
            self.rate_limit = max(rate_limit, 0.1)
            self.max_batch_size = 30
            self.max_tokens_per_batch = DEEPSEEK_LIMITS['safe_input_tokens']
        
        # 多线程配置（支持大规模并发）
        self.max_workers = 100  # 100个线程并发
        
        # 配置OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # 初始化缓存和统计
        self.cache = {}
        self.stats = APICallStats()
        
        logger.info(f"✅ DeepSeek分析器初始化完成 (模型: {self.model}, 并发: {self.max_workers}线程)")

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 简化的token估算：中文1字符≈1token，英文1字符≈0.25token
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars + int(other_chars * 0.25)
    
    def _build_batch_prompt(self, news_texts: List[str]) -> str:
        """构建批量分析提示词"""
        numbered_news = []
        for i, text in enumerate(news_texts, 1):
            numbered_news.append(f"{i}. {text}")
        
        news_list = "\n".join(numbered_news)
        
        prompt = f"""请对以下{len(news_texts)}条加密货币新闻进行情感分析，严格按照JSON格式输出结果。

分析标准：
- sentiment_score: -1到1的数值（-1极度负面，0中性，1极度正面）
- confidence: 0到1的置信度
- sentiment_label: "负面"/"中性"/"正面"
- market_impact: "利空"/"中性"/"利好"

新闻内容：
{news_list}

请严格按照以下JSON格式输出，每条新闻一个对象：
```json
[
  {{"id": 1, "sentiment_score": 0.5, "confidence": 0.8, "sentiment_label": "正面", "market_impact": "利好"}},
  {{"id": 2, "sentiment_score": -0.3, "confidence": 0.7, "sentiment_label": "负面", "market_impact": "利空"}}
]
```"""
        return prompt
    
    def _make_batch_request(self, news_texts: List[str]) -> Optional[Dict]:
        """发送批量请求到DeepSeek API"""
        prompt = self._build_batch_prompt(news_texts)
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=min(4000, len(news_texts) * 200)
                )
                
                end_time = time.time()
                
                # 记录统计
                self.stats.record_call(
                    response_time=end_time - start_time,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    success=True
                )
                
                return {
                    'content': response.choices[0].message.content,
                    'usage': response.usage
                }
                
            except Exception as e:
                end_time = time.time()
                self.stats.record_call(
                    response_time=end_time - start_time,
                    input_tokens=0,
                    output_tokens=0,
                    success=False
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit * (attempt + 1))
                    continue
                else:
                    logger.error(f"批量请求失败: {e}")
                    return None
        
        return None
    
    def _parse_batch_result(self, response, news_count: int) -> List[Dict]:
        """解析批量分析结果"""
        try:
            content = response['content']
            
            # 提取JSON部分
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("未找到有效的JSON格式")
            
            json_str = content[start_idx:end_idx]
            results = json.loads(json_str)
            
            # 验证和处理结果
            processed_results = []
            for i in range(news_count):
                if i < len(results):
                    result = results[i]
                    processed_results.append({
                        'deepseek_sentiment_score': result.get('sentiment_score', 0.0),
                        'deepseek_confidence': result.get('confidence', 0.5),
                        'deepseek_sentiment_label': result.get('sentiment_label', '中性'),
                        'deepseek_market_impact': result.get('market_impact', '中性'),
                        'deepseek_key_factors': '',
                        'deepseek_reasoning': ''
                    })
                else:
                    # 缺失结果的默认值
                    processed_results.append({
                        'deepseek_sentiment_score': 0.0,
                        'deepseek_confidence': 0.5,
                        'deepseek_sentiment_label': '中性',
                        'deepseek_market_impact': '中性',
                        'deepseek_key_factors': '',
                        'deepseek_reasoning': ''
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"解析批量结果失败: {e}")
            # 返回默认结果
            return [{
                'deepseek_sentiment_score': 0.0,
                'deepseek_confidence': 0.5,
                'deepseek_sentiment_label': '中性',
                'deepseek_market_impact': '中性',
                'deepseek_key_factors': '',
                'deepseek_reasoning': ''
            } for _ in range(news_count)]
    
    def analyze_batch_single_request(self, news_texts: List[str]) -> List[Dict]:
        """使用单个API请求分析多条新闻"""
        if len(news_texts) > self.max_batch_size:
            logger.warning(f"批量大小({len(news_texts)})超过限制({self.max_batch_size})，将分批处理")
            
            results = []
            for i in range(0, len(news_texts), self.max_batch_size):
                batch = news_texts[i:i + self.max_batch_size]
                batch_results = self.analyze_batch_single_request(batch)
                results.extend(batch_results)
            return results
        
        response = self._make_batch_request(news_texts)
        if response:
            return self._parse_batch_result(response, len(news_texts))
        else:
            logger.error("批量请求失败，返回默认结果")
            return [{
                'deepseek_sentiment_score': 0.0,
                'deepseek_confidence': 0.5,
                'deepseek_sentiment_label': '中性',
                'deepseek_market_impact': '中性',
                'deepseek_key_factors': '',
                'deepseek_reasoning': ''
            } for _ in news_texts]
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def analyze_single_threaded(self, text: str, thread_id: int = 0) -> Dict:
        """单条新闻分析（线程安全）"""
        cache_key = self._get_cache_key(text)
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        prompt = f"""请对以下加密货币新闻进行情感分析：

新闻内容: {text}

请返回JSON格式的分析结果：
{{"sentiment_score": 0.0, "confidence": 0.8, "sentiment_label": "中性", "market_impact": "中性"}}"""
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                
                end_time = time.time()
                
                # 记录统计
                self.stats.record_call(
                    response_time=end_time - start_time,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    success=True
                )
                
                result = self._parse_single_result(response.choices[0].message.content)
                self.cache[cache_key] = result
                return result
        
            except Exception as e:
                end_time = time.time()
                self.stats.record_call(
                    response_time=end_time - start_time,
                    input_tokens=0,
                    output_tokens=0,
                    success=False
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit)
                    continue
                else:
                    logger.error(f"线程{thread_id}分析失败: {e}")
                    break
            
            # 高并发模式下的速率限制（避免阻塞）
            if self.rate_limit > 0:
                time.sleep(min(self.rate_limit, 0.1))  # 最大只sleep 0.1秒
        
        # 失败时返回默认结果
        return {
            'deepseek_sentiment_score': 0.0,
            'deepseek_confidence': 0.5,
            'deepseek_sentiment_label': '中性',
            'deepseek_market_impact': '中性',
            'deepseek_key_factors': '',
            'deepseek_reasoning': ''
        }
    
    def _parse_single_result(self, content: str) -> Dict:
        """解析单个分析结果"""
        try:
            # 尝试解析JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                
                return {
                    'deepseek_sentiment_score': result.get('sentiment_score', 0.0),
                    'deepseek_confidence': result.get('confidence', 0.5),
                    'deepseek_sentiment_label': result.get('sentiment_label', '中性'),
                    'deepseek_market_impact': result.get('market_impact', '中性'),
                    'deepseek_key_factors': '',
                    'deepseek_reasoning': ''
                }
        except:
            pass
        
        # 解析失败时返回默认值
        return {
            'deepseek_sentiment_score': 0.0,
            'deepseek_confidence': 0.5,
            'deepseek_sentiment_label': '中性',
            'deepseek_market_impact': '中性',
            'deepseek_key_factors': '',
            'deepseek_reasoning': ''
        }
    
    def analyze_batch_multithreaded(self, texts: List[str], max_workers: Optional[int] = None) -> List[Dict]:
        """多线程批量分析"""
        if not texts:
            return []
        
        max_workers = max_workers or self.max_workers
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.analyze_single_threaded, text, i): i 
                for i, text in enumerate(texts)
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_index), total=len(texts), desc="🤖 DeepSeek分析"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"线程{index}执行失败: {e}")
                    results[index] = {
                        'deepseek_sentiment_score': 0.0,
                        'deepseek_confidence': 0.5,
                        'deepseek_sentiment_label': '中性',
                        'deepseek_market_impact': '中性',
                        'deepseek_key_factors': '',
                        'deepseek_reasoning': ''
                    }
        
        return results
    
    def analyze_batch(self, news_data: Union[List[str], List[Dict], pd.DataFrame], 
                     title_col: str = 'title', content_col: str = 'description',
                     use_batch_request: bool = False, use_multithreading: bool = True,
                     max_workers: Optional[int] = None) -> pd.DataFrame:
        """批量分析新闻情感"""
        
        # 标准化输入数据
        if isinstance(news_data, pd.DataFrame):
            df = news_data.copy()
        elif isinstance(news_data, list):
            if news_data and isinstance(news_data[0], dict):
                df = pd.DataFrame(news_data)
            else:
                df = pd.DataFrame({title_col: news_data})
        else:
            raise ValueError("不支持的数据类型")
        
        # 构建分析文本
        if content_col in df.columns and df[content_col].notna().any():
            texts = (df[title_col].fillna('') + ' ' + df[content_col].fillna('')).tolist()
        else:
            texts = df[title_col].fillna('').tolist()
        
        logger.info(f"开始分析 {len(texts)} 条新闻...")
        
        # 执行分析
        if use_batch_request and len(texts) <= self.max_batch_size:
            results = self.analyze_batch_single_request(texts)
        else:
            if use_multithreading:
                results = self.analyze_batch_multithreaded(texts, max_workers)
            else:
                results = [self.analyze_single_threaded(text, i) for i, text in enumerate(tqdm(texts, desc="🤖 DeepSeek分析"))]
        
        # 将结果添加到DataFrame
        results_df = pd.DataFrame(results)
        for col in results_df.columns:
            df[col] = results_df[col]
        
        logger.info(f"✅ 分析完成！成功率: {self.stats.get_stats()['success_rate']}")
        return df
    
    def load_cache(self, cache_file: Optional[str] = None):
        """加载缓存"""
        cache_file = cache_file or "deepseek_sentiment_cache.json"
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"✅ 缓存加载成功，共 {len(self.cache)} 条记录")
            else:
                logger.warning(f"缓存文件不存在: {cache_file}")
                self.cache = {}
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
            self.cache = {}
    
    def save_cache(self, cache_file: Optional[str] = None):
        """保存缓存"""
        cache_file = cache_file or "deepseek_sentiment_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 缓存保存成功，共 {len(self.cache)} 条记录")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.get_stats()
    
    def print_stats(self):
        """打印统计信息"""
        self.stats.print_stats()


def main():
    """测试DeepSeek情感分析器"""
    import sys
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 检查API密钥
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("⚠️  请设置环境变量 DEEPSEEK_API_KEY")
        print("    export DEEPSEEK_API_KEY='your-api-key-here'")
        return
    
    # 创建分析器
    print("🚀 创建DeepSeek情感分析器...")
    analyzer = DeepSeekSentimentAnalyzer(fast_mode=len(sys.argv) > 1 and sys.argv[1] == '--fast')
    
    # 测试数据
    test_news = [
        "Bitcoin ETF获得SEC批准，机构资金大举涌入。美国证券交易委员会(SEC)正式批准了首批比特币现货ETF，这一里程碑事件预计将为加密货币市场带来大量机构资金。",
        "加密货币交易所遭黑客攻击，损失超2亿美元。全球知名加密货币交易所昨日遭受大规模黑客攻击，损失金额超过2亿美元，引发市场恐慌情绪。",
        "央行数字货币研发取得重大进展。中国人民银行宣布数字人民币技术研发取得重大突破，将在更多城市开展试点应用。"
    ]
    
    print(f"🧪 测试分析 {len(test_news)} 条新闻...")
    
    # 分析情感
    results = analyzer.analyze_batch_single_request(test_news)
    
    # 显示结果
    print("\n📊 分析结果：")
    for i, (news, result) in enumerate(zip(test_news, results)):
        print(f"\n{i+1}. 新闻: {news[:50]}...")
        print(f"   情感得分: {result['deepseek_sentiment_score']}")
        print(f"   情感标签: {result['deepseek_sentiment_label']}")
        print(f"   置信度: {result['deepseek_confidence']}")
        print(f"   市场影响: {result['deepseek_market_impact']}")
        print(f"   关键因素: {result['deepseek_key_factors']}")
        print(f"   分析理由: {result['deepseek_reasoning']}")
    
    # 显示统计信息
    analyzer.print_stats()
    
    # 保存缓存
    analyzer.save_cache()
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main() 