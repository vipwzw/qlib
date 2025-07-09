#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻情感因子分析 - 数据采集脚本
支持BTC/USDT价格数据和多源新闻数据采集
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml
import logging
import time
import requests
import feedparser
from urllib.parse import urljoin

# 导入非小号API抓取器
try:
    from feixiaohao_api_scraper import FeixiaohaoAPIScraper
    HAS_FEIXIAOHAO_SCRAPER = True
except ImportError:
    HAS_FEIXIAOHAO_SCRAPER = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 添加utils路径以使用配置加载器
utils_path = project_root / "utils"
if utils_path.exists():
    sys.path.append(str(utils_path))
    try:
        from config_loader import ConfigLoader
        HAS_CONFIG_LOADER = True
    except ImportError:
        HAS_CONFIG_LOADER = False
else:
    HAS_CONFIG_LOADER = False

class DataCollectionManager:
    """数据采集管理器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化数据采集管理器"""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        self._setup_directories()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if HAS_CONFIG_LOADER:
            try:
                # 使用配置加载器加载环境变量
                loader = ConfigLoader(config_file=self.config_path)
                return loader.load_config()
            except Exception as e:
                self.logger.warning(f"配置加载器失败，使用传统方式: {e}")
        
        # 传统方式加载配置
        config_file = project_root / self.config_path
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        """创建必要的目录"""
        directories = [
            "data/raw/price",
            "data/raw/news", 
            "data/processed",
            "data/factors",
            "logs"
        ]
        
        for directory in directories:
            (project_root / directory).mkdir(parents=True, exist_ok=True)

    def collect_price_data(self, days: int = 30) -> bool:
        """
        从Binance API采集真实的BTC/USDT 1分钟K线数据
        根据配置文件中的回测时间范围下载数据
        """
        
        # 优先从配置文件读取时间范围
        start_date = None
        end_date = None
        try:
            backtest_config = self.config.get('evaluation', {}).get('backtest', {})
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if start_date and end_date:
                self.logger.info(f"📅 使用配置文件时间范围: {start_date} - {end_date}")
            else:
                self.logger.info(f"⚠️ 配置文件未指定时间范围，使用默认回看 {days} 天")
                
        except Exception as e:
            self.logger.warning(f"读取配置文件时间范围失败: {e}")
            
        self.logger.info(f"开始从Binance API采集BTC/USDT价格数据")
        
        try:
            # 计算时间范围（Binance API使用毫秒时间戳）
            if start_date and end_date:
                # 使用配置文件中的时间范围
                start_time = datetime.strptime(start_date, '%Y-%m-%d')
                end_time = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # 包含结束日期
            else:
                # 使用默认天数
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
            
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            self.logger.info(f"数据时间范围: {start_time} - {end_time}")
            
            # Binance API配置
            base_url = "https://api.binance.com"
            symbol = "BTCUSDT"
            interval = "1m"  # 1分钟K线
            limit = 1000  # 每次请求最多1000条数据
            
            all_data = []
            current_start = start_timestamp
            
            # 分批获取数据（因为API有单次请求限制）
            while current_start < end_timestamp:
                try:
                    # 构建API请求URL
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': current_start,
                        'endTime': end_timestamp,
                        'limit': limit
                    }
                    
                    url = f"{base_url}/api/v3/klines"
                    
                    self.logger.info(f"请求数据: {datetime.fromtimestamp(current_start/1000)}")
                    
                    # 发送请求
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code != 200:
                        self.logger.error(f"API请求失败: {response.status_code} - {response.text}")
                        break
                    
                    data = response.json()
                    
                    if not data:
                        self.logger.info("没有更多数据")
                        break
                    
                    # 添加数据到列表
                    all_data.extend(data)
                    
                    # 更新下一批的开始时间（最后一条数据的时间 + 1分钟）
                    last_timestamp = data[-1][0]
                    current_start = last_timestamp + 60000  # 加1分钟（60000毫秒）
                    
                    self.logger.info(f"获取到 {len(data)} 条数据，总计 {len(all_data)} 条")
                    
                    # 添加请求间隔，避免触发API限制
                    import time
                    time.sleep(0.1)
                    
                except requests.RequestException as e:
                    self.logger.error(f"网络请求失败: {e}")
                    break
                except Exception as e:
                    self.logger.error(f"数据处理失败: {e}")
                    break
            
            if not all_data:
                self.logger.error("未能获取任何价格数据")
                return False
            
            # 转换数据格式
            self.logger.info(f"开始处理 {len(all_data)} 条K线数据")
            
            # Binance K线数据格式:
            # [时间戳, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, 主动买入成交量, 主动买入成交额, 忽略]
            processed_data = []
            
            for kline in all_data:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                processed_data.append({
                    'timestamp': timestamp,
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            # 创建DataFrame
            price_data = pd.DataFrame(processed_data)
            
            # 按时间排序
            price_data = price_data.sort_values('timestamp').reset_index(drop=True)
            
            # 去重（以防API返回重复数据）
            price_data = price_data.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # 验证数据质量
            self.logger.info(f"数据验证:")
            self.logger.info(f"  时间范围: {price_data['timestamp'].min()} - {price_data['timestamp'].max()}")
            self.logger.info(f"  价格范围: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
            self.logger.info(f"  平均成交量: {price_data['volume'].mean():.2f}")
            
            # 保存数据
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"btc_usdt_1m_real_{timestamp_str}.csv"
            output_path = project_root / "data" / "raw" / "price" / filename
            
            price_data.to_csv(output_path, index=False)
            
            self.logger.info(f"✅ 真实价格数据采集完成，共 {len(price_data)} 条记录")
            self.logger.info(f"📁 数据已保存至: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 价格数据采集失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False

    def collect_news_data(self) -> bool:
        """采集新闻数据"""
        self.logger.info("开始采集新闻数据")
        
        # 从配置文件中读取时间范围
        start_date = None
        end_date = None
        try:
            backtest_config = self.config.get('evaluation', {}).get('backtest', {})
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if start_date:
                self.logger.info(f"📅 配置文件指定开始日期: {start_date}")
            if end_date:
                self.logger.info(f"📅 配置文件指定结束日期: {end_date}")
                
        except Exception as e:
            self.logger.warning(f"读取配置文件时间范围失败: {e}")
        
        try:
            news_sources = self.config.get('news_sources', {})
            all_news = []
            
            for source_name, source_config in news_sources.items():
                if not source_config.get('enabled', False):
                    continue
                
                self.logger.info(f"从 {source_config.get('name', source_name)} 采集新闻...")
                
                if source_config.get('type') == 'rss':
                    news_items = self._collect_rss_news(source_config, start_date, end_date)
                    all_news.extend(news_items)
                    self.logger.info(f"从 {source_name} 采集到 {len(news_items)} 条新闻")
                elif source_config.get('type') == 'api' and source_name == 'feixiaohao':
                    news_items = self._collect_feixiaohao_news(source_config, start_date, end_date)
                    all_news.extend(news_items)
                    self.logger.info(f"从 {source_name} 采集到 {len(news_items)} 条新闻")
            
            if all_news:
                # 创建DataFrame
                news_df = pd.DataFrame(all_news)
                
                # 去重和过滤
                news_df = news_df.drop_duplicates(subset=['title'])
                news_df = self._filter_crypto_news(news_df)
                
                # 应用时间范围过滤
                if start_date or end_date:
                    original_count = len(news_df)
                    news_df = self._filter_by_date_range(news_df, start_date, end_date)
                    filtered_count = len(news_df)
                    self.logger.info(f"时间范围过滤：{original_count} -> {filtered_count} 条新闻")
                
                # 保存数据
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"crypto_news_{timestamp}.csv"
                output_path = project_root / "data" / "raw" / "news" / filename
                
                news_df.to_csv(output_path, index=False)
                
                self.logger.info(f"过滤后剩余 {len(news_df)} 条相关新闻")
                self.logger.info(f"新闻数据已保存至: {output_path}")
                
                return True
            else:
                self.logger.warning("未采集到任何新闻数据")
                return False
                
        except Exception as e:
            self.logger.error(f"新闻数据采集失败: {e}")
            return False

    def _collect_rss_news(self, source_config: Dict, start_date: str = None, end_date: str = None) -> List[Dict]:
        """从RSS源采集新闻
        
        Args:
            source_config: 新闻源配置
            start_date: 开始日期 (配置文件中获取)
            end_date: 结束日期 (配置文件中获取)
        """
        news_items = []
        
        try:
            url = source_config.get('url')
            if not url:
                self.logger.warning(f"RSS源配置缺少URL: {source_config.get('name')}")
                return news_items
            
            self.logger.info(f"正在访问RSS源: {url}")
            
            # 设置请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache'
            }
            
            # 使用requests获取RSS内容
            response = requests.get(url, headers=headers, timeout=15)
            self.logger.info(f"HTTP状态码: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.warning(f"HTTP请求失败: {response.status_code}")
                return news_items
            
            # 使用feedparser解析响应内容
            feed = feedparser.parse(response.text)
            
            # 检查解析结果
            if hasattr(feed, 'bozo') and feed.bozo:
                self.logger.warning(f"RSS解析可能有问题: {getattr(feed, 'bozo_exception', 'Unknown error')}")
            
            # 检查是否有条目
            if not hasattr(feed, 'entries') or not feed.entries:
                self.logger.warning(f"RSS源没有找到任何条目: {url}")
                return news_items
            
            self.logger.info(f"RSS源返回 {len(feed.entries)} 个条目")
            
            for entry in feed.entries[:20]:  # 限制为最新20条
                try:
                    # 处理发布时间
                    published = entry.get('published', '')
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6]).isoformat()
                        except:
                            published = entry.get('published', '')
                    
                    # 提取描述或摘要
                    description = entry.get('description', '')
                    if not description:
                        description = entry.get('summary', '')
                    
                    news_item = {
                        'title': entry.get('title', '').strip(),
                        'description': self._clean_text(description),
                        'link': entry.get('link', ''),
                        'published': published,
                        'source': source_config.get('name', 'Unknown')
                    }
                    
                    # 只添加有标题的新闻
                    if news_item['title']:
                        news_items.append(news_item)
                        
                except Exception as e:
                    self.logger.error(f"处理RSS条目时出错: {e}")
                    continue
                
        except requests.RequestException as e:
            self.logger.error(f"网络请求失败 {source_config.get('url')}: {e}")
        except Exception as e:
            self.logger.error(f"RSS采集失败 {source_config.get('url')}: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
        
        return news_items
    
    def _collect_feixiaohao_news(self, source_config: Dict, start_date: str = None, end_date: str = None) -> List[Dict]:
        """从非小号API采集新闻
        
        Args:
            source_config: 新闻源配置  
            start_date: 开始日期 (配置文件中获取)
            end_date: 结束日期 (配置文件中获取)
        """
        news_items = []
        
        try:
            if not HAS_FEIXIAOHAO_SCRAPER:
                self.logger.error("非小号API抓取器未找到，请确保feixiaohao_api_scraper.py文件存在")
                return news_items
            
            # 从配置中获取参数
            max_pages = source_config.get('max_pages', 50)
            per_page = source_config.get('per_page', 100)
            
            self.logger.info(f"开始从非小号API采集新闻，最多{max_pages}页，每页{per_page}条")
            
            # 初始化非小号抓取器
            output_dir = project_root / "data" / "raw" / "news"
            scraper = FeixiaohaoAPIScraper(str(output_dir))
            
            # 抓取新闻数据，传递时间范围参数
            news_data = scraper.scrape_news(max_pages=max_pages, per_page=per_page, start_date=start_date, end_date=end_date)
            
            if news_data:
                # 转换数据格式以匹配其他新闻源的格式
                for item in news_data:
                    news_item = {
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'link': item.get('link', ''),
                        'published': item.get('published', ''),
                        'source': item.get('source', '非小号')
                    }
                    
                    # 只添加有标题的新闻
                    if news_item['title']:
                        news_items.append(news_item)
                        
                self.logger.info(f"成功从非小号API获取到 {len(news_items)} 条新闻")
            else:
                self.logger.warning("非小号API未返回任何新闻数据")
                
        except Exception as e:
            self.logger.error(f"非小号API采集失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
        
        return news_items
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""
        
        import re
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _filter_by_date_range(self, news_df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """根据时间范围过滤新闻
        
        Args:
            news_df: 新闻数据框
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            
        Returns:
            过滤后的新闻数据框
        """
        if news_df.empty:
            return news_df
        
        try:
            # 处理发布时间列
            if 'published' not in news_df.columns:
                self.logger.warning("新闻数据中没有 'published' 列，无法进行时间过滤")
                return news_df
            
            # 转换发布时间为datetime
            news_df['published_dt'] = pd.to_datetime(news_df['published'], errors='coerce')
            
            # 移除无法解析时间的记录
            before_count = len(news_df)
            news_df = news_df.dropna(subset=['published_dt'])
            after_count = len(news_df)
            
            if before_count != after_count:
                self.logger.info(f"移除了 {before_count - after_count} 条无法解析时间的新闻")
            
            # 应用时间范围过滤
            if start_date:
                start_dt = pd.to_datetime(start_date)
                news_df = news_df[news_df['published_dt'] >= start_dt]
                self.logger.info(f"过滤开始日期 {start_date} 之前的新闻")
            
            if end_date:
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # 包含结束日期当天
                news_df = news_df[news_df['published_dt'] < end_dt]
                self.logger.info(f"过滤结束日期 {end_date} 之后的新闻")
            
            # 删除临时列
            news_df = news_df.drop('published_dt', axis=1)
            
            return news_df
            
        except Exception as e:
            self.logger.error(f"时间范围过滤失败: {e}")
            return news_df

    def _filter_crypto_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """过滤与加密货币相关的新闻"""
        # 英文关键词
        crypto_keywords_en = [
            'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'blockchain',
            'ethereum', 'trading', 'market', 'price', 'investment', 'eth',
            'usdt', 'tether', 'binance', 'coinbase', 'defi', 'nft',
            # 稳定币相关
            'stablecoin', 'stable coin', 'usdc', 'dai', 'busd',
            # 挖矿相关公司
            'core scientific', 'coreweave', 'marathon', 'riot', 'bitfarms',
            'hut 8', 'cleanspark', 'bitdeer', 'canaan', 'bitmain',
            # 挖矿相关术语
            'mining', 'miner', 'hashrate', 'hash rate', 'asic', 'proof of work',
            # 其他重要术语
            'wallet', 'exchange', 'custody', 'hodl', 'satoshi', 'wei',
            'smart contract', 'dapp', 'layer 2', 'lightning network',
            # 重要公司和项目
            'microstrategy', 'tesla', 'grayscale', 'blackrock', 'fidelity',
            'solana', 'cardano', 'polkadot', 'chainlink', 'uniswap'
        ]
        
        # 中文关键词（适用于非小号等中文新闻源）
        crypto_keywords_cn = [
            '比特币', 'BTC', '以太坊', 'ETH', '加密货币', '数字货币', '虚拟货币',
            '区块链', '币安', '交易所', '挖矿', '钱包', 'USDT', '泰达币',
            '去中心化', 'DeFi', 'NFT', '代币', '合约', '公链', '私链',
            '矿机', '矿池', '分叉', '硬分叉', '软分叉', '稳定币',
            '挖矿', '矿工', '算力', '哈希率', '减半', '链上', '链下',
            '钱包地址', '私钥', '助记词', '冷钱包', '热钱包',
            # 重要公司和项目名称
            '微策略', 'MicroStrategy', '灰度', 'Grayscale', '贝莱德', 'BlackRock',
            '嘉楠科技', '比特大陆', 'Bitmain', '蚂蚁矿机', 'Antminer',
            '币安网', 'Binance', '欧易', 'OKX', '火币', 'Huobi',
            # 其他术语
            '智能合约', '去中心化应用', 'DApp', '元宇宙', '闪电网络',
            '分布式账本', '共识机制', '工作量证明', '权益证明',
            'Layer2', '二层网络', '跨链', '侧链', '原子交换',
            '流动性', '做市商', '套利', '量化交易', '高频交易'
        ]
        
        # 合并所有关键词
        all_keywords = crypto_keywords_en + crypto_keywords_cn
        
        # 创建关键词过滤条件（对于中文关键词，不转换大小写）
        # 先检查英文关键词（转小写）
        title_filter_en = news_df['title'].str.lower().str.contains('|'.join(crypto_keywords_en), na=False)
        desc_filter_en = news_df['description'].str.lower().str.contains('|'.join(crypto_keywords_en), na=False)
        
        # 再检查中文关键词（不转大小写）
        title_filter_cn = news_df['title'].str.contains('|'.join(crypto_keywords_cn), na=False)
        desc_filter_cn = news_df['description'].str.contains('|'.join(crypto_keywords_cn), na=False)
        
        # 返回包含任何关键词的新闻
        return news_df[(title_filter_en | desc_filter_en) | (title_filter_cn | desc_filter_cn)]

    def collect_data(self, data_type: str, days: int = 30) -> bool:
        """统一数据采集接口"""
        success = True
        
        if data_type in ['price', 'all']:
            if not self.collect_price_data(days):
                success = False
        
        if data_type in ['news', 'all']:
            if not self.collect_news_data():
                success = False
        
        return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻情感因子数据采集工具")
    parser.add_argument("--data-type", choices=["price", "news", "all"], 
                       default="all", help="采集的数据类型")
    parser.add_argument("--days", type=int, default=30, 
                       help="价格数据回看天数")
    parser.add_argument("--config", default="configs/config.yaml", 
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化数据采集管理器
    try:
        manager = DataCollectionManager(args.config)
        
        # 执行数据采集
        success = manager.collect_data(args.data_type, args.days)
        
        if success:
            print("✅ 数据采集任务完成！")
        else:
            print("⚠️ 数据采集完成，但部分任务失败")
            return 1
            
    except Exception as e:
        print(f"❌ 数据采集失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 