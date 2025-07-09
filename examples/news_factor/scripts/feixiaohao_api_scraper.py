#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
非小号API新闻抓取器
使用正确的分页逻辑：时间戳递减向历史方向翻页
"""

import time
import requests
import pandas as pd
import json
import brotli
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional
import argparse


class FeixiaohaoAPIScraper:
    """非小号API新闻抓取器"""
    
    def __init__(self, output_dir: str = "data/raw/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_url = "https://api.fxhnews.com/api/v4/news/news"
        self._setup_logging()
        self._setup_session()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_session(self):
        """设置请求会话"""
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh-Hans;q=0.9',
            'Origin': 'https://feixiaohao.com.cn',
            'Referer': 'https://feixiaohao.com.cn/',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15'
        })
        
    def _decode_response(self, response) -> Dict:
        """解码Brotli压缩的响应"""
        try:
            return response.json()
        except json.JSONDecodeError:
            # API使用Brotli压缩
            try:
                decompressed = brotli.decompress(response.content)
                return json.loads(decompressed.decode('utf-8'))
            except Exception as e:
                self.logger.error(f"解码响应失败: {e}")
                return {}
    
    def scrape_news(self, max_pages: int = 50, per_page: int = 100, start_date: str = None, end_date: str = None) -> List[Dict]:
        """抓取新闻数据"""
        if start_date and end_date:
            self.logger.info(f"开始抓取新闻数据，时间范围: {start_date} - {end_date}，最多{max_pages}页，每页{per_page}条...")
        else:
            self.logger.info(f"开始抓取新闻数据，最多{max_pages}页，每页{per_page}条...")
        
        all_news = []
        current_timestamp = None
        
        # 转换时间范围为时间戳
        start_timestamp = None
        end_timestamp = None
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                start_timestamp = int(start_dt.timestamp())
            except ValueError:
                self.logger.warning(f"开始日期格式错误: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # 包含结束日期
                end_timestamp = int(end_dt.timestamp())
            except ValueError:
                self.logger.warning(f"结束日期格式错误: {end_date}")
        
        self.logger.info(f"时间戳范围: {start_timestamp} - {end_timestamp}")
        
        # 如果提供了end_date，从end_timestamp开始抓取
        if end_timestamp:
            current_timestamp = end_timestamp
            self.logger.info(f"从结束时间开始抓取: {end_date} (timestamp: {end_timestamp})")
        
        base_params = {
            'channelid': 24,
            'direction': 1,
            'per_page': per_page,
            'isfxh': 0,
            'webp': 0
        }
        
        for page in range(max_pages):
            try:
                params = base_params.copy()
                
                if page == 0 and current_timestamp is None:
                    self.logger.info(f"第{page+1}页: 首页")
                else:
                    if current_timestamp is None:
                        self.logger.warning("无法获取时间戳，停止抓取")
                        break
                    params['timestamp'] = current_timestamp
                    self.logger.info(f"第{page+1}页: timestamp={current_timestamp}")
                
                response = self.session.get(self.api_url, params=params, timeout=15)
                
                if response.status_code != 200:
                    self.logger.error(f"请求失败: {response.status_code}")
                    break
                
                data = self._decode_response(response)
                news_items = self._parse_news_list(data)
                
                if not news_items:
                    self.logger.info("没有更多数据，停止抓取")
                    break
                
                # 时间范围过滤
                filtered_news = []
                should_stop = False
                
                for news in news_items:
                    # 解析新闻时间戳
                    published_str = news.get('published', '')
                    try:
                        published_dt = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                        news_timestamp = int(published_dt.timestamp())
                        
                        # 检查是否已经抓取到开始时间之前的数据
                        if start_timestamp and news_timestamp < start_timestamp:
                            self.logger.info(f"新闻时间 {published_str} 早于开始时间 {start_date}，停止抓取")
                            should_stop = True
                            break
                        
                        # 检查是否在时间范围内
                        if end_timestamp and news_timestamp > end_timestamp:
                            # 新闻太新，跳过
                            continue
                        
                        # 符合时间范围的新闻
                        filtered_news.append(news)
                        
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"解析新闻时间失败: {published_str}, {e}")
                        # 如果没有时间限制，仍然保留这条新闻
                        if not start_timestamp and not end_timestamp:
                            filtered_news.append(news)
                
                # 添加符合条件的新闻
                if filtered_news:
                    all_news.extend(filtered_news)
                    self.logger.info(f"第{page+1}页获取到 {len(news_items)} 条新闻，过滤后 {len(filtered_news)} 条")
                else:
                    self.logger.info(f"第{page+1}页获取到 {len(news_items)} 条新闻，无符合时间范围的新闻")
                
                # 检查是否需要停止抓取
                if should_stop:
                    self.logger.info(f"已抓取到目标时间范围之前的数据，停止抓取")
                    break
                
                # 获取下一页时间戳
                current_timestamp = self._get_next_timestamp(data)
                
                if current_timestamp is None:
                    self.logger.warning("无法获取下一页时间戳，停止抓取")
                    break
                
                time.sleep(1)  # 避免请求过快
                
            except Exception as e:
                self.logger.error(f"第{page+1}页抓取失败: {e}")
                break
        
        self.logger.info(f"总共抓取到 {len(all_news)} 条新闻")
        return all_news
    
    def _parse_news_list(self, data: Dict) -> List[Dict]:
        """解析新闻列表"""
        try:
            # 检查数据结构
            if not data or 'data' not in data:
                self.logger.warning("响应数据结构不正确")
                return []
            
            news_list = data['data'].get('list', [])
            news_items = []
            
            for item in news_list:
                news_item = self._parse_news_item(item)
                if news_item:
                    news_items.append(news_item)
            
            return news_items
            
        except (KeyError, TypeError) as e:
            self.logger.error(f"解析新闻列表失败: {e}")
            return []
    
    def _parse_news_item(self, item: Dict) -> Optional[Dict]:
        """解析单条新闻"""
        try:
            # 提取标题
            title = item.get('title', '').strip()
            if not title or len(title) < 5:
                return None
            
            # 提取内容（如果有的话，否则使用标题）
            content = item.get('content', title).strip()
            
            # 解析时间戳
            timestamp = item.get('issuetime')
            if timestamp:
                try:
                    published = datetime.fromtimestamp(int(timestamp)).isoformat()
                except (ValueError, TypeError):
                    published = datetime.now().isoformat()
            else:
                published = datetime.now().isoformat()
            
            # 提取其他字段
            url = item.get('url', '')
            news_id = item.get('id', '')
            
            return {
                'title': title,
                'description': content,
                'link': url,
                'published': published,
                'source': '非小号',
                'source_id': f'feixiaohao_api_{news_id}' if news_id else 'feixiaohao_api'
            }
            
        except Exception as e:
            self.logger.warning(f"解析新闻项目失败: {e}")
            return None
    
    def _get_next_timestamp(self, data: Dict) -> Optional[int]:
        """获取下一页时间戳（当前页最小时间戳-1）"""
        try:
            if not data or 'data' not in data:
                return None
                
            news_list = data['data'].get('list', [])
            timestamps = []
            
            for item in news_list:
                ts = item.get('issuetime')
                if ts and isinstance(ts, (int, float)):
                    timestamps.append(int(ts))
            
            if timestamps:
                min_timestamp = min(timestamps)
                next_timestamp = min_timestamp - 1  # 向历史方向翻页
                self.logger.debug(f"时间戳范围: {min(timestamps)} - {max(timestamps)}, 下一页: {next_timestamp}")
                return next_timestamp
                
        except Exception as e:
            self.logger.warning(f"提取时间戳失败: {e}")
        
        return None
    
    def save_data(self, news_data: List[Dict]) -> str:
        """保存数据到CSV文件"""
        if not news_data:
            raise ValueError("没有数据可保存")
        
        # 去重（基于标题）
        seen_titles = set()
        unique_news = []
        
        for news in news_data:
            title = news.get('title', '').strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        # 创建DataFrame
        df = pd.DataFrame(unique_news)
        
        # 保存到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feixiaohao_news_{timestamp}.csv"
        file_path = self.output_dir / filename
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        self.logger.info(f"数据已保存至: {file_path}")
        self.logger.info(f"去重后共 {len(unique_news)} 条新闻")
        
        return str(file_path)
    
    def run(self, max_pages: int = 50, per_page: int = 100, start_date: str = None, end_date: str = None) -> str:
        """运行完整的抓取流程"""
        try:
            # 抓取新闻
            news_data = self.scrape_news(max_pages=max_pages, per_page=per_page, start_date=start_date, end_date=end_date)
            
            if not news_data:
                if start_date and end_date:
                    raise ValueError(f"在指定时间范围 {start_date} - {end_date} 内未获取到任何新闻数据")
                else:
                    raise ValueError("未获取到任何新闻数据")
            
            # 保存数据
            file_path = self.save_data(news_data)
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"抓取流程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="非小号API新闻抓取器")
    parser.add_argument("--max-pages", type=int, default=200, help="最大抓取页数")
    parser.add_argument("--per-page", type=int, default=100, help="每页新闻数量") 
    parser.add_argument("--output-dir", default="data/raw/news", help="输出目录")
    parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        scraper = FeixiaohaoAPIScraper(args.output_dir)
        file_path = scraper.run(
            max_pages=args.max_pages, 
            per_page=args.per_page,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        print(f"✅ 抓取完成！数据保存至: {file_path}")
        
        if args.start_date and args.end_date:
            print(f"📅 时间范围: {args.start_date} - {args.end_date}")
        
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 