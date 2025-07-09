#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动数据管理系统
在策略运行前自动检测并下载缺失的数据，生成因子
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import subprocess
import glob

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class AutoDataManager:
    """自动数据管理器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化数据管理器"""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
        self._setup_directories()
        
        # 数据路径
        self.price_data_dir = project_root / "data" / "raw" / "price"
        self.news_data_dir = project_root / "data" / "raw" / "news"
        self.factors_data_dir = project_root / "data" / "factors"
        self.processed_data_dir = project_root / "data" / "processed"
        
        # 脚本路径
        self.scripts_dir = project_root / "scripts"
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_file = project_root / self.config_path
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
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
            "data/results",
            "logs"
        ]
        
        for directory in directories:
            (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """检查数据可用性"""
        self.logger.info("🔍 检查数据可用性...")
        
        availability = {
            'price_data': self._check_price_data(),
            'news_data': self._check_news_data(),
            'factor_data': self._check_factor_data()
        }
        
        self.logger.info(f"数据检查结果: {availability}")
        return availability
    
    def _check_price_data(self) -> bool:
        """检查价格数据是否存在"""
        # 检查是否有价格数据文件
        price_files = list(self.price_data_dir.glob("*.csv")) + list(self.price_data_dir.glob("*.parquet"))
        
        if not price_files:
            self.logger.info("❌ 未找到价格数据文件")
            return False
        
        # 检查最新文件的时间范围
        latest_file = sorted(price_files, key=lambda x: x.stat().st_mtime)[-1]
        
        try:
            if latest_file.suffix == '.csv':
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            else:
                df = pd.read_parquet(latest_file)
            
            # 检查数据时间范围
            lookback_days = self.config.get('data', {}).get('price', {}).get('lookback_days', 30)
            required_start = datetime.now() - timedelta(days=lookback_days)
            
            if df.index.max() < required_start:
                self.logger.info(f"⚠️ 价格数据过期，最新数据: {df.index.max()}")
                return False
            
            self.logger.info(f"✅ 价格数据可用，时间范围: {df.index.min()} - {df.index.max()}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 价格数据文件损坏: {e}")
            return False
    
    def _check_news_data(self) -> bool:
        """检查新闻数据是否存在"""
        # 检查是否有新闻数据文件
        news_files = list(self.news_data_dir.glob("*.csv")) + list(self.news_data_dir.glob("*.parquet"))
        
        if not news_files:
            self.logger.info("❌ 未找到新闻数据文件")
            return False
        
        # 检查最新文件的内容
        latest_file = sorted(news_files, key=lambda x: x.stat().st_mtime)[-1]
        
        try:
            if latest_file.suffix == '.csv':
                df = pd.read_csv(latest_file)
            else:
                df = pd.read_parquet(latest_file)
            
            if len(df) == 0:
                self.logger.info("❌ 新闻数据文件为空")
                return False
            
            # 检查关键列是否存在
            required_columns = ['title', 'description', 'published']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.info(f"⚠️ 新闻数据缺少列: {missing_columns}")
                return False
            
            self.logger.info(f"✅ 新闻数据可用，共 {len(df)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 新闻数据文件损坏: {e}")
            return False
    
    def _check_factor_data(self) -> bool:
        """检查因子数据是否存在"""
        # 检查是否有因子数据文件
        factor_files = list(self.factors_data_dir.glob("*.parquet")) + list(self.factors_data_dir.glob("*.csv"))
        
        if not factor_files:
            self.logger.info("❌ 未找到因子数据文件")
            return False
        
        # 检查最新文件的内容
        latest_file = sorted(factor_files, key=lambda x: x.stat().st_mtime)[-1]
        
        try:
            if latest_file.suffix == '.csv':
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            else:
                df = pd.read_parquet(latest_file)
            
            if len(df) == 0:
                self.logger.info("❌ 因子数据文件为空")
                return False
            
            # 检查是否有基本的因子列
            expected_factors = ['sentiment_score', 'news_volume', 'price_return']
            available_factors = [col for col in expected_factors if any(col in df_col for df_col in df.columns)]
            
            if len(available_factors) == 0:
                self.logger.info("⚠️ 因子数据中未找到预期的因子")
                return False
            
            self.logger.info(f"✅ 因子数据可用，共 {df.shape[1]} 个因子，{df.shape[0]} 个时间点")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 因子数据文件损坏: {e}")
            return False
    
    def download_price_data(self) -> bool:
        """下载价格数据（根据配置文件中的时间范围）"""
        self.logger.info("📥 开始下载价格数据...")
        
        # 从配置文件中读取时间范围
        start_date = None
        end_date = None
        try:
            backtest_config = self.config.get('evaluation', {}).get('backtest', {})
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if start_date:
                self.logger.info(f"📅 配置的开始日期: {start_date}")
            if end_date:
                self.logger.info(f"📅 配置的结束日期: {end_date}")
                
        except Exception as e:
            self.logger.warning(f"读取配置文件时间范围失败: {e}")
        
        try:
            # 构建命令 - 不传递days参数，让data_collection.py使用配置文件中的时间范围
            cmd = [
                sys.executable,
                str(self.scripts_dir / "data_collection.py"),
                "--data-type", "price"
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                self.logger.info("✅ 价格数据下载完成")
                return True
            else:
                self.logger.error(f"❌ 价格数据下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 价格数据下载异常: {e}")
            return False
    
    def download_news_data(self) -> bool:
        """下载新闻数据（根据配置文件中的时间范围）"""
        self.logger.info("📰 开始下载新闻数据...")
        
        # 从配置文件中读取时间范围
        start_date = None
        end_date = None
        try:
            backtest_config = self.config.get('evaluation', {}).get('backtest', {})
            start_date = backtest_config.get('start_date')
            end_date = backtest_config.get('end_date')
            
            if start_date:
                self.logger.info(f"📅 配置的开始日期: {start_date}")
            if end_date:
                self.logger.info(f"📅 配置的结束日期: {end_date}")
                
        except Exception as e:
            self.logger.warning(f"读取配置文件时间范围失败: {e}")
        
        try:
            # 构建命令
            cmd = [
                sys.executable,
                str(self.scripts_dir / "data_collection.py"),
                "--data-type", "news"
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                self.logger.info("✅ 新闻数据下载完成")
                return True
            else:
                self.logger.error(f"❌ 新闻数据下载失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 新闻数据下载异常: {e}")
            return False
    
    def generate_factors(self) -> bool:
        """生成因子数据"""
        self.logger.info("🔧 开始生成因子数据...")
        
        try:
            # 构建命令
            cmd = [
                sys.executable,
                str(self.scripts_dir / "factor_construction.py")
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                self.logger.info("✅ 因子数据生成完成")
                return True
            else:
                self.logger.error(f"❌ 因子数据生成失败: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 因子数据生成异常: {e}")
            return False
    
    def ensure_data_ready(self) -> bool:
        """确保所有数据都准备就绪"""
        self.logger.info("🚀 开始自动数据管理流程...")
        
        # 检查数据可用性
        availability = self.check_data_availability()
        
        success = True
        
        # 处理价格数据
        if not availability['price_data']:
            self.logger.info("需要下载价格数据...")
            if not self.download_price_data():
                success = False
        
        # 处理新闻数据
        if not availability['news_data']:
            self.logger.info("需要下载新闻数据...")
            if not self.download_news_data():
                success = False
        
        # 处理因子数据
        if not availability['factor_data']:
            self.logger.info("需要生成因子数据...")
            # 如果价格数据或新闻数据刚下载，也需要重新生成因子
            if not self.generate_factors():
                success = False
        elif not availability['price_data'] or not availability['news_data']:
            # 如果价格或新闻数据更新了，重新生成因子
            self.logger.info("数据更新，需要重新生成因子...")
            if not self.generate_factors():
                success = False
        
        if success:
            self.logger.info("🎉 所有数据已准备就绪！")
        else:
            self.logger.error("⚠️ 数据准备过程中出现问题")
        
        return success
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        summary = {
            'price_data': {},
            'news_data': {},
            'factor_data': {}
        }
        
        # 价格数据摘要
        try:
            price_files = list(self.price_data_dir.glob("*.csv")) + list(self.price_data_dir.glob("*.parquet"))
            if price_files:
                latest_file = sorted(price_files, key=lambda x: x.stat().st_mtime)[-1]
                if latest_file.suffix == '.csv':
                    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                else:
                    df = pd.read_parquet(latest_file)
                
                summary['price_data'] = {
                    'file_count': len(price_files),
                    'latest_file': latest_file.name,
                    'record_count': len(df),
                    'date_range': f"{df.index.min()} - {df.index.max()}",
                    'columns': list(df.columns)
                }
        except Exception as e:
            summary['price_data']['error'] = str(e)
        
        # 新闻数据摘要
        try:
            news_files = list(self.news_data_dir.glob("*.csv")) + list(self.news_data_dir.glob("*.parquet"))
            if news_files:
                latest_file = sorted(news_files, key=lambda x: x.stat().st_mtime)[-1]
                if latest_file.suffix == '.csv':
                    df = pd.read_csv(latest_file)
                else:
                    df = pd.read_parquet(latest_file)
                
                summary['news_data'] = {
                    'file_count': len(news_files),
                    'latest_file': latest_file.name,
                    'record_count': len(df),
                    'columns': list(df.columns)
                }
        except Exception as e:
            summary['news_data']['error'] = str(e)
        
        # 因子数据摘要
        try:
            factor_files = list(self.factors_data_dir.glob("*.parquet")) + list(self.factors_data_dir.glob("*.csv"))
            if factor_files:
                latest_file = sorted(factor_files, key=lambda x: x.stat().st_mtime)[-1]
                if latest_file.suffix == '.csv':
                    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                else:
                    df = pd.read_parquet(latest_file)
                
                summary['factor_data'] = {
                    'file_count': len(factor_files),
                    'latest_file': latest_file.name,
                    'factor_count': df.shape[1],
                    'record_count': df.shape[0],
                    'factors': list(df.columns)
                }
        except Exception as e:
            summary['factor_data']['error'] = str(e)
        
        return summary
    
    def clean_old_data(self, keep_days: int = 7) -> bool:
        """清理旧数据文件"""
        self.logger.info(f"🧹 清理 {keep_days} 天前的旧数据...")
        
        cutoff_time = datetime.now() - timedelta(days=keep_days)
        cleaned_count = 0
        
        for data_dir in [self.price_data_dir, self.news_data_dir, self.factors_data_dir]:
            if not data_dir.exists():
                continue
                
            for file_path in data_dir.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            self.logger.info(f"已删除旧文件: {file_path.name}")
                        except Exception as e:
                            self.logger.error(f"删除文件失败 {file_path.name}: {e}")
        
        self.logger.info(f"✅ 清理完成，删除了 {cleaned_count} 个文件")
        return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动数据管理工具")
    parser.add_argument("--check", action="store_true", help="只检查数据可用性")
    parser.add_argument("--download-price", action="store_true", help="强制下载价格数据")
    parser.add_argument("--download-news", action="store_true", help="强制下载新闻数据")
    parser.add_argument("--generate-factors", action="store_true", help="强制生成因子数据")
    parser.add_argument("--summary", action="store_true", help="显示数据摘要")
    parser.add_argument("--clean", type=int, metavar="DAYS", help="清理N天前的旧数据")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        manager = AutoDataManager(args.config)
        
        if args.check:
            # 只检查数据可用性
            availability = manager.check_data_availability()
            print(f"\n数据可用性检查结果:")
            print(f"价格数据: {'✅' if availability['price_data'] else '❌'}")
            print(f"新闻数据: {'✅' if availability['news_data'] else '❌'}")
            print(f"因子数据: {'✅' if availability['factor_data'] else '❌'}")
            
        elif args.download_price:
            # 强制下载价格数据
            success = manager.download_price_data()
            return 0 if success else 1
            
        elif args.download_news:
            # 强制下载新闻数据
            success = manager.download_news_data()
            return 0 if success else 1
            
        elif args.generate_factors:
            # 强制生成因子数据
            success = manager.generate_factors()
            return 0 if success else 1
            
        elif args.summary:
            # 显示数据摘要
            summary = manager.get_data_summary()
            print(f"\n📊 数据摘要报告:")
            print(f"{'='*50}")
            
            for data_type, info in summary.items():
                print(f"\n{data_type.upper()}:")
                if 'error' in info:
                    print(f"  ❌ 错误: {info['error']}")
                else:
                    for key, value in info.items():
                        print(f"  {key}: {value}")
            
        elif args.clean:
            # 清理旧数据
            success = manager.clean_old_data(args.clean)
            return 0 if success else 1
            
        else:
            # 默认执行完整的数据准备流程
            success = manager.ensure_data_ready()
            return 0 if success else 1
            
    except Exception as e:
        print(f"❌ 自动数据管理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 