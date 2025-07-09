#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻数据统计分析脚本
分析每天的新闻数量分布和时间模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime, timedelta
import argparse
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NewsStatistics:
    """新闻数据统计分析器"""
    
    def __init__(self, data_dir: str = "data/raw/news"):
        """初始化统计分析器"""
        self.data_dir = Path(data_dir)
        self.news_data = None
        
    def load_latest_news_data(self):
        """加载最新的新闻数据"""
        try:
            # 查找最新的crypto_news文件
            news_files = list(self.data_dir.glob("crypto_news_*.csv"))
            
            if not news_files:
                print("❌ 未找到新闻数据文件")
                return False
            
            # 选择最新的文件
            latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
            print(f"📰 加载新闻数据: {latest_file.name}")
            
            # 读取数据
            self.news_data = pd.read_csv(latest_file)
            
            # 处理时间列
            self.news_data['published_dt'] = pd.to_datetime(self.news_data['published'], errors='coerce')
            self.news_data = self.news_data.dropna(subset=['published_dt'])
            
            # 提取日期和时间特征
            self.news_data['date'] = self.news_data['published_dt'].dt.date
            self.news_data['hour'] = self.news_data['published_dt'].dt.hour
            self.news_data['day_of_week'] = self.news_data['published_dt'].dt.day_name()
            
            print(f"✅ 成功加载 {len(self.news_data)} 条新闻数据")
            print(f"📅 时间范围: {self.news_data['published_dt'].min()} - {self.news_data['published_dt'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载新闻数据失败: {e}")
            return False
    
    def daily_news_statistics(self):
        """分析每天的新闻数量"""
        if self.news_data is None:
            print("❌ 请先加载新闻数据")
            return
        
        print("\n" + "="*60)
        print("📊 每日新闻数量统计")
        print("="*60)
        
        # 按日期统计
        daily_counts = self.news_data.groupby('date').size().reset_index(name='news_count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.sort_values('date')
        
        # 基础统计
        total_days = len(daily_counts)
        total_news = daily_counts['news_count'].sum()
        avg_daily = daily_counts['news_count'].mean()
        max_daily = daily_counts['news_count'].max()
        min_daily = daily_counts['news_count'].min()
        
        print(f"📈 统计周期: {total_days} 天")
        print(f"📰 总新闻数: {total_news:,} 条")
        print(f"📊 日均新闻: {avg_daily:.1f} 条")
        print(f"🔝 最多一天: {max_daily} 条")
        print(f"🔻 最少一天: {min_daily} 条")
        
        # 显示每日详细数据
        print(f"\n📅 每日新闻数量详情:")
        print("-" * 40)
        for _, row in daily_counts.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            weekday = row['date'].strftime('%A')
            count = row['news_count']
            print(f"{date_str} ({weekday}): {count:3d} 条")
        
        return daily_counts
    
    def hourly_distribution(self):
        """分析小时分布"""
        if self.news_data is None:
            print("❌ 请先加载新闻数据")
            return
        
        print("\n" + "="*60)
        print("🕐 小时分布统计")
        print("="*60)
        
        hourly_counts = self.news_data.groupby('hour').size()
        
        print("小时分布:")
        for hour, count in hourly_counts.items():
            bar = "█" * min(50, int(count * 50 / hourly_counts.max()))
            print(f"{hour:2d}时: {count:4d} 条 {bar}")
        
        return hourly_counts
    
    def weekday_distribution(self):
        """分析星期分布"""
        if self.news_data is None:
            print("❌ 请先加载新闻数据")
            return
        
        print("\n" + "="*60)
        print("📅 星期分布统计")
        print("="*60)
        
        # 确保星期顺序正确
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = self.news_data['day_of_week'].value_counts().reindex(weekday_order, fill_value=0)
        
        weekday_names_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        for i, (weekday, count) in enumerate(weekday_counts.items()):
            bar = "█" * min(30, int(count * 30 / weekday_counts.max()))
            print(f"{weekday_names_cn[i]} ({weekday}): {count:4d} 条 {bar}")
        
        return weekday_counts
    
    def source_distribution(self):
        """分析新闻源分布"""
        if self.news_data is None:
            print("❌ 请先加载新闻数据")
            return
        
        print("\n" + "="*60)
        print("📡 新闻源分布统计")
        print("="*60)
        
        source_counts = self.news_data['source'].value_counts()
        
        for source, count in source_counts.items():
            percentage = count / len(self.news_data) * 100
            bar = "█" * min(40, int(count * 40 / source_counts.max()))
            print(f"{source}: {count:4d} 条 ({percentage:.1f}%) {bar}")
        
        return source_counts
    
    def create_visualizations(self, output_dir: str = "data/results"):
        """创建可视化图表"""
        if self.news_data is None:
            print("❌ 请先加载新闻数据")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置图表风格
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('新闻数据统计分析', fontsize=16, fontweight='bold')
        
        # 1. 每日新闻数量趋势
        daily_counts = self.news_data.groupby('date').size().reset_index(name='news_count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.sort_values('date')
        
        axes[0, 0].plot(daily_counts['date'], daily_counts['news_count'], 
                       marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('每日新闻数量趋势')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('新闻数量')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 小时分布
        hourly_counts = self.news_data.groupby('hour').size()
        axes[0, 1].bar(hourly_counts.index, hourly_counts.values, alpha=0.7)
        axes[0, 1].set_title('小时分布')
        axes[0, 1].set_xlabel('小时')
        axes[0, 1].set_ylabel('新闻数量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 星期分布
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_counts = self.news_data['day_of_week'].value_counts().reindex(weekday_order, fill_value=0)
        weekday_names_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        axes[1, 0].bar(weekday_names_cn, weekday_counts.values, alpha=0.7)
        axes[1, 0].set_title('星期分布')
        axes[1, 0].set_xlabel('星期')
        axes[1, 0].set_ylabel('新闻数量')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 新闻源分布
        source_counts = self.news_data['source'].value_counts()
        axes[1, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('新闻源分布')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = output_path / f"news_statistics_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 图表已保存: {chart_file}")
        
        plt.show()
    
    def generate_report(self, output_dir: str = "data/results"):
        """生成完整统计报告"""
        if not self.load_latest_news_data():
            return False
        
        print("🔍 开始新闻数据统计分析...")
        
        # 执行各项统计
        daily_counts = self.daily_news_statistics()
        hourly_counts = self.hourly_distribution()
        weekday_counts = self.weekday_distribution()
        source_counts = self.source_distribution()
        
        # 创建可视化
        try:
            self.create_visualizations(output_dir)
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
        
        # 保存统计数据
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存每日统计
        daily_file = output_path / f"daily_news_stats_{timestamp}.csv"
        daily_counts.to_csv(daily_file, index=False)
        print(f"💾 每日统计已保存: {daily_file}")
        
        print("\n✅ 统计分析完成！")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻数据统计分析工具")
    parser.add_argument("--data-dir", default="data/raw/news", help="新闻数据目录")
    parser.add_argument("--output-dir", default="data/results", help="输出目录")
    parser.add_argument("--no-chart", action="store_true", help="不生成图表")
    
    args = parser.parse_args()
    
    try:
        analyzer = NewsStatistics(args.data_dir)
        
        if not analyzer.load_latest_news_data():
            return 1
        
        # 执行统计分析
        analyzer.daily_news_statistics()
        analyzer.hourly_distribution()
        analyzer.weekday_distribution()
        analyzer.source_distribution()
        
        # 生成图表
        if not args.no_chart:
            try:
                analyzer.create_visualizations(args.output_dir)
            except Exception as e:
                print(f"⚠️ 图表生成失败: {e}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 统计分析失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 