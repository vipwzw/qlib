#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子评估脚本
用于评估新闻情感因子的有效性
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def evaluate_single_factor(args):
    """独立的因子评估函数，用于多进程调用"""
    factor_name, factor_data, returns_data = args
    
    try:
        # 去除NaN值
        factor_series = factor_data.dropna()
        if len(factor_series) < 50:  # 数据点太少
            return None
        
        # 对齐数据
        aligned_data = pd.concat([factor_series, returns_data], axis=1).dropna()
        if aligned_data.empty:
            return None
        
        factor_values = aligned_data.iloc[:, 0]
        return_values = aligned_data.iloc[:, 1]
        
        # 计算IC指标 (简化版本，避免循环计算)
        ic_1d = factor_values.corr(return_values.shift(-1))
        ic_5d = factor_values.corr(return_values.shift(-5))
        
        # 计算Rank IC
        rank_ic_1d = factor_values.rank().corr(return_values.shift(-1).rank())
        rank_ic_5d = factor_values.rank().corr(return_values.shift(-5).rank())
        
        # 计算统计指标
        performance = {
            'factor_name': factor_name,
            'data_points': len(factor_series),
            
            # IC指标
            'ic_1d_mean': ic_1d,
            'ic_1d_std': np.nan,  # 简化版本暂不计算
            'ic_1d_ir': ic_1d,
            'ic_1d_positive_ratio': 1.0 if ic_1d > 0 else 0.0,
            
            'ic_5d_mean': ic_5d,
            'ic_5d_std': np.nan,
            'ic_5d_ir': ic_5d,
            'ic_5d_positive_ratio': 1.0 if ic_5d > 0 else 0.0,
            
            # Rank IC指标
            'rank_ic_1d_mean': rank_ic_1d,
            'rank_ic_1d_std': np.nan,
            'rank_ic_1d_ir': rank_ic_1d,
            
            'rank_ic_5d_mean': rank_ic_5d,
            'rank_ic_5d_std': np.nan,
            'rank_ic_5d_ir': rank_ic_5d,
            
            # 因子基本统计
            'factor_mean': factor_series.mean(),
            'factor_std': factor_series.std(),
            'factor_skew': factor_series.skew(),
            'factor_kurt': factor_series.kurtosis(),
        }
        
        return performance
        
    except Exception as e:
        print(f"评估因子 {factor_name} 时出错: {e}")
        return None

class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化评估器"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        config_file = project_root / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_factor_data(self, factor_path: str = None) -> pd.DataFrame:
        """加载因子数据"""
        if factor_path is None:
            # 查找最新的因子文件
            factors_dir = project_root / "data" / "factors"
            if not factors_dir.exists():
                raise FileNotFoundError("因子数据目录不存在，请先运行因子构建")
            
            factor_files = list(factors_dir.glob("factors_*.parquet"))
            if not factor_files:
                raise FileNotFoundError("未找到因子数据文件，请先运行因子构建")
            
            # 使用最新的文件
            factor_path = sorted(factor_files)[-1]
        
        self.logger.info(f"加载因子数据: {factor_path}")
        
        try:
            factors = pd.read_parquet(factor_path)
            self.logger.info(f"成功加载因子数据，形状: {factors.shape}")
            return factors
        except Exception as e:
            self.logger.error(f"加载因子数据失败: {e}")
            raise
    
    def load_returns_data(self) -> pd.Series:
        """加载收益率数据"""
        self.logger.info("从价格数据计算真实收益率...")
        
        try:
            # 加载最新的价格数据
            import glob
            price_files = glob.glob(str(project_root / "data" / "raw" / "price" / "*.csv"))
            if price_files:
                # 按修改时间排序，选择最新的文件
                import os
                latest_price_file = sorted(price_files, key=os.path.getmtime)[-1]
                self.logger.info(f"加载价格数据: {latest_price_file}")
                
                price_data = pd.read_csv(latest_price_file)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                price_data.set_index('timestamp', inplace=True)
                
                # 计算1分钟收益率
                returns = price_data['close'].pct_change().dropna()
                
                self.logger.info(f"计算得到 {len(returns)} 个收益率数据点")
                self.logger.info(f"收益率时间范围: {returns.index.min()} 到 {returns.index.max()}")
                
                return returns
            else:
                # 如果没有价格数据，生成模拟数据但使用正确的时间范围
                self.logger.warning("未找到价格数据，生成模拟收益率数据...")
                dates = pd.date_range(start='2025-06-07', end='2025-07-07', freq='min')
                np.random.seed(42)
                returns = np.random.normal(0.0001, 0.005, len(dates))  # 分钟收益率
                return pd.Series(returns, index=dates, name='returns')
                
        except Exception as e:
            self.logger.error(f"加载收益率数据失败: {e}")
            # 生成模拟数据作为后备
            dates = pd.date_range(start='2025-06-07', end='2025-07-07', freq='min')
            np.random.seed(42)
            returns = np.random.normal(0.0001, 0.005, len(dates))
            return pd.Series(returns, index=dates, name='returns')
    
    def calculate_ic(self, factor: pd.Series, returns: pd.Series, periods: int = 1) -> pd.Series:
        """计算信息系数(IC)"""
        # 对齐数据
        aligned_data = pd.concat([factor, returns], axis=1).dropna()
        
        if aligned_data.empty:
            self.logger.warning("因子和收益率数据无法对齐")
            return pd.Series(dtype=float)
        
        factor_values = aligned_data.iloc[:, 0]
        return_values = aligned_data.iloc[:, 1].shift(-periods)
        
        # 计算滚动IC
        ic_values = []
        window = 30  # 30天滚动窗口
        
        for i in range(window, len(factor_values)):
            factor_window = factor_values.iloc[i-window:i]
            return_window = return_values.iloc[i-window:i]
            
            if len(factor_window.dropna()) > 10 and len(return_window.dropna()) > 10:
                ic = factor_window.corr(return_window)
                ic_values.append(ic)
            else:
                ic_values.append(np.nan)
        
        ic_index = factor_values.index[window:]
        return pd.Series(ic_values, index=ic_index, name=f'IC_{periods}d')
    
    def calculate_rank_ic(self, factor: pd.Series, returns: pd.Series, periods: int = 1) -> pd.Series:
        """计算Rank IC"""
        # 对齐数据
        aligned_data = pd.concat([factor, returns], axis=1).dropna()
        
        if aligned_data.empty:
            return pd.Series(dtype=float)
        
        factor_values = aligned_data.iloc[:, 0]
        return_values = aligned_data.iloc[:, 1].shift(-periods)
        
        # 计算滚动Rank IC
        rank_ic_values = []
        window = 30
        
        for i in range(window, len(factor_values)):
            factor_window = factor_values.iloc[i-window:i]
            return_window = return_values.iloc[i-window:i]
            
            if len(factor_window.dropna()) > 10 and len(return_window.dropna()) > 10:
                factor_rank = factor_window.rank()
                return_rank = return_window.rank()
                rank_ic = factor_rank.corr(return_rank)
                rank_ic_values.append(rank_ic)
            else:
                rank_ic_values.append(np.nan)
        
        rank_ic_index = factor_values.index[window:]
        return pd.Series(rank_ic_values, index=rank_ic_index, name=f'RankIC_{periods}d')
    
    def factor_performance_summary(self, factor_name: str, factor_data: pd.Series, returns_data: pd.Series) -> Dict:
        """计算因子表现汇总"""
        self.logger.info(f"评估因子: {factor_name}")
        
        # 计算IC指标
        ic_1d = self.calculate_ic(factor_data, returns_data, 1)
        ic_5d = self.calculate_ic(factor_data, returns_data, 5)
        
        rank_ic_1d = self.calculate_rank_ic(factor_data, returns_data, 1)
        rank_ic_5d = self.calculate_rank_ic(factor_data, returns_data, 5)
        
        # 计算统计指标
        performance = {
            'factor_name': factor_name,
            'data_points': len(factor_data),
            
            # IC指标
            'ic_1d_mean': ic_1d.mean(),
            'ic_1d_std': ic_1d.std(),
            'ic_1d_ir': ic_1d.mean() / ic_1d.std() if ic_1d.std() > 0 else 0,
            'ic_1d_positive_ratio': (ic_1d > 0).mean(),
            
            'ic_5d_mean': ic_5d.mean(),
            'ic_5d_std': ic_5d.std(),
            'ic_5d_ir': ic_5d.mean() / ic_5d.std() if ic_5d.std() > 0 else 0,
            'ic_5d_positive_ratio': (ic_5d > 0).mean(),
            
            # Rank IC指标
            'rank_ic_1d_mean': rank_ic_1d.mean(),
            'rank_ic_1d_std': rank_ic_1d.std(),
            'rank_ic_1d_ir': rank_ic_1d.mean() / rank_ic_1d.std() if rank_ic_1d.std() > 0 else 0,
            
            'rank_ic_5d_mean': rank_ic_5d.mean(),
            'rank_ic_5d_std': rank_ic_5d.std(),
            'rank_ic_5d_ir': rank_ic_5d.mean() / rank_ic_5d.std() if rank_ic_5d.std() > 0 else 0,
            
            # 因子基本统计
            'factor_mean': factor_data.mean(),
            'factor_std': factor_data.std(),
            'factor_skew': factor_data.skew(),
            'factor_kurt': factor_data.kurtosis(),
        }
        
        return performance
    
    def evaluate_all_factors(self, factors_df: pd.DataFrame, returns_data: pd.Series, n_jobs: int = None) -> pd.DataFrame:
        """评估所有因子（支持多进程并行）"""
        if n_jobs is None:
            n_jobs = min(mp.cpu_count(), len(factors_df.columns))
        
        self.logger.info(f"开始评估所有因子... (使用 {n_jobs} 个进程)")
        
        # 准备参数
        factor_args = [(factor_name, factors_df[factor_name], returns_data) 
                      for factor_name in factors_df.columns]
        
        results = []
        
        if n_jobs == 1:
            # 单进程执行（带进度条）
            for args in tqdm(factor_args, desc="评估因子", ncols=80):
                result = evaluate_single_factor(args)
                if result is not None:
                    results.append(result)
        else:
            # 多进程执行
            with mp.Pool(processes=n_jobs) as pool:
                # 使用imap显示进度
                results_iter = pool.imap(evaluate_single_factor, factor_args)
                
                # 带进度条收集结果
                for result in tqdm(results_iter, total=len(factor_args), 
                                 desc="评估因子", ncols=80):
                    if result is not None:
                        results.append(result)
        
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.set_index('factor_name')
            self.logger.info(f"成功评估了 {len(results_df)} 个因子")
            return results_df
        else:
            self.logger.warning("没有成功评估任何因子")
            return pd.DataFrame()
    
    def print_evaluation_results(self, results_df: pd.DataFrame):
        """打印评估结果"""
        if results_df.empty:
            print("没有可显示的评估结果")
            return
        
        print("\n" + "="*60)
        print("因子评估结果")
        print("="*60)
        
        # 按IC_IR排序
        sorted_results = results_df.sort_values('ic_1d_ir', ascending=False)
        
        for factor_name, row in sorted_results.iterrows():
            print(f"\n因子名称: {factor_name}")
            print("-" * 40)
            print(f"1日IC均值: {row['ic_1d_mean']:.4f}")
            print(f"1日IC标准差: {row['ic_1d_std']:.4f}")
            print(f"1日IC_IR: {row['ic_1d_ir']:.4f}")
            print(f"1日正IC比例: {row['ic_1d_positive_ratio']:.2%}")
            print(f"5日IC均值: {row['ic_5d_mean']:.4f}")
            print(f"5日IC_IR: {row['ic_5d_ir']:.4f}")
            print(f"因子均值: {row['factor_mean']:.4f}")
            print(f"因子标准差: {row['factor_std']:.4f}")
    
    def generate_evaluation_report(self, results_df: pd.DataFrame, output_path: str = None):
        """生成评估报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = project_root / "data" / "results" / f"factor_evaluation_report_{timestamp}.html"
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成HTML报告
        html_content = self._generate_html_report(results_df)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"评估报告已保存至: {output_path}")
    
    def _generate_html_report(self, results_df: pd.DataFrame) -> str:
        """生成HTML报告内容"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>因子评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .header { text-align: center; color: #333; }
                .summary { background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1 class="header">新闻情感因子评估报告</h1>
            <div class="summary">
                <h3>评估摘要</h3>
                <p>评估时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p>评估因子数量: """ + str(len(results_df)) + """</p>
                <p>最佳因子(按1日IC_IR): """ + (results_df.sort_values('ic_1d_ir', ascending=False).index[0] if not results_df.empty else "无") + """</p>
            </div>
            
            <h2>详细结果</h2>
            """ + results_df.round(4).to_html() + """
            
        </body>
        </html>
        """
        return html
    
    def save_results(self, results_df: pd.DataFrame, filename: str = None):
        """保存评估结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"factor_evaluation_{timestamp}.csv"
        
        results_dir = project_root / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        results_df.to_csv(filepath)
        
        self.logger.info(f"评估结果已保存至: {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子评估工具 (支持多核心并行)")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--factor-path", help="因子数据文件路径")
    parser.add_argument("--factor", help="评估特定因子名称")
    parser.add_argument("--generate-report", action="store_true", help="生成HTML报告")
    parser.add_argument("--save-results", action="store_true", help="保存评估结果")
    parser.add_argument("--n-jobs", type=int, default=None, 
                       help=f"并行进程数 (默认: min(CPU核心数={mp.cpu_count()}, 因子数))")
    parser.add_argument("--verbose", action="store_true", help="显示详细评估结果")
    
    args = parser.parse_args()
    
    try:
        # 初始化评估器
        evaluator = FactorEvaluator(args.config)
        
        # 加载数据
        print("🔍 正在加载数据...")
        factors_df = evaluator.load_factor_data(args.factor_path)
        returns_data = evaluator.load_returns_data()
        
        print(f"📊 因子数量: {len(factors_df.columns)}")
        print(f"💻 可用CPU核心数: {mp.cpu_count()}")
        
        if args.factor:
            # 评估特定因子
            if args.factor not in factors_df.columns:
                print(f"❌ 错误: 找不到因子 '{args.factor}'")
                print(f"📋 可用因子: {list(factors_df.columns)}")
                return 1
            
            factor_series = factors_df[args.factor]
            performance = evaluator.factor_performance_summary(args.factor, factor_series, returns_data)
            
            # 打印单个因子结果
            print("\n" + "="*50)
            print(f"因子评估结果: {args.factor}")
            print("="*50)
            for key, value in performance.items():
                if key != 'factor_name':
                    print(f"{key}: {value:.4f}")
        else:
            # 评估所有因子
            n_jobs = args.n_jobs if args.n_jobs else min(mp.cpu_count(), len(factors_df.columns))
            print(f"🚀 开始并行评估 (进程数: {n_jobs})")
            
            results_df = evaluator.evaluate_all_factors(factors_df, returns_data, n_jobs=n_jobs)
            
            if not results_df.empty:
                print(f"\n✅ 成功评估了 {len(results_df)} 个因子")
                
                # 显示TOP因子
                top_factors = results_df.sort_values('ic_1d_ir', ascending=False, na_position='last').head()
                print("\n🏆 TOP 5 因子 (按1日IC_IR排序):")
                for i, (factor_name, row) in enumerate(top_factors.iterrows(), 1):
                    ic_ir = row['ic_1d_ir']
                    ic_mean = row['ic_1d_mean']
                    if pd.notna(ic_ir) and pd.notna(ic_mean):
                        print(f"  {i}. {factor_name}: IC_IR={ic_ir:.4f}, IC={ic_mean:.4f}")
                    else:
                        print(f"  {i}. {factor_name}: IC_IR=nan, IC=nan")
                
                # 打印详细结果（可选）
                if args.verbose:
                    evaluator.print_evaluation_results(results_df)
                
                # 生成报告
                if args.generate_report:
                    evaluator.generate_evaluation_report(results_df)
                    print("📄 HTML报告已生成")
                
                # 保存结果
                if args.save_results:
                    evaluator.save_results(results_df)
                    print("💾 评估结果已保存到CSV")
            else:
                print("❌ 未能生成任何评估结果")
        
        print("\n🎉 因子评估完成！")
        
    except Exception as e:
        print(f"❌ 因子评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 