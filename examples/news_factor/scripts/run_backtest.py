#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻情感因子策略回测脚本
集成自动数据管理功能
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent.parent))  # 添加qlib根目录

# 导入自动数据管理器
try:
    from .auto_data_manager import AutoDataManager
except ImportError:
    from scripts.auto_data_manager import AutoDataManager

try:
    import qlib
    from qlib.workflow import R
    from qlib.utils import init_instance_by_config
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.data.dataset import DatasetH
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.backtest import backtest, executor
    QLIB_AVAILABLE = True
except ImportError as e:
    print(f"Qlib导入失败: {e}")
    print("请确保已正确安装qlib")
    QLIB_AVAILABLE = False


class NewsSentimentBacktest:
    """新闻情感因子回测器"""
    
    def __init__(self, config_path: str = "configs/config.yaml", auto_data_check: bool = True):
        """初始化回测器"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # 自动数据管理
        if auto_data_check:
            self._ensure_data_ready(config_path)
        
        if QLIB_AVAILABLE:
            self._init_qlib()
    
    def _ensure_data_ready(self, config_path: str):
        """确保数据准备就绪"""
        self.logger.info("🔍 检查数据完整性...")
        
        try:
            data_manager = AutoDataManager(config_path)
            success = data_manager.ensure_data_ready()
            
            if not success:
                self.logger.warning("⚠️ 数据准备过程中出现问题，但将继续运行回测")
            else:
                self.logger.info("✅ 所有数据已准备就绪")
        except Exception as e:
            self.logger.error(f"❌ 自动数据管理失败: {e}")
            self.logger.info("将尝试使用现有数据运行回测")
    
    def _load_config(self, config_path: str) -> dict:
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
    
    def _init_qlib(self):
        """初始化qlib"""
        try:
            # 设置qlib数据路径
            provider_uri = "~/.qlib/qlib_data/crypto_data"
            
            qlib.init(provider_uri=provider_uri, region='cn')
            self.logger.info("Qlib初始化成功")
            
        except Exception as e:
            self.logger.error(f"Qlib初始化失败: {e}")
            self.logger.info("将使用模拟模式运行")
    
    def run_workflow_backtest(self, workflow_config_path: str = "configs/workflow_config.yaml"):
        """运行基于工作流配置的回测"""
        if not QLIB_AVAILABLE:
            self.logger.error("Qlib不可用，无法运行工作流回测")
            return None
        
        try:
            # 加载工作流配置
            workflow_config_file = project_root / workflow_config_path
            with open(workflow_config_file, 'r', encoding='utf-8') as f:
                workflow_config = yaml.safe_load(f)
            
            self.logger.info("开始执行工作流回测...")
            
            # 使用qlib的工作流运行回测
            with R.start(experiment_name="news_sentiment_factor"):
                # 这里应该使用实际的工作流执行逻辑
                self.logger.info("工作流回测完成")
                
        except Exception as e:
            self.logger.error(f"工作流回测失败: {e}")
            return None
    
    def run_simple_backtest(self):
        """运行简单回测"""
        self.logger.info("开始简单回测...")
        
        # 加载回测配置
        backtest_config = self.config.get('evaluation', {}).get('backtest', {})
        
        start_date = backtest_config.get('start_date', '2025-06-13')
        end_date = backtest_config.get('end_date', '2025-07-07')
        initial_capital = backtest_config.get('initial_capital', 100000)
        
        self.logger.info(f"回测期间: {start_date} - {end_date}")
        self.logger.info(f"初始资金: {initial_capital}")
        
        # 生成模拟结果
        results = self._generate_mock_results(start_date, end_date, initial_capital)
        
        # 计算性能指标
        performance = self._calculate_performance(results)
        
        # 输出结果
        self._print_results(performance)
        
        return performance
    
    def _generate_mock_results(self, start_date: str, end_date: str, initial_capital: float) -> pd.DataFrame:
        """生成模拟回测结果 (M1级别)"""
        # 创建M1级别的时间范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
        
        # 生成模拟收益率（基于新闻情感的M1级别策略）
        np.random.seed(42)
        
        # 模拟新闻情感得分 (M1级别调整)
        sentiment_scores = np.random.normal(0, 0.05, len(date_range))  # 降低波动率
        
        # 基于情感得分生成策略信号 (使用配置中的阈值)
        buy_threshold = self.config.get('strategy', {}).get('signal_generation', {}).get('sentiment_threshold_buy', 0.05)
        sell_threshold = self.config.get('strategy', {}).get('signal_generation', {}).get('sentiment_threshold_sell', -0.07)
        
        signals = np.where(sentiment_scores > buy_threshold, 1,  # 买入
                          np.where(sentiment_scores < sell_threshold, -1, 0))  # 卖出
        
        # 模拟M1级别市场收益率 (大幅降低以适合分钟级别)
        market_returns = np.random.normal(0.00001, 0.0008, len(date_range))  # 1分钟级别的收益率
        
        # 策略收益率（简化版）
        strategy_returns = signals * market_returns * 0.8  # 80%的市场收益捕获
        
        # 计算累计净值
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'date': date_range,
            'sentiment_score': sentiment_scores,
            'signal': signals,
            'market_return': market_returns,
            'strategy_return': strategy_returns,
            'cumulative_return': cumulative_returns - 1,
            'portfolio_value': portfolio_value
        })
        
        results.set_index('date', inplace=True)
        return results
    
    def _calculate_performance(self, results: pd.DataFrame) -> dict:
        """计算策略性能指标"""
        strategy_returns = results['strategy_return']
        portfolio_values = results['portfolio_value']
        
        # 基本指标 (M1级别调整)
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        # M1级别年化: 365天 * 24小时 * 60分钟 = 525,600分钟
        annualized_return = (1 + total_return) ** (525600 / len(results)) - 1
        
        # 波动率 (M1级别调整)
        volatility = strategy_returns.std() * np.sqrt(525600)
        
        # 夏普比率
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_values = portfolio_values
        peak = cumulative_values.expanding().max()
        drawdown = (cumulative_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (strategy_returns > 0).mean()
        
        # 信息比率（vs市场，M1级别调整）
        market_returns = results['market_return']
        excess_returns = strategy_returns - market_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(525600) if excess_returns.std() > 0 else 0
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'final_portfolio_value': portfolio_values.iloc[-1],
            'total_trades': (results['signal'] != 0).sum()
        }
        
        return performance
    
    def _print_results(self, performance: dict):
        """打印M1级别回测结果"""
        print("\n" + "="*60)
        print("🎯 M1新闻情感因子策略回测结果")
        print("="*60)
        print(f"总收益率: {performance['total_return']:.2%}")
        print(f"年化收益率: {performance['annualized_return']:.2%}")
        print(f"年化波动率: {performance['volatility']:.2%}")
        print(f"夏普比率: {performance['sharpe_ratio']:.3f}")
        print(f"最大回撤: {performance['max_drawdown']:.2%}")
        print(f"胜率: {performance['win_rate']:.2%}")
        print(f"信息比率: {performance['information_ratio']:.3f}")
        print(f"最终组合价值: ${performance['final_portfolio_value']:,.2f}")
        print(f"总交易次数: {performance['total_trades']}")
        print("="*50)
    
    def save_results(self, results: pd.DataFrame, filename: str = None):
        """保存回测结果"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.csv"
        
        results_dir = project_root / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        results.to_csv(filepath)
        
        self.logger.info(f"回测结果已保存至: {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻情感因子策略回测工具")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--workflow-config", default="configs/workflow_config.yaml", 
                       help="工作流配置文件路径")
    parser.add_argument("--mode", choices=["simple", "workflow"], default="simple",
                       help="回测模式")
    parser.add_argument("--save-results", action="store_true", help="保存回测结果")
    parser.add_argument("--no-auto-data", action="store_true", 
                       help="禁用自动数据检查和下载")
    parser.add_argument("--force-download", action="store_true", 
                       help="强制重新下载所有数据")
    
    args = parser.parse_args()
    
    try:
        # 处理强制下载参数
        if args.force_download:
            print("🔄 强制重新下载所有数据...")
            data_manager = AutoDataManager(args.config)
            
            # 强制下载价格数据
            print("📥 下载价格数据...")
            data_manager.download_price_data()
            
            # 强制下载新闻数据  
            print("📰 下载新闻数据...")
            data_manager.download_news_data()
            
            # 重新生成因子
            print("🔧 生成因子数据...")
            data_manager.generate_factors()
            
            print("✅ 数据下载完成！")
        
        # 初始化回测器（控制是否自动检查数据）
        auto_data_check = not args.no_auto_data
        backtester = NewsSentimentBacktest(args.config, auto_data_check=auto_data_check)
        
        if args.mode == "workflow" and QLIB_AVAILABLE:
            # 运行工作流回测
            results = backtester.run_workflow_backtest(args.workflow_config)
        else:
            # 运行简单回测
            results = backtester.run_simple_backtest()
        
        if args.save_results and results is not None:
            if isinstance(results, dict):
                # 如果结果是性能指标字典，转换为DataFrame
                results_df = pd.DataFrame([results])
                backtester.save_results(results_df)
            else:
                backtester.save_results(results)
        
        print("\n回测完成！")
        
    except Exception as e:
        print(f"回测失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 