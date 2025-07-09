#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的新闻情感因子策略
确保买卖信号一一对应，避免无效信号
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from typing import Tuple, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root.parent.parent))

try:
    import qlib
    from qlib.workflow import R
    QLIB_AVAILABLE = True
except ImportError as e:
    print(f"Qlib导入失败: {e}")
    QLIB_AVAILABLE = False


class ImprovedNewsSentimentStrategy:
    """改进的新闻情感因子策略"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化策略"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # 策略参数
        self.buy_threshold = 0.05    # 买入阈值
        self.sell_threshold = -0.05  # 卖出阈值
        self.max_holding_days = 10   # 最大持仓天数
        self.stop_loss = -0.05       # 止损阈值
        self.take_profit = 0.08      # 止盈阈值
        
        if QLIB_AVAILABLE:
            self._init_qlib()
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        config_file = project_root / config_path
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # 返回默认配置
            return {
                'evaluation': {
                    'backtest': {
                        'start_date': '2023-01-01',
                        'end_date': '2024-01-01',
                        'initial_capital': 100000
                    }
                }
            }
    
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
            provider_uri = "~/.qlib/qlib_data/crypto_data"
            qlib.init(provider_uri=provider_uri, region='cn')
            self.logger.info("Qlib初始化成功")
        except Exception as e:
            self.logger.warning(f"Qlib初始化失败: {e}")
    
    def generate_improved_signals(self, sentiment_scores: pd.Series, 
                                 prices: pd.Series = None) -> Tuple[pd.Series, pd.DataFrame]:
        """
        生成改进的交易信号，确保买卖一一对应
        
        Args:
            sentiment_scores: 情感得分序列
            prices: 价格序列（用于止盈止损）
            
        Returns:
            signals: 交易信号序列 (1=买入, -1=卖出, 0=持有)
            trade_log: 详细交易记录
        """
        signals = pd.Series(0, index=sentiment_scores.index, name='signal')
        trade_log = []
        
        # 策略状态
        position = 0  # 0=空仓, 1=持仓
        entry_date = None
        entry_price = None
        holding_days = 0
        
        for i, (date, sentiment) in enumerate(sentiment_scores.items()):
            current_price = prices.iloc[i] if prices is not None else 100 * (1 + sentiment * 0.1)
            
            # 更新持仓天数
            if position == 1:
                holding_days += 1
            
            # 空仓状态：寻找买入机会
            if position == 0:
                if sentiment > self.buy_threshold:
                    # 产生买入信号
                    signals[date] = 1
                    position = 1
                    entry_date = date
                    entry_price = current_price
                    holding_days = 0
                    
                    trade_log.append({
                        'date': date,
                        'action': 'BUY',
                        'price': current_price,
                        'sentiment': sentiment,
                        'reason': f'情感得分{sentiment:.4f} > 买入阈值{self.buy_threshold}'
                    })
                    
                    self.logger.debug(f"🟢 {date.strftime('%Y-%m-%d')} 买入信号，情感得分: {sentiment:.4f}")
            
            # 持仓状态：寻找卖出机会
            elif position == 1:
                current_return = (current_price - entry_price) / entry_price
                sell_reason = None
                
                # 检查各种卖出条件
                if sentiment < self.sell_threshold:
                    sell_reason = f'情感得分{sentiment:.4f} < 卖出阈值{self.sell_threshold}'
                elif holding_days >= self.max_holding_days:
                    sell_reason = f'持仓{holding_days}天 >= 最大持仓{self.max_holding_days}天'
                elif current_return <= self.stop_loss:
                    sell_reason = f'止损：当前收益{current_return:.2%} <= {self.stop_loss:.2%}'
                elif current_return >= self.take_profit:
                    sell_reason = f'止盈：当前收益{current_return:.2%} >= {self.take_profit:.2%}'
                
                if sell_reason:
                    # 产生卖出信号
                    signals[date] = -1
                    position = 0
                    
                    trade_log.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'sentiment': sentiment,
                        'return': current_return,
                        'holding_days': holding_days,
                        'reason': sell_reason
                    })
                    
                    self.logger.debug(f"🔴 {date.strftime('%Y-%m-%d')} 卖出信号，收益: {current_return:.2%}")
                    
                    # 重置状态
                    entry_date = None
                    entry_price = None
                    holding_days = 0
        
        # 如果最后仍持仓，强制平仓
        if position == 1:
            last_date = sentiment_scores.index[-1]
            last_price = prices.iloc[-1] if prices is not None else current_price
            final_return = (last_price - entry_price) / entry_price
            
            signals[last_date] = -1
            trade_log.append({
                'date': last_date,
                'action': 'SELL',
                'price': last_price,
                'sentiment': sentiment_scores.iloc[-1],
                'return': final_return,
                'holding_days': holding_days,
                'reason': '回测结束强制平仓'
            })
        
        trade_log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        
        return signals, trade_log_df
    
    def calculate_strategy_returns(self, signals: pd.Series, market_returns: pd.Series) -> pd.Series:
        """
        根据信号计算策略收益率
        
        Args:
            signals: 交易信号
            market_returns: 市场收益率
            
        Returns:
            strategy_returns: 策略收益率
        """
        strategy_returns = pd.Series(0.0, index=signals.index)
        position = 0
        
        for date in signals.index:
            signal = signals[date]
            market_ret = market_returns[date]
            
            if signal == 1:  # 买入
                position = 1
                # 买入当天不产生收益
                strategy_returns[date] = 0
            elif signal == -1:  # 卖出
                if position == 1:
                    # 卖出当天获得市场收益
                    strategy_returns[date] = market_ret * 0.8  # 80%的市场收益捕获
                position = 0
            else:  # 持有
                if position == 1:
                    # 持仓期间获得市场收益
                    strategy_returns[date] = market_ret * 0.8
                else:
                    # 空仓期间无收益
                    strategy_returns[date] = 0
        
        return strategy_returns
    
    def run_improved_backtest(self, start_date: str = '2023-01-01', 
                             end_date: str = '2024-01-01', 
                             initial_capital: float = 100000) -> pd.DataFrame:
        """运行改进的回测"""
        self.logger.info("🚀 开始运行改进的新闻情感策略回测...")
        
        # 创建日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成模拟数据
        np.random.seed(42)
        sentiment_scores = pd.Series(
            np.random.normal(0, 0.1, len(date_range)), 
            index=date_range, 
            name='sentiment_score'
        )
        
        # 生成价格数据
        market_returns = pd.Series(
            np.random.normal(0.0005, 0.02, len(date_range)),
            index=date_range,
            name='market_return'
        )
        
        # 计算价格序列
        initial_price = 45000.0
        prices = pd.Series(
            initial_price * (1 + market_returns).cumprod(),
            index=date_range,
            name='price'
        )
        
        # 生成改进的交易信号
        self.logger.info("📊 生成交易信号...")
        signals, trade_log = self.generate_improved_signals(sentiment_scores, prices)
        
        # 计算策略收益率
        strategy_returns = self.calculate_strategy_returns(signals, market_returns)
        
        # 计算组合净值
        cumulative_returns = (1 + strategy_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'signal': signals,
            'market_return': market_returns,
            'strategy_return': strategy_returns,
            'cumulative_return': cumulative_returns - 1,
            'portfolio_value': portfolio_value,
            'price': prices
        })
        
        # 统计信号分布
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        self.logger.info(f"✅ 策略回测完成！")
        self.logger.info(f"📊 信号统计：买入信号 {buy_signals} 个，卖出信号 {sell_signals} 个")
        self.logger.info(f"🎯 信号匹配：{'✅ 完全匹配' if buy_signals == sell_signals else '❌ 不匹配'}")
        
        # 保存交易记录
        if not trade_log.empty:
            results_dir = project_root / "data" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            trade_log_file = results_dir / f"trade_log_{timestamp}.csv"
            trade_log.to_csv(trade_log_file, index=False)
            self.logger.info(f"📋 交易记录已保存至: {trade_log_file}")
        
        return results, trade_log
    
    def analyze_strategy_performance(self, results: pd.DataFrame, 
                                   trade_log: pd.DataFrame) -> Dict:
        """分析策略表现"""
        strategy_returns = results['strategy_return']
        portfolio_values = results['portfolio_value']
        signals = results['signal']
        
        # 基本指标
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(results)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # 交易统计
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        # 交易分析
        if not trade_log.empty:
            trade_pairs = []
            buys = trade_log[trade_log['action'] == 'BUY']
            sells = trade_log[trade_log['action'] == 'SELL']
            
            for i in range(min(len(buys), len(sells))):
                trade_pairs.append({
                    'entry_date': buys.iloc[i]['date'],
                    'exit_date': sells.iloc[i]['date'],
                    'return': sells.iloc[i]['return'],
                    'holding_days': sells.iloc[i]['holding_days']
                })
            
            if trade_pairs:
                trade_pairs_df = pd.DataFrame(trade_pairs)
                win_rate = (trade_pairs_df['return'] > 0).mean()
                avg_return = trade_pairs_df['return'].mean()
                avg_holding_days = trade_pairs_df['holding_days'].mean()
                best_trade = trade_pairs_df['return'].max()
                worst_trade = trade_pairs_df['return'].min()
            else:
                win_rate = avg_return = avg_holding_days = best_trade = worst_trade = 0
        else:
            win_rate = avg_return = avg_holding_days = best_trade = worst_trade = 0
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_match': buy_signals == sell_signals,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'avg_holding_days': avg_holding_days,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'final_portfolio_value': portfolio_values.iloc[-1]
        }
        
        return performance
    
    def print_performance_report(self, performance: Dict):
        """打印策略表现报告"""
        print("\n" + "="*60)
        print("🎯 改进的新闻情感因子策略回测报告")
        print("="*60)
        
        print("\n📈 策略表现")
        print(f"• 总收益率: {performance['total_return']:.2%}")
        print(f"• 年化收益率: {performance['annualized_return']:.2%}")
        print(f"• 年化波动率: {performance['volatility']:.2%}")
        print(f"• 夏普比率: {performance['sharpe_ratio']:.3f}")
        print(f"• 最大回撤: {performance['max_drawdown']:.2%}")
        print(f"• 最终净值: ${performance['final_portfolio_value']:,.2f}")
        
        print("\n🔄 交易信号统计")
        print(f"• 买入信号数量: {performance['buy_signals']}")
        print(f"• 卖出信号数量: {performance['sell_signals']}")
        print(f"• 信号匹配状态: {'✅ 完全匹配' if performance['signal_match'] else '❌ 不匹配'}")
        
        print("\n📊 交易表现")
        print(f"• 胜率: {performance['win_rate']:.2%}")
        print(f"• 平均单笔收益: {performance['avg_return_per_trade']:.2%}")
        print(f"• 平均持仓天数: {performance['avg_holding_days']:.1f}天")
        print(f"• 最佳交易: {performance['best_trade']:.2%}")
        print(f"• 最差交易: {performance['worst_trade']:.2%}")
        
        print("\n💡 策略评估")
        if performance['signal_match']:
            print("✅ 买卖信号完全匹配，避免了无效信号")
        else:
            print("❌ 买卖信号不匹配，策略逻辑需要调整")
        
        if performance['win_rate'] > 0.5:
            print("✅ 胜率超过50%，策略具有预测能力")
        else:
            print("⚠️ 胜率低于50%，需要优化信号质量")
        
        if performance['total_return'] > 0:
            print("✅ 策略产生正收益")
        else:
            print("❌ 策略产生负收益，需要重新评估")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="改进的新闻情感因子策略")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--start-date", default="2023-01-01", help="回测开始日期")
    parser.add_argument("--end-date", default="2024-01-01", help="回测结束日期")
    parser.add_argument("--initial-capital", type=float, default=100000, help="初始资金")
    parser.add_argument("--save-results", action="store_true", help="保存回测结果")
    
    args = parser.parse_args()
    
    try:
        # 初始化策略
        strategy = ImprovedNewsSentimentStrategy(args.config)
        
        # 运行回测
        results, trade_log = strategy.run_improved_backtest(
            args.start_date, 
            args.end_date, 
            args.initial_capital
        )
        
        # 分析表现
        performance = strategy.analyze_strategy_performance(results, trade_log)
        
        # 打印报告
        strategy.print_performance_report(performance)
        
        # 保存结果
        if args.save_results:
            results_dir = project_root / "data" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"improved_backtest_results_{timestamp}.csv"
            results.to_csv(results_file)
            
            print(f"\n📁 回测结果已保存至: {results_file}")
        
        return results, trade_log, performance
        
    except Exception as e:
        print(f"❌ 策略回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    results, trade_log, performance = main() 