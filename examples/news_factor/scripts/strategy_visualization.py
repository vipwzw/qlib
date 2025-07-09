#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略可视化工具
在K线图上显示买卖信号，评估交易决策的合理性
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

# 导入回测器
try:
    from run_backtest import NewsSentimentBacktest
except ImportError:
    print("⚠️ 无法导入回测器，将使用模拟数据")


class StrategyVisualizer:
    """策略可视化器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化可视化器"""
        self.config_path = config_path
        self.trade_log = pd.DataFrame()  # 初始化交易记录
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_ohlc_data(self, dates: pd.DatetimeIndex, returns: pd.Series, 
                          initial_price: float = 45000.0) -> pd.DataFrame:
        """基于收益率生成OHLC数据"""
        # 计算价格序列
        prices = initial_price * (1 + returns).cumprod()
        
        # 为每日生成OHLC数据
        ohlc_data = []
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 生成日内波动
            np.random.seed(i + 42)  # 确保可重现
            
            # 开盘价：前一日收盘价（第一天为初始价格）
            if i == 0:
                open_price = initial_price
            else:
                open_price = ohlc_data[i-1]['close']
            
            # 收盘价：根据收益率计算
            close_price = price
            
            # 生成高低价（在开盘和收盘价基础上加随机波动）
            daily_volatility = 0.01  # 1%的日内波动
            high_low_range = abs(close_price - open_price) + close_price * daily_volatility
            
            high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range * 0.5)
            low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range * 0.5)
            
            # 生成成交量
            volume = np.random.exponential(1000000)  # 指数分布的成交量
            
            ohlc_data.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(ohlc_data)
    
    def get_backtest_data(self) -> pd.DataFrame:
        """获取回测数据"""
        # 优先使用优化的策略
        try:
            from optimized_strategy import OptimizedNewsSentimentStrategy
            
            self.logger.info("🎯 使用优化的策略生成数据...")
            strategy = OptimizedNewsSentimentStrategy(self.config_path)
            detailed_data, trade_log = strategy.run_optimized_backtest('2023-01-01', '2024-01-01', 100000)
            
            self.logger.info(f"✅ 获取优化策略数据，共 {len(detailed_data)} 个交易日")
            self.logger.info(f"📊 交易记录：{len(trade_log)} 笔交易")
            
            # 保存交易记录供后续分析
            self.trade_log = trade_log
            return detailed_data
            
        except Exception as e:
            self.logger.warning(f"无法运行优化策略: {e}")
            
            # 备选：使用改进策略
            try:
                from improved_strategy import ImprovedNewsSentimentStrategy
                strategy = ImprovedNewsSentimentStrategy(self.config_path)
                detailed_data, trade_log = strategy.run_improved_backtest('2023-01-01', '2024-01-01', 100000)
                self.logger.info(f"✅ 使用改进策略数据，共 {len(detailed_data)} 个交易日")
                self.trade_log = trade_log
                return detailed_data
                
            except Exception as e2:
                self.logger.warning(f"无法运行改进策略: {e2}")
                
                # 最后备选：使用原始回测器
                try:
                    from run_backtest import NewsSentimentBacktest
                    backtester = NewsSentimentBacktest(self.config_path)
                    detailed_data = backtester._generate_mock_results('2023-01-01', '2024-01-01', 100000)
                    self.logger.info(f"✅ 使用原始策略数据，共 {len(detailed_data)} 个交易日")
                    self.trade_log = pd.DataFrame()  # 空的交易记录
                    return detailed_data
                    
                except Exception as e3:
                    self.logger.warning(f"无法运行原始回测: {e3}")
                    # 最后备选：生成模拟数据
                    self.trade_log = pd.DataFrame()
                    return self._generate_demo_data()
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """生成演示数据"""
        self.logger.info("生成演示数据...")
        
        # 创建日期范围
        date_range = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # 生成模拟数据
        np.random.seed(42)
        sentiment_scores = np.random.normal(0, 0.1, len(date_range))
        signals = np.where(sentiment_scores > 0.05, 1,
                          np.where(sentiment_scores < -0.05, -1, 0))
        market_returns = np.random.normal(0.0005, 0.02, len(date_range))
        strategy_returns = signals * market_returns * 0.8
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        results = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'signal': signals,
            'market_return': market_returns,
            'strategy_return': strategy_returns,
            'cumulative_return': cumulative_returns - 1,
            'portfolio_value': 100000 * cumulative_returns
        }, index=date_range)
        
        return results
    
    def create_strategy_chart(self, backtest_data: pd.DataFrame, 
                            show_volume: bool = True,
                            show_sentiment: bool = True) -> go.Figure:
        """创建策略可视化图表"""
        
        # 生成OHLC数据
        ohlc_data = self.generate_ohlc_data(
            backtest_data.index, 
            backtest_data['strategy_return']
        )
        
        # 确定子图数量
        subplot_count = 2  # K线图 + 净值曲线
        if show_volume:
            subplot_count += 1
        if show_sentiment:
            subplot_count += 1
        
        # 创建子图
        subplot_titles = ['价格走势与交易信号', '策略净值']
        if show_volume:
            subplot_titles.append('成交量')
        if show_sentiment:
            subplot_titles.append('新闻情感得分')
        
        fig = make_subplots(
            rows=subplot_count, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=subplot_titles,
            row_heights=[0.5] + [0.5/(subplot_count-1)]*(subplot_count-1)
        )
        
        # 1. 添加K线图
        fig.add_trace(go.Candlestick(
            x=ohlc_data['date'],
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name='BTC/USDT',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ), row=1, col=1)
        
        # 2. 添加买卖信号
        buy_signals = backtest_data[backtest_data['signal'] == 1]
        sell_signals = backtest_data[backtest_data['signal'] == -1]
        
        if len(buy_signals) > 0:
            buy_prices = []
            for date in buy_signals.index:
                # 找到对应的价格
                ohlc_row = ohlc_data[ohlc_data['date'].dt.date == date.date()]
                if not ohlc_row.empty:
                    buy_prices.append(ohlc_row.iloc[0]['low'] * 0.995)  # 在低点下方显示
                else:
                    buy_prices.append(None)
            
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='买入信号',
                hovertemplate='<b>买入信号</b><br>' +
                             '日期: %{x}<br>' +
                             '价格: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ), row=1, col=1)
        
        if len(sell_signals) > 0:
            sell_prices = []
            for date in sell_signals.index:
                ohlc_row = ohlc_data[ohlc_data['date'].dt.date == date.date()]
                if not ohlc_row.empty:
                    sell_prices.append(ohlc_row.iloc[0]['high'] * 1.005)  # 在高点上方显示
                else:
                    sell_prices.append(None)
            
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_prices,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='卖出信号',
                hovertemplate='<b>卖出信号</b><br>' +
                             '日期: %{x}<br>' +
                             '价格: $%{y:,.2f}<br>' +
                             '<extra></extra>'
            ), row=1, col=1)
        
        # 3. 添加策略净值曲线
        fig.add_trace(go.Scatter(
            x=backtest_data.index,
            y=backtest_data['portfolio_value'],
            mode='lines',
            name='策略净值',
            line=dict(color='blue', width=2),
            hovertemplate='<b>策略净值</b><br>' +
                         '日期: %{x}<br>' +
                         '净值: $%{y:,.2f}<br>' +
                         '<extra></extra>'
        ), row=2, col=1)
        
        # 添加基准净值线（初始值）
        initial_value = backtest_data['portfolio_value'].iloc[0]
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="gray",
            annotation_text="初始净值",
            row=2, col=1
        )
        
        current_row = 3
        
        # 4. 添加成交量（可选）
        if show_volume:
            fig.add_trace(go.Bar(
                x=ohlc_data['date'],
                y=ohlc_data['volume'],
                name='成交量',
                marker_color='lightblue',
                opacity=0.7
            ), row=current_row, col=1)
            current_row += 1
        
        # 5. 添加情感得分（可选）
        if show_sentiment:
            fig.add_trace(go.Scatter(
                x=backtest_data.index,
                y=backtest_data['sentiment_score'],
                mode='lines',
                name='情感得分',
                line=dict(color='purple', width=1),
                hovertemplate='<b>新闻情感得分</b><br>' +
                             '日期: %{x}<br>' +
                             '得分: %{y:.4f}<br>' +
                             '<extra></extra>'
            ), row=current_row, col=1)
            
            # 添加情感阈值线
            fig.add_hline(y=0.05, line_dash="dot", line_color="green", 
                         annotation_text="买入阈值", row=current_row, col=1)
            fig.add_hline(y=-0.05, line_dash="dot", line_color="red", 
                         annotation_text="卖出阈值", row=current_row, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                         annotation_text="中性", row=current_row, col=1)
        
        # 更新布局
        fig.update_layout(
            title={
                'text': '新闻情感因子策略可视化分析',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_rangeslider_visible=False,
            height=800 if subplot_count <= 3 else 1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 更新x轴
        fig.update_xaxes(title_text="日期", row=subplot_count, col=1)
        
        # 更新y轴
        fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="净值 (USD)", row=2, col=1)
        
        if show_volume:
            fig.update_yaxes(title_text="成交量", row=3, col=1)
        
        if show_sentiment:
            fig.update_yaxes(title_text="情感得分", row=current_row, col=1)
        
        return fig
    
    def analyze_trade_performance(self, backtest_data: pd.DataFrame) -> Dict:
        """分析交易表现"""
        signals = backtest_data['signal']
        returns = backtest_data['strategy_return']
        prices = backtest_data['portfolio_value']
        
        # 找到所有交易点
        trade_points = backtest_data[signals != 0].copy()
        
        # 计算交易统计
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        
        # 计算持仓期收益
        trade_analysis = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, row in trade_points.iterrows():
            if row['signal'] == 1 and position == 0:  # 买入
                position = 1
                entry_price = row['portfolio_value']
                entry_date = date
            elif row['signal'] == -1 and position == 1:  # 卖出
                exit_price = row['portfolio_value']
                trade_return = (exit_price - entry_price) / entry_price
                holding_days = (date - entry_date).days
                
                trade_analysis.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'holding_days': holding_days
                })
                
                position = 0
                entry_price = 0
                entry_date = None
        
        # 计算统计指标
        if trade_analysis:
            trade_df = pd.DataFrame(trade_analysis)
            avg_return = trade_df['return'].mean()
            win_rate = (trade_df['return'] > 0).mean()
            avg_holding_days = trade_df['holding_days'].mean()
            best_trade = trade_df['return'].max()
            worst_trade = trade_df['return'].min()
        else:
            avg_return = win_rate = avg_holding_days = best_trade = worst_trade = 0
        
        analysis = {
            'total_trades': len(trade_analysis),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'avg_holding_days': avg_holding_days,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'trade_details': trade_analysis
        }
        
        return analysis
    
    def generate_report(self, backtest_data: pd.DataFrame, trade_analysis: Dict) -> str:
        """生成分析报告"""
        total_return = (backtest_data['portfolio_value'].iloc[-1] / backtest_data['portfolio_value'].iloc[0] - 1)
        max_drawdown = ((backtest_data['portfolio_value'] / backtest_data['portfolio_value'].cummax()) - 1).min()
        
        # 检查是否为改进策略（买卖信号匹配）
        signals_matched = trade_analysis['buy_signals'] == trade_analysis['sell_signals']
        strategy_type = "改进的新闻情感因子策略" if signals_matched else "基础新闻情感因子策略"
        
        report = f"""
📊 {strategy_type}分析报告
{'='*50}

📈 策略表现
• 总收益率: {total_return:.2%}
• 最大回撤: {max_drawdown:.2%}
• 最终净值: ${backtest_data['portfolio_value'].iloc[-1]:,.2f}

🔄 交易信号统计
• 买入信号: {trade_analysis['buy_signals']}
• 卖出信号: {trade_analysis['sell_signals']}
• 信号匹配: {'✅ 完全匹配' if signals_matched else '❌ 不匹配'}
• 总交易次数: {trade_analysis['total_trades']}

📊 交易表现
• 胜率: {trade_analysis['win_rate']:.2%}
• 平均单笔收益: {trade_analysis['avg_return_per_trade']:.2%}
• 平均持仓天数: {trade_analysis['avg_holding_days']:.1f}天

🎯 极值表现
• 最佳交易: {trade_analysis['best_trade']:.2%}
• 最差交易: {trade_analysis['worst_trade']:.2%}

💡 策略评估
"""
        
        # 添加信号匹配评估
        if signals_matched:
            report += "✅ 买卖信号完全匹配，避免了无效信号\n"
        else:
            report += "❌ 买卖信号不匹配，存在无效信号问题\n"
        
        # 添加策略评估
        if trade_analysis['win_rate'] > 0.5:
            report += "✅ 胜率超过50%，策略有一定的预测能力\n"
        else:
            report += "⚠️ 胜率低于50%，需要优化信号质量\n"
        
        if total_return > 0:
            report += "✅ 策略产生正收益\n"
        else:
            report += "❌ 策略产生负收益，需要重新评估\n"
        
        if max_drawdown > -0.2:
            report += "✅ 最大回撤控制在合理范围内\n"
        else:
            report += "⚠️ 最大回撤较大，风险控制需要加强\n"
        
        # 添加改进建议
        if not signals_matched:
            report += "\n🛠️ 改进建议:\n"
            report += "• 建议使用状态管理策略，确保买卖信号一一对应\n"
            report += "• 添加持仓检查逻辑，避免空仓时卖出或持仓时重复买入\n"
            report += "• 考虑使用改进策略: python scripts/improved_strategy.py\n"
        
        return report
    
    def visualize_strategy(self, save_html: bool = True, 
                          show_volume: bool = True, 
                          show_sentiment: bool = True) -> str:
        """运行完整的策略可视化分析"""
        self.logger.info("🚀 开始策略可视化分析...")
        
        # 1. 获取回测数据
        self.logger.info("📊 加载回测数据...")
        backtest_data = self.get_backtest_data()
        
        # 2. 创建可视化图表
        self.logger.info("📈 创建可视化图表...")
        fig = self.create_strategy_chart(backtest_data, show_volume, show_sentiment)
        
        # 3. 分析交易表现
        self.logger.info("🔍 分析交易表现...")
        trade_analysis = self.analyze_trade_performance(backtest_data)
        
        # 4. 生成报告
        report = self.generate_report(backtest_data, trade_analysis)
        print(report)
        
        # 5. 保存图表
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_visualization_{timestamp}.html"
            
            output_dir = project_root / "data" / "analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = output_dir / filename
            fig.write_html(str(filepath))
            
            self.logger.info(f"📁 可视化图表已保存至: {filepath}")
            return str(filepath)
        else:
            # 在浏览器中显示
            fig.show()
            return "图表已在浏览器中显示"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="策略可视化工具")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--no-save", action="store_true", help="不保存HTML文件，直接在浏览器显示")
    parser.add_argument("--no-volume", action="store_true", help="不显示成交量")
    parser.add_argument("--no-sentiment", action="store_true", help="不显示情感得分")
    
    args = parser.parse_args()
    
    try:
        visualizer = StrategyVisualizer(args.config)
        
        result = visualizer.visualize_strategy(
            save_html=not args.no_save,
            show_volume=not args.no_volume,
            show_sentiment=not args.no_sentiment
        )
        
        print(f"\n🎉 可视化完成！{result}")
        
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 