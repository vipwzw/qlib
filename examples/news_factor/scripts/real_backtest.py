#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实新闻情感因子回测系统
基于实际的新闻数据和价格数据进行回测
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import warnings
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# 抑制HTTP请求日志，只显示进度条
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class RealNewsBacktest:
    """真实新闻情感因子回测器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化回测器"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        self.price_data = None
        self.news_data = None
        self.factor_data = None
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return {}
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_real_data(self):
        """加载真实数据"""
        self.logger.info("📊 开始加载真实数据...")
        
        success = True
        
        # 1. 加载价格数据
        if not self._load_price_data():
            success = False
        
        # 2. 加载新闻数据
        if not self._load_news_data():
            success = False
        
        # 3. 加载因子数据
        if not self._load_factor_data():
            success = False
        
        if success:
            self.logger.info("✅ 所有数据加载完成")
        else:
            self.logger.error("❌ 数据加载失败")
        
        return success
    
    def _load_price_data(self):
        """加载价格数据"""
        try:
            price_dir = project_root / "data" / "raw" / "price"
            
            # 优先查找真实数据文件（带"real"标识的）
            real_price_files = list(price_dir.glob("btc_usdt_1m_real_*.csv"))
            
            if real_price_files:
                # 使用最新的真实数据文件
                latest_price_file = max(real_price_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"📈 使用真实市场数据: {latest_price_file.name}")
            else:
                # 回退到模拟数据
                price_files = list(price_dir.glob("btc_usdt_1m_*.csv"))
                if not price_files:
                    self.logger.error("❌ 未找到任何价格数据文件")
                    return False
                latest_price_file = max(price_files, key=lambda x: x.stat().st_mtime)
                self.logger.info(f"📈 使用模拟数据: {latest_price_file.name}")
            
            # 加载数据，根据文件格式选择正确的索引列
            try:
                # 先尝试读取数据查看列名
                temp_df = pd.read_csv(latest_price_file, nrows=5)
                
                if 'timestamp' in temp_df.columns:
                    # 真实数据格式
                    self.price_data = pd.read_csv(latest_price_file)
                    self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                    self.price_data.set_index('timestamp', inplace=True)
                else:
                    # 旧格式（第一列是索引）
                    self.price_data = pd.read_csv(latest_price_file, index_col=0, parse_dates=True)
                
            except Exception as e:
                self.logger.warning(f"解析失败，尝试其他格式: {e}")
                self.price_data = pd.read_csv(latest_price_file, index_col=0, parse_dates=True)
            
            self.logger.info(f"✅ 价格数据: {len(self.price_data)} 条记录")
            self.logger.info(f"📅 价格时间范围: {self.price_data.index.min()} - {self.price_data.index.max()}")
            
            # 显示价格统计信息
            price_stats = {
                'min_price': self.price_data['close'].min(),
                'max_price': self.price_data['close'].max(),
                'avg_price': self.price_data['close'].mean(),
                'price_change': ((self.price_data['close'].iloc[-1] / self.price_data['close'].iloc[0]) - 1) * 100
            }
            
            self.logger.info(f"💰 价格统计:")
            self.logger.info(f"  最低价: ${price_stats['min_price']:,.2f}")
            self.logger.info(f"  最高价: ${price_stats['max_price']:,.2f}")
            self.logger.info(f"  平均价: ${price_stats['avg_price']:,.2f}")
            self.logger.info(f"  总变化: {price_stats['price_change']:+.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 价格数据加载失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    def _load_news_data(self):
        """加载新闻数据"""
        try:
            news_dir = project_root / "data" / "raw" / "news"
            news_files = list(news_dir.glob("crypto_news_*.csv"))
            
            if not news_files:
                self.logger.error("❌ 未找到新闻数据文件")
                return False
            
            # 选择最新的新闻文件
            latest_news_file = max(news_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"📰 加载新闻数据: {latest_news_file.name}")
            
            self.news_data = pd.read_csv(latest_news_file)
            
            # 处理时间列
            self.news_data['published_dt'] = pd.to_datetime(self.news_data['published'], errors='coerce')
            self.news_data = self.news_data.dropna(subset=['published_dt'])
            self.news_data.set_index('published_dt', inplace=True)
            
            self.logger.info(f"✅ 新闻数据: {len(self.news_data)} 条记录")
            self.logger.info(f"📅 新闻时间范围: {self.news_data.index.min()} - {self.news_data.index.max()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 新闻数据加载失败: {e}")
            return False
    
    def _load_factor_data(self):
        """加载因子数据"""
        try:
            factor_dir = project_root / "data" / "factors"
            factor_files = list(factor_dir.glob("factors_*.parquet"))
            
            if not factor_files:
                self.logger.warning("⚠️ 未找到因子数据文件，将使用价格数据计算基础因子")
                return self._calculate_basic_factors()
            
            # 选择最新的因子文件
            latest_factor_file = max(factor_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"🔧 加载因子数据: {latest_factor_file.name}")
            
            self.factor_data = pd.read_parquet(latest_factor_file)
            
            # 确保时间索引
            if 'datetime' in self.factor_data.columns:
                self.factor_data.set_index('datetime', inplace=True)
            
            self.logger.info(f"✅ 因子数据: {len(self.factor_data)} 条记录，{len(self.factor_data.columns)} 个因子")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 因子数据加载失败: {e}")
            return False
    
    def _calculate_basic_factors(self):
        """计算基础技术因子"""
        if self.price_data is None:
            return False
        
        try:
            self.logger.info("🔧 计算基础技术因子...")
            
            df = self.price_data.copy()
            
            # 基础价格因子
            df['returns_1m'] = df['close'].pct_change()
            df['returns_5m'] = df['close'].pct_change(5)
            
            # 移动平均
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_60'] = df['close'].rolling(60).mean()
            
            # 价格相对位置
            df['price_ma5_ratio'] = df['close'] / df['ma_5'] - 1
            df['price_ma20_ratio'] = df['close'] / df['ma_20'] - 1
            
            # 波动率
            df['volatility_20m'] = df['returns_1m'].rolling(20).std()
            df['volatility_60m'] = df['returns_1m'].rolling(60).std()
            
            # 动量指标
            df['momentum_20m'] = (df['close'] / df['close'].shift(20)) - 1
            
            # 成交量因子
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            self.factor_data = df
            self.logger.info(f"✅ 计算了 {len(df.columns)} 个基础因子")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 基础因子计算失败: {e}")
            return False
    
    def create_sentiment_factor(self):
        """创建情感因子（使用现有的因子数据）"""
        try:
            # 直接使用现有的因子数据
            if self.factor_data is not None:
                self.logger.info("✅ 使用现有的因子数据")
                
                # 创建情感因子DataFrame
                sentiment_factor = pd.DataFrame()
                sentiment_factor['sentiment_mean'] = self.factor_data['sentiment_1h_sentiment_score_mean']
                sentiment_factor['sentiment_std'] = self.factor_data['sentiment_1h_sentiment_score_std']
                sentiment_factor['news_count'] = self.factor_data['sentiment_1h_sentiment_score_count']
                sentiment_factor['sentiment_sum'] = self.factor_data['sentiment_1h_sentiment_score_sum']
                
                # 填充缺失值
                sentiment_factor = sentiment_factor.fillna(0)
                
                self.logger.info(f"✅ 情感因子创建完成，覆盖 {len(sentiment_factor)} 个时间点")
                
                return sentiment_factor
            
            else:
                self.logger.error("❌ 因子数据不存在")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 情感因子创建失败: {e}")
            return None
    
    def _apply_simple_sentiment_analysis(self):
        """简单的关键词情感分析（作为DeepSeek的回退方案）"""
        positive_keywords = [
            'rise', 'up', 'bull', 'growth', 'increase', 'profit', 'gain', 
            'positive', 'optimistic', 'surge', 'rally', 'breakout',
            '上涨', '牛市', '增长', '利好', '积极', '突破', '涨幅'
        ]
        
        negative_keywords = [
            'fall', 'down', 'bear', 'decline', 'decrease', 'loss', 'drop',
            'negative', 'pessimistic', 'crash', 'correction', 'dump',
            '下跌', '熊市', '下降', '利空', '消极', '暴跌', '调整'
        ]
        
        def calculate_sentiment(text):
            if pd.isna(text):
                return 0
            
            text_lower = str(text).lower()
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
            negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)
            
            # 简单的情感得分：正面-负面
            sentiment = positive_score - negative_score
            
            # 归一化到[-1, 1]区间
            if sentiment > 0:
                return min(sentiment / 3, 1)  # 最多3个正面词得满分
            elif sentiment < 0:
                return max(sentiment / 3, -1)  # 最多3个负面词得满分（负）
            else:
                return 0
        
        return self.news_data['title'].apply(calculate_sentiment)
    
    def run_backtest(self):
        """运行真实数据回测"""
        self.logger.info("🚀 开始真实数据回测...")
        
        # 加载数据
        if not self.load_real_data():
            return None
        
        # 创建情感因子
        sentiment_factor = self.create_sentiment_factor()
        if sentiment_factor is None:
            return None
        
        # 获取配置参数
        strategy_config = self.config.get('strategy', {})
        signal_config = strategy_config.get('signal_generation', {})
        risk_config = strategy_config.get('risk_management', {})
        backtest_config = self.config.get('evaluation', {}).get('backtest', {})
        
        # 交易模式
        trading_mode = strategy_config.get('trading_mode', 'future')  # 默认合约模式
        
        buy_threshold = signal_config.get('sentiment_threshold_buy', 0.1)
        sell_threshold = signal_config.get('sentiment_threshold_sell', -0.1)
        max_holding_minutes = risk_config.get('max_holding_minutes', 60)
        
        # 动态止损止盈参数
        volatility_lookback = risk_config.get('volatility_lookback', 60)
        stop_loss_multiplier = risk_config.get('stop_loss_multiplier', 1.5)
        take_profit_multiplier = risk_config.get('take_profit_multiplier', 2.5)
        min_stop_loss = risk_config.get('min_stop_loss', 0.005)
        max_stop_loss = risk_config.get('max_stop_loss', 0.05)
        min_take_profit = risk_config.get('min_take_profit', 0.01)
        max_take_profit = risk_config.get('max_take_profit', 0.10)
        
        # 交易成本参数
        transaction_cost = backtest_config.get('transaction_cost', 0.0015)  # 0.15%手续费
        slippage = backtest_config.get('slippage', 0.0000)  # 无滑点
        
        self.logger.info(f"📊 策略参数:")
        self.logger.info(f"  交易模式: {trading_mode}")
        self.logger.info(f"  买入阈值: {buy_threshold}")
        self.logger.info(f"  卖出阈值: {sell_threshold}")
        self.logger.info(f"  动态止损: {stop_loss_multiplier}x波动率 ({min_stop_loss:.1%}-{max_stop_loss:.1%})")
        self.logger.info(f"  动态止盈: {take_profit_multiplier}x波动率 ({min_take_profit:.1%}-{max_take_profit:.1%})")
        self.logger.info(f"  手续费: {transaction_cost:.2%}")
        self.logger.info(f"  滑点: {slippage:.2%}")
        
        # 确保数据对齐
        common_index = self.price_data.index.intersection(sentiment_factor.index)
        if len(common_index) == 0:
            self.logger.error("❌ 价格数据和情感数据时间范围不重叠")
            return None
        
        # 获取对齐的数据
        price_aligned = self.price_data.loc[common_index]
        sentiment_aligned = sentiment_factor.loc[common_index]
        
        self.logger.info(f"📅 回测时间范围: {common_index.min()} - {common_index.max()}")
        self.logger.info(f"📊 有效数据点: {len(common_index)}")
        
        # 初始化回测变量
        initial_capital = self.config.get('evaluation', {}).get('backtest', {}).get('initial_capital', 100000)
        portfolio_value = initial_capital
        position = 0  # 0: 空仓, 1: 多头, -1: 空头
        entry_price = 0
        entry_time = None
        
        trades = []
        portfolio_values = []
        signals = []
        
        # 计算动态止损止盈的函数
        def calculate_dynamic_thresholds(current_index):
            """根据波动率计算动态止损止盈阈值"""
            if current_index < volatility_lookback:
                # 数据不足时使用最小值
                return min_stop_loss, min_take_profit
            
            # 计算过去N分钟的收益率波动率
            start_idx = max(0, current_index - volatility_lookback)
            recent_prices = price_aligned.iloc[start_idx:current_index]['close']
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) < 10:  # 需要至少10个观测值
                return min_stop_loss, min_take_profit
            
            volatility = returns.std()
            
            # 动态止损止盈
            dynamic_stop_loss = stop_loss_multiplier * volatility
            dynamic_take_profit = take_profit_multiplier * volatility
            
            # 限制在合理范围内
            dynamic_stop_loss = max(min_stop_loss, min(max_stop_loss, dynamic_stop_loss))
            dynamic_take_profit = max(min_take_profit, min(max_take_profit, dynamic_take_profit))
            
            return dynamic_stop_loss, dynamic_take_profit
        
        # 逐个时间点进行回测
        for i, (timestamp, price_row) in enumerate(price_aligned.iterrows()):
            current_price = price_row['close']
            sentiment_row = sentiment_aligned.loc[timestamp]
            
            # 当前情感信号
            if 'sentiment_mean' in sentiment_aligned.columns:
                sentiment_signal = sentiment_row['sentiment_mean']
                news_count = sentiment_row['news_count']
            else:
                # 如果没有sentiment_mean列，尝试使用其他列名
                sentiment_cols = [col for col in sentiment_aligned.columns if 'sentiment' in col and 'mean' in col]
                if sentiment_cols:
                    sentiment_signal = sentiment_row[sentiment_cols[0]]
                else:
                    sentiment_signal = 0
                
                news_count_cols = [col for col in sentiment_aligned.columns if 'count' in col]
                if news_count_cols:
                    news_count = sentiment_row[news_count_cols[0]]
                else:
                    news_count = 0
            
            # 计算当前的动态止损止盈阈值
            current_stop_loss, current_take_profit = calculate_dynamic_thresholds(i)
            
            # 记录信号
            signal = 0
            
            # 平仓逻辑
            if position != 0:
                # 计算持仓收益（扣除交易成本）
                if position == 1:  # 多头
                    # 开仓成本：手续费 + 滑点
                    entry_cost = transaction_cost + slippage
                    # 平仓成本：手续费 + 滑点
                    exit_cost = transaction_cost + slippage
                    # 净收益 = 价格收益 - 总交易成本
                    pnl_pct = (current_price - entry_price) / entry_price - entry_cost - exit_cost
                else:  # 空头（现货模式下不应该出现）
                    # 开仓成本：手续费 + 滑点
                    entry_cost = transaction_cost + slippage
                    # 平仓成本：手续费 + 滑点
                    exit_cost = transaction_cost + slippage
                    # 净收益 = 价格收益 - 总交易成本
                    pnl_pct = (entry_price - current_price) / entry_price - entry_cost - exit_cost
                
                # 检查平仓条件
                should_exit = False
                exit_reason = ""
                
                # 动态止盈止损
                if pnl_pct >= current_take_profit:
                    should_exit = True
                    exit_reason = f"止盈({current_take_profit:.1%})"
                elif pnl_pct <= -current_stop_loss:
                    should_exit = True
                    exit_reason = f"止损({current_stop_loss:.1%})"
                
                # 最大持仓时间
                if entry_time and (timestamp - entry_time).total_seconds() / 60 >= max_holding_minutes:
                    should_exit = True
                    exit_reason = "超时"
                
                # 反向信号（现货模式）
                if trading_mode == "spot":
                    # 现货模式：只在持有多头时检查卖出信号
                    if position == 1 and sentiment_signal <= sell_threshold:
                        should_exit = True
                        exit_reason = "反向信号"
                else:
                    # 合约模式：检查双向反向信号
                    if position == 1 and sentiment_signal <= sell_threshold:
                        should_exit = True
                        exit_reason = "反向信号"
                    elif position == -1 and sentiment_signal >= buy_threshold:
                        should_exit = True
                        exit_reason = "反向信号"
                
                # 执行平仓
                if should_exit:
                    # 记录交易
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position': position,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': portfolio_value * pnl_pct,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # 更新组合价值
                    portfolio_value *= (1 + pnl_pct)
                    
                    # 重置持仓
                    position = 0
                    entry_price = 0
                    entry_time = None
                    
                    signal = -position  # 平仓信号
            
            # 开仓逻辑（仅在空仓时）
            if position == 0:  # 不需要限制当前时刻有新闻，因为情感因子是基于历史新闻计算的
                if sentiment_signal >= buy_threshold:
                    # 开多头
                    position = 1
                    entry_price = current_price
                    entry_time = timestamp
                    signal = 1
                elif sentiment_signal <= sell_threshold and trading_mode != "spot":
                    # 开空头（仅在合约模式下）
                    position = -1
                    entry_price = current_price
                    entry_time = timestamp
                    signal = -1
            
            # 记录状态
            portfolio_values.append(portfolio_value)
            signals.append(signal)
        
        # 如果最后还有持仓，强制平仓
        if position != 0:
            final_price = price_aligned.iloc[-1]['close']
            if position == 1:
                # 扣除开仓和平仓成本
                entry_cost = transaction_cost + slippage
                exit_cost = transaction_cost + slippage
                pnl_pct = (final_price - entry_price) / entry_price - entry_cost - exit_cost
            else:
                # 扣除开仓和平仓成本
                entry_cost = transaction_cost + slippage
                exit_cost = transaction_cost + slippage
                pnl_pct = (entry_price - final_price) / entry_price - entry_cost - exit_cost
            
            trade = {
                'entry_time': entry_time,
                'exit_time': common_index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position': position,
                'pnl_pct': pnl_pct,
                'pnl_amount': portfolio_value * pnl_pct,
                'exit_reason': "回测结束"
            }
            trades.append(trade)
            portfolio_value *= (1 + pnl_pct)
            portfolio_values[-1] = portfolio_value
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'signal': signals
        }, index=common_index)
        
        # 计算性能指标
        performance = self._calculate_real_performance(results, trades, initial_capital)
        
        # 输出结果
        self._print_real_results(performance, trades)
        
        return {
            'performance': performance,
            'trades': trades,
            'results': results
        }
    
    def _calculate_real_performance(self, results: pd.DataFrame, trades: list, initial_capital: float) -> dict:
        """计算真实回测性能指标"""
        
        portfolio_values = results['portfolio_value']
        
        # 基本指标
        total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
        
        # 时间跨度（天）
        time_span_days = (results.index[-1] - results.index[0]).total_seconds() / (24 * 3600)
        annualized_return = (1 + total_return) ** (365 / time_span_days) - 1 if time_span_days > 0 else 0
        
        # 计算收益率序列
        portfolio_returns = portfolio_values.pct_change().fillna(0)
        
        # 波动率（年化）
        volatility = portfolio_returns.std() * np.sqrt(365 * 24 * 60) if len(portfolio_returns) > 1 else 0
        
        # 夏普比率
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # 交易统计
        if trades:
            trade_pnls = [t['pnl_pct'] for t in trades]
            win_trades = [t for t in trades if t['pnl_pct'] > 0]
            lose_trades = [t for t in trades if t['pnl_pct'] <= 0]
            
            win_rate = len(win_trades) / len(trades) if trades else 0
            avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in lose_trades]) if lose_trades else 0
            profit_factor = abs(avg_win * len(win_trades) / (avg_loss * len(lose_trades))) if lose_trades and avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        performance = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_portfolio_value': portfolio_values.iloc[-1],
            'total_trades': len(trades),
            'time_span_days': time_span_days
        }
        
        return performance
    
    def _print_real_results(self, performance: dict, trades: list):
        """输出真实回测结果"""
        print("\n" + "="*60)
        print("🎯 真实新闻情感因子策略回测结果 (现货交易/已扣除手续费)")
        print("="*60)
        print(f"总收益率: {performance['total_return']:.2%}")
        print(f"年化收益率: {performance['annualized_return']:.2%}")
        print(f"年化波动率: {performance['volatility']:.2%}")
        print(f"夏普比率: {performance['sharpe_ratio']:.3f}")
        print(f"最大回撤: {performance['max_drawdown']:.2%}")
        print(f"胜率: {performance['win_rate']:.2%}")
        print(f"平均盈利: {performance['avg_win']:.2%}")
        print(f"平均亏损: {performance['avg_loss']:.2%}")
        print(f"盈亏比: {performance['profit_factor']:.2f}")
        print(f"最终组合价值: ${performance['final_portfolio_value']:,.2f}")
        print(f"总交易次数: {performance['total_trades']}")
        print(f"回测时长: {performance['time_span_days']:.1f} 天")
        print("="*60)
        
        # 显示最近几笔交易
        if trades:
            print(f"\n📋 最近5笔交易详情:")
            print("-" * 80)
            for trade in trades[-5:]:
                entry_time = trade['entry_time'].strftime('%m-%d %H:%M')
                exit_time = trade['exit_time'].strftime('%m-%d %H:%M')
                position_str = "多头" if trade['position'] == 1 else "空头"
                pnl_str = f"{trade['pnl_pct']:+.2%}"
                
                print(f"{entry_time} -> {exit_time} | {position_str} | {pnl_str} | {trade['exit_reason']}")


def main():
    """主函数"""
    try:
        backtester = RealNewsBacktest()
        result = backtester.run_backtest()
        
        if result:
            print("\n✅ 真实数据回测完成！")
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存交易记录
            trades_df = pd.DataFrame(result['trades'])
            trades_file = project_root / "data" / "results" / f"real_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 交易记录已保存: {trades_file}")
            
            # 保存性能指标
            performance_df = pd.DataFrame([result['performance']])
            performance_file = project_root / "data" / "results" / f"real_performance_{timestamp}.csv"
            performance_df.to_csv(performance_file, index=False)
            print(f"💾 性能指标已保存: {performance_file}")
            
        else:
            print("❌ 真实数据回测失败")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ 回测运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 