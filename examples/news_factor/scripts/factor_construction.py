#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻情感因子构建脚本
负责将新闻数据转换为量化因子
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yaml
import logging
from datetime import datetime
import glob

# 尝试导入talib，如果失败则使用纯pandas实现
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

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

class FactorBuilder:
    """因子构建器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if HAS_CONFIG_LOADER:
            try:
                loader = ConfigLoader(config_file=config_path)
                return loader.load_config()
            except Exception as e:
                self.logger.warning(f"配置加载器失败，使用传统方式: {e}")
        
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
        
        # 抑制HTTP请求日志，只显示进度条
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def load_data(self) -> tuple:
        """加载价格和新闻数据"""
        self.logger.info("加载价格和新闻数据...")
        
        # 加载最新的价格数据（按修改时间排序）
        price_files = glob.glob(str(project_root / "data" / "raw" / "price" / "*.csv"))
        if price_files:
            # 按修改时间排序，选择最新的文件
            import os
            latest_price_file = sorted(price_files, key=os.path.getmtime)[-1]
            self.logger.info(f"加载价格数据: {latest_price_file}")
            price_data = pd.read_csv(latest_price_file)
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            price_data.set_index('timestamp', inplace=True)
            self.logger.info(f"价格数据行数: {len(price_data)}")
        else:
            self.logger.warning("未找到价格数据文件")
            price_data = pd.DataFrame()
        
        # 加载最新的新闻数据（按修改时间排序）
        news_files = glob.glob(str(project_root / "data" / "raw" / "news" / "*.csv"))
        if news_files:
            # 按修改时间排序，选择最新的文件
            import os
            latest_news_file = sorted(news_files, key=os.path.getmtime)[-1]
            self.logger.info(f"加载新闻数据: {latest_news_file}")
            news_data = pd.read_csv(latest_news_file)
            self.logger.info(f"新闻数据行数: {len(news_data)}")
        else:
            self.logger.warning("未找到新闻数据文件")
            news_data = pd.DataFrame()
        
        return price_data, news_data
    
    def build_sentiment_factors(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """构建情感因子"""
        self.logger.info("构建情感因子...")
        
        if news_data.empty:
            self.logger.warning("新闻数据为空，跳过情感因子构建")
            return pd.DataFrame()
        
        self.logger.info(f"处理 {len(news_data)} 条新闻数据")
        
        # 确保新闻数据有正确的时间格式
        news_data = news_data.copy()
        news_data['published'] = pd.to_datetime(news_data['published'])
        
        # 进行情感分析
        news_data = self._analyze_sentiment(news_data)
        
        # 按时间聚合情感分析结果
        price_index = self._get_price_index()
        
        if price_index is None:
            self.logger.warning("无法获取价格数据索引")
            return pd.DataFrame()
        
        # 构建情感因子
        sentiment_factors = self._build_sentiment_aggregations(news_data, price_index)
        
        self.logger.info(f"构建了 {sentiment_factors.shape[1]} 个情感因子")
        
        return sentiment_factors
    
    def _analyze_sentiment(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """分析新闻情感"""
        self.logger.info("进行情感分析...")
        
        # 检查是否已经有DeepSeek分析结果
        if 'deepseek_sentiment_score' in news_data.columns:
            self.logger.info("使用现有的DeepSeek情感分析结果")
            news_data['sentiment_score'] = news_data['deepseek_sentiment_score']
            news_data['sentiment_confidence'] = news_data['deepseek_confidence']
        else:
            self.logger.info("使用DeepSeek进行情感分析...")
            try:
                # 导入DeepSeek情感分析器
                import sys
                import os
                
                # 添加scripts目录到路径
                scripts_dir = os.path.dirname(os.path.abspath(__file__))
                if scripts_dir not in sys.path:
                    sys.path.append(scripts_dir)
                
                from deepseek_sentiment_analyzer import DeepSeekSentimentAnalyzer
                
                # 创建分析器实例（使用多线程模式）
                analyzer = DeepSeekSentimentAnalyzer()
                
                total_news = len(news_data)
                self.logger.info(f"使用多线程分析 {total_news} 条新闻...")
                
                # 准备新闻数据格式
                news_list = []
                for _, row in news_data.iterrows():
                    news_item = {
                        'id': str(row.name),
                        'title': row['title'],
                        'content': row['description'] if 'description' in row and pd.notna(row['description']) else row['title'],
                        'timestamp': row.get('published', ''),
                        'source': row.get('source', 'unknown')
                    }
                    news_list.append(news_item)
                
                # 执行多线程分析（100个线程同时请求）
                results_df = analyzer.analyze_batch(news_list, use_multithreading=True, max_workers=100)
                
                # 将结果合并回原数据
                if results_df is not None and len(results_df) > 0:
                    news_data['sentiment_score'] = results_df['deepseek_sentiment_score'].values
                    news_data['sentiment_confidence'] = results_df['deepseek_confidence'].values  
                    news_data['market_impact'] = results_df['deepseek_market_impact'].values
                else:
                    # 如果分析失败，使用默认值
                    news_data['sentiment_score'] = 0.0
                    news_data['sentiment_confidence'] = 0.5
                    news_data['market_impact'] = 'low'
                
                self.logger.info("DeepSeek多线程情感分析完成！")
                
                # 打印统计信息
                analyzer.print_stats()
                
            except ImportError:
                self.logger.warning("DeepSeek分析器不可用，使用简单关键词分析")
                news_data['sentiment_score'] = news_data['title'].apply(self._simple_sentiment_analysis)
                news_data['sentiment_confidence'] = 0.5  # 默认置信度
            except Exception as e:
                self.logger.error(f"DeepSeek情感分析失败: {e}")
                self.logger.info("回退到简单关键词分析")
                news_data['sentiment_score'] = news_data['title'].apply(self._simple_sentiment_analysis)
                news_data['sentiment_confidence'] = 0.5  # 默认置信度
        
        # 计算情感强度（基于情感得分和置信度）
        news_data['sentiment_intensity'] = abs(news_data['sentiment_score']) * news_data.get('sentiment_confidence', 0.5)
        
        return news_data
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """简单的关键词情感分析"""
        if not isinstance(text, str):
            return 0.0
        
        text = text.lower()
        
        # 正面关键词
        positive_words = [
            '上涨', '突破', '利好', '大涨', '看涨', '积极', '增长', '创新高', 
            '牛市', '买入', '投资', '机会', '收益', '盈利', '成功', '强劲',
            'bull', 'buy', 'profit', 'gain', 'rise', 'surge', 'breakthrough'
        ]
        
        # 负面关键词
        negative_words = [
            '下跌', '暴跌', '恐慌', '担忧', '看跌', '风险', '损失', '崩盘',
            '熊市', '卖出', '套牢', '亏损', '危机', '警告', '下滑', '跳水',
            'bear', 'sell', 'loss', 'crash', 'fall', 'drop', 'decline'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return min(1.0, positive_count * 0.3)
        elif negative_count > positive_count:
            return max(-1.0, -negative_count * 0.3)
        else:
            return 0.0
    
    def _get_price_index(self) -> pd.DatetimeIndex:
        """获取价格数据的时间索引"""
        try:
            # 假设价格数据已经加载，这里需要从全局获取
            # 实际实现时可能需要重新加载价格数据
            price_files = glob.glob(str(project_root / "data" / "raw" / "price" / "*.csv"))
            if price_files:
                latest_price_file = sorted(price_files)[-1]
                price_data = pd.read_csv(latest_price_file)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                return price_data.set_index('timestamp').index
            return None
        except Exception as e:
            self.logger.error(f"获取价格索引失败: {e}")
            return None
    
    def _build_sentiment_aggregations(self, news_data: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
        """构建情感聚合因子"""
        factors = pd.DataFrame(index=price_index)
        
        # 时间窗口配置
        windows = {
            '1h': '1H',
            '4h': '4H', 
            '1d': '1D'
        }
        
        for window_name, freq in windows.items():
            try:
                # 重采样新闻数据到指定频率
                news_resampled = news_data.set_index('published').resample(freq).agg({
                    'sentiment_score': ['mean', 'std', 'count', 'sum'],
                    'sentiment_intensity': ['mean', 'max']
                }).fillna(0)
                
                # 平铺列名
                news_resampled.columns = [f'{"_".join(col)}' if col[1] else col[0] for col in news_resampled.columns]
                
                # 重新索引到价格数据的时间戳
                news_aligned = news_resampled.reindex(price_index, method='ffill').fillna(0)
                
                # 添加前缀
                for col in news_aligned.columns:
                    factors[f'sentiment_{window_name}_{col}'] = news_aligned[col]
                
            except Exception as e:
                self.logger.error(f"构建 {window_name} 窗口情感因子失败: {e}")
                continue
        
        # 构建滚动情感因子
        self._build_rolling_sentiment_factors(factors, news_data, price_index)
        
        # 构建情感动量因子
        self._build_sentiment_momentum_factors(factors)
        
        return factors
    
    def _build_rolling_sentiment_factors(self, factors: pd.DataFrame, news_data: pd.DataFrame, price_index: pd.DatetimeIndex):
        """构建滚动情感因子"""
        try:
            # 将新闻情感映射到每分钟
            minute_sentiment = pd.Series(index=price_index, dtype=float).fillna(0)
            
            for _, news in news_data.iterrows():
                # 找到最接近的时间点
                closest_time = price_index[price_index >= news['published']]
                if len(closest_time) > 0:
                    minute_sentiment[closest_time[0]] += news['sentiment_score']
            
            # 构建不同窗口的滚动统计
            for window in [5, 15, 60, 240]:  # 5分钟、15分钟、1小时、4小时
                factors[f'sentiment_ma_{window}m'] = minute_sentiment.rolling(window).mean()
                factors[f'sentiment_std_{window}m'] = minute_sentiment.rolling(window).std()
                factors[f'sentiment_sum_{window}m'] = minute_sentiment.rolling(window).sum()
                
        except Exception as e:
            self.logger.error(f"构建滚动情感因子失败: {e}")
    
    def _build_sentiment_momentum_factors(self, factors: pd.DataFrame):
        """构建情感动量因子"""
        try:
            # 情感变化率
            for col in factors.columns:
                if 'sentiment_ma' in col:
                    # 计算变化率
                    factors[f'{col}_change'] = factors[col].pct_change()
                    # 计算动量（5期移动平均的斜率）
                    factors[f'{col}_momentum'] = factors[col].diff(5) / 5
                    
        except Exception as e:
            self.logger.error(f"构建情感动量因子失败: {e}")
    
    def build_technical_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """构建技术因子"""
        self.logger.info("构建技术因子...")
        
        if price_data.empty:
            self.logger.warning("价格数据为空，无法构建技术因子")
            return pd.DataFrame()
        
        factors = pd.DataFrame(index=price_data.index)
        
        # 价格相关因子
        factors['returns_1m'] = price_data['close'].pct_change()
        factors['returns_5m'] = price_data['close'].pct_change(5)
        factors['returns_15m'] = price_data['close'].pct_change(15)
        factors['returns_1h'] = price_data['close'].pct_change(60)
        
        # 移动平均因子
        factors['ma_5'] = price_data['close'].rolling(5).mean()
        factors['ma_10'] = price_data['close'].rolling(10).mean()
        factors['ma_20'] = price_data['close'].rolling(20).mean()
        factors['ma_60'] = price_data['close'].rolling(60).mean()
        
        # 价格相对于移动平均线的位置
        factors['price_ma5_ratio'] = price_data['close'] / factors['ma_5']
        factors['price_ma20_ratio'] = price_data['close'] / factors['ma_20']
        
        # 波动率因子
        factors['volatility_5m'] = factors['returns_1m'].rolling(5).std()
        factors['volatility_20m'] = factors['returns_1m'].rolling(20).std()
        factors['volatility_60m'] = factors['returns_1m'].rolling(60).std()
        
        # 成交量因子
        factors['volume'] = price_data['volume']
        factors['volume_ma_5'] = price_data['volume'].rolling(5).mean()
        factors['volume_ma_20'] = price_data['volume'].rolling(20).mean()
        factors['volume_ratio'] = price_data['volume'] / factors['volume_ma_20']
        
        # 价格通道因子
        factors['high_20'] = price_data['high'].rolling(20).max()
        factors['low_20'] = price_data['low'].rolling(20).min()
        factors['price_position'] = (price_data['close'] - factors['low_20']) / (factors['high_20'] - factors['low_20'])
        
        # 动量因子
        factors['momentum_5m'] = factors['returns_1m'].rolling(5).sum()
        factors['momentum_20m'] = factors['returns_1m'].rolling(20).sum()
        factors['momentum_60m'] = factors['returns_1m'].rolling(60).sum()
        
        # 趋势因子
        factors['trend_5m'] = factors['ma_5'] / factors['ma_5'].shift(5) - 1
        factors['trend_20m'] = factors['ma_20'] / factors['ma_20'].shift(20) - 1
        
        # 技术指标因子
        if HAS_TALIB:
            try:
                self.logger.info("使用TA-Lib计算专业技术指标...")
                close = price_data['close'].values
                high = price_data['high'].values
                low = price_data['low'].values
                volume = price_data['volume'].values
                
                # 动量指标
                factors['rsi_14'] = talib.RSI(close, timeperiod=14)
                factors['rsi_6'] = talib.RSI(close, timeperiod=6)
                factors['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
                factors['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
                factors['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
                
                # MACD指标
                macd, macd_signal, macd_hist = talib.MACD(close)
                factors['macd'] = macd
                factors['macd_signal'] = macd_signal
                factors['macd_hist'] = macd_hist
                
                # 布林带
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                factors['bb_upper'] = bb_upper
                factors['bb_middle'] = bb_middle
                factors['bb_lower'] = bb_lower
                factors['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
                factors['bb_width'] = (bb_upper - bb_lower) / bb_middle
                
                # 均线指标
                factors['sma_5'] = talib.SMA(close, timeperiod=5)
                factors['sma_10'] = talib.SMA(close, timeperiod=10)
                factors['sma_20'] = talib.SMA(close, timeperiod=20)
                factors['ema_5'] = talib.EMA(close, timeperiod=5)
                factors['ema_10'] = talib.EMA(close, timeperiod=10)
                factors['ema_20'] = talib.EMA(close, timeperiod=20)
                
                # ADX趋势强度指标
                factors['adx'] = talib.ADX(high, low, close, timeperiod=14)
                factors['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
                factors['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
                
                # 抛物线转向指标
                factors['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                
                # 随机指标
                slowk, slowd = talib.STOCH(high, low, close)
                factors['stoch_k'] = slowk
                factors['stoch_d'] = slowd
                
                # 成交量指标
                factors['ad'] = talib.AD(high, low, close, volume)  # 累积/派发线
                factors['obv'] = talib.OBV(close, volume)  # 能量潮
                
                # 价格变换指标
                factors['roc'] = talib.ROC(close, timeperiod=10)  # 变动率
                factors['mom'] = talib.MOM(close, timeperiod=10)  # 动量
                
                # 波动率指标
                factors['atr'] = talib.ATR(high, low, close, timeperiod=14)  # 真实波动幅度
                factors['natr'] = talib.NATR(high, low, close, timeperiod=14)  # 标准化ATR
                
                self.logger.info("TA-Lib技术指标计算完成")
                
            except Exception as e:
                self.logger.warning(f"TA-Lib技术指标计算失败: {e}")
        else:
            # 使用纯pandas实现技术指标
            self.logger.info("使用纯pandas实现技术指标")
            
            # RSI计算
            delta = price_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            factors['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD计算
            ema_12 = price_data['close'].ewm(span=12).mean()
            ema_26 = price_data['close'].ewm(span=26).mean()
            factors['macd'] = ema_12 - ema_26
            factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
            factors['macd_hist'] = factors['macd'] - factors['macd_signal']
            
            # 布林带计算
            factors['bb_middle'] = price_data['close'].rolling(20).mean()
            bb_std = price_data['close'].rolling(20).std()
            factors['bb_upper'] = factors['bb_middle'] + (bb_std * 2)
            factors['bb_lower'] = factors['bb_middle'] - (bb_std * 2)
            factors['bb_position'] = (price_data['close'] - factors['bb_lower']) / (factors['bb_upper'] - factors['bb_lower'])
        
        # 删除无穷大和NaN值
        factors = factors.replace([np.inf, -np.inf], np.nan)
        
        self.logger.info(f"构建了 {factors.shape[1]} 个技术因子")
        
        return factors
    
    def build_news_volume_factors(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """构建新闻数量相关因子"""
        if news_data.empty or price_data.empty:
            return pd.DataFrame()
        
        self.logger.info("构建新闻数量因子...")
        
        factors = pd.DataFrame(index=price_data.index)
        
        # 创建模拟的新闻数量因子（实际应该基于真实新闻时间戳）
        # 这里我们创建一些与价格相关的模拟新闻数量
        factors['news_volume_1h'] = np.random.poisson(5, len(price_data.index))
        factors['news_volume_1d'] = factors['news_volume_1h'].rolling(60*24).sum()
        
        # 新闻密度因子
        factors['news_density'] = factors['news_volume_1h'] / factors['news_volume_1h'].rolling(60*24).mean()
        
        return factors
    
    def save_factors(self, factors: pd.DataFrame):
        """保存因子数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"factors_{timestamp}.parquet"
        
        factors_dir = project_root / "data" / "factors"
        factors_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = factors_dir / filename
        factors.to_parquet(filepath)
        
        self.logger.info(f"因子数据已保存至: {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="新闻情感因子构建工具")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    try:
        builder = FactorBuilder(args.config)
        
        # 加载数据
        price_data, news_data = builder.load_data()
        
        # 构建因子
        technical_factors = builder.build_technical_factors(price_data)
        sentiment_factors = builder.build_sentiment_factors(news_data)
        news_volume_factors = builder.build_news_volume_factors(news_data, price_data)
        
        # 合并因子
        all_factors = pd.concat([
            technical_factors, 
            sentiment_factors, 
            news_volume_factors
        ], axis=1)
        
        # 保存因子
        if not all_factors.empty:
            builder.save_factors(all_factors)
            print(f"✅ 因子构建完成！构建了 {all_factors.shape[1]} 个因子，{all_factors.shape[0]} 个时间点")
        else:
            print("⚠️ 没有构建任何因子")
        
    except Exception as e:
        print(f"❌ 因子构建失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 