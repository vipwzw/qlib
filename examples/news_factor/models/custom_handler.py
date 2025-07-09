#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义数据处理器
用于处理新闻情感因子和价格数据
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.data import D
import warnings
warnings.filterwarnings('ignore')

class CryptoNewsSentimentHandler(DataHandlerLP):
    """
    加密货币新闻情感数据处理器
    继承自qlib的DataHandlerLP，专门处理新闻情感因子
    """
    
    def __init__(
        self,
        instruments="btcusdt",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        news_factor_path=None,
        **kwargs
    ):
        """
        初始化处理器
        
        Parameters:
        -----------
        news_factor_path: str
            新闻因子数据文件路径
        """
        self.news_factor_path = news_factor_path
        
        # 定义特征表达式
        fields = self._get_feature_expressions()
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            fields=fields,
            **kwargs
        )
    
    def _get_feature_expressions(self) -> List[str]:
        """
        定义特征表达式
        
        Returns:
        --------
        List[str]: 特征表达式列表
        """
        features = []
        
        # 价格相关特征
        price_features = [
            "($close-$open)/$open",  # 日内收益率
            "($high-$low)/$close",   # 日内波动率
            "$volume",               # 成交量
            "Ref($close, 1)",        # 前一日收盘价
            "($close/Ref($close, 1)-1)",  # 日收益率
        ]
        
        # 技术指标特征
        technical_features = [
            "Mean($close, 5)",       # 5日均价
            "Mean($close, 20)",      # 20日均价
            "Std($close, 20)",       # 20日价格标准差
            "Max($high, 20)",        # 20日最高价
            "Min($low, 20)",         # 20日最低价
        ]
        
        # 新闻情感特征（如果有新闻因子数据）
        if self.news_factor_path:
            sentiment_features = [
                "sentiment_score_1m",     # 1分钟情感得分
                "sentiment_score_5m",     # 5分钟情感得分
                "sentiment_score_1h",     # 1小时情感得分
                "news_volume_1h",         # 1小时新闻量
                "sentiment_volatility",   # 情感波动率
                "weighted_sentiment",     # 加权情感得分
            ]
            features.extend(sentiment_features)
        
        features.extend(price_features)
        features.extend(technical_features)
        
        return features
    
    def fetch_data(self) -> pd.DataFrame:
        """
        获取数据
        
        Returns:
        --------
        pd.DataFrame: 合并后的特征数据
        """
        # 获取价格数据
        price_data = super().fetch_data()
        
        # 如果有新闻因子数据，进行合并
        if self.news_factor_path and hasattr(self, 'news_factor_path'):
            try:
                news_factors = self._load_news_factors()
                if not news_factors.empty:
                    # 时间对齐和合并
                    combined_data = self._merge_data(price_data, news_factors)
                    return combined_data
            except Exception as e:
                print(f"加载新闻因子数据失败: {e}")
                print("使用纯价格数据继续...")
        
        return price_data
    
    def _load_news_factors(self) -> pd.DataFrame:
        """
        加载新闻因子数据
        
        Returns:
        --------
        pd.DataFrame: 新闻因子数据
        """
        import os
        from pathlib import Path
        
        factor_path = Path(self.news_factor_path)
        
        if not factor_path.exists():
            print(f"新闻因子文件不存在: {factor_path}")
            return pd.DataFrame()
        
        # 根据文件格式读取
        if factor_path.suffix == '.parquet':
            news_factors = pd.read_parquet(factor_path)
        elif factor_path.suffix == '.csv':
            news_factors = pd.read_csv(factor_path, index_col=0, parse_dates=True)
        else:
            print(f"不支持的文件格式: {factor_path.suffix}")
            return pd.DataFrame()
        
        # 确保索引为datetime
        if not isinstance(news_factors.index, pd.DatetimeIndex):
            news_factors.index = pd.to_datetime(news_factors.index)
        
        return news_factors
    
    def _merge_data(self, price_data: pd.DataFrame, news_factors: pd.DataFrame) -> pd.DataFrame:
        """
        合并价格数据和新闻因子数据
        
        Parameters:
        -----------
        price_data: pd.DataFrame
            价格数据
        news_factors: pd.DataFrame
            新闻因子数据
        
        Returns:
        --------
        pd.DataFrame: 合并后的数据
        """
        # 重采样新闻因子到价格数据频率
        if self.freq == "day":
            # 日频数据：对新闻因子进行日度聚合
            news_daily = news_factors.resample('D').agg({
                'sentiment_score_1m': 'mean',
                'sentiment_score_5m': 'mean', 
                'sentiment_score_1h': 'mean',
                'news_volume_1h': 'sum',
                'sentiment_volatility': 'mean',
                'weighted_sentiment': 'mean'
            }).dropna()
        else:
            # 其他频率：直接使用
            news_daily = news_factors
        
        # 合并数据
        combined_data = price_data.join(news_daily, how='left')
        
        # 前向填充缺失的新闻因子
        sentiment_cols = [col for col in news_daily.columns if col in combined_data.columns]
        combined_data[sentiment_cols] = combined_data[sentiment_cols].fillna(method='ffill')
        
        # 如果仍有缺失值，用0填充
        combined_data[sentiment_cols] = combined_data[sentiment_cols].fillna(0)
        
        return combined_data


class NewsFactorProcessor(Processor):
    """
    新闻因子预处理器
    """
    
    def __init__(self, fields_group=None, **kwargs):
        self.fields_group = fields_group or ["sentiment"]
        super().__init__(**kwargs)
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理新闻因子数据
        
        Parameters:
        -----------
        df: pd.DataFrame
            输入数据
        
        Returns:
        --------
        pd.DataFrame: 处理后的数据
        """
        # 标识新闻因子列
        sentiment_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['sentiment', 'news_volume', 'weighted']
        )]
        
        if not sentiment_cols:
            return df
        
        # 异常值处理
        for col in sentiment_cols:
            if col in df.columns:
                # 使用3倍标准差方法处理异常值
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # 设置上下界
                upper_bound = mean_val + 3 * std_val
                lower_bound = mean_val - 3 * std_val
                
                # 异常值截断
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df


class NewsFactorNormalizer(Processor):
    """
    新闻因子标准化处理器
    """
    
    def __init__(self, method="zscore", **kwargs):
        """
        初始化标准化处理器
        
        Parameters:
        -----------
        method: str
            标准化方法，支持 'zscore', 'minmax', 'robust'
        """
        self.method = method
        super().__init__(**kwargs)
    
    def fit(self, df: pd.DataFrame) -> "NewsFactorNormalizer":
        """
        训练标准化参数
        
        Parameters:
        -----------
        df: pd.DataFrame
            训练数据
        
        Returns:
        --------
        self
        """
        # 识别新闻因子列
        sentiment_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['sentiment', 'news_volume', 'weighted']
        )]
        
        if not sentiment_cols:
            return self
        
        self.stats = {}
        
        for col in sentiment_cols:
            if col in df.columns:
                if self.method == "zscore":
                    self.stats[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
                elif self.method == "minmax":
                    self.stats[col] = {
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                elif self.method == "robust":
                    self.stats[col] = {
                        'median': df[col].median(),
                        'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                    }
        
        return self
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用标准化
        
        Parameters:
        -----------
        df: pd.DataFrame
            输入数据
        
        Returns:
        --------
        pd.DataFrame: 标准化后的数据
        """
        if not hasattr(self, 'stats'):
            return df
        
        result_df = df.copy()
        
        for col, stats in self.stats.items():
            if col in result_df.columns:
                if self.method == "zscore":
                    if stats['std'] > 0:
                        result_df[col] = (result_df[col] - stats['mean']) / stats['std']
                elif self.method == "minmax":
                    if stats['max'] > stats['min']:
                        result_df[col] = (result_df[col] - stats['min']) / (stats['max'] - stats['min'])
                elif self.method == "robust":
                    if stats['iqr'] > 0:
                        result_df[col] = (result_df[col] - stats['median']) / stats['iqr']
        
        return result_df


# 预定义的处理器配置
DEFAULT_NEWS_PROCESSORS = [
    NewsFactorProcessor(),
    NewsFactorNormalizer(method="zscore")
] 