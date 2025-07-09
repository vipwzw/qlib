
# 因子分析报告

## 基本信息
- 因子文件: factors_20250707_195816.parquet
- 因子数量: 97
- 时间点数量: 43201
- 时间范围: 2025-06-07 19:12:19.710743 至 2025-07-07 19:12:19.710743

## 因子列表
- returns_1m
- returns_5m
- returns_15m
- returns_1h
- ma_5
- ma_10
- ma_20
- ma_60
- price_ma5_ratio
- price_ma20_ratio
- volatility_5m
- volatility_20m
- volatility_60m
- volume
- volume_ma_5
- volume_ma_20
- volume_ratio
- high_20
- low_20
- price_position
- momentum_5m
- momentum_20m
- momentum_60m
- trend_5m
- trend_20m
- rsi_14
- rsi_6
- cci_14
- williams_r
- mfi
- macd
- macd_signal
- macd_hist
- bb_upper
- bb_middle
- bb_lower
- bb_position
- bb_width
- sma_5
- sma_10
- sma_20
- ema_5
- ema_10
- ema_20
- adx
- plus_di
- minus_di
- sar
- stoch_k
- stoch_d
- ad
- obv
- roc
- mom
- atr
- natr
- sentiment_1h_sentiment_score_mean
- sentiment_1h_sentiment_score_std
- sentiment_1h_sentiment_score_count
- sentiment_1h_sentiment_score_sum
- sentiment_1h_sentiment_intensity_mean
- sentiment_1h_sentiment_intensity_max
- sentiment_4h_sentiment_score_mean
- sentiment_4h_sentiment_score_std
- sentiment_4h_sentiment_score_count
- sentiment_4h_sentiment_score_sum
- sentiment_4h_sentiment_intensity_mean
- sentiment_4h_sentiment_intensity_max
- sentiment_1d_sentiment_score_mean
- sentiment_1d_sentiment_score_std
- sentiment_1d_sentiment_score_count
- sentiment_1d_sentiment_score_sum
- sentiment_1d_sentiment_intensity_mean
- sentiment_1d_sentiment_intensity_max
- sentiment_ma_5m
- sentiment_std_5m
- sentiment_sum_5m
- sentiment_ma_15m
- sentiment_std_15m
- sentiment_sum_15m
- sentiment_ma_60m
- sentiment_std_60m
- sentiment_sum_60m
- sentiment_ma_240m
- sentiment_std_240m
- sentiment_sum_240m
- sentiment_ma_5m_change
- sentiment_ma_5m_momentum
- sentiment_ma_15m_change
- sentiment_ma_15m_momentum
- sentiment_ma_60m_change
- sentiment_ma_60m_momentum
- sentiment_ma_240m_change
- sentiment_ma_240m_momentum
- news_volume_1h
- news_volume_1d
- news_density

## 数据质量
- 完整因子数量: 22
- 有缺失值的因子数量: 75
- 平均缺失率: 3.68%

## 建议
1. 对于高相关性的因子，可以考虑只保留其中一个，减少冗余
2. 对于缺失值较多的因子，需要进一步检查数据质量
3. 建议进行因子有效性测试，评估预测能力
4. 可以考虑对因子进行标准化处理
