# 文件路径: crypto_orderbook_analysis.py

import ccxt
import numpy as np
from collections import defaultdict

def get_all_symbols(exchange, quote_currency=None, active_only=True):
    """获取指定交易所的交易对列表"""
    try:
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        
        if active_only:
            symbols = [s for s in symbols if markets[s].get('active', True)]
        if quote_currency:
            symbols = [s for s in symbols if s.endswith(f'/{quote_currency}')]
            
        symbols.sort()
        return symbols
    except Exception as e:
        print(f"获取交易对失败: {e}")
        return []


def get_top_volumes(exchange, quote_currency="USDT", N = 500):
    # 获取所有交易对
    markets = get_all_symbols(exchange, quote_currency)
    symbol_stats = defaultdict(dict)
    print(f"load markets : {len(markets)}")
    i = 0
    for symbol in markets:
        try:
            # 获取订单簿（默认返回前100档）
            orderbook = exchange.fetch_order_book(symbol, limit=N)
            #print(f"load {symbol} orderbook", orderbook)
            # 计算price（bid和ask的平均值）
            price = (orderbook['bids'][0][0] + orderbook['asks'][0][0])/2
            bid_volumes = np.sum([bid[1] * bid[0] for bid in orderbook['bids'][:N]])
            
            # 计算卖盘总量（asks）
            ask_volumes = np.sum([ask[1] for ask in orderbook['asks'][:N]]) * price
            
            #美金计价的买卖盘总量
            symbol_stats[symbol] = {
                'bid_total': bid_volumes,
                'ask_total': ask_volumes,
                'rate' : bid_volumes / ask_volumes,
            }
            i += 1
            if i % 10 == 0:
                print(f"已处理 {i} 个交易对")
            
        except Exception as e:
            print(f"获取{symbol}数据失败: {str(e)}")
    
    return symbol_stats

def analyze_volumes(stats):
    """执行统计分析"""
    bid_totals = [v['bid_total'] for v in stats.values()]
    ask_totals = [v['ask_total'] for v in stats.values()]
    
    return {
        'all_pairs': {
            'total_bid': np.sum(bid_totals),
            'total_ask': np.sum(ask_totals),
            'avg_bid': np.mean(bid_totals),
            'avg_ask': np.mean(ask_totals),
            'max_bid': np.max(bid_totals),
            'max_ask': np.max(ask_totals)
        },
        'bid_pairs': sorted(
            stats.items(), 
            key=lambda x: x[1]['bid_total'], 
            reverse=True
        ),
        'ask_pairs': sorted(
            stats.items(), 
            key=lambda x: x[1]['ask_total'], 
            reverse=True
        ),
        'rate_pairs': sorted(
            stats.items(), 
            key=lambda x: x[1]['rate'], 
            reverse=True
        )
    }

# 使用示例
if __name__ == "__main__":
    # 获取币安数据
    """获取指定交易所前100档位买卖盘总量"""
    # 初始化交易所
    exchange_name = 'binance'
    exchange = getattr(ccxt, exchange_name)({
        'enableRateLimit': True,
    })
    
    binance_stats = get_top_volumes(exchange)
    analysis = analyze_volumes(binance_stats)
    
    #打印 top20 买盘总量 和 货币对
    print("top20 买盘总量")
    for i in range(20):
        print(f"{analysis['bid_pairs'][i][0]}: {analysis['bid_pairs'][i][1]['bid_total']}")
    
    #打印 top20 卖盘总量 和 货币对
    print("top20 卖盘总量")
    for i in range(20):
        print(f"{analysis['ask_pairs'][i][0]}: {analysis['ask_pairs'][i][1]['ask_total']}")

    #打印 top20 卖盘总量 和 货币对
    print("top20 买盘/卖盘 比例")
    for i in range(20):
        print(f"{analysis['rate_pairs'][i][0]}: {analysis['rate_pairs'][i][1]['rate']}")
    
    # 打印汇总信息
    print("汇总信息")
    print(f"总的买盘总量: {analysis['all_pairs']['total_bid']}")
    print(f"总的卖盘总量: {analysis['all_pairs']['total_ask']}")
    print(f"平均买盘总量: {analysis['all_pairs']['avg_bid']}")
    print(f"平均卖盘总量: {analysis['all_pairs']['avg_ask']}")
    print(f"最大买盘总量: {analysis['all_pairs']['max_bid']}")
    print(f"最大卖盘总量: {analysis['all_pairs']['max_ask']}")
    
