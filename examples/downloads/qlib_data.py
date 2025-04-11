import ccxt
import requests

# 配置socks5h代理会话
session = requests.Session()
session.proxies = {
    'http': 'socks5h://127.0.0.1:1080',  # 关键：使用socks5h协议
    'https': 'socks5h://127.0.0.1:1080'
}

# 将会话传递给CCXT
exchange = ccxt.binance({
    'session': session,  # 注入自定义会话
    'enableRateLimit': True
})

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

# 测试请求（DNS由代理服务器解析）
ticker = exchange.fetch_ticker('BTC/USDT')
print(ticker)
print(get_all_symbols(exchange, 'USDT'))