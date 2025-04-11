import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import concurrent.futures
from queue import Queue
import time
import pyarrow as pa
import pyarrow.parquet as pq
import argparse

# 第一部分：获取交易对列表
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

def csv_path(exchange, symbol, timeframe) -> str:
    normalized_symbol = symbol.replace('/', '-')
    return os.path.join(SAVE_DIR, f"{exchange}/{timeframe}/{normalized_symbol}.csv")
# 第二部分：数据维护核心功能
def maintain_ohlcv_data(exchange, symbol, timeframe, days=365):
    """维护OHLCV数据的核心函数"""
    exchange_id = exchange.id
    root_path = parquet_root_path(exchange_id, timeframe)
    normalized_symbol = symbol.replace('/', '-')
    parquet_path = os.path.join(
        root_path,
        f'symbol={normalized_symbol}'
    )
    #csv file path
    filename = csv_path(exchange_id, symbol, timeframe)
    try:
        if os.path.exists(parquet_path):
            return update_existing_data(exchange, parquet_path, symbol, timeframe)
        elif os.path.exists(filename):
            # 更新模式
            return update_existing_data(exchange, filename, symbol, timeframe)
        else:
            # 全新下载模式
            return download_new_data(exchange, symbol, timeframe, days)
    except Exception as e:
        print(f"处理 {symbol} 时发生错误: {str(e)}")

def read_table(filename):
    """读取数据文件"""
    #判断是目录还是文件
    if os.path.isdir(filename):
        table = pq.read_table(filename)
        df = table.to_pandas()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df
    else:
        df = pd.read_csv(filename, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        return df

def update_existing_data(exchange, filename, symbol, timeframe):
    """更新现有数据文件"""
    print(f"\n▶ 开始更新 {symbol} 数据")
    df_existing = read_table(filename)
    last_timestamp = int(df_existing.index[-1].timestamp() * 1000)
    
    new_data = []
    since = last_timestamp + 1000 * 60  # 从下个分钟开始
    end_time = exchange.milliseconds()
    needretry = False
    while since < end_time:
        try:
            # 每次获取1500根K线（交易所通常允许的最大值）
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1500)
            if not ohlcv:
                break
                
            new_data += ohlcv
            since = ohlcv[-1][0] + 1
            print(f"已获取 {symbol} {len(ohlcv)} 条新数据，最新时间: {exchange.iso8601(ohlcv[-1][0])}")
            
        except Exception as e:
            print(f"获取数据中断: {str(e)}")
            needretry = True
            break

    if new_data:
        # 处理新数据
        df_new = process_dataframe(new_data)
        df_combined = pd.concat([df_existing, df_new]).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        csv_file = csv_path(exchange.id, symbol, timeframe)
        df_combined.to_csv(csv_file)
        save_as_parquet(exchange.id, symbol, timeframe, df_combined)
        print(f"√ 成功更新 {symbol}，新增 {len(df_new)} 条记录")
        return (df_combined, needretry)
    else:
        print(f"√ {symbol} 数据已经是最新")
        return (df_existing, needretry)

def download_new_data(exchange, symbol, timeframe, days):
    """下载全新数据集"""
    print(f"\n▶ 开始下载 {symbol} 历史数据")
    
    all_ohlcv = []
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    needretry = False
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1500)
            if not ohlcv:
                break
                
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            #print(f"已下载 {len(ohlcv)} 条数据，时间: {exchange.iso8601(ohlcv[-1][0])}")
            
            # 达到当前时间则停止
            if ohlcv[-1][0] > exchange.milliseconds():
                break
                
        except Exception as e:
            print(f"下载中断: {str(e)}")
            needretry = True
            break

    if all_ohlcv:
        df = process_dataframe(all_ohlcv)
        csv_file = csv_path(exchange.id, symbol, timeframe)
        df.to_csv(csv_file)
        save_as_parquet(exchange.id, symbol, timeframe, df)
        print(f"√ 成功下载 {symbol}，共 {len(df)} 条记录")
        return (df, needretry)
    else:
        print(f"× 无法获取 {symbol} 数据")
        return (None, needretry)

def process_dataframe(ohlcv):
    """统一处理数据框格式"""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df[~df.index.duplicated(keep='last')].sort_index()

def threaded_download():
    symbols = get_all_symbols(getattr(ccxt, EXCHANGE)(), QUOTE_CURRENCY)
    total = len(symbols)
    
    # 使用队列控制请求节奏
    task_queue = Queue()
    for idx, symbol in enumerate(symbols, 1):
        task_queue.put((symbol, total, idx))
    
    # 带动态间隔的线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        last_time = time.time()
        
        while not task_queue.empty():
            # 控制请求间隔
            elapsed = time.time() - last_time
            if elapsed < REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL - elapsed)
            
            symbol, total, idx = task_queue.get()
            future = executor.submit(worker, symbol, total, idx)
            futures.append(future)
            last_time = time.time()
        
        # 等待所有任务完成
        success_count = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                success_count += 1
    
    print(f"\n处理完成! 成功 {success_count}/{total} 个交易对")
def worker(symbol, total, index):
    """带重试机制的下载工作线程"""
    retry_count = 0
    success = False
    
    while retry_count < MAX_RETRIES and not success:
        # 每次重试创建新交易所实例
        exchange = getattr(ccxt, EXCHANGE)(exchange_config)
        
        try:
            # 带时间戳的进度提示
            ts = datetime.now().strftime("%H:%M:%S")
            status_msg = "重试" if retry_count > 0 else "开始"
            print(f"[{ts}] {status_msg}处理 ({index}/{total}) {symbol} 第{retry_count+1}次尝试")
            
            # 调用核心逻辑并获取重试标志
            _, needretry = maintain_ohlcv_data(exchange, symbol, TIMEFRAME, HISTORICAL_DAYS)
            
            if not needretry:
                success = True
                print(f"[{ts}] √ 成功完成 ({index}/{total}) {symbol}")
                return symbol
            else:
                retry_count += 1
                print(f"[{ts}] ! 需要重试 ({index}/{total}) {symbol}")
                
        except Exception as e:
            retry_count += 1
            print(f"[{ts}] × 尝试失败 ({index}/{total}) {symbol}: {str(e)}")
            
        finally:
            exchange.session.close()
            time.sleep(2)  # 指数退避等待
    
    # 重试耗尽处理
    print(f"[ERROR] 无法完成 ({index}/{total}) {symbol}，已达最大重试次数")
    return None

def parquet_root_path(exchange_id, timeframe):
    """获取Parquet数据根目录"""
    return os.path.join(f'{SAVE_DIR}/parquet_data', exchange_id, timeframe)
def save_as_parquet(exchange_name, symbol, timeframe, data):
    """存储为Parquet格式"""
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['symbol'] = symbol.replace('/', '-')  # 分区友好格式
    
    # 写入分区数据集
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=parquet_root_path(exchange_name, timeframe),
        partition_cols=['symbol'],
        compression='snappy',
        existing_data_behavior='delete_matching'
    )

# 第三部分：主程序
if __name__ == "__main__":
    # 配置参数
    EXCHANGE = 'binance'
    TIMEFRAME = '1m'
    QUOTE_CURRENCY = 'USDT'
    HISTORICAL_DAYS = 365
    SAVE_DIR = "data"
    MAX_WORKERS = 10
    REQUEST_INTERVAL = 0.2  # 每个请求间隔（秒）
    MAX_RETRIES = 3  # 最大重试次数

    parser = argparse.ArgumentParser(description='交易所历史数据下载工具')
    parser.add_argument('--update', 
                       type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                       default=True,
                       help='是否执行数据更新（默认True）')
    parser.add_argument("--exchange", 
                        type=str, 
                        default="binance",
                        help="交易所名称 (例如: binance, okex, bybit)")
    parser.add_argument("--proxy", 
                        type=str, 
                        default=None,
                        help="代理地址 (例如: http://127.0.0.1:7890)")
    parser.add_argument("--timeframe", 
                        type=str, 
                        default="1m",
                        help="K线时间间隔 (例如: 1m, 5m, 1h, 4h, 1d)")
    args = parser.parse_args()
    EXCHANGE = args.exchange
    TIMEFRAME = args.timeframe
    # 交易所配置字典
    exchange_config = {
        'enableRateLimit': True,
        'rateLimit': 3000,  # 默认3秒间隔
    }

    # 代理配置
    if args.proxy:
        exchange_config.update({
            'proxies': {
                'http': args.proxy,
                'https': args.proxy,
                'ws': args.proxy  # WebSocket代理
            }
        })
    # 检查保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(os.path.join(SAVE_DIR, f"{EXCHANGE}/{TIMEFRAME}")):
        os.makedirs(os.path.join(SAVE_DIR, f"{EXCHANGE}/{TIMEFRAME}"))
    root_path=parquet_root_path(EXCHANGE, TIMEFRAME)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if args.update:
        threaded_download()
        print("\n所有交易对数据处理完成！")
    else:
        print("跳过数据更新")
