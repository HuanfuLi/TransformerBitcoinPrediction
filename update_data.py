# 文件名: update_data.py (已修正TypeError)

import requests
import pandas as pd
import datetime as dt
from datetime import timedelta
import time
import os

def get_bitmex_historical_data(
        symbol="XBTUSD",
        interval="1d",
        start_time="2018-01-01",
        end_time=None,
        limit=1000,
        max_retries=5
):
    """
    从 BitMEX 获取指定时间范围内的历史K线数据。
    """
    # [核心修正] 确保所有 datetime 对象都是时区感知的 (UTC)
    if end_time is None:
        # pd.Timestamp.utcnow() 返回一个时区感知的 UTC 时间
        end_time = pd.Timestamp.utcnow()
    else:
        # 如果用户提供了字符串，解析后本地化为 UTC
        end_time = pd.to_datetime(end_time).tz_localize('UTC')

    # 将开始时间字符串也解析并本地化为 UTC
    current_start_time = pd.to_datetime(start_time).tz_localize('UTC')

    url = "https://www.bitmex.com/api/v1/trade/bucketed"
    all_data = []
    retry_count = 0

    print(f"正在从 BitMEX 获取 {symbol} 从 {current_start_time.strftime('%Y-%m-%d')} 到 {end_time.strftime('%Y-%m-%d')} 的 {interval} K线数据...")

    while current_start_time < end_time:
        params = {
            "symbol": symbol,
            "binSize": interval,
            "count": limit,
            "startTime": current_start_time.isoformat(),
            "partial": "false"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            candles = response.json()

            if not candles:
                print("\n没有更多数据了。")
                break

            all_data.extend(candles)
            
            last_timestamp_str = candles[-1]['timestamp']
            # pd.to_datetime 会自动将带'Z'的ISO字符串解析为时区感知的UTC时间
            last_timestamp = pd.to_datetime(last_timestamp_str)
            
            current_start_time = last_timestamp + pd.Timedelta(days=1)

            retry_count = 0
            print(f"已获取 {len(all_data)} 条记录... (当前获取到 {last_timestamp.strftime('%Y-%m-%d')})", end="\r")
            time.sleep(0.5)

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n错误: {str(e)}")
                print(f"已达到最大重试次数 ({max_retries})，终止获取。")
                break
            print(f"\n请求时发生错误: {str(e)}。将在 {1.5 ** retry_count:.1f} 秒后重试...")
            time.sleep(1.5 ** retry_count)

    if not all_data:
        print("未能获取到任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'timestamp': 'time'}, inplace=True)

    # [核心修正] 将 'time' 列转换为不带时区的 datetime 对象，以保持与项目其他部分的一致性
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"])
    
    # 筛选掉在指定结束日期之后的数据
    df = df[df['time'] <= end_time.tz_localize(None)]

    print(f"\n数据获取完成！共 {len(df)} 条记录 (时间范围: {df['time'].min().strftime('%Y-%m-%d')} 至 {df['time'].max().strftime('%Y-%m-%d')})")
    return df

if __name__ == "__main__":
    DATASET_DIR = 'dataset'
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    btc_df = get_bitmex_historical_data()

    if not btc_df.empty:
        output_path = os.path.join(DATASET_DIR, 'btc.csv')
        btc_df.to_csv(output_path, index=False)
        print(f"数据已成功保存到: {output_path}")
    else:
        print("无法获取新数据，请检查网络连接或API状态。")