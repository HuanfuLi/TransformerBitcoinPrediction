import requests
import pandas as pd
import datetime as dt
import time
import os
import json
import logging
from pathlib import Path
from typing import Optional

# ===================================================================
# Part 1: Bitcoin Price Data Fetching (from BitMEX)
# ===================================================================

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
    if end_time is None:
        end_time = pd.Timestamp.utcnow()
    else:
        end_time = pd.to_datetime(end_time).tz_localize('UTC')

    current_start_time = pd.to_datetime(start_time).tz_localize('UTC')
    url = "https://www.bitmex.com/api/v1/trade/bucketed"
    all_data = []
    retry_count = 0

    print(f"正在从 BitMEX 获取 {symbol} 从 {current_start_time.strftime('%Y-%m-%d')} 到 {end_time.strftime('%Y-%m-%d')} 的 {interval} K线数据...")

    while current_start_time < end_time:
        params = {"symbol": symbol, "binSize": interval, "count": limit, "startTime": current_start_time.isoformat(), "partial": "false"}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            candles = response.json()

            if not candles:
                print("\n没有更多数据了。")
                break

            all_data.extend(candles)
            last_timestamp = pd.to_datetime(candles[-1]['timestamp'])
            current_start_time = last_timestamp + pd.Timedelta(days=1)
            retry_count = 0
            print(f"已获取 {len(all_data)} 条记录... (当前获取到 {last_timestamp.strftime('%Y-%m-%d')})", end="\r")
            time.sleep(0.5)

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n错误: {str(e)} - 已达到最大重试次数，终止获取。")
                break
            print(f"\n请求时发生错误: {str(e)}。将在 {1.5 ** retry_count:.1f} 秒后重试...")
            time.sleep(1.5 ** retry_count)

    if not all_data:
        print("未能获取到任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'timestamp': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"])
    df = df[df['time'] <= end_time.tz_localize(None)]

    print(f"\n数据获取完成！共 {len(df)} 条记录 (时间范围: {df['time'].min().strftime('%Y-%m-%d')} 至 {df['time'].max().strftime('%Y-%m-%d')})")
    return df

# ===================================================================
# Part 2: Fear & Greed Index Data Fetching
# ===================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fear_greed_data(api_url="https://api.alternative.me/fng/", limit=0) -> Optional[pd.DataFrame]:
    """
    从 Alternative.me API 获取 Fear & Greed Index 数据。
    """
    try:
        # 移除 date_format=cn 参数，直接处理时间戳更可靠
        url = f"{api_url}?limit={limit}"
        print("正在从 alternative.me 获取 Fear & Greed Index 数据...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get('data', [])

        if not data:
            logger.error("API响应中未找到有效数据。")
            return None

        df = pd.DataFrame(data)

        # [核心修正] 从 'timestamp' 列创建 'date' 列
        # API返回的时间戳是字符串格式的秒级Unix时间戳
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        # 将其转换为datetime对象，并只取日期部分
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        # 再次转换为datetime64[ns]类型，以确保与btc.csv中的'time'列类型一致
        df['date'] = pd.to_datetime(df['date'])

        # 重命名其他列
        df.rename(columns={'value': 'fear_greed_value', 'value_classification': 'fear_greed_classification'}, inplace=True)
        df['fear_greed_value'] = pd.to_numeric(df['fear_greed_value'])
        
        # 选择并排序最终的列
        df = df[['date', 'fear_greed_value', 'fear_greed_classification']]
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Fear & Greed Index 获取完成！共 {len(df)} 条记录 (时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')})")
        return df

    except Exception as e:
        logger.error(f"获取 Fear & Greed Index 数据时发生错误: {e}")
        return None

# ===================================================================
# Part 3: Main Execution Block
# ===================================================================

if __name__ == "__main__":
    DATASET_DIR = 'dataset'
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # --- Step 1: Updating Bitcoin (BTC) Price Data ---
    print("--- Step 1: Updating Bitcoin (BTC) Price Data ---")
    btc_df = get_bitmex_historical_data()
    if btc_df is not None and not btc_df.empty:
        output_path = os.path.join(DATASET_DIR, 'btc.csv')
        btc_df.to_csv(output_path, index=False)
        print(f"Bitcoin data successfully saved to: {output_path}")
    else:
        print("Could not fetch new Bitcoin data.")

    print("\n" + "="*50 + "\n")

    # --- Step 2: Updating Fear & Greed Index Data ---
    print("--- Step 2: Updating Fear & Greed Index Data ---")
    fg_df = get_fear_greed_data()
    if fg_df is not None and not fg_df.empty:
        output_path = os.path.join(DATASET_DIR, 'fear_greed_index.csv')
        fg_df.to_csv(output_path, index=False)
        print(f"Fear & Greed Index data successfully saved to: {output_path}")
    else:
        print("Could not fetch new Fear & Greed Index data.")