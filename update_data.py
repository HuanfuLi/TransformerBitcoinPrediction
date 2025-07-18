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
) -> pd.DataFrame:
    """
    Fetches historical candlestick data from BitMEX for a specified time range.
    
    Args:
        symbol (str): The trading symbol (e.g., "XBTUSD").
        interval (str): The candlestick interval (e.g., "1d").
        start_time (str): The start date in "YYYY-MM-DD" format.
        end_time (str, optional): The end date. Defaults to the current UTC time.
        limit (int): The number of records to fetch per API call.
        max_retries (int): The maximum number of retries for a failed request.

    Returns:
        pd.DataFrame: A DataFrame containing the historical price data.
    """
    if end_time is None:
        end_time = pd.Timestamp.utcnow()
    else:
        end_time = pd.to_datetime(end_time).tz_localize('UTC')

    current_start_time = pd.to_datetime(start_time).tz_localize('UTC')
    url = "https://www.bitmex.com/api/v1/trade/bucketed"
    all_data = []
    retry_count = 0

    print(f"Fetching {interval} candlestick data for {symbol} from {current_start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} from BitMEX...")

    while current_start_time < end_time:
        params = {"symbol": symbol, "binSize": interval, "count": limit, "startTime": current_start_time.isoformat(), "partial": "false"}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            candles = response.json()

            if not candles:
                print("\nNo more data available.")
                break

            all_data.extend(candles)
            last_timestamp = pd.to_datetime(candles[-1]['timestamp'])
            current_start_time = last_timestamp + pd.Timedelta(minutes=1) # Move to the next interval start
            retry_count = 0
            print(f"Fetched {len(all_data)} records... (Current date: {last_timestamp.strftime('%Y-%m-%d')})", end="\r")
            time.sleep(0.5) # Respect API rate limits

        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\nError: {str(e)} - Maximum retries reached, aborting fetch.")
                break
            print(f"\nAn error occurred during the request: {str(e)}. Retrying in {1.5 ** retry_count:.1f} seconds...")
            time.sleep(1.5 ** retry_count)

    if not all_data:
        print("Failed to fetch any data.")
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

    print(f"\nData fetching complete! Total {len(df)} records (Date Range: {df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')})")
    return df

# ===================================================================
# Part 2: Fear & Greed Index Data Fetching
# ===================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fear_greed_data(api_url="https://api.alternative.me/fng/", limit=0) -> Optional[pd.DataFrame]:
    """
    Fetches the Fear & Greed Index data from the Alternative.me API.

    Args:
        api_url (str): The API endpoint URL.
        limit (int): The number of results to return (0 for all).

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the F&G data, or None on failure.
    """
    try:
        url = f"{api_url}?limit={limit}"
        print("Fetching Fear & Greed Index data from alternative.me...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get('data', [])

        if not data:
            logger.error("No valid data found in the API response.")
            return None

        df = pd.DataFrame(data)

        # [Core Fix] Create 'date' column from 'timestamp'
        # The API returns the timestamp as a string of seconds (Unix time)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        # Convert it to datetime objects and take only the date part
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        # Convert again to datetime64[ns] to ensure type consistency with btc.csv's 'time' column
        df['date'] = pd.to_datetime(df['date'])

        # Rename other columns
        df.rename(columns={'value': 'fear_greed_value', 'value_classification': 'fear_greed_classification'}, inplace=True)
        df['fear_greed_value'] = pd.to_numeric(df['fear_greed_value'])
        
        # Select and sort the final columns
        df = df[['date', 'fear_greed_value', 'fear_greed_classification']]
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Fear & Greed Index fetching complete! Total {len(df)} records (Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
        return df

    except Exception as e:
        logger.error(f"An error occurred while fetching Fear & Greed Index data: {e}")
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