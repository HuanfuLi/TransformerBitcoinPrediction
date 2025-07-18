import pandas as pd
import numpy as np
import os
from typing import Tuple

class FeatureEngineer:
    """
    A class for processing Bitcoin data, integrating sentiment indices, and performing
    comprehensive feature engineering.
    
    Workflow:
    1. Load Bitcoin price data and Fear & Greed Index data.
    2. Merge the two data sources.
    3. Calculate various technical indicators and derived features.
    4. Handle missing values generated during calculations.
    5. Dynamically trim invalid rows with zero values from the beginning of the dataset.
    6. Reorder feature columns by category.
    7. Save the final dataset containing all features.
    """

    def __init__(self, data_dir: str = 'dataset'):
        """
        Initializes the FeatureEngineer.

        Args:
            data_dir (str): The data directory for input and output CSV files.
        """
        self.data_dir = data_dir
        self.btc_path = os.path.join(data_dir, 'btc.csv')
        self.fg_path = os.path.join(data_dir, 'fear_greed_index.csv')
        self.output_path = os.path.join(data_dir, 'btc_features.csv')
        print(f"FeatureEngineer initialized. Input files: {self.btc_path}, {self.fg_path}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads Bitcoin price and Fear & Greed Index data from CSV files.
        
        Returns:
            A tuple containing two DataFrames: (btc_df, fg_df).
        """
        print("Step 1/7: Loading input data...")
        if not os.path.exists(self.btc_path) or not os.path.exists(self.fg_path):
            raise FileNotFoundError(
                f"Error: Missing input files. Please ensure both '{self.btc_path}' and '{self.fg_path}' exist.\n"
                "Please run `update_data.py` first to generate these files."
            )
        
        btc_df = pd.read_csv(self.btc_path, parse_dates=['time'])
        fg_df = pd.read_csv(self.fg_path, parse_dates=['date'])
        
        print(f"Successfully loaded {len(btc_df)} BTC records and {len(fg_df)} Fear & Greed Index records.")
        return btc_df, fg_df

    def merge_data(self, btc_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the Bitcoin and Fear & Greed Index data on the date column.

        Args:
            btc_df (pd.DataFrame): The Bitcoin price DataFrame.
            fg_df (pd.DataFrame): The Fear & Greed Index DataFrame.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        print("Step 2/7: Merging data sources...")
        fg_df.rename(columns={'date': 'time'}, inplace=True)
        merged_df = pd.merge(btc_df, fg_df, on='time', how='left')
        print("Data merging complete.")
        return merged_df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all derived features.

        Args:
            df (pd.DataFrame): The DataFrame to engineer features on.

        Returns:
            pd.DataFrame: DataFrame with all new features.
        """
        print("Step 3/7: Performing feature engineering calculations...")
        df = self._calculate_moving_averages(df)
        df = self._calculate_technical_indicators(df)
        df = self._calculate_sentiment_features(df)
        print("All feature calculations complete.")
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates moving averages for various windows."""
        ma_windows = [5, 10, 20, 50, 100]
        for window in ma_windows:
            df[f'MA_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all technical indicators."""
        df['price_change_pct'] = df['close'].pct_change()
        df['volume_change_pct'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_std'] = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        return df

    def _calculate_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates derived features based on the Fear & Greed Index."""
        if 'fear_greed_value' not in df.columns: return df
        df['fear_greed_normalized'] = df['fear_greed_value'] / 100.0
        df['fear_greed_change'] = df['fear_greed_value'].diff()
        for window in [3, 7, 14, 30]:
            df[f'fear_greed_ma_{window}'] = df['fear_greed_value'].rolling(window=window, min_periods=1).mean()
        df['fear_greed_volatility_7d'] = df['fear_greed_value'].rolling(window=7, min_periods=1).std()
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in the data.

        Args:
            df (pd.DataFrame): The DataFrame with missing values.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        print("Step 4/7: Handling missing values...")
        # Forward fill sentiment-related columns first
        cols_to_fill = [col for col in df.columns if 'fear_greed' in col]
        df[cols_to_fill] = df[cols_to_fill].ffill()
        # Forward fill all other columns
        df.ffill(inplace=True)
        # Fill any remaining NaNs with 0
        df.fillna(0, inplace=True)
        print("Missing value handling complete.")
        return df

    def trim_initial_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [New Feature] Trims all rows containing zero values from the start of the DataFrame
        until the first fully non-zero row is found. This is mainly to remove invalid
        initial data caused by moving averages, delayed sentiment data, etc.

        Args:
            df (pd.DataFrame): The DataFrame to be trimmed.

        Returns:
            pd.DataFrame: The DataFrame after trimming invalid header rows.
        """
        print("Step 5/7: Trimming dataset to remove initial zero-value rows...")
        
        # We only care about derived feature columns; base price and volume can be 0.
        # Here, 'BB_upper' and 'fear_greed_value' are used as key check columns
        # because one is a long-period technical indicator and the other is an external data source.
        key_check_cols = ['BB_upper', 'fear_greed_value']
        
        first_valid_index = 0
        for col in key_check_cols:
            if col in df.columns:
                # Find the index of the first non-zero value in this column
                first_nonzero_index_for_col = (df[col].dropna() != 0).idxmax()
                # Update the starting index we need to keep, taking the max of all key column indices
                first_valid_index = max(first_valid_index, first_nonzero_index_for_col)
        
        if first_valid_index > 0:
            original_rows = len(df)
            trimmed_df = df.loc[first_valid_index:].reset_index(drop=True)
            print(f"Data trimming complete. Keeping data from index {first_valid_index}, removed {original_rows - len(trimmed_df)} rows.")
            print(f"New start date is: {trimmed_df['time'].iloc[0].strftime('%Y-%m-%d')}")
            return trimmed_df
        else:
            print("No initial zero-value rows found to trim, returning original data.")
            return df

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the DataFrame columns by category.

        Args:
            df (pd.DataFrame): The DataFrame with unordered columns.

        Returns:
            pd.DataFrame: The DataFrame with reordered columns.
        """
        print("Step 6/7: Reordering feature columns by category...")
        base_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        ma_cols = sorted([col for col in df.columns if col.startswith('MA_')], key=lambda x: int(x.split('_')[1]))
        indicator_cols = sorted([col for col in df.columns if col not in base_cols and col not in ma_cols and 'fear_greed' not in col])
        sentiment_cols = sorted([col for col in df.columns if 'fear_greed' in col])
        
        final_order = base_cols + ma_cols + indicator_cols + sentiment_cols
        existing_cols = [col for col in final_order if col in df.columns]
        
        return df[existing_cols]

    def save_features(self, df: pd.DataFrame):
        """
        Saves the final feature DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The final DataFrame to save.
        """
        print(f"Step 7/7: Saving final feature file to {self.output_path}...")
        df.to_csv(self.output_path, index=False)
        print("="*50)
        print(f"Success! The final feature file has been saved.")
        print(f"Total {len(df)} rows, {len(df.columns)} features.")
        print("="*50)

    def run(self):
        """
        Executes the complete feature engineering pipeline.
        """
        try:
            btc_df, fg_df = self.load_data()
            merged_df = self.merge_data(btc_df, fg_df)
            features_df = self.calculate_features(merged_df)
            cleaned_df = self.handle_missing_values(features_df)
            # [New] Call the trimming function
            trimmed_df = self.trim_initial_zeros(cleaned_df)
            ordered_df = self.reorder_columns(trimmed_df)
            self.save_features(ordered_df)
        except Exception as e:
            print(f"\nA critical error occurred during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    feature_engineer = FeatureEngineer()
    feature_engineer.run()