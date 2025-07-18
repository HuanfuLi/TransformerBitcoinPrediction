import pandas as pd
import numpy as np
import os
from typing import Tuple

class FeatureEngineer:
    """
    一个用于处理比特币数据、整合情绪指数并进行全面特征工程的类。
    
    工作流程:
    1. 加载比特币价格数据和恐惧贪婪指数数据。
    2. 合并两个数据源。
    3. 计算各类技术指标和衍生特征。
    4. 处理因计算产生的缺失值。
    5. 动态裁切掉数据头部的无效0值行。
    6. 按类别重新排序特征列。
    7. 保存包含所有特征的最终数据集。
    """

    def __init__(self, data_dir: str = 'dataset'):
        """
        初始化FeatureEngineer。

        Args:
            data_dir (str): 存放输入和输出CSV文件的数据目录。
        """
        self.data_dir = data_dir
        self.btc_path = os.path.join(data_dir, 'btc.csv')
        self.fg_path = os.path.join(data_dir, 'fear_greed_index.csv')
        self.output_path = os.path.join(data_dir, 'btc_features.csv')
        print(f"数据处理器已初始化。输入文件: {self.btc_path}, {self.fg_path}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        从CSV文件加载比特币价格和恐惧贪婪指数数据。
        """
        print("步骤 1/7: 正在加载输入数据...")
        if not os.path.exists(self.btc_path) or not os.path.exists(self.fg_path):
            raise FileNotFoundError(
                f"错误: 缺少输入文件。请确保 '{self.btc_path}' 和 '{self.fg_path}' 都存在。\n"
                "请先运行 `update_data.py` 来生成这些文件。"
            )
        
        btc_df = pd.read_csv(self.btc_path, parse_dates=['time'])
        fg_df = pd.read_csv(self.fg_path, parse_dates=['date'])
        
        print(f"成功加载 {len(btc_df)} 条BTC数据和 {len(fg_df)} 条恐惧贪婪指数数据。")
        return btc_df, fg_df

    def merge_data(self, btc_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
        """
        将比特币数据和恐惧贪婪指数数据按日期合并。
        """
        print("步骤 2/7: 正在合并数据源...")
        fg_df.rename(columns={'date': 'time'}, inplace=True)
        merged_df = pd.merge(btc_df, fg_df, on='time', how='left')
        print("数据合并完成。")
        return merged_df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有衍生特征。
        """
        print("步骤 3/7: 正在进行特征工程计算...")
        df = self._calculate_moving_averages(df)
        df = self._calculate_technical_indicators(df)
        df = self._calculate_sentiment_features(df)
        print("所有特征计算完成。")
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算不同周期的移动平均线。"""
        ma_windows = [5, 10, 20, 50, 100]
        for window in ma_windows:
            df[f'MA_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标。"""
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
        """计算基于恐惧贪婪指数的衍生特征。"""
        if 'fear_greed_value' not in df.columns: return df
        df['fear_greed_normalized'] = df['fear_greed_value'] / 100.0
        df['fear_greed_change'] = df['fear_greed_value'].diff()
        for window in [3, 7, 14, 30]:
            df[f'fear_greed_ma_{window}'] = df['fear_greed_value'].rolling(window=window, min_periods=1).mean()
        df['fear_greed_volatility_7d'] = df['fear_greed_value'].rolling(window=7, min_periods=1).std()
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理数据中的缺失值。
        """
        print("步骤 4/7: 正在处理缺失值...")
        cols_to_fill = [col for col in df.columns if 'fear_greed' in col]
        df[cols_to_fill] = df[cols_to_fill].ffill()
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        print("缺失值处理完成。")
        return df

    def trim_initial_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [新增功能] 从DataFrame的开头裁切掉所有包含0值的行，直到找到第一个完全非0的行。
        这主要用于移除因移动平均线、情绪指数数据延迟等原因造成的无效初始数据。

        Args:
            df (pd.DataFrame): 待裁切的DataFrame。

        Returns:
            pd.DataFrame: 裁切掉头部无效行后的DataFrame。
        """
        print("步骤 5/7: 正在裁切数据集以移除初始的0值行...")
        
        # 我们只关心衍生特征列，基础价格和交易量本身可能为0
        # 这里以 'BB_upper' 和 'fear_greed_value' 作为关键检查列
        # 因为它们一个是计算周期最长的技术指标之一，一个是外部数据源
        key_check_cols = ['BB_upper', 'fear_greed_value']
        
        first_valid_index = 0
        for col in key_check_cols:
            if col in df.columns:
                # 找到该列第一个非零值的索引
                first_nonzero_index_for_col = (df[col] != 0).idxmax()
                # 更新我们需要保留的起始索引，取所有关键列中非零索引的最大值
                first_valid_index = max(first_valid_index, first_nonzero_index_for_col)
        
        if first_valid_index > 0:
            original_rows = len(df)
            trimmed_df = df.loc[first_valid_index:].reset_index(drop=True)
            print(f"数据裁切完成。从索引 {first_valid_index} 开始保留数据，移除了 {original_rows - len(trimmed_df)} 行。")
            print(f"新的起始日期为: {trimmed_df['time'].iloc[0].strftime('%Y-%m-%d')}")
            return trimmed_df
        else:
            print("未发现需要裁切的初始0值行，返回原始数据。")
            return df

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按类别对DataFrame的列进行排序。
        """
        print("步骤 6/7: 正在按类别重新排序特征列...")
        base_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        ma_cols = sorted([col for col in df.columns if col.startswith('MA_')], key=lambda x: int(x.split('_')[1]))
        indicator_cols = sorted([col for col in df.columns if col not in base_cols and col not in ma_cols and 'fear_greed' not in col])
        sentiment_cols = sorted([col for col in df.columns if 'fear_greed' in col])
        final_order = base_cols + ma_cols + indicator_cols + sentiment_cols
        existing_cols = [col for col in final_order if col in df.columns]
        return df[existing_cols]

    def save_features(self, df: pd.DataFrame):
        """
        将最终的特征DataFrame保存到CSV文件。
        """
        print(f"步骤 7/7: 正在保存最终特征文件到 {self.output_path}...")
        df.to_csv(self.output_path, index=False)
        print("="*50)
        print(f"成功！最终特征文件已保存。")
        print(f"总计 {len(df)} 行, {len(df.columns)} 个特征。")
        print("="*50)

    def run(self):
        """
        执行完整的特征工程流水线。
        """
        try:
            btc_df, fg_df = self.load_data()
            merged_df = self.merge_data(btc_df, fg_df)
            features_df = self.calculate_features(merged_df)
            cleaned_df = self.handle_missing_values(features_df)
            # [新增] 调用裁切功能
            trimmed_df = self.trim_initial_zeros(cleaned_df)
            ordered_df = self.reorder_columns(trimmed_df)
            self.save_features(ordered_df)
        except Exception as e:
            print(f"\n在执行过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    feature_engineer = FeatureEngineer()
    feature_engineer.run()