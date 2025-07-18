import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import traceback
import warnings

# 忽略警告以保持输出干净
warnings.filterwarnings("ignore")

class FeatureSelector:
    """
    一个用于分析比特币特征数据、计算特征相关性和重要性，并建议最佳特征的类。
    
    工作流程:
    1. 加载包含所有特征的CSV文件。
    2. 处理缺失值和异常。
    3. 计算Pearson相关系数。
    4. 使用随机森林计算特征重要性。
    5. 基于相关性和重要性建议特征。
    6. 在终端打印结果。
    """

    def __init__(self, data_path: str = 'dataset/btc_features.csv', target_col: str = 'close', top_n: int = 10):
        """
        初始化FeatureSelector。

        Args:
            data_path (str): 特征数据的CSV路径。
            target_col (str): 目标列名，通常为'close'。
            top_n (int): 建议的前N个特征。
        """
        self.data_path = data_path
        self.target_col = target_col
        self.top_n = top_n
        self.df = None
        print("特征选择器已初始化。")

    def load_data(self) -> bool:
        """
        加载数据并进行基本验证。

        Returns:
            bool: 加载是否成功。
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"错误: 文件 '{self.data_path}' 不存在。请先运行 feature_engineer.py 生成它。")
            
            self.df = pd.read_csv(self.data_path, parse_dates=['time'])
            if self.df.empty:
                raise ValueError("错误: 加载的数据为空。")
            
            if self.target_col not in self.df.columns:
                raise ValueError(f"错误: 目标列 '{self.target_col}' 不存在于数据中。")
            
            # 移除非数值列（除了'time'）
            self.df = self.df.select_dtypes(include=[np.number])
            if self.df.shape[1] < 2:
                raise ValueError("错误: 数据中没有足够的数值特征。")
            
            print(f"成功加载 {len(self.df)} 行数据，包含 {len(self.df.columns)} 个数值列。")
            return True
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            traceback.print_exc()
            return False

    def handle_missing_values(self):
        """
        处理缺失值和无限值。
        """
        try:
            # 替换无限值为NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # 前向填充缺失值
            self.df.ffill(inplace=True)
            # 剩余缺失值填充为0
            self.df.fillna(0, inplace=True)
            print("缺失值和异常值处理完成。")
        except Exception as e:
            print(f"处理缺失值时发生错误: {e}")
            traceback.print_exc()

    def calculate_correlations(self) -> pd.Series:
        """
        计算每个特征与目标的相关系数。

        Returns:
            pd.Series: 特征的相关系数（绝对值排序）。
        """
        try:
            corr = self.df.corr()[self.target_col].drop(self.target_col)
            corr_abs = corr.abs().sort_values(ascending=False)
            print("Pearson相关系数计算完成。")
            return corr_abs
        except Exception as e:
            print(f"计算相关系数时发生错误: {e}")
            traceback.print_exc()
            return pd.Series()

    def calculate_feature_importances(self) -> pd.Series:
        """
        使用随机森林计算特征重要性。

        Returns:
            pd.Series: 特征重要性（排序）。
        """
        try:
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            # 拆分数据以避免过拟合
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            
            # 计算MSE以验证模型
            y_pred = rf.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print(f"随机森林验证MSE: {mse:.4f}")
            
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("特征重要性计算完成。")
            return importances
        except Exception as e:
            print(f"计算特征重要性时发生错误: {e}")
            traceback.print_exc()
            return pd.Series()

    def suggest_features(self, corr: pd.Series, imp: pd.Series) -> list:
        """
        基于相关性和重要性建议特征。

        Args:
            corr (pd.Series): 相关系数。
            imp (pd.Series): 重要性。

        Returns:
            list: 建议的特征列表。
        """
        try:
            # (existing combined score logic)
            if not corr.empty and not imp.empty:
                combined = 0.4 * corr / corr.max() + 0.6 * imp / imp.max()
                combined = combined.sort_values(ascending=False)
                suggested = combined.head(self.top_n).index.tolist()
            # ... (existing elifs)
            
            if not suggested:
                raise ValueError("无法生成建议特征。")
            
            # Always include 'close' as first (essential for time-series)
            if 'close' not in suggested:
                suggested = ['close'] + suggested
            else:
                suggested = ['close'] + [f for f in suggested if f != 'close']
            
            print("\n建议在实际预测中使用的特征（基于相关性和重要性）：")
            for i, feat in enumerate(suggested, 1):
                print(f"{i}. {feat}")
            
            # Save to YAML for optimizer to read
            config_dir = 'config'
            os.makedirs(config_dir, exist_ok=True)
            save_path = os.path.join(config_dir, 'suggested_features.yaml')
            with open(save_path, 'w') as f:
                yaml.dump({'features_to_use': suggested}, f)
            print(f"\n建议特征已保存到: {save_path}")
            
            return suggested
        except Exception as e:
            print(f"生成建议时发生错误: {e}")
            traceback.print_exc()
            return []

    def run(self):
        """
        执行完整的特征选择流程。
        """
        if not self.load_data():
            return
        
        self.handle_missing_values()
        
        corr = self.calculate_correlations()
        if not corr.empty:
            print("\nTop 10 相关系数（绝对值）：")
            print(corr.head(10))
        
        imp = self.calculate_feature_importances()
        if not imp.empty:
            print("\nTop 10 特征重要性：")
            print(imp.head(10))
        
        self.suggest_features(corr, imp)

if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()