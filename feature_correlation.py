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

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

class FeatureSelector:
    """
    A class to analyze Bitcoin feature data, calculate feature correlation and importance,
    and recommend the best features for the model.
    
    Workflow:
    1. Load the CSV file containing all features.
    2. Handle missing values and anomalies.
    3. Calculate the Pearson correlation coefficient.
    4. Calculate feature importance using a Random Forest model.
    5. Recommend features based on a combination of correlation and importance.
    6. Print the results to the terminal.
    """

    def __init__(self, data_path: str = 'dataset/btc_features.csv', target_col: str = 'close', top_n: int = 10):
        """
        Initializes the FeatureSelector.

        Args:
            data_path (str): The path to the CSV file with feature data.
            target_col (str): The name of the target column, typically 'close'.
            top_n (int): The number of top features to recommend.
        """
        self.data_path = data_path
        self.target_col = target_col
        self.top_n = top_n
        self.df = None
        print("FeatureSelector has been initialized.")

    def load_data(self) -> bool:
        """
        Loads the data from the CSV file and performs basic validation.

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Error: File '{self.data_path}' does not exist. Please run feature_engineer.py first to generate it.")
            
            self.df = pd.read_csv(self.data_path, parse_dates=['time'])
            if self.df.empty:
                raise ValueError("Error: The loaded data is empty.")
            
            if self.target_col not in self.df.columns:
                raise ValueError(f"Error: Target column '{self.target_col}' does not exist in the data.")
            
            # Keep only numeric columns (except for 'time', which is already parsed)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'time' in self.df.columns:
                 self.df = self.df[['time'] + numeric_cols]
            else:
                 self.df = self.df[numeric_cols]

            if self.df.shape[1] < 2:
                raise ValueError("Error: Not enough numeric features in the data.")
            
            print(f"Successfully loaded {len(self.df)} rows of data with {len(self.df.columns)} numeric columns.")
            return True
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            traceback.print_exc()
            return False

    def handle_missing_values(self):
        """
        Handles missing values and infinite values in the DataFrame.
        It replaces infinities with NaN and then uses forward fill,
        filling any remaining NaNs with 0.
        """
        try:
            # Replace infinite values with NaN
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Forward-fill missing values
            self.df.ffill(inplace=True)
            # Fill any remaining missing values with 0
            self.df.fillna(0, inplace=True)
            print("Missing values and anomalies have been handled.")
        except Exception as e:
            print(f"An error occurred while handling missing values: {e}")
            traceback.print_exc()

    def calculate_correlations(self) -> pd.Series:
        """
        Calculates the Pearson correlation coefficient of each feature with the target column.

        Returns:
            pd.Series: A Series of feature correlations (absolute values, sorted).
        """
        try:
            corr = self.df.corr(numeric_only=True)[self.target_col].drop(self.target_col)
            corr_abs = corr.abs().sort_values(ascending=False)
            print("Pearson correlation coefficients calculated.")
            return corr_abs
        except Exception as e:
            print(f"An error occurred while calculating correlations: {e}")
            traceback.print_exc()
            return pd.Series(dtype='float64')

    def calculate_feature_importances(self) -> pd.Series:
        """
        Calculates feature importances using a Random Forest Regressor.

        Returns:
            pd.Series: A Series of feature importances, sorted in descending order.
        """
        try:
            X = self.df.drop(columns=[self.target_col, 'time'])
            y = self.df[self.target_col]
            
            # Split data to avoid overfitting during importance calculation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            
            # Calculate MSE for model validation
            y_pred = rf.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Random Forest validation MSE: {mse:.4f}")
            
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("Feature importance calculation complete.")
            return importances
        except Exception as e:
            print(f"An error occurred while calculating feature importances: {e}")
            traceback.print_exc()
            return pd.Series(dtype='float64')

    def suggest_features(self, corr: pd.Series, imp: pd.Series) -> list:
        """
        Recommends features based on a weighted score of correlation and importance.

        Args:
            corr (pd.Series): The correlation series.
            imp (pd.Series): The importance series.

        Returns:
            list: A list of recommended feature names.
        """
        suggested = []
        try:
            if not corr.empty and not imp.empty:
                # Combine correlation and importance with weights
                combined = 0.4 * (corr / corr.max()) + 0.6 * (imp / imp.max())
                combined = combined.sort_values(ascending=False)
                suggested = combined.head(self.top_n).index.tolist()
            elif not corr.empty:
                suggested = corr.head(self.top_n).index.tolist()
            elif not imp.empty:
                suggested = imp.head(self.top_n).index.tolist()

            if not suggested:
                raise ValueError("Could not generate suggested features.")
            
            # Always include 'close' as the first feature, essential for time-series analysis
            if 'close' not in suggested:
                suggested = ['close'] + suggested
            else:
                # Move 'close' to the front if it's already in the list
                suggested.remove('close')
                suggested = ['close'] + suggested
            
            # Ensure the final list does not exceed top_n+1 items if 'close' was added
            suggested = suggested[:self.top_n+1]

            print("\nSuggested features to use in actual prediction (based on correlation and importance):")
            for i, feat in enumerate(suggested, 1):
                print(f"{i}. {feat}")
            
            # Save the list to a YAML file for the optimizer to read
            config_dir = 'config'
            os.makedirs(config_dir, exist_ok=True)
            save_path = os.path.join(config_dir, 'suggested_features.yaml')
            with open(save_path, 'w') as f:
                yaml.dump({'features_to_use': suggested}, f)
            print(f"\nSuggested features have been saved to: {save_path}")
            
            return suggested
        except Exception as e:
            print(f"An error occurred while suggesting features: {e}")
            traceback.print_exc()
            return []

    def run(self):
        """
        Executes the complete feature selection workflow.
        """
        if not self.load_data():
            return
        
        self.handle_missing_values()
        
        corr = self.calculate_correlations()
        if not corr.empty:
            print("\nTop 10 Correlation Coefficients (absolute values):")
            print(corr.head(10))
        
        imp = self.calculate_feature_importances()
        if not imp.empty:
            print("\nTop 10 Feature Importances:")
            print(imp.head(10))
        
        if corr.empty and imp.empty:
             print("\nCould not calculate correlations or feature importances. Aborting feature suggestion.")
             return

        self.suggest_features(corr, imp)

if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()