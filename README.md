# Easy Bitcoin on Transformer Project

## Project Overview

Easy Bitcoin on Transformer is a Python-based Bitcoin price prediction system that uses a Transformer model for time-series forecasting. This project integrates historical Bitcoin price data with the Fear & Greed Index, enhancing prediction accuracy through feature engineering, correlation analysis, and hyperparameter optimization. The project supports multi-feature prediction, including technical indicators (like Moving Averages, RSI, MACD, Bollinger Bands) and sentiment indicators.

The main functionalities include data fetching, feature engineering, feature selection, model optimization, training, prediction, and results visualization. It is suitable for short-term Bitcoin price prediction (defaulting to 5 days) and handles the non-stationary nature of price data through relative normalization.

**Current Version Date:** July 17, 2025

**Author/Maintainer:** HuanfuLi

**Dependencies:** Python 3.8+

## Key Features

1.  **Data Update (`update_data.py`)**:
    * Fetches historical Bitcoin candlestick data (OHLCV) from the BitMEX API.
    * Fetches Fear & Greed Index data from the Alternative.me API.
    * Saves the data to `dataset/btc.csv` and `dataset/fear_greed_index.csv`.

2.  **Feature Engineering (`feature_engineer.py`)**:
    * Merges Bitcoin price data with the Fear & Greed Index data.
    * Calculates moving averages (MA_5, MA_10, MA_20, MA_50, MA_100).
    * Computes technical indicators (e.g., price change percentage, volume change percentage, high-low ratio, RSI, MACD, MACD signal line, MACD histogram, Bollinger Bands).
    * Generates sentiment-derived features (e.g., normalized F&G value, change, moving averages, volatility).
    * Handles missing values, trims initial invalid data, and reorders columns by category.
    * Saves the final feature-rich dataset to `dataset/btc_features.csv`.

3.  **Feature Selection (`feature_correlation.py`)**:
    * Loads `btc_features.csv` and calculates the Pearson correlation of each feature with the target (close price).
    * Uses a Random Forest model to calculate feature importance and validates the model's MSE.
    * Recommends the Top N features (default 10) based on a weighted average of correlation and importance.
    * Saves the recommended features to `config/suggested_features.yaml`, always ensuring 'close' is the first feature.

4.  **Model Optimization (`optimizer.py`)**:
    * Loads the suggested features (or falls back to ['close', 'volume']).
    * Uses data from `btc_features.csv`, splitting it chronologically into training, validation, and test sets (default: 1000 days total, 90 for validation, 90 for testing).
    * Performs hyperparameter optimization for the Transformer model using Optuna (tuning `d_model`, `nhead`, `num_encoder_layers`, etc.).
    * Generates a configuration file `config/config_transformer.yaml` with model settings, data date ranges, and the feature list.

5.  **Model Training & Prediction (`main.py`)**:
    * Reads the configuration and feature list from `config_transformer.yaml`.
    * Trains the Transformer model using data from `btc_features.csv` (on the combined training + validation sets).
    * Makes predictions on the test set (defaulting to the last 5 days) and generates a visualization chart showing 30 days of history + 5 days of prediction.
    * Saves the chart to the `outputs/plots/` directory.

## Installation Requirements

-   **Python Version:** 3.8 or higher.
-   **Required Libraries:** Install via `pip install -r requirements.txt`:
    ```
    pandas
    numpy
    torch
    optuna
    PyYAML
    matplotlib
    scikit-learn
    requests
    ```
-   **Hardware:** A GPU with CUDA or MPS support will accelerate training (CPU is used otherwise).
-   **Directory Structure:**
    ```
    Easy-Bitcoin-on-Transformer/
    ├── dataset/              # Data files (btc.csv, fear_greed_index.csv, btc_features.csv)
    ├── config/               # Configuration files (suggested_features.yaml, config_transformer.yaml)
    ├── outputs/plots/        # Visualization charts
    ├── update_data.py
    ├── feature_engineer.py
    ├── feature_correlation.py
    ├── optimizer.py
    ├── main.py
    ├── requirements.txt
    └── README.md
    ```

## Usage Instructions

### Execution Order
The project is designed with a modular workflow. Run the scripts in the following order to complete the full process:

1.  **Update Data:**
    ```bash
    python update_data.py
    ```
    -   This will fetch the latest data and save it to the `dataset/` directory.
    -   Note: APIs may have rate limits; a retry mechanism is built-in.

2.  **Perform Feature Engineering:**
    ```bash
    python feature_engineer.py
    ```
    -   Inputs: `btc.csv` and `fear_greed_index.csv`.
    -   Output: `btc_features.csv` (containing all engineered features).
    -   Will throw an error if input files are missing.

3.  **Select Features:**
    ```bash
    python feature_correlation.py
    ```
    -   Input: `btc_features.csv`.
    -   Output: Prints the Top 10 correlations/importances/suggestions to the terminal and saves the feature list to `suggested_features.yaml`.
    -   You can customize the `top_n` parameter in the script (e.g., `FeatureSelector(top_n=15)`).

4.  **Optimize the Model:**
    ```bash
    python optimizer.py
    ```
    -   Inputs: `btc_features.csv` and `suggested_features.yaml` (uses defaults if not found).
    -   Output: `config_transformer.yaml` (containing optimized hyperparameters and the feature list).
    -   You can customize `n_trials` (default is 50; 15-50 is recommended to save time).
    -   Note: The optimization process can be time-consuming, depending on GPU availability.

5.  **Train and Predict:**
    ```bash
    python main.py
    ```
    -   Inputs: `config_transformer.yaml` and `btc_features.csv`.
    -   Outputs: Training logs, prediction results (printed to terminal), and a visualization chart saved to `outputs/plots/`.
    -   The chart displays the last 30 days of historical price vs. the 5-day prediction.

### Important Notes
-   **Error Handling:** Each script includes exception handling and will print stack traces. If a file is missing, it will prompt you to run the prerequisite script.
-   **Custom Parameters:**
    -   In `optimizer.py`, adjust `TOTAL_DAYS_FOR_OPTIMIZATION`, `TEST_SET_SIZE_DAYS`, etc., to change the data range for optimization.
    -   In `feature_correlation.py`, adjust `top_n` to control the number of suggested features (too many may lead to overfitting).
-   **Multi-Feature Support:** The system automatically uses the suggested features for training to improve accuracy (e.g., by integrating RSI, MACD, etc.).
-   **Visualization:** The generated PNG chart includes historical actual prices (green), predicted prices (red dashed line), and a line indicating the prediction start.
-   **Performance Tip:** Using a GPU will significantly speed up optimization and training. Predictions are based on relative changes, making the model suitable for non-stationary time series.
-   **Potential Issues:** If data APIs are unavailable, check your network or the API status manually. The project does not guarantee prediction accuracy and is for educational/referential purposes only.