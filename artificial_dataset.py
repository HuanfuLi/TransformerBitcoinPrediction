import pandas as pd
import numpy as np
import os
import datetime as dt

# ===================================================================
# --- User Configuration Area ---
# ===================================================================

# Choose the shape of the dataset to generate.
# Available options: "UPWARD_TREND", "DOWNWARD_TREND", "SINUSOID", "INVERSE"
DATASET_SHAPE = "INVERSE" 

# General settings for the dataset
TOTAL_DAYS = 1500              # Total number of data points (days) to generate.
BASE_PRICE = 50000             # The starting price for the simulation.
BASE_VOLUME = 10000            # Base trading volume (will be constant).

# Settings specific to SINUSOID shape
SINE_WAVE_AMPLITUDE = 15000    # The peak amplitude of the sine wave from the base price.
SINE_WAVE_FREQUENCY = 0.05     # The frequency of the sine wave (higher value = more cycles).

# Settings specific to TREND shapes
TREND_STRENGTH = 25            # The daily price change for UPWARD or DOWNWARD trends.

# Output file settings
OUTPUT_DIR = 'dataset'
OUTPUT_FILENAME = 'btc_artificial_pure.csv' # Changed filename to reflect pure data

# ===================================================================
# --- Data Generation Logic ---
# ===================================================================

def generate_dataset(shape: str):
    """
    Main function to generate and save the dataset based on the selected shape.
    """
    print(f"Generating PURE artificial dataset with shape: {shape}")

    # Create a date range for the time series
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=TOTAL_DAYS - 1)
    time_index = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
    
    # Generate the core 'close' price based on the selected shape
    if shape == "UPWARD_TREND":
        close_prices = generate_trend(upward=True)
    elif shape == "DOWNWARD_TREND":
        close_prices = generate_trend(upward=False)
    elif shape == "SINUSOID":
        close_prices = generate_sinusoid()
    elif shape == "INVERSE":
        close_prices = generate_inverse()
    else:
        raise ValueError(f"Unknown shape '{shape}'. Please choose from the available options.")

    # Create a complete DataFrame with OHLCV data
    df = create_full_dataframe(time_index, close_prices)

    # Save the generated data to a CSV file
    save_data(df)

def generate_trend(upward: bool = True) -> np.ndarray:
    """Generates a perfect linear upward or downward trend."""
    direction = 1 if upward else -1
    trend = np.arange(TOTAL_DAYS) * TREND_STRENGTH * direction
    return BASE_PRICE + trend

def generate_sinusoid() -> np.ndarray:
    """Generates a perfect sinusoidal wave pattern."""
    x = np.arange(TOTAL_DAYS)
    sine_wave = SINE_WAVE_AMPLITUDE * np.sin(SINE_WAVE_FREQUENCY * x)
    return BASE_PRICE + sine_wave

def generate_inverse() -> np.ndarray:
    """Generates a perfect f(x) = 1/x style curve."""
    x = np.linspace(1, 10, TOTAL_DAYS)
    # Scale to make the curve more pronounced and start from the base price
    inverse_curve = (BASE_PRICE * 4) / x 
    # Adjust to start near the base price
    inverse_curve = inverse_curve - inverse_curve[0] + BASE_PRICE
    return inverse_curve

def create_full_dataframe(time_index: pd.DatetimeIndex, close_prices: np.ndarray) -> pd.DataFrame:
    """
    Creates a full OHLCV DataFrame from the generated close prices.
    For pure data, O, H, L are the same as Close, and Volume is constant.
    """
    df = pd.DataFrame(index=time_index)
    
    # Set all price points to be identical for a pure curve
    df['close'] = np.maximum(1, close_prices)
    df['open'] = df['close']
    df['high'] = df['close']
    df['low'] = df['close']

    # Set volume to a constant base value
    df['volume'] = BASE_VOLUME

    df = df.reset_index().rename(columns={'index': 'time'})
    
    # Reorder columns to match the project's format
    return df[['time', 'open', 'high', 'low', 'close', 'volume']]

def save_data(df: pd.DataFrame):
    """
    Saves the DataFrame to a CSV file.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"Successfully generated and saved PURE artificial dataset!")
    print(f"Shape: {DATASET_SHAPE}")
    print(f"Rows: {len(df)}")
    print(f"Location: {output_path}")
    print("-" * 50)


if __name__ == "__main__":
    try:
        generate_dataset(DATASET_SHAPE)
    except ValueError as e:
        print(f"Error: {e}")