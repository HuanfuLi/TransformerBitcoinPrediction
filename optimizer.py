# filename: optimizer.py (Corrected)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import yaml
import os
import math
import datetime
from sklearn.preprocessing import StandardScaler

# --- User Configuration: Data Range for Optimization ---
TOTAL_DAYS_FOR_OPTIMIZATION = 1000
TEST_SET_SIZE_DAYS = 90
VALIDATION_SET_SIZE_DAYS = 90
# The rest (820 days) will be used for training.
# --- End of Configuration ---

# --- Global Settings (CUDA and MPS Support) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Detected CUDA device, using GPU acceleration.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Detected Apple MPS device, using GPU acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("No GPU detected, using CPU.")

# --- Model Definition (Unchanged) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model); position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_size):
        super(TransformerModel, self).__init__()
        self.d_model = d_model; self.input_projection = nn.Linear(input_size, d_model); self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers); self.decoder = nn.Linear(d_model, output_size)
    def forward(self, src):
        src = self.input_projection(src); src = self.pos_encoder(src); output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :]); return output

# --- Data Handling Functions ---
def get_btc_data():
    """Loads data from CSV."""
    data_path = 'dataset/btc.csv'
    if not os.path.exists(data_path): raise FileNotFoundError(f"Data file not found: '{data_path}'. Please run 'update_data.py' first.")
    df = pd.read_csv(data_path, parse_dates=['time']); df = df.sort_values('time').reset_index(drop=True)
    return df

def extract_sequences(data, time_step, predicted_days):
    """Extracts sequences and applies relative normalization to 'close' (assumed to be column 0)."""
    X, Y = [], []
    for i in range(len(data) - time_step - predicted_days + 1):
        input_seq = data[i:i + time_step].copy()
        last_close = input_seq[-1, 0]
        if last_close == 0: last_close = 1e-8
        
        # Normalize the 'close' column of the input sequence
        input_seq[:, 0] /= last_close
        X.append(input_seq)
        
        # Normalize the target 'close' values
        y_seq = data[i + time_step:i + time_step + predicted_days, 0] / last_close
        Y.append(y_seq)
        
    return np.array(X), np.array(Y)

def objective(trial, X_train, y_train, X_val, y_val, input_size, output_size):
    """Optuna objective function."""
    nhead = trial.suggest_categorical('nhead', [2, 4, 8]); d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    if d_model % nhead != 0: raise optuna.exceptions.TrialPruned()
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 6); dim_feedforward = trial.suggest_int('dim_feedforward', 128, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.4); learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = TransformerModel(input_size=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, output_size=output_size).to(DEVICE)
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_train_tensor, y_train_tensor = torch.FloatTensor(X_train).to(DEVICE), torch.FloatTensor(y_train).to(DEVICE)
    X_val_tensor, y_val_tensor = torch.FloatTensor(X_val).to(DEVICE), torch.FloatTensor(y_val).to(DEVICE)
    for epoch in range(30):
        model.train(); permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            optimizer.zero_grad(); indices = permutation[i:i+batch_size]; batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            outputs = model(batch_X); loss = criterion(outputs, batch_y); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): val_outputs = model(X_val_tensor); val_loss = criterion(val_outputs, y_val_tensor)
    if DEVICE.type in ['mps', 'cuda']: torch.cuda.empty_cache() if DEVICE.type == 'cuda' else torch.mps.empty_cache()
    return val_loss.item()

# --- Main Optimization Flow ---
def run_optimization(n_trials=50):
    """Runs the hyperparameter optimization."""
    print("Starting hyperparameter optimization for Transformer model...")
    df = get_btc_data(); end_date = df['time'].max(); start_date = end_date - pd.DateOffset(days=TOTAL_DAYS_FOR_OPTIMIZATION - 1)
    print(f"Latest date in dataset: {end_date.strftime('%Y-%m-%d')}. Using this as the base for date calculations.")
    data_block_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
    test_split_date = end_date - pd.DateOffset(days=TEST_SET_SIZE_DAYS)
    validation_split_date = test_split_date - pd.DateOffset(days=VALIDATION_SET_SIZE_DAYS)
    train_df = data_block_df[data_block_df['time'] < validation_split_date].copy()
    val_df = data_block_df[(data_block_df['time'] >= validation_split_date) & (data_block_df['time'] < test_split_date)].copy()
    test_df = data_block_df[data_block_df['time'] >= test_split_date].copy()
    for d_set in [train_df, val_df, test_df]: d_set.ffill(inplace=True); d_set.fillna(0, inplace=True)
    print("\nDynamically split dataset based on latest data:")
    print(f"  Total Data Block: {data_block_df['time'].min().strftime('%Y-%m-%d')} to {data_block_df['time'].max().strftime('%Y-%m-%d')}")
    print(f"  Training Set:     {train_df['time'].min().strftime('%Y-%m-%d')} to {train_df['time'].max().strftime('%Y-%m-%d')} ({len(train_df)} days)")
    print(f"  Validation Set:   {val_df['time'].min().strftime('%Y-%m-%d')} to {val_df['time'].max().strftime('%Y-%m-%d')} ({len(val_df)} days)")
    print(f"  Test Set:         {test_df['time'].min().strftime('%Y-%m-%d')} to {test_df['time'].max().strftime('%Y-%m-%d')} ({len(test_df)} days)\n")
    
    # ** Key Change: No more StandardScaler **
    feature_cols = ['close', 'volume']
    train_data = train_df[feature_cols].values
    val_data = val_df[feature_cols].values
    
    time_step = 60; predicted_days = 5
    X_train, y_train = extract_sequences(train_data, time_step, predicted_days)
    X_val, y_val = extract_sequences(val_data, time_step, predicted_days)

    if len(X_train) == 0 or len(X_val) == 0: raise ValueError("Error: Dataset is too small to create valid training or validation sequences.")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_train.shape[2], y_train.shape[1]), n_trials=n_trials)
    
    best_params = study.best_params
    print(f"\nOptimization complete! Best validation loss: {study.best_value:.6f}\nBest hyperparameters: {best_params}")
    
    config = {
        'model_settings': { 'time_step': time_step, 'predicted_days': predicted_days, 'learning_rate': best_params['learning_rate'], 'batch_size': best_params['batch_size'], 'epochs': 200 },
        'data_settings': {
            'train_start_date': train_df['time'].min().strftime('%Y-%m-%d'), 'train_end_date': train_df['time'].max().strftime('%Y-%m-%d'),
            'validation_start_date': val_df['time'].min().strftime('%Y-%m-%d'), 'validation_end_date': val_df['time'].max().strftime('%Y-%m-%d'),
            'test_start_date': test_df['time'].min().strftime('%Y-%m-%d'), 'test_end_date': test_df['time'].max().strftime('%Y-%m-%d'),
        },
        'model_specific': { 'Transformer': { 'd_model': best_params['d_model'], 'nhead': best_params['nhead'], 'num_encoder_layers': best_params['num_encoder_layers'], 'dim_feedforward': best_params['dim_feedforward'], 'dropout': best_params['dropout'] } },
        'feature_settings': { 'features_to_use': feature_cols }
    }
    config_dir = 'config'; os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, 'config_transformer.yaml'), 'w') as f: yaml.dump(config, f, default_flow_style=False)
    print(f"\nBest configuration saved to: config/config_transformer.yaml")

if __name__ == "__main__":
    try: run_optimization(n_trials=15)
    except (FileNotFoundError, ValueError) as e: print(f"\nOperation terminated: {e}")