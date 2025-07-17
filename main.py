# filename: main.py (Corrected)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import os
import math
import datetime as dt
import matplotlib.pyplot as plt

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
def load_config(file_path='config/config_transformer.yaml'):
    """Loads YAML configuration file."""
    if not os.path.exists(file_path): raise FileNotFoundError(f"Configuration file not found: '{file_path}'. Please run 'optimizer.py' first.")
    with open(file_path, 'r') as f: return yaml.safe_load(f)

def extract_sequences(data, time_step, predicted_days):
    """Extracts sequences and applies relative normalization to 'close' (assumed to be column 0)."""
    X, Y = [], []
    for i in range(len(data) - time_step - predicted_days + 1):
        input_seq = data[i:i + time_step].copy()
        last_close = input_seq[-1, 0]
        if last_close == 0: last_close = 1e-8
        
        input_seq[:, 0] /= last_close
        X.append(input_seq)
        
        y_seq = data[i + time_step:i + time_step + predicted_days, 0] / last_close
        Y.append(y_seq)
        
    return np.array(X), np.array(Y)

def main():
    """Main function for training and validation."""
    print("Starting model training and validation process...")

    # 1. Load Configuration
    config = load_config()
    model_settings, data_settings = config['model_settings'], config['data_settings']
    transformer_config, feature_columns = config['model_specific']['Transformer'], config['feature_settings']['features_to_use']
    print("Configuration loaded successfully.")

    # 2. Load and Split Data based on Config
    df = pd.read_csv('dataset/btc.csv', parse_dates=['time'])
    train_start, train_end = pd.to_datetime(data_settings['train_start_date']), pd.to_datetime(data_settings['train_end_date'])
    val_start, val_end = pd.to_datetime(data_settings['validation_start_date']), pd.to_datetime(data_settings['validation_end_date'])
    test_start, test_end = pd.to_datetime(data_settings['test_start_date']), pd.to_datetime(data_settings['test_end_date'])
    
    train_val_df = df[df['time'].between(train_start, val_end)].copy()
    test_df = df[df['time'].between(test_start, test_end)].copy()
    
    for d_set in [train_val_df, test_df]: d_set.ffill(inplace=True); d_set.fillna(0, inplace=True)
    
    print(f"Final training data range: {train_val_df['time'].min().strftime('%Y-%m-%d')} to {train_val_df['time'].max().strftime('%Y-%m-%d')}")
    print(f"Final test data range: {test_df['time'].min().strftime('%Y-%m-%d')} to {test_df['time'].max().strftime('%Y-%m-%d')}")

    # 3. Prepare Training Data with Relative Normalization
    train_val_data = train_val_df[feature_columns].values
    X_train, y_train = extract_sequences(train_val_data, model_settings['time_step'], model_settings['predicted_days'])
    X_train_tensor, y_train_tensor = torch.FloatTensor(X_train).to(DEVICE), torch.FloatTensor(y_train).to(DEVICE)
    
    # 4. Initialize and Train Model
    model = TransformerModel(input_size=X_train.shape[2], output_size=y_train.shape[1], **transformer_config).to(DEVICE)
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=model_settings['learning_rate'])
    
    print("\nStarting model training...")
    for epoch in range(model_settings['epochs']):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0)); epoch_loss, num_batches = 0.0, 0
        for i in range(0, X_train_tensor.size(0), model_settings['batch_size']):
            optimizer.zero_grad(); indices = permutation[i:i + model_settings['batch_size']]; batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            outputs = model(batch_X); loss = criterion(outputs, batch_y); loss.backward(); optimizer.step()
            epoch_loss += loss.item(); num_batches += 1
        if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{model_settings['epochs']}, Training Loss: {epoch_loss/num_batches:.6f}")
    print("Model training complete!")

    # 5. Validation and Plotting on the Test Set
    print(f"\nValidating on test set and plotting the last {len(test_df.tail(30))} days for fit visualization...")

    predict_target_start_date = test_df['time'].iloc[-5]
    pred_input_start_date = predict_target_start_date - pd.DateOffset(days=model_settings['time_step'])
    pred_input_df = df[df['time'].between(pred_input_start_date, predict_target_start_date, inclusive='left')].copy()
    
    # ** Key Change: Prepare prediction input with relative normalization **
    pred_features = pred_input_df[feature_columns].values
    last_close_for_prediction = pred_features[-1, 0]
    if last_close_for_prediction == 0: last_close_for_prediction = 1e-8
    
    pred_input_normalized = pred_features.copy()
    pred_input_normalized[:, 0] /= last_close_for_prediction
    pred_input_tensor = torch.FloatTensor(pred_input_normalized.reshape(1, model_settings['time_step'], -1)).to(DEVICE)

    # Make prediction (output will be relative ratios)
    model.eval()
    with torch.no_grad(): prediction_tensor = model(pred_input_tensor)
    prediction_relative = prediction_tensor.cpu().numpy()[0]
    
    # ** Key Change: Convert relative prediction back to absolute price **
    final_predictions = prediction_relative * last_close_for_prediction
    pred_dates = pd.date_range(start=predict_target_start_date, periods=model_settings['predicted_days'])
    
    # 6. Plotting
    plot_data_df = test_df.tail(30)
    plt.figure(figsize=(15, 8))
    plt.plot(plot_data_df['time'], plot_data_df['close'], 'g-', label='Historical Actual Price', linewidth=2)
    plt.plot(pred_dates, final_predictions, 'r--', label=f'Model Prediction ({model_settings["predicted_days"]} days)', marker='o')
    plt.axvline(x=predict_target_start_date, color='black', linestyle=':', alpha=0.7, label='Prediction Start')
    plt.title(f"Model Validation: Last 30 Days of Test Set vs. 5-Day Prediction", fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Price (USD)'); plt.legend(); plt.grid(True, alpha=0.3); plt.xticks(rotation=45); plt.tight_layout()
    
    plot_dir = 'outputs/plots'
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{plot_dir}/final_validation_plot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nValidation plot saved successfully to: {filename}")

if __name__ == '__main__':
    try: main()
    except (FileNotFoundError, ValueError, KeyError) as e: print(f"\nError: {e}")