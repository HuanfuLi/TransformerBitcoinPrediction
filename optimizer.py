# 文件名: optimize_transformer.py (已修正并重构)

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

# --- 用户配置区域: 定义用于优化的数据范围 ---
# 我们将从数据最新的那天往前取一个大的数据块，然后在这个块里划分训练/验证/测试集
TOTAL_DAYS_FOR_OPTIMIZATION = 1000 # 使用最近1000天的数据作为总的数据池
TEST_SET_SIZE_DAYS = 90          # 在数据池的末尾，留出90天作为测试集
VALIDATION_SET_SIZE_DAYS = 90    # 在测试集之前，再留出90天作为验证集
# 剩下的部分 (1000 - 90 - 90 = 820天) 将作为训练集
# --- 配置结束 ---


# --- 全局设置 ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("检测到 Apple MPS 设备，将使用 GPU 加速。")
else:
    DEVICE = torch.device("cpu")
    print("未检测到 MPS 设备，将使用 CPU。")


# --- 模型定义 (与你提供的旧版本一致) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_size):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# --- 数据处理函数 (与你提供的旧版本一致) ---
def get_btc_data():
    """从CSV文件加载数据"""
    data_path = 'dataset/btc.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['time'])
        df = df.sort_values('time').reset_index(drop=True)
        # 在分割数据集后独立处理缺失值，此处不再填充
        return df
    else:
        raise FileNotFoundError(f"未找到数据文件 '{data_path}'。请先运行 'update_data.py'。")

def extract_sequences(data, time_step, predicted_days):
    """提取序列数据"""
    X, Y = [], []
    for i in range(len(data) - time_step - predicted_days + 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[(i + time_step):(i + time_step + predicted_days), 0])
    return np.array(X), np.array(Y)

# --- Optuna 目标函数 (与你提供的旧版本一致) ---
def objective(trial, X_train, y_train, X_val, y_val, input_size, output_size):
    """Optuna的优化目标函数"""
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 6)
    dim_feedforward = trial.suggest_int('dim_feedforward', 128, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = TransformerModel(input_size=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, output_size=output_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    y_val_tensor = torch.FloatTensor(y_val).to(DEVICE)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    return val_loss.item()

# --- 主优化流程 (已重构) ---
def run_optimization(n_trials=50):
    """运行超参数优化"""
    print("开始为 Transformer 模型进行超参数优化...")

    # 1. 加载数据并确定日期范围
    df = get_btc_data()
    end_date = df['time'].max()
    start_date = end_date - pd.DateOffset(days=TOTAL_DAYS_FOR_OPTIMIZATION - 1)
    
    # 从完整数据集中切分出用于本次优化的数据块
    data_block_df = df[(df['time'] >= start_date) & (df['time'] <= end_date)].copy()
    
    # 2. 在数据块内划分训练、验证、测试集 (严格按时间)
    test_split_date = end_date - pd.DateOffset(days=TEST_SET_SIZE_DAYS)
    validation_split_date = test_split_date - pd.DateOffset(days=VALIDATION_SET_SIZE_DAYS)

    train_df = data_block_df[data_block_df['time'] < validation_split_date].copy()
    val_df = data_block_df[(data_block_df['time'] >= validation_split_date) & (data_block_df['time'] < test_split_date)].copy()
    test_df = data_block_df[data_block_df['time'] >= test_split_date].copy()
    
    # 3. [无数据泄露] 对分割后的数据集独立进行缺失值填充
    for d_set in [train_df, val_df, test_df]:
        d_set.ffill(inplace=True)
        d_set.fillna(0, inplace=True) # 用0填充可能存在的头部NaN

    print("\n已根据最新数据动态划分数据集:")
    print(f"  总数据块: {data_block_df['time'].min().strftime('%Y-%m-%d')} to {data_block_df['time'].max().strftime('%Y-%m-%d')}")
    print(f"  训练集:    {train_df['time'].min().strftime('%Y-%m-%d')} to {train_df['time'].max().strftime('%Y-%m-%d')} ({len(train_df)}天)")
    print(f"  验证集:    {val_df['time'].min().strftime('%Y-%m-%d')} to {val_df['time'].max().strftime('%Y-%m-%d')} ({len(val_df)}天)")
    print(f"  测试集:    {test_df['time'].min().strftime('%Y-%m-%d')} to {test_df['time'].max().strftime('%Y-%m-%d')} ({len(test_df)}天)\n")

    # 4. 准备数据进行优化
    feature_cols = ['close', 'volume']
    scaler = StandardScaler()
    
    train_data_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_data_scaled = scaler.transform(val_df[feature_cols].values)
    
    time_step = 60
    predicted_days = 5

    X_train, y_train = extract_sequences(train_data_scaled, time_step, predicted_days)
    X_val, y_val = extract_sequences(val_data_scaled, time_step, predicted_days)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("错误：数据集过小，无法创建出有效的训练或验证序列。请在脚本顶部增加 *_SIZE_DAYS。")

    input_size = X_train.shape[2]
    output_size = y_train.shape[1]

    # 5. 创建并运行 Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_size, output_size), n_trials=n_trials)

    # 6. 保存最佳配置
    best_params = study.best_params
    print(f"\n优化完成！最佳验证损失: {study.best_value:.6f}")
    print("最佳超参数:", best_params)

    config = {
        'model_settings': { 'active_model': 'Transformer', 'time_step': time_step, 'predicted_days': predicted_days, 'predict_start_date': '2025-07-01', 'learning_rate': best_params['learning_rate'], 'batch_size': best_params['batch_size'], 'epochs': 200, 'early_stop_patience': 20 },
        'model_specific': { 'Transformer': { 'd_model': best_params['d_model'], 'nhead': best_params['nhead'], 'num_encoder_layers': best_params['num_encoder_layers'], 'dim_feedforward': best_params['dim_feedforward'], 'dropout': best_params['dropout'] } },
        'feature_settings': { 'features_to_use': feature_cols }
    }

    config_dir = 'config'
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    config_file = os.path.join(config_dir, 'config_transformer.yaml')
    with open(config_file, 'w') as f: yaml.dump(config, f, default_flow_style=False)

    print(f"\n最佳配置已保存到: {config_file}")
    return study

if __name__ == "__main__":
    try:
        run_optimization(n_trials=50)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n操作终止: {e}")