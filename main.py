# 文件名: main.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import os
import math
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 全局设置 (添加了MPS支持) ---
# 自动检测并选择最佳可用设备 (MPS for Apple Silicon > CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("检测到 Apple MPS 设备，将使用 GPU 加速。")
else:
    DEVICE = torch.device("cpu")
    print("未检测到 MPS 设备，将使用 CPU。")


# --- 模型定义 ---
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

# --- 辅助函数 ---
def load_config(file_path='config/config_transformer.yaml'):
    """加载YAML配置文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到配置文件 '{file_path}'。请先运行 'optimize_transformer.py'。")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def extract_data(data, time_step, predicted_days):
    """从数据中提取序列"""
    X, Y = [], []
    for i in range(len(data) - time_step - predicted_days + 1):
        X.append(data[i:(i + time_step)])
        Y.append(data[i + time_step:i + time_step + predicted_days, 0]) # 目标是收盘价
    return np.array(X), np.array(Y)

def main():
    """主执行函数"""
    print("开始执行主流程：模型训练与预测...")

    # 1. 加载配置
    config = load_config()
    model_settings = config['model_settings']
    transformer_config = config['model_specific']['Transformer']
    feature_columns = config['feature_settings']['features_to_use']

    print(f"模型配置加载成功，将使用以下特征: {feature_columns}")

    # 2. 加载并准备数据
    data_path = 'dataset/btc.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件 '{data_path}'。请先运行 'update_data.py'。")
    
    df = pd.read_csv(data_path, parse_dates=['time'])
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 根据配置划分数据
    predict_start_date = pd.to_datetime(model_settings['predict_start_date'])
    # 使用预测开始日期前的所有数据进行训练
    train_df = df[df['time'] < predict_start_date].copy()
    
    if len(train_df) < model_settings['time_step'] * 2:
        raise ValueError("训练数据不足，请确保拥有足够长的历史数据。")

    print(f"训练数据时间范围: {train_df['time'].min()} to {train_df['time'].max()} ({len(train_df)} 条)")
    
    # 3. 数据缩放和序列化
    scaler = StandardScaler()
    train_features = train_df[feature_columns].values
    scaled_train_data = scaler.fit_transform(train_features)
    
    X_train, y_train = extract_data(
        scaled_train_data, 
        model_settings['time_step'], 
        model_settings['predicted_days']
    )
    
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    
    # 4. 初始化并训练模型
    model = TransformerModel(
        input_size=X_train.shape[2],
        output_size=y_train.shape[1],
        **transformer_config
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_settings['learning_rate'])
    
    print("\n开始模型训练...")
    for epoch in range(model_settings['epochs']):
        model.train()
        permutation = torch.randperm(X_train_tensor.size(0))
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, X_train_tensor.size(0), model_settings['batch_size']):
            optimizer.zero_grad()
            indices = permutation[i:i + model_settings['batch_size']]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{model_settings['epochs']}, 训练损失: {avg_loss:.6f}")
    
    print("模型训练完成！")

    # 5. 进行预测
    print(f"\n开始为从 {predict_start_date.strftime('%Y-%m-%d')} 起的 {model_settings['predicted_days']} 天进行预测...")
    
    # 准备预测所需输入数据
    pred_input_start = predict_start_date - pd.DateOffset(days=model_settings['time_step'])
    pred_data = df[(df['time'] >= pred_input_start) & (df['time'] < predict_start_date)].copy()
    
    if len(pred_data) < model_settings['time_step']:
        raise ValueError(f"预测所需数据不足。需要 {model_settings['time_step']} 天，但只有 {len(pred_data)} 天。")
    
    pred_features = pred_data[feature_columns].values
    pred_scaled = scaler.transform(pred_features)
    
    pred_input = pred_scaled.reshape(1, model_settings['time_step'], -1)
    pred_input_tensor = torch.FloatTensor(pred_input).to(DEVICE)

    # 进行预测
    model.eval()
    with torch.no_grad():
        prediction_tensor = model(pred_input_tensor)
    
    prediction_scaled = prediction_tensor.cpu().numpy()[0]
    
    # 将预测结果逆缩放回原始价格
    dummy_array = np.zeros((len(prediction_scaled), len(feature_columns)))
    dummy_array[:, 0] = prediction_scaled
    
    unscaled_prediction = scaler.inverse_transform(dummy_array)
    final_predictions = unscaled_prediction[:, 0]

    pred_dates = pd.date_range(start=predict_start_date, periods=model_settings['predicted_days'])
    
    print("\n预测结果 (美元):")
    for i, date in enumerate(pred_dates):
        print(f"{date.strftime('%Y-%m-%d')}: ${final_predictions[i]:.2f}")
        
    # ===================================================================
    # 6. [已修改] 生成用于模型验证的30天窗口对比图表
    # ===================================================================
    print("\n正在生成30天窗口验证图表...")

    predicted_days_val = model_settings['predicted_days']
    
    # 定义绘图的30天窗口范围
    plot_end_date = predict_start_date + pd.DateOffset(days=predicted_days_val - 1)
    plot_start_date = plot_end_date - pd.DateOffset(days=29)

    # 获取这30天的全部历史数据用于绘图
    historical_data_for_plot = df[(df['time'] >= plot_start_date) & (df['time'] <= plot_end_date)]

    # 单独获取预测时间段内的真实数据
    actuals_for_prediction_period = df[(df['time'] >= predict_start_date) & (df['time'] <= plot_end_date)]

    plt.figure(figsize=(15, 8))
    
    # 绘制30天的真实历史价格作为基准
    plt.plot(historical_data_for_plot['time'], historical_data_for_plot['close'], 'g-', label='历史真实价格', alpha=0.8, linewidth=2)

    # 在同一时间段上，绘制模型的5天预测价格，用于对比
    plt.plot(pred_dates, final_predictions, 'r--', label=f'模型预测价格 (5天)', alpha=0.9, linewidth=2, marker='o')

    # 添加垂直线，标记预测开始的位置
    plt.axvline(x=predict_start_date, color='black', linestyle=':', alpha=0.7, label='预测开始点')

    plt.title(f"模型验证：30天历史价格 vs 5天预测价格", fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格 (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    plot_dir = 'outputs/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_dir}/validation_plot_{predict_start_date.strftime('%Y%m%d')}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"验证图表已成功保存到: {filename}")

if __name__ == '__main__':
    # 确保输出目录存在
    for dir_name in ['config', 'dataset', 'outputs/plots']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {e}")
        print("请按以下顺序执行脚本：")
        print("1. python update_data.py  (获取数据)")
        print("2. python optimize_transformer.py (优化模型并生成配置)")
        print("3. python main.py (训练、预测并绘图)")