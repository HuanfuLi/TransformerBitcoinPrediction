# BitcoinConcise On Transformer

## 项目概述

BitcoinConcise on Transformer 是一个基于 Python 的比特币价格预测系统，使用 Transformer 模型进行时间序列预测。该项目整合了比特币历史价格数据、恐惧贪婪指数（Fear & Greed Index），并通过特征工程、相关性分析和超参数优化来提升预测精度。项目支持多特征预测，包括技术指标（如移动平均线、RSI、MACD、布林带）和情绪指标。

项目主要功能包括数据获取、特征工程、特征选择、模型优化、训练与预测，以及可视化结果。适用于比特币价格短期预测（默认5天），并通过相对归一化处理非平稳性问题。

**当前版本日期：** 2025年7月17日  
**作者/维护者：** HuanfuLi
**依赖环境：** Python 3.8+  

## 主要功能

1. **数据更新（update_data.py）**：
   - 从 BitMEX API 获取比特币历史K线数据（开盘价、最高价、最低价、收盘价、交易量）。
   - 从 Alternative.me API 获取恐惧贪婪指数数据。
   - 保存数据到 `dataset/btc.csv` 和 `dataset/fear_greed_index.csv`。

2. **特征工程（feature_engineer.py）**：
   - 合并比特币价格和恐惧贪婪指数数据。
   - 计算移动平均线（MA_5, MA_10, MA_20, MA_50, MA_100）。
   - 计算技术指标（如价格变化百分比、交易量变化百分比、高低价比率、RSI、MACD、MACD信号线、MACD柱状图、布林带上轨/下轨/宽度/中轨/标准差）。
   - 计算情绪衍生特征（如归一化恐惧贪婪值、变化值、移动平均、波动率）。
   - 处理缺失值，裁切初始无效数据，按类别排序列。
   - 保存最终特征数据集到 `dataset/btc_features.csv`。

3. **特征选择（feature_correlation.py）**：
   - 加载 `btc_features.csv`，计算每个特征与目标（收盘价）的Pearson相关系数。
   - 使用随机森林模型计算特征重要性，并验证模型MSE。
   - 基于相关性和重要性（加权平均）建议Top N特征（默认10个）。
   - 保存建议特征到 `config/suggested_features.yaml`（始终将'close'置于首位）。

4. **模型优化（optimizer.py）**：
   - 加载建议特征（若无则回退到['close', 'volume']）。
   - 使用 `btc_features.csv` 中的数据，按时间划分训练/验证/测试集（默认总1000天，验证90天，测试90天）。
   - 使用Optuna进行Transformer模型超参数优化（d_model, nhead, num_encoder_layers 等）。
   - 生成配置文件 `config/config_transformer.yaml`，包括模型设置、数据日期范围、特征列表。

5. **模型训练与预测（main.py）**：
   - 读取 `config_transformer.yaml` 中的配置和特征列表。
   - 使用 `btc_features.csv` 中的数据训练Transformer模型（结合训练+验证集）。
   - 在测试集上进行预测（默认最后5天），生成30天历史+5天预测的可视化图表。
   - 保存图表到 `outputs/plots/` 目录。

## 安装要求

- **Python 版本：** 3.8 或更高。
- **所需库：** 通过 `pip install -r requirements.txt` 安装：
  ```
  pandas
  numpy
  torch
  optuna
  yaml
  matplotlib
  scikit-learn
  requests
  ```
- **硬件要求：** 支持CUDA或MPS的GPU可加速训练（否则使用CPU）。
- **目录结构：**
  ```
  BitcoinConcise/
  ├── dataset/              # 数据文件 (btc.csv, fear_greed_index.csv, btc_features.csv)
  ├── config/               # 配置文件 (suggested_features.yaml, config_transformer.yaml)
  ├── outputs/plots/        # 可视化图表
  ├── update_data.py
  ├── feature_engineer.py
  ├── feature_correlation.py
  ├── optimizer.py
  ├── main.py
  └── README.md
  ```

## 使用说明

### 运行顺序
项目采用模块化设计，按以下顺序运行脚本以完成完整工作流：

1. **更新数据：**
   ```
   python update_data.py
   ```
   - 这将获取最新数据并保存到 `dataset/` 目录。
   - 注意：API可能有速率限制，重试机制已内置。

2. **进行特征工程：**
   ```
   python feature_engineer.py
   ```
   - 输入：`btc.csv` 和 `fear_greed_index.csv`。
   - 输出：`btc_features.csv`（包含所有特征）。
   - 如果数据缺失，会抛出错误提示。

3. **特征选择：**
   ```
   python feature_correlation.py
   ```
   - 输入：`btc_features.csv`。
   - 输出：终端打印Top 10相关系数/重要性/建议特征，并保存到 `suggested_features.yaml`。
   - 可自定义 `top_n` 参数（如 `FeatureSelector(top_n=15)`）。

4. **模型优化：**
   ```
   python optimizer.py
   ```
   - 输入：`btc_features.csv` 和 `suggested_features.yaml`（若无则使用默认）。
   - 输出：`config_transformer.yaml`（包含优化后的超参数和特征列表）。
   - 可自定义 `n_trials`（默认50，建议15-50以节省时间）。
   - 注意：优化过程可能耗时，取决于GPU可用性。

5. **训练与预测：**
   ```
   python main.py
   ```
   - 输入：`config_transformer.yaml` 和 `btc_features.csv`。
   - 输出：训练日志、预测结果（终端打印）、可视化图表（保存到 `outputs/plots/`）。
   - 图表显示测试集最后30天历史价格 vs. 5天预测价格。

### 注意事项
- **错误处理：** 每个脚本内置异常捕获和打印栈追踪。如果缺少文件，会提示运行前置脚本。
- **自定义参数：**
  - 在 `optimizer.py` 中调整 `TOTAL_DAYS_FOR_OPTIMIZATION`、`TEST_SET_SIZE_DAYS` 等以改变数据范围。
  - 在 `feature_correlation.py` 中调整 `top_n` 以控制建议特征数量（过多可能导致过拟合）。
- **多特征支持：** 系统自动使用建议特征进行训练，提高精度（例如整合RSI、MACD等指标）。
- **可视化：** 生成的PNG图表包含历史真实价格（绿色）、预测价格（红色虚线）和预测起始线。
- **性能提示：** 使用GPU可显著加速优化和训练。预测基于相对变化，适合非平稳时间序列。
- **潜在问题：** 如果数据API不可用，手动检查网络或API状态。项目不保证预测准确性，仅供参考。

## 贡献与反馈
欢迎提交Issue或Pull Request。如果有问题，请提供运行日志和环境细节。

**免责声明：** 本项目用于教育和研究目的。比特币投资有风险，请勿基于预测进行实际交易。