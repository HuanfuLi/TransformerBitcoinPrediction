# Easy Bitcoin on Transformer 项目

## 项目概述

Easy Bitcoin on Transformer 是一个基于 Python 的比特币价格预测系统，使用 Transformer 模型进行时间序列预测。该项目整合了比特币历史价格数据、恐惧贪婪指数（Fear & Greed Index），并通过特征工程、相关性分析和超参数优化来提升预测精度。项目支持多特征预测，包括技术指标（如移动平均线、RSI、MACD、布林带）和情绪指标。

项目主要功能包括数据获取、特征工程、特征选择、模型优化、训练与预测，以及可视化结果。它还包含了**早停（Early Stopping）机制**以防止过拟合，能够**自动保存最佳模型**，并提供了一个独立的**基准测试框架**，用于在纯净的人工数据集上验证模型性能。

**当前版本日期：** 2025年7月19日  
**作者/维护者：** HuanfuLi  
**依赖环境：** Python 3.8+

## 主要功能

1.  **数据更新（update\_data.py）**：

      * 从 BitMEX API 获取比特币历史K线数据（开盘价、最高价、最低价、收盘价、交易量）。
      * 从 Alternative.me API 获取恐惧贪婪指数数据。
      * 保存数据到 `dataset/btc.csv` 和 `dataset/fear_greed_index.csv`。

2.  **特征工程（feature\_engineer.py）**：

      * 合并比特币价格和恐惧贪婪指数数据。 
      * 计算移动平均线（MA）、相对强弱指数（RSI）、平滑异同移动平均线（MACD）、布林带（Bollinger Bands）等多种技术指标。
      * 计算情绪衍生特征。 
      * 处理缺失值，裁切初始无效数据，并按类别排序列。
      * 保存最终特征数据集到 `dataset/btc_features.csv`。

3.  **特征选择（feature\_correlation.py）**：

      * 加载 `btc_features.csv`，计算每个特征与目标（收盘价）的Pearson相关系数。 
      * 使用随机森林模型计算特征重要性。 
      * 基于相关性和重要性建议Top N个特征，并保存到 `config/suggested_features.yaml`。 

4.  **模型优化（optimizer.py）**：

      * 加载建议的特征列表。
      * 使用Optuna进行Transformer模型超参数优化（如 `d_model`, `nhead`, `dropout` 等）。
      * 生成包含最优参数、数据范围和特征列表的配置文件 `config/config_transformer.yaml`。

5.  **模型训练与预测（main.py）**：

      * 读取 `config_transformer.yaml` 中的配置。
      * **新增功能**：训练过程中监控验证集的损失，并实现**早停机制**。如果验证损失在设定的耐心值（如40个周期）内没有改善，将自动停止训练以防止过拟合。
      * **新增功能**：在训练结束后，自动将验证集上表现**最佳的模型权重**保存到 `dataset/models/` 目录中，以便未来直接使用。
      * 加载最佳模型，在测试集上进行最终预测，并生成可视化图表。

## 安装要求

  * **Python 版本：** 3.8 或更高。
  * **所需库：** 通过 `pip install -r requirements.txt` 安装：
    ```
    pandas
    numpy
    torch
    optuna
    pyyaml
    matplotlib
    scikit-learn
    requests
    ```
  * **硬件要求：** 支持CUDA或MPS的GPU可显著加速训练（否则使用CPU）。
  * **目录结构：**
    ```
    BitcoinConcise/
    ├── dataset/
    │   ├── models/           # (新增) 保存训练好的模型
    │   ├── btc.csv
    │   ├── btc_artificial_pure.csv  # (新增) 人工数据集
    │   ├── fear_greed_index.csv
    │   └── btc_features.csv
    ├── config/
    ├── outputs/plots/
    ├── artificial_dataset.py     # (新增) 人工数据生成脚本
    ├── bench_test_main.py        # (新增) 基准测试主脚本
    ├── update_data.py
    ├── feature_engineer.py
    ├── feature_correlation.py
    ├── optimizer.py
    ├── main.py
    └── README.md
    ```

## 使用说明

### 标准工作流（真实数据预测）

项目采用模块化设计，按以下顺序运行脚本以完成完整工作流：

1.  **更新数据：**

    ```bash
    python update_data.py
    ```

2.  **进行特征工程：**

    ```bash
    python feature_engineer.py
    ```

3.  **（可选）特征选择：**

    ```bash
    python feature_correlation.py
    ```

4.  **（可选）模型优化：**

    ```bash
    python optimizer.py
    ```

5.  **训练与预测：**

    ```bash
    python main.py
    ```

      * 此脚本会自动加载配置文件，训练模型，并使用早停和模型保存功能。最终生成对真实数据的预测图。

-----

## 🔬 基准测试框架 (Benchmark Testing)

为了独立验证模型的核心学习能力，项目提供了一个基准测试框架。它使用**纯净无噪声的数学曲线**（如直线、正弦波）来测试模型是否能正确学习基本模式，而不是记忆噪声。

### 基准测试工作流

1.  **步骤一：生成人工数据集**

      * 打开 `artificial_dataset.py` 文件。
      * 在顶部的用户配置区，设置 `DATASET_SHAPE`为你想要的曲线形状（如 `"UPWARD_TREND"`, `"SINUSOID"`, `"INVERSE"`）。
      * 运行脚本：
        ```bash
        python artificial_dataset.py
        ```
      * 这将在 `dataset/` 目录下生成一个名为 `btc_artificial_pure.csv` 的纯净数据集。 

2.  **步骤二：基于人工数据进行特征工程**

      * 这是**关键一步**。打开 `feature_engineer.py` 文件。
      * 找到 `__init__` 方法中的这一行：
        ```python
        self.btc_path = os.path.join(data_dir, 'btc.csv')
        ```
      * **临时修改**它，将输入文件名从 `'btc.csv'` 改为 `'btc_artificial_pure.csv'`：
        ```python
        self.btc_path = os.path.join(data_dir, 'btc_artificial_pure.csv') # 临时修改
        ```
      * 保存文件后，运行特征工程脚本：
        ```bash
        python feature_engineer.py
        ```
      * 这将使用你生成的人工数据覆盖 `dataset/btc_features.csv`。

3.  **步骤三：运行基准测试**

      * 直接运行 `bench_test_main.py`：
        ```bash
        python bench_test_main.py
        ```
      * 该脚本被专门设计用于处理纯净数据：
          * 它会自动启用 `IS_BENCHMARK_TRAINING = True` 标志。
          * 它会使用一个**简化的、经过正则化处理的模型配置** (`BENCHMARK_MODEL_CONFIG`) 来防止在简单数据上发生过拟合。
          * 它使用了一个包含**方向性损失**的 `CombinedLoss` 函数，迫使模型学习曲线的正确走向，而不是输出一条平线。
      * 测试完成后，会生成一张图表，展示模型对纯净数学曲线的拟合情况。

**重要提示：** 完成基准测试后，请记得将 `feature_engineer.py` 中的 `self.btc_path` **改回** `'btc.csv'`，以便继续处理真实数据。