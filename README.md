# 比特币价格预测精简版系统

这是一个基于 Transformer 模型的比特币价格预测系统（精简版）。它整合了从数据获取、模型自动优化到训练和预测的完整流程，旨在提供一个清晰、易于理解且功能完备的深度学习预测范例。

## 系统原理

本系统的工作流程遵循数据驱动的机器学习项目标准步骤，具体如下：

1.  **数据获取 (`update_data.py`)**:
    系统首先通过调用币安（Binance）的公共 API 来获取最新的比特币兑美元（BTC/USDT）的日K历史数据。这些数据包含了开盘价、收盘价、最高价、最低价和交易量，并被保存为 `dataset/btc.csv` 文件，作为后续所有分析和训练的基础。

2.  **超参数优化 (`optimize_transformer.py`)**:
    为了让 Transformer 模型达到最佳性能，需要为其选择合适的超参数（如网络层数、学习率等）。此脚本使用 `Optuna` 这个强大的贝叶斯优化框架，自动进行数十次训练试验，以找到能让模型在验证集上损失最小的超参数组合。找到的最佳参数将被自动写入 `config/config_transformer.yaml` 配置文件中。

3.  **模型训练与预测 (`main.py`)**:
    这是系统的核心执行脚本。它首先加载 `config_transformer.yaml` 中的最优配置，然后读取全部历史数据进行模型训练。训练完成后，模型会根据最近的历史数据，对未来5天的比特币收盘价进行预测。

4.  **结果可视化**:
    为了直观地评估预测结果，`main.py` 会生成一个图表。该图表将预测开始日期前的 **25天历史价格** 与模型预测的 **5天未来价格** 绘制在一起，形成一个30天的连续窗口。这使得我们可以直观地看到预测价格与历史趋势的衔接关系。

## 文件结构

```
.
├── config/
│   └── config_transformer.yaml   # (自动生成) 优化后的模型配置文件
├── dataset/
│   └── btc.csv                   # (自动生成) 比特币历史价格数据
├── outputs/
│   └── plots/
│       └── *.png                 # (自动生成) 最终的预测结果图表
├── update_data.py                # 脚本1: 用于获取和更新数据
├── optimize_transformer.py       # 脚本2: 用于模型超参数优化
├── main.py                       # 脚本3: 用于训练、预测和绘图的核心程序
└── requirements.txt              # 项目所需的Python依赖库
```

## 使用方法

请严格按照以下步骤操作，以确保系统正常运行。

### 1. 环境准备

首先，你需要安装所有必需的 Python 库。

**a. 创建虚拟环境 (推荐)**
```bash
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`
```

**b. 安装依赖**
将上面的 `requirements.txt` 文件内容保存后，运行以下命令进行安装：
```bash
pip install -r requirements.txt
```
> **注意**: 对于 **Apple Silicon (M1/M2/M3) 用户**，为了启用 MPS GPU 加速，建议访问 [PyTorch 官网](https://pytorch.org/) 获取并运行针对你的设备推荐的 `pip` 或 `conda` 安装命令，以确保安装了正确版本的 PyTorch。

### 2. 执行流程

请按顺序执行以下三个脚本：

* **第一步：获取数据**
    运行 `update_data.py` 来下载最新的比特币历史价格数据。
    ```bash
    python update_data.py
    ```
    执行成功后，你会在 `dataset` 文件夹下看到 `btc.csv` 文件。

* **第二步：优化模型参数**
    运行 `optimizer.py` 为模型寻找最佳超参数。这个过程会进行多次试验，可能需要几分钟到十几分钟不等。
    ```bash
    python optimizer.py
    ```
    执行成功后，你会在 `config` 文件夹下看到 `config_transformer.yaml` 文件。

* **第三步：训练模型并进行预测**
    运行 `main.py` 来启动最终的训练和预测流程。
    ```bash
    python main.py
    ```
    脚本会加载配置和数据，训练模型，在控制台输出未来5天的预测价格，并最终在 `outputs/plots` 文件夹下生成一张可视化的预测图表。

## 输出文件说明

-   `dataset/btc.csv`: 包含比特币历史价格（UTC时间）的 CSV 文件。
-   `config/config_transformer.yaml`: 包含 Transformer 模型最佳超参数和训练设置的 YAML 文件。
-   `outputs/plots/transformer_30day_window_*.png`: 包含历史价格与预测价格对比的30天窗口图表，文件名中的 `*` 是生成图表时的时间戳。