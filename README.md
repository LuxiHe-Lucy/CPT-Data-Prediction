## 项目简介

**CPT-Prediction** 是一个用于 CPT（静力触探）数据预测的模型集合，包含：

- **CNN 模型**：卷积神经网络，用于基于栅格化地震属性进行回归预测；
- **ANN 模型**：全连接神经网络（MLP），支持栅格展平或统计特征输入；
- **SVR 模型**：支持向量回归；
- **RF 模型**：随机森林回归。

四类模型使用统一的数据预处理与评估框架，适合做对比实验和工程应用。

## 代码结构

- `frame/framework.py`：数据预处理、统一框架与工具函数。
- `CNN.py`：CNN 模型定义与训练脚本。
- `ANN.py`：ANN 模型定义与训练脚本。
- `SVR.py`：SVR 模型定义与训练脚本。
- `RF.py`：RF 模型定义与训练脚本。
- `framework_cnn.yaml`：CNN 相关配置。
- `framework_ann.yaml`：ANN 相关配置。
- `framework_svr.yaml`：SVR 相关配置。
- `framework_rf.yaml`：RF 相关配置。

> 说明：目前部分配置文件中仍使用 Windows 绝对路径（例如 `D:\CNN\...`），在自己的环境中使用时请根据实际路径进行修改。

## 环境依赖

建议使用 Python 3.9+，并在虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- PyTorch（`torch`、`torchvision`）
- scikit-learn（`scikit-learn`）
- NumPy / SciPy
- Matplotlib
- PyYAML
- tqdm

## 数据准备

1. 将数据预处理为 `pickle` 文件（项目原始代码假定为 14 通道、固定大小的栅格数据）。
2. 将所有 `pickle` 文件放在一个目录下，例如：
   - `./data/dataset_pickle_14通道_64x64`
3. 打开相应的 `framework_*.yaml`，修改以下字段为你的实际路径：

```yaml
data:
  data_dir: '你的数据目录'
  output_base_dir: '你的输出结果目录'
logging:
  log_dir: '你的日志目录（可以与输出目录相同）'
```

## 使用示例

下面给出最常用的脚本调用示例（在项目根目录 `CPT-Prediction` 下执行）。

- **训练 CNN 模型**

```bash
python CNN.py
```

对应配置来自 `framework_cnn.yaml`，会：

- 扫描 `data.data_dir` 下所有 `*.pickle` 文件；
- 按配置划分训练 / 验证 / 测试集；
- 训练 CNN 模型，并在 `data.output_base_dir` 下保存：
  - 训练日志；
  - 预测结果 CSV；
  - 真实值与预测值对比图；
  - 训练好的模型权重。

- **训练 ANN 模型**

```bash
python ANN.py
```

- **训练 SVR 模型**

```bash
python SVR.py
```

- **训练 RF 模型**

```bash
python RF.py
```

> 提示：四个脚本的整体流程类似，仅模型类型和对应的 `framework_*.yaml` 配置不同。初次使用时建议先从 CNN 或 ANN 开始，确认路径配置无误后再运行其它模型。

## GitHub 建议

如果要将本项目上传到 GitHub，推荐：

- 删除或改成相对路径的本地绝对路径（例如 `D:\CNN\...`），避免与其他环境冲突；
- 在 `data/`、`output/` 等目录下放置一个 `.gitkeep`（或在 README 中说明目录结构），而不要直接上传真实数据；
- 在 `requirements.txt` 中维护依赖，方便其他人一键安装并复现结果。

