# 模型代码合集

本仓库包含常用模型的网络结构与基础封装，便于查看与复用。

## 目录结构

- `CNN/`：卷积神经网络结构（SEBlock + ResidualBlock + SimpleCNN）
- `ANN/`：全连接神经网络结构（SimpleANN）
- `RF/`：随机森林回归封装（RandomForestRegressor）
- `SVR/`：支持向量回归封装（SVR）

## 使用说明（示例）

示例仅展示模型定义与调用方式。

```python
from CNN.model import SimpleCNN
model = SimpleCNN(num_channels=13, grid_size=32, num_outputs=1)
```

```python
from RF.model import RFModel
model = RFModel()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

如需完整训练/评估流程，请自行补充数据预处理与训练管线。
