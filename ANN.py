"""
ANN模型实现模块（支持栅格展平输入）

本模块实现了用于CPT（静力触探）数据预测的人工神经网络模型。

主要组件：
    SimpleANN: 全连接神经网络架构
    ANNModel: 完整的模型封装类，包含预处理、训练、预测和评估功能

输入模式：
    - 'flatten_raster': 栅格展平输入（与CNN公平对比）
      输入维度: (N, C×H×W) - 例如13通道×32×32=13312维
    - 'statistics': 统计特征输入（原方案，向后兼容）
      输入维度: (N, C×4+1) - 例如13通道×4+1=53维

特性：
    - 支持任意通道组合、任意栅格尺寸
    - 复用CNN的分通道标准化策略
    - 支持单输出回归（根据target_param选择qc/fs/u2）
    - 支持早停机制防止过拟合
    - 支持GPU/CPU自动切换
    - 完整的训练日志记录
"""
from __future__ import annotations
import sys
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Modle.frame.framework import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from tqdm.auto import tqdm


class SimpleANN(nn.Module):
    """
    全连接神经网络 (MLP)

    网络结构：
        - 输入: (batch, input_size) - 任意维度输入（自适应）
        - 输出: (batch, 1) - 单个参数值（qc、fs或u2）
        - 支持多隐藏层、BatchNorm、Dropout、ReLU激活

    Args:
        input_size: 输入特征维度（自动检测）
        hidden_sizes: 隐藏层大小列表（默认[256, 128, 64]）
        num_outputs: 输出维度（默认1，单输出回归）
        dropout_rate: Dropout率
        activation: 激活函数类型（'relu'或'tanh'）
        use_batch_norm: 是否使用BatchNorm（默认True）
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], num_outputs: int = 1,
                 dropout_rate: float = 0.3, activation: str = 'relu', use_batch_norm: bool = True):
        super().__init__()
        layers = []
        prev = input_size

        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, h))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0 and i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))

            prev = h

        layers.append(nn.Linear(prev, num_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ANNModel(BaseModel):
    """
    ANN模型封装类

    实现单输出回归任务，用于预测CPT参数（qc、fs或u2中的一个）。

    主要特性：
        - 支持两种输入模式：
          1. 'flatten_raster': 栅格展平（与CNN公平对比）
          2. 'statistics': 统计特征（原方案）
        - 复用CNN的标准化策略（分通道标准化 + RobustScaler）
        - 支持任意通道组合、任意栅格尺寸
        - 支持早停机制防止过拟合
        - 支持学习率调度（ReduceLROnPlateau）
        - 支持GPU/CPU自动切换
        - 完整的训练日志记录

    Attributes:
        device: 计算设备（CPU或CUDA）
        model: ANN网络模型实例
        target_param: 目标参数类型（'qc', 'fs', 'u2'）
        input_mode: 输入模式（'flatten_raster' 或 'statistics'）
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        patience: 早停耐心值
        weight_decay: 权重衰减系数
        cnn_channel_scalers: CNN的通道标准化器（flatten_raster模式使用）
        qc_scaler: QC值的RobustScaler（复用CNN的）
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        device_config = config.get('device', 'auto').lower()
        if device_config == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"ANN 使用设备: {self.device} (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self.logger.warning("配置要求使用CUDA，但未检测到GPU，将使用CPU")
                self.device = torch.device('cpu')
        elif device_config == 'cpu':
            self.device = torch.device('cpu')
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"ANN 使用设备: {self.device} (强制使用CPU)")
        else:  # 'auto'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger = logging.getLogger(__name__)
            if self.device.type == 'cuda':
                self.logger.info(f"ANN 使用设备: {self.device} (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self.logger.info(f"ANN 使用设备: {self.device}")

        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.lr = float(config.get('learning_rate', 0.001))
        self.weight_decay = float(config.get('weight_decay', 1e-5))
        self.patience = config.get('patience', 15)

        self.target_param = config.get('target_param', 'qc').lower()
        if self.target_param not in ['qc', 'fs', 'u2']:
            raise ValueError(f"target_param必须是'qc', 'fs'或'u2'之一，当前值: {self.target_param}")

        self.input_mode = config.get('input_mode', 'flatten_raster')
        if self.input_mode not in ['flatten_raster', 'statistics']:
            raise ValueError(f"input_mode必须是'flatten_raster'或'statistics'，当前值: {self.input_mode}")

        self.crop_size = config.get('crop_size', 32)
        self.selected_channels = config.get('selected_channels', None)

        self.cnn_channel_scalers = None
        self.target_scaler = None

        self.num_outputs = 1
        self.model = None
        self.logger = logging.getLogger(__name__)

        if self.input_mode == 'flatten_raster':
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * self.crop_size * self.crop_size
            self.logger.info(
                f"ANN模型初始化: 输入模式=栅格展平, "
                f"栅格尺寸={self.crop_size}×{self.crop_size}, "
                f"通道数={num_channels}, "
                f"预期输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )
        else:
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * 4 + 1
            self.logger.info(
                f"ANN模型初始化: 输入模式=统计特征, "
                f"通道数={num_channels}, "
                f"预期输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )

    def _create_model(self, input_size: int):
        """
        创建ANN模型实例
        Args:
            input_size: 输入特征维度
        """
        if self.input_mode == 'flatten_raster':
            default_hidden = [256, 128, 64]
        else:
            default_hidden = [128, 64, 32]

        hidden = self.config.get('hidden_sizes', default_hidden)
        dropout = self.config.get('dropout_rate', 0.3)
        act = self.config.get('activation', 'relu')
        use_batch_norm = self.config.get('use_batch_norm', True)

        self.model = SimpleANN(
            input_size, hidden, self.num_outputs, dropout, act, use_batch_norm
        ).to(self.device)

        self.logger.info(
            f"创建ANN: 输入维度={input_size}, 隐藏层={hidden}, "
            f"激活={act}, BatchNorm={use_batch_norm}, 输出数={self.num_outputs}"
        )

    def preprocess(self, file_paths: List[str], fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理
        Args:
            file_paths: pickle文件路径列表
            fit_scaler: 是否拟合标准化器
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标值数组)
        """
        from Modle.frame.framework import UnifiedDataPreprocessor

        preprocessor = UnifiedDataPreprocessor(self.config)

        if not fit_scaler:
            if self.cnn_channel_scalers is not None or self.target_scaler is not None:
                scalers_dict = {
                    'cnn_channel_scalers': self.cnn_channel_scalers,
                    'target_scaler': self.target_scaler,
                    'qc_scaler': self.target_scaler,  # 向后兼容
                    'target_mean': None,
                    'target_std': None
                }
                preprocessor.set_scalers(scalers_dict)

        X, y = preprocessor.preprocess(file_paths, fit_scaler=fit_scaler)

        if fit_scaler:
            scalers = preprocessor.get_scalers()
            self.cnn_channel_scalers = scalers['cnn_channel_scalers']
            self.target_scaler = scalers['target_scaler']

        self.logger.info("ANN 目标缩放: 使用固定因子 10 对标准化后的目标再缩放一次 (y_scaled = y / 10)")
        y_scaled = y / 10.0

        return X, y_scaled

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        训练ANN模型
        Args:
            X_train: 训练特征 (N, input_dim)
            y_train: 训练目标 (N,)
            X_val: 验证特征 (M, input_dim)，可选
            y_val: 验证目标 (M,)，可选
        Returns:
            Dict[str, float]: 评估指标字典
        """
        if len(X_train) == 0:
            raise ValueError("训练数据为空")

        input_size = X_train.shape[1]

        if y_train.ndim > 1:
            y_train = y_train.flatten()
        if y_val is not None and y_val.ndim > 1:
            y_val = y_val.flatten()

        self.num_outputs = 1

        self._create_model(input_size)

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)

        best_val = float('inf')
        counter = 0

        self.logger.info(f"开始训练ANN模型: {len(X_train)} 训练样本, {len(X_val) if X_val is not None else 0} 验证样本")
        self.logger.info(
            f"训练配置: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.lr}, "
            f"weight_decay={self.weight_decay}, patience={self.patience}"
        )
        self.logger.info(f"输入模式: {self.input_mode}, 输入维度: {input_size}, 预测参数: {self.target_param.upper()}")
        print(f"开始训练ANN模型: {self.epochs} 个epoch...", file=sys.stderr)

        train_losses = []
        val_losses = []

        from tqdm.auto import tqdm
        epoch_pbar = tqdm(range(1, self.epochs + 1), desc="训练进度", unit="epoch", file=sys.stderr)

        for epoch in epoch_pbar:
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                batch_count += 1

            avg_train = train_loss / batch_count if batch_count > 0 else float('inf')
            train_losses.append(avg_train)

            val_loss = None
            val_metrics_epoch = None
            if X_val is not None and y_val is not None:
                val_metrics = self._eval_internal(X_val, y_val)
                val_loss = val_metrics.get('mse', float('inf'))
                val_losses.append(val_loss)

                try:
                    self.model.eval()
                    with torch.no_grad():
                        Xv_t = torch.FloatTensor(X_val).to(self.device)
                        yv_pred = self.model(Xv_t).cpu().numpy().flatten()

                    yv_true = y_val.flatten()
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    mse_v = mean_squared_error(yv_true, yv_pred)
                    rmse_v = np.sqrt(mse_v)
                    mae_v = mean_absolute_error(yv_true, yv_pred)
                    r2_v = r2_score(yv_true, yv_pred)

                    val_metrics_epoch = {
                        'mse': mse_v,
                        'rmse': rmse_v,
                        'mae': mae_v,
                        'r2': r2_v
                    }
                except Exception as e:
                    self.logger.warning(f"Epoch {epoch}: 计算验证集评估指标时出错: {e}")

                scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    counter = 0
                    torch.save(self.model.state_dict(), 'temp_ann.pth')
                else:
                    counter += 1

                if counter >= self.patience:
                    self.logger.info(f"早停触发 at epoch {epoch}")
                    epoch_pbar.set_postfix({'status': 'Early stopping'})
                    break

                current_lr = optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({
                    "Train Loss": f"{avg_train:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Best": f"{best_val:.4f}",
                    "Patience": f"{counter}/{self.patience}",
                    "LR": f"{current_lr:.2e}"
                })
            else:
                epoch_pbar.set_postfix({"Train Loss": f"{avg_train:.4f}"})

            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Epoch {epoch:03d}/{self.epochs}")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"训练损失 ({self.target_param.upper()}): {avg_train:.6f}")
            if val_loss is not None:
                self.logger.info(f"验证损失 ({self.target_param.upper()}): {val_loss:.6f}")
                if val_metrics_epoch is not None:
                    self.logger.info(
                        f"验证集评估指标 ({self.target_param.upper()}，标准化目标空间): "
                        f"MSE={val_metrics_epoch['mse']:.6f}, "
                        f"RMSE={val_metrics_epoch['rmse']:.6f}, "
                        f"MAE={val_metrics_epoch['mae']:.6f}, "
                        f"R²={val_metrics_epoch['r2']:.6f}"
                    )
            self.logger.info(f"当前学习率: {current_lr:.2e}")
            self.logger.info(f"早停计数器: {counter}/{self.patience}")
            self.logger.info(f"{'='*80}")

        epoch_pbar.close()

        if X_val is not None and os.path.exists('temp_ann.pth'):
            self.model.load_state_dict(torch.load('temp_ann.pth', map_location=self.device))
            os.remove('temp_ann.pth')
            self.logger.info(f"已加载最佳模型（验证损失: {best_val:.6f}）")

        self._plot_training_curves(train_losses, val_losses)

        self.is_trained = True

        train_metrics = self.evaluate(X_train, y_train)
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            for k, v in val_metrics.items():
                train_metrics[f"val_{k}"] = v

        return train_metrics

    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float]):
        """
        绘制并保存训练/验证损失曲线图
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
        """
        if len(train_losses) == 0:
            self.logger.warning("没有训练损失数据，跳过曲线绘制")
            return

        import matplotlib.pyplot as plt

        model_dir = self.config.get('model_dir', '.')
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        curve_file = os.path.join(model_dir, 'ann_training_curve.png')

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if len(val_losses) > 0:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title(f'ANN Training Curve - {self.target_param.upper()}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(curve_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ 训练曲线已保存到: {curve_file}")
        print(f"训练曲线已保存: {curve_file}", file=sys.stderr)

    def _eval_internal(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        内部评估（用于早停）
        Args:
            X_val: 验证特征 (M, input_dim)
            y_val: 验证目标 (M,)
        Returns:
            Dict[str, float]: 评估指标字典
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_val).to(self.device)
            y_p = self.model(X_t).cpu().numpy()

        if y_val.ndim > 1:
            y_val = y_val.flatten()
        if y_p.ndim > 1:
            y_p = y_p.flatten()

        metrics = self._calculate_metrics(y_val, y_p)
        metrics['mse'] = metrics.get(f'{self.target_param}_mse', mean_squared_error(y_val, y_p))
        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算指标（单输出回归）
        Args:
            y_true: (N,) - 真实值
            y_pred: (N,) - 预测值
        Returns:
            Dict[str, float]: 评估指标字典
        """
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        param_name = self.target_param
        mse = mean_squared_error(y_true, y_pred)
        return {
            f"{param_name}_mse": mse,
            f"{param_name}_rmse": np.sqrt(mse),
            f"{param_name}_mae": mean_absolute_error(y_true, y_pred),
            f"{param_name}_r2": r2_score(y_true, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测（单输出回归：返回单个参数值，已反标准化到原始量纲）
        Args:
            X_test: 测试特征
        Returns:
            np.ndarray: 预测值数组 (N,)，已反标准化到原始量纲
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(self.device)
            pred = self.model(X_t).cpu().numpy()

        pred = pred.flatten()

        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            self.logger.info(f"✓ 使用RobustScaler反标准化{self.target_param.upper()}预测值")
            self.logger.info(f"  反标准化前预测值范围: [{pred.min():.4f}, {pred.max():.4f}], 均值={pred.mean():.4f}")
            pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            self.logger.info(f"  反标准化后预测值范围: [{pred.min():.4f}, {pred.max():.4f}], 均值={pred.mean():.4f}")
            if hasattr(self.target_scaler, 'center_') and hasattr(self.target_scaler, 'scale_'):
                self.logger.info(f"  RobustScaler参数: center={self.target_scaler.center_[0]:.3f}, scale={self.target_scaler.scale_[0]:.3f}")
        else:
            self.logger.warning(f"反标准化参数缺失: target_scaler={hasattr(self, 'target_scaler') and self.target_scaler is not None}")
            self.logger.warning("反标准化参数缺失，返回标准化后的预测值（约 -1~1 区间）")

        return pred

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, already_normalized: bool = True) -> Dict[str, float]:
        """
        计算评估指标（单输出回归，在原始量纲下计算）
        Args:
            X_test: 测试特征 (N, input_dim)
            y_test: 测试目标 (N,)，如果already_normalized=True，则为标准化后的值
            already_normalized: y_test是否已经标准化（默认True）
        Returns:
            Dict[str, float]: 评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        if y_test.ndim > 1:
            y_test = y_test.flatten()

        y_pred = self.predict(X_test)

        if already_normalized:
            y_test_original = y_test.copy()
            if hasattr(self, 'target_scaler') and self.target_scaler is not None:
                y_test_original = self.target_scaler.inverse_transform(y_test_original.reshape(-1, 1)).flatten()
                self.logger.info(f"✓ 已反标准化{self.target_param.upper()}真实值到原始量纲")
            else:
                self.logger.warning(f"目标值反标准化参数缺失，使用标准化后的值进行评估")
        else:
            y_test_original = y_test

        if self.target_param == 'qc':
            qc_valid_mask = (y_test_original >= 0) & (y_test_original <= 100)
            invalid_qc_count = (~qc_valid_mask).sum()
            if invalid_qc_count > 0:
                self.logger.warning(f"[评估] 发现{invalid_qc_count}个qc异常值（不在0-100范围），将过滤掉这些样本")
                y_pred = y_pred[qc_valid_mask]
                y_test_original = y_test_original[qc_valid_mask]

        valid_mask = ~(np.isnan(y_test_original) | np.isinf(y_test_original) | 
                       np.isnan(y_pred) | np.isinf(y_pred))
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            self.logger.warning(f"[评估] 发现{invalid_count}个NaN/Inf值，将过滤掉这些样本")
            y_pred = y_pred[valid_mask]
            y_test_original = y_test_original[valid_mask]

        if len(y_test_original) == 0:
            raise ValueError("过滤后没有有效样本用于评估")

        return self._calculate_metrics(y_test_original, y_pred)

    def save(self, path: str):
        """
        保存模型状态、scaler和配置
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        input_size = next(iter(self.model.parameters())).shape[1] if hasattr(self.model, 'network') else None
        if input_size is None:
            input_size = getattr(self.scaler, 'n_features_in_', None)

        data = {
            'state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
            'input_size': input_size,
            'num_outputs': self.num_outputs,
            'target_param': self.target_param,
            'input_mode': self.input_mode,
            'cnn_channel_scalers': self.cnn_channel_scalers,
            'target_scaler': self.target_scaler,
            'target_scaler': self.target_scaler
        }

        if hasattr(self, '_target_scaler_file'):
            data['_target_scaler_file'] = self._target_scaler_file

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"ANN模型保存: {path}")

    @classmethod
    def load(cls, path: str, config: Dict) -> 'ANNModel':
        """
        加载模型
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(config)
        instance.scaler = data['scaler']
        instance.num_outputs = data.get('num_outputs', 1)
        instance.target_param = data.get('target_param', 'qc')
        instance.input_mode = data.get('input_mode', 'flatten_raster')
        instance.cnn_channel_scalers = data.get('cnn_channel_scalers', None)
        instance.target_scaler = data.get('target_scaler', None)

        if '_target_scaler_file' in data:
            instance._target_scaler_file = data['_target_scaler_file']

        instance.target_scaler = data.get('target_scaler', None)
        instance.is_trained = True

        instance._create_model(data['input_size'])
        instance.model.load_state_dict(data['state_dict'])
        instance.model.to(instance.device)

        instance.logger.info(f"ANN模型加载: {path}")
        return instance

if __name__ == "__main__":
    import glob
    import yaml
    from Modle.frame.framework import CPTModelFramework, setup_logging

    config_path = 'Modle/framework_ann.yaml'
    if not os.path.exists(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'framework_ann.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', None)
    log_file_name = log_config.get('framework_log_prefix', None)
    if log_file_name:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f'{log_file_name}_{timestamp}.txt'
    
    if setup_logging:
        log_file_path = setup_logging(log_dir=log_dir, log_file_name=log_file_name)
    else:
        logging.basicConfig(level=logging.INFO)
        log_file_path = None

    logger = logging.getLogger(__name__)

    data_config = config.get('data', {})
    DATA_DIR = data_config.get('data_dir', None)
    if not DATA_DIR:
        raise ValueError("配置文件中未指定 data.data_dir，请检查配置文件")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")
    
    file_pattern = data_config.get('data_file_pattern', '*.pickle')
    
    all_files = glob.glob(os.path.join(DATA_DIR, file_pattern))
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"文件模式: {file_pattern}")
    logger.info(f"总文件数: {len(all_files)}")
    print(f"找到 {len(all_files)} 个数据文件")

    if len(all_files) == 0:
        logger.error(f"错误: 在 {DATA_DIR} 中未找到匹配模式 '{file_pattern}' 的数据文件")
        print(f"错误: 在 {DATA_DIR} 中未找到匹配模式 '{file_pattern}' 的数据文件")
        print(f"提示: 请检查数据目录路径和文件扩展名是否正确")
        exit(1)

    if 'test_files' not in data_config:
        raise ValueError("配置文件缺少必需项: data.test_files，请配置训练、验证、测试集划分")

    test_file_names = data_config['test_files']
    if not isinstance(test_file_names, list) or len(test_file_names) == 0:
        raise ValueError("data.test_files 必须是非空列表")

    test_files = []
    for filename in test_file_names:
        if os.path.isabs(filename):
            test_path = filename
        else:
            test_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(test_path):
            test_files.append(test_path)
        else:
            logger.warning(f"测试文件不存在: {test_path}")

    if len(test_files) == 0:
        auto_select_test = data_config.get('auto_select_test', False)
        auto_test_count = data_config.get('auto_test_count', 4)
        if auto_select_test and len(all_files) >= auto_test_count:
            test_files = all_files[:auto_test_count]
            logger.warning(f"警告: 未找到指定的测试文件，自动选择前{auto_test_count}个文件作为测试集")
            print(f"警告: 未找到指定的测试文件，自动选择前{auto_test_count}个文件作为测试集")
        else:
            logger.error("错误: 未找到测试文件且auto_select_test=False")
            print("错误: 未找到测试文件，请检查配置文件中的test_files设置")
            exit(1)

    logger.info(f"测试集文件数: {len(test_files)}")
    for f in test_files:
        logger.info(f"  - {os.path.basename(f)}")

    test_basenames = {os.path.basename(f) for f in test_files}
    train_val_files = [f for f in all_files if os.path.basename(f) not in test_basenames]

    logger.info(f"\n训练+验证集文件数: {len(train_val_files)} (排除{len(test_files)}个测试文件)")
    logger.info(f"测试集文件数: {len(test_files)}")
    logger.info("数据划分: 训练集:验证集 = 9:1 (使用非测试文件)")
    print(f"使用 {len(train_val_files)} 个文件进行训练 (90%训练集 + 10%验证集)，{len(test_files)} 个文件作为测试集")

    print(f"\n{'=' * 60}")
    print("训练 ANN 模型（栅格展平输入模式）")
    print(f"{'=' * 60}")

    framework = CPTModelFramework(config_dict=config)

    print("\n正在加载和预处理数据...")
    logger.info(f"使用输入模式: {config['model'].get('input_mode', 'flatten_raster')}")

    train_metrics = framework.train(train_val_files, val_size=config.get('training', {}).get('val_size', 0.111))
    logger.info(f"\n训练指标: {train_metrics}")

    OUTPUT_BASE_DIR = data_config.get('output_base_dir', './output')
    output_subdir = data_config.get('output_subdir', 'ANN_results')
    model_filename = data_config.get('model_filename', 'ann_model.pkl')
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    output_dir = os.path.join(OUTPUT_BASE_DIR, output_subdir)
    model_path = os.path.join(OUTPUT_BASE_DIR, model_filename)

    os.makedirs(output_dir, exist_ok=True)

    print("\n正在对测试孔位生成预测结果...")

    test_metrics = framework.evaluate(test_files, output_dir=output_dir)

    framework.save(model_path)

    print(f"\n{'=' * 60}")
    print("ANN 模型训练完成！")
    print(f"{'=' * 60}")
    print(f"模型保存路径: {model_path}")
    print(f"结果保存目录: {output_dir}")
    print(f"  - CSV文件: 每个钻孔的预测结果")
    print(f"  - PNG图片: 真实值与预测值对比曲线")
    print(f"\n测试集评估指标 (基于{len(test_files)}个孔位):")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"{'=' * 60}\n")

    if log_file_path:
        print(f"详细日志已保存至: {log_file_path}")
