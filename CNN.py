"""
CNN模型实现模块：用于CPT（静力触探）数据预测的卷积神经网络模型。
主要组件：
    SimpleCNN: CNN网络架构
    CNNModel: 模型封装类（预处理、训练、预测、评估）
"""
from __future__ import annotations


import sys
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Modle.frame.framework import BaseModel, fill_nan_with_nearest, preprocess_qc_values, process_nan_values, preprocess_raster_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
from Modle.frame.framework import PreprocessResult
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from collections import Counter
import json
import joblib
from typing import Dict, Any, List, Tuple, Optional
import logging
import gc
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
from multiprocessing import Pool, cpu_count
from functools import partial
import torch.nn.functional as F
import torchvision.transforms as transforms

EXPECTED_ATTRIBUTES = [
    'Amplitude', 'Envelope', 'InstFreq', 'RMS_5', 'Coherency_5',
    'InstPhase', 'ParaPhase', 'AvgAbsAmp_5', 'AvgPeakAmp_5', 'AvgEnergy_5',
    'depth_to_upper', 'depth_to_surface', 'layer_index', 'layer_thickness',
]


class SeismicDataAugmentation:
    """地震数据增强类"""
    def __init__(self, grid_size=32, p_flip=0.1, p_rotate=0.05, p_noise=0.1):
        self.grid_size = grid_size
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise

    def __call__(self, grid_continuous, grid_layer_index=None):
        """
        对连续通道进行数据增强
        Args:
            grid_continuous: (C, H, W) 连续通道栅格
            grid_layer_index: (H, W) layer_index栅格，可选
        Returns:
            增强后的栅格数据
        """
        if np.random.random() < self.p_flip:
            grid_continuous = np.flip(grid_continuous, axis=2)
            if grid_layer_index is not None:
                grid_layer_index = np.flip(grid_layer_index, axis=1)

        if np.random.random() < self.p_flip:
            grid_continuous = np.flip(grid_continuous, axis=1)
            if grid_layer_index is not None:
                grid_layer_index = np.flip(grid_layer_index, axis=0)

        if np.random.random() < self.p_rotate:
            k = np.random.randint(1, 4)
            grid_continuous = np.rot90(grid_continuous, k, axes=(1, 2))
            if grid_layer_index is not None:
                grid_layer_index = np.rot90(grid_layer_index, k, axes=(0, 1))

        if np.random.random() < self.p_noise:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, grid_continuous.shape)
            grid_continuous = grid_continuous + noise

        return grid_continuous, grid_layer_index


class SeismicDataset(TensorDataset):
    """支持数据增强的地震数据集"""
    def __init__(self, grids_continuous, grids_layer_index, cpts, targets,
                 augment=False, augmentor=None):
        self.augment = augment
        self.augmentor = augmentor

        self.grids_continuous = torch.FloatTensor(grids_continuous)
        self.cpts = torch.FloatTensor(cpts)
        self.targets = torch.FloatTensor(targets)

        if grids_layer_index is not None:
            self.grids_layer_index = torch.LongTensor(grids_layer_index)
            self.use_layer_index = True
        else:
            self.grids_layer_index = None
            self.use_layer_index = False

    def __len__(self):
        return len(self.grids_continuous)

    def __getitem__(self, idx):
        grid_cont = self.grids_continuous[idx]
        cpt = self.cpts[idx]
        target = self.targets[idx]

        if self.augment and self.augmentor is not None:
            grid_cont_np = grid_cont.numpy()
            layer_idx_np = self.grids_layer_index[idx].numpy() if self.use_layer_index else None

            augmented_cont, augmented_idx = self.augmentor(grid_cont_np, layer_idx_np)

            grid_cont = torch.FloatTensor(augmented_cont.copy())
            if self.use_layer_index and augmented_idx is not None:
                layer_idx = torch.LongTensor(augmented_idx.copy())
            else:
                layer_idx = self.grids_layer_index[idx] if self.use_layer_index else None
        else:
            layer_idx = self.grids_layer_index[idx] if self.use_layer_index else None

        if self.use_layer_index:
            return grid_cont, layer_idx, cpt, target
        else:
            return grid_cont, cpt, target


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """残差块，包含跳跃连接"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    """
    卷积神经网络模型（单输出回归）- 空间Embedding融合版本
    输入：
        - 连续通道栅格：(B, C, H, W)
        - 可选 layer_index 栅格：(B, H, W)，通过空间Embedding编码后融合
    输出：
        - (B, 1) 单个参数值

    改进：layer_index通过空间Embedding编码为(B, embedding_dim, H, W)后，
    与连续通道在通道维度拼接，一同输入ResNet进行特征提取。
    """
    def __init__(self, num_channels: int = 13, grid_size: int = 32, num_outputs: int = 1,
                 hidden_sizes: List[int] = [128, 128], dropout_rate: float = 0.5,
                 conv_dropout: float = 0.2, num_layer_indices: int = 10, embedding_dim: int = 16,
                 use_layer_index: bool = True):
        super(SimpleCNN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.use_layer_index = use_layer_index

        total_input_channels = num_channels
        if self.use_layer_index:
            total_input_channels += embedding_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(total_input_channels, 64, kernel_size=3, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_layer_index:
            self.spatial_embedding = nn.Embedding(num_layer_indices, embedding_dim)
        else:
            self.spatial_embedding = None

        conv_feature_size = 256
        fc_input_size = conv_feature_size

        layers = []
        prev_size = fc_input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_outputs))

        self.fc = nn.Sequential(*layers)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x_continuous: torch.Tensor, layer_index: torch.Tensor = None) -> torch.Tensor:
        """前向传播"""
        if self.use_layer_index and (layer_index is not None):
            layer_embedding = self.spatial_embedding(layer_index)
            layer_embedding = layer_embedding.permute(0, 3, 1, 2)
            x = torch.cat([x_continuous, layer_embedding], dim=1)
        else:
            x = x_continuous

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        conv_out = self.adaptive_pool(x)
        conv_out = conv_out.flatten(1)

        return self.fc(conv_out)


class CNNModel(BaseModel):
    """
    CNN 模型封装类：用于预测单个 CPT 参数（qc / fs / u2）
    特性：
        - 分通道标准化（Log1p+Standard / Z-score / MinMax）
        - Embedding 编码 layer_index 类别通道
        - 支持 GPU / CPU 自动切换和 AMP 混合精度
        - 完整的训练 / 验证 / 评估 / 保存 / 加载流程
    """
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any] = None):
        super().__init__(config)
        self._parent_config = parent_config

        device_config = config.get('device', 'auto').lower()
        if device_config == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"CNN 使用设备: {self.device} (GPU: {torch.cuda.get_device_name(0)})")
                self.logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.logger.warning("配置要求使用CUDA，但未检测到GPU，将使用CPU")
                self.device = torch.device('cpu')
                self.logger.info(f"CNN 使用设备: {self.device}")
        elif device_config == 'cpu':
            self.device = torch.device('cpu')
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"CNN 使用设备: {self.device} (强制使用CPU)")
        else:  # 'auto'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger = logging.getLogger(__name__)
            if self.device.type == 'cuda':
                self.logger.info(f"CNN 使用设备: {self.device} (GPU: {torch.cuda.get_device_name(0)})")
                self.logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                self.logger.info(f"CNN 使用设备: {self.device}")

        self.epochs = config.get('epochs', 150)
        self.batch_size = config.get('batch_size', 128)
        self.lr = float(config.get('learning_rate', 1e-2))
        self.weight_decay = float(config.get('weight_decay', 1e-4))
        self.patience = config.get('patience', 20)
        self.gradient_clip = float(config.get('gradient_clip', 1.0))
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'

        self.use_augmentation = config.get('use_augmentation', True)
        if self.use_augmentation:
            self.augmentor = SeismicDataAugmentation(grid_size=self.config.get('grid_size', 32))

        self.logger.info(f"训练配置 - Epochs: {self.epochs}, Batch Size: {self.batch_size}, LR: {self.lr}")
        self.logger.info(f"正则化 - Weight Decay: {self.weight_decay}, Gradient Clip: {self.gradient_clip}")
        self.logger.info(f"早停 - Patience: {self.patience}")

        self.target_param = config.get('target_param', 'qc').lower()
        if self.target_param not in ['qc', 'fs', 'u2']:
            raise ValueError(f"target_param必须是'qc', 'fs'或'u2'之一，当前值: {self.target_param}")
        self.logger.info(f"CNN模型将预测单个参数: {self.target_param.upper()}")

        self.selected_channels = config.get('selected_channels', list(range(14)))
        if not isinstance(self.selected_channels, list):
            raise ValueError("selected_channels必须是整数列表")
        if not all(isinstance(ch, int) and 0 <= ch < 14 for ch in self.selected_channels):
            raise ValueError("selected_channels中的所有值必须是0-13之间的整数")
        self.logger.info(f"选择的输入通道: {self.selected_channels} (共{len(self.selected_channels)}个通道)")

        self.continuous_channels = [ch for ch in self.selected_channels if ch != 12]
        self.use_layer_index = 12 in self.selected_channels

        self.model = None
        self.grid_scalers = None
        self.cpt_scaler = None
        self.target_scalers = None
        self.target_max_abs = None
        self.target_scaler = None
        self.channel_scalers = None
        self.channel_scale_factors = None
        self.num_layer_indices = None

    def _create_model(self):
        """根据配置创建 CNN 模型实例"""
        num_channels = len(self.continuous_channels)
        grid_size = self.config.get('grid_size', 32)
        num_layer_indices = self.num_layer_indices if self.num_layer_indices is not None else self.config.get('num_layer_indices', 10)

        if hasattr(self, 'layer_index_mapping') and self.layer_index_mapping is not None:
            actual_num_classes = self.layer_index_mapping.get('num_classes', num_layer_indices)
            if actual_num_classes > num_layer_indices:
                self.logger.info(f"根据数据调整num_layer_indices: {num_layer_indices} -> {actual_num_classes}")
                num_layer_indices = actual_num_classes

        embedding_dim = self.config.get('embedding_dim', 16)

        self.model = SimpleCNN(
            num_channels=num_channels,
            grid_size=grid_size,
            num_outputs=1,
            hidden_sizes=self.config.get('hidden_sizes', [128, 128]),
            dropout_rate=float(self.config.get('dropout', 0.5)),
            conv_dropout=float(self.config.get('conv_dropout', 0.2)),
            num_layer_indices=num_layer_indices,
            embedding_dim=embedding_dim,
            use_layer_index=self.use_layer_index
        ).to(self.device)

    def preprocess(
            self,
            file_paths: List[str],
            normalize: bool = False,
            fit_scaler: bool = True,
            crop_size: int = 32
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        数据预处理
        Returns:
            grids_continuous: (N, num_continuous_channels, H, W)
            grids_layer_index: (N, H, W) 或 None
            cpts: (N, 1)
            targets: (N,)
        """
        from frame.framework import preprocess_raster_data, PreprocessResult

        if fit_scaler:
            result: PreprocessResult = preprocess_raster_data(
                file_paths,
                self.config,
                fit_scaler=True,
                crop_size=crop_size,
                channel_scalers=None
            )
        else:
            if self.channel_scalers is None:
                raise ValueError(
                    "测试阶段（fit_scaler=False）模型的channel_scalers为None\n"
                    "请确保先进行训练或加载已训练的模型"
                )
            result: PreprocessResult = preprocess_raster_data(
                file_paths,
                self.config,
                fit_scaler=False,
                crop_size=crop_size,
                channel_scalers=self.channel_scalers,
                qc_scaler=self.target_scaler,
            )

        X_grid_continuous = result.grid_continuous
        X_grid_layer_index = result.grid_layer_index
        X_cpt = result.cpt
        y = result.targets
        scalers = result.scalers

        if fit_scaler:
            if scalers is None:
                raise RuntimeError("训练阶段返回的scalers不应为None")

            self.channel_scalers = scalers['channel_scalers']
            self.target_scaler = scalers['target_scaler']
            self.use_layer_index = scalers.get('use_layer_index', False)
            self.num_layer_indices = scalers.get('num_layer_indices', None)
            self.layer_index_mapping = scalers.get('layer_index_mapping', None)

            if f'_{self.target_param}_scaler_file' in scalers:
                self._qc_scaler_file = scalers[f'_{self.target_param}_scaler_file']

            self.logger.info("✓ 训练阶段：已保存所有标准化器")
        else:
            if scalers is not None:
                self.logger.warning("测试阶段返回了非None的scalers，这可能导致数据泄露！")
            self.logger.info("✓ 测试阶段：使用已训练的标准化器")

        return X_grid_continuous, X_grid_layer_index, X_cpt, y

    def train(self, X_grid, X_grid_layer_index, X_cpt, y_train, X_val_grid=None, X_val_grid_layer_index=None, X_val_cpt=None, y_val=None):
        """
        训练模型
        Args:
            X_grid: 连续通道栅格
            X_grid_layer_index: layer_index 栅格（可选）
            X_cpt: CPT 特征
            y_train: 标准化后的训练标签
            X_val_* / y_val: 验证集（可选）
        Returns:
            评估指标字典（MSE, RMSE, MAE, R²）
        """
        self._create_model()

        if len(y_train) > 0:
            output_layer = None
            for layer in reversed(self.model.fc):
                if isinstance(layer, nn.Linear):
                    output_layer = layer
                    break

            if output_layer is not None:
                nn.init.normal_(output_layer.weight, mean=0.0, std=0.1)
                if hasattr(self, 'target_scaler') and self.target_scaler is not None:
                    nn.init.constant_(output_layer.bias, 0.0)
                    self.logger.info(f"输出层初始化: 偏置=0.0 (RobustScaler标准化后的中位数), "
                                   f"标准化后目标值范围=[{np.min(y_train):.2f}, {np.max(y_train):.2f}], "
                                   f"原始中位数={self.target_scaler.center_[0]:.2f}, IQR scale={self.target_scaler.scale_[0]:.2f}")
                elif hasattr(self, 'target_mean') and self.target_mean is not None:
                    nn.init.constant_(output_layer.bias, 0.0)
                    self.logger.info(f"输出层初始化: 偏置=0.0 (标准化后的均值), "
                                   f"标准化后目标值范围=[{np.min(y_train):.2f}, {np.max(y_train):.2f}], "
                                   f"原始目标值均值={self.target_mean:.2f}, 标准差={self.target_std:.2f}")
                else:
                    nn.init.constant_(output_layer.bias, 0.0)
                    self.logger.info(f"输出层初始化: 偏置=0.0, 目标值范围=[{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        else:
            self.logger.warning("训练数据为空，使用默认输出层初始化")

        # 确保 y 为 1D
        if y_train.ndim > 1:
            y_train = y_train.flatten()
        if y_val is not None and y_val.ndim > 1:
            y_val = y_val.flatten()

        self.logger.info(f"训练开始前目标值统计: y_train范围=[{y_train.min():.4f}, {y_train.max():.4f}]")

        X_grid_continuous = X_grid

        self.logger.info(f"训练数据形状: X_grid_continuous={X_grid_continuous.shape}, "
                         f"X_grid_layer_index={X_grid_layer_index.shape if X_grid_layer_index is not None else None}, "
                         f"X_cpt={X_cpt.shape}, y_train={y_train.shape}")

        if self.use_layer_index and X_grid_layer_index is not None:
            train_ds = SeismicDataset(
                X_grid_continuous, X_grid_layer_index, X_cpt, y_train,
                augment=self.use_augmentation, augmentor=self.augmentor if self.use_augmentation else None
            )
        else:
            train_ds = SeismicDataset(
                X_grid_continuous, None, X_cpt, y_train,
                augment=self.use_augmentation, augmentor=self.augmentor if self.use_augmentation else None
            )
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )

        loss_type = self.config.get('loss_type', 'mse').lower()
        self.loss_type = loss_type

        if loss_type == 'mae':
            criterion = nn.L1Loss(reduction='mean')
        elif loss_type == 'huber':
            criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        elif loss_type == 'mse':
            criterion = nn.MSELoss(reduction='mean')
        else:
            criterion = nn.MSELoss(reduction='mean')

        self.logger.info(f"使用损失函数: {loss_type.upper()}, 预测参数: {self.target_param.upper()}")

        if self.use_amp:
            self.scaler = GradScaler('cuda')
            self.logger.info("启用混合精度训练 (AMP)")

        from datetime import datetime
        log_dir = None
        parent_config = getattr(self, '_parent_config', None)
        if parent_config and 'logging' in parent_config:
            log_dir = parent_config['logging'].get('log_dir')

        if log_dir is None:
            log_dir = self.config.get('log_dir', 'logs')
            if not os.path.isabs(log_dir):
                model_dir = self.config.get('model_dir', '.')
                if not os.path.isabs(model_dir):
                    model_dir = os.path.abspath(model_dir)
                log_dir = os.path.join(model_dir, log_dir)

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')

        self.model_save_dir = log_dir
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        self.logger.propagate = False
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.info(f"训练日志保存到: {log_file}")
        self.logger.info("="*80)
        self.logger.info("训练配置:")
        self.logger.info(f"  设备: {self.device}")
        self.logger.info(f"  批次大小: {self.batch_size}")
        self.logger.info(f"  学习率: {self.lr}")
        self.logger.info(f"  训练集样本数: {len(X_grid)}")
        self.logger.info(f"  验证集样本数: {len(X_val_grid) if X_val_grid is not None else 0}")
        self.logger.info(f"  混合精度训练: {self.use_amp}")
        self.logger.info("="*80)

        best_loss = float('inf')
        patience_cnt = 0
        final_train_loss = None

        epoch_pbar = tqdm(range(1, self.epochs + 1), desc="训练进度", unit="epoch",
                         file=sys.stderr,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for epoch in epoch_pbar:
            self.model.train()
            loss_sum = 0.0
            batch_count = 0

            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}",
                             leave=False, unit="batch",
                             file=sys.stderr,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

            for batch_data in batch_pbar:
                if self.use_layer_index:
                    g_cont, g_idx, c, t = batch_data
                    g_cont, g_idx, c, t = g_cont.to(self.device), g_idx.to(self.device), c.to(self.device), t.to(self.device)
                else:
                    g_cont, c, t = batch_data
                    g_cont, c, t = g_cont.to(self.device), c.to(self.device), t.to(self.device)
                    g_idx = None

                if t.ndim > 1:
                    t = t.flatten()

                if batch_count == 0 and epoch == 1:
                    log_msg = f"输入数据统计: 连续通道范围=[{g_cont.min().item():.2f}, {g_cont.max().item():.2f}], "
                    if self.use_layer_index:
                        log_msg += f"layer_index范围=[{g_idx.min().item()}, {g_idx.max().item()}], "
                    log_msg += f"t范围=[{t.min().item():.2f}, {t.max().item():.2f}], "
                    log_msg += f"g_cont均值={g_cont.mean().item():.2f}, t均值={t.mean().item():.2f}"
                    self.logger.info(log_msg)
                    if torch.any(torch.isnan(g_cont)) or torch.any(torch.isinf(g_cont)):
                        self.logger.error(f"输入栅格数据包含NaN/Inf: NaN={torch.sum(torch.isnan(g_cont)).item()}, Inf={torch.sum(torch.isinf(g_cont)).item()}")
                    if torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                        self.logger.error(f"目标值包含NaN/Inf: NaN={torch.sum(torch.isnan(t)).item()}, Inf={torch.sum(torch.isinf(t)).item()}")

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast(device_type='cuda'):
                        pred = self.model(g_cont, layer_index=g_idx if self.use_layer_index else None)
                        if pred.ndim > 1:
                            pred = pred.flatten()

                        if batch_count == 0 and epoch == 1:
                            self.logger.info(f"模型输出统计: pred范围=[{pred.min().item():.2f}, {pred.max().item():.2f}], "
                                           f"pred均值={pred.mean().item():.2f}")
                            if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
                                self.logger.error(f"模型输出包含NaN/Inf: NaN={torch.sum(torch.isnan(pred)).item()}, Inf={torch.sum(torch.isinf(pred)).item()}")

                        loss = criterion(pred, t)

                    if torch.isnan(loss) or torch.isinf(loss):
                        if batch_count < 5:
                            self.logger.warning(f"Epoch {epoch}, Batch {batch_count}: 损失值为 {loss.item()}")
                            self.logger.warning(f"  - 输入g_cont范围: [{g_cont.min().item():.2f}, {g_cont.max().item():.2f}]")
                            self.logger.warning(f"  - 目标t范围: [{t.min().item():.2f}, {t.max().item():.2f}]")
                            self.logger.warning(f"  - 预测pred范围: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
                        continue

                    loss_value = loss.item()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(g_cont, layer_index=g_idx if self.use_layer_index else None)
                    if pred.ndim > 1:
                        pred = pred.flatten()

                    if batch_count == 0 and epoch == 1:
                        self.logger.info(f"模型输出统计: pred范围=[{pred.min().item():.2f}, {pred.max().item():.2f}], "
                                       f"pred均值={pred.mean().item():.2f}")
                        if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
                            self.logger.error(f"模型输出包含NaN/Inf: NaN={torch.sum(torch.isnan(pred)).item()}, Inf={torch.sum(torch.isinf(pred)).item()}")

                    loss = criterion(pred, t)
                    if torch.isnan(loss) or torch.isinf(loss):
                        if batch_count < 5:
                            self.logger.warning(f"Epoch {epoch}, Batch {batch_count}: 损失值为 {loss.item()}")
                            self.logger.warning(f"  - 输入g_cont范围: [{g_cont.min().item():.2f}, {g_cont.max().item():.2f}]")
                            self.logger.warning(f"  - 目标t范围: [{t.min().item():.2f}, {t.max().item():.2f}]")
                            self.logger.warning(f"  - 预测pred范围: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
                        continue
                    loss_value = loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    optimizer.step()

                loss_sum += loss_value * g_cont.size(0)
                batch_count += 1

                scheduler.step()

                current_loss = loss_sum / (batch_count * self.batch_size)
                current_lr = optimizer.param_groups[0]['lr']
                batch_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'lr': f'{current_lr:.2e}'})
                if self.device.type == 'cuda' and batch_count % 10 == 0:
                    torch.cuda.empty_cache()

            train_loss = loss_sum/len(train_ds) if batch_count > 0 else float('inf')
            final_train_loss = train_loss

            if np.isnan(train_loss) or np.isinf(train_loss):
                self.logger.error(f"Epoch {epoch}: 训练损失为 {train_loss}, 训练失败！")
                self.logger.error("可能原因：1) 数据未标准化导致数值过大 2) 学习率过大 3) 数据中包含异常值")
                break

            if X_val_grid is not None:
                val_loss = self._val_loss(X_val_grid, X_val_grid_layer_index, X_val_cpt, y_val, criterion)
                if np.isnan(val_loss) or np.isinf(val_loss):
                    self.logger.error(f"Epoch {epoch}: 验证损失为 {val_loss}, 训练失败！")
                    break

                try:
                    val_metrics = self.evaluate(
                        X_val_grid,
                        X_val_layer_index,
                        X_val_cpt,
                        y_val,
                        already_normalized=True
                    )
                except Exception as e:
                    self.logger.error(f"Epoch {epoch}: 计算验证指标时出错: {e}")
                    val_metrics = None

                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    patience_cnt = 0
                    best_model_path = os.path.join(self.model_save_dir, 'best_cnn.tmp.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    self.logger.info(f"保存最好模型: {best_model_path}")

                    if epoch % 5 == 0:
                        epoch_model_path = os.path.join(self.model_save_dir, f'best_cnn_epoch_{epoch}.pth')
                        torch.save(self.model.state_dict(), epoch_model_path)
                        self.logger.info(f"保存第{epoch}轮最好模型: {epoch_model_path}")
                else:
                    patience_cnt += 1

                postfix = {
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'best': f'{best_loss:.4f}',
                    'patience': f'{patience_cnt}/{self.patience}'
                }
                epoch_pbar.set_postfix(postfix)

                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Epoch {epoch:03d}/{self.epochs}")
                self.logger.info(f"{'='*80}")
                self.logger.info(f"训练损失 ({self.target_param.upper()}): {train_loss:.6f}")
                self.logger.info(f"验证损失 ({self.target_param.upper()}): {val_loss:.6f}")

                if val_metrics is not None:
                    param_name = self.target_param.upper()
                    mse = val_metrics.get(f'{self.target_param}_mse', float('nan'))
                    rmse = val_metrics.get(f'{self.target_param}_rmse', float('nan'))
                    mae = val_metrics.get(f'{self.target_param}_mae', float('nan'))
                    r2 = val_metrics.get(f'{self.target_param}_r2', float('nan'))
                    self.logger.info(
                        f"验证集评估指标 ({param_name}，反归一化后): "
                        f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}"
                    )
                else:
                    self.logger.warning("本轮未能计算验证集评估指标。")

                self.logger.info(f"最佳验证损失: {best_loss:.6f}")
                self.logger.info(f"当前学习率: {current_lr:.2e}")
                self.logger.info(f"早停计数器: {patience_cnt}/{self.patience}")
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"GPU显存使用: {gpu_memory:.2f} GB")
                self.logger.info(f"{'='*80}")

                if patience_cnt >= self.patience:
                    self.logger.info("Early stopping triggered!")
                    epoch_pbar.set_postfix({'status': 'Early stopping'})
                    break
            else:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}'})
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Epoch {epoch:03d}/{self.epochs}")
                self.logger.info(f"{'='*80}")
                self.logger.info(f"训练损失 ({self.target_param.upper()}): {train_loss:.6f}")
                self.logger.info(f"当前学习率: {current_lr:.2e}")
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    self.logger.info(f"GPU显存使用: {gpu_memory:.2f} GB")
                self.logger.info(f"{'='*80}")

        epoch_pbar.close()

        best_model_path = os.path.join(self.model_save_dir, 'best_cnn.tmp.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=False))
            os.remove(best_model_path)
            self.logger.info(f"已加载最好模型: {best_model_path}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info(f"\n{'='*80}")
        self.logger.info("训练完成！")
        self.logger.info(f"{'='*80}")
        if X_val_grid is not None:
            self.logger.info(f"最终最佳验证损失: {best_loss:.6f}")
            if final_train_loss is not None:
                self.logger.info(f"最终训练损失: {final_train_loss:.6f}")
            self.logger.info(f"总训练轮数: {epoch}/{self.epochs}")
        else:
            if final_train_loss is not None:
                self.logger.info(f"最终训练损失: {final_train_loss:.6f}")
            self.logger.info(f"总训练轮数: {epoch}/{self.epochs}")
        self.logger.info(f"{'='*80}")

        handlers = self.logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()

        self.is_trained = True

        eval_grid = X_val_grid if X_val_grid is not None and len(X_val_grid) > 0 else X_grid
        eval_cpt = X_val_cpt if X_val_cpt is not None and len(X_val_cpt) > 0 else X_cpt
        eval_y = y_val if y_val is not None and len(y_val) > 0 else y_train
        eval_grid_layer_index = X_val_grid_layer_index if X_val_grid is not None and len(
            X_val_grid) > 0 else X_grid_layer_index
        return self.evaluate(eval_grid, eval_grid_layer_index, eval_cpt, eval_y, already_normalized=False)

    def _val_loss(self, g, g_layer_index, c, y, criterion):
        """计算验证集平均损失"""
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_samples = 0

            if y.ndim > 1:
                y = y.flatten()


            g_cont = g
            g_idx = g_layer_index

            if self.use_layer_index and g_idx is not None:
                ds = TensorDataset(
                    torch.FloatTensor(g_cont),
                    torch.LongTensor(g_idx),
                    torch.FloatTensor(c),
                    torch.FloatTensor(y)
                )
            else:
                ds = TensorDataset(
                    torch.FloatTensor(g_cont),
                    torch.FloatTensor(c),
                    torch.FloatTensor(y)
                )

            loader = DataLoader(ds, batch_size=self.batch_size * 4)
            for batch_data in loader:
                if self.use_layer_index and g_idx is not None:
                    gb_cont, gb_idx, cb, tb = batch_data
                    gb_cont, gb_idx, cb, tb = gb_cont.to(self.device), gb_idx.to(self.device), cb.to(
                        self.device), tb.to(self.device)
                else:
                    gb_cont, cb, tb = batch_data
                    gb_cont, cb, tb = gb_cont.to(self.device), cb.to(self.device), tb.to(self.device)
                    gb_idx = None

                if tb.ndim > 1:
                    tb = tb.flatten()

                if self.use_amp:
                    with autocast(device_type='cuda'):
                        pred = self.model(gb_cont, layer_index=gb_idx)
                        if pred.ndim > 1:
                            pred = pred.flatten()
                        batch_loss = criterion(pred, tb)
                else:
                    pred = self.model(gb_cont, layer_index=gb_idx)
                    if pred.ndim > 1:
                        pred = pred.flatten()
                    batch_loss = criterion(pred, tb)

                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    continue

                batch_size = gb_cont.size(0)
                total_loss += batch_loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss

    def predict(self, grids, grids_layer_index, cpts, already_normalized: bool = False) -> np.ndarray:
        """
        预测单个参数值（返回真实值范围）
        Args:
            grids: 连续通道栅格 (N, C, H, W)
            grids_layer_index: layer_index 栅格 (N, H, W) 或 None
            cpts: CPT 特征
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        grids_continuous = grids

        self.model.eval()
        if self.use_layer_index and grids_layer_index is not None:
            grids_cont_tensor = torch.FloatTensor(grids_continuous)
            grids_idx_tensor = torch.LongTensor(grids_layer_index)
            loader = DataLoader(TensorDataset(grids_cont_tensor, grids_idx_tensor), batch_size=self.batch_size * 4)
        else:
            grids_cont_tensor = torch.FloatTensor(grids_continuous)
            loader = DataLoader(TensorDataset(grids_cont_tensor), batch_size=self.batch_size * 4)

        preds = []
        with torch.no_grad():
            for batch in loader:
                if self.use_layer_index and grids_layer_index is not None:
                    g_cont = batch[0].to(self.device)
                    g_idx = batch[1].to(self.device)
                    pred = self.model(g_cont, layer_index=g_idx).cpu().numpy()
                else:
                    g_cont = batch[0].to(self.device)
                    pred = self.model(g_cont, layer_index=None).cpu().numpy()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)

        if preds.ndim > 1:
            preds = preds.flatten()

        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            self.logger.info(f"✓ 使用RobustScaler反标准化{self.target_param.upper()}预测值")
            self.logger.info(f"  反标准化前预测值范围: [{preds.min():.4f}, {preds.max():.4f}], 均值={preds.mean():.4f}")
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            self.logger.info(f"  反标准化后预测值范围: [{preds.min():.4f}, {preds.max():.4f}], 均值={preds.mean():.4f}")
            self.logger.info(f"  RobustScaler参数: center={self.target_scaler.center_[0]:.3f}, scale={self.target_scaler.scale_[0]:.3f}")
        else:
            self.logger.warning(f"反归一化参数缺失: target_scaler={hasattr(self, 'target_scaler') and self.target_scaler is not None}")
            self.logger.warning("反归一化参数缺失，返回标准化后的预测值（约 -1~1 区间）")

        return preds

    def evaluate(self, grids, grids_layer_index, cpts, y_true, already_normalized=False):
        """
        评估模型性能，返回 MSE / RMSE / MAE / R²（目标值为原始量纲）
        """
        pred = self.predict(grids, grids_layer_index, cpts, already_normalized)

        if y_true.ndim > 1:
            y_true = y_true.flatten()

        if already_normalized:
            y_true_original = y_true.copy()
            if hasattr(self, 'target_scaler') and self.target_scaler is not None:
                y_true_original = self.target_scaler.inverse_transform(y_true_original.reshape(-1, 1)).flatten()
            else:
                y_true_original = y_true
                self.logger.warning(f"目标值反归一化参数缺失，使用标准化后的值进行评估")
        else:
            y_true_original = y_true

        if self.target_param == 'qc':
            qc_valid_mask = (y_true_original >= -1) & (y_true_original <= 200)
            invalid_qc_count = (~qc_valid_mask).sum()
            if invalid_qc_count > 0:
                self.logger.warning(f"[评估] 发现{invalid_qc_count}个qc异常值（不在-1到200范围），将过滤掉这些样本")
                pred = pred[qc_valid_mask]
                y_true_original = y_true_original[qc_valid_mask]

        valid_mask = ~(np.isnan(pred) | np.isinf(pred) | np.isnan(y_true_original) | np.isinf(y_true_original))
        n_valid = np.sum(valid_mask)
        n_total = len(pred)

        if n_valid == 0:
            self.logger.error("所有预测值都是NaN/Inf，无法计算评估指标")
            param_name = self.target_param
            return {
                f'{param_name}_mse': float('inf'),
                f'{param_name}_rmse': float('inf'),
                f'{param_name}_mae': float('inf'),
                f'{param_name}_r2': float('-inf')
            }

        if n_valid < n_total:
            n_invalid = n_total - n_valid
            self.logger.warning(f"评估时发现 {n_invalid}/{n_total} 个无效预测值（NaN/Inf），将只使用有效值计算指标")
            pred_valid = pred[valid_mask]
            y_true_valid = y_true_original[valid_mask]
        else:
            pred_valid = pred
            y_true_valid = y_true_original

        metrics = {}
        metrics['mse'] = mean_squared_error(y_true_valid, pred_valid)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true_valid, pred_valid)
        metrics['r2'] = r2_score(y_true_valid, pred_valid)


        param_name = self.target_param
        result = {
            f'{param_name}_mse': metrics['mse'],
            f'{param_name}_rmse': metrics['rmse'],
            f'{param_name}_mae': metrics['mae'],
            f'{param_name}_r2': metrics['r2']
        }

        return result

    def save(self, path: str):
        """保存模型（权重 + 标准化参数 + 配置）"""

        save_target_mean = getattr(self, 'target_mean', None)
        save_target_std = getattr(self, 'target_std', None)

        if self.target_param.lower() == 'fs':
            expected_mean = 0.0932
            expected_std = 0.0882

            if save_target_mean is not None and save_target_std is not None:
                mean_diff = abs(save_target_mean - expected_mean)
                std_diff = abs(save_target_std - expected_std)

                if mean_diff > 0.01 or std_diff > 0.01:
                    print(f"⚠️ 保存前FS标准化参数异常!")
                    print(f"  当前值: mean={save_target_mean:.4f}, std={save_target_std:.4f}")
                    print(f"  修正为: mean={expected_mean:.4f}, std={expected_std:.4f}")

                    save_target_mean = expected_mean
                    save_target_std = expected_std
            else:
                save_target_mean = expected_mean
                save_target_std = expected_std
                print(f"✓ 保存使用默认FS标准化参数: mean={save_target_mean:.4f}, std={save_target_std:.4f}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'grid_scalers': self.grid_scalers,
            'grid_max_abs': getattr(self, 'grid_max_abs', None),
            'cpt_scaler': self.cpt_scaler,
            'target_scaler': getattr(self, 'target_scaler', None),
            'channel_scalers': getattr(self, 'channel_scalers', None) or {str(ch): {'method': 'zscore', 'mean': 0.0, 'std': 1.0} for ch in [0, 1, 2, 3, 4, 5, 7]},
            'num_layer_indices': getattr(self, 'num_layer_indices', None),
            'layer_index_mapping': getattr(self, 'layer_index_mapping', None),
            'config': self.config
        }, path)

    @classmethod
    def load(cls, path: str, config: Dict):
        """从文件加载模型（包含权重、标准化参数、配置等）"""
        data = torch.load(path, map_location='cpu')
        obj = cls(config)
        channel_scalers_data = data.get('channel_scalers', None)
        obj.num_layer_indices = data.get('num_layer_indices', None)
        obj.layer_index_mapping = data.get('layer_index_mapping', None)
        if obj.num_layer_indices is None and obj.layer_index_mapping is not None:
            obj.num_layer_indices = len(obj.layer_index_mapping)

        obj._create_model()
        obj.model.load_state_dict(data['model_state_dict'])
        obj.model.to(obj.device)

        obj.grid_scalers = data.get('grid_scalers', None)
        obj.grid_max_abs = data.get('grid_max_abs', None)
        obj.cpt_scaler = data.get('cpt_scaler', None)
        obj.target_scalers = data.get('target_scalers', None)
        obj.target_max_abs = data.get('target_max_abs', None)
        obj.target_mean = data.get('target_mean', None)
        obj.target_std = data.get('target_std', None)
        obj.qc_scaler = data.get('qc_scaler', None)

        obj.target_scaler = data.get('target_scaler', None)
        if obj.target_scaler is not None:
            obj.logger.info(f"加载模型: {obj.target_param.upper()}目标值RobustScaler已加载")
        else:
            obj.logger.warning(f"加载模型: 未找到{obj.target_param.upper()}目标值scaler")

        if channel_scalers_data is not None:
            continuous_channels = [0, 1, 2, 3, 4, 5, 7]

            if 'unified_max_abs' in channel_scalers_data:
                obj.logger.warning("加载模型: 检测到旧版本的统一归一化参数，将转换为Z-score标准化")
                global_max_abs = float(channel_scalers_data['unified_max_abs'])
                obj.channel_scalers = {}
                for orig_ch in continuous_channels:
                    obj.channel_scalers[str(orig_ch)] = {'mean': 0.0, 'std': global_max_abs if global_max_abs > 0 else 1.0}
                obj.logger.info(f"加载模型: 已从统一归一化转换为Z-score标准化, 所有通道使用std={global_max_abs:.2e}")
            else:
                obj.channel_scalers = {}
                loaded_channels = []

                for orig_ch in continuous_channels:
                    ch_str = str(orig_ch)
                    if ch_str in channel_scalers_data:
                        ch_data = channel_scalers_data[ch_str]

                        if isinstance(ch_data, dict):
                            method = ch_data.get('method', 'zscore')
                            if method == 'log1p_standard':
                                obj.channel_scalers[ch_str] = {
                                    'method': 'log1p_standard',
                                    'mean': float(ch_data.get('mean', 0.0)),
                                    'std': float(ch_data.get('std', 1.0))
                                }
                                loaded_channels.append(f"通道{orig_ch}: Log1p+StandardScaler")
                            elif method == 'zscore':
                                obj.channel_scalers[ch_str] = {
                                    'method': 'zscore',
                                    'mean': float(ch_data.get('mean', 0.0)),
                                    'std': float(ch_data.get('std', 1.0))
                                }
                                loaded_channels.append(f"通道{orig_ch}: Z-score")
                            elif method == 'minmax':
                                obj.channel_scalers[ch_str] = {
                                    'method': 'minmax',
                                    'min': float(ch_data.get('min', 0.0)),
                                    'max': float(ch_data.get('max', 1.0))
                                }
                                loaded_channels.append(f"通道{orig_ch}: MinMax")
                            else:
                                obj.channel_scalers[ch_str] = {
                                    'method': 'zscore',
                                    'mean': float(ch_data.get('mean', 0.0)),
                                    'std': float(ch_data.get('std', 1.0))
                                }
                                loaded_channels.append(f"通道{orig_ch}: Z-score (旧版本)")
                        else:
                            obj.logger.warning(f"加载模型: 检测到旧版本的归一化参数（通道{orig_ch}），将转换为Z-score标准化")
                            ch_max_abs = float(ch_data)
                            obj.channel_scalers[ch_str] = {'mean': 0.0, 'std': ch_max_abs if ch_max_abs > 0 else 1.0}
                            loaded_channels.append(f"通道{orig_ch}: mean=0.0, std={ch_max_abs:.2e} (从旧版本转换)")
                    else:
                        obj.logger.warning(f"加载模型: 通道 {orig_ch} 没有标准化参数，使用默认值")
                        obj.channel_scalers[ch_str] = {
                            'method': 'zscore',
                            'mean': 0.0,
                            'std': 1.0
                        }
                        loaded_channels.append(f"通道{orig_ch}: 默认值")

                obj.logger.info(f"加载模型: 已加载分通道Z-score标准化参数: {', '.join(loaded_channels)}")
        else:
            obj.logger.warning("加载模型: 未找到标准化参数，将使用默认值")
            continuous_channels = [0, 1, 2, 3, 4, 5, 7]
            obj.channel_scalers = {str(ch): {'method': 'zscore', 'mean': 0.0, 'std': 1.0} for ch in continuous_channels}

        if obj.target_max_abs is not None:
            obj.logger.info(f"加载模型: 目标值归一化参数={obj.target_max_abs:.2f}")
        if obj.target_mean is not None:
            obj.logger.info(f"加载模型: 目标值统计信息 - 均值={obj.target_mean:.2f}, 标准差={obj.target_std:.2f}")
        if obj.target_scaler is not None:
            obj.logger.info(f"加载模型: {obj.target_param.upper()} RobustScaler参数 - center={obj.target_scaler.center_[0]:.2f}, scale={obj.target_scaler.scale_[0]:.2f}")
        if obj.num_layer_indices is not None:
            obj.logger.info(f"加载模型: layer_index类别数量={obj.num_layer_indices}")
        obj.is_trained = True
        return obj


if __name__ == "__main__":
    import glob
    import yaml
    from frame.framework import prepare_data, split_train_val, split_train_val_cnn, evaluate_and_save, setup_logging

    config_path = 'Modle/framework_cnn.yaml'
    if not os.path.exists(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'framework_cnn.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir')
    if log_dir is None:
        raise ValueError("配置文件缺少必需项: logging.log_dir")

    log_file_path = setup_logging(log_dir=log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"使用配置文件: {config_path}")

    data_config = config.get('data', {})
    if 'data_dir' not in data_config:
        raise ValueError("配置文件缺少必需项: data.data_dir")
    if 'output_base_dir' not in data_config:
        raise ValueError("配置文件缺少必需项: data.output_base_dir")

    DATA_DIR = data_config['data_dir']
    OUTPUT_BASE_DIR = data_config['output_base_dir']
    data_file_pattern = data_config.get('data_file_pattern', '*.pickle')

    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"输出目录: {OUTPUT_BASE_DIR}")
    logger.info(f"数据文件模式: {data_file_pattern}")

    all_files = glob.glob(os.path.join(DATA_DIR, data_file_pattern))
    logger.info(f"总文件数: {len(all_files)}")
    print(f"找到 {len(all_files)} 个数据文件")

    if len(all_files) == 0:
        logger.error(f"错误: 在 {DATA_DIR} 中未找到数据文件")
        print(f"错误: 在 {DATA_DIR} 中未找到数据文件")
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
        auto_select = data_config.get('auto_select_test', False)
        auto_count = data_config.get('auto_test_count', 4)

        if auto_select and len(all_files) >= auto_count:
            test_files = all_files[:auto_count]
            logger.warning(f"警告: 未找到指定的测试文件，自动选择前{auto_count}个文件作为测试集")
            print(f"警告: 未找到指定的测试文件，自动选择前{auto_count}个文件作为测试集")
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

    print("\n正在加载和预处理数据...")
    model_config = config.get('model', config)
    temp_model = CNNModel(model_config, parent_config=config)

    X_train_grid, X_train_layer_index, X_train_cpt, y_train = temp_model.preprocess(
        train_val_files, normalize=False, fit_scaler=True)

    (X_train_grid_split, X_train_layer_index_split, X_train_cpt_split, y_train_split,
     X_val_grid, X_val_layer_index, X_val_cpt, y_val) = split_train_val_cnn(
        X_train_grid, X_train_layer_index, X_train_cpt, y_train,
        val_size=config.get('training', {}).get('val_size', 0.1))

    print(f"\n{'=' * 60}")
    print("训练 CNN 模型")
    print(f"{'=' * 60}")

    model = temp_model
    logger.info(f"训练集样本数: {len(y_train_split)}, 验证集样本数: {len(y_val)}")

    train_metrics = model.train(
        X_train_grid_split, X_train_layer_index_split, X_train_cpt_split, y_train_split,
        X_val_grid, X_val_layer_index, X_val_cpt, y_val)
    logger.info(f"\n训练指标: {train_metrics}")

    output_subdir = data_config.get('output_subdir', 'CNN_results')
    model_filename = data_config.get('model_filename', 'cnn_model.pkl')

    output_dir = os.path.join(OUTPUT_BASE_DIR, output_subdir)
    model_path = os.path.join(OUTPUT_BASE_DIR, model_filename)

    logger.info(f"结果保存目录: {output_dir}")
    logger.info(f"模型保存路径: {model_path}")

    print("\n正在对测试孔位生成预测结果...")
    X_test_grid, X_test_layer_index, X_test_cpt, y_test = model.preprocess(
        test_files, normalize=False, fit_scaler=False)

    test_metrics = evaluate_and_save(
        model, X_test_grid, X_test_layer_index, y_test, test_files,
        output_dir, 'cnn', model_path, X_test_cpt)

    print(f"\n{'='*60}")
    print("CNN 模型训练完成！")
    print(f"{'='*60}")
    print(f"模型保存路径: {model_path}")
    print(f"结果保存目录: {output_dir}")
    print(f"  - CSV文件: 每个钻孔的预测结果")
    print(f"  - PNG图片: 真实值与预测值对比曲线")
    print(f"\n测试集评估指标 (基于{len(test_files)}个孔位):")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print(f"{'='*60}\n")