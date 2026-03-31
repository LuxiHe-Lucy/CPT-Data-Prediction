"""
SVR模型实现模块

实现用于CPT数据预测的支持向量回归模型。

输入模式：
    - 'flatten_raster': 栅格展平输入，维度 (N, C×H×W)
    - 'statistics': 统计特征输入，维度 (N, C×4+1)
"""
from __future__ import annotations
import sys
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Modle.frame.framework import BaseModel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging


class SVRModel(BaseModel):
    """
    支持向量回归模型

    支持单输出回归（qc/fs/u2）和两种输入模式：
    - 'flatten_raster': 栅格展平输入
    - 'statistics': 统计特征输入

    Attributes:
        target_param: 目标参数类型（'qc', 'fs', 'u2'）
        input_mode: 输入模式
        model: SVR实例
        crop_size: 栅格裁剪尺寸
        selected_channels: 选择的通道列表
        cnn_channel_scalers: CNN的通道标准化器
        target_scaler: 目标值标准化器
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.target_param = config.get('target_param', 'qc').lower()
        if self.target_param not in ['qc', 'fs', 'u2']:
            raise ValueError(f"target_param必须是'qc', 'fs'或'u2'之一，当前值: {self.target_param}")

        self.input_mode = config.get('input_mode', 'flatten_raster')
        if self.input_mode not in ['flatten_raster', 'statistics']:
            raise ValueError(f"input_mode必须是'flatten_raster'或'statistics'，当前值: {self.input_mode}")

        self.crop_size = config.get('crop_size', 8)
        self.selected_channels = config.get('selected_channels', None)
        self.cnn_channel_scalers = None
        self.target_scaler = None

        kernel = config.get('kernel', 'rbf')
        tol_value = float(config.get('tol', 0.001))

        gamma_value = config.get('gamma', 'scale')
        if isinstance(gamma_value, str) and gamma_value not in ['scale', 'auto']:
            gamma_value = float(gamma_value)

        import random
        self.random_seed = random.randint(0, 2**32 - 1)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        base_svr = SVR(
            kernel=kernel,
            C=float(config.get('C', 1.0)),
            epsilon=float(config.get('epsilon', 0.1)),
            gamma=gamma_value,
            degree=int(config.get('degree', 3)),
            coef0=float(config.get('coef0', 0.0)),
            shrinking=bool(config.get('shrinking', True)),
            tol=tol_value,
            cache_size=int(config.get('cache_size', 200)),
            verbose=bool(config.get('verbose', False)),
            max_iter=int(config.get('max_iter', -1))
        )

        self.model = base_svr
        self.logger = logging.getLogger(__name__)

        if self.input_mode == 'flatten_raster':
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * self.crop_size * self.crop_size

            if expected_dim > 1000:
                self.logger.warning(
                    f"⚠️ SVR警告: 输入维度过高（{expected_dim}维），训练可能非常慢！"
                    f"建议：(1) 减小crop_size到8或更小；(2) 减少通道数到5个以下"
                )

            self.logger.info(
                f"SVR模型初始化: 输入模式=栅格展平, "
                f"栅格尺寸={self.crop_size}×{self.crop_size}, "
                f"通道数={num_channels}, "
                f"输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )
        else:
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * 4 + 1
            self.logger.info(
                f"SVR模型初始化: 输入模式=统计特征, "
                f"通道数={num_channels}, "
                f"输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )

        self.logger.info(
            f"SVR超参数: kernel={kernel}, "
            f"C={self.model.C}, "
            f"epsilon={self.model.epsilon}, "
            f"gamma={self.model.gamma}"
        )

    def preprocess(self, file_paths: List[str], fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理

        Args:
            file_paths: 数据文件路径列表
            fit_scaler: 是否拟合标准化器

        Returns:
            预处理后的特征和目标值
        """
        from Modle.frame.framework import UnifiedDataPreprocessor

        preprocessor = UnifiedDataPreprocessor(self.config)
        X, y = preprocessor.preprocess(file_paths, fit_scaler=fit_scaler)

        if fit_scaler:
            scalers = preprocessor.get_scalers()
            self.cnn_channel_scalers = scalers['cnn_channel_scalers']
            self.target_scaler = scalers['target_scaler']

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        训练SVR模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征，可选
            y_val: 验证目标，可选

        Returns:
            训练指标字典
        """
        if y_train.ndim > 1:
            y_train = y_train.flatten()

        self._X_train = X_train.copy()
        self._y_train = y_train.copy()

        kernel = self.model.kernel if hasattr(self.model, 'kernel') else 'unknown'
        C = self.model.C if hasattr(self.model, 'C') else 'unknown'
        epsilon = self.model.epsilon if hasattr(self.model, 'epsilon') else 'unknown'
        input_dim = X_train.shape[1]

        self.logger.info(f"开始训练SVR模型: {len(X_train)} 训练样本, {len(X_val) if X_val is not None else 0} 验证样本")
        self.logger.info(
            f"训练配置: kernel={kernel}, C={C}, epsilon={epsilon}"
        )
        self.logger.info(f"输入模式: {self.input_mode}, 输入维度: {input_dim}, 预测参数: {self.target_param.upper()}")

        if input_dim > 500:
            print(f"⚠️ 警告: 输入维度很高（{input_dim}维），SVR训练可能需要很长时间...", file=sys.stderr)
            self.logger.warning(f"⚠️ 输入维度过高（{input_dim}维），建议减小crop_size或通道数")

        print(f"开始训练SVR模型 (kernel: {kernel})，这可能需要一些时间...", file=sys.stderr)

        self.model.fit(X_train, y_train)

        n_support = self.model.n_support_ if hasattr(self.model, 'n_support_') else None
        if n_support is not None:
            total_support = sum(n_support)
            support_ratio = total_support / len(X_train) * 100
            self.logger.info(
                f"SVR训练完成 (kernel: {kernel}, 单输出回归: {self.target_param.upper()})"
            )
            self.logger.info(f"支持向量数: {total_support} ({support_ratio:.1f}% 的训练样本)")
            print(f"SVR训练完成，支持向量数: {total_support} ({support_ratio:.1f}%)", file=sys.stderr)
        else:
            self.logger.info(f"SVR训练完成 (kernel: {kernel}, 单输出回归: {self.target_param.upper()})")
            print(f"SVR训练完成", file=sys.stderr)

        self._plot_prediction_scatter(X_train, y_train, X_val, y_val)

        self.is_trained = True

        train_metrics = self.evaluate(X_train, y_train)
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        else:
            val_metrics = None

        try:
            param_name = self.target_param.upper()
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"SVR 最终训练集评估指标 ({param_name}): "
                             f"MSE={train_metrics.get(f'{self.target_param}_mse', float('nan')):.6f}, "
                             f"RMSE={train_metrics.get(f'{self.target_param}_rmse', float('nan')):.6f}, "
                             f"MAE={train_metrics.get(f'{self.target_param}_mae', float('nan')):.6f}, "
                             f"R²={train_metrics.get(f'{self.target_param}_r2', float('nan')):.6f}")
            if val_metrics is not None:
                self.logger.info(f"SVR 最终验证集评估指标 ({param_name}): "
                                 f"MSE={val_metrics.get(f'{self.target_param}_mse', float('nan')):.6f}, "
                                 f"RMSE={val_metrics.get(f'{self.target_param}_rmse', float('nan')):.6f}, "
                                 f"MAE={val_metrics.get(f'{self.target_param}_mae', float('nan')):.6f}, "
                                 f"R²={val_metrics.get(f'{self.target_param}_r2', float('nan')):.6f}")
            self.logger.info(f"{'='*80}")
        except Exception as e:
            self.logger.warning(f"记录SVR评估指标到日志时出错: {e}")

        return train_metrics

    def _plot_prediction_scatter(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        绘制训练集和验证集的预测vs真实值散点图

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征，可选
            y_val: 验证目标，可选
        """
        import matplotlib.pyplot as plt

        model_dir = self.config.get('model_dir', '.')
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        scatter_file = os.path.join(model_dir, 'svr_prediction_scatter.png')

        y_train_pred = self.model.predict(X_train)

        fig, axes = plt.subplots(1, 2 if X_val is not None else 1, figsize=(14, 6))

        if X_val is None:
            axes = [axes]

        axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20, c='blue', label='Training Data')
        axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                     'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('True Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title(f'SVR Training Set - {self.target_param.upper()}', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            axes[1].scatter(y_val, y_val_pred, alpha=0.5, s=20, c='green', label='Validation Data')
            axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
                         'r--', lw=2, label='Perfect Prediction')
            axes[1].set_xlabel('True Values', fontsize=12)
            axes[1].set_ylabel('Predicted Values', fontsize=12)
            axes[1].set_title(f'SVR Validation Set - {self.target_param.upper()}', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ 预测散点图已保存到: {scatter_file}")
        print(f"预测散点图已保存: {scatter_file}", file=sys.stderr)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X_test: 测试特征

        Returns:
            预测值数组
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        pred = self.model.predict(X_test)
        return pred.flatten() if pred.ndim > 1 else pred

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            X_test: 测试特征
            y_test: 测试目标

        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        y_pred = self.predict(X_test)

        if y_test.ndim > 1:
            y_test = y_test.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        param_name = self.target_param
        mse = mean_squared_error(y_test, y_pred)
        return {
            f"{param_name}_mse": mse,
            f"{param_name}_rmse": np.sqrt(mse),
            f"{param_name}_mae": mean_absolute_error(y_test, y_pred),
            f"{param_name}_r2": r2_score(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }

    def save(self, path: str):
        """
        保存模型、scaler和配置

        Args:
            path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'target_param': self.target_param,
            'input_mode': self.input_mode,
            'cnn_channel_scalers': self.cnn_channel_scalers,
            'target_scaler': self.target_scaler
        }

        if hasattr(self, '_target_scaler_file'):
            data['_target_scaler_file'] = self._target_scaler_file

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"SVR模型保存: {path}")

    @classmethod
    def load(cls, path: str, config: Dict) -> 'SVRModel':
        """
        加载模型

        Args:
            path: 模型路径
            config: 配置字典

        Returns:
            加载的模型实例
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(config)
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.target_param = data.get('target_param', 'qc')
        instance.input_mode = data.get('input_mode', 'flatten_raster')
        instance.cnn_channel_scalers = data.get('cnn_channel_scalers', None)
        instance.target_scaler = data.get('target_scaler', None)

        if '_target_scaler_file' in data:
            instance._target_scaler_file = data['_target_scaler_file']

        instance.is_trained = True

        instance.logger.info(f"SVR模型加载: {path}")
        return instance


if __name__ == "__main__":
    import glob
    import yaml
    from Modle.frame.framework import CPTModelFramework, setup_logging

    config_path = 'Modle/framework_svr.yaml'
    if not os.path.exists(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'framework_svr.yaml')

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

    print("正在扫描数据文件...")
    all_files = glob.glob(os.path.join(DATA_DIR, data_file_pattern))
    logger.info(f"总文件数: {len(all_files)}")
    print(f"找到 {len(all_files)} 个数据文件")

    if len(all_files) == 0:
        logger.error(f"错误: 在 {DATA_DIR} 中未找到数据文件")
        print(f"错误: 在 {DATA_DIR} 中未找到数据文件")
        exit(1)

    print("正在准备测试集...")
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

    print(f"\n{'=' * 60}")
    print("训练 SVR 模型（栅格展平输入模式）")
    print(f"{'=' * 60}")
    print("⚠️ 警告: SVR训练可能需要较长时间，请耐心等待...")

    framework = CPTModelFramework(config_dict=config)

    print("\n正在加载和预处理数据...")
    logger.info(f"使用输入模式: {config['model'].get('input_mode', 'flatten_raster')}")
    logger.info(f"栅格尺寸: {config['model'].get('crop_size', 8)}×{config['model'].get('crop_size', 8)}")

    print("开始训练（详细日志请查看日志文件）...")
    train_metrics = framework.train(train_val_files, val_size=config.get('training', {}).get('val_size', 0.111))
    logger.info(f"\n训练指标: {train_metrics}")

    output_subdir = data_config.get('output_subdir', 'SVR_results')
    model_filename = data_config.get('model_filename', 'svr_model.pkl')
    output_dir = os.path.join(OUTPUT_BASE_DIR, output_subdir)
    model_path = os.path.join(OUTPUT_BASE_DIR, model_filename)

    logger.info(f"结果保存目录: {output_dir}")
    logger.info(f"模型保存路径: {model_path}")

    print("\n正在对测试孔位生成预测结果...")

    test_metrics = framework.evaluate(test_files, output_dir=output_dir)

    framework.save_model(model_path)

    print(f"\n{'=' * 60}")
    print("SVR 模型训练完成！")
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
