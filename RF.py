"""
RF模型实现模块

实现用于CPT数据预测的随机森林回归模型。

输入模式：
    - 'flatten_raster': 栅格展平输入，维度 (N, C×H×W)
    - 'statistics': 统计特征输入，维度 (N, C×4+1)

特性：
    - 支持单输出回归（qc/fs/u2）
    - OOB分数评估和特征重要性分析
    - 多核并行训练
"""
from __future__ import annotations
import sys
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Modle.frame.framework import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging


class RFModel(BaseModel):
    """
    Random Forest回归模型

    支持单输出回归：预测qc、fs或u2中的一个。
    支持两种输入模式：
        1. 'flatten_raster': 栅格展平（与CNN公平对比）
        2. 'statistics': 统计特征（原方案）

    主要特性：
        - OOB分数评估（袋外误差估计）
        - 并行训练（n_jobs=-1使用所有CPU核心）
        - 特征重要性分析（用于解释性）
        - 复用CNN的标准化策略（flatten_raster模式）

    Attributes:
        target_param: 目标参数类型（'qc', 'fs', 'u2'）
        input_mode: 输入模式（'flatten_raster' 或 'statistics'）
        model: RandomForestRegressor实例
        crop_size: 栅格裁剪尺寸（flatten_raster模式使用）
        selected_channels: 选择的通道列表
        cnn_channel_scalers: CNN的通道标准化器
        qc_scaler: QC值的RobustScaler
    """

    def __init__(self, config: Dict):
        super().__init__(config)

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
        base_rf = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            criterion=config.get('criterion', 'squared_error'),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=config.get('min_weight_fraction_leaf', 0.0),
            max_features=config.get('max_features', 'sqrt'),
            max_leaf_nodes=config.get('max_leaf_nodes', None),
            min_impurity_decrease=config.get('min_impurity_decrease', 0.0),
            bootstrap=config.get('bootstrap', True),
            oob_score=config.get('oob_score', True),
            n_jobs=config.get('n_jobs', -1),
            random_state=config.get('random_state', 42),
            verbose=config.get('verbose', 0),
            warm_start=config.get('warm_start', False),
            ccp_alpha=config.get('ccp_alpha', 0.0),
            max_samples=config.get('max_samples', None)
        )

        self.model = base_rf
        self.logger = logging.getLogger(__name__)

        if self.input_mode == 'flatten_raster':
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * self.crop_size * self.crop_size
            self.logger.info(
                f"RF模型初始化: 输入模式=栅格展平, "
                f"栅格尺寸={self.crop_size}×{self.crop_size}, "
                f"通道数={num_channels}, "
                f"预期输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )
        else:
            num_channels = len(self.selected_channels) if self.selected_channels else 14
            expected_dim = num_channels * 4 + 1
            self.logger.info(
                f"RF模型初始化: 输入模式=统计特征, "
                f"通道数={num_channels}, "
                f"预期输入维度={expected_dim}, "
                f"目标参数={self.target_param.upper()}"
            )

        self.logger.info(
            f"RF超参数: n_estimators={self.model.n_estimators}, "
            f"max_depth={self.model.max_depth}, "
            f"min_samples_split={self.model.min_samples_split}"
        )

    def preprocess(self, file_paths: List[str], fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理

        Args:
            file_paths: 数据文件路径列表
            fit_scaler: 是否拟合scaler

        Returns:
            预处理后的特征和目标值
        """
        from Modle.frame.framework import UnifiedDataPreprocessor

        preprocessor = UnifiedDataPreprocessor(self.config)

        if not fit_scaler:
            if self.cnn_channel_scalers is not None:
                preprocessor.set_scalers({
                    'cnn_channel_scalers': self.cnn_channel_scalers,
                    'target_scaler': self.target_scaler,
                    'qc_scaler': None,
                    'target_mean': None,
                    'target_std': None
                })

        X, y = preprocessor.preprocess(file_paths, fit_scaler=fit_scaler)

        if fit_scaler:
            scalers = preprocessor.get_scalers()
            self.cnn_channel_scalers = scalers['cnn_channel_scalers']
            self.target_scaler = scalers['target_scaler']

        return X, y

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                param_grid: Optional[Dict] = None) -> Dict:
        """
        使用网格搜索优化超参数

        Args:
            X_train: 训练特征
            y_train: 训练目标
            param_grid: 参数网格，默认为常用参数组合

        Returns:
            Dict: 最佳参数和分数
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            }

        self.logger.info("开始超参数优化...")
        self.logger.info(f"参数网格大小: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])}")

        base_model = RandomForestRegressor(
            criterion='squared_error',
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='r2',
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        self.logger.info(f"最佳参数: {best_params}")
        self.logger.info(f"最佳交叉验证R²分数: {best_score:.6f}")

        for param, value in best_params.items():
            if param in self.model.__dict__ or hasattr(self.model, param):
                setattr(self.model, param, value)
            self.config[param] = value

        return {
            'best_params': best_params,
            'best_cv_score': best_score
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None, optimize_params: bool = False) -> Dict[str, float]:
        """
        训练Random Forest模型（单输出回归：预测qc、fs或u2中的一个）

        Args:
            X_train: 训练特征 (N, feature_dim)
            y_train: 训练目标 (N,) - 单个参数值
            X_val: 验证特征 (M, feature_dim)，可选
            y_val: 验证目标 (M,) - 单个参数值，可选

        Returns:
            Dict[str, float]: 训练指标字典
        """
        if y_train.ndim > 1:
            y_train = y_train.flatten()

        if optimize_params:
            self.logger.info("执行超参数优化...")
            opt_result = self.optimize_hyperparameters(X_train, y_train)
            self.logger.info(f"超参数优化完成，最佳CV分数: {opt_result['best_cv_score']:.6f}")

        n_estimators = self.model.n_estimators
        max_depth = self.model.max_depth
        input_dim = X_train.shape[1]

        self.logger.info(f"开始训练RF模型: {len(X_train)} 训练样本, {len(X_val) if X_val is not None else 0} 验证样本")
        self.logger.info(
            f"训练配置: n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={self.model.min_samples_split}, n_jobs={self.model.n_jobs}"
        )
        self.logger.info(f"输入模式: {self.input_mode}, 输入维度: {input_dim}, 预测参数: {self.target_param.upper()}")
        print(f"开始训练RF模型: {n_estimators} 棵树，这可能需要一些时间...", file=sys.stderr)

        import threading
        import time

        progress_message = "训练进度: 正在训练决策树..."
        print(progress_message, file=sys.stderr)

        def train_with_progress():
            self.model.fit(X_train, y_train)

        train_thread = threading.Thread(target=train_with_progress)
        train_thread.start()

        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_idx = 0
        start_time = time.time()

        while train_thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed > 1:
                spinner_char = spinner[spinner_idx % len(spinner)]
                estimated_progress = min(95, int(elapsed / max(1, n_estimators * 0.05) * 100))
                sys.stderr.write(f'\r{spinner_char} 训练进度: {estimated_progress}% ({int(elapsed)}s)  ')
                sys.stderr.flush()
                spinner_idx += 1
                time.sleep(0.5)

        sys.stderr.write(f'\r✓ 训练完成! 耗时: {int(time.time() - start_time)}s{" " * 30}\n')
        sys.stderr.flush()

        train_thread.join()

        oob_score = getattr(self.model, 'oob_score_', None)
        if oob_score is not None:
            self.logger.info(f"RF训练完成 (单输出回归: {self.target_param.upper()}, OOB Score: {oob_score:.6f})")
            print(f"RF训练完成，OOB Score: {oob_score:.6f}", file=sys.stderr)
        else:
            self.logger.info(f"RF训练完成 (单输出回归: {self.target_param.upper()})")
            print(f"RF训练完成", file=sys.stderr)

        if hasattr(self.model, 'feature_importances_') and input_dim <= 100:
            feature_importance = self.model.feature_importances_
            top_k = min(10, input_dim)
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            self.logger.info(f"特征重要性 Top-{top_k}:")
            for idx in top_indices:
                self.logger.info(f"  特征 {idx}: {feature_importance[idx]:.6f}")

        self._plot_feature_importance(X_train.shape[1])

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
            self.logger.info(f"RF 最终训练集评估指标 ({param_name}): "
                             f"MSE={train_metrics.get(f'{self.target_param}_mse', float('nan')):.6f}, "
                             f"RMSE={train_metrics.get(f'{self.target_param}_rmse', float('nan')):.6f}, "
                             f"MAE={train_metrics.get(f'{self.target_param}_mae', float('nan')):.6f}, "
                             f"R²={train_metrics.get(f'{self.target_param}_r2', float('nan')):.6f}")
            if val_metrics is not None:
                self.logger.info(f"RF 最终验证集评估指标 ({param_name}): "
                                 f"MSE={val_metrics.get(f'{self.target_param}_mse', float('nan')):.6f}, "
                                 f"RMSE={val_metrics.get(f'{self.target_param}_rmse', float('nan')):.6f}, "
                                 f"MAE={val_metrics.get(f'{self.target_param}_mae', float('nan')):.6f}, "
                                 f"R²={val_metrics.get(f'{self.target_param}_r2', float('nan')):.6f}")
            self.logger.info(f"{'='*80}")
        except Exception as e:
            self.logger.warning(f"记录RF评估指标到日志时出错: {e}")

        return train_metrics

    def _plot_feature_importance(self, n_features: int):
        """
        绘制并保存特征重要性柱状图

        Args:
            n_features: 特征总数
        """
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning("模型没有feature_importances_属性，跳过绘图")
            return

        import matplotlib.pyplot as plt

        model_dir = self.config.get('model_dir', '.')
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        importance_file = os.path.join(model_dir, 'rf_feature_importance.png')
        feature_importance = self.model.feature_importances_

        top_k = min(20, n_features)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_importance = feature_importance[top_indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(top_k), top_importance, color='steelblue', alpha=0.8)
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title(f'RF Feature Importance (Top-{top_k}) - {self.target_param.upper()}',
                  fontsize=14, fontweight='bold')
        plt.xticks(range(top_k), [f'F{i}' for i in top_indices], rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ 特征重要性图已保存到: {importance_file}")
        print(f"特征重要性图已保存: {importance_file}", file=sys.stderr)


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测（单输出回归：返回单个参数值）

        Args:
            X_test: 测试特征

        Returns:
            np.ndarray: 预测值数组 (N,)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        pred = self.model.predict(X_test)
        return pred.flatten() if pred.ndim > 1 else pred

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标（单输出回归）

        Args:
            X_test: 测试特征
            y_test: 测试目标 (N,) - 单个参数值

        Returns:
            Dict[str, float]: 评估指标字典
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

        self.logger.info(f"RF模型保存: {path}")

    @classmethod
    def load(cls, path: str, config: Dict) -> 'RFModel':
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

        instance.logger.info(f"RF模型加载: {path}")
        return instance


if __name__ == "__main__":
    import glob
    import yaml
    from datetime import datetime
    from Modle.frame.framework import CPTModelFramework, setup_logging

    config_path = 'Modle/framework_rf.yaml'
    if not os.path.exists(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'framework_rf.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging_config = config.get('logging', {})
    log_dir = logging_config.get('log_dir', None)
    framework_log_prefix = logging_config.get('framework_log_prefix', 'framework_log')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f"{framework_log_prefix}_{timestamp}.txt"
    log_file_path = setup_logging(log_dir=log_dir, log_file_name=log_file_name)

    logger = logging.getLogger(__name__)

    data_config = config.get('data', {})
    DATA_DIR = data_config.get('data_dir', r'D:\CNN\数据集\dataset_pickle_14通道_64x64')
    data_file_pattern = data_config.get('data_file_pattern', '*.pickle')

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
        auto_select_test = data_config.get('auto_select_test', False)
        auto_test_count = int(data_config.get('auto_test_count', 2))
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
    print("训练 RF 模型（栅格展平输入模式）")
    print(f"{'=' * 60}")

    framework = CPTModelFramework(config_dict=config)

    print("\n正在加载和预处理数据...")
    logger.info(f"使用输入模式: {config['model'].get('input_mode', 'flatten_raster')}")

    print("开始训练（详细日志请查看日志文件）...")
    train_metrics = framework.train(train_val_files, val_size=config.get('training', {}).get('val_size', 0.111))
    logger.info(f"\n训练指标: {train_metrics}")

    OUTPUT_BASE_DIR = data_config.get('output_base_dir', r'D:\CNN\预测结果\RF')
    output_subdir = data_config.get('output_subdir', 'RF_results')
    model_filename = data_config.get('model_filename', 'rf_model.pkl')

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    output_dir = os.path.join(OUTPUT_BASE_DIR, output_subdir)
    model_path = os.path.join(OUTPUT_BASE_DIR, model_filename)

    os.makedirs(output_dir, exist_ok=True)

    print("\n正在对测试孔位生成预测结果...")

    test_metrics = framework.evaluate(test_files, output_dir=output_dir)
    framework.model.save(model_path)

    print(f"\n{'=' * 60}")
    print("RF 模型训练完成！")
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