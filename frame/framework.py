"""
CPT预测模型框架模块

本模块提供了CPT（静力触探）数据预测的统一框架，支持多种模型类型：
- CNN: 卷积神经网络（处理8通道栅格数据）
- ANN: 人工神经网络
- RF: 随机森林
- SVR: 支持向量回归

主要功能：
1. CPTModelFramework: 主框架类，负责模型管理、训练、评估和预测
2. BaseModel: 模型基类，定义通用接口
"""

import pickle
import numpy as np
import yaml
import os
from typing import List, Dict, Any, Optional, Tuple, Union, NamedTuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import importlib
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm.auto import tqdm

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray



def setup_logging(log_dir: str = 'logs', log_level: int = logging.INFO) -> str:
    """
    配置日志系统

    Args:
        log_dir: 日志目录路径
        log_level: 日志级别（默认INFO）

    Returns:
        str: 日志文件路径
    """
    from datetime import datetime

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"日志系统已初始化，日志文件: {log_file}")
    return log_file


def _crop_raster(raster: np.ndarray, crop_size: int) -> np.ndarray:
    """中心裁剪64×64栅格数据到指定尺寸"""
    if crop_size >= 64:
        return raster

    start = (64 - crop_size) // 2
    end = start + crop_size
    return raster[:, start:end, start:end]


def _load_single_file(
        file_path: str,
        target_param: str = 'qc',
        fit_scaler: bool = True,
        crop_size: int = 32
) -> List[Tuple]:
    """
    加载单个pickle文件数据（多进程调用）

    Args:
        file_path: pickle文件路径
        target_param: 目标参数类型（'qc', 'fs', 'u2'）
        fit_scaler: 是否拟合标准化器
        crop_size: 裁剪尺寸

    Returns:
        List[Tuple]: 包含(raster, cpt_point, target_value)的列表
    """
    results = []
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for item in data.get('rasters', []):
            raster = item.get('raster')
            pt = item.get('cpt_point')

            if not isinstance(raster, np.ndarray) or pt is None or len(pt) < 4:
                continue

            if raster.ndim == 3:
                if raster.shape[2] == 14:
                    raster = raster.transpose(2, 0, 1)
                elif raster.shape[0] != 14:
                    continue
            else:
                continue

            if raster.shape != (14, 64, 64):
                continue

            raster = _crop_raster(raster, crop_size)

            if np.all(raster == 0):
                continue

            has_full_nan_channel = False
            for ch in range(14):
                if np.all(np.isnan(raster[ch])):
                    has_full_nan_channel = True
                    break

            if has_full_nan_channel:
                continue

            if target_param == 'qc':
                qc_value = float(pt[3])
                if not (0 <= qc_value <= 100):
                    continue
                target_value = qc_value
            elif target_param == 'fs':
                fs_value = float(pt[1])
                if fs_value < 0 or fs_value > 5 or np.isnan(fs_value) or np.isinf(fs_value):
                    continue
                target_value = fs_value
            elif target_param == 'u2':
                target_value = float(pt[2])
            else:
                target_value = float(pt[3])

            results.append((raster.astype(np.float32), pt, target_value))

    except Exception:
        pass

    return results


def fill_nan_with_nearest(data_2d: np.ndarray) -> np.ndarray:
    """
    使用最邻近插值填充2D数组中的NaN值

    Args:
        data_2d: 2D数组 (H, W)，可能包含NaN值

    Returns:
        np.ndarray: 填充后的2D数组，所有NaN值都被最邻近的有效值替换

    Note:
        如果整个数组都是NaN，将使用0填充
    """
    if not np.any(np.isnan(data_2d)):
        return data_2d.copy()

    h, w = data_2d.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    valid_mask = ~np.isnan(data_2d)
    if not np.any(valid_mask):
        return np.zeros_like(data_2d)

    valid_points = np.column_stack([y_coords[valid_mask], x_coords[valid_mask]])
    valid_values = data_2d[valid_mask]
    nan_mask = np.isnan(data_2d)
    nan_points = np.column_stack([y_coords[nan_mask], x_coords[nan_mask]])

    if len(nan_points) > 0:
        interpolated_values = griddata(
            valid_points, valid_values, nan_points,
            method='nearest', fill_value=0.0
        )
        filled_data = data_2d.copy()
        filled_data[nan_mask] = interpolated_values
        return filled_data

    return data_2d.copy()


def crop_raster(raster: np.ndarray, crop_size: int) -> np.ndarray:
    """
    对64×64的栅格数据进行中心裁剪，得到指定尺寸的栅格

    Args:
        raster: 原始栅格数据 (C, 64, 64) 或 (64, 64, C)
        crop_size: 裁剪后的尺寸（9-64之间）

    Returns:
        np.ndarray: 裁剪后的栅格数据 (C, crop_size, crop_size)
    """
    if raster.ndim == 3:
        if raster.shape[2] == raster.shape[0] or raster.shape[2] < raster.shape[0]:
            raster = raster.transpose(2, 0, 1)

    if crop_size >= 64:
        return raster

    start = (64 - crop_size) // 2
    end = start + crop_size
    return raster[:, start:end, start:end]


def preprocess_qc_values(targets: np.ndarray, cpts: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    预处理qc值（-1到200范围过滤 + 前向填充）

    Args:
        targets: 目标值数组 (N,)
        cpts: CPT特征数组 (N, 4) - [depth, qc, fs, u2]
        logger: 日志记录器

    Returns:
        np.ndarray: 处理后的目标值数组
    """
    if len(targets) == 0:
        return targets

    initial_range = (targets.min(), targets.max())
    invalid_mask = (targets < -1) | (targets > 200)
    invalid_count = np.sum(invalid_mask)

    if invalid_count == 0:
        logger.info("qc值预处理: 所有值都在-1到200范围内，无需处理")
        return targets

    logger.info(f"qc值预处理: 发现{invalid_count}个异常值（不在-1到200范围内），原始范围=[{initial_range[0]:.4f}, {initial_range[1]:.4f}]")

    if len(cpts) == 0 or cpts.shape[1] < 1:
        logger.warning("CPT数据缺少深度信息，将使用原始顺序进行前向填充")
        sorted_indices = np.arange(len(targets))
        depths = None
    else:
        depths = cpts[:, 0] if cpts.ndim > 1 else cpts
        sorted_indices = np.argsort(depths)

    targets_sorted = targets[sorted_indices].copy()
    invalid_mask_sorted = invalid_mask[sorted_indices]

    last_valid_value = None
    filled_count = 0

    with tqdm(total=len(targets_sorted), desc="预处理qc值", unit="样本", file=sys.stderr) as pbar:
        for i in range(len(targets_sorted)):
            if invalid_mask_sorted[i]:
                if last_valid_value is not None:
                    targets_sorted[i] = last_valid_value
                    filled_count += 1
                else:
                    targets_sorted[i] = 0.0
                    filled_count += 1
                    if depths is not None:
                        logger.warning(f"qc值前向填充: 深度={depths[sorted_indices[i]]:.2f}是第一个值且为异常值"
                                     f"{targets[sorted_indices[i]]:.4f}，填充为0.0")
                    else:
                        logger.warning(f"qc值前向填充: 索引{sorted_indices[i]}是第一个值且为异常值"
                                     f"{targets[sorted_indices[i]]:.4f}，填充为0.0")
            else:
                last_valid_value = targets_sorted[i]

            pbar.update(1)
            if (i + 1) % 1000 == 0:
                pbar.set_postfix({"已填充": filled_count})

    targets_restored = np.zeros_like(targets)
    targets_restored[sorted_indices] = targets_sorted
    targets = targets_restored

    logger.info(f"qc值前向填充完成: 填充了{filled_count}个异常值，填充后范围=[{targets.min():.4f}, {targets.max():.4f}]")

    return targets


def process_nan_values(grids: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """
    处理栅格数据中的NaN和Inf值

    Args:
        grids: 栅格数据 (N, 8, 32, 32)
        logger: 日志记录器

    Returns:
        np.ndarray: 处理后的栅格数据
    """
    if np.any(np.isnan(grids)) or np.any(np.isinf(grids)):
        nan_count = np.sum(np.isnan(grids))
        inf_count = np.sum(np.isinf(grids))
        logger.info(f"栅格数据中包含异常值: NaN={nan_count}, Inf={inf_count}")

        filled_count = 0
        full_nan_channels = 0
        full_nan_samples = 0
        total_samples = grids.shape[0]
        total_channels = grids.shape[1]
        total_operations = total_samples * total_channels

        with tqdm(total=total_operations, desc="处理NaN值", unit="通道", file=sys.stderr) as pbar:
            for i in range(grids.shape[0]):
                if np.all(np.isnan(grids[i])):
                    full_nan_samples += 1
                    logger.warning(f"警告: 样本 {i} 全为NaN，应该在加载阶段被过滤，跳过处理")
                    pbar.update(total_channels)
                    continue

                for ch in range(grids.shape[1]):
                    ch_data = grids[i, ch, :, :]

                    if np.all(np.isnan(ch_data)):
                        grids[i, ch, :, :] = 0.0
                        full_nan_channels += 1
                        if full_nan_channels <= 5:
                            logger.warning(f"样本 {i} 通道 {ch} 全为NaN，使用0填充")
                    elif np.any(np.isnan(ch_data)):
                        grids[i, ch, :, :] = fill_nan_with_nearest(ch_data)
                        filled_count += 1

                    pbar.update(1)
                    if (i * total_channels + ch + 1) % 100 == 0:
                        pbar.set_postfix({"部分NaN": filled_count, "全NaN通道": full_nan_channels})

        if full_nan_samples > 0:
            logger.warning(f"发现 {full_nan_samples} 个全为NaN的栅格样本（应该在加载阶段被过滤）")

        if np.any(np.isinf(grids)):
            grids = np.nan_to_num(grids, nan=0.0, posinf=0.0, neginf=0.0)

        if filled_count > 0 or full_nan_channels > 0:
            logger.info(f"NaN值填充完成: {filled_count} 个部分NaN通道使用最邻近填充, "
                       f"{full_nan_channels} 个全NaN通道使用0填充")

    return grids


class PreprocessResult(NamedTuple):
    """预处理结果的命名元组"""
    grid_continuous: np.ndarray
    grid_layer_index: Optional[np.ndarray]
    cpt: np.ndarray
    targets: np.ndarray
    scalers: Optional[Dict]


def preprocess_raster_data(
    file_paths: List[str],
    config: Dict,
    fit_scaler: bool = True,
    crop_size: int = 32,
    channel_scalers: Optional[Dict] = None,
    qc_scaler: Optional[Any] = None,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None
) -> PreprocessResult:
    """
    栅格数据预处理

    Args:
        file_paths: pickle文件路径列表
        config: 模型配置字典
        fit_scaler: True=训练阶段(拟合scaler), False=测试阶段(使用已有scaler)
        crop_size: 裁剪尺寸（9-64）
        channel_scalers: 测试阶段必须传入的已训练标准化器

    Returns:
        PreprocessResult: 命名元组，包含5个字段

    Raises:
        ValueError: 测试阶段未提供channel_scalers时抛出
    """
    from sklearn.preprocessing import RobustScaler
    from multiprocessing import Pool, cpu_count
    from functools import partial
    import joblib

    logger = logging.getLogger(__name__)

    target_param = config.get('target_param', 'qc').lower()
    selected_channels = config.get('selected_channels', list(range(14)))

    continuous_channels = [ch for ch in selected_channels if ch != 12]
    use_layer_index = 12 in selected_channels

    if not fit_scaler:
        validate_scalers(channel_scalers, qc_scaler, target_param, fit_scaler)
        logger.info("✓ 测试阶段：使用已加载的标准化器")
    total_files = len(file_paths)
    num_workers = min(config.get('num_workers', cpu_count()), total_files)

    print(f"正在使用{num_workers}个进程并行加载数据...", file=sys.stderr)

    load_func = partial(
        _load_single_file,
        target_param=target_param,
        fit_scaler=fit_scaler,
        crop_size=crop_size
    )

    results_list = []

    try:
        with Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(
                load_func,
                file_paths,
                chunksize=max(1, total_files // (num_workers * 4))
            )

            with tqdm(total=total_files, desc="加载数据文件", unit="文件",
                      position=0, leave=True, file=sys.stderr) as pbar:
                for results in results_iter:
                    results_list.extend(results)
                    pbar.update(1)
                    pbar.set_postfix({"样本数": len(results_list)})

    except Exception as e:
        logger.error(f"多进程数据加载失败: {e}")
        raise

    if len(results_list) == 0:
        raise ValueError("没有有效样本（所有文件加载失败或数据被过滤）")

    print(f"\n✓ 数据加载完成，共 {len(results_list)} 个样本", file=sys.stderr)

    grids = []
    cpts = []
    targets = []

    for raster, pt, target_value in results_list:
        grids.append(raster)
        if fit_scaler:
            cpts.append([float(pt[0]), float(pt[3]), float(pt[1]), float(pt[2])])
        else:
            cpts.append([float(pt[0])])
        targets.append(target_value)

    grids = np.array(grids, dtype=np.float32)
    cpts = np.array(cpts, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    logger.info(f"数据形状: grids={grids.shape}, cpts={cpts.shape}, targets={targets.shape}")

    grids = _batch_fill_nan_optimized(grids, logger)
    valid_mask = ~(np.isnan(targets) | np.isinf(targets))
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        logger.warning(f"目标值中包含{invalid_count}个NaN/Inf，将过滤掉")
        grids = grids[valid_mask]
        cpts = cpts[valid_mask]
        targets = targets[valid_mask]

    if len(targets) == 0:
        raise ValueError("过滤NaN/Inf后没有有效样本")

    if target_param == 'qc':
        grids, cpts, targets = filter_qc_range(grids, cpts, targets, -1.0, 200.0, logger)

        if len(targets) == 0:
            raise ValueError("QC值过滤后没有有效样本")
    scalers_output = None

    if fit_scaler:
        if target_param == 'qc':
            qc_min, qc_max = -1.0, 200.0
            original_count = len(targets)
            valid_mask = (targets >= qc_min) & (targets <= qc_max)
            filtered_targets = targets[valid_mask]
            filtered_count = len(filtered_targets)
            outlier_ratio = (original_count - filtered_count) / original_count * 100

            logger.info(f"✓ QC范围过滤: 有效范围=[{qc_min:.1f}, {qc_max:.1f}], 过滤比例={outlier_ratio:.1f}%")
            logger.info(f"  原始样本: {original_count}, 过滤后: {filtered_count}")

            targets_for_scaling = filtered_targets
        else:
            targets_for_scaling = targets

        target_scaler = RobustScaler()
        targets_original = targets_for_scaling.copy()
        targets_normalized = target_scaler.fit_transform(targets_for_scaling.reshape(-1, 1)).flatten()

        logger.info(f"✓ {target_param.upper()}目标值使用RobustScaler标准化:")
        logger.info(f"  原始值范围: [{targets_original.min():.2f}, {targets_original.max():.2f}]")
        logger.info(f"  原始均值: {targets_original.mean():.3f}, 中位数: {np.median(targets_original):.3f}")
        logger.info(f"  标准化范围: [{targets_normalized.min():.2f}, {targets_normalized.max():.2f}]")
        logger.info(f"  标准化均值: {targets_normalized.mean():.3f}, 中位数: {np.median(targets_normalized):.3f}")
        logger.info(f"  RobustScaler参数: center={target_scaler.center_[0]:.3f}, scale={target_scaler.scale_[0]:.3f}")

        scaler_file = _get_scaler_path(config, f'{target_param}_scaler_file', f'{target_param}_robust_scaler.pkl')
        joblib.dump(target_scaler, str(scaler_file))
        logger.info(f"  已保存到: {scaler_file}")

        if target_param == 'qc':
            targets = np.full_like(targets, np.nan)
            targets[valid_mask] = targets_normalized
            logger.info(f"  QC标准化完成: 有效样本={filtered_count}, 异常值样本设为NaN")
        else:
            targets = targets_normalized

        _scaler_file = str(scaler_file)

    else:
        if qc_scaler is not None:
            targets = qc_scaler.transform(targets.reshape(-1, 1)).flatten()
            logger.info(f"✓ [测试] 使用已加载的RobustScaler标准化{target_param.upper()}值")
            _scaler_file = None
        else:
            logger.warning(f"[测试] 缺少{target_param.upper()}标准化器，跳过标准化")
            _scaler_file = None

    if use_layer_index:
        layer_index_data = grids[:, 12, :, :].copy()

        if layer_index_data is not None:
            max_layer_index = int(np.max(layer_index_data)) if layer_index_data.size > 0 else 0
            min_layer_index = int(np.min(layer_index_data)) if layer_index_data.size > 0 else 0

            logger.info(f"Layer_index数据范围: min={min_layer_index}, max={max_layer_index}")

            if min_layer_index < 0:
                logger.warning(f"检测到负的layer_index值 (min={min_layer_index})，将调整为从0开始")
                layer_index_data = layer_index_data - min_layer_index
                max_layer_index = max_layer_index - min_layer_index

            num_layer_indices_config = config.get('num_layer_indices', 10)
            if max_layer_index >= num_layer_indices_config:
                logger.warning(f"layer_index最大值 {max_layer_index} 超过配置的num_layer_indices {num_layer_indices_config}，将裁剪")
                layer_index_data = np.clip(layer_index_data, 0, num_layer_indices_config - 1)

            actual_max_index = int(np.max(layer_index_data)) if layer_index_data.size > 0 else 0
            actual_num_classes = actual_max_index + 1

            logger.info(f"Layer_index处理后: 实际类别数量={actual_num_classes}, 范围=[0, {actual_max_index}]")

            if fit_scaler and scalers_output is not None:
                scalers_output['num_layer_indices'] = actual_num_classes
                scalers_output['layer_index_mapping'] = {
                    'min_original': min_layer_index,
                    'max_original': max_layer_index,
                    'num_classes': actual_num_classes
                }

        continuous_grids = grids[:, continuous_channels, :, :]
    else:
        layer_index_data = None
        continuous_grids = grids[:, continuous_channels, :, :]

    full_channel_normalization_strategies = {
        0: 'log1p_standard',
        1: 'log1p_standard',
        2: 'log1p_standard',
        3: 'log1p_standard',
        4: 'log1p_standard',
        5: 'log1p_standard',
        6: 'log1p_standard',
        7: 'log1p_standard',
        8: 'log1p_standard',
        9: 'log1p_standard',
        10: 'raw',
        11: 'raw',
        13: 'raw',
    }

    channel_normalization_strategies = {
        orig_ch: full_channel_normalization_strategies[orig_ch]
        for orig_ch in continuous_channels
        if orig_ch in full_channel_normalization_strategies
    }

    full_minmax_ranges = {
        4: (0.0, 1.0),
        5: (-np.pi, np.pi),
        6: (-np.pi, np.pi)
    }

    minmax_ranges = {
        orig_ch: full_minmax_ranges[orig_ch]
        for orig_ch in continuous_channels
        if orig_ch in full_minmax_ranges
    }

    if fit_scaler:
        channel_scalers = {}

        print("\n正在计算通道标准化参数...", file=sys.stderr)
        with tqdm(total=len(continuous_channels), desc="拟合通道Scaler",
                  unit="通道", position=0, leave=True, file=sys.stderr) as pbar:
            for ch_idx, orig_ch in enumerate(continuous_channels):
                ch_data = continuous_grids[:, ch_idx, :, :].flatten()
                ch_data = ch_data[~np.isnan(ch_data) & ~np.isinf(ch_data)]

                if len(ch_data) == 0:
                    channel_scalers[str(orig_ch)] = {
                        'method': 'zscore',
                        'mean': 0.0,
                        'std': 1.0
                    }
                    pbar.update(1)
                    continue

                strategy = channel_normalization_strategies.get(orig_ch, 'zscore')

                if strategy == 'log1p_standard':
                    ch_data_log1p = np.log1p(np.maximum(ch_data, 0))
                    ch_mean = float(np.mean(ch_data_log1p))
                    ch_std = float(np.std(ch_data_log1p))
                    if ch_std < 1e-6:
                        ch_std = 1.0
                    channel_scalers[str(orig_ch)] = {
                        'method': 'log1p_standard',
                        'mean': ch_mean,
                        'std': ch_std
                    }
                    pbar.set_postfix_str(f"Ch{orig_ch}: Log1p+Std")

                elif strategy == 'zscore':
                    ch_mean = float(np.mean(ch_data))
                    ch_std = float(np.std(ch_data))
                    if ch_std < 1e-6:
                        ch_std = 1.0
                    channel_scalers[str(orig_ch)] = {
                        'method': 'zscore',
                        'mean': ch_mean,
                        'std': ch_std
                    }
                    pbar.set_postfix_str(f"Ch{orig_ch}: Z-score")

                elif strategy == 'minmax':
                    if orig_ch in minmax_ranges:
                        data_min, data_max = minmax_ranges[orig_ch]
                        actual_min = float(np.min(ch_data))
                        actual_max = float(np.max(ch_data))
                        scale_min = min(data_min, actual_min)
                        scale_max = max(data_max, actual_max)
                    else:
                        scale_min = float(np.min(ch_data))
                        scale_max = float(np.max(ch_data))

                    if scale_max - scale_min < 1e-6:
                        scale_min = 0.0
                        scale_max = 1.0

                    channel_scalers[str(orig_ch)] = {
                        'method': 'minmax',
                        'min': scale_min,
                        'max': scale_max
                    }
                    pbar.set_postfix_str(f"Ch{orig_ch}: MinMax")

                elif strategy == 'raw':
                    channel_scalers[str(orig_ch)] = {'method': 'raw'}
                    pbar.set_postfix_str(f"Ch{orig_ch}: Raw")

                pbar.update(1)

        logger.info(f"✓ 已拟合{len(channel_scalers)}个通道的标准化器")

    else:
        logger.info(f"✓ [测试] 使用已加载的{len(channel_scalers)}个通道标准化器")

    print("\n正在应用通道标准化...", file=sys.stderr)
    with tqdm(total=len(continuous_channels), desc="标准化通道",
              unit="通道", position=0, leave=True, file=sys.stderr) as pbar:
        for ch_idx, orig_ch in enumerate(continuous_channels):
            scaler_info = channel_scalers.get(str(orig_ch))
            if not scaler_info:
                pbar.update(1)
                continue

            method = scaler_info['method']

            if method == 'log1p_standard':
                continuous_grids[:, ch_idx, :, :] = np.log1p(
                    np.maximum(continuous_grids[:, ch_idx, :, :], 0)
                )
                continuous_grids[:, ch_idx, :, :] = (
                                                            continuous_grids[:, ch_idx, :, :] - scaler_info['mean']
                                                    ) / scaler_info['std']
                pbar.set_postfix_str(f"Ch{orig_ch}: Log1p+Std")

            elif method == 'zscore':
                continuous_grids[:, ch_idx, :, :] = (
                                                            continuous_grids[:, ch_idx, :, :] - scaler_info['mean']
                                                    ) / scaler_info['std']
                pbar.set_postfix_str(f"Ch{orig_ch}: Z-score")

            elif method == 'minmax':
                scale_min = scaler_info['min']
                scale_max = scaler_info['max']
                continuous_grids[:, ch_idx, :, :] = 2.0 * (
                        continuous_grids[:, ch_idx, :, :] - scale_min
                ) / (scale_max - scale_min) - 1.0
                pbar.set_postfix_str(f"Ch{orig_ch}: MinMax")

            pbar.update(1)

    X_grid_continuous = continuous_grids
    X_grid_layer_index = layer_index_data
    X_cpt = cpts[:, 0:1] if fit_scaler else cpts

    if fit_scaler:
        scalers_output = {
            'channel_scalers': channel_scalers,
            'target_scaler': target_scaler,
            'target_param': target_param,
            'use_layer_index': use_layer_index
        }

        if '_scaler_file' in locals():
            scalers_output[f'_{target_param}_scaler_file'] = _scaler_file

    else:
        scalers_output = None

    logger.info(f"✓ 预处理完成: {len(X_grid_continuous)} 样本, "
                f"连续通道={len(continuous_channels)}, "
                f"Layer_index={'是' if use_layer_index else '否'}")

    return PreprocessResult(
        grid_continuous=X_grid_continuous,
        grid_layer_index=X_grid_layer_index,
        cpt=X_cpt,
        targets=targets,
        scalers=scalers_output
    )



class BaseModel:
    """
    模型接口基类

    Attributes:
        config (Dict[str, Any]): 模型配置字典
        scaler (StandardScaler): 特征标准化器（可选，传统模型使用）
        is_trained (bool): 模型是否已训练
        logger (logging.Logger): 日志记录器
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基类

        Args:
            config: 模型配置字典
        """
        self.config = config
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)

    def fit_scaler(self, X: np.ndarray):
        """
        拟合特征标准化器

        Args:
            X: 训练特征数据
        """
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标准化特征数据

        Args:
            X: 特征数据

        Returns:
            标准化后的特征数据
        """
        return self.scaler.transform(X)



class UnifiedDataPreprocessor:
    """
    统一的数据预处理工具类
    """

    def __init__(self, config: Dict):
        self.config = config
        self.target_param = config.get('target_param', 'qc').lower()
        self.input_mode = config.get('input_mode', 'flatten_raster')
        self.crop_size = config.get('crop_size', 32)
        self.selected_channels = config.get('selected_channels', None)

        self.cnn_channel_scalers = None
        self.qc_scaler = None
        self.target_scaler = None  # 统一的目标值scaler（支持qc/fs/u2）
        self.target_mean = None
        self.target_std = None

        self.logger = logging.getLogger(__name__)

    def preprocess_flatten_raster(self, file_paths: List[str],
                                  fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """栅格展平预处理"""
        X_grid, X_grid_layer_index, X_cpt, y, scalers = preprocess_raster_data(
            file_paths,
            self.config,
            fit_scaler=fit_scaler,
            crop_size=self.crop_size,
            channel_scalers=self.cnn_channel_scalers if not fit_scaler else None,
            qc_scaler=self.target_scaler if not fit_scaler and self.target_param == 'qc' else None,
            target_mean=None,
            target_std=None
        )

        N, C, H, W = X_grid.shape
        X = X_grid.reshape(N, C * H * W)

        if fit_scaler:
            self.cnn_channel_scalers = scalers['channel_scalers']
            self.target_scaler = scalers['target_scaler']

            if f'_{self.config.get("target_param", "qc")}_scaler_file' in scalers:
                self._target_scaler_file = scalers[f'_{self.config.get("target_param", "qc")}_scaler_file']

            self.logger.info(
                f"✓ 栅格展平预处理完成: {N} 样本, "
                f"输入维度={C}×{H}×{W}={C * H * W}维"
            )

        return X, y

    def preprocess_statistics(self, file_paths: List[str],
                              fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        统计特征预处理

        Args:
            file_paths: pickle文件路径列表
            fit_scaler: 是否拟合标准化器

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标值数组)
        """
        temp_framework_config = {
            'model_type': 'ann',
            'model': self.config
        }
        temp_framework = CPTModelFramework(config_dict=temp_framework_config)

        X, y = temp_framework._load_and_preprocess_data(
            file_paths,
            skip_zero=True,
            skip_invalid_qc=(self.target_param == 'qc'),
            target_param=self.target_param,
            crop_size=self.crop_size,
            selected_channels=self.selected_channels
        )

        self.logger.info(
            f"✓ 统计特征预处理完成: {len(X)} 样本, "
            f"输入维度={X.shape[1] if len(X) > 0 else 0}维, "
            f"目标参数={self.target_param.upper()}"
        )

        return X, y

    def preprocess(self, file_paths: List[str],
                   fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        统一预处理接口

        Args:
            file_paths: pickle文件路径列表
            fit_scaler: 是否拟合标准化器

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标值数组)
        """
        if self.input_mode == 'flatten_raster':
            return self.preprocess_flatten_raster(file_paths, fit_scaler)
        elif self.input_mode == 'statistics':
            return self.preprocess_statistics(file_paths, fit_scaler)
        else:
            raise ValueError(f"不支持的输入模式: {self.input_mode}")

    def get_scalers(self) -> Dict:
        """
        获取所有标准化器（用于保存）

        Returns:
            Dict: 包含所有标准化器的字典
        """
        return {
            'cnn_channel_scalers': self.cnn_channel_scalers,
            'target_scaler': self.target_scaler,
            'qc_scaler': self.qc_scaler,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }

    def set_scalers(self, scalers: Dict):
        """
        设置标准化器（用于加载）

        Args:
            scalers: 包含标准化器的字典
        """
        self.cnn_channel_scalers = scalers.get('cnn_channel_scalers')
        self.target_scaler = scalers.get('target_scaler')
        self.qc_scaler = scalers.get('qc_scaler')
        self.target_mean = scalers.get('target_mean')
        self.target_std = scalers.get('target_std')


class CPTModelFramework:
    """
    CPT预测模型主框架

    支持的模型类型：
    - 'cnn': 卷积神经网络
    - 'ann': 人工神经网络
    - 'rf': 随机森林
    - 'svr': 支持向量回归

    Attributes:
        config (Dict[str, Any]): 配置字典
        model_type (str): 模型类型（小写）
        model: 模型实例
        scaler (StandardScaler): 全局特征标准化器
        is_trained (bool): 是否已训练
        logger (logging.Logger): 日志记录器
    """

    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("必须提供config_path或config_dict")

        self.name = self.config.get('name', 'CPTModelFramework')
        self.model_type = self.config.get('model_type', 'ann').lower()
        self.logger = logging.getLogger(self.name)

        module_mapping = {
            'rf': {'module': 'RF', 'class': 'RFModel'},
            'ann': {'module': 'ANN', 'class': 'ANNModel'},
            'cnn': {'module': 'CNN', 'class': 'CNNModel'},
            'svr': {'module': 'SVR', 'class': 'SVRModel'},
            'gbdt': {'module': 'GBDT', 'class': 'GBDTModel'},
        }

        if self.model_type not in module_mapping:
            raise ValueError(f"不支持的模型类型: {self.model_type}. 支持: {list(module_mapping.keys())}")

        mapping = module_mapping[self.model_type]
        module_name = mapping['module']
        class_name = mapping['class']

        try:
            model_module = None
            import_error = None

            try:
                model_module = importlib.import_module(f"Modle.{module_name}")
            except ImportError as e1:
                import_error = e1
                try:
                    current_dir = Path(__file__).parent
                    parent_dir = current_dir.parent
                    if str(parent_dir) not in sys.path:
                        sys.path.insert(0, str(parent_dir))
                    model_module = importlib.import_module(f"Modle.{module_name}")
                except ImportError as e2:
                    try:
                        model_module = importlib.import_module(module_name)
                    except ImportError as e3:
                        raise ImportError(f"无法导入模块 '{module_name}': {e1}, {e2}, {e3}")

            model_class = getattr(model_module, class_name)
            self.model = model_class(self.config.get('model', {}))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"模型模块 '{module_name}.py' 或类 '{class_name}' 未找到: {e}")

        self.scaler = StandardScaler()
        self.is_trained = False

        self.train_config = self.config.get('training', {})
        self.val_size = self.train_config.get('val_size', 0.1)
        self.random_state = self.train_config.get('random_state', 42)

        model_config = self.config.get('model', {})
        self.target_param = model_config.get('target_param', 'qc').lower()
        if self.target_param not in ['qc', 'fs', 'u2']:
            raise ValueError(f"target_param必须是'qc', 'fs'或'u2'之一，当前值: {self.target_param}")

        if hasattr(self.model, 'target_param'):
            self.model.target_param = self.target_param

        self.logger.info(f"框架初始化: 模型类型={self.model_type}, 使用模块={module_name}, 目标参数={self.target_param.upper()}")

    def _load_raster_data(self, file_paths: List[str],
                          skip_zero: bool = True,
                          process_nan: bool = True,
                          crop_size: int = None,
                          selected_channels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载栅格数据（通用函数，供所有模型使用）
        支持14通道64×64数据，支持裁剪和通道选择

        注意：全为NaN的栅格样本会被自动过滤掉，不参与训练和预测。
        Args:
            file_paths: pickle文件路径列表
            skip_zero: 是否跳过全零栅格
            process_nan: 是否处理NaN值（处理部分NaN的情况）
            crop_size: 裁剪后的尺寸（9-64之间，None表示不裁剪）
            selected_channels: 选择的通道列表（0-13，None表示使用所有通道）
        Returns:
            Tuple[np.ndarray, np.ndarray]: (栅格数据, CPT点数据)
                - 栅格数据: (N, C, H, W) - C为选择的通道数，H×W为裁剪后的尺寸
                - CPT点数据: (N, 4) - [depth, fs, u2, qc] - 已过滤全NaN样本
        """
        grids, cpts = [], []
        total_files = len(file_paths)

        with tqdm(total=total_files, desc="加载数据文件", unit="文件", file=sys.stderr) as pbar:
            for path in file_paths:
                if not os.path.exists(path):
                    pbar.update(1)
                    continue

                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)

                    file_samples = 0
                    for item in data.get('rasters', []):
                        raster = item.get('raster')
                        pt = item.get('cpt_point')

                        if not isinstance(raster, np.ndarray) or pt is None or len(pt) < 4:
                            continue

                        if raster.ndim == 3:
                            if raster.shape[2] == 14 or raster.shape[2] == 8:
                                raster = raster.transpose(2, 0, 1)
                            elif raster.shape[0] not in [8, 14]:
                                continue
                        else:
                            continue

                        if raster.shape[0] not in [8, 14]:
                            continue
                        if raster.shape[0] == 8 and raster.shape[1:] != (32, 32):
                            continue
                        if raster.shape[0] == 14 and raster.shape[1:] != (64, 64):
                            continue

                        if crop_size is not None and raster.shape[1] == 64:
                            raster = crop_raster(raster, crop_size)
                        elif crop_size is not None and raster.shape[1] != 64:
                            self.logger.warning(f"数据不是64×64，无法裁剪到{crop_size}×{crop_size}，跳过")
                            continue

                        if selected_channels is not None:
                            raster = raster[selected_channels, :, :]

                        if np.all(np.isnan(raster)):
                            continue

                        has_full_nan_channel = False
                        for ch in range(raster.shape[0]):
                            if np.all(np.isnan(raster[ch])):
                                has_full_nan_channel = True
                                break
                        if has_full_nan_channel:
                            continue

                        if skip_zero and np.all(raster == 0):
                            continue

                        grids.append(raster.astype(np.float32))
                        cpts.append([float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])])
                        file_samples += 1

                    pbar.set_postfix({"样本数": len(grids), "当前文件样本": file_samples})

                except Exception as e:
                    self.logger.error(f"读取 {path} 失败: {e}")

                pbar.update(1)

        print(f"数据加载完成，共 {len(grids)} 个样本", file=sys.stderr)

        if len(grids) > 0:
            num_channels = grids[0].shape[0]
            grid_size = grids[0].shape[1]
            grids = np.array(grids, dtype=np.float32)
        else:
            num_channels = len(selected_channels) if selected_channels is not None else 14
            grid_size = crop_size if crop_size is not None else 64
            grids = np.empty((0, num_channels, grid_size, grid_size), dtype=np.float32)
        cpts = np.array(cpts, dtype=np.float32) if cpts else np.empty((0, 4), dtype=np.float32)

        if len(grids) > 0:
            full_nan_mask = np.array([np.all(np.isnan(grids[i])) for i in range(len(grids))])
            full_nan_count = np.sum(full_nan_mask)

            if full_nan_count > 0:
                self.logger.warning(f"发现 {full_nan_count} 个全为NaN的栅格样本，将被过滤掉")
                valid_mask = ~full_nan_mask
                grids = grids[valid_mask]
                cpts = cpts[valid_mask]
                print(f"已过滤 {full_nan_count} 个全NaN样本，剩余 {len(grids)} 个有效样本", file=sys.stderr)

        if process_nan and len(grids) > 0:
            grids = process_nan_values(grids, self.logger)

        if len(cpts) > 0:
            targets_temp = cpts[:, 3]
            if np.any(np.isnan(targets_temp)) or np.any(np.isinf(targets_temp)):
                nan_count = np.sum(np.isnan(targets_temp))
                inf_count = np.sum(np.isinf(targets_temp))
                self.logger.warning(f"目标值中包含异常值: NaN={nan_count}, Inf={inf_count}")
                cpts = np.nan_to_num(cpts, nan=0.0, posinf=0.0, neginf=0.0)

        return grids, cpts

    def _load_and_preprocess_data(self, file_paths: List[str],
                                   skip_zero: bool = True,
                                   skip_invalid_qc: bool = True,
                                   target_param: str = 'qc',
                                   crop_size: int = None,
                                   selected_channels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载并预处理传统模型（ANN/RF/SVR）的数据

        将栅格数据转换为统计特征向量（每个通道提取mean、std、max、min），
        加上depth特征。支持14通道64×64数据，支持裁剪和通道选择。
        支持单输出回归（根据target_param选择qc/fs/u2）。

        Args:
            file_paths: pickle文件路径列表
            skip_zero: 是否跳过全零栅格
            skip_invalid_qc: 是否跳过无效qc值（不在-1到200范围内，仅对qc参数有效）
            target_param: 目标参数类型（'qc', 'fs', 'u2'），默认'qc'
            crop_size: 裁剪后的尺寸（9-64之间，None表示不裁剪）
            selected_channels: 选择的通道列表（0-13，None表示使用所有通道）

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标值数组)
                - 特征矩阵: (N, C×4+1) - C为选择的通道数，4个统计量（mean、std、max、min）+ depth
                - 目标值数组: (N,) - 单个参数值（根据target_param选择）
        """
        grids, cpts = self._load_raster_data(file_paths, skip_zero=skip_zero, process_nan=True,
                                            crop_size=crop_size, selected_channels=selected_channels)

        if len(grids) == 0:
            # 计算特征维度
            num_channels = len(selected_channels) if selected_channels is not None else 14
            feature_dim = num_channels * 4 + 1  # 每个通道4个统计量 + depth
            return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # 获取实际通道数
        num_channels = grids.shape[1]
        feature_dim = num_channels * 4 + 1  # 每个通道4个统计量 + depth

        features_list = []
        targets_list = []
        skipped_invalid = 0

        # 使用tqdm显示特征提取进度
        with tqdm(total=len(grids), desc="提取统计特征", unit="样本", file=sys.stderr) as pbar:
            for i in range(len(grids)):
                raster = grids[i]
                cpt = cpts[i]  # [depth, fs, u2, qc]

                # 跳过无效qc值（仅对qc参数）：qc值应在-1到200范围内
                if target_param == 'qc' and skip_invalid_qc and (cpt[3] < -1 or cpt[3] > 200):
                    skipped_invalid += 1
                    if cpt[3] > 200:
                        self.logger.debug(f"跳过异常QC值: {cpt[3]:.2f} (超出200上限)")
                    elif cpt[3] < -1:
                        self.logger.debug(f"跳过异常QC值: {cpt[3]:.2f} (小于-1下限)")
                    pbar.update(1)
                    continue

                # 提取统计特征：每个通道提取4个统计量（mean、std、max、min）
                feats = []
                for ch in range(num_channels):
                    d = raster[ch].flatten()
                    d_valid = d[~np.isnan(d) & ~np.isinf(d)]
                    if len(d_valid) > 0:
                        feats.extend([float(d_valid.mean()), float(d_valid.std()),
                                    float(d_valid.max()), float(d_valid.min())])
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0])

                feats.append(cpt[0])
                features_list.append(feats)

                if target_param == 'qc':
                    targets_list.append(float(cpt[3]))
                elif target_param == 'fs':
                    targets_list.append(float(cpt[1]))
                elif target_param == 'u2':
                    targets_list.append(float(cpt[2]))
                else:
                    targets_list.append(float(cpt[3]))

                pbar.update(1)
                if (i + 1) % 100 == 0:
                    pbar.set_postfix({"有效样本": len(features_list), "跳过": skipped_invalid})

        X = np.array(features_list, dtype=np.float32) if features_list else np.empty((0, feature_dim), dtype=np.float32)
        y = np.array(targets_list, dtype=np.float32) if targets_list else np.empty((0,), dtype=np.float32)

        if target_param == 'qc' and len(y) > 0:
            y = preprocess_qc_values(y, cpts[:len(y)], self.logger)

        self.logger.info(f"传统模型预处理完成: {len(X)} 样本 (跳过无效qc {skipped_invalid})")
        self.logger.info(f"特征维度: {X.shape[1] if len(X) > 0 else 0}, 目标参数: {target_param.upper()}")

        if len(y) > 0:
            valid_targets = y[~np.isnan(y) & ~np.isinf(y) & (y != 0)]
            if len(valid_targets) > 0:
                self.logger.info(f"目标值({target_param})统计: 总数={len(y)}, 有效非零值={len(valid_targets)}, "
                               f"范围=[{valid_targets.min():.4f}, {valid_targets.max():.4f}], "
                               f"均值={valid_targets.mean():.4f}, 标准差={valid_targets.std():.4f}")

        return X, y

    def _load_and_preprocess_raster_flatten(self, file_paths: List[str],
                                            crop_size: int = 32,
                                            selected_channels: List[int] = None,
                                            target_param: str = 'qc') -> Tuple[np.ndarray, np.ndarray]:
        """
        加载栅格数据并展平为向量（用于传统模型与CNN公平对比）

        处理流程：
            1. 调用 _load_raster_data 加载完整栅格（与CNN相同）
            2. 应用与CNN相同的标准化策略（分通道标准化）
            3. 展平栅格为1D向量：(N, C, H, W) → (N, C×H×W)
            4. 提取目标值（根据target_param选择qc/fs/u2）

        Args:
            file_paths: pickle文件路径列表
            crop_size: 裁剪尺寸（默认32，与CNN保持一致）
            selected_channels: 选择的通道列表（默认None表示全部14通道）
            target_param: 目标参数类型（'qc', 'fs', 'u2'）

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标值数组)
                - 特征矩阵: (N, C×H×W) - 展平后的栅格向量
                - 目标值数组: (N,) - 单个参数值

        Note:
            - 使用与CNN相同的标准化策略（通过CNN模型的preprocess实现）
            - 支持任意通道组合和栅格尺寸
            - 自动处理QC值的RobustScaler标准化
        """
        try:
            from Modle.CNN import CNNModel
        except ImportError:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from CNN import CNNModel

        temp_cnn_config = self.config.get('model', {}).copy()
        temp_cnn_config['crop_size'] = crop_size
        temp_cnn_config['selected_channels'] = selected_channels
        temp_cnn_config['target_param'] = target_param

        temp_cnn = CNNModel(temp_cnn_config)

        X_grid, X_grid_layer_index, X_cpt, y = temp_cnn.preprocess(
            file_paths,
            normalize=False,
            fit_scaler=True,
            crop_size=crop_size
        )

        N, C, H, W = X_grid.shape
        X_flatten = X_grid.reshape(N, C * H * W)

        self.logger.info(
            f"栅格展平预处理完成: {N} 样本, "
            f"输入维度={C}通道×{H}×{W}={C * H * W}维, "
            f"目标参数={target_param.upper()}"
        )

        self.cnn_scaler_info = {
            'channel_scalers': temp_cnn.channel_scalers,
            'qc_scaler': getattr(temp_cnn, 'qc_scaler', None),
            'target_mean': getattr(temp_cnn, 'target_mean', None),
            'target_std': getattr(temp_cnn, 'target_std', None)
        }

        return X_flatten, y

    def train(self, data_paths: List[str], val_size: float = None) -> Dict[str, float]:
        """
        训练模型

        根据模型类型选择不同的训练流程：
        - CNN模型：使用栅格数据，支持BatchNorm归一化
        - 传统模型：使用统计特征向量

        Args:
            data_paths: 训练数据文件路径列表（已排除测试集）
            val_size: 验证集比例（默认0.1，即9:1分割）

        Returns:
            Dict[str, float]: 训练指标字典（MSE、RMSE、MAE、R²）
        """
        if val_size is None:
            val_size = self.val_size

        self.logger.info(f"开始训练 {self.model_type.upper()} 模型，样本文件数: {len(data_paths)}")
        self.logger.info(f"数据划分：训练集:验证集 = {1-val_size:.1%}:{val_size:.1%} (9:1)")

        if self.model_type == 'cnn':
            # CNN模型：使用栅格数据，支持BatchNorm归一化
            X_grid, X_grid_layer_index, X_cpt, y = self.model.preprocess(data_paths, normalize=False, fit_scaler=True,
                                                    crop_size=self.model.config.get('crop_size', 32))
            from sklearn.model_selection import train_test_split
            if X_grid_layer_index is not None:
                Xg_tr, Xg_val, Xg_idx_tr, Xg_idx_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
                    X_grid, X_grid_layer_index, X_cpt, y, test_size=val_size, random_state=self.random_state, stratify=None
                )
            else:
                Xg_tr, Xg_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
                    X_grid, X_cpt, y, test_size=val_size, random_state=self.random_state, stratify=None
                )
                Xg_idx_tr, Xg_idx_val = None, None
            self.logger.info(f"训练集样本数: {len(y_tr)}, 验证集样本数: {len(y_val)}")
            metrics = self.model.train(Xg_tr, Xg_idx_tr, Xc_tr, y_tr, Xg_val, Xg_idx_val, Xc_val, y_val)
        else:
            model_config = self.config.get('model', {})
            input_mode = model_config.get('input_mode', 'statistics')

            if input_mode == 'flatten_raster':
                X, y = self.model.preprocess(data_paths, fit_scaler=True)
            else:
                crop_size = model_config.get('crop_size', None)
                selected_channels = model_config.get('selected_channels', None)

                X, y = self._load_and_preprocess_data(
                    data_paths,
                    skip_zero=True,
                    skip_invalid_qc=(self.target_param == 'qc'),
                    target_param=self.target_param,
                    crop_size=crop_size,
                    selected_channels=selected_channels
                )
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=self.random_state
            )
            self.logger.info(f"训练集样本数: {len(y_train)}, 验证集样本数: {len(y_val)}")

            print(f"正在拟合特征标准化器...", file=sys.stderr)
            self.model.fit_scaler(X_train)
            X_train_scaled = self.model.transform(X_train)
            X_val_scaled = self.model.transform(X_val) if X_val is not None and len(X_val) > 0 else None

            print(f"开始训练 {self.model_type.upper()} 模型...", file=sys.stderr)

            optimize_params = self.config.get('training', {}).get('optimize_hyperparams', False)
            if self.model_type == 'rf' and optimize_params and hasattr(self.model, 'optimize_hyperparameters'):
                self.logger.info("启用RF超参数优化")
                metrics = self.model.train(X_train_scaled, y_train, X_val_scaled, y_val, optimize_params=True)
            else:
                metrics = self.model.train(X_train_scaled, y_train, X_val_scaled, y_val)

            print(f"{self.model_type.upper()} 模型训练完成！", file=sys.stderr)

        self.is_trained = True
        self.logger.info(f"训练完成 → {metrics}")
        return metrics

    def save_model(self, model_path):
        """
        保存模型到文件

        Args:
            model_path: 模型保存路径
        """
        import pickle
        import os

        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 保存模型和相关组件
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': getattr(self, 'feature_selector', None),
            'model_type': getattr(self, 'model_type', 'unknown')
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"模型已保存至: {model_path}")
        print(f"模型已保存至: {model_path}")

    def load_model(self, model_path):
        """
        从文件加载模型

        Args:
            model_path: 模型文件路径
        """
        import pickle

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        if 'feature_selector' in model_data:
            self.feature_selector = model_data['feature_selector']

        self.logger.info(f"模型已从 {model_path} 加载")
        print(f"模型已从 {model_path} 加载")

    def predict(self, test_paths: List[str]) -> np.ndarray:
        """
        预测测试集

        Args:
            test_paths: 测试数据文件路径列表

        Returns:
            np.ndarray: 预测值数组 (N,)
                - CNN模型：根据target_param返回qc/fs/u2中的一个
                - 传统模型：返回qc值
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        if self.model_type == 'cnn':
            preprocess_result = self.model.preprocess(test_paths, normalize=False, fit_scaler=False,
                                                      crop_size=self.model.config.get('crop_size', 32))
            if len(preprocess_result) == 4:
                X_test_grid, X_test_layer_index, X_test_cpt, _ = preprocess_result
                if len(X_test_grid) == 0:
                    return np.empty((0,), dtype=np.float32)
                return self.model.predict(X_test_grid, X_test_layer_index, X_test_cpt, already_normalized=True)
            else:
                X_test_grid, X_test_cpt, _ = preprocess_result
                if len(X_test_grid) == 0:
                    return np.empty((0,), dtype=np.float32)
                return self.model.predict(X_test_grid, X_test_cpt, already_normalized=True)
        else:
            model_config = self.config.get('model', {})
            input_mode = model_config.get('input_mode', 'statistics')
            
            if input_mode == 'flatten_raster':
                X_test, _ = self.model.preprocess(test_paths, fit_scaler=False)
                if len(X_test) == 0:
                    return np.empty((0,), dtype=np.float32)
                pred = self.model.predict(X_test)
                if pred.ndim > 1:
                    pred = pred.flatten()
                
                if hasattr(self.model, 'target_scaler') and self.model.target_scaler is not None:
                    pred = self.model.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                
                return pred
            else:
                crop_size = model_config.get('crop_size', None)
                selected_channels = model_config.get('selected_channels', None)

                X_test, _ = self._load_and_preprocess_data(
                    test_paths,
                    skip_zero=True,
                    skip_invalid_qc=False,
                    target_param=self.target_param,
                    crop_size=crop_size,
                    selected_channels=selected_channels
                )
                if len(X_test) == 0:
                    return np.empty((0,), dtype=np.float32)
                X_test_scaled = self.model.transform(X_test)
                pred = self.model.predict(X_test_scaled)
                if pred.ndim > 1:
                    pred = pred.flatten()
                
                if hasattr(self.model, 'target_scaler') and self.model.target_scaler is not None:
                    pred = self.model.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                
                return pred

    def predict_with_confidence(self, test_paths: List[str], confidence_level: float = 0.95,
                               n_bootstraps: int = 50, n_forward_passes: int = 50) -> Dict[str, np.ndarray]:
        """
        预测测试集并计算置信区间（支持RF、SVR和ANN模型）

        Args:
            test_paths: 测试数据文件路径列表
            confidence_level: 置信水平 (默认0.95表示95%置信区间)
            n_bootstraps: Bootstrap样本数量 (仅SVR使用，默认50)
            n_forward_passes: Monte Carlo前向传播次数 (仅ANN使用，默认50)

        Returns:
            Dict: 包含mean、lower、upper、std等信息的字典
                - 'mean': 均值预测
                - 'lower': 置信区间下界
                - 'upper': 置信区间上界
                - 'std': 预测标准差
                - 'confidence_level': 置信水平
                - 模型特定的计数: 'n_estimators'(RF), 'n_bootstraps'(SVR), 'n_forward_passes'(ANN)
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        if self.model_type not in ['rf', 'svr', 'ann', 'cnn']:
            raise ValueError("置信区间计算仅支持RF、SVR、ANN和CNN模型")

        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("置信水平必须在(0,1)范围内")

        # 根据模型类型选择置信区间计算方法
        model_config = self.config.get('model', {})
        input_mode = model_config.get('input_mode', 'statistics')

        if self.model_type == 'cnn':
            # CNN特殊处理：需要grids, grids_layer_index, cpts
            crop_size = model_config.get('crop_size', 32)
            grids, grids_layer_index, cpts, _ = self.model.preprocess(
                test_paths, normalize=False, fit_scaler=False, crop_size=crop_size)

            if len(grids) == 0:
                return {
                    'mean': np.empty((0,), dtype=np.float32),
                    'lower': np.empty((0,), dtype=np.float32),
                    'upper': np.empty((0,), dtype=np.float32),
                    'std': np.empty((0,), dtype=np.float32),
                    'confidence_level': confidence_level,
                    'n_forward_passes': 0
                }

            pred_result = self.model.predict_with_confidence_mc_dropout(
                grids, grids_layer_index, cpts, confidence_level, n_forward_passes)

        elif input_mode == 'flatten_raster':
            # 栅格展平模式（用于RF/SVR/ANN）
            X_test, _ = self.model.preprocess(test_paths, fit_scaler=False)
            if len(X_test) == 0:
                empty_result = {
                    'mean': np.empty((0,), dtype=np.float32),
                    'lower': np.empty((0,), dtype=np.float32),
                    'upper': np.empty((0,), dtype=np.float32),
                    'std': np.empty((0,), dtype=np.float32),
                    'confidence_level': confidence_level
                }
                empty_result['n_estimators' if self.model_type == 'rf' else 'n_bootstraps'] = 0
                return empty_result

            if self.model_type == 'rf':
                pred_result = self.model.predict_with_confidence(X_test, confidence_level)
            elif self.model_type == 'svr':
                pred_result = self.model.predict_with_confidence_bootstrap(X_test, confidence_level, n_bootstraps)
            else:  # ANN
                pred_result = self.model.predict_with_confidence_mc_dropout(X_test, confidence_level, n_forward_passes)
        else:
            # 统计特征模式
            crop_size = model_config.get('crop_size', None)
            selected_channels = model_config.get('selected_channels', None)

            X_test, _ = self._load_and_preprocess_data(
                test_paths,
                skip_zero=True,
                skip_invalid_qc=False,  # 预测时不跳过
                target_param=self.target_param,
                crop_size=crop_size,
                selected_channels=selected_channels
            )
            if len(X_test) == 0:
                empty_result = {
                    'mean': np.empty((0,), dtype=np.float32),
                    'lower': np.empty((0,), dtype=np.float32),
                    'upper': np.empty((0,), dtype=np.float32),
                    'std': np.empty((0,), dtype=np.float32),
                    'confidence_level': confidence_level
                }
                empty_result['n_estimators' if self.model_type == 'rf' else 'n_bootstraps'] = 0
                return empty_result

            X_test_scaled = self.model.transform(X_test)

            if self.model_type == 'rf':
                pred_result = self.model.predict_with_confidence(X_test_scaled, confidence_level)
            elif self.model_type == 'svr':
                pred_result = self.model.predict_with_confidence_bootstrap(X_test_scaled, confidence_level, n_bootstraps)
            else:  # ANN
                pred_result = self.model.predict_with_confidence_mc_dropout(X_test_scaled, confidence_level, n_forward_passes)

        # 如果模型使用了target_scaler，需要反标准化所有预测值
        if hasattr(self.model, 'target_scaler') and self.model.target_scaler is not None:
            pred_result['mean'] = self.model.target_scaler.inverse_transform(
                pred_result['mean'].reshape(-1, 1)).flatten()
            pred_result['lower'] = self.model.target_scaler.inverse_transform(
                pred_result['lower'].reshape(-1, 1)).flatten()
            pred_result['upper'] = self.model.target_scaler.inverse_transform(
                pred_result['upper'].reshape(-1, 1)).flatten()
            pred_result['std'] = pred_result['std'] * self.model.target_scaler.scale_[0]

        return pred_result

    def evaluate(self, test_paths: List[str], output_dir: str = None, include_confidence: bool = False,
                 confidence_level: float = 0.95, n_bootstraps: int = 50, n_forward_passes: int = 50) -> Dict[str, float]:
        """
        评估测试集性能，并可选地保存预测结果到CSV和绘制对比曲线

        Args:
            test_paths: 测试数据文件路径列表
            output_dir: 可选，输出目录路径。如果提供，将保存CSV和对比曲线图

        Returns:
            Dict[str, float]: 评估指标字典
                - {param}_mse: 均方误差
                - {param}_rmse: 均方根误差
                - {param}_mae: 平均绝对误差
                - {param}_r2: R²决定系数
        """
        if not self.is_trained:
            raise ValueError("模型未训练")

        # 评估模型
        if self.model_type == 'cnn':
            X_test_grid, X_test_layer_index, X_test_cpt, y_test = self.model.preprocess(
                test_paths, normalize=False, fit_scaler=False,
                crop_size=self.model.config.get('crop_size', 32))
            if len(X_test_grid) == 0:
                return {"error": "测试数据为空"}
            metrics = self.model.evaluate(X_test_grid, X_test_layer_index, X_test_cpt, y_test, already_normalized=True)
        else:
            # 传统模型评估：支持统计特征或栅格展平
            # 从已训练的模型实例中获取input_mode，而不是配置文件
            model_config = self.config.get('model', {})
            if hasattr(self.model, 'input_mode'):
                input_mode = self.model.input_mode
            else:
                input_mode = model_config.get('input_mode', 'statistics')

            if input_mode == 'flatten_raster':
                # 栅格展平模式
                X_test, y_test = self.model.preprocess(test_paths, fit_scaler=False)
                if len(X_test) == 0:
                    return {"error": "测试数据为空"}
                metrics = self.model.evaluate(X_test, y_test)
            else:
                # 统计特征模式
                crop_size = model_config.get('crop_size', None)
                selected_channels = model_config.get('selected_channels', None)

                X_test, y_test = self._load_and_preprocess_data(
                    test_paths,
                    skip_zero=True,
                    skip_invalid_qc=False,  # 评估时不跳过
                    target_param=self.target_param,
                    crop_size=crop_size,
                    selected_channels=selected_channels
                )
                if len(X_test) == 0:
                    return {"error": "测试数据为空"}
                X_test_scaled = self.model.transform(X_test)
                metrics = self.model.evaluate(X_test_scaled, y_test)

        # 如果提供了output_dir，保存预测结果和对比曲线
        if output_dir is not None:
            self.save_results_to_csv(test_paths, output_dir, include_confidence=include_confidence,
                                   confidence_level=confidence_level, n_bootstraps=n_bootstraps,
                                   n_forward_passes=n_forward_passes)
            self.logger.info(f"预测结果已保存到: {output_dir}")

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实值数组
            y_pred: 预测值数组

        Returns:
            Dict[str, float]: 包含MSE、RMSE、MAE、R²的字典
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def save_results_to_csv(self, test_paths: List[str], output_dir: str, test_data=None,
                           include_confidence: bool = False, confidence_level: float = 0.95,
                           n_bootstraps: int = 50, n_forward_passes: int = 50):
        """
        保存预测结果到CSV文件并生成对比曲线图

        为每个测试文件生成：
        - CSV文件：包含depth、真实值、预测值
        - PNG图片：真实值与预测值的对比曲线

        Args:
            test_paths: 测试集文件路径列表
            output_dir: 输出目录路径
            test_data: 可选的预处理数据 (X_test, y_test, predictions_all)，用于避免重复处理
        """
        os.makedirs(output_dir, exist_ok=True)

        # 获取目标参数名称
        if self.model_type == 'cnn':
            target_param = self.model.target_param  # 'qc', 'fs', 'u2'
        else:
            # 传统模型（ANN、RF、SVR等）也可能有target_param属性
            if hasattr(self.model, 'target_param'):
                target_param = self.model.target_param
            else:
                # 从配置中获取
                model_config = self.config.get('model', {})
                target_param = model_config.get('target_param', 'qc')

        saved_count = 0

        if test_data is not None and self.model_type == 'cnn':
            # 使用提供的预处理数据，避免重复处理
            X_test, y_test, predictions_all = test_data
            self.logger.info(f"使用提供的预处理数据: {len(predictions_all)} 个预测结果")

            # 对于CNN，需要按文件重新组织数据
            pred_idx = 0
            for file_path in test_paths:
                well_name = os.path.basename(file_path).replace('.pickle', '')

                # 为单个文件生成预测
                try:
                    # 获取该文件对应的预测结果数量
                    # 这里需要重新预处理单个文件来获取预测数量
                    grids, grids_layer_index, cpts, _ = self.model.preprocess(
                        [file_path], normalize=False, fit_scaler=False,
                        crop_size=self.model.config.get('crop_size', 32)
                    )

                    n_predictions = len(grids)
                    single_file_predictions = predictions_all[pred_idx:pred_idx + n_predictions]

                    # 计算置信区间（如果启用）
                    if include_confidence and self.model_type in ['rf', 'svr', 'ann', 'cnn']:
                        # CNN模型：重新计算置信区间
                        confidence_result = self.model.predict_with_confidence_mc_dropout(
                            grids, grids_layer_index, cpts, confidence_level, n_forward_passes
                        )
                        confidence_interval = {
                            'lower': confidence_result['lower'],
                            'upper': confidence_result['upper'],
                            'std': confidence_result['std']
                        }
                    else:
                        confidence_interval = None

                    if len(single_file_predictions) > 0:
                        self._save_single_file_results(file_path, single_file_predictions, output_dir, target_param,
                                                     confidence_interval)
                        saved_count += 1
                        pred_idx += n_predictions
                    else:
                        self.logger.warning(f"文件 {well_name} 没有有效预测结果")
                except Exception as e:
                    self.logger.error(f"处理文件 {well_name} 失败: {e}")
        else:
            # 传统方式：为每个文件单独预测（向后兼容）
            for file_path in test_paths:
                try:
                    if include_confidence and self.model_type in ['rf', 'svr', 'ann', 'cnn']:
                        # RF/SVR/ANN/CNN模型：同时计算预测值和置信区间
                        confidence_result = self.predict_with_confidence([file_path], confidence_level, n_bootstraps, n_forward_passes)
                        single_file_predictions = confidence_result['mean']
                        confidence_interval = {
                            'lower': confidence_result['lower'],
                            'upper': confidence_result['upper'],
                            'std': confidence_result['std']
                        }
                    else:
                        single_file_predictions = self.predict([file_path])
                        confidence_interval = None

                    if len(single_file_predictions) > 0:
                        self._save_single_file_results(file_path, single_file_predictions, output_dir, target_param,
                                                     confidence_interval)
                        saved_count += 1
                except Exception as e:
                    well_name = os.path.basename(file_path).replace('.pickle', '')
                    self.logger.error(f"处理文件 {well_name} 失败: {e}")

        self.logger.info(f"CSV保存完成: {saved_count}/{len(test_paths)} 文件")

    def _save_single_file_results(self, file_path: str, predictions: np.ndarray, output_dir: str, target_param: str,
                                 confidence_interval: Dict = None):
        """
        保存单个文件的结果到CSV和生成对比图

        Args:
            file_path: 文件路径
            predictions: 该文件的预测结果
            output_dir: 输出目录
            target_param: 目标参数名称
        """
        well_name = os.path.basename(file_path).replace('.pickle', '')
        output_file = os.path.join(output_dir, f"{well_name}_results.csv")

        # 获取crop_size参数（与训练/预测阶段保持一致）
        crop_size = self.model.config.get('crop_size', 32) if hasattr(self.model, 'config') else 32
        if self.model_type == 'cnn':
            crop_size = self.model.config.get('crop_size', 32)
        else:
            # 对于传统模型，从框架配置中获取
            model_config = self.config.get('model', {})
            crop_size = model_config.get('crop_size', 32)

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            depths, true_vals = [], []
            pred_idx = 0

            for raster_info in data.get('rasters', []):
                cpt_point = raster_info['cpt_point']
                raster = raster_info['raster']

                # 适配多种数据集格式
                is_valid_shape = False
                if isinstance(raster, np.ndarray) and raster.ndim == 3:
                    if (raster.shape == (8, 32, 32) or raster.shape == (32, 32, 8) or
                        raster.shape == (14, 64, 64) or raster.shape == (64, 64, 14)):
                        is_valid_shape = True

                if not is_valid_shape:
                    continue

                # ========== 与_load_single_file保持完全一致的过滤逻辑 ==========

                # 处理raster形状
                if raster.ndim == 3:
                    if raster.shape[2] == 14:
                        raster = raster.transpose(2, 0, 1)
                    elif raster.shape[0] != 14:
                        continue
                else:
                    continue

                if raster.shape != (14, 64, 64):
                    continue

                # 中心裁剪（与训练/预测阶段完全一致）
                raster = _crop_raster(raster, crop_size)

                # 跳过全零栅格
                if np.all(raster == 0):
                    continue

                # 跳过有全NaN通道的样本
                has_full_nan_channel = False
                for ch in range(14):
                    if np.all(np.isnan(raster[ch])):
                        has_full_nan_channel = True
                        break

                if has_full_nan_channel:
                    continue

                # 根据target_param进行范围过滤（与_load_single_file完全一致）
                if target_param == 'qc':
                    qc_value = float(cpt_point[3])
                    if not (0 <= qc_value <= 100):
                        continue  # 过滤超出0-100范围的QC值
                    true_val = qc_value
                elif target_param == 'fs':
                    fs_value = float(cpt_point[1])
                    # FS值范围过滤：合理的FS值应该在0-5范围内，过滤异常值
                    if fs_value < 0 or fs_value > 5 or np.isnan(fs_value) or np.isinf(fs_value):
                        continue  # 过滤异常FS值
                    true_val = fs_value
                elif target_param == 'u2':
                    true_val = float(cpt_point[2])
                else:
                    true_val = float(cpt_point[3])

                # 添加到结果列表
                depths.append(cpt_point[0])
                true_vals.append(true_val)

            # 只在有有效数据时保存
            if depths and len(predictions) > 0:
                # 现在depths的数量应该与predictions完全匹配（都经过相同的过滤）
                if len(depths) != len(predictions):
                    self.logger.warning(
                        f"文件 {well_name}: 即使应用相同的过滤逻辑，预测结果数量({len(predictions)})仍与有效样本数量({len(depths)})不匹配"
                    )
                    # 仍然使用较小的数量作为安全措施
                    min_len = min(len(depths), len(predictions))
                    pred_vals = predictions[:min_len]
                    depths_used = depths[:min_len]
                    true_vals_used = true_vals[:min_len]
                else:
                    # 数量匹配，使用全部数据
                    pred_vals = predictions
                    depths_used = depths
                    true_vals_used = true_vals

                # 保存CSV
                df_dict = {
                    'depth': depths_used,
                    f'true_{target_param}': true_vals_used,
                    f'pred_{target_param}': pred_vals
                }

                # 如果有置信区间数据，添加到DataFrame
                if confidence_interval is not None:
                    ci_lower = confidence_interval.get('lower', [])
                    ci_upper = confidence_interval.get('upper', [])
                    ci_std = confidence_interval.get('std', [])

                    # 确保置信区间数据的长度与预测数据一致
                    if len(ci_lower) >= min_len:
                        df_dict[f'pred_{target_param}_lower'] = ci_lower[:min_len]
                        df_dict[f'pred_{target_param}_upper'] = ci_upper[:min_len]
                        df_dict[f'pred_{target_param}_std'] = ci_std[:min_len]

                df = pd.DataFrame(df_dict)

                df.to_csv(output_file, index=False)
                self.logger.info(f"保存: {output_file} ({min_len} 点)")

                # 绘制对比曲线
                filter_max = CHART_FILTER_QC_MAX if target_param == 'qc' else None
                xlim = CHART_XLIM if target_param == 'qc' else None

                plot_file = plot_comparison_curve(
                    df=df,
                    well_name=well_name,
                    output_dir=output_dir,
                    target_param=target_param,
                    filter_max=filter_max,
                    figsize=CHART_FIGSIZE,
                    xlim=xlim,
                    invert_yaxis=CHART_INVERT_YAXIS
                )
                self.logger.info(f"保存: {plot_file}")
            else:
                self.logger.warning(f"文件 {well_name}: 无有效预测结果或真实数据")

        except Exception as e:
            self.logger.error(f"保存 {well_name} 失败: {e}", exc_info=True)

    def save(self, model_path: str):
        """
        保存模型到文件

        Args:
            model_path: 模型保存路径

        Raises:
            ValueError: 如果模型未训练
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
        self.model.save(model_path)
        self.logger.info(f"框架模型保存: {model_path}")

    @classmethod
    def load(cls, model_path: str, config_path: str = None,
             config_dict: Dict[str, Any] = None) -> 'CPTModelFramework':
        """
        从文件加载已训练的模型

        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径（可选）
            config_dict: 配置字典（可选，与config_path二选一）

        Returns:
            CPTModelFramework: 加载后的框架实例

        Raises:
            ValueError: 如果未提供配置
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_dict:
            config = config_dict
        else:
            raise ValueError("加载需提供config_path或config_dict")

        # 标准化 model_type 为小写
        model_type = config.get('model_type', 'ann').lower()

        # 映射
        module_mapping = {
            'rf': {'module': 'RF', 'class': 'RFModel'},
            'ann': {'module': 'ANN', 'class': 'ANNModel'},
            'cnn': {'module': 'CNN', 'class': 'CNNModel'},
            'svr': {'module': 'SVR', 'class': 'SVRModel'},
            'gbdt': {'module': 'GBDT', 'class': 'GBDTModel'},
        }

        if model_type not in module_mapping:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持: {list(module_mapping.keys())}")

        mapping = module_mapping[model_type]
        module_name = mapping['module']
        class_name = mapping['class']

        instance = cls(config_dict=config)
        model_config = config.get('model', {})

        # 使用与__init__相同的导入逻辑
        model_module = None
        try:
            model_module = importlib.import_module(f"Modle.{module_name}")
        except ImportError:
            try:
                current_dir = Path(__file__).parent
                parent_dir = current_dir.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                model_module = importlib.import_module(f"Modle.{module_name}")
            except ImportError:
                model_module = importlib.import_module(module_name)

        model_class = getattr(model_module, class_name)
        instance.model = model_class.load(model_path, model_config)
        instance.is_trained = True
        return instance



def _get_scaler_path(config: Dict, key: str, default_name: str) -> str:
    """获取scaler文件的绝对路径"""
    scaler_file = config.get(key, default_name)
    if not os.path.isabs(scaler_file):
        model_dir = config.get('model_dir', '..')
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        scaler_file = os.path.join(model_dir, scaler_file)
    return scaler_file


def validate_scalers(
        channel_scalers: Optional[Dict],
        qc_scaler: Optional[Any],
        target_param: str,
        fit_scaler: bool
) -> None:
    """验证测试阶段的标准化器是否存在
    
    Args:
        channel_scalers: 通道标准化器字典
        qc_scaler: QC标准化器
        target_param: 目标参数名称
        fit_scaler: 是否为训练阶段
        
    Raises:
        ValueError: 如果测试阶段缺少必要的标准化器
    """
    if not fit_scaler:
        # 测试阶段：验证标准化器存在
        if channel_scalers is None:
            raise ValueError("测试阶段：缺少channel_scalers，请先训练模型或加载已保存的标准化器")
        
        if target_param == 'qc' and qc_scaler is None:
            raise ValueError("测试阶段：缺少qc_scaler，请先训练模型或加载已保存的标准化器")


def _batch_fill_nan_optimized(grids: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """向量化处理所有通道的NaN值（优化版本）"""
    if not (np.any(np.isnan(grids)) or np.any(np.isinf(grids))):
        return grids

    nan_count = np.sum(np.isnan(grids))
    inf_count = np.sum(np.isinf(grids))
    logger.info(f"栅格数据中包含异常值: NaN={nan_count}, Inf={inf_count}")

    N, C, H, W = grids.shape
    result = grids.copy()

    # 向量化：找出需要处理的样本和通道
    has_nan = np.any(np.isnan(grids) | np.isinf(grids), axis=(2, 3))  # (N, C)
    indices = np.argwhere(has_nan)  # [[i1, ch1], [i2, ch2], ...]

    filled_count = 0
    full_nan_channels = 0

    with tqdm(total=len(indices), desc="处理NaN值", unit="通道", file=sys.stderr) as pbar:
        for i, ch in indices:
            ch_data = result[i, ch, :, :]
            if np.all(np.isnan(ch_data)):
                result[i, ch, :, :] = 0.0
                full_nan_channels += 1
            else:
                result[i, ch, :, :] = fill_nan_with_nearest(ch_data)
                filled_count += 1
            pbar.update(1)

    # 处理Inf值
    if np.any(np.isinf(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"NaN值填充完成: {filled_count} 个部分NaN通道使用最邻近填充, "
                f"{full_nan_channels} 个全NaN通道使用0填充")

    return result


def filter_qc_range(
        grids: np.ndarray,
        cpts: np.ndarray,
        targets: np.ndarray,
        min_value: float,
        max_value: float,
        logger: logging.Logger
) -> tuple:
    """过滤QC值范围，保留在[min_value, max_value]范围内的样本
    
    Args:
        grids: 栅格数据 (N, C, H, W)
        cpts: CPT数据 (N, 1)
        targets: 目标值 (N,)
        min_value: 最小值
        max_value: 最大值
        logger: 日志记录器
        
    Returns:
        过滤后的 (grids, cpts, targets)
    """
    valid_mask = (targets >= min_value) & (targets <= max_value)
    invalid_count = np.sum(~valid_mask)
    
    if invalid_count > 0:
        logger.warning(f"QC值范围过滤: 移除了{invalid_count}个样本（不在[{min_value}, {max_value}]范围内）")
        logger.info(f"过滤前样本数: {len(targets)}, 过滤后样本数: {np.sum(valid_mask)}")
        
        grids = grids[valid_mask]
        cpts = cpts[valid_mask]
        targets = targets[valid_mask]
    else:
        logger.info(f"所有QC值都在[{min_value}, {max_value}]范围内，无需过滤")
    
    return grids, cpts, targets


def _fit_channel_scalers(
        continuous_grids: np.ndarray,
        continuous_channels: List[int],
        config: Dict,
        logger: logging.Logger
) -> Dict:
    """拟合分通道标准化器（训练阶段）"""
    full_channel_normalization_strategies = {
        0: 'zscore', 1: 'log1p_standard', 2: 'zscore', 3: 'log1p_standard',
        4: 'minmax', 5: 'minmax', 6: 'minmax', 7: 'log1p_standard',
        8: 'log1p_standard', 9: 'log1p_standard', 10: 'raw', 11: 'raw', 13: 'raw'
    }

    full_minmax_ranges = {
        4: (0.0, 1.0), 5: (-np.pi, np.pi), 6: (-np.pi, np.pi)
    }

    channel_scalers = {}

    for ch_idx, orig_ch in enumerate(continuous_channels):
        ch_data = continuous_grids[:, ch_idx, :, :].flatten()
        ch_data = ch_data[~np.isnan(ch_data) & ~np.isinf(ch_data)]

        if len(ch_data) == 0:
            channel_scalers[str(orig_ch)] = {'method': 'zscore', 'mean': 0.0, 'std': 1.0}
            continue

        strategy = full_channel_normalization_strategies.get(orig_ch, 'zscore')

        if strategy == 'log1p_standard':
            ch_data_log1p = np.log1p(np.maximum(ch_data, 0))
            ch_mean = float(np.mean(ch_data_log1p))
            ch_std = float(np.std(ch_data_log1p))
            channel_scalers[str(orig_ch)] = {
                'method': 'log1p_standard',
                'mean': ch_mean,
                'std': ch_std
            }
        elif strategy == 'zscore':
            ch_mean = float(np.mean(ch_data))
            ch_std = float(np.std(ch_data))
            channel_scalers[str(orig_ch)] = {
                'method': 'zscore',
                'mean': ch_mean,
                'std': ch_std
            }
        elif strategy == 'minmax':
            if orig_ch in full_minmax_ranges:
                data_min, data_max = full_minmax_ranges[orig_ch]
                actual_min = float(np.min(ch_data))
                actual_max = float(np.max(ch_data))
                scale_min = min(data_min, actual_min)
                scale_max = max(data_max, actual_max)
            else:
                scale_min = float(np.min(ch_data))
                scale_max = float(np.max(ch_data))

            if scale_max - scale_min < 1e-6:
                scale_min = 0.0
                scale_max = 1.0

            channel_scalers[str(orig_ch)] = {
                'method': 'minmax',
                'min': scale_min,
                'max': scale_max
            }
        elif strategy == 'raw':
            channel_scalers[str(orig_ch)] = {'method': 'raw'}

    logger.info(f"✓ 已拟合{len(channel_scalers)}个通道的标准化器")
    return channel_scalers


def _apply_channel_scalers(
        continuous_grids: np.ndarray,
        continuous_channels: List[int],
        channel_scalers: Dict
) -> np.ndarray:
    """应用分通道标准化"""
    for ch_idx, orig_ch in enumerate(continuous_channels):
        scaler_info = channel_scalers.get(str(orig_ch))
        if not scaler_info:
            continue

        method = scaler_info['method']

        if method == 'log1p_standard':
            continuous_grids[:, ch_idx, :, :] = np.log1p(
                np.maximum(continuous_grids[:, ch_idx, :, :], 0)
            )
            continuous_grids[:, ch_idx, :, :] = (
                                                        continuous_grids[:, ch_idx, :, :] - scaler_info['mean']
                                                ) / scaler_info['std']
        elif method == 'zscore':
            continuous_grids[:, ch_idx, :, :] = (
                                                        continuous_grids[:, ch_idx, :, :] - scaler_info['mean']
                                                ) / scaler_info['std']
        elif method == 'minmax':
            scale_min = scaler_info['min']
            scale_max = scaler_info['max']
            continuous_grids[:, ch_idx, :, :] = 2.0 * (
                    continuous_grids[:, ch_idx, :, :] - scale_min
            ) / (scale_max - scale_min) - 1.0
        # 'raw' 不做处理

    return continuous_grids

def prepare_data(train_val_files: List[str], test_files: List[str],
                config: Dict, model_type: str, trained_model=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    通用数据准备函数

    根据模型类型加载和预处理数据，返回训练和测试数据。

    Args:
        train_val_files: 训练+验证集文件路径列表
        test_files: 测试集文件路径列表
        config: 模型配置字典
        model_type: 模型类型 ('cnn', 'ann', 'rf', 'svr')
        trained_model: 已训练的模型实例（CNN模型需要，用于保持归一化参数）

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: 训练特征
            - y_train: 训练目标
            - X_test: 测试特征
            - y_test: 测试目标
    """
    logger = logging.getLogger(__name__)

    if model_type.lower() == 'cnn':
        if trained_model is not None:
            model = trained_model
        else:
            try:
                from Modle.CNN import CNNModel
            except ImportError:
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from CNN import CNNModel
            model_config = config.get('model', config)
            model = CNNModel(model_config)

        X_train_grid, X_train_layer_index, X_train_cpt, y_train = model.preprocess(train_val_files, normalize=False, fit_scaler=True)
        X_test_grid, X_test_layer_index, X_test_cpt, y_test = model.preprocess(test_files, normalize=False, fit_scaler=False)

        return X_train_grid, X_train_layer_index, X_train_cpt, y_train, X_test_grid, X_test_layer_index, X_test_cpt, y_test, model

    else:
        model_config = config.get('model', {})

        temp_config = {'model_type': model_type, 'model': model_config}
        temp_framework = CPTModelFramework(config_dict=temp_config)

        X_train, y_train = temp_framework._load_and_preprocess_data(
            train_val_files,
            skip_zero=True,
            skip_invalid_qc=(model_config.get('target_param', 'qc') == 'qc'),
            target_param=model_config.get('target_param', 'qc'),
            crop_size=model_config.get('crop_size', None),
            selected_channels=model_config.get('selected_channels', None)
        )

        X_test, y_test = temp_framework._load_and_preprocess_data(
            test_files,
            skip_zero=True,
            skip_invalid_qc=False,
            target_param=model_config.get('target_param', 'qc'),
            crop_size=model_config.get('crop_size', None),
            selected_channels=model_config.get('selected_channels', None)
        )

    logger.info(f"数据准备完成: 训练样本={len(X_train)}, 测试样本={len(X_test)}")
    return X_train, y_train, X_test, y_test


def split_train_val(X_train: np.ndarray, y_train: np.ndarray, val_size: float = 0.1, random_state: int = 42):
    """
    将训练数据分割为训练集和验证集

    Args:
        X_train: 训练特征
        y_train: 训练目标
        val_size: 验证集比例
        random_state: 随机种子

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train, X_val, y_train, y_val)
    """
    from sklearn.model_selection import train_test_split

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    return X_train_split, X_val, y_train_split, y_val


def split_train_val_cnn(X_train_grid: np.ndarray, X_train_layer_index: np.ndarray,
                       X_train_cpt: np.ndarray, y_train: np.ndarray,
                       val_size: float = 0.1, random_state: int = 42):
    """将CNN训练数据分割为训练集和验证集"""
    from sklearn.model_selection import train_test_split

    indices = np.arange(X_train_grid.shape[0])
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state
    )

    X_train_grid_split = X_train_grid[train_indices]
    X_train_layer_index_split = X_train_layer_index[train_indices] if X_train_layer_index is not None else None
    X_train_cpt_split = X_train_cpt[train_indices]
    y_train_split = y_train[train_indices]

    X_val_grid = X_train_grid[val_indices]
    X_val_layer_index = X_train_layer_index[val_indices] if X_train_layer_index is not None else None
    X_val_cpt = X_train_cpt[val_indices]
    y_val = y_train[val_indices]

    return (X_train_grid_split, X_train_layer_index_split, X_train_cpt_split, y_train_split,
            X_val_grid, X_val_layer_index, X_val_cpt, y_val)



def plot_comparison_curve(df: pd.DataFrame, well_name: str, output_dir: str,
                         target_param: str = 'qc', filter_max: float = None,
                         figsize: tuple = (4, 16), xlim: tuple = None,
                         invert_yaxis: bool = True):
    """
    绘制真实值与预测值的对比曲线图

    Args:
        df: 包含depth、true_{param}、pred_{param}列的DataFrame
        well_name: 钻孔名称
        output_dir: 输出目录
        target_param: 目标参数名称（'qc', 'fs', 'u2'）
        filter_max: 过滤上限（仅用于显示，不影响数据）
        figsize: 图表尺寸（宽, 高）
        xlim: X轴范围（tuple: (min, max)）
        invert_yaxis: 是否反转Y轴（深度向下）

    Returns:
        str: 保存的图片文件路径
    """
    import matplotlib.pyplot as plt

    # 过滤异常值（仅用于显示）
    if filter_max is not None:
        df_filtered = df[df[f'true_{target_param}'] <= filter_max]
        if len(df_filtered) == 0:
            df_filtered = df
    else:
        df_filtered = df

    # 确定ylabel
    ylabel_map = {'qc': 'QC (kPa)', 'fs': 'FS (kPa)', 'u2': 'U2 (kPa)'}
    ylabel = ylabel_map.get(target_param, f'{target_param.upper()} (kPa)')

    plt.figure(figsize=figsize)
    plt.plot(
        df_filtered[f'true_{target_param}'], df_filtered['depth'],
        label=f'True {target_param.upper()}', color='blue', linewidth=2
    )
    plt.plot(
        df_filtered[f'pred_{target_param}'], df_filtered['depth'],
        label=f'Predicted {target_param.upper()}', color='red', linewidth=2
    )

    plt.xlabel(ylabel)
    plt.ylabel('Depth (m)')
    plt.title(f'Comparison of True vs Predicted {target_param.upper()} over Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if invert_yaxis:
        plt.gca().invert_yaxis()

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    plt.tight_layout()

    plot_file = os.path.join(output_dir, f"{well_name}_{target_param}_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_file


def evaluate_and_save(model, X_test, X_test_layer_index, y_test,
                     test_files: List[str], output_dir: str,
                     model_type: str, model_path: str, X_test_cpt=None):
    """评估模型并保存结果"""
    logger = logging.getLogger(__name__)

    logger.info(f"\n评估测试集...")
    if model_type.lower() == 'cnn':
        test_metrics = model.evaluate(X_test, X_test_layer_index, X_test_cpt, y_test, already_normalized=True)
    else:
        test_metrics = model.evaluate(X_test, y_test)
    logger.info(f"测试指标: {test_metrics}")

    model.save(model_path)
    logger.info(f"模型已保存: {model_path}")

    if model_type.lower() == 'cnn':
        try:
            import yaml
            with open('Modle/framework_cnn.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"无法加载CNN配置文件，使用默认配置: {e}")
            config = {'model_type': 'cnn', 'model': getattr(model, 'config', {})}

        ci_config = config.get('confidence_interval', {})
        include_ci = ci_config.get('enabled', False)

        if include_ci:
            ci_level = ci_config.get('level', 0.95)
            n_passes = ci_config.get('n_forward_passes', 30)
            save_results_to_csv_cnn(model, test_files, output_dir, test_data=None,
                                   confidence_level=ci_level, n_forward_passes=n_passes)
        else:
            predictions_all = model.predict(X_test, X_test_layer_index, X_test_cpt, already_normalized=True)
            test_data = (X_test, y_test, predictions_all)
            save_results_to_csv_cnn(model, test_files, output_dir, test_data)
        logger.info(f"预测结果已保存到: {output_dir}")

    return test_metrics





# 数据目录
DATA_DIR = r'D:\项目数据\深度学习数据集\dataset_pickle_14通道_64x64'

# 输出目录
OUTPUT_BASE_DIR = r'D:\项目数据\预测结果\CNN-16×16'

# 图表配置
CHART_FILTER_QC_MAX = 300  # qc值过滤上限（用于图表显示）
CHART_FIGSIZE = (4, 16)    # 图表尺寸（宽, 高）
CHART_XLIM = (0, 50)       # X轴范围（仅用于qc）
CHART_INVERT_YAXIS = True  # 是否反转Y轴（深度向下）


def setup_logging(log_dir: str = None, log_file_name: str = None) -> str:
    """
    配置日志系统

    将所有日志输出写入文件，不在控制台显示（tqdm进度条除外）。

    Args:
        log_dir: 日志文件目录（建议从配置文件读取）
        log_file_name: 日志文件名，默认为 framework_log_YYYYMMDD_HHMMSS.txt

    Returns:
        str: 日志文件完整路径
        
    Note:
        建议从yaml配置文件的logging.log_dir读取log_dir参数，避免使用默认值
    """
    from datetime import datetime

    # 设置日志目录（仅作为后备默认值）
    if log_dir is None:
        import warnings
        warnings.warn("未指定log_dir，使用默认值。建议从配置文件读取。", UserWarning)
        log_dir = '../logs'  # 使用相对路径作为默认值

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名
    if log_file_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = f'framework_log_{timestamp}.txt'

    log_file_path = os.path.join(log_dir, log_file_name)

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清除现有的所有处理器（包括可能存在的控制台处理器）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.handlers.clear()

    # 创建文件处理器（只输出到文件，不输出到控制台）
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 不创建控制台处理器，所有日志只写入文件
    # tqdm进度条会输出到stderr，不受日志系统影响，可以在控制台正常显示

    # 记录日志文件路径（使用print输出到控制台，因为此时日志只写入文件）
    print(f"{'='*80}")
    print(f"日志文件已创建: {log_file_path}")
    print(f"所有训练信息（INFO、WARNING、ERROR等）将写入日志文件")
    print(f"控制台将显示tqdm进度条，详细日志请查看日志文件")
    print(f"{'='*80}")

    return log_file_path



if __name__ == "__main__":
    import glob
    from datetime import datetime

    # 配置日志系统
    log_file_path = setup_logging()
    logger = logging.getLogger(__name__)

    # 1. 获取数据文件
    print("正在扫描数据文件...")
    all_files = glob.glob(os.path.join(DATA_DIR, '*.pickle'))
    logger.info(f"总文件数: {len(all_files)}")
    print(f"找到 {len(all_files)} 个数据文件")

    if len(all_files) == 0:
        logger.error(f"错误: 在 {DATA_DIR} 中未找到数据文件")
        print(f"错误: 在 {DATA_DIR} 中未找到数据文件")
        exit(1)

    # 2. 手动指定测试集（4个钻孔）
    print("正在准备测试集...")
    test_files = [
        os.path.join(DATA_DIR, 'CZ7_CPT1.pickle'),
        os.path.join(DATA_DIR, 'CZ7_CPT24.pickle'),
        os.path.join(DATA_DIR, 'CZ7_CPT29.pickle'),
        os.path.join(DATA_DIR, 'CZ7_CPT59.pickle'),
    ]

    # 过滤存在的测试文件
    test_files = [f for f in test_files if os.path.exists(f)]
    if len(test_files) == 0:
        test_files = all_files[:4]
        logger.warning(f"警告: 未找到指定的测试文件，自动选择前4个文件作为测试集")
        print(f"警告: 未找到指定的测试文件，自动选择前4个文件作为测试集")

    logger.info(f"测试集文件数: {len(test_files)}")
    for f in test_files:
        logger.info(f"  - {os.path.basename(f)}")

    # 3. 获取训练+验证集文件 (使用所有可用数据进行训练)
    train_val_files = all_files  # 使用所有文件进行训练和验证
    logger.info(f"\n训练+验证集文件数: {len(train_val_files)}")
    logger.info(f"数据划分: 训练集:验证集 = 9:1 (使用所有{len(all_files)}个文件)")
    print(f"使用全部 {len(all_files)} 个文件进行训练 (90%训练集 + 10%验证集)")


def save_results_to_csv_cnn(model, test_paths: List[str], output_dir: str, test_data=None,
                           confidence_level: float = 0.95, n_forward_passes: int = 30):
    """
    专门为CNN模型保存预测结果到CSV文件

    Args:
        model: 已训练的CNN模型实例
        test_paths: 测试集文件路径列表
        output_dir: 输出目录路径
        test_data: 可选的预处理数据 (X_test, y_test, predictions_all)
        confidence_level: 置信水平 (默认0.95)
        n_forward_passes: Monte Carlo Dropout前向传播次数 (默认30)
    """
    import pandas as pd
    import pickle
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # 获取目标参数名称
    target_param = model.target_param


    saved_count = 0

    if test_data is not None:
        # 使用提供的预处理数据
        X_test, y_test, predictions_all = test_data

        # 对于CNN，需要按文件重新组织数据
        pred_idx = 0
        for file_path in test_paths:
            well_name = os.path.basename(file_path).replace('.pickle', '')

            # 为单个文件生成预测
            try:
                # 获取该文件对应的预测结果数量
                grids, grids_layer_index, cpts, _ = model.preprocess(
                    [file_path], normalize=False, fit_scaler=False,
                    crop_size=model.config.get('crop_size', 32)
                )

                n_predictions = len(grids)
                single_file_predictions = predictions_all[pred_idx:pred_idx + n_predictions]

                # 计算置信区间（CNN模型使用Bootstrap方法）
                confidence_result = model.predict_with_confidence_bootstrap(
                    grids, grids_layer_index, cpts, confidence_level, n_forward_passes
                )
                confidence_interval = {
                    'lower': confidence_result['lower'],
                    'upper': confidence_result['upper'],
                    'std': confidence_result['std']
                }

                # 使用Monte Carlo Dropout的均值替换原有预测，确保预测值在置信区间内
                single_file_predictions = confidence_result['mean']

                if len(single_file_predictions) > 0:
                    _save_single_file_results_cnn(file_path, single_file_predictions, output_dir, target_param,
                                                 confidence_interval)
                    saved_count += 1
                    pred_idx += n_predictions
                else:
                    print(f"文件 {well_name} 没有有效预测结果")
            except Exception as e:
                print(f"处理文件 {well_name} 失败: {e}")
    else:
        # 传统方式：为每个文件单独预测（向后兼容）
        for file_path in test_paths:
            try:
                # 预处理单个文件
                grids, grids_layer_index, cpts, _ = model.preprocess(
                    [file_path], normalize=False, fit_scaler=False,
                    crop_size=model.config.get('crop_size', 32)
                )

                # 计算置信区间（包含预测均值）
                confidence_result = model.predict_with_confidence_bootstrap(
                    grids, grids_layer_index, cpts, confidence_level, n_forward_passes
                )

                # 使用Bootstrap的均值作为预测值，确保预测值在置信区间内
                single_file_predictions = confidence_result['mean']
                confidence_interval = {
                    'lower': confidence_result['lower'],
                    'upper': confidence_result['upper'],
                    'std': confidence_result['std']
                }

                if len(single_file_predictions) > 0:
                    _save_single_file_results_cnn(file_path, single_file_predictions, output_dir, target_param,
                                                 confidence_interval)
                    saved_count += 1
            except Exception as e:
                well_name = os.path.basename(file_path).replace('.pickle', '')
                print(f"处理文件 {well_name} 失败: {e}")

    print(f"CSV保存完成: {saved_count}/{len(test_paths)} 文件")


def _save_single_file_results_cnn(file_path: str, predictions: np.ndarray, output_dir: str, target_param: str,
                                 confidence_interval: Dict = None):
    """
    保存单个CNN文件的结果到CSV（专门为CNN模型优化）

    Args:
        file_path: 文件路径
        predictions: 该文件的预测结果
        output_dir: 输出目录
        target_param: 目标参数名称
        confidence_interval: 置信区间数据
    """
    import pandas as pd
    import pickle
    import numpy as np

    well_name = os.path.basename(file_path).replace('.pickle', '')
    output_file = os.path.join(output_dir, f"{well_name}_results.csv")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        depths, true_vals = [], []
        pred_idx = 0

        for raster_info in data.get('rasters', []):
            cpt_point = raster_info['cpt_point']
            raster = raster_info['raster']

            # 适配多种数据集格式
            is_valid_shape = False
            if isinstance(raster, np.ndarray) and raster.ndim == 3:
                if (raster.shape == (8, 32, 32) or raster.shape == (32, 32, 8) or
                    raster.shape == (14, 64, 64) or raster.shape == (64, 64, 14)):
                    is_valid_shape = True

            # 根据target_param选择对应的真实值
            if target_param == 'qc':
                true_val = cpt_point[3]
                valid_check = (cpt_point[3] >= 0) & (cpt_point[3] <= 100)
            elif target_param == 'fs':
                true_val = cpt_point[1]
                valid_check = True
            elif target_param == 'u2':
                true_val = cpt_point[2]
                valid_check = True
            else:
                true_val = cpt_point[3]
                valid_check = (cpt_point[3] >= 0) & (cpt_point[3] <= 100)

            if (valid_check and is_valid_shape and not np.all(raster == 0)):
                # 检查是否有全NaN通道
                has_full_nan_channel = False
                if raster.shape[0] in [8, 14]:  # (C, H, W) 格式
                    num_channels = raster.shape[0]
                    for ch in range(num_channels):
                        if np.all(np.isnan(raster[ch])):
                            has_full_nan_channel = True
                            break
                elif raster.shape[2] in [8, 14]:  # (H, W, C) 格式
                    num_channels = raster.shape[2]
                    for ch in range(num_channels):
                        if np.all(np.isnan(raster[:, :, ch])):
                            has_full_nan_channel = True
                            break

                # 跳过有全NaN通道的样本
                if has_full_nan_channel:
                    continue

                if pred_idx < len(predictions):
                    depths.append(cpt_point[0])
                    true_vals.append(true_val)
                    pred_idx += 1

        # 只在有有效数据时保存
        if depths and len(predictions) > 0:
            # 使用较小的数量，确保不会越界
            min_len = min(len(depths), len(predictions))
            pred_vals = predictions[:min_len]
            depths_used = depths[:min_len]
            true_vals_used = true_vals[:min_len]

            if min_len < len(depths) or min_len < len(predictions):
                print(f"文件 {well_name}: 预测结果数量({len(predictions)})与有效样本数量({len(depths)})不匹配，使用前{min_len}个样本")

            # 保存CSV
            df_dict = {
                'depth': depths_used,
                f'true_{target_param}': true_vals_used,
                f'pred_{target_param}': pred_vals
            }

            # 如果有置信区间数据，添加到DataFrame
            if confidence_interval is not None:
                ci_lower = confidence_interval.get('lower', [])
                ci_upper = confidence_interval.get('upper', [])
                ci_std = confidence_interval.get('std', [])

                # 确保置信区间数据的长度与预测数据一致
                if len(ci_lower) >= min_len:
                    df_dict[f'pred_{target_param}_lower'] = ci_lower[:min_len]
                    df_dict[f'pred_{target_param}_upper'] = ci_upper[:min_len]
                    df_dict[f'pred_{target_param}_std'] = ci_std[:min_len]

            df = pd.DataFrame(df_dict)
            df.to_csv(output_file, index=False)
            print(f"保存: {output_file} ({min_len} 点)")

            # 绘制对比曲线
            filter_max = CHART_FILTER_QC_MAX if target_param == 'qc' else None
            xlim = CHART_XLIM if target_param == 'qc' else None

            plot_file = plot_comparison_curve(
                df=df,
                well_name=well_name,
                output_dir=output_dir,
                target_param=target_param,
                filter_max=filter_max,
                figsize=CHART_FIGSIZE,
                xlim=xlim,
                invert_yaxis=CHART_INVERT_YAXIS
            )
            print(f"保存: {plot_file}")
        else:
            print(f"文件 {well_name}: 无有效预测结果或真实数据")

    except Exception as e:
        print(f"保存 {well_name} 失败: {e}", exc_info=True)