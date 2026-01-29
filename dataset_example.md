## 数据集pickle文件格式示例

{
    # 基本元信息
    "well_name":              "WellA-12",                  # str，井名
    "cpt_surface_coords":     (435672.18, 3859124.67),     # tuple[float, float]，孔口 XY 坐标（投影坐标系，单位：m）
    "elevation_value":        12.45,                       # float，地面高程（m）
    "start_depth":            0.10,                        # float，CPT记录的起始深度（通常很浅，m）
    "depth_offset":           1.82,                        # float，深度对齐偏移量（根据第一层地层对齐，m），可能为0.0
    "step":                   0.05,                        # float，原始CPT采样间隔（m）
    "downsample_step":        5,                           # int，每隔多少原始点取一个（下采样倍数）
    "matched_profile":        "Line_Seis_023",             # str，匹配到的地震剖面名称
    "match_dist":             8.74,                        # float，CPT孔口到最近trace的水平距离（m）
    "central_trace_idx":      142,                         # int，匹配到的中心trace索引（从0开始）
    "used_attributes":        ["Envelope", "InstFreq", "RMS", "Coherency", ......,"Depth to upper", "Depth to surface", "Layer index", "Layer thickness"],
                               # List[str]，实际成功读取并使用的属性名称

    # 完整的原始CPT曲线数据（下采样前）
    "cpt_points": [                                        # List[List[float]]，形状 ≈ (原始采样点数, 4)
        [0.10,  1.24,  0.45,  3.82],     # [DEPTH, fs, u2, qc]
        [0.15,  1.31,  0.48,  4.15],
        ...
        [29.85, 18.76, 4.92, 42.31],
        ...
    ],

    # 最核心部分：每个采样点对应的特征窗口 + 标签 + 坐标
    "rasters": [                                           # List[dict]，长度 = 下采样后的点数（通常几百～一千多）
        {
            "cpt_point":          [12.50, 4.82, 1.96, 26.4],   # List[float]，当前点的 [DEPTH, fs, u2, qc]
            "xyz_coords":         (435672.18, 3859124.67, -0.32),  # Tuple[float,float,float]，真实空间坐标 X,Y,Z（Z向下为负）
            "raster":             <numpy.ndarray shape=(8, 64, 64) dtype=float32>,  # 8通道 × 64×64 栅格
            "trace_idx":          142,                         # int，中心trace索引（与上层一致）
            "used_attrs":         ["Envelope", ...],           # List[str]，与文件级 used_attributes 相同
            "original_index":     250                          # int，该点在原始 cpt_points 中的索引（下采样前）
        },

        {
            "cpt_point":          [12.75, 5.14, 2.11, 28.9],
            "xyz_coords":         (435672.18, 3859124.67, -0.57),
            "raster":             <ndarray shape=(8,64,64)>,
            "trace_idx":          142,
            "used_attrs":         [...],
            "original_index":     255
        },

        ...  # 通常还有几百到一千多个这样的字典
    ]
}
