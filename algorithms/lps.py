# -*- coding: utf-8 -*-
"""
传统LPS算法 - 瞬时频率(IF)提取
Local Peak Search Algorithm with Adaptive IF Search Range
支持策略注入，通过 strategy 参数选择不同的惩罚函数
"""

import numpy as np
from .time_freq import compute_stft
from .Entropy.penalty_factory import get_strategy, PENALTY_STRATEGIES


def find_initial_if(power_spectrum, frequencies, min_freq, max_freq):
    """在搜索范围内找到功率最大的频率作为IF起点

    参数:
        power_spectrum: 可以是单帧 (1D) 或多帧 (2D, shape: [n_freq, n_frames])
        frequencies: 频率数组
        min_freq, max_freq: 搜索范围
    """
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    search_indices = np.where(freq_mask)[0]

    if len(search_indices) == 0:
        raise ValueError(f"搜索范围 [{min_freq}, {max_freq}] Hz 内没有有效频率点")

    # 支持多帧平均
    if power_spectrum.ndim == 2:
        # 多帧：取平均功率谱
        avg_power = np.mean(power_spectrum, axis=1)
        power_in_range = avg_power[search_indices]
    else:
        # 单帧
        power_in_range = power_spectrum[search_indices]

    best_idx = search_indices[np.argmax(power_in_range)]
    return frequencies[best_idx], best_idx


def parabolic_interpolation(power_spectrum, peak_idx, frequencies):
    """抛物线插值：获得亚像素级频率精度"""
    if peak_idx <= 0 or peak_idx >= len(power_spectrum) - 1:
        return frequencies[peak_idx]
    
    y_left = power_spectrum[peak_idx - 1]
    y_center = power_spectrum[peak_idx]
    y_right = power_spectrum[peak_idx + 1]
    
    denominator = 2 * (2 * y_center - y_left - y_right)
    if abs(denominator) < 1e-10:
        return frequencies[peak_idx]
    
    delta = np.clip((y_left - y_right) / denominator, -0.5, 0.5)
    freq_resolution = frequencies[1] - frequencies[0]
    return frequencies[peak_idx] + delta * freq_resolution


def _track_if_single_direction(power, f, start_idx, end_idx, step,
                               min_freq, max_freq, c1, c2,
                               adaptive_range, lambda_smooth, use_interpolation,
                               compute_penalty_func, strategy_params=None,
                               init_frames=10):
    """单方向追踪IF脊

    参数:
        min_freq/max_freq: 频率搜索范围 (Hz)，由外部传入
        compute_penalty_func: 惩罚函数 (callable)，通过策略注入
        strategy_params: 策略额外参数 (dict)
        init_frames: 用于估计初始频率的帧数 (默认10帧)
    """
    if strategy_params is None:
        strategy_params = {}

    num_time_points = power.shape[1]
    if_path = np.zeros(num_time_points)

    # 使用多帧平均来估计初始频率，更稳定
    if step > 0:
        init_end = min(start_idx + init_frames, num_time_points)
        init_power = power[:, start_idx:init_end]
    else:
        init_start = max(start_idx - init_frames + 1, 0)
        init_power = power[:, init_start:start_idx + 1]

    # 初始频率在指定范围内搜索
    initial_if, _ = find_initial_if(init_power, f, min_freq, max_freq)
    if_path[start_idx] = initial_if

    for k in range(start_idx + step, end_idx + step, step):
        prev_if = if_path[k - step]

        # 自适应搜索范围，限制在 min_freq ~ max_freq
        if adaptive_range:
            search_min = max(c1 * prev_if, min_freq)
            search_max = min(c2 * prev_if, max_freq)
        else:
            search_min = min_freq
            search_max = max_freq

        search_indices = np.where((f >= search_min) & (f <= search_max))[0]
        if len(search_indices) == 0:
            if_path[k] = prev_if
            continue

        # 调用注入的惩罚函数
        best_freq, _, _ = compute_penalty_func(
            power_spectrum=power[:, k],
            freq_indices=search_indices,
            frequencies=f,
            prev_if=prev_if,
            lambda_smooth=lambda_smooth,
            use_interpolation=use_interpolation,
            parabolic_interpolation_func=parabolic_interpolation,
            **strategy_params
        )

        # 确保结果在指定范围内
        best_freq = np.clip(best_freq, min_freq, max_freq)
        if_path[k] = best_freq

    return if_path


def _compute_path_smoothness(if_path):
    """计算路径平滑度（一阶差分标准差）"""
    return np.std(np.diff(if_path))


def _fuse_bidirectional_paths(if_forward, if_backward, verbose=False):
    """融合正向和反向追踪路径"""
    smoothness_fwd = _compute_path_smoothness(if_forward)
    smoothness_bwd = _compute_path_smoothness(if_backward)
    mean_diff = np.mean(np.abs(if_forward - if_backward))
    
    if verbose:
        print(f"  正向平滑度: {smoothness_fwd:.4f}, 反向平滑度: {smoothness_bwd:.4f}")
        print(f"  路径平均差异: {mean_diff:.4f} Hz")
    
    if mean_diff < 0.5:
        if_fused = (if_forward + if_backward) / 2
        fusion_method = "平均融合"
    elif smoothness_fwd < smoothness_bwd * 0.8:
        if_fused = if_forward.copy()
        fusion_method = "选择正向"
    elif smoothness_bwd < smoothness_fwd * 0.8:
        if_fused = if_backward.copy()
        fusion_method = "选择反向"
    else:
        if_fused = np.minimum(if_forward, if_backward)
        fusion_method = "逐点选择低频"
    
    if verbose:
        print(f"  融合策略: {fusion_method}")

    return if_fused


def entropy_based_lps(signal_data, fs, nperseg=131072, noverlap=None,
                      min_freq=5, max_freq=50, c1=0.9, c2=1.1,
                      adaptive_range=True, lambda_smooth=0.0,
                      use_interpolation=True, bidirectional=True, verbose=True,
                      strategy='baseline', strategy_params=None):
    """
    基于熵的LPS算法主函数

    参数:
        signal_data: 输入振动信号
        fs: 采样频率 (Hz)
        nperseg: STFT窗口长度
        noverlap: STFT重叠长度，默认75%
        min_freq/max_freq: 频率搜索范围 (Hz)
        c1/c2: 自适应搜索范围系数
        adaptive_range: 是否使用自适应搜索范围
        lambda_smooth: 平滑约束系数
        use_interpolation: 是否使用抛物线插值
        bidirectional: 是否使用双向追踪
        verbose: 是否打印进度信息
        strategy: 惩罚函数策略名称 ('baseline' 等)
        strategy_params: 策略额外参数 (dict)

    返回:
        t: 时间数组, if_estimated: IF数组, f: 频率数组, Zxx: STFT结果
    """
    # 获取惩罚函数策略
    compute_penalty = get_strategy(strategy)
    if strategy_params is None:
        strategy_params = {}
    
    if verbose:
        print(f"LPS算法: fs={fs}Hz, nperseg={nperseg}, freq=[{min_freq},{max_freq}]Hz")
        print(f"  策略: {strategy}")

    # 计算STFT
    f, t, Zxx = compute_stft(signal_data, fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    num_time_points = len(t)

    if verbose:
        print(f"  时间点: {num_time_points}, 频率分辨率: {f[1]-f[0]:.4f}Hz")

    # 正向追踪
    if_forward = _track_if_single_direction(
        power, f, 0, num_time_points - 1, +1,
        min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation,
        compute_penalty, strategy_params
    )

    if not bidirectional:
        if_estimated = if_forward
    else:
        # 反向追踪
        if_backward = _track_if_single_direction(
            power, f, num_time_points - 1, 0, -1,
            min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation,
            compute_penalty, strategy_params
        )
        # 融合
        if_estimated = _fuse_bidirectional_paths(if_forward, if_backward, verbose=verbose)

    if verbose:
        print(f"  结果: 平均IF={np.mean(if_estimated):.2f}Hz, RPM={np.mean(if_estimated)*60:.1f}")

    return t, if_estimated, f, Zxx


def convert_if_to_rpm(if_array):
    """将瞬时频率(Hz)转换为转速(RPM)"""
    return if_array * 60

