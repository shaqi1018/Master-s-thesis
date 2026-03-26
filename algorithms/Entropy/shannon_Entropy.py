# -*- coding: utf-8 -*-
"""
熵计算与惩罚函数模块
包含概率分布计算、熵计算、基于熵的惩罚函数等功能
用于IF脊线提取中的最优频率选择
"""

import numpy as np


def compute_probability_distribution(power_spectrum, freq_indices):
    """
    计算频率搜索范围内的概率分布
    
    公式: d_k(μ_ki) = S_k(μ_ki) / Σ S_k(μ_kj)
    
    参数:
        power_spectrum: 功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
    
    返回:
        prob: 概率分布 (ndarray)，与 freq_indices 等长
    """
    power = power_spectrum[freq_indices]
    total_power = np.sum(power)
    if total_power > 1e-10:
        return power / total_power
    return np.ones(len(freq_indices)) / len(freq_indices)


def compute_entropy(prob_distribution):
    """
    计算香农熵
    
    公式: H = -Σ d_k · log(d_k)
    
    参数:
        prob_distribution: 概率分布 (ndarray)
    
    返回:
        entropy: 熵值 (float)
    """
    prob = np.clip(prob_distribution, 1e-10, 1.0)
    return -np.sum(prob * np.log(prob))


def compute_local_entropy(power_spectrum, center_idx, window_size=3):
    """
    计算以 center_idx 为中心的局部熵
    
    参数:
        power_spectrum: 功率谱 (ndarray)
        center_idx: 中心频率索引 (int)
        window_size: 局部窗口半宽度 (int)
    
    返回:
        local_entropy: 局部熵值 (float)
    """
    left = max(0, center_idx - window_size)
    right = min(len(power_spectrum), center_idx + window_size + 1)
    local_indices = np.arange(left, right)
    local_prob = compute_probability_distribution(power_spectrum, local_indices)
    return compute_entropy(local_prob)


def compute_penalty_function(power_spectrum, freq_indices, frequencies,
                             prev_if=None, lambda_smooth=0.0, 
                             use_interpolation=True, entropy_weight=0.5,
                             parabolic_interpolation_func=None):
    """
    计算惩罚函数并返回最优频率
    
    惩罚函数 (论文公式8): 
        χ = -|TFD|² + entropy_weight * H + lambda_smooth * |Δf/f|
    
    参数:
        power_spectrum: 当前时刻的功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
        frequencies: 完整频率轴 (ndarray)
        prev_if: 上一时刻的 IF 值 (float)，用于平滑约束
        lambda_smooth: 平滑约束系数 (float)
        use_interpolation: 是否使用抛物线插值 (bool)
        entropy_weight: 熵项权重 (float)
        parabolic_interpolation_func: 抛物线插值函数 (callable)，可选
    
    返回:
        best_freq: 最优频率 (float)
        best_idx: 最优频率索引 (int)
        global_entropy: 全局熵值 (float)
    """
    power_in_range = power_spectrum[freq_indices]
    max_power = np.max(power_in_range)
    
    # 归一化功率
    normalized_power = power_in_range / max_power if max_power > 1e-10 else np.zeros(len(freq_indices))
    
    # 计算每个候选频率点的局部熵
    local_entropies = np.zeros(len(freq_indices))
    window_size = max(3, len(freq_indices) // 10)
    
    for i, idx in enumerate(freq_indices):
        local_entropies[i] = compute_local_entropy(power_spectrum, idx, window_size)
    
    # 归一化局部熵
    max_entropy = np.max(local_entropies)
    min_entropy = np.min(local_entropies)
    if max_entropy - min_entropy > 1e-10:
        normalized_entropy = (local_entropies - min_entropy) / (max_entropy - min_entropy)
    else:
        normalized_entropy = np.zeros(len(freq_indices))
    
    # 构建惩罚函数: χ = -P_norm + w * H_norm + λ * |Δf/f|
    penalty = -normalized_power + entropy_weight * normalized_entropy
    
    # 平滑约束项
    if lambda_smooth > 0 and prev_if is not None:
        freq_deviation = np.abs(frequencies[freq_indices] - prev_if) / prev_if
        penalty += lambda_smooth * freq_deviation
    
    # 找到最小惩罚点
    local_idx = np.argmin(penalty)
    best_idx = freq_indices[local_idx]
    
    # 可选的抛物线插值
    if use_interpolation and parabolic_interpolation_func is not None:
        best_freq = parabolic_interpolation_func(power_spectrum, best_idx, frequencies)
    else:
        best_freq = frequencies[best_idx]
    
    # 计算全局熵（用于信息输出）
    global_prob = compute_probability_distribution(power_spectrum, freq_indices)
    global_entropy = compute_entropy(global_prob)
    
    return best_freq, best_idx, global_entropy
