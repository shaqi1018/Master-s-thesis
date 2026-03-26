# -*- coding: utf-8 -*-
"""
高级熵计算数学库
包含各种熵的计算方法，用于惩罚函数的构建
"""

import numpy as np


def shannon_entropy(prob_distribution):
    """
    计算香农熵 (Shannon Entropy)
    
    公式: H = -Σ p_i · log(p_i)
    
    参数:
        prob_distribution: 概率分布 (ndarray)，需满足 Σp_i = 1
    
    返回:
        entropy: 熵值 (float)
    """
    prob = np.clip(prob_distribution, 1e-12, 1.0)
    return -np.sum(prob * np.log(prob))


def compute_probability_distribution(power_spectrum, freq_indices):
    """
    计算频率搜索范围内的概率分布
    
    公式: p_i = S_i / Σ S_j
    
    参数:
        power_spectrum: 功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
    
    返回:
        prob: 概率分布 (ndarray)
    """
    power = power_spectrum[freq_indices]
    total_power = np.sum(power)
    if total_power > 1e-10:
        return power / total_power
    return np.ones(len(freq_indices)) / len(freq_indices)


def compute_local_entropy(power_spectrum, center_idx, window_size=3, 
                          entropy_func=None):
    """
    计算以 center_idx 为中心的局部熵
    
    参数:
        power_spectrum: 功率谱 (ndarray)
        center_idx: 中心频率索引 (int)
        window_size: 局部窗口半宽度 (int)
        entropy_func: 熵计算函数，默认使用香农熵
    
    返回:
        local_entropy: 局部熵值 (float)
    """
    if entropy_func is None:
        entropy_func = shannon_entropy
    
    left = max(0, center_idx - window_size)
    right = min(len(power_spectrum), center_idx + window_size + 1)
    local_indices = np.arange(left, right)
    local_prob = compute_probability_distribution(power_spectrum, local_indices)
    return entropy_func(local_prob)


def renyi_entropy(prob_distribution, alpha=3):
    """
    计算瑞利熵 (Rényi Entropy)
    
    公式: H_α(P) = 1/(1-α) · log(Σ p_i^α)
    
    特性:
    - α > 1 时，对峰值更敏感，能更好地聚焦能量集中的频率点
    - α → 1 时，退化为香农熵
    - α = 2 称为碰撞熵 (Collision Entropy)
    - α = 3 或 5 常用于峰值检测，忽略噪声
    
    参数:
        prob_distribution: 概率分布 (ndarray)，需满足 Σp_i = 1
        alpha: 熵的阶数 (float)，默认 3
    
    返回:
        entropy: 瑞利熵值 (float)
    """
    prob = np.clip(prob_distribution, 1e-12, 1.0)
    
    if abs(alpha - 1) < 1e-10:
        # α → 1 时退化为香农熵
        return shannon_entropy(prob)
    
    return (1 / (1 - alpha)) * np.log(np.sum(prob ** alpha) + 1e-12)


def compute_local_renyi_entropy(power_spectrum, center_idx, window_size=3, alpha=3):
    """
    计算以 center_idx 为中心的局部瑞利熵
    
    参数:
        power_spectrum: 功率谱 (ndarray)
        center_idx: 中心频率索引 (int)
        window_size: 局部窗口半宽度 (int)
        alpha: 瑞利熵阶数 (float)
    
    返回:
        local_entropy: 局部瑞利熵值 (float)
    """
    left = max(0, center_idx - window_size)
    right = min(len(power_spectrum), center_idx + window_size + 1)
    local_indices = np.arange(left, right)
    local_prob = compute_probability_distribution(power_spectrum, local_indices)
    return renyi_entropy(local_prob, alpha=alpha)
