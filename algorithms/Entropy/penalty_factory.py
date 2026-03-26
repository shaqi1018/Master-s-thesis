# -*- coding: utf-8 -*-
"""
惩罚函数工厂模块
使用工厂模式组装不同的惩罚函数策略
支持策略注入，便于对比实验
"""

import numpy as np
from .shannon_Entropy import compute_penalty_function as baseline_penalty_function
from .advanced_entropies import (
    shannon_entropy,
    compute_probability_distribution,
    compute_local_entropy,
    renyi_entropy,
    compute_local_renyi_entropy
)


def strategy_baseline(power_spectrum, freq_indices, frequencies,
                      prev_if=None, lambda_smooth=0.0,
                      use_interpolation=True, entropy_weight=0.5,
                      parabolic_interpolation_func=None, **kwargs):
    """
    策略 A: 原版论文逻辑 (Baseline)
    直接调用 cross_Entropy.py 中的原版代码
    
    参数:
        power_spectrum: 当前时刻的功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
        frequencies: 完整频率轴 (ndarray)
        prev_if: 上一时刻的 IF 值 (float)
        lambda_smooth: 平滑约束系数 (float)
        use_interpolation: 是否使用抛物线插值 (bool)
        entropy_weight: 熵项权重 (float)
        parabolic_interpolation_func: 抛物线插值函数 (callable)
        **kwargs: 其他参数（兼容性）
    
    返回:
        best_freq: 最优频率 (float)
        best_idx: 最优频率索引 (int)
        global_entropy: 全局熵值 (float)
    """
    return baseline_penalty_function(
        power_spectrum=power_spectrum,
        freq_indices=freq_indices,
        frequencies=frequencies,
        prev_if=prev_if,
        lambda_smooth=lambda_smooth,
        use_interpolation=use_interpolation,
        entropy_weight=entropy_weight,
        parabolic_interpolation_func=parabolic_interpolation_func
    )


def strategy_renyi(power_spectrum, freq_indices, frequencies,
                   prev_if=None, lambda_smooth=5.0,
                   use_interpolation=True,
                   parabolic_interpolation_func=None,
                   alpha=3, w1=1.0, w2=2.0, **kwargs):
    """
    策略 B: 瑞利熵策略 (Renyi Entropy)

    惩罚函数: J(f,t) = -w1·S̃(f,t)² + w2·H_α(P_t) + λ·|f - f_prev|/f_prev

    参数:
        power_spectrum: 当前时刻的功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
        frequencies: 完整频率轴 (ndarray)
        prev_if: 上一时刻的 IF 值 (float)
        lambda_smooth: 平滑约束系数 (float)
        use_interpolation: 是否使用抛物线插值 (bool)
        parabolic_interpolation_func: 抛物线插值函数 (callable)
        alpha: 瑞利熵阶数 (float)，默认 3
        w1: 能量项权重 (float)，默认 1.0
        w2: 熵项权重 (float)，默认 2.0
        **kwargs: 其他参数（兼容性）

    返回:
        best_freq: 最优频率 (float)
        best_idx: 最优频率索引 (int)
        global_entropy: 全局瑞利熵值 (float)
    """
    power_in_range = power_spectrum[freq_indices]
    max_power = np.max(power_in_range)

    # 1. 能量项: 归一化功率 (越大越好)
    normalized_power = power_in_range / max_power if max_power > 1e-10 else np.zeros(len(freq_indices))

    # 2. 熵项: 计算每个候选频率点的局部瑞利熵 (越尖越好，即越小越好)
    local_entropies = np.zeros(len(freq_indices))
    window_size = max(3, len(freq_indices) // 10)

    for i, idx in enumerate(freq_indices):
        local_entropies[i] = compute_local_renyi_entropy(
            power_spectrum, idx, window_size, alpha=alpha
        )

    # 归一化局部熵
    max_entropy = np.max(local_entropies)
    min_entropy = np.min(local_entropies)
    if max_entropy - min_entropy > 1e-10:
        normalized_entropy = (local_entropies - min_entropy) / (max_entropy - min_entropy)
    else:
        normalized_entropy = np.zeros(len(freq_indices))

    # 3. 构建惩罚函数: J = -w1·P_norm + w2·H_norm + λ·|Δf/f|
    penalty = -w1 * normalized_power + w2 * normalized_entropy

    # 平滑约束项 (跳变越小越好)
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

    # 计算全局瑞利熵（用于信息输出）
    global_prob = compute_probability_distribution(power_spectrum, freq_indices)
    global_entropy = renyi_entropy(global_prob, alpha=alpha)

    return best_freq, best_idx, global_entropy


def strategy_pure_energy(power_spectrum, freq_indices, frequencies,
                         prev_if=None, lambda_smooth=0.0,
                         use_interpolation=True,
                         parabolic_interpolation_func=None, **kwargs):
    """
    策略 C: 纯能量策略（不用熵，不用平滑）

    惩罚函数: J(f,t) = -P(f,t)

    最纯粹的传统LPS算法：只根据功率谱能量选择峰值
    不使用任何熵项，也不使用平滑约束

    参数:
        power_spectrum: 当前时刻的功率谱 (ndarray)
        freq_indices: 搜索范围内的频率索引 (ndarray)
        frequencies: 完整频率轴 (ndarray)
        prev_if: 上一时刻的 IF 值 (float) - 未使用，保持接口一致
        lambda_smooth: 平滑约束系数 (float) - 未使用，保持接口一致
        use_interpolation: 是否使用抛物线插值 (bool)
        parabolic_interpolation_func: 抛物线插值函数 (callable)
        **kwargs: 其他参数（兼容性）

    返回:
        best_freq: 最优频率 (float)
        best_idx: 最优频率索引 (int)
        global_entropy: 0.0 (占位，保持接口一致)
    """
    power_in_range = power_spectrum[freq_indices]

    # 直接找到功率最大的频率点（不用归一化，不用惩罚函数）
    local_idx = np.argmax(power_in_range)
    best_idx = freq_indices[local_idx]

    # 可选的抛物线插值
    if use_interpolation and parabolic_interpolation_func is not None:
        best_freq = parabolic_interpolation_func(power_spectrum, best_idx, frequencies)
    else:
        best_freq = frequencies[best_idx]

    return best_freq, best_idx, 0.0  # 返回0作为熵值占位


# ============================================================
# 策略注册字典：通过字符串名字找到对应的惩罚函数
# ============================================================
PENALTY_STRATEGIES = {
    'baseline': strategy_baseline,      # 香农熵 - 论文原版
    'renyi_v1': strategy_renyi,         # 瑞利熵 - 创新策略
    'pure_energy': strategy_pure_energy, # 纯能量 - 不用熵（对比实验）
    # 后续可添加更多策略:
    # 'spectral_kurtosis': strategy_sk,
}


def get_strategy(strategy_name):
    """
    根据策略名称获取对应的惩罚函数
    
    参数:
        strategy_name: 策略名称 (str)
    
    返回:
        strategy_func: 惩罚函数 (callable)
    
    异常:
        ValueError: 如果策略名称不存在
    """
    if strategy_name not in PENALTY_STRATEGIES:
        available = list(PENALTY_STRATEGIES.keys())
        raise ValueError(f"未知策略: '{strategy_name}'，可用策略: {available}")
    return PENALTY_STRATEGIES[strategy_name]


def register_strategy(name, func):
    """
    动态注册新的惩罚函数策略
    
    参数:
        name: 策略名称 (str)
        func: 惩罚函数 (callable)
    """
    PENALTY_STRATEGIES[name] = func


def list_strategies():
    """
    列出所有可用的策略名称
    
    返回:
        strategies: 策略名称列表 (list)
    """
    return list(PENALTY_STRATEGIES.keys())
