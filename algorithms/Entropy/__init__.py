# -*- coding: utf-8 -*-
"""
Entropy 模块
包含熵计算、惩罚函数工厂等功能
"""

# Baseline 版本（论文复现）
from .shannon_Entropy import (
    compute_probability_distribution,
    compute_entropy,
    compute_local_entropy,
    compute_penalty_function
)

# 高级熵计算库
from .advanced_entropies import (
    shannon_entropy,
    compute_probability_distribution as adv_compute_prob,
    compute_local_entropy as adv_compute_local_entropy
)

# 惩罚函数工厂
from .penalty_factory import (
    PENALTY_STRATEGIES,
    get_strategy,
    register_strategy,
    list_strategies,
    strategy_baseline
)

__all__ = [
    # Baseline
    'compute_probability_distribution',
    'compute_entropy',
    'compute_local_entropy',
    'compute_penalty_function',
    # Advanced
    'shannon_entropy',
    # Factory
    'PENALTY_STRATEGIES',
    'get_strategy',
    'register_strategy',
    'list_strategies',
    'strategy_baseline',
]
