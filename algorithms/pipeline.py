# -*- coding: utf-8 -*-
"""
信号处理流水线
封装对单个振动信号的完整处理流程
"""

from .lps import entropy_based_lps, convert_if_to_rpm
from .if_tools import smooth_and_interpolate_if


def process_single_file(signal_data, fs, lps_config, if_smooth_config, verbose=True):
    """
    处理单个振动信号：提取IF + 平滑插值
    
    参数:
        signal_data: 振动信号数据 (1D numpy array)
        fs: 采样频率 (Hz)
        lps_config: LPS算法参数字典
        if_smooth_config: IF平滑插值参数字典
        verbose: 是否打印进度信息
    
    返回:
        result: 包含所有处理结果的字典
            - t_if: IF时间轴 (原始)
            - if_estimated: 原始IF脊线
            - t_full: 插值后时间轴
            - if_interp: 插值后IF脊线
            - trend_interp: 趋势线
            - rpm: 转速 (RPM)
            - f: STFT频率轴
            - Zxx: STFT结果
    """
    # 1. LPS算法提取IF
    t_if, if_estimated, f, Zxx = entropy_based_lps(
        signal_data, fs,
        **lps_config,
        verbose=verbose
    )
    
    # 2. 转换为RPM
    rpm = convert_if_to_rpm(if_estimated)
    
    if verbose:
        print(f"转速范围: {rpm.min():.1f} - {rpm.max():.1f} RPM")
    
    # 3. 平滑和插值
    t_full, if_interp, trend_interp = smooth_and_interpolate_if(
        t_if, if_estimated, **if_smooth_config
    )
    
    if verbose:
        print(f"IF插值完成: {len(if_estimated)} 点 → {len(if_interp)} 点")
    
    return {
        't_if': t_if,
        'if_estimated': if_estimated,
        't_full': t_full,
        'if_interp': if_interp,
        'trend_interp': trend_interp,
        'rpm': rpm,
        'f': f,
        'Zxx': Zxx,
    }


def downsample_signal(signal, factor):
    """
    对信号进行降采样
    
    参数:
        signal: 原始信号 (ndarray)
        factor: 降采样因子 (int)
    
    返回:
        downsampled_signal: 降采样后的信号 (ndarray)
    """
    return signal[::factor]

