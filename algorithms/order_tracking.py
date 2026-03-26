# -*- coding: utf-8 -*-
"""
阶次分析模块 (Computed Order Tracking)

核心逻辑:
1. 转速变相位 (积分): IF (Hz) -> 累积相位 (弧度)
2. 角度重采样 (插值): 等时间间隔 -> 等角度间隔
3. 变换求谱 (FFT): 角域信号 FFT -> 阶次谱

扩展功能:
- 阶次包络谱 (Order Envelope Spectrum): 针对轴承冲击性故障的包络解调
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import get_window, hilbert, butter, filtfilt


def compute_order_spectrum(time_signal, if_curve, fs=200000, target_length=1537):
    """
    计算阶次谱 (Computed Order Tracking)
    
    将变速工况下的时域振动信号转换为阶次谱, 消除转速波动的影响.
    
    参数:
        time_signal: 时域振动信号, 形状 (N,)
        if_curve: 瞬时频率曲线 (Hz), 形状 (N,), 与 time_signal 等长
        fs: 采样频率 (Hz), 默认 200000
        target_length: 输出阶次谱长度, 默认 1537 (便于神经网络输入)
    
    返回:
        order_spec: 归一化阶次谱, 形状 (target_length,), 值域 [0, 1]
    """
    # 0. 基础校验: 确保信号与IF曲线等长
    if len(time_signal) != len(if_curve):
        min_len = min(len(time_signal), len(if_curve))
        time_signal = time_signal[:min_len]
        if_curve = if_curve[:min_len]

    # 1. 转速变相位 (积分)
    # 物理公式: phase(t) = 2*pi * integral(f(t)) dt
    dt = 1.0 / fs
    phase = np.cumsum(if_curve) * dt * 2 * np.pi
    
    # 2. 角度重采样 (插值)
    # 建立等角度网格, 点数与原信号一致以保证分辨率
    num_samples = len(time_signal)
    uniform_angle_grid = np.linspace(phase[0], phase[-1], num_samples)
    
    # 三次样条插值: 时域信号 -> 角域信号
    interpolator = interp1d(phase, time_signal, kind='cubic', fill_value='extrapolate')
    angle_domain_signal = interpolator(uniform_angle_grid)
    
    # 3. 加窗 (减少频谱泄漏)
    window = get_window('hann', len(angle_domain_signal))
    angle_domain_signal = angle_domain_signal * window
    
    # 4. FFT 变换得到阶次谱
    fft_result = np.fft.rfft(angle_domain_signal)
    order_amp = np.abs(fft_result)
    
    # 5. 尺寸对齐 (插值到目标长度)
    if target_length is not None and len(order_amp) != target_length:
        x_old = np.linspace(0, 1, len(order_amp))
        x_new = np.linspace(0, 1, target_length)
        order_amp = interp1d(x_old, order_amp, kind='linear')(x_new)
        
    # 6. 归一化到 [0, 1]
    order_spec = order_amp / (np.max(order_amp) + 1e-8)

    return order_spec


def compute_order_envelope_spectrum(time_signal, if_curve, fs=200000,
                                     filter_band=None, target_length=1537):
    """
    计算阶次包络谱 (Order Envelope Spectrum)

    针对轴承冲击性故障的包络解调方法:
    1. 带通滤波 (提取高频共振带)
    2. Hilbert变换提取包络
    3. 角度重采样
    4. FFT得到阶次包络谱

    参数:
        time_signal: 时域振动信号, 形状 (N,)
        if_curve: 瞬时频率曲线 (Hz), 形状 (N,), 与 time_signal 等长
        fs: 采样频率 (Hz), 默认 200000
        filter_band: 带通滤波频率范围 (low, high) Hz,
                     默认 None 表示自动选择 (0.1*fs/2, 0.8*fs/2)
        target_length: 输出阶次谱长度, 默认 1537

    返回:
        order_env_spec: 归一化阶次包络谱, 形状 (target_length,), 值域 [0, 1]
    """
    # 0. 基础校验: 确保信号与IF曲线等长
    if len(time_signal) != len(if_curve):
        min_len = min(len(time_signal), len(if_curve))
        time_signal = time_signal[:min_len]
        if_curve = if_curve[:min_len]

    # 1. 带通滤波 (提取高频共振带，轴承故障冲击能量集中区域)
    nyq = fs / 2  # 奈奎斯特频率

    if filter_band is None:
        # 自动选择: 10%-80% 奈奎斯特频率
        low_freq = 0.1 * nyq
        high_freq = 0.8 * nyq
    else:
        low_freq, high_freq = filter_band

    # 归一化截止频率
    low = low_freq / nyq
    high = high_freq / nyq

    # 确保频率在有效范围内
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    # 设计4阶Butterworth带通滤波器
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, time_signal)

    # 2. Hilbert变换提取包络
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # 3. 转速变相位 (积分)
    dt = 1.0 / fs
    phase = np.cumsum(if_curve) * dt * 2 * np.pi

    # 4. 角度重采样 (对包络信号重采样)
    num_samples = len(envelope)
    uniform_angle_grid = np.linspace(phase[0], phase[-1], num_samples)

    # 三次样条插值: 包络信号 -> 角域包络信号
    interpolator = interp1d(phase, envelope, kind='cubic', fill_value='extrapolate')
    angle_domain_envelope = interpolator(uniform_angle_grid)

    # 5. 加窗 (减少频谱泄漏)
    window = get_window('hann', len(angle_domain_envelope))
    angle_domain_envelope = angle_domain_envelope * window

    # 6. FFT 变换得到阶次包络谱
    fft_result = np.fft.rfft(angle_domain_envelope)
    order_env_amp = np.abs(fft_result)

    # 7. 尺寸对齐 (插值到目标长度)
    if target_length is not None and len(order_env_amp) != target_length:
        x_old = np.linspace(0, 1, len(order_env_amp))
        x_new = np.linspace(0, 1, target_length)
        order_env_amp = interp1d(x_old, order_env_amp, kind='linear')(x_new)

    # 8. 归一化到 [0, 1]
    order_env_spec = order_env_amp / (np.max(order_env_amp) + 1e-8)

    return order_env_spec
