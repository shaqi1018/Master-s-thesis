# -*- coding: utf-8 -*-
"""
时频变换模块
包含STFT、FFT等基础变换函数
"""

import numpy as np
from scipy import signal


def remove_dc_component(signal_data):
    """去除信号的直流分量（均值）"""
    return signal_data - np.mean(signal_data)


def compute_stft(signal_data, fs, nperseg=1024, noverlap=None, nfft=None, window='hann'):
    """
    计算信号的短时傅里叶变换（STFT）
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
        nperseg: 每个段的长度（样本数），默认1024
        noverlap: 段之间的重叠样本数，默认为nperseg的75%
        nfft: FFT的长度，默认等于nperseg
        window: 窗函数类型，默认'hann'
    
    返回:
        f: 频率数组 (Hz)
        t: 时间数组 (秒)
        Zxx: STFT结果的复数矩阵
    """
    signal_no_dc = remove_dc_component(signal_data)
    
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    
    f, t, Zxx = signal.stft(signal_no_dc, fs=fs, window=window, 
                            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return f, t, Zxx


def compute_fft(signal_data, fs):
    """
    计算信号的快速傅里叶变换（FFT）
    
    参数:
        signal_data: 输入信号数组 (1D numpy array)
        fs: 采样频率 (Hz)
    
    返回:
        f: 频率数组 (Hz)
        magnitude: 归一化后的幅度谱
        phase: 相位谱
    """
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    
    f = np.fft.fftfreq(n, 1/fs)[:n//2]
    magnitude = np.abs(fft_result[:n//2]) / n
    phase = np.angle(fft_result[:n//2])
    
    # 去掉直流分量
    return f[1:], magnitude[1:], phase[1:]

