# -*- coding: utf-8 -*-
"""
双流数据集：同时产出 时域信号 + 阶次谱

用途：
    为双流神经网络提供数据，一路输入时域信号，一路输入阶次谱
    
数据流:
    原始振动信号 --> 时域信号 (time_signal)
                --> 阶次谱 (order_spec)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# 添加父目录到路径，兼容直接运行
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from data.data_utils import load_channel1_data
from algorithms.pipeline import process_single_file
from algorithms.order_tracking import compute_order_envelope_spectrum


class DualStreamDataset(Dataset):
    """
    双流轴承数据集
    
    同时提供时域信号和阶次谱两路数据
    """
    
    FAULT_TYPES = {
        0: 'Healthy',
        1: 'Inner',
        2: 'Outer',
        3: 'Ball',
        4: 'Combination'
    }
    
    def __init__(self, base_path, fs=200000, signal_length=2000000):
        """
        初始化数据集
        
        参数:
            base_path: 数据集根目录
            fs: 采样频率 (Hz)
            signal_length: 每个文件的信号长度
        """
        self.base_path = base_path
        self.fs = fs
        self.signal_length = signal_length
        self.subdirs = self._get_subdirs()
        
        # 数据容器 (build_dataset后填充)
        self.all_time_signals = []   # 时域信号切片
        self.all_if_curves = []       # IF曲线切片 (用于在线计算阶次谱)
        self.all_order_specs = []     # 预计算阶次谱切片 (推荐)
        self.all_labels = []          # 故障标签
        self.all_groups = []          # 分组ID（同一原始文件共享同一group）
        
        # 阶次谱参数
        self.order_spec_length = 1537  # 阶次谱输出长度
        self.downsample_fs = None      # 降采样后的采样频率
    
    def _get_subdirs(self):
        """获取子文件夹列表"""
        subdirs = [d for d in os.listdir(self.base_path)
                   if os.path.isdir(os.path.join(self.base_path, d))]
        subdirs.sort()
        return subdirs
    
    def get_file_path(self, folder_idx, filename):
        """获取文件完整路径"""
        return os.path.join(self.base_path, self.subdirs[folder_idx], filename)
    
    def load_signal(self, folder_idx, filename):
        """加载单个信号文件"""
        file_path = self.get_file_path(folder_idx, filename)
        return load_channel1_data(file_path)
    
    def get_label(self, folder_idx):
        """获取故障类型标签"""
        return folder_idx
    
    def get_label_name(self, folder_idx):
        """获取故障类型名称"""
        return self.FAULT_TYPES.get(folder_idx, 'Unknown')
    
    def list_files(self, folder_idx):
        """列出指定文件夹中的所有.mat文件"""
        folder_path = os.path.join(self.base_path, self.subdirs[folder_idx])
        return [f for f in os.listdir(folder_path) if f.endswith('.mat')]
    
    def build_dataset(self, lps_config, if_smooth_config,
                      window_size=3072, hop_size=None,
                      downsample_factor=10, order_spec_length=1537,
                      precompute_order_spec=True,
                      folder_indices=None, verbose=True):
        """
        遍历所有文件，提取IF并同步切片存储
        
        参数:
            lps_config: LPS算法参数字典
            if_smooth_config: IF平滑插值参数字典
            window_size: 切片窗口长度，默认3072
            hop_size: 切片步长，默认为window_size（无重叠）
            downsample_factor: 降采样因子，默认10
            order_spec_length: 阶次谱输出长度，默认1537
            precompute_order_spec: 是否在构建阶段预计算阶次谱，默认True
            folder_indices: 要处理的文件夹索引列表
            verbose: 是否打印进度信息
        """
        # 默认步长 = 窗口长度（无重叠）
        if hop_size is None:
            hop_size = window_size
        
        # 默认处理全部5个文件夹（0-4）
        if folder_indices is None:
            folder_indices = [0, 1, 2, 3, 4]
        
        # 保存参数
        self.order_spec_length = order_spec_length
        self.downsample_fs = self.fs // downsample_factor
        
        # 清空容器
        self.all_time_signals = []
        self.all_if_curves = []
        self.all_order_specs = []
        self.all_labels = []
        self.all_groups = []
        file_count = 0
        
        # 外层循环：遍历指定的故障类型
        for folder_idx in folder_indices:
            folder_name = self.subdirs[folder_idx]
            label = self.get_label(folder_idx)
            label_name = self.get_label_name(folder_idx)
            
            if verbose:
                print(f"\n[{folder_idx}] {label_name}: {folder_name}")
            
            # 内层循环：遍历.mat文件
            mat_files = self.list_files(folder_idx)
            
            for filename in mat_files:
                file_count += 1
                group_id = file_count - 1
                if verbose:
                    print(f"  处理: {filename}", end=" ... ")
                
                # 加载原始信号
                signal_data = self.load_signal(folder_idx, filename)
                
                # 调用核心处理函数 (提取IF曲线)
                result = process_single_file(
                    signal_data, self.fs, lps_config, if_smooth_config,
                    verbose=False
                )
                if_interp = result['if_interp']  # 拉伸后IF (2000000,)
                
                # 降采样
                signal_data = signal_data[::downsample_factor]
                if_interp = if_interp[::downsample_factor]
                
                # 计算切片数量
                n_slices = (len(signal_data) - window_size) // hop_size + 1
                
                # 切片循环
                for i in range(n_slices):
                    start = i * hop_size
                    end = start + window_size
                    
                    signal_slice = signal_data[start:end]
                    if_slice = if_interp[start:end]
                    
                    self.all_time_signals.append(signal_slice)
                    self.all_if_curves.append(if_slice)

                    if precompute_order_spec:
                        order_spec = compute_order_envelope_spectrum(
                            time_signal=signal_slice,
                            if_curve=if_slice,
                            fs=self.downsample_fs,
                            filter_band=None,
                            target_length=self.order_spec_length
                        )
                        self.all_order_specs.append(order_spec.astype(np.float32))

                    self.all_labels.append(label)
                    self.all_groups.append(group_id)
                
                if verbose:
                    print(f"切片: {n_slices} 片")
        
        # 转换为numpy数组
        self.all_time_signals = np.array(self.all_time_signals, dtype=np.float32)
        self.all_if_curves = np.array(self.all_if_curves, dtype=np.float32)
        self.all_labels = np.array(self.all_labels, dtype=np.int64)
        self.all_groups = np.array(self.all_groups, dtype=np.int64)
        if precompute_order_spec:
            self.all_order_specs = np.array(self.all_order_specs, dtype=np.float32)
        else:
            self.all_order_specs = None
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"双流数据集构建完成!")
            print(f"  文件总数: {file_count}")
            print(f"  降采样: 每{downsample_factor}取1, fs: {self.downsample_fs} Hz")
            print(f"  窗口长度: {window_size}, 步长: {hop_size}")
            print(f"  总切片数: {len(self.all_time_signals)}")
            print(f"  time_signals shape: {self.all_time_signals.shape}")
            print(f"  if_curves shape: {self.all_if_curves.shape}")
            if self.all_order_specs is not None:
                print(f"  order_specs shape: {self.all_order_specs.shape}")
            print(f"  labels shape: {self.all_labels.shape}")
            print(f"  groups shape: {self.all_groups.shape}, 文件组数: {len(np.unique(self.all_groups))}")
            print(f"  阶次谱长度: {self.order_spec_length}")
            print(f"  阶次谱模式: {'预计算缓存' if precompute_order_spec else '在线计算'}")
            print(f"  标签分布: {self._count_labels()}")
    
    def _count_labels(self):
        """统计各标签数量"""
        from collections import Counter
        counts = Counter(self.all_labels)
        return {self.get_label_name(k): v for k, v in sorted(counts.items())}
    
    # ========== PyTorch Dataset 接口 ==========
    def __len__(self):
        """返回数据集大小"""
        return len(self.all_time_signals)
    
    def __getitem__(self, idx):
        """
        获取单个样本 - 双流数据
        
        返回:
            time_signal: 时域振动信号, shape (1, window_size), float32
            order_spec: 阶次谱, shape (1, order_spec_length), float32
            label: 故障类型标签, long
        """
        # 获取时域信号和IF曲线
        time_signal = self.all_time_signals[idx]
        label = self.all_labels[idx]

        # 优先使用预计算阶次谱；若未预计算则在线计算
        if self.all_order_specs is not None:
            order_spec = self.all_order_specs[idx]
        else:
            if_curve = self.all_if_curves[idx]
            order_spec = compute_order_envelope_spectrum(
                time_signal=time_signal,
                if_curve=if_curve,
                fs=self.downsample_fs,
                filter_band=None,  # 自动选择滤波频带
                target_length=self.order_spec_length
            ).astype(np.float32)
        
        # 转换为PyTorch张量，添加通道维度
        time_signal = torch.from_numpy(time_signal[np.newaxis, :]).float()  # (1, 3072)
        order_spec = torch.from_numpy(order_spec[np.newaxis, :]).float()    # (1, 1537)
        label = torch.tensor(label).long()
        
        return time_signal, order_spec, label
