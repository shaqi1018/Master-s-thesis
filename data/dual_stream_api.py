# -*- coding: utf-8 -*-
"""
DualStreamDataset API - External Callable Interface

Usage:
    from data.dual_stream_api import (
        create_dataset,
        create_dataloader,
        get_default_config,
        DualStreamDatasetBuilder
    )

    # Quick Start
    train_loader, val_loader = create_dataloader(
        base_path="path/to/data",
        split_ratio=0.8
    )

    # Custom Config
    config = get_default_config()
    config['window_size'] = 4096
    dataset = create_dataset(base_path="path/to/data", **config)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from data.dual_stream_dataset import DualStreamDataset


# ==================== Default Configurations ====================

DEFAULT_LPS_CONFIG = {
    'nperseg': 32768,
    'noverlap': 31130,
    'min_freq': 5.0,
    'max_freq': 50.0,
    'c1': 0.8,
    'c2': 1.2,
    'adaptive_range': True,
    'lambda_smooth': 0.5,
    'use_interpolation': False,
    'bidirectional': True,
    'strategy': 'renyi_v1',
    'strategy_params': {'alpha': 3, 'w1': 1.0, 'w2': 0.5},
}

DEFAULT_IF_SMOOTH_CONFIG = {
    'n_samples': 2000000,
    'pad_size': 20,
    'smooth_factor': 0.5,
    't_start': 0,
    't_end': 10
}

DEFAULT_DATASET_CONFIG = {
    'fs': 200000,
    'signal_length': 2000000,
    'file_ext': 'auto',
    'signal_col': 1,
    'window_size': 3072,
    'hop_size': None,  # None means same as window_size (no overlap)
    'downsample_factor': 10,
    'order_spec_length': 1537,
    'precompute_order_spec': True,
    'folder_indices': [0, 1, 2, 3, 4],  # 包含全部5个类别
    'verbose': True,
}

DEFAULT_DATALOADER_CONFIG = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 0,
    'pin_memory': True,
    'drop_last': False,
}


def get_default_config() -> Dict:
    """
    Get all default configurations.

    Returns:
        Dict containing all default configs
    """
    return {
        'lps_config': DEFAULT_LPS_CONFIG.copy(),
        'if_smooth_config': DEFAULT_IF_SMOOTH_CONFIG.copy(),
        'dataset_config': DEFAULT_DATASET_CONFIG.copy(),
        'dataloader_config': DEFAULT_DATALOADER_CONFIG.copy(),
    }


def create_dataset(
    base_path: str,
    lps_config: Optional[Dict] = None,
    if_smooth_config: Optional[Dict] = None,
    fs: int = 200000,
    signal_length: int = 2000000,
    file_ext: str = 'auto',
    signal_col: int = 1,
    window_size: int = 3072,
    hop_size: Optional[int] = None,
    downsample_factor: int = 10,
    order_spec_length: int = 1537,
    precompute_order_spec: bool = True,
    folder_indices: Optional[List[int]] = None,
    verbose: bool = True,
) -> DualStreamDataset:
    """
    Create a DualStreamDataset instance.

    Args:
        base_path: Root directory of the dataset
        lps_config: LPS algorithm parameters (optional)
        if_smooth_config: IF smoothing parameters (optional)
        fs: Sampling frequency (Hz)
        signal_length: Signal length per file
        file_ext: File extension ('auto' | 'mat' | 'csv')
        signal_col: CSV signal column index
        window_size: Slice window size
        hop_size: Slice step size (None = window_size)
        downsample_factor: Downsampling factor
        order_spec_length: Order spectrum output length
        precompute_order_spec: Whether to precompute order spectrum during build
        folder_indices: List of folder indices to process
        verbose: Print progress info

    Returns:
        Configured DualStreamDataset instance
    """
    # Use default configs if not provided
    if lps_config is None:
        lps_config = DEFAULT_LPS_CONFIG.copy()
    if if_smooth_config is None:
        if_smooth_config = DEFAULT_IF_SMOOTH_CONFIG.copy()
    if folder_indices is None:
        folder_indices = [0, 1, 2, 3, 4]  # 包含全部5个类别

    # Create dataset
    dataset = DualStreamDataset(
        base_path,
        fs=fs,
        signal_length=signal_length,
        file_ext=file_ext,
        signal_col=signal_col,
    )

    # Build dataset
    dataset.build_dataset(
        lps_config=lps_config,
        if_smooth_config=if_smooth_config,
        window_size=window_size,
        hop_size=hop_size,
        downsample_factor=downsample_factor,
        order_spec_length=order_spec_length,
        precompute_order_spec=precompute_order_spec,
        folder_indices=folder_indices,
        verbose=verbose,
    )

    return dataset


def create_dataloader(
    base_path: str,
    split_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    lps_config: Optional[Dict] = None,
    if_smooth_config: Optional[Dict] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        base_path: Root directory of the dataset
        split_ratio: Train/Val split ratio (default 0.8)
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU
        drop_last: Whether to drop last incomplete batch
        lps_config: LPS algorithm parameters
        if_smooth_config: IF smoothing parameters
        **dataset_kwargs: Additional arguments for create_dataset

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    dataset = create_dataset(
        base_path=base_path,
        lps_config=lps_config,
        if_smooth_config=if_smooth_config,
        **dataset_kwargs
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


class DualStreamDatasetBuilder:
    """
    Builder class for creating DualStreamDataset with fluent API.

    Example:
        dataset = (DualStreamDatasetBuilder("path/to/data")
            .set_sampling(fs=200000, downsample_factor=10)
            .set_window(size=4096, hop=2048)
            .set_folders([0, 1, 2, 3])
            .set_lps_config(nperseg=131072)
            .build())
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self._lps_config = DEFAULT_LPS_CONFIG.copy()
        self._if_smooth_config = DEFAULT_IF_SMOOTH_CONFIG.copy()
        self._fs = 200000
        self._signal_length = 2000000
        self._window_size = 3072
        self._hop_size = None
        self._downsample_factor = 10
        self._order_spec_length = 1537
        self._folder_indices = [0, 1, 2, 3]
        self._verbose = True

    def set_sampling(self, fs: int = None, downsample_factor: int = None) -> 'DualStreamDatasetBuilder':
        """Set sampling parameters."""
        if fs is not None:
            self._fs = fs
        if downsample_factor is not None:
            self._downsample_factor = downsample_factor
        return self

    def set_window(self, size: int = None, hop: int = None) -> 'DualStreamDatasetBuilder':
        """Set window parameters."""
        if size is not None:
            self._window_size = size
        if hop is not None:
            self._hop_size = hop
        return self

    def set_folders(self, indices: List[int]) -> 'DualStreamDatasetBuilder':
        """Set folder indices to process."""
        self._folder_indices = indices
        return self

    def set_lps_config(self, **kwargs) -> 'DualStreamDatasetBuilder':
        """Update LPS configuration."""
        self._lps_config.update(kwargs)
        return self

    def set_if_smooth_config(self, **kwargs) -> 'DualStreamDatasetBuilder':
        """Update IF smooth configuration."""
        self._if_smooth_config.update(kwargs)
        return self

    def set_order_spec_length(self, length: int) -> 'DualStreamDatasetBuilder':
        """Set order spectrum output length."""
        self._order_spec_length = length
        return self

    def set_verbose(self, verbose: bool) -> 'DualStreamDatasetBuilder':
        """Set verbose mode."""
        self._verbose = verbose
        return self

    def build(self) -> DualStreamDataset:
        """Build and return the dataset."""
        return create_dataset(
            base_path=self.base_path,
            lps_config=self._lps_config,
            if_smooth_config=self._if_smooth_config,
            fs=self._fs,
            signal_length=self._signal_length,
            window_size=self._window_size,
            hop_size=self._hop_size,
            downsample_factor=self._downsample_factor,
            order_spec_length=self._order_spec_length,
            folder_indices=self._folder_indices,
            verbose=self._verbose,
        )

    def build_loaders(
        self,
        split_ratio: float = 0.8,
        batch_size: int = 32,
        **loader_kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Build dataset and return train/val DataLoaders."""
        dataset = self.build()

        total_size = len(dataset)
        train_size = int(total_size * split_ratio)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs
        )

        return train_loader, val_loader


# ==================== Utility Functions ====================

def get_class_names() -> Dict[int, str]:
    """Get fault type class names mapping."""
    return DualStreamDataset.FAULT_TYPES.copy()


def get_num_classes() -> int:
    """Get number of fault classes."""
    return len(DualStreamDataset.FAULT_TYPES)


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_lps_keys = ['nperseg', 'min_freq', 'max_freq', 'strategy']
    required_if_keys = ['n_samples', 'pad_size', 'smooth_factor', 't_start', 't_end']

    if 'lps_config' in config:
        for key in required_lps_keys:
            if key not in config['lps_config']:
                raise ValueError(f"Missing required LPS config key: {key}")

    if 'if_smooth_config' in config:
        for key in required_if_keys:
            if key not in config['if_smooth_config']:
                raise ValueError(f"Missing required IF smooth config key: {key}")

    return True


# ==================== Module Exports ====================

__all__ = [
    'create_dataset',
    'create_dataloader',
    'get_default_config',
    'get_class_names',
    'get_num_classes',
    'validate_config',
    'DualStreamDatasetBuilder',
    'DEFAULT_LPS_CONFIG',
    'DEFAULT_IF_SMOOTH_CONFIG',
    'DEFAULT_DATASET_CONFIG',
    'DEFAULT_DATALOADER_CONFIG',
]