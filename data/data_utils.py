# -*- coding: utf-8 -*-
"""
数据工具模块
通用的文件读取函数
"""

import scipy.io as scio


def load_channel1_data(mat_path):
    """
    从.mat文件中加载Channel_1数据

    参数:
        mat_path: .mat文件路径

    返回:
        channel1_data: Channel_1的振动数据（1D numpy array）
    """
    try:
        data = scio.loadmat(mat_path)

        possible_keys = ['Channel_1', 'channel1', 'Channel1', 'ch1', 'Ch1', 'channel_1']

        for key in possible_keys:
            if key in data:
                channel1_data = data[key].squeeze()
                return channel1_data

        raise KeyError(f"在文件 {mat_path} 中未找到Channel_1数据")

    except Exception as e:
        print(f"加载文件 {mat_path} 时出错: {str(e)}")
        raise


def load_channel2_data(mat_path):
    """
    从.mat文件中加载Channel_2数据（编码器转速数据）

    参数:
        mat_path: .mat文件路径

    返回:
        channel2_data: Channel_2的编码器数据（1D numpy array）
    """
    try:
        data = scio.loadmat(mat_path)

        possible_keys = ['Channel_2', 'channel2', 'Channel2', 'ch2', 'Ch2', 'channel_2']

        for key in possible_keys:
            if key in data:
                channel2_data = data[key].squeeze()
                return channel2_data

        raise KeyError(f"在文件 {mat_path} 中未找到Channel_2数据")

    except Exception as e:
        print(f"加载文件 {mat_path} 时出错: {str(e)}")
        raise

