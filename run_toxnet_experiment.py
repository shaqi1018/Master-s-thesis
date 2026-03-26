# -*- coding: utf-8 -*-
"""
TO-XNet 实验运行脚本（精简版）
仅保留：LPS + Renyi 熵 + 双流双向交叉注意力 TO-XNet

使用方法:
    python run_toxnet_experiment.py
    python run_toxnet_experiment.py --data_path "path/to/UO_Bearing"
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


DEFAULT_DATA_PATH = r'D:/Variable_speed/University_of_Ottawa/Original_data/UO_Bearing'


def print_experiment_info(args):
    """打印实验信息"""
    print("=" * 60)
    print("TO-XNet 实验配置（精简版）")
    print("=" * 60)
    print("数据集: University of Ottawa Bearing Dataset")
    print(f"数据路径: {args.data_path}")
    print("采样频率: 200000 Hz")
    print("类别数: 5")
    print("类别名: ['Healthy', 'Inner', 'Outer', 'Ball', 'Combination']")
    print("窗口大小: 3072")
    print("降采样因子: 10")
    print("LPS策略: renyi_v1")
    print("交互方式: bidirectional cross-attention")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='TO-XNet Focused Experiment')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径 (不指定则使用默认路径)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--lite', action='store_true', help='使用轻量版TOXNetLite')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式')
    
    args = parser.parse_args()

    # 设置数据路径
    if args.data_path is None:
        args.data_path = DEFAULT_DATA_PATH

    # 打印实验信息
    print_experiment_info(args)
    
    # 导入训练模块并运行
    from train_toxnet import main as train_main
    
    # 构造训练参数
    class TrainArgs:
        pass
    
    train_args = TrainArgs()
    train_args.data_path = args.data_path
    train_args.save_dir = os.path.join(args.save_dir, 'ottawa')
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.lr = args.lr
    train_args.num_classes = 5
    train_args.window_size = 3072
    train_args.downsample_factor = 10
    train_args.lite = args.lite
    
    # 运行训练
    train_main(train_args)


if __name__ == '__main__':
    main()

