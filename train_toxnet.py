# -*- coding: utf-8 -*-
"""
TO-XNet Training Script
训练脚本

Usage:
    python train_toxnet.py --data_path "path/to/data" --epochs 100
"""

import os
# 解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from models.toxnet import TOXNet, TOXNetLite
from data.dual_stream_api import create_dataset, get_default_config


def stratified_group_split(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    按文件分组进行分层划分（避免同一原始文件切片泄漏到不同集合）

    参数:
        dataset: 完整数据集
        train_ratio: 训练集比例 (默认0.6)
        val_ratio: 验证集比例 (默认0.2)
        test_ratio: 测试集比例 (默认0.2)
        random_state: 随机种子

    返回:
        train_indices, val_indices, test_indices
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    all_groups = dataset.all_groups
    all_labels = dataset.all_labels

    # 提取文件级样本（每个group一个标签）
    unique_groups = np.unique(all_groups)
    group_labels = np.array([all_labels[np.where(all_groups == g)[0][0]] for g in unique_groups])

    # 第一次划分：训练文件组 vs (验证+测试 文件组)
    try:
        train_groups, temp_groups, _, temp_group_labels = train_test_split(
            unique_groups, group_labels,
            test_size=(val_ratio + test_ratio),
            stratify=group_labels,
            random_state=random_state
        )
    except ValueError:
        # 某些类别文件数过少时回退为非分层
        train_groups, temp_groups, _, temp_group_labels = train_test_split(
            unique_groups, group_labels,
            test_size=(val_ratio + test_ratio),
            stratify=None,
            random_state=random_state
        )

    # 第二次划分：验证文件组 vs 测试文件组
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    try:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=val_test_ratio,
            stratify=temp_group_labels,
            random_state=random_state
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=val_test_ratio,
            stratify=None,
            random_state=random_state
        )

    # 映射回切片级索引
    all_indices = np.arange(len(dataset))
    train_indices = all_indices[np.isin(all_groups, train_groups)]
    val_indices = all_indices[np.isin(all_groups, val_groups)]
    test_indices = all_indices[np.isin(all_groups, test_groups)]

    return train_indices, val_indices, test_indices


def print_split_distribution(dataset, train_indices, val_indices, test_indices, class_names):
    """打印数据集划分的类别分布"""
    all_labels = dataset.all_labels

    print("\n" + "="*60)
    print("数据集划分统计 (分层采样，类别分布近似)")
    print("="*60)

    # 统计各集合的类别分布
    train_labels = all_labels[train_indices]
    val_labels = all_labels[val_indices]
    test_labels = all_labels[test_indices]

    print(f"\n{'类别':<15} {'训练集':>10} {'验证集':>10} {'测试集':>10} {'总计':>10}")
    print("-"*60)

    for class_idx, class_name in enumerate(class_names):
        train_count = np.sum(train_labels == class_idx)
        val_count = np.sum(val_labels == class_idx)
        test_count = np.sum(test_labels == class_idx)
        total_count = train_count + val_count + test_count
        print(f"{class_name:<15} {train_count:>10} {val_count:>10} {test_count:>10} {total_count:>10}")

    print("-"*60)
    print(f"{'总计':<15} {len(train_indices):>10} {len(val_indices):>10} {len(test_indices):>10} {len(all_labels):>10}")
    print(f"{'比例':<15} {len(train_indices)/len(all_labels)*100:>9.1f}% {len(val_indices)/len(all_labels)*100:>9.1f}% {len(test_indices)/len(all_labels)*100:>9.1f}%")
    print("="*60)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (time_signal, order_spec, labels) in enumerate(dataloader):
        time_signal = time_signal.to(device, non_blocking=True)
        order_spec = order_spec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(time_signal, order_spec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for time_signal, order_spec, labels in dataloader:
            time_signal = time_signal.to(device, non_blocking=True)
            order_spec = order_spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(time_signal, order_spec)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get default config
    config = get_default_config()

    # Create dataset
    print("\n" + "="*50)
    print("Loading and processing dataset...")
    print("="*50)

    dataset = create_dataset(
        base_path=args.data_path,
        lps_config=config['lps_config'],
        if_smooth_config=config['if_smooth_config'],
        window_size=args.window_size,
        downsample_factor=args.downsample_factor,
        precompute_order_spec=not args.online_order_spec,
        folder_indices=[0, 1, 2, 3, 4],  # 包含全部5个类别
        verbose=True
    )

    # 分层划分数据集 (6:2:2)
    class_names = ['Healthy', 'Inner', 'Outer', 'Ball', 'Combination']

    # 按文件分组分层划分 (6:2:2)
    train_indices, val_indices, test_indices = stratified_group_split(
        dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42
    )

    # 输出文件组统计，确认无泄漏
    train_groups = np.unique(dataset.all_groups[train_indices])
    val_groups = np.unique(dataset.all_groups[val_indices])
    test_groups = np.unique(dataset.all_groups[test_indices])
    print(f"\n文件组划分: 训练={len(train_groups)}, 验证={len(val_groups)}, 测试={len(test_groups)}")

    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 打印数据集划分统计
    print_split_distribution(dataset, train_indices, val_indices, test_indices, class_names)

    # Create DataLoaders
    pin_memory = (device.type == 'cuda')
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    eval_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = True
        eval_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = args.prefetch_factor
        eval_loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    # Create model
    print("\n" + "="*50)
    print("Creating TO-XNet model...")
    print("="*50)

    if args.lite:
        model = TOXNetLite(num_classes=args.num_classes).to(device)
        print("Using TOXNetLite (ablation version)")
    else:
        model = TOXNet(
            num_classes=args.num_classes,
            dim=64,
            num_heads=4,
            use_multiscale=True,
            use_pos_encoding=True,
            use_bidirectional=True
        ).to(device)
        print("Using TOXNet (full version)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 30)

        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - start_time

        # Log
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_toxnet.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"*** Best model saved (Acc: {val_acc:.2f}%) ***")

    print("\n" + "="*50)
    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%")
    print("="*50)

    # ==================== 在测试集上进行最终评估 ====================
    print("\n" + "="*50)
    print("在测试集上进行最终评估...")
    print("="*50)

    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, 'best_toxnet.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")

    # 在测试集上评估
    test_loss, test_acc, all_preds, all_labels_test = evaluate(model, test_loader, criterion, device)
    print(f"\n*** 测试集结果 ***")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # ==================== 生成图表 ====================
    print("\n生成训练结果图表...")

    # 类别名称
    class_names = ['Healthy', 'Inner', 'Outer', 'Ball', 'Combination']
    
    # 创建图表保存目录
    fig_dir = os.path.join(args.save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========== 图1: 损失曲线 ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_dir, 'loss_curve.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'loss_curve.png')}")
    
    # ========== 图2: 准确率曲线 ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs_range, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_range, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, 'accuracy_curve.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'accuracy_curve.png')}")
    
    # ========== 图3: 混淆矩阵 (测试集) ==========
    cm = confusion_matrix(all_labels_test, all_preds)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:args.num_classes],
                yticklabels=class_names[:args.num_classes],
                ax=ax3, annot_kws={'size': 14})
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_title('Confusion Matrix (Test Set)', fontsize=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_dir, 'confusion_matrix.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'confusion_matrix.png')}")
    
    # ========== 图4: 归一化混淆矩阵 ==========
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names[:args.num_classes],
                yticklabels=class_names[:args.num_classes],
                ax=ax4, annot_kws={'size': 12})
    ax4.set_xlabel('Predicted Label', fontsize=12)
    ax4.set_ylabel('True Label', fontsize=12)
    ax4.set_title('Normalized Confusion Matrix', fontsize=14)
    fig4.tight_layout()
    fig4.savefig(os.path.join(fig_dir, 'confusion_matrix_normalized.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'confusion_matrix_normalized.png')}")
    
    # ========== 图5: 综合图 (2x2布局) ==========
    fig5, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 子图1: 损失曲线
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 准确率曲线
    axes[0, 1].plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names[:args.num_classes],
                yticklabels=class_names[:args.num_classes],
                ax=axes[1, 0], annot_kws={'size': 10})
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_title('Confusion Matrix')
    
    # 子图4: 每类准确率柱状图
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(per_class_acc)))
    bars = axes[1, 1].bar(class_names[:args.num_classes], per_class_acc, color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Per-Class Accuracy')
    axes[1, 1].set_ylim([0, 105])
    # 添加数值标签
    for bar, acc in zip(bars, per_class_acc):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    fig5.suptitle(f'TO-XNet Training Results (Test Acc: {test_acc:.2f}%)', fontsize=16)
    fig5.tight_layout()
    fig5.savefig(os.path.join(fig_dir, 'training_summary.png'), dpi=150)
    print(f"  保存: {os.path.join(fig_dir, 'training_summary.png')}")

    # ========== 保存分类报告 (测试集) ==========
    report = classification_report(all_labels_test, all_preds,
                                   target_names=class_names[:args.num_classes],
                                   digits=4)
    report_path = os.path.join(fig_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TO-XNet Classification Report (Test Set)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")
        f.write(report)
    print(f"  保存: {report_path}")

    # 打印分类报告
    print("\n" + "="*50)
    print("Classification Report (Test Set):")
    print("="*50)
    print(report)

    # ========== 图6: 多阶段 t-SNE 特征可视化 (测试集) ==========
    print("\n生成多阶段 t-SNE 特征可视化 (测试集)...")
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    # 提取各阶段特征
    model.eval()
    stage_features = {'raw': [], 'before_cross_attn': [], 'before_gated_fusion': [], 'final': []}
    labels_list = []

    with torch.no_grad():
        for time_signal, order_spec, labels in test_loader:
            time_signal = time_signal.to(device, non_blocking=True)
            order_spec = order_spec.to(device, non_blocking=True)
            # 提取多阶段特征
            features_dict = model.extract_multi_stage_features(time_signal, order_spec)
            for key in stage_features:
                stage_features[key].append(features_dict[key].cpu().numpy())
            labels_list.extend(labels.numpy())

    # 合并特征
    for key in stage_features:
        stage_features[key] = np.vstack(stage_features[key])
    labels_array = np.array(labels_list)
    print(f"  样本数量: {len(labels_array)}")

    # 随机采样（t-SNE计算量大）
    max_samples = 800
    if len(labels_array) > max_samples:
        print(f"  随机采样 {max_samples} 个样本...")
        np.random.seed(42)
        indices = np.random.choice(len(labels_array), max_samples, replace=False)
        for key in stage_features:
            stage_features[key] = stage_features[key][indices]
        labels_array = labels_array[indices]

    # 阶段名称和标题
    stage_names = ['raw', 'before_cross_attn', 'before_gated_fusion', 'final']
    stage_titles = [
        'Raw Input Data',
        'Before Cross-Attention',
        'Before Gated Fusion',
        'Final Features (Before Classifier)'
    ]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']

    # 创建2x2子图
    fig_tsne, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (stage_name, title) in enumerate(zip(stage_names, stage_titles)):
        print(f"  执行 t-SNE: {title}...")
        feat = stage_features[stage_name]
        # 对原始数据进行PCA降维（维度太高）
        if stage_name == 'raw' and feat.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=100, random_state=42)
            feat = pca.fit_transform(feat)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800, verbose=0)
        feat_2d = tsne.fit_transform(feat)
        silhouette = silhouette_score(feat_2d, labels_array)

        ax = axes[idx]
        for class_idx in range(args.num_classes):
            mask = labels_array == class_idx
            ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1], c=colors[class_idx],
                      label=class_names[class_idx], alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('t-SNE Dim 1', fontsize=11)
        ax.set_ylabel('t-SNE Dim 2', fontsize=11)
        ax.set_title(f'{title}\nSilhouette: {silhouette:.3f}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    fig_tsne.suptitle(f'TO-XNet Multi-Stage Feature Visualization (Val Acc: {best_val_acc:.2f}%)',
                      fontsize=14, fontweight='bold')
    fig_tsne.tight_layout(rect=[0, 0, 1, 0.96])
    tsne_path = os.path.join(fig_dir, 'tsne_multi_stage.png')
    fig_tsne.savefig(tsne_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 多阶段t-SNE图已保存: {tsne_path}")

    # 显示所有图表
    plt.show()

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TO-XNet')
    parser.add_argument('--data_path', type=str, 
                        default=r'D:/Variable_speed/University_of_Ottawa/Original_data/UO_Bearing',
                        help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--window_size', type=int, default=3072, help='Window size')
    parser.add_argument('--downsample_factor', type=int, default=10, help='Downsample factor')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker count')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch batches per worker')
    parser.add_argument('--online_order_spec', action='store_true',
                        help='在线计算阶次谱（默认关闭，建议使用预计算缓存）')
    parser.add_argument('--lite', action='store_true', help='Use TOXNetLite')

    args = parser.parse_args()
    main(args)

