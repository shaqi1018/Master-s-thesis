# -*- coding: utf-8 -*-
"""
TO-XNet: Time-Order Cross-Domain Interaction Network
时域-阶次跨域交互网络

创新点:
    1. 双向交叉注意力 (Bidirectional Cross-Attention)
    2. 多尺度时域特征提取 (Multi-Scale Temporal Feature)
    3. 阶次感知位置编码 (Order-Aware Positional Encoding)
    4. 自适应门控融合 (Adaptive Gated Fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock1D(nn.Module):
    """1D残差块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MultiScaleConv1D(nn.Module):
    """多尺度卷积模块 - 创新点2: 捕获不同时间尺度的冲击特征"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class OrderPositionalEncoding(nn.Module):
    """阶次位置编码 - 创新点3: 让网络感知阶次位置"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BidirectionalCrossAttention(nn.Module):
    """双向交叉注意力 - 创新点1: 时域与阶次双向信息交互"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_time = nn.Linear(dim, dim)
        self.kv_order = nn.Linear(dim, dim * 2)
        self.q_order = nn.Linear(dim, dim)
        self.kv_time = nn.Linear(dim, dim * 2)
        self.proj_time = nn.Linear(dim, dim)
        self.proj_order = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f_time, f_order):
        B, L_t, C = f_time.shape
        _, L_o, _ = f_order.shape
        # Time queries Order
        q_t = self.q_time(f_time).view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)
        kv_o = self.kv_order(f_order).view(B, L_o, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_o, v_o = kv_o[0], kv_o[1]
        attn_t2o = F.softmax((q_t @ k_o.transpose(-2, -1)) * self.scale, dim=-1)
        out_t = self.proj_time((self.dropout(attn_t2o) @ v_o).transpose(1, 2).reshape(B, L_t, C))
        # Order queries Time
        q_o = self.q_order(f_order).view(B, L_o, self.num_heads, self.head_dim).transpose(1, 2)
        kv_t = self.kv_time(f_time).view(B, L_t, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_t, v_t = kv_t[0], kv_t[1]
        attn_o2t = F.softmax((q_o @ k_t.transpose(-2, -1)) * self.scale, dim=-1)
        out_o = self.proj_order((self.dropout(attn_o2t) @ v_t).transpose(1, 2).reshape(B, L_o, C))
        return out_t, out_o


class AdaptiveGatedFusion(nn.Module):
    """自适应门控融合 - 创新点4: 学习最优融合权重"""
    def __init__(self, dim):
        super().__init__()
        self.gate_time = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_order = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.fuse_proj = nn.Linear(dim * 2, dim)

    def forward(self, f_t, f_o):
        concat = torch.cat([f_t, f_o], dim=-1)
        return self.fuse_proj(torch.cat([self.gate_time(concat) * f_t,
                                          self.gate_order(concat) * f_o], dim=-1))


class TimeStreamModel(nn.Module):
    """时域流模型 - ResNet-1D + 多尺度 + SE注意力"""
    def __init__(self, use_multiscale=True):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.head = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=32, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        if use_multiscale:
            self.multiscale = MultiScaleConv1D(16, 32)
            self.layer1 = ResBlock1D(32, 32, stride=2)
        else:
            self.layer1 = ResBlock1D(16, 32, stride=2)
        self.layer2 = ResBlock1D(32, 64, stride=2)
        self.layer3 = ResBlock1D(64, 64, stride=2)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                 nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.Sigmoid())

    def forward(self, x):
        x = self.head(x)
        if self.use_multiscale:
            x = self.multiscale(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # SE注意力加权
        se_weight = self.se(x).unsqueeze(-1)
        return x * se_weight


class OrderStreamModel(nn.Module):
    """阶次流模型 - 浅层宽核CNN + 位置编码"""
    def __init__(self, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU())
        if use_pos_encoding:
            self.pos_encoding = OrderPositionalEncoding(d_model=64, max_len=200)

    def forward(self, x):
        x = self.features(x)
        if self.use_pos_encoding:
            x = x.permute(0, 2, 1)
            x = self.pos_encoding(x)
            x = x.permute(0, 2, 1)
        return x


class TOXNet(nn.Module):
    """
    TO-XNet: Time-Order Cross-Domain Interaction Network

    创新点总结:
        1. 双向交叉注意力: 时域与阶次信息双向流动
        2. 多尺度时域特征: 捕获不同尺度的冲击特征
        3. 阶次位置编码: 让网络感知故障阶次位置
        4. 自适应门控融合: 学习最优融合权重
    """
    def __init__(self, num_classes=4, dim=64, num_heads=4,
                 use_multiscale=True, use_pos_encoding=True, use_bidirectional=True):
        super().__init__()
        self.use_bidirectional = use_bidirectional

        # 双流特征提取
        self.stream_time = TimeStreamModel(use_multiscale=use_multiscale)
        self.stream_order = OrderStreamModel(use_pos_encoding=use_pos_encoding)

        # 交叉注意力
        if use_bidirectional:
            self.cross_attn = BidirectionalCrossAttention(dim=dim, num_heads=num_heads)
        else:
            # 单向: 仅时域查询阶次
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 自适应门控融合
        self.gated_fusion = AdaptiveGatedFusion(dim)

        # 全局池化 + 分类器
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, x_time, x_order):
        # 特征提取
        f_t = self.stream_time(x_time)   # (B, 64, L1)
        f_o = self.stream_order(x_order)  # (B, 64, L2)

        # 维度变换: (B, C, L) -> (B, L, C)
        f_t = f_t.permute(0, 2, 1)
        f_o = f_o.permute(0, 2, 1)

        # 交叉注意力融合
        if self.use_bidirectional:
            attn_t, attn_o = self.cross_attn(f_t, f_o)
            f_t = f_t + attn_t  # 残差连接
            f_o = f_o + attn_o
        else:
            attn_out, _ = self.cross_attn(f_t, f_o, f_o)
            f_t = f_t + attn_out

        # 全局池化: (B, L, C) -> (B, C)
        f_t_global = f_t.mean(dim=1)
        f_o_global = f_o.mean(dim=1)

        # 门控融合
        fused = self.gated_fusion(f_t_global, f_o_global)

        # 分类
        return self.classifier(fused)

    def extract_features(self, x_time, x_order):
        """
        提取融合后的特征（用于t-SNE可视化等）
        """
        f_t = self.stream_time(x_time)
        f_o = self.stream_order(x_order)
        f_t = f_t.permute(0, 2, 1)
        f_o = f_o.permute(0, 2, 1)
        if self.use_bidirectional:
            attn_t, attn_o = self.cross_attn(f_t, f_o)
            f_t = f_t + attn_t
            f_o = f_o + attn_o
        else:
            attn_out, _ = self.cross_attn(f_t, f_o, f_o)
            f_t = f_t + attn_out
        f_t_global = f_t.mean(dim=1)
        f_o_global = f_o.mean(dim=1)
        fused = self.gated_fusion(f_t_global, f_o_global)
        return fused

    def extract_multi_stage_features(self, x_time, x_order):
        """
        提取各阶段的特征（用于多阶段t-SNE可视化）

        返回字典包含:
            - 'raw': 原始输入拼接 (时域+阶次)
            - 'before_cross_attn': 交叉注意力前的特征
            - 'before_gated_fusion': 门控融合前的特征
            - 'final': 分类器前的最终特征
        """
        features = {}

        # 1. 原始数据特征 (展平后拼接)
        raw_time = x_time.view(x_time.size(0), -1)  # (B, L)
        raw_order = x_order.view(x_order.size(0), -1)  # (B, L)
        features['raw'] = torch.cat([raw_time, raw_order], dim=1)  # (B, 2L)

        # 2. 双流特征提取后 (交叉注意力前)
        f_t = self.stream_time(x_time)   # (B, 64, L1)
        f_o = self.stream_order(x_order)  # (B, 64, L2)
        # 全局池化后拼接
        f_t_pool = f_t.mean(dim=2)  # (B, 64)
        f_o_pool = f_o.mean(dim=2)  # (B, 64)
        features['before_cross_attn'] = torch.cat([f_t_pool, f_o_pool], dim=1)  # (B, 128)

        # 3. 交叉注意力后 (门控融合前)
        f_t = f_t.permute(0, 2, 1)  # (B, L1, 64)
        f_o = f_o.permute(0, 2, 1)  # (B, L2, 64)
        if self.use_bidirectional:
            attn_t, attn_o = self.cross_attn(f_t, f_o)
            f_t = f_t + attn_t
            f_o = f_o + attn_o
        else:
            attn_out, _ = self.cross_attn(f_t, f_o, f_o)
            f_t = f_t + attn_out
        f_t_global = f_t.mean(dim=1)  # (B, 64)
        f_o_global = f_o.mean(dim=1)  # (B, 64)
        features['before_gated_fusion'] = torch.cat([f_t_global, f_o_global], dim=1)  # (B, 128)

        # 4. 门控融合后 (分类器前)
        fused = self.gated_fusion(f_t_global, f_o_global)  # (B, 64)
        features['final'] = fused

        return features


# ==================== 简化版本 (消融实验用) ====================

class TOXNetLite(nn.Module):
    """轻量版TO-XNet - 用于消融实验和快速验证"""
    def __init__(self, num_classes=4, dim=64):
        super().__init__()
        self.stream_time = TimeStreamModel(use_multiscale=False)
        self.stream_order = OrderStreamModel(use_pos_encoding=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(dim, num_classes))

    def forward(self, x_time, x_order):
        f_t = self.stream_time(x_time).permute(0, 2, 1)
        f_o = self.stream_order(x_order).permute(0, 2, 1)
        attn_out, _ = self.cross_attn(f_t, f_o, f_o)
        fused = (f_t + attn_out).permute(0, 2, 1)
        global_feat = self.gap(fused).squeeze(-1)
        return self.classifier(global_feat)


# ==================== 模块导出 ====================
__all__ = [
    'TOXNet', 'TOXNetLite',
    'TimeStreamModel', 'OrderStreamModel',
    'BidirectionalCrossAttention', 'AdaptiveGatedFusion',
    'MultiScaleConv1D', 'OrderPositionalEncoding', 'ResBlock1D'
]
