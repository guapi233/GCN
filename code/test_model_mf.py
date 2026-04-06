"""
测试矩阵分解模型
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import world
import torch
from model import PureMF, BasicModel
from dataloader import LastFM
import numpy as np


def test_pure_mf():
    """测试矩阵分解模型"""
    print("=" * 60)
    print("测试矩阵分解模型（PureMF）")
    print("=" * 60)

    # 加载 LastFM 数据集
    print("\n[1] 加载数据集...")
    dataset = LastFM()
    print(f"✓ 数据集: {dataset.n_users} 用户, {dataset.m_items} 物品")
    print(f"✓ 训练集: {dataset.trainDataSize} 交互")

    # 创建模型配置
    print("\n[2] 创建模型配置...")
    config = {
        'latent_dim_rec': 64,  # 嵌入维度
    }

    # 创建模型
    print("\n[3] 创建模型...")
    model = PureMF(config, dataset)
    print(f"✓ 模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 测试评分预测
    print("\n[4] 测试评分预测...")
    test_users = torch.tensor([0, 1, 2])
    ratings = model.getUsersRating(test_users)
    print(f"✓ 输入用户: {test_users}")
    print(f"✓ 评分矩阵形状: {ratings.shape}")
    print(f"✓ 评分范围: [{ratings.min():.4f}, {ratings.max():.4f}]")

    # 测试 BPR 损失
    print("\n[5] 测试 BPR 损失...")
    users = torch.tensor([0, 1, 2])
    pos_items = torch.tensor([10, 20, 30])
    neg_items = torch.tensor([100, 200, 300])

    loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
    print(f"✓ BPR 损失: {loss:.4f}")
    print(f"✓ L2 正则化: {reg_loss:.4f}")
    print(f"✓ 总损失: {loss + reg_loss:.4f}")

    # 测试前向传播
    print("\n[6] 测试前向传播...")
    scores = model.forward(users, pos_items)
    print(f"✓ 输出得分: {scores}")

    print("\n" + "=" * 60)
    print("测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_pure_mf()