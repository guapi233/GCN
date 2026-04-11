"""
测试 LastFM 数据加载器
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataloader import LastFM
import torch
import numpy as np


def test_lastfm_loader():
    """测试 LastFM 数据加载器"""
    print("=" * 60)
    print("测试 LastFM 数据加载器")
    print("=" * 60)

    # 检查数据集是否存在
    data_path = "../data/lastfm"
    if not os.path.exists(data_path):
        print(f"\n✗ 数据集不存在: {data_path}")
        print("  请先下载 LastFM 数据集并放在 data/lastfm/ 目录下")
        return

    # 初始化数据集
    print("\n[1] 初始化数据集...")
    dataset = LastFM(path=data_path)
    print("✓ 数据集初始化成功")

    # 测试基本属性
    print("\n[2] 测试基本属性...")
    print(f"✓ 用户数: {dataset.n_users}")
    print(f"✓ 物品数: {dataset.m_items}")
    print(f"✓ 训练集大小: {dataset.trainDataSize}")
    print(f"✓ 训练集唯一用户数: {len(dataset.trainUniqueUsers)}")
    print(f"✓ 测试集唯一用户数: {len(dataset.testUniqueUsers)}")

    # 测试社交网络
    print("\n[3] 测试社交网络...")
    print(f"✓ 社交关系数: {len(dataset.trustNet)}")
    print(f"✓ 社交网络形状: {dataset.socialNet.shape}")
    print(f"  前5条社交关系:")
    for i in range(min(5, len(dataset.trustNet))):
        print(f"    {dataset.trustNet[i][0]} → {dataset.trustNet[i][1]}")

    # 测试交互矩阵
    print("\n[4] 测试交互矩阵...")
    print(f"✓ UserItemNet 形状: {dataset.UserItemNet.shape}")
    print(f"✓ UserItemNet 非零元素数: {dataset.UserItemNet.nnz}")
    print(f"  平均每个用户交互数: {dataset.UserItemNet.nnz / dataset.n_users:.2f}")

    # 测试正样本
    print("\n[5] 测试正样本查询...")
    test_users = [0, 1, 2]
    pos_items = dataset.getUserPosItems(test_users)
    print(f"✓ 查询用户 {test_users} 的正样本:")
    for i, user in enumerate(test_users):
        print(f"    用户 {user}: {list(pos_items[i][:5])}... (共{len(pos_items[i])}个)")

    # 测试负样本
    print("\n[6] 测试负样本查询...")
    neg_items = dataset.getUserNegItems(test_users)
    print(f"✓ 查询用户 {test_users} 的负样本:")
    for i, user in enumerate(test_users):
        print(f"    用户 {user}: 共{len(neg_items[i])}个负样本")

    # 测试交互反馈
    print("\n[7] 测试交互反馈查询...")
    users = [0, 1, 2]
    items = [10, 20, 30]
    feedback = dataset.getUserItemFeedback(users, items)
    print(f"✓ 用户-物品交互反馈:")
    for i in range(len(users)):
        print(f"    用户 {users[i]} - 物品 {items[i]}: {'有交互' if feedback[i] else '无交互'}")

    # 测试测试集字典
    print("\n[8] 测试测试集字典...")
    test_dict = dataset.testDict
    print(f"✓ 测试集用户数: {len(test_dict)}")
    print(f"  前3个测试用户的物品:")
    for i, (user, items) in enumerate(list(test_dict.items())[:3]):
        print(f"    用户 {user}: {items}")

    # 测试稀疏图构建
    print("\n[9] 测试稀疏图构建...")
    print("  正在构建稀疏图...")
    graph = dataset.getSparseGraph()
    print(f"✓ 稀疏图构建成功")
    print(f"✓ 稀疏图形状: {graph.shape}")
    print(f"✓ 稀疏图类型: {type(graph)}")
    print(f"✓ 稀疏图设备: {graph.device}")

    # 测试 PyTorch Dataset 接口
    print("\n[10] 测试 PyTorch Dataset 接口...")
    print(f"✓ 数据集长度: {len(dataset)}")
    print(f"✓ 第一个样本: {dataset[0]}")
    print(f"✓ 第二个样本: {dataset[1]}")

    print("\n" + "=" * 60)
    print("✓ LastFM 数据加载器测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_lastfm_loader()