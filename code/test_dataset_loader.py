"""
测试通用数据加载器
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataloader import Loader
import torch
import numpy as np


def test_loader(dataset_name="gowalla"):
    """测试通用数据加载器"""
    print("=" * 60)
    print(f"测试通用数据加载器 - {dataset_name}")
    print("=" * 60)

    # 检查数据集是否存在
    data_path = f"../data/{dataset_name}"
    if not os.path.exists(data_path):
        print(f"\n✗ 数据集不存在: {data_path}")
        print("  请先下载数据集并放在对应目录下")
        return

    # 初始化数据集
    print(f"\n[1] 初始化数据集...")
    import world
    config = world.config.copy()
    config['A_split'] = False  # 不分片
    dataset = Loader(config=config, path=data_path)
    print("✓ 数据集初始化成功")

    # 测试基本属性
    print("\n[2] 测试基本属性...")
    print(f"✓ 用户数: {dataset.n_users:,}")
    print(f"✓ 物品数: {dataset.m_items:,}")
    print(f"✓ 训练集大小: {dataset.trainDataSize:,}")
    print(f"✓ 训练集唯一用户数: {len(dataset.trainUniqueUsers):,}")
    print(f"✓ 测试集唯一用户数: {len(dataset.testUniqueUsers):,}")

    # 测试交互矩阵
    print("\n[3] 测试交互矩阵...")
    print(f"✓ UserItemNet 形状: {dataset.UserItemNet.shape}")
    print(f"✓ UserItemNet 非零元素数: {dataset.UserItemNet.nnz:,}")
    avg_interactions = dataset.UserItemNet.nnz / dataset.n_users
    print(f"✓ 平均每个用户交互数: {avg_interactions:.2f}")

    # 测试正样本
    print("\n[4] 测试正样本查询...")
    test_users = [0, 1, 2]
    pos_items = dataset.getUserPosItems(test_users)
    print(f"✓ 查询用户 {test_users} 的正样本:")
    for i, user in enumerate(test_users):
        print(f"    用户 {user}: {list(pos_items[i][:5])}... (共{len(pos_items[i])}个)")

    # 测试交互反馈
    print("\n[5] 测试交互反馈查询...")
    users = [0, 1, 2]
    items = [10, 20, 30]
    feedback = dataset.getUserItemFeedback(users, items)
    print(f"✓ 用户-物品交互反馈:")
    for i in range(len(users)):
        print(f"    用户 {users[i]} - 物品 {items[i]}: {'有交互' if feedback[i] else '无交互'}")

    # 测试测试集字典
    print("\n[6] 测试测试集字典...")
    test_dict = dataset.testDict
    print(f"✓ 测试集用户数: {len(test_dict):,}")
    print(f"  前3个测试用户的物品:")
    for i, (user, items) in enumerate(list(test_dict.items())[:3]):
        print(f"    用户 {user}: {items[:5]}...")

    # 测试稀疏图构建
    print("\n[7] 测试稀疏图构建...")
    print("  正在构建稀疏图...")
    import time
    start = time.time()
    graph = dataset.getSparseGraph()
    elapsed = time.time() - start
    print(f"✓ 稀疏图构建成功，耗时 {elapsed:.2f}s")
    
    if hasattr(graph, 'shape'):
        print(f"✓ 稀疏图形状: {graph.shape}")
        print(f"✓ 稀疏图非零元素数: {graph._nnz():,}")
        print(f"✓ 稀疏图设备: {graph.device}")
    else:
        print(f"√ 稀疏图分块数：{len(graph)}")
        print(f"√ 单个分块形状：{graph[0].shape}")
        print(f"✓ 稀疏图非零元素数: {sum([block._nnz() for block in graph]):,}")
        print(f"✓ 稀疏图设备: {graph[0].device}")
 

    # 测试预计算矩阵加载
    print("\n[8] 测试预计算矩阵加载...")
    import scipy.sparse as sp
    pre_adj_path = f"{data_path}/s_pre_adj_mat.npz"
    if os.path.exists(pre_adj_path):
        print("✓ 预计算矩阵已存在")
        pre_adj = sp.load_npz(pre_adj_path)
        print(f"✓ 预计算矩阵形状: {pre_adj.shape}")
        print(f"✓ 预计算矩阵非零元素数: {pre_adj.nnz:,}")
    else:
        print("✗ 预计算矩阵不存在（首次运行会自动生成）")

    # 测试 PyTorch Dataset 接口
    print("\n[9] 测试 PyTorch Dataset 接口...")
    print(f"✓ 数据集长度: {len(dataset):,}")
    print(f"✓ 第一个样本: {dataset[0]}")
    print(f"✓ 最后一个样本: {dataset[-1]}")

    print("\n" + "=" * 60)
    print(f"✓ {dataset_name} 数据加载器测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    # 测试不同的数据集
    datasets = ["gowalla", "yelp2018", "amazon-book"]

    for dataset in datasets:
        print("\n\n")
        test_loader(dataset)