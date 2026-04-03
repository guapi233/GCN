"""
测试 BasicDataset 基类
"""
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataloader import BasicDataset
import torch


class TestDataset(BasicDataset):
    """测试用数据集实现"""
    def __init__(self):
        super().__init__()
        self._n_users = 100
        self._m_items = 500
        self._trainDataSize = 1000
        self._testDict = {}
        self._allPos = [[] for _ in range(self._n_users)]
        self._trainUniqueUsers = list(range(self._n_users))

    @property
    def n_users(self):
        return self._n_users

    @property
    def m_items(self):
        return self._m_items

    @property
    def trainDataSize(self):
        return self._trainDataSize

    @property
    def testDict(self):
        return self._testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserItemFeedback(self, users, items):
        return np.zeros(len(users), dtype=np.uint8)

    def getUserPosItems(self, users):
        return [self._allPos[u] for u in users]

    def getUserNegItems(self, users):
        return [list(range(self._m_items)) for u in users]

    def getSparseGraph(self):
        return None

    def __getitem__(self, index):
        return self._trainUniqueUsers[index]

    def __len__(self):
        return len(self._trainUniqueUsers)


# 测试基类
if __name__ == '__main__':
    print("=" * 50)
    print("测试 BasicDataset 基类")
    print("=" * 50)

    dataset = TestDataset()

    # 测试 1: 属性访问
    print("\n[1] 测试属性访问...")
    print(f"✓ 用户数: {dataset.n_users}")
    print(f"✓ 物品数: {dataset.m_items}")
    print(f"✓ 训练集大小: {dataset.trainDataSize}")

    # 测试 2: 方法调用
    print("\n[2] 测试方法调用...")
    users = [0, 1, 2]
    items = [10, 20, 30]
    feedback = dataset.getUserItemFeedback(users, items)
    print(f"✓ 用户-物品反馈: {feedback}")

    pos_items = dataset.getUserPosItems(users)
    print(f"✓ 用户正样本: {pos_items}")

    # 测试 3: PyTorch Dataset 接口
    print("\n[3] 测试 PyTorch Dataset 接口...")
    print(f"✓ 数据集长度: {len(dataset)}")
    print(f"✓ 第一个样本: {dataset[0]}")
    print(f"✓ 最后一个样本: {dataset[-1]}")

    # 测试 4: 稀疏矩阵转换（模拟）
    print("\n[4] 测试稀疏矩阵转换...")
    import scipy.sparse as sp
    import numpy as np

    # 创建测试稀疏矩阵
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    data = np.array([1.0, 1.0, 1.0])
    sparse_mat = sp.coo_matrix((data, (row, col)), shape=(3, 3))

    # 转换为 PyTorch 稀疏张量
    sparse_tensor = dataset._convert_sp_mat_to_sp_tensor(sparse_mat)
    print(f"✓ 稀疏张量形状: {sparse_tensor.shape}")
    print(f"✓ 稀疏张量类型: {type(sparse_tensor)}")

    print("\n" + "=" * 50)
    print("✓ 所有测试通过！")
    print("=" * 50)