"""
数据加载模块
设计原则：
1. 数据集索引从 0 开始
2. 使用稀疏矩阵存储交互数据
3. 提供统一的接口供不同数据集实现
"""

import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    """
    数据集抽象基类

    所有具体数据集都需要继承这个类并实现抽象方法
    """
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        """
        用户数量（属性，不是方法）

        Returns:
            int: 用户总数
        """
        raise NotImplementedError("子类必须实现 n_users 属性")

    @property
    def m_items(self):
        """
        物品数量（属性，不是方法）

        Returns:
            int: 物品总数
        """
        raise NotImplementedError("子类必须实现 m_items 属性")

    @property
    def trainDataSize(self):
        """
        训练集交互数量

        Returns:
            int: 训练集的总交互数（边数）
        """
        raise NotImplementedError("子类必须实现 trainDataSize 属性")

    @property
    def testDict(self):
        """
        测试集字典

        Returns:
            dict: {用户ID: [物品列表]}
                  例如：{0: [5, 8, 10], 1: [2, 3]}
        """
        raise NotImplementedError("子类必须实现 testDict 属性")

    @property
    def allPos(self):
        """
        所有用户的训练集正样本

        Returns:
            list: 每个用户的正样本物品列表
                  例如：[[5, 8], [2, 3, 10], [1]]
        """
        raise NotImplementedError("子类必须实现 allPos 属性")

    def getUserItemFeedback(self, users, items):
        """
        查询用户-物品是否存在交互

        Args:
            users: 用户ID列表，shape=[-1]
            items: 物品ID列表，shape=[-1]

        Returns:
            np.ndarray: 交互状态数组，shape=[-1]
                        1=有交互，0=无交互
        """
        raise NotImplementedError("子类必须实现 getUserItemFeedback 方法")

    def getUserPosItems(self, users):
        """
        获取指定用户的正样本物品

        Args:
            users: 用户ID列表

        Returns:
            list: 每个用户的正样本物品列表
        """
        raise NotImplementedError("子类必须实现 getUserPosItems 方法")

    def getUserNegItems(self, users):
        """
        获取指定用户的负样本物品

        注意：对于大型数据集，预计算所有负样本会导致内存爆炸
              因此这个方法可能不被实现

        Args:
            users: 用户ID列表

        Returns:
            list: 每个用户的负样本物品列表
        """
        raise NotImplementedError("子类必须实现 getUserNegItems 方法")

    def getSparseGraph(self):
        """
        获取用户-物品二分稀疏图

        构建对称归一化的邻接矩阵：
        A = [ I   R ]
            [R^T I ]

        归一化：A_norm = D^(-0.5) × A × D^(-0.5)

        Returns:
            torch.sparse.FloatTensor: 归一化的稀疏邻接矩阵
                                       shape=[n_users+m_items, n_users+m_items]
        """
        raise NotImplementedError("子类必须实现 getSparseGraph 方法")

    def __getitem__(self, index):
        """
        PyTorch Dataset 必须实现的方法

        Args:
            index: 样本索引

        Returns:
            用户ID（用于后续训练时的负采样）
        """
        raise NotImplementedError("子类必须实现 __getitem__ 方法")

    def __len__(self):
        """
        PyTorch Dataset 必须实现的方法

        Returns:
            int: 数据集大小（训练集唯一用户数）
        """
        raise NotImplementedError("子类必须实现 __len__ 方法")
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量

        Args:
            X: scipy.sparse 矩阵（CSR 或 COO 格式）

        Returns:
            torch.sparse.FloatTensor: PyTorch 稀疏张量
        """
        # 转换为 COO 格式（更易转换为 PyTorch 稀疏张量）
        coo = X.tocoo().astype(np.float32)

        # 提取行索引和列索引，并转为 PyTorch LongTensor
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()

        # 合并为 2×E 的边索引（E 为边数）
        index = torch.stack([row, col])

        # 提取边权重
        data = torch.FloatTensor(coo.data)

        # 创建 PyTorch 稀疏张量
        is_Train = True # 【占位⭐️】根据环境判断是否打开向量检查
        
        # 训练时：开启检查（默认）
        if is_Train:
            with torch.sparse.check_sparse_tensor_invariants(True):
                return torch.sparse_coo_tensor(
                    index, data, torch.Size(coo.shape), check_invariants=True
                )
        # 推理时：显式关闭全局检查，避免警告
        else:
            with torch.sparse.check_sparse_tensor_invariants(False):
                return torch.sparse_coo_tensor(
                    index, data, torch.Size(coo.shape), check_invariants=False
                )
        
    def _split_A_hat(self, A, n_folds):
        """
        将大型邻接矩阵分片（用于内存优化）

        Args:
            A: 归一化后的邻接矩阵（CSR 格式）
            n_folds: 分片数量

        Returns:
            list: 分片后的稀疏张量列表
        """
        A_fold = []

        # 计算每个分片的节点数
        n_nodes = self.n_users + self.m_items
        fold_len = n_nodes // n_folds

        for i_fold in range(n_folds):
            start = i_fold * fold_len
            if i_fold == n_folds - 1:
                # 最后一个分片包含剩余所有节点
                end = n_nodes
            else:
                end = (i_fold + 1) * fold_len

            # 提取分片并转换为 PyTorch 稀疏张量
            sub_graph = self._convert_sp_mat_to_sp_tensor(A[start:end])
            sub_graph = sub_graph.coalesce().to(world.device)
            A_fold.append(sub_graph)

        return A_fold