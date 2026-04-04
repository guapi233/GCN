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
        with torch.sparse.check_sparse_tensor_invariants(is_Train):
            return torch.sparse_coo_tensor(
                index, data, torch.Size(coo.shape), check_invariants=True
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
    
class LastFM(BasicDataset):
    """
    LastFM 数据集加载器

    特点：
    - 包含用户-物品交互数据
    - 包含用户社交网络（信任关系）
    - 小型数据集（1892用户，4489物品）
    - 适合快速测试和学习
    """
    
    def __init__(self, path="../data/lastfm"):
        """
        初始化 LastFM 数据集

        Args:
            path: 数据集路径
        """
        cprint(f"正在加载 [lastfm] 数据集...")

        # 训练/测试模式映射
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        # 读取原始数据
        # ----------------------------------------
        # 1. 读取训练集
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # 2. 读取测试集
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        # 3. 读取社交网络
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()

        # ID 归一化（从 1-based 转为 0-based）
        # ----------------------------------------
        # 社交网络 ID 归一化
        trustNet -= 1  # 1, 2, ... → 0, 1, ...
        # 训练集 ID 归一化
        trainData -= 1
        # 测试集 ID 归一化
        testData -= 1
        
        # 存储预处理后的数据
        # ----------------------------------------
        self.trustNet = trustNet         # 用户社交关系矩阵 [N×2]
        self.trainData = trainData       # 训练集原始数据
        self.testData = testData         # 测试集原始数据
        
        # 提取用户和物品列表
        # ----------------------------------------
        # 训练集
        self.trainUser = np.array(trainData.iloc[:, 0])      # 所有交互的用户ID
        self.trainUniqueUsers = np.unique(self.trainUser)   # 唯一用户ID
        self.trainItem = np.array(trainData.iloc[:, 1])     # 所有交互的物品ID（展平）

        # 测试集
        self.testUser = np.array(testData.iloc[:, 0])       # 所有测试用户ID
        self.testUniqueUsers = np.unique(self.testUser)    # 唯一测试用户ID
        self.testItem = np.array(testData.iloc[:, 1])       # 所有测试物品ID（展平）

        # 初始化图结构
        # ----------------------------------------
        self.Graph = None
        
        # 计算并打印稀疏度
        # ----------------------------------------
        n_interactions = len(self.trainUser) + len(self.testUser)
        sparsity = n_interactions / self.n_users / self.m_items
        print(f"LastFM 数据集稀疏度: {sparsity:.4f}")
        
        # 构建稀疏矩阵
        # ----------------------------------------
        # 社交网络矩阵：(users, users)
        self.socialNet = csr_matrix(
            (np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
            shape=(self.n_users, self.n_users)
        )
        
        # 用户-物品交互矩阵：(users, items)
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_users, self.m_items)
        )
        
        # 预计算数据（避免重复计算）
        # ----------------------------------------
        # 所有用户的正样本
        self._allPos = self.getUserPosItems(list(range(self.n_users)))

        # 所有用户的负样本（LastFM 是小型数据集，可以预计算）
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])           # 该用户的正样本
            neg = allItems - pos                 # 负样本 = 所有物品 - 正样本
            self.allNeg.append(np.array(list(neg)))

        # 构建测试集字典
        # ----------------------------------------
        self.__testDict = self.__build_test()
    
    @property
    def n_users(self):
        """用户数量"""
        return 1892

    @property
    def m_items(self):
        """物品数量"""
        return 4489

    @property
    def trainDataSize(self):
        """训练集交互数量"""
        return len(self.trainUser)

    @property
    def testDict(self):
        """测试集字典"""
        return self.__testDict

    @property
    def allPos(self):
        """所有用户正样本"""
        return self._allPos
    
    def __build_test(self):
        """
        构建测试集字典

        Returns:
            dict: {用户ID: [物品列表]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        查询用户-物品是否存在交互

        Args:
            users: 用户ID列表, shape=[-1]
            items: 物品ID列表, shape=[-1]

        Returns:
            np.ndarray: 交互状态 [0 或 1], shape=[-1]
        """
        # 从稀疏矩阵查询交互状态
        feedback = self.UserItemNet[users, items]
        return np.array(feedback).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        """
        获取指定用户的正样本物品

        Args:
            users: 用户ID列表

        Returns:
            list: 每个用户的正样本物品列表
        """
        posItems = []
        for user in users:
            # 从 UserItemNet 查该用户的非零列（即正样本）
            pos_items = self.UserItemNet[user].nonzero()[1]
            posItems.append(pos_items)
        return posItems
    
    def getUserNegItems(self, users):
        """
        获取指定用户的负样本物品

        注意：仅适用于小型数据集（如 LastFM）
              大型数据集因负样本过多，不能预计算

        Args:
            users: 用户ID列表

        Returns:
            list: 每个用户的负样本物品列表
        """
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
    
    def __getitem__(self, index):
        """
        PyTorch Dataset 必须实现的方法

        Args:
            index: 样本索引

        Returns:
            int: 用户ID
        """
        user = self.trainUniqueUsers[index]
        return user
    
    def __len__(self):
        """
        PyTorch Dataset 必须实现的方法

        Returns:
            int: 数据集大小（训练集唯一用户数）
        """
        return len(self.trainUniqueUsers)
    
    def getSparseGraph(self):
        """
        构建归一化的用户-物品二分稀疏图

        核心逻辑：
        1. 构建 [users, items] → [items, users] 的无向边
        2. 物品 ID 偏移，避免与用户 ID 冲突
        3. 对称归一化：A = D^(-0.5) × A × D^(-0.5)

        Returns:
            torch.sparse.FloatTensor: 归一化稀疏图
        """
        if self.Graph is None:
            # 1. 处理用户/物品 ID
            # ----------------------------------------
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            # 2. 生成二分图的边索引（无向图）
            # ----------------------------------------
            # 用户 → 物品：用户ID | (物品ID + 用户数)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])

            # 物品 → 用户：(物品ID + 用户数) | 用户ID
            second_sub = torch.stack([item_dim + self.n_users, user_dim])

            # 合并两类边，形成完整无向图
            index = torch.cat([first_sub, second_sub], dim=1)

            # 3. 生成边权重（初始为 1）
            # ----------------------------------------
            data = torch.ones(index.size(-1)).int()

            # 4. 构建初始稀疏图（未归一化）
            # ----------------------------------------
            with torch.sparse.check_sparse_tensor_invariants(True):
                self.Graph = torch.sparse_coo_tensor(
                    index, data,
                    torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
                )
            

            # 5. 图归一化（避免度数大的节点权重过高）
            # ----------------------------------------
            # 转换为稠密矩阵（小型数据集可行）
            dense = self.Graph.to_dense()

            # 计算每个节点的度数
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.  # 度数为 0 的节点设为 1（避免除零）

            # 对称归一化：A = D^(-0.5) × A × D^(-0.5)
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt           # 行归一化
            dense = dense / D_sqrt.t()       # 列归一化

            # 6. 转回稀疏张量（节省内存）
            # ----------------------------------------
            index = dense.nonzero()          # 提取非零元素索引
            data = dense[dense >= 1e-9]      # 提取非零元素权重

            # 构建最终归一化稀疏图
            with torch.sparse.check_sparse_tensor_invariants(True):
                self.Graph = torch.sparse_coo_tensor(
                    index.t(), data,
                    torch.Size([self.n_users + self.m_items, self.n_users + self.m_items])
                )
            self.Graph = self.Graph.coalesce().to(world.device)

        return self.Graph