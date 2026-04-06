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

class Loader(BasicDataset):
    """
    通用数据集加载器

    支持：
    - Gowalla
    - Yelp2018
    - Amazon-book
    - 大规模数据处理
    - 稀疏图预计算
    - 图分片优化

    特点：
    - 动态计算用户/物品数量
    - 预计算并保存归一化邻接矩阵
    - 支持图分片（应对超大规模数据）
    """
    def __init__(self, config=world.config, path="../data/gowalla"):
        """
        初始化通用数据加载器

        Args:
            config: 配置字典
            path: 数据集路径
        """
        cprint(f"正在加载 [{path}] 数据集...")

        # 图分片配置
        self.split = config['A_split']    # 是否分片
        self.folds = config['A_n_fold']   # 分片数量

        # 训练/测试模式映射
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        # 动态计算的用户/物品数
        self.n_user = 0
        self.m_item = 0

        # 文件路径
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path

        # 初始化数据存储
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        # 读取训练集
        # ----------------------------------------
        print(f"正在读取训练集: {train_file}")
        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    # 解析每行：用户ID 物品1 物品2 ...
                    parts = line.strip('\n').split(' ')
                    items = [int(i) for i in parts[1:]]
                    uid = int(parts[0])

                    # 添加到训练集
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)

                    # 动态更新最大 ID
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)

                    self.traindataSize += len(items)

        # 读取测试集
        # ----------------------------------------
        print(f"正在读取测试集: {test_file}")
        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    parts = line.strip('\n').split(' ')
                    
                    # items = [int(i) for i in parts[1:]]
                    items = [int(i) for i in parts[1:] if i.strip() != ''] # 跳过无交互用户项
                    
                    if not items:
                        continue  # 不加载无交互用户项
                    
                    uid = int(parts[0])

                    # 添加到测试集
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)

                    # 继续更新最大 ID（测试集可能有训练集未出现的物品）
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)

                    self.testDataSize += len(items)

        # 转换为 NumPy 数组
        # ----------------------------------------
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        # ID 归一化：最大 ID + 1 = 总数量
        # ----------------------------------------
        self.m_item += 1
        self.n_user += 1

        # 初始化图结构
        # ----------------------------------------
        self.Graph = None

        # 打印数据统计信息
        # ----------------------------------------
        print(f"{self.traindataSize} 个交互用于训练")
        print(f"{self.testDataSize} 个交互用于测试")
        sparsity = (self.traindataSize + self.testDataSize) / self.n_user / self.m_item
        print(f"{path.split('/')[-1]} 数据集稀疏度: {sparsity:.6f}")

        # 构建用户-物品交互稀疏矩阵
        # ----------------------------------------
        print("正在构建 UserItemNet 稀疏矩阵...")
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        # 计算用户/物品的度数（用于归一化）
        # ----------------------------------------
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1

        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # 预计算数据
        # ----------------------------------------
        print("正在预计算用户正样本...")
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

        print("正在构建测试集字典...")
        self.__testDict = self.__build_test()

        print(f"{path.split('/')[-1]} 数据集已准备就绪！")

    @property
    def n_users(self):
        """用户数量"""
        return self.n_user

    @property
    def m_items(self):
        """物品数量"""
        return self.m_item

    @property
    def trainDataSize(self):
        """训练集交互数量"""
        return self.traindataSize

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
            users: 用户ID列表
            items: 物品ID列表

        Returns:
            np.ndarray: 交互状态 [0 或 1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

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
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

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
            int: 数据集大小
        """
        return len(self.trainUniqueUsers)
    
    def getSparseGraph(self):
        """
        构建归一化的用户-物品二分稀疏图（优化版，全程稀疏运算，支持amazon-book）
        核心优化：
        1. 用COO格式直接构建邻接矩阵，杜绝稠密化赋值导致的内存溢出
        2. 全程稀疏矩阵运算，仅存储非零值，内存占用骤降
        3. 保留原逻辑：预计算缓存、图分片、PyTorch稀疏张量转换
        Returns:
            torch.sparse.FloatTensor: 归一化稀疏图（分片则返回列表）
        """
        print("正在加载邻接矩阵...")
        if self.Graph is None:
            # 尝试加载预计算的归一化邻接矩阵（避免重复计算）
            cache_path = self.path + '/s_pre_adj_mat.npz'
            try:
                pre_adj_mat = sp.load_npz(cache_path)
                print("✓ 成功加载预计算的邻接矩阵")
                norm_adj = pre_adj_mat
            except FileNotFoundError:
                # 无缓存时重新生成（全程稀疏，核心优化段）
                print("✗ 未找到预计算的邻接矩阵，正在重新生成...")
                s = time()
                n_users = self.n_users
                m_items = self.m_items
                total_nodes = n_users + m_items

                # 步骤1：提取用户-物品交互对（从稀疏矩阵中获取非零值）
                print("  步骤1: 提取用户-物品交互对...")
                user_indices, item_indices = self.UserItemNet.nonzero()  # 直接取稀疏矩阵的非零索引
                data = np.ones_like(user_indices, dtype=np.float32)       # 交互边权重为1

                # 步骤2：构建二部图邻接矩阵（COO格式，全程稀疏，无稠密转换）
                print("  步骤2: 构建稀疏二部图邻接矩阵...")
                # 构建用户→物品的边：用户ID | 物品ID+用户数（偏移避免ID冲突）
                row_u2i = user_indices
                col_u2i = item_indices + n_users
                # 构建物品→用户的边：物品ID+用户数 | 用户ID（无向图，双向边）
                row_i2u = col_u2i
                col_i2u = row_u2i
                # 合并双向边的行、列索引和权重
                all_rows = np.concatenate([row_u2i, row_i2u])
                all_cols = np.concatenate([col_u2i, col_i2u])
                all_data = np.concatenate([data, data])
                # 用COO格式构建邻接矩阵（仅存储非零值，内存效率最高）
                adj_mat = sp.coo_matrix(
                    (all_data, (all_rows, all_cols)),
                    shape=(total_nodes, total_nodes),
                    dtype=np.float32
                ).tocsr()  # 转为CSR格式，方便后续归一化运算
                print(f"  稀疏邻接矩阵形状: {adj_mat.shape}")
                print(f"  非零元素数: {adj_mat.nnz}")

                # 步骤3：对称归一化（全程稀疏矩阵运算，无稠密转换）
                print("  步骤3: 对称归一化...")
                rowsum = np.array(adj_mat.sum(axis=1)).flatten()  # 计算每个节点的度数
                d_inv = np.zeros_like(rowsum, dtype=np.float32)  # 初始化度数倒数矩阵
                mask = rowsum != 0  # 筛选度数非零的节点
                d_inv[mask] = np.power(rowsum[mask], -0.5)       # 仅对非零节点计算-0.5次方
                d_mat = sp.diags(d_inv, dtype=np.float32)         # 构建度数对角矩阵
                # 归一化公式：A_norm = D^(-0.5) · A · D^(-0.5)
                norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()

                # 步骤4：保存缓存（后续运行直接加载，无需重新计算）
                print("  步骤4: 保存归一化邻接矩阵缓存...")
                sp.save_npz(cache_path, norm_adj)
                end = time()
                print(f"✓ 邻接矩阵构建完成，耗时 {end-s:.2f}s")

            # 步骤5：处理图分片/转换为PyTorch稀疏张量（保留原代码逻辑，兼容配置）
            if self.split:
                print("正在分片邻接矩阵...")
                self.Graph = self._split_A_hat(norm_adj, n_folds=self.folds)  # 用配置的分片数，而非硬编码5
                print(f"✓ 邻接矩阵已分为 {len(self.Graph)} 片")
            else:
                print("正在转换为 PyTorch 稀疏张量...")
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("✓ 邻接矩阵转换完成并迁移至设备")
        return self.Graph