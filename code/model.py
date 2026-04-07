"""
模型定义模块
"""

import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

class BasicModel(nn.Module):
    """
    推荐模型基类

    所有推荐模型都必须继承这个类
    """
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        """
        获取用户对所有物品的评分预测

        Args:
            users: 用户ID列表，shape=[batch_size]

        Returns:
            torch.Tensor: 评分矩阵，shape=[batch_size, n_items]
                         每个用户对每个物品的预测得分
        """
        raise NotImplementedError("子类必须实现 getUsersRating 方法")
    
class PairWiseModel(BasicModel):
    """
    Pairwise 损失模型基类

    继承 BasicModel，添加 BPR 损失接口
    """
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        计算 BPR 损失

        Args:
            users: 用户ID列表，shape=[batch_size]
            pos: 正样本物品ID列表，shape=[batch_size]
            neg: 负样本物品ID列表，shape=[batch_size]

        Returns:
            tuple: (log_loss, l2_loss)
                   - log_loss: BPR 对数损失
                   - l2_loss: L2 正则化损失
        """
        raise NotImplementedError("子类必须实现 bpr_loss 方法")

class PureMF(BasicModel):
    """
    纯矩阵分解模型（Pure Matrix Factorization）

    特点：
    - 简单的嵌入层，不使用图卷积
    - 作为 baseline 对比模型
    - 快速训练，易于理解

    结构：
    - 用户嵌入：[n_users × latent_dim]
    - 物品嵌入：[m_items × latent_dim]
    """
    def __init__(self, config: dict, dataset: BasicDataset):
        """
        初始化矩阵分解模型

        Args:
            config: 配置字典
            dataset: 数据集对象
        """
        super(PureMF, self).__init__()

        # 获取配置参数
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']  # 嵌入维度

        # 激活函数（sigmoid，将得分压缩到 [0, 1]）
        self.f = nn.Sigmoid()

        # 初始化嵌入层
        self._init_weight()
        
    def _init_weight(self):
        """初始化用户和物品嵌入层"""
        # 用户嵌入层
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )

        # 物品嵌入层
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )

        # 初始化策略：标准正态分布 N(0, 1)
        nn.init.normal_(self.embedding_user.weight, std=1.0)
        nn.init.normal_(self.embedding_item.weight, std=1.0)
        print("使用标准正态分布 N(0,1) 初始化嵌入矩阵")
        
    def getUsersRating(self, users):
        """
        预测用户对所有物品的评分

        Args:
            users: 用户ID列表，shape=[batch_size]

        Returns:
            torch.Tensor: 评分矩阵，shape=[batch_size, n_items]
        """
        users = users.long()

        # 获取用户嵌入
        users_emb = self.embedding_user(users)  # [batch_size, latent_dim]

        # 获取所有物品嵌入
        items_emb = self.embedding_item.weight  # [n_items, latent_dim]

        # 计算得分：用户嵌入 × 物品嵌入转置（点积）
        scores = torch.matmul(users_emb, items_emb.t())  # [batch_size, n_items]

        # 通过 sigmoid 激活
        return self.f(scores)  # [batch_size, n_items]
    
    def bpr_loss(self, users, pos, neg):
        """
        计算 BPR 损失

        Args:
            users: 用户ID列表
            pos: 正样本物品ID列表
            neg: 负样本物品ID列表

        Returns:
            tuple: (log_loss, l2_loss)
        """
        users_emb = self.embedding_user(users.long())  # [batch_size, latent_dim]
        pos_emb = self.embedding_item(pos.long())       # [batch_size, latent_dim]
        neg_emb = self.embedding_item(neg.long())     # [batch_size, latent_dim]

        # 计算正样本得分
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)  # [batch_size]

        # 计算负样本得分
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)  # [batch_size]

        # BPR 损失：softplus(neg_scores - pos_scores)
        # 目标：pos_scores > neg_scores
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        # L2 正则化损失
        reg_loss = (1 / 2) * (
            users_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / float(len(users))

        return loss, reg_loss
    
    def forward(self, users, items):
        """
        前向传播（训练时使用）

        Args:
            users: 用户ID列表
            items: 物品ID列表

        Returns:
            torch.Tensor: 得分，shape=[batch_size]
        """
        users = users.long()
        items = items.long()

        users_emb = self.embedding_user(users)  # [batch_size, latent_dim]
        items_emb = self.embedding_item(items)  # [batch_size, latent_dim]

        # 计算得分：点积
        inner_pro = torch.mul(users_emb, items_emb)  # [batch_size, latent_dim]
        scores = torch.sum(inner_pro, dim=1)       # [batch_size]

        return scores

class LightGCN(BasicModel):
    """
    LightGCN 模型

    特点：
    - 简化图卷积网络
    - 去除特征变换和非线性激活
    - 通过层间聚合获得最终表示

    结构：
    - 嵌入层：用户和物品嵌入
    - 图卷积：多层传播
    - 聚合：层间加权平均
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        """
        初始化 LightGCN 模型

        Args:
            config: 配置字典，包含模型超参数
            dataset: 数据集对象，提供用户/物品数量和稀疏图
        """
        super(LightGCN, self).__init__()

        # 保存配置和数据集
        self.config = config
        self.dataset = dataset

        # 初始化
        self._init_weight()

    def _init_weight(self):
        """初始化模型参数"""
        # 1. 获取超参数
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # 2. 嵌入层
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )

        # 3. 嵌入初始化
        if self.config['pretrain'] == 0:
            # 正态分布 N(0, 0.1)
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("使用正态分布 N(0, 0.1) 初始化 LightGCN")
        else:
            # 预训练嵌入
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))
            print("使用预训练数据初始化")

        # 4. 激活函数
        self.f = nn.Sigmoid()

        # 5. 稀疏图
        self.Graph = self.dataset.getSparseGraph()
        print(f"LightGCN 准备就绪 (dropout: {self.config['dropout']})")

    def _dropout(self, keep_prob):
        """
        应用 Dropout

        Args:
            keep_prob: 保留概率

        Returns:
            torch.Tensor 或 list: Dropout 后的图
        """
        if self.A_split:
            # 分片图 Dropout
            graph = []
            for g in self.Graph:
                graph.append(self._dropout_x(g, keep_prob))
        else:
            # 完整图 Dropout
            graph = self._dropout_x(self.Graph, keep_prob)
        return graph
    
    def _dropout_x(self, x, keep_prob):
        """
        稀疏图的 Dropout

        Args:
            x: 稀疏图张量
            keep_prob: 保留概率

        Returns:
            torch.Tensor: Dropout 后的稀疏图
        """
        # 获取稀疏图信息
        size = x.size()
        index = x.indices().t()   # 边索引 (n_edges, 2)
        values = x.values()      # 边权重

        # 随机生成保留掩码
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()

        # 筛选保留的边
        index = index[random_index]
        values = values[random_index] / keep_prob

        # 重构稀疏图
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def computer(self):
        """
        图卷积传播算子

        执行多层图卷积传播，融合各层嵌入

        Returns:
            tuple: (users_emb, items_emb)
                - users_emb: 用户最终嵌入 [n_users, latent_dim]
                - items_emb: 物品最终嵌入 [n_items, latent_dim]
        """
        # 1. 获取初始嵌入
        users_emb = self.embedding_user.weight     # [n_users, latent_dim]
        items_emb = self.embedding_item.weight    # [n_items, latent_dim]

        # 2. 合并用户/物品嵌入
        all_emb = torch.cat([users_emb, items_emb])  # [n_users+n_items, latent_dim]

        # 3. 初始化嵌入列表
        embs = [all_emb]

        # 4. 获取稀疏图（训练时可能 Dropout）
        if self.config['dropout']:
            if self.training:
                g_droped = self._dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        # 5. 多层图卷积传播
        for layer in range(self.n_layers):
            if self.A_split:
                # 分片图处理
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                all_emb = torch.cat(temp_emb, dim=0)
            else:
                # 完整图处理
                all_emb = torch.sparse.mm(g_droped, all_emb)

            # 保存当前层嵌入
            embs.append(all_emb)

        # 6. 多层嵌入融合
        embs = torch.stack(embs, dim=1)        # [n_nodes, n_layers+1, latent_dim]
        light_out = torch.mean(embs, dim=1)    # [n_nodes, latent_dim]

        # 7. 拆分用户/物品嵌入
        users, items = torch.split(
            light_out,
            [self.num_users, self.num_items]
        )

        return users, items

    def getUsersRating(self, users):
        """
        预测用户对所有物品的评分

        Args:
            users: 用户 ID 列表，shape=[batch_size]

        Returns:
            torch.Tensor: 评分矩阵，shape=[batch_size, n_items]
        """
        # 1. 获取优化后的嵌入
        all_users, all_items = self.computer()

        # 2. 提取用户嵌入
        users_emb = all_users[users.long()]

        # 3. 获取所有物品嵌入
        items_emb = all_items

        # 4. 计算评分
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        """
        获取损失计算所需的嵌入

        Args:
            users: 用户 ID 列表
            pos_items: 正样本物品 ID 列表
            neg_items: 负样本物品 ID 列表

        Returns:
            tuple: (users_emb, pos_emb, neg_emb,
                    users_emb_ego, pos_emb_ego, neg_emb_ego)
        """
        # 1. 获取优化后的嵌入
        all_users, all_items = self.computer()

        # 2. 提取正负样本嵌入
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # 3. 获取初始嵌入（用于 L2 正则化）
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        计算 BPR 损失

        Args:
            users: 用户 ID 列表
            pos: 正样本物品 ID 列表
            neg: 负样本物品 ID 列表

        Returns:
            tuple: (loss, reg_loss)
                - loss: BPR 损失
                - reg_loss: L2 正则化损失
        """
        # 1. 获取嵌入
        (users_emb, pos_emb, neg_emb,
        users_emb_ego, pos_emb_ego, neg_emb_ego) = self.getEmbedding(
            users.long(), pos.long(), neg.long())

        # 2. 计算 L2 正则化损失
        reg_loss = (1/2) * (
            users_emb_ego.norm(2).pow(2) +
            pos_emb_ego.norm(2).pow(2) +
            neg_emb_ego.norm(2).pow(2)
        ) / float(len(users))

        # 3. 计算预测得分
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        # 4. 计算 BPR 损失
        loss = torch.mean(
            torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        """
        前向传播（训练时使用）

        Args:
            users: 用户 ID 列表
            items: 物品 ID 列表

        Returns:
            torch.Tensor: 预测得分
        """
        # 1. 获取所有用户/物品嵌入
        all_users, all_items = self.computer()

        # 2. 提取批量的嵌入
        users_emb = all_users[users]
        items_emb = all_items[items]

        # 3. 计算得分
        scores = torch.sum(users_emb * items_emb, dim=1)

        return self.f(scores)