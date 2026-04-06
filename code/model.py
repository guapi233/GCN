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