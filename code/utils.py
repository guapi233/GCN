import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os

## CPP加速负样本采样
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False
    
def UniformSample_Python(dataset):
    """
    优化版负采样

    使用向量化操作加速

    Args:
        dataset: 数据集对象

    Returns:
        tuple: (users, pos_items, neg_items)
    """
    # 训练数据
    users = dataset.trainUser
    pos_items = dataset.trainItem

    # 用户数量
    n_samples = len(users)

    # 负采样
    neg_items = np.random.randint(
        0,
        dataset.m_items,
        size=n_samples
    )

    # 过滤：确保负样本不是正样本
    for i in range(n_samples):
        user = users[i]
        pos = dataset.allPos[user]

        # 重采样直到不是正样本
        attempts = 0
        while neg_items[i] in pos and attempts < 10:
            neg_items[i] = np.random.randint(0, dataset.m_items)
            attempts += 1

    return users, pos_items, neg_items

def batch_sampling(dataset, users, n_neg=1):
    """
    批量负采样

    Args:
        dataset: 数据集对象
        users: 用户列表
        n_neg: 每个正样本的负采样数量

    Returns:
        dict: {user: [neg_items]}
    """
    results = {
        'users': [],
        'pos_items': [],
        'neg_items': []
    }

    for user in users:
        # 获取用户的正样本
        pos_items = dataset.allPos[user]

        # 获取所有物品
        all_items = set(range(dataset.m_items))

        # 负采样：排除正样本
        neg_candidates = all_items - set(pos_items)
        neg_items = np.random.choice(
            list(neg_candidates),
            size=n_neg,
            replace=False
        )

        results['users'].extend([user] * n_neg)
        results['pos_items'].extend(pos_items * n_neg)
        results['neg_items'].extend(neg_items)

    return results

class NegativeSampler:
    """
    负采样器

    预计算每个用户的负样本候选集，加速采样
    """

    def __init__(self, dataset):
        """
        初始化负采样器

        Args:
            dataset: 数据集对象
        """
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items

        # 预计算每个用户的负样本候选集
        self._init_neg_pool()

    def _init_neg_pool(self):
        """初始化负样本候选池"""
        all_items = set(range(self.m_items))

        self.neg_pool = []
        for user in range(self.n_users):
            pos_items = set(self.dataset.allPos[user])
            neg_items = all_items - pos_items
            self.neg_pool.append(list(neg_items))

    def sample(self, user, n_neg=1):
        """
        为用户采样负样本

        Args:
            user: 用户 ID
            n_neg: 采样数量

        Returns:
            list: 负样本物品 ID 列表
        """
        return np.random.choice(
            self.neg_pool[user],
            size=n_neg,
            replace=False
        )
        
def analyze_sampling(dataset, n_samples=10000):
    """
    分析负采样分布

    Args:
        dataset: 数据集对象
        n_samples: 采样数量
    """
    # 执行采样
    users, pos_items, neg_items = UniformSample_Python(dataset)

    # 统计
    print(f"总样本数: {len(users)}")
    print(f"唯一用户: {len(set(users))}")
    print(f"唯一正样本: {len(set(pos_items))}")
    print(f"唯一负样本: {len(set(neg_items))}")

    # 检查负样本质量
    conflict = 0
    for u, neg in zip(users, neg_items):
        if neg in dataset.allPos[u]:
            conflict += 1

    print(f"冲突样本（负样本是正样本）: {conflict}")
    print(f"冲突率: {conflict / len(users) * 100:.2f}%")