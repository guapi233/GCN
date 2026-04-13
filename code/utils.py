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
    优化版负采样（向量化加速）

    Args:
        dataset: 数据集对象

    Returns:
        np.array: 采样结果，每行 [user, pos_item, neg_item]
    """
    users = dataset.trainUser
    pos_items = dataset.trainItem
    n_samples = len(users)
    m_items = dataset.m_items

    # 预构建用户-正样本集合字典（O(1) 查找）
    user_pos_set = {u: set(dataset.allPos[u]) for u in np.unique(users)}

    # 向量化采样
    neg_items = np.random.randint(0, m_items, size=n_samples)

    # 向量化过滤：标记需要重采样的位置
    mask = np.array([neg_items[i] in user_pos_set.get(users[i], set()) 
                     for i in range(n_samples)])
    
    # 最多重试 10 次
    for _ in range(10):
        retry_indices = np.where(mask)[0]
        if len(retry_indices) == 0:
            break
        new_neg = np.random.randint(0, m_items, size=len(retry_indices))
        neg_items[retry_indices] = new_neg
        mask[retry_indices] = [new_neg[i] in user_pos_set.get(users[retry_indices[i]], set()) 
                               for i in range(len(retry_indices))]

    S = np.column_stack((users, pos_items, neg_items))
    return S

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
    
    
## BPR损失函数类
class BPRLoss:
    """
    BPR 损失函数类

    封装了 BPR 损失计算和优化器更新

    Attributes:
        model: 推荐模型
        weight_decay: L2 正则化系数
        lr: 学习率
        opt: Adam 优化器
    """

    def __init__(self, recmodel: PairWiseModel, config: dict):
        """
        初始化 BPR 损失函数

        Args:
            recmodel: 推荐模型（继承 PairWiseModel）
            config: 配置字典，包含 'decay' 和 'lr'
        """
        self.model = recmodel
        self.weight_decay = config['decay']  # 权重衰退系数
        self.lr = config['lr']  # 学习率
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)  # 优化器

    ## 核心训练步骤
    def stageOne(self, users, pos, neg):
        """
        执行一次训练步骤

        Args:
            users: 用户 ID 张量
            pos: 正样本物品 ID 张量
            neg: 负样本物品 ID 张量

        Returns:
            float: 损失值
        """
        # 计算 BPR 损失和 L2 正则化损失
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)

        # L2 损失乘以衰退系数后叠加到总损失上
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        # 梯度清零 → 反向传播计算梯度 → 优化器更新参数
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()
    
def minibatch(*tensors, batch_size):
    """
    分批生成器

    将输入张量按批次大小分割，用于批量训练

    Args:
        *tensors: 输入张量（多个）
        batch_size: 批量大小

    Yields:
        tuple: 批量数据
    """
    n_samples = tensors[0].shape[0]

    for i in range(0, n_samples, batch_size):
        yield tuple(tensor[i:i+batch_size] for tensor in tensors)
        
def shuffle(*arrays):
    """
    打乱数据顺序

    Args:
        *arrays: 输入数组（多个）

    Returns:
        tuple: 打乱后的数组
    """
    # 获取随机索引
    indices = np.random.permutation(len(arrays[0]))

    # 打乱所有数组
    return tuple(array[indices] for array in arrays)

class timer:
    """
    计时上下文管理器

    用于测量代码块的执行时间

    使用示例:
        with timer(name="Sample"):
            do_something()
        print(timer.dict())  # 输出: |Sample:0.12|
    """
    from time import time
    TAPE = [-1]  # 全局时间记录
    NAMED_TAPE = {}

    @staticmethod
    def get():
        """获取最后一次计时结果"""
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        """获取所有命名计时结果"""
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        """清零计时器"""
        if select_keys is None:
            for key in timer.NAMED_TAPE:
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        """初始化计时器"""
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE.get(kwargs['name'], 0.)
            self.named = kwargs['name']
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        """进入上下文，开始计时"""
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，结束计时"""
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

def RecallPrecision_ATk(rating_np, test_data, k):
    """
    计算 Recall@K 和 Precision@K

    Args:
        rating_np: 评分矩阵 (n_users, n_items)
        test_data: 测试集字典 {user: [items]}
        k: Top-K

    Returns:
        tuple: (recall, precision)
    """
    # 用户总数
    n_users = rating_np.shape[0]

    # 统计
    hit = 0
    precision_sum = 0
    recall_sum = 0

    for user in range(n_users):
        # 测试集物品
        test_items = test_data.get(user, [])

        if len(test_items) == 0:
            continue

        # Top-K 推荐
        _, top_k = torch.topk(
            torch.from_numpy(rating_np[user]),
            k=k
        )
        top_k = top_k.numpy()

        # 命中
        hit_items = set(top_k) & set(test_items)
        hit += len(hit_items)

        # Recall
        recall_sum += len(hit_items) / len(test_items)

        # Precision
        precision_sum += len(hit_items) / k

    # 平均
    recall = hit / sum(len(test_data.get(u, [])) for u in range(n_users))
    precision = precision_sum / n_users

    return recall, precision


def NDCGatK_r(rating_np, test_data, k):
    """
    计算 NDCG@K

    Args:
        rating_np: 评分矩阵
        test_data: 测试集字典
        k: Top-K

    Returns:
        float: NDCG@K
    """
    n_users = rating_np.shape[0]

    ndcg = 0.0

    for user in range(n_users):
        test_items = test_data.get(user, [])

        if len(test_items) == 0:
            continue

        # Top-K 推荐
        _, top_k = torch.topk(
            torch.from_numpy(rating_np[user]),
            k=k
        )
        top_k = top_k.numpy()

        # 计算 DCG
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in test_items:
                dcg += 1.0 / np.log2(i + 2)

        # 计算 IDCG
        idcg = 0.0
        for i in range(min(len(test_items), k)):
            idcg += 1.0 / np.log2(i + 2)

        # NDCG
        if idcg > 0:
            ndcg += dcg / idcg

    return ndcg / n_users

def set_seed(seed):
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_checkpoint_path(config):
    """
    生成 checkpoint 路径

    Args:
        config: 配置字典

    Returns:
        str: 模型保存路径
    """
    path = config.get('path', './checkpoints')

    # 文件名：模型_数据集_维度_层数.pt
    filename = f"{config['model']}_{config['dataset']}_"
    filename += f"{config['recdim']}_{config['layer']}.pt"

    return os.path.join(path, filename)

def dynamic_sampling(model, dataset, epoch, max_epochs):
    """
    动态采样

    采样比例随训练进度变化

    Args:
        model: 模型对象
        dataset: 数据集对象
        epoch: 当前轮数
        max_epochs: 总轮数

    Returns:
        float: 采样比例
    """
    # 采样比例随训练进度变化
    ratio = 1.0 + epoch / max_epochs

    return min(ratio, 5.0)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)