import numpy as np
from dataloader import LastFM
from utils import UniformSample_Python

# 加载数据集
dataset = LastFM()

print(f"数据集信息:")
print(f"  用户数: {dataset.n_users}")
print(f"  物品数: {dataset.m_items}")
print(f"  训练样本: {dataset.trainDataSize}")

# 测试采样
users, pos_items, neg_items = UniformSample_Python(dataset)

print(f"\n采样结果:")
print(f"  样本数: {len(users)}")
print(f"  用户范围: [{users.min()}, {users.max()}]")
print(f"  正样本范围: [{pos_items.min()}, {pos_items.max()}]")
print(f"  负样本范围: [{neg_items.min()}, {neg_items.max()}]")

# 验证负样本
print(f"\n验证负样本:")
for i in range(5):
    u, pos, neg = users[i], pos_items[i], neg_items[i]
    is_valid = neg not in dataset.allPos[u]
    print(f"  用户 {u}: 正样本 {pos}, 负样本 {neg}, 有效: {is_valid}")