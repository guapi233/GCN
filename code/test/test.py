import torch
import numpy as np
import world
from model import LightGCN
from dataloader import LastFM
from Procedure import Test

# 加载数据集
dataset = LastFM()

# 创建模型
model = LightGCN(world.config, dataset)
model = model.to(world.device)
world.config['topks'] = [1, 2, 3, 4, 5]

# 加载训练好的权重（如果有）
# model.load_state_dict(torch.load('checkpoint.pt'))

# 测试
print("开始测试...")
result = Test(model, dataset, world.config)

# 汇总结果
print("\n测试结果汇总:")
for i, k in enumerate(world.config['topks']):
    print(f"Top-{k}:")
    print(f"  Recall:    {result['recall'][i]:.4f}")
    print(f"  Precision: {result['precision'][i]:.4f}")
    print(f"  NDCG:      {result['ndcg'][i]:.4f}")