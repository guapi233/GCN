import torch
import world
from model import LightGCN
from dataloader import LastFM

# 加载数据集
dataset = LastFM()

# 创建模型配置
config = {
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'dropout': 0,
    'keep_prob': 0.6,
    'A_split': False,
    'pretrain': 0,
}

# 创建模型
model = LightGCN(config, dataset)

# 测试参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"用户嵌入: {model.num_users} × {model.latent_dim}")
print(f"物品嵌入: {model.num_items} × {model.latent_dim}")

# 测试 computer()
users_emb, items_emb = model.computer()
print(f"用户嵌入形状: {users_emb.shape}")
print(f"物品嵌入形状: {items_emb.shape}")

# 测试 getUsersRating()
rating = model.getUsersRating(torch.LongTensor([0, 1, 2]))
print(f"评分矩阵形状: {rating.shape}")

# 测试 bpr_loss()
users = torch.LongTensor([0, 1, 2])
pos = torch.LongTensor([10, 20, 30])
neg = torch.LongTensor([100, 200, 300])

loss, reg_loss = model.bpr_loss(users, pos, neg)
print(f"BPR 损失: {loss.item():.4f}")
print(f"正则化损失: {reg_loss.item():.4f}")