# test_model.py - 验证模型
import torch
import world
import dataloader
import model

def test_model():
    """测试模型前向传播"""
    print("测试模型...")
    
    # 加载数据
    dataset = dataloader.Loader(path="../data/gowalla")
    
    # 创建模型
    Recmodel = model.LightGCN(world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    print(f"模型创建成功")
    print(f"嵌入维度: {world.config['latent_dim_rec']}")
    print(f"图卷积层数: {world.config['lightGCN_n_layers']}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    users = torch.tensor([0, 1, 2]).to(world.device)
    pos = torch.tensor([100, 200, 300]).to(world.device)
    neg = torch.tensor([50, 150, 250]).to(world.device)
    
    loss, reg_loss = Recmodel.bpr_loss(users, pos, neg)
    print(f"BPR损失: {loss.item():.4f}")
    print(f"正则化损失: {reg_loss.item():.4f}")
    
    # 测试评分预测
    print("\n测试评分预测...")
    ratings = Recmodel.getUsersRating(users)
    print(f"评分矩阵形状: {ratings.shape}")
    print(f"评分范围: [{ratings.min():.4f}, {ratings.max():.4f}]")
    
    print("\n✅ 模型测试通过!")
    return True


if __name__ == '__main__':
    test_model()