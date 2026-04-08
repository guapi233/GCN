import world
import model
import dataloader
import utils
from Procedure import BPR_train_original
import torch


world.config['lr'] = 0.001
world.config['decay'] = 1e-4
world.config['bpr_batch_size'] = 2048
world.config['epochs'] = 1000

world.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化数据集
dataset = dataloader.Loader(world.config)

# 初始化模型
Recmodel = model.LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)

# 创建 BPR 损失类（包含优化器）
bpr = utils.BPRLoss(Recmodel, world.config)

# 训练轮数
epochs = world.config['epochs']

print(f"开始训练，共 {epochs} 轮")

# 训练循环
for epoch in range(epochs):
    # 训练一个 epoch
    output = BPR_train_original(
        dataset, Recmodel, bpr, epoch, neg_k=1, w=None
    )
    print(f"第 {epoch+1}/{epochs} 轮: {output}")

print("训练完成！")