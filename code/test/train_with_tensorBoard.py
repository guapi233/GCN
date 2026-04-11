import world
import model
import dataloader
import utils
from Procedure import BPR_train_original
import torch
from torch.utils.tensorboard import SummaryWriter
from Procedure import BPR_train_original, Test


# 初始化 writer
writer = SummaryWriter('../runs/lightgcn_experiment')

world.config['lr'] = 0.001
world.config['decay'] = 1e-4
world.config['bpr_batch_size'] = 2048
world.config['epochs'] = 10
world.config['topks'] = [1, 5, 10, 20]

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
    # 训练（传入 writer）
    output = BPR_train_original(
        dataset, Recmodel, bpr, epoch, neg_k=1, w=writer
    )
    
    # 每10轮测试并记录指标
    if (epoch + 1) % 10 == 0:
        result = Test(Recmodel, dataset, world.config)
        writer.add_scalar('Recall@20', result['recall'][0], epoch)
        writer.add_scalar('NDCG@20', result['ndcg'][0], epoch)

writer.close()
print("训练完成！")