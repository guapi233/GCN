import world
import numpy as np
import torch
import utils
import dataloader
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    """
    BPR 训练一个 epoch

    执行完整的训练流程：采样 → 分批次 → 前向传播 → 反向传播

    Args:
        dataset: 数据集对象
        recommend_model: 推荐模型
        loss_class: BPRLoss 损失类实例
        epoch: 当前轮次
        neg_k: 负采样比例（默认 1）
        w: TensorBoard 写入器（可选）

    Returns:
        str: 训练结果字符串，包含损失和时间信息
    """
    Recmodel = recommend_model
    Recmodel.train()  # 将模型设置为训练模式（启用 dropout 等训练层）
    bpr: utils.BPRLoss = loss_class  # 简化损失函数变量名

    # 采样负样本，并计时
    with timer(name="Sample"):
        S = utils.UniformSample_Python(dataset)  # 调用均匀采样函数

    print(f"使用 {len(S)} 个样本进行训练")

    # 将采样结果转换为 Tensor，并指定为长整型（适配 PyTorch 索引）
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    # 将数据迁移到配置的设备（GPU/CPU）上
    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)

    # 打乱数据顺序，避免训练过拟合
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    # 计算总批次数
    total_batch = len(users) // world.config['bpr_batch_size'] + 1

    # 初始化平均损失值
    aver_loss = 0.

    # 分批次训练
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
        utils.minibatch(
            users, posItems, negItems,
            batch_size=world.config['bpr_batch_size']
        )
    ):
        # 调用 BPR 损失的 stageOne 方法，计算当前批次的损失值
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri

        # 如果启用了 tensorboard 可视化，记录损失值
        if world.tensorboard and w is not None:
            w.add_scalar(
                f'BPRLoss/BPR',
                cri,
                epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i
            )

    # 计算平均损失（总损失/批次数）
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()

    return f"损失{aver_loss:.3f}-{time_info}"