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
from utils import RecallPrecision_ATk, NDCGatK_r

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

def test_one_batch(X):
    # 解包输入：排序后的推荐物品列表（numpy数组）
    sorted_items = X[0].numpy()
    # 解包输入：用户的真实交互物品列表
    groundTrue = X[1]
    # 生成标签：判断推荐物品是否在真实列表中（1表示命中，0表示未命中）
    r = utils.getLabel(groundTrue, sorted_items)

    # 初始化存储精度、召回率、NDCG的列表,计算指定K值下的精度和召回率
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        # 存储当前K值的精度，召回率，NDCG
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size'] # 获取测试时的用户批次大小配置
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # 获取测试集的用户-物品交互字典（key:用户ID，value:真实交互物品列表）
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()  # 将模型设置为评估模式（关闭dropout等训练层）
    max_K = max(world.topks) # 获取最大的Top-K值，用于统一截取推荐列表  
    
    # 如果启用多进程测试，初始化进程池
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    # 初始化评估结果字典，存储各Top-K下的平均指标
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}

    # 禁用梯度计算（测试阶段无需反向传播，节省显存）
    with torch.no_grad():
        users = list(testDict.keys()) # 获取测试集中所有用户ID列表
        
        # 断言：测试批次大小不超过用户总数的1/10，避免批次过大导致显存溢出
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        
        # 初始化存储用户批次、推荐评分、真实标签的列表
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1 # 计算测试总批次数
        for batch_users in utils.minibatch(np.array(users), batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users) # 获取当前批次用户的所有正样本（训练+验证），用于推荐时排除
            # 获取当前批次用户的测试集真实交互物品列表
            groundTrue = [testDict[u] for u in batch_users]
            # 将用户ID转换为Tensor并指定为长整型并迁移至设备
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            # 模型预测当前批次用户对所有物品的评分
            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            
            # 初始化需要排除的索引和物品列表（排除用户已交互的物品）
            exclude_index = []
            exclude_items = []
            # 遍历每个用户的已交互物品，生成排除索引
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items)) # 重复用户索引，匹配物品数量
                exclude_items.extend(items) # 收集用户已交互的物品ID
            
            # 将已交互物品的评分置为极小值，确保不会被推荐
            rating[exclude_index, exclude_items] = -(1<<10)
            # 截取评分最高的max_K个物品（按评分降序）
            _, rating_K = torch.topk(rating, k=max_K)
            # 释放评分Tensor的显存（仅保留Top-K结果）
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating

            # 存储当前批次的用户ID，Top-K推荐结果（迁移到CPU），真实交互物品列表
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        # 验证批次数是否匹配（防止数据遗漏）
        assert total_batch == len(users_list)
        
        # 将推荐结果和真实标签打包为迭代器
        X = zip(rating_list, groundTrue_list)
        # 多进程模式：使用进程池并行计算评估指标
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else: # 单进程模式：逐批次计算评估指标
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        
        # 计算批次缩放比例（用于平均指标）
        scale = float(u_batch_size/len(users))
        # 累加所有批次的评估指标，并计算最终的平均指标（除以用户总数）
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        # 如果启用tensorboard，记录测试指标
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        
        if multicore == 1: # 关闭多进程池（释放资源）
            pool.close()
        print(results)
        return results
