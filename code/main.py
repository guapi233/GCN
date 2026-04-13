import world
import utils
import Procedure
from register import MODELS, dataset
from pprint import pprint
import torch
import numpy as np
import time
from os.path import join
from torch.utils.tensorboard import SummaryWriter

def main():
    """
    主训练入口

    完整的训练流程：初始化 → 训练循环 → 测试评估 → 保存模型
    """
    # ========== 初始化 ==========
    
    # 设置随机种子（保证可复现）
    utils.set_seed(world.seed)
    
    # 打印配置信息
    print('=========== 配置信息 ===========')
    pprint(world.config)
    print(f"使用数据集: {world.dataset}")
    print(f"使用设备: {world.device}")
    print(f"测试Top-K: {world.topks}")
    print('=========== 配置结束 ===========')
    
    # ========== 初始化模型 ==========
    
    # 从注册器获取模型类
    model_class = MODELS[world.model_name]
    
    # 创建模型实例
    Recmodel = model_class(world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    print(f"\n模型: {world.model_name}")
    print(f"可训练参数: {sum(p.numel() for p in Recmodel.parameters() if p.requires_grad):,}")
    
    # ========== 加载已有模型（断点续训）==========
    
    # 生成权重文件路径
    weight_file = utils.getFileName()
    
    if world.LOAD:
        # 尝试加载已有权重
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
            print(f"\n已加载模型: {weight_file}")
        except FileNotFoundError:
            print(f"\n未找到模型文件: {weight_file}，将从头训练")
    else:
        print(f"\n从头训练（不加载预训练模型）")
    
    # ========== 初始化训练组件 ==========
    
    # 创建 BPR 损失类（包含优化器）
    bpr = utils.BPRLoss(Recmodel, world.config)
    
    # 初始化 TensorBoard（如果启用）
    if world.tensorboard:
        w = SummaryWriter(
            join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
        )
    else:
        w = None
    
    # ========== 训练循环 ==========
    
    print(f"\n开始训练，共 {world.TRAIN_epochs} 轮")
    print("-" * 50)
    
    best_recall = 0.0
    early_stop_count = 0
    
    for epoch in range(world.TRAIN_epochs):
        # 记录开始时间
        start_time = time.time()
        
        # 训练一个 epoch
        output = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=1, w=w
        )
        
        # 计算 epoch 耗时
        epoch_time = time.time() - start_time
        
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output} | 耗时: {epoch_time:.2f}s")
        
        # ========== 定期测试 ==========
        
        # 每 10 轮测试一次
        if (epoch + 1) % 10 == 0:
            print("\n开始测试...")
            
            # 切换到评估模式
            Recmodel.eval()
            
            # 执行测试
            with torch.no_grad():
                results = Procedure.Test(dataset, Recmodel, epoch, w=w)
            
            # 切回训练模式
            Recmodel.train()
            
            # 获取当前最佳指标（以 Recall@20 为例）
            current_recall = results['recall'][0]  # 第一个 K 值对应的 recall
            
            print(f"测试结果 - Recall@20: {current_recall:.4f}")
            
            # 保存最佳模型
            if current_recall > best_recall:
                best_recall = current_recall
                torch.save(Recmodel.state_dict(), weight_file)
                print(f"保存最佳模型 (Recall@20={best_recall:.4f})")
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(f"未改善 ({early_stop_count}/{world.early_stop_patience})")
            
            print("-" * 50)
            
            # 早停检查
            if world.early_stop and early_stop_count >= world.early_stop_patience:
                print(f"\n早停！连续 {world.early_stop_patience} 轮无改善")
                break
    
    # ========== 训练结束 ==========
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳 Recall@20: {best_recall:.4f}")
    print(f"模型保存于: {weight_file}")
    
    # 关闭 TensorBoard
    if w is not None:
        w.close()


# ========== 程序入口 ==========

if __name__ == '__main__':
    main()