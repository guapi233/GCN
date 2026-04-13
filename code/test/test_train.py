# test_train.py - 快速训练验证
import world
import utils
import Procedure
from register import MODELS, dataset
import torch

def test_training():
    """测试训练流程（快速版本）"""
    print("测试训练流程...")
    
    # 设置小参数快速测试
    world.config['epochs'] = 5
    world.config['bpr_batch_size'] = 1024
    
    # 创建模型
    model_class = MODELS[world.model_name]
    Recmodel = model_class(world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    
    # 创建损失
    bpr = utils.BPRLoss(Recmodel, world.config)
    
    print(f"\n快速训练 {world.config['epochs']} 轮...")
    
    for epoch in range(world.config['epochs']):
        output = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=1, w=None
        )
        print(f"轮次 {epoch+1}: {output}")
    
    print("\n✅ 训练流程测试通过!")
    return True


if __name__ == '__main__':
    test_training()