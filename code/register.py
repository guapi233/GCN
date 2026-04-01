'''
模型与数据集注册器

使用注册器模式实现动态加载，便于扩展新模型和数据集
'''

import world
import dataloader
import model
from pprint import pprint

# 根据数据集名称加载数据集
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']: ## 这三个数据集合lastfm有区别，需要分析下，看能否改进下
    dataset = dataloader.Loader(path=f"../data/{world.dataset}")
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
else:
    raise ValueError(f"未知的数据集: {world.dataset}")

# 模型注册器（模型名称 -> 模型类）(后续统一迁移到world)
MODELS = {
    'mf': model.PureMF,      # 矩阵分解
    'lgn': model.LightGCN    # LightGCN
}

# 根据模型名称获取模型实例
def get_model(model_name, config, dataset):
    if model_name not in MODELS:
        raise ValueError(f"未知的模型: {model_name}，支持: {list(MODELS.keys())}")
    return MODELS[model_name](config, dataset)


if __name__ == '__main__':
    # 测试注册器（需要先实现 dataloader 和 model）
    print('=========== 配置 ============')
    print(world.config)
    print("测试核心数:", world.CORES)
    print("实验备注:", world.comment)
    print("TensorBoard:", world.tensorboard)
    print("加载预训练:", world.LOAD)
    print("权重路径:", world.PATH)
    print("测试 Top-K:", world.topks)
    print("使用 BPR 损失")
    print('=========== 结束 ============')