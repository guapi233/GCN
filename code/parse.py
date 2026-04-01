'''
命令行参数解析模块
'''

import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LightGCN: Lightweight Graph Convolutional Network for Recommendation")

    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="数据集选择: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="模型权重保存路径")

    # 模型相关参数
    parser.add_argument('--model', type=str, default='lgn',
                        help="模型选择: [mf, lgn] (mf=矩阵分解, lgn=LightGCN)")
    parser.add_argument('--recdim', type=int, default=64,
                        help="嵌入维度（用户/物品向量长度）")
    parser.add_argument('--layer', type=int, default=3,
                        help="LightGCN 的图卷积层数")

    # 训练相关参数
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="BPR 训练的批量大小")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="测试时的用户批量大小")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="训练轮数")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="学习率")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="L2 正则化系数（权重衰减）")
    parser.add_argument('--dropout', type=int, default=0,
                        help="是否使用 Dropout (0=否, 1=是)")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="Dropout 的保留概率")

    # 优化相关参数
    parser.add_argument('--a_fold', type=int, default=100,
                        help="邻接矩阵分片数量（用于大规模数据内存优化）")
    parser.add_argument('--multicore', type=int, default=0,
                        help="测试时是否使用多进程 (0=否, 1=是)")
    parser.add_argument('--pretrain', type=int, default=0,
                        help="是否使用预训练权重 (0=否, 1=是)")

    # 评估相关参数
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="Top-K 评估列表，如 '[10, 20]'")

    # 其他参数
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="是否启用 TensorBoard (0=否, 1=是)")
    parser.add_argument('--comment', type=str, default="lgn",
                        help="实验备注（用于 TensorBoard 日志区分）")
    parser.add_argument('--load', type=int, default=0,
                        help="是否加载预训练模型 (0=否, 1=是)")
    parser.add_argument('--seed', type=int, default=2020,
                        help="随机种子（确保实验可复现）")

    return parser.parse_args()