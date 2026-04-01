'''
全局配置管理模块
'''

import os
import torch
from parse import parse_args
from os.path import join
import multiprocessing

# 设置环境变量，解决 Windows/Linux 下 KMP 库重复加载的报错问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 执行参数解析
args = parse_args()

# 定义路径
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))  # 项目根目录
CODE_PATH = join(ROOT_PATH, 'code')                    # 代码目录
DATA_PATH = join(ROOT_PATH, 'data')                    # 数据目录
BOARD_PATH = join(ROOT_PATH, 'runs')                  # TensorBoard 日志目录
FILE_PATH = join(ROOT_PATH, 'checkpoints')             # 模型 checkpoint 保存路径

# 创建必要的目录
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)
if not os.path.exists(BOARD_PATH):
    os.makedirs(BOARD_PATH, exist_ok=True)

# 初始化配置字典
config = {}

# 支持的数据集
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']

# 支持的模型列表
all_models = ['mf', 'lgn']

# 训练配置
config['bpr_batch_size'] = args.bpr_batch          # BPR 训练批量大小
config['latent_dim_rec'] = args.recdim              # 嵌入维度
config['lightGCN_n_layers'] = args.layer           # LightGCN 层数
config['dropout'] = args.dropout                    # 是否使用 Dropout
config['keep_prob'] = args.keepprob                 # Dropout 保留概率
config['A_n_fold'] = args.a_fold                   # 邻接矩阵分片数量
config['test_u_batch_size'] = args.testbatch       # 测试时的用户批量大小
config['multicore'] = args.multicore               # 是否使用多进程
config['lr'] = args.lr                             # 学习率
config['decay'] = args.decay                       # L2 正则化系数
config['pretrain'] = args.pretrain                 # 是否使用预训练权重

# 高级配置
config['A_split'] = False    # 是否对邻接矩阵进行拆分（大规模数据优化）
config['bigdata'] = False    # 是否处理超大规模数据

# 自动检测最佳可用设备
if torch.cuda.is_available():
    # 检测 CUDA 并指定设备索引
    device = torch.device('cuda:0')
    print(f'✅ 使用 CUDA 加速: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    # 适配 Apple Silicon (M1/M2/M3)
    device = torch.device('mps')
    print('✅ 使用 MPS 加速 (Apple Silicon)')
else:
    #  CPU 兜底，提示计算速度可能较慢
    device = torch.device('cpu')
    print('⚠️ 未检测到 GPU 加速，正在使用 CPU 计算（训练/推理可能较慢）')
    
# 并行配置
CORES = multiprocessing.cpu_count() // 2

# 随机种子
seed = args.seed

# 数据集与模型
dataset = args.dataset
model_name = args.model

# 参数校验
if dataset not in all_dataset:
    raise NotImplementedError(f"暂不支持数据集 '{dataset}'，请选择: {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"暂不支持模型 '{model_name}'，请选择: {all_models}")

# 训练配置
TRAIN_epochs = args.epochs          # 训练轮数
LOAD = args.load                   # 是否加载预训练模型
PATH = args.path                   # 模型路径

# 评估配置
topks = eval(args.topks)            # Top-K 列表，如 [20]

# TensorBoard 配置
tensorboard = args.tensorboard      # 是否启用 TensorBoard
comment = args.comment             # 实验备注

# 忽略 Pandas 的 FutureWarning
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

# 彩色打印（黄色背景）
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
 
if __name__ == '__main__':
    # 测试配置是否正确加载
    print("=== 配置测试 ===")
    print(f"数据集: {dataset}")
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"嵌入维度: {config['latent_dim_rec']}")
    print(f"层数: {config['lightGCN_n_layers']}")
    print(f"学习率: {config['lr']}")
    print(f"批量大小: {config['bpr_batch_size']}")
    print(f"随机种子: {seed}")
    print(f"训练轮数: {TRAIN_epochs}")
    print(f"Top-K: {topks}")
    print(f"TensorBoard: {tensorboard}")
    print("=" * 30)