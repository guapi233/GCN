"""
测试项目初始化
"""
import sys
import os

from parse import parse_args
import world

print("=" * 50)
print("项目初始化测试")
print("=" * 50)

# 测试 1: 参数解析
print("\n[1] 测试参数解析...")
args = parse_args()
print(f"✓ 参数解析成功")
print(f"  数据集: {args.dataset}")
print(f"  模型: {args.model}")

# 测试 2: 配置加载
print("\n[2] 测试配置加载...")
print(f"✓ 配置加载成功")
print(f"  设备: {world.device}")
print(f"  嵌入维度: {world.config['latent_dim_rec']}")
print(f"  学习率: {world.config['lr']}")

# 测试 3: 路径检查
print("\n[3] 测试路径检查...")
print(f"✓ 项目根目录: {world.ROOT_PATH}")
print(f"  代码目录: {world.CODE_PATH}")
print(f"  数据目录: {world.DATA_PATH}")
print(f"  日志目录: {world.BOARD_PATH}")
print(f"  Checkpoint目录: {world.FILE_PATH}")

# 测试 4: 库导入
print("\n[4] 测试库导入...")
import torch
import numpy as np
import pandas as pd
import scipy as sp
from torch.utils.tensorboard import SummaryWriter

print(f"✓ 所有库导入成功")
print(f"  PyTorch: {torch.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  SciPy: {sp.__version__}")

print("\n" + "=" * 50)
print("✓ 项目初始化测试通过！")
print("=" * 50)