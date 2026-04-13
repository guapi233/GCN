# test_data.py - 验证数据加载
import world
import dataloader

def test_dataloader():
    """测试数据加载器"""
    print("测试数据加载...")
    
    # 加载数据集
    dataset = dataloader.Loader(path="../data/gowalla")
    
    print(f"用户数: {dataset.n_users}")
    print(f"物品数: {dataset.m_items}")
    print(f"训练样本: {dataset.trainDataSize}")
    print(f"测试样本: {dataset.testDataSize}")
    
    # 验证图结构
    print(f"\n正在构建稀疏图...")
    dataset.Graph = dataset.getSparseGraph()
    print(f"稀疏图形状: {dataset.Graph.shape}")
    print(f"图非零元素: {dataset.Graph._nnz() if hasattr(dataset.Graph, '_nnz') else 'N/A'}")
    
    # 验证测试集
    print(f"\n测试用户: {len(dataset.testDict)}")
    
    print("\n✅ 数据加载测试通过!")
    return True


if __name__ == '__main__':
    test_dataloader()