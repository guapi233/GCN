# test_imports.py - 验证所有导入
"""
验证所有模块可正常导入
"""

def test_imports():
    """测试所有模块导入"""
    errors = []
    
    # 测试基础模块
    try:
        import world
        print("✅ world.py")
    except Exception as e:
        errors.append(f"❌ world.py: {e}")
    
    try:
        import parse
        print("✅ parse.py")
    except Exception as e:
        errors.append(f"❌ parse.py: {e}")
    
    # 测试数据加载
    try:
        import dataloader
        print("✅ dataloader.py")
    except Exception as e:
        errors.append(f"❌ dataloader.py: {e}")
    
    # 测试模型
    try:
        import model
        print("✅ model.py")
    except Exception as e:
        errors.append(f"❌ model.py: {e}")
    
    # 测试工具
    try:
        import utils
        print("✅ utils.py")
    except Exception as e:
        errors.append(f"❌ utils.py: {e}")
    
    # 测试流程
    try:
        import Procedure
        print("✅ Procedure.py")
    except Exception as e:
        errors.append(f"❌ Procedure.py: {e}")
    
    # 测试注册器
    try:
        import register
        print("✅ register.py")
    except Exception as e:
        errors.append(f"❌ register.py: {e}")
    
    if errors:
        print("\n导入错误:")
        for e in errors:
            print(e)
        return False
    
    print("\n✅ 所有模块导入成功!")
    return True


if __name__ == '__main__':
    test_imports()
