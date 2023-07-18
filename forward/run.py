from net import Net
from forward import printStr

# 当模块被直接执行时，执行 printStr() 函数
if __name__ == "__main__":
    # 测试数据
    import torch
    data = torch.rand(2, 64, 64)

    result = printStr(data)
    print(result)
