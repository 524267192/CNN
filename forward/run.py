import random
from net import Net
from forward import printStr

# 当模块被直接执行时，执行 printStr() 函数
if __name__ == "__main__":
    # 测试数据
    data = [[random.random() for _ in range(2)] for _ in range(4096)]

    #调用函数接口
    result = printStr(data)

    # print(result)
    # print(len(result))
    # print(len(result[0]))
    # print(len(result[1]))
    # for sublist in result[1]:
    #     print(len(sublist))

