import re
import numpy as np
np.set_printoptions(threshold=np.inf)


def get_data():
    # 从文件中读取数据并转换为三维列表格式
    input_file = 'data/data_64x64x2.txt'

    all_data = []#存放所有数据
    single_file_data = []#存放一个文件的数据
    i = 0#计数，计每个文件的行个数，每4096行保存一次
    with open(input_file, 'r') as f:
        lines = f.readlines()

        # 匹配浮点数和科学计数法表示的浮点数的正则表达式
        pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

        matches = re.findall(pattern, str(lines))

        result = [float(match) for match in matches]

        for a in range(len(result)):
            if a==0 or a%(4096*2)!=0:
                single_file_data.append(result[a])

            else:
                # print(len(single_file_data))#8192
                single_file_data = np.array(single_file_data).reshape(64,64,2)
                all_data.append(single_file_data)
                single_file_data=[]
                single_file_data.append(result[a])
        single_file_data = np.array(single_file_data).reshape(64, 64, 2)
        all_data.append(single_file_data)
        all_data = np.array(all_data)
        print(all_data.shape)


    return all_data
all_data  = get_data()

# # 将数据写入文件，格式只能为二进制,二进制数据无法查看，想查看的话可以将其写入到txt里面或者用deebug查看
# np.save('./data/all_data.npy', all_data)
# #直接加载npy文件为numpy格式
# all_data = np.load('./data/all_data.npy')

print(all_data.shape)

with open("./data/formatted_data.txt", "w", encoding="utf-8") as f:
    f.write(str(all_data))
    f.close()


