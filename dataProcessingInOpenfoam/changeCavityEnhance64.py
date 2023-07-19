import os
import re

"""
功能:
将kolmogorov文件夹下第一层文件夹中的所有U文件中第14行中的数字50改为0，
第22行的4096改为262144，同时将24行到4119行（包括4119行）的每一行复制成64行，
即复制的4096行放大64倍变成262144行
"""
# 定义要处理的文件夹路径
folder_path = "kolmogorov_64"
a = 0
# 遍历第一层文件夹
for root, dirs, files in os.walk(folder_path):
    for folder in dirs:
        # 获取文件夹的路径
        folder_path = os.path.join(root, folder)

        # 遍历文件夹中的所有U文件
        for filename in os.listdir(folder_path):
            if filename=="U" or filename=="p":
                file_path = os.path.join(folder_path, filename)

                # 打开文件进行逐行替换和复制操作
                with open(file_path, "r") as file:
                    lines = file.readlines()

                tmp_str = ""#存放一行512个元素，64个矩阵每个矩阵的行复制八个。到时候列再复制八份，就是将tmp_list添加八次即可
                with open(file_path, "w") as file:
                    for i, line in enumerate(lines):
                        if i == 13:  # 第14行
                            line = re.sub(r'\d+', '0', line)  # 将行中的所有数字替换为0
                            file.write(line)
                        elif i == 21:  # 第22行
                            line = line.replace("4096", "262144")
                            file.write(line)
                        elif 23 <= i <= 4118:  # 第24行到第4119行（包括）
                            line = line * 8
                            tmp_str+=line
                            if (i - 22) % 64 == 0:#新的一行，一行共64个矩阵，每行的第一个元素
                                file.write(tmp_str*8)
                                tmp_str = ""
                        else:
                            file.write(line)

