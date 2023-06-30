import os
import shutil

"""
功能：用于将"kolmogorov_64"中所有非0文件夹分别复制到"kolmogorov_512”中。
其中有多少个非零数据的文件夹，就创建多少个kolmogorov_512文件夹
"""
# 定义要复制的文件夹名称
source_folder_name = "kolmogorov_512"

# 定义存放目标文件夹的文件夹名称
destination_folder_name = "all_data"

# 获取当前工作目录
current_directory = os.getcwd()

# 创建存放目标文件夹的路径
destination_folder_path = os.path.join(current_directory, destination_folder_name)

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(destination_folder_path):
    os.mkdir(destination_folder_path)

# 获取名字为kolmogorov_64文件夹下子文件夹中为数字文件夹的名字
kolmogorov_64_folder_path = os.path.join(current_directory, "kolmogorov_64")
subfolders = [subfolder for subfolder in os.listdir(kolmogorov_64_folder_path) if os.path.isdir(os.path.join(kolmogorov_64_folder_path, subfolder)) and subfolder.isdigit()]

# 复制kolmogorov_512文件夹
for x in subfolders:
    source_folder_path = os.path.join(current_directory, source_folder_name)
    destination_folder_path_x = os.path.join(destination_folder_path, f"{source_folder_name}_{x}")

    # 如果目标文件夹已经存在，先删除再复制
    if os.path.exists(destination_folder_path_x) and os.path.isdir(destination_folder_path_x):
        shutil.rmtree(destination_folder_path_x)

    shutil.copytree(source_folder_path, destination_folder_path_x)


# 将kolmogorov_64文件夹下第一层文件夹中的所有U和p文件分别替换到kolmogorov_512_x文件夹中的0文件夹内
for root, dirs, files in os.walk(kolmogorov_64_folder_path):
    for folder in dirs:
        if folder != "0" and folder.isdigit():
            source_folder_path = os.path.join(root, folder)
            destination_folder_path_x = os.path.join(destination_folder_path, f"{source_folder_name}_{folder}")
            destination_folder_path_x_0 = os.path.join(destination_folder_path_x, "0")

            # 替换U文件
            for filename in os.listdir(source_folder_path):
                if filename=="U":
                    source_file_path = os.path.join(source_folder_path, filename)
                    destination_file_path = os.path.join(destination_folder_path_x_0, filename)
                    shutil.copy2(source_file_path, destination_file_path)

            # 替换p文件
            for filename in os.listdir(source_folder_path):
                if filename=="p":
                    source_file_path = os.path.join(source_folder_path, filename)
                    destination_file_path = os.path.join(destination_folder_path_x_0, filename)
                    shutil.copy2(source_file_path, destination_file_path)
