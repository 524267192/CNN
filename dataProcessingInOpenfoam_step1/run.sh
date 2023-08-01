#!/bin/bash

directory="$HOME/jie/app/OpenFOAM-v2212/run/code/all_data/"

# 切换到 all_data 目录
cd "${directory}"

# 遍历 all_data 目录下的所有文件夹
for folder in */; do
  folder_name="${folder%/}"
  screen_name="${folder_name}"

  # 创建新的 screen 会话，并执行 blockMesh
  screen -S "${screen_name}" bash -c "cd ${folder_name} && blockMesh &&  myicoFoam"

  # 退出 screen 会话
  screen -d
  #screen -S "${screen_name}" -X detach
done

