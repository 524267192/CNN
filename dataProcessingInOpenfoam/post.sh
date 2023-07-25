#!/bin/bash

directory="$HOME/jie/app/OpenFOAM-v2212/run/code/all_data/"

# 切换到 all_data 目录
cd "${directory}"

# 遍历 all_data 目录下的所有文件夹
for folder in */; do
  folder_name="${folder%/}"
  cd ${folder_name} && postProcess -func vorticity
  cd ..
done

