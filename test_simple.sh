#!/bin/bash

# 设置环境变量
export OMP_NUM_THREADS=16

# 确保目录存在
mkdir -p files

# 编译程序
echo "编译程序..."
g++ -std=c++11 -O3 -fopenmp -pthread main.cc -o simple_test

# 检查编译结果
if [ $? -eq 0 ]; then
  echo "编译成功，开始运行测试..."
  ./simple_test
  echo "测试完成！"
else
  echo "编译失败"
fi 