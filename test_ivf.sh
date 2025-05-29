#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p record

# 清空历史记录
rm -rf record/*

# 创建结果文件
echo "Nlist,Nprobe,Recall,Latency_us" > ivf_results.csv

# Nlist值列表 (聚类中心/簇的数量)
NLIST_VALUES=(4 8 16 32 64 128 256)

# Nprobe值列表 (搜索时检查的簇数量)
NPROBE_VALUES=(1 2 4 8 16 32 64)

# 对每个Nlist和Nprobe组合进行测试
for nlist in "${NLIST_VALUES[@]}"; do
  for nprobe in "${NPROBE_VALUES[@]}"; do
    echo "==============================================="
    echo "测试参数: Nlist=$nlist, Nprobe=$nprobe"
    echo "==============================================="
    
    # 修改main.cc中的nlist和nprobe值
    sed -i "s/const size_t nlist = [0-9]*;/const size_t nlist = $nlist;/" main.cc
    sed -i "s/const size_t nprobe = [0-9]*;/const size_t nprobe = $nprobe;/" main.cc
    
    # 确保索引文件被删除
    echo "删除旧索引文件..."
    rm -f files/*
    
    # 获取主要输出
    output=$(bash test.sh ann 2 1)

    echo "$output" > record/ivf_output_nlist${nlist}_nprobe${nprobe}.txt
    
    # 提取召回率和延迟时间
    recall=$(echo "$output" | grep "average recall" | awk '{print $3}')
    latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
    
    # 保存结果
    echo "$nlist,$nprobe,$recall,$latency" >> ivf_results.csv
    
    # 打印当前结果
    echo "Nlist=$nlist Nprobe=$nprobe 召回率: $recall 延迟: $latency 微秒"
    echo "----------------------------------------"
  done
done

# 分析结果并创建简单报告
echo "生成结果分析..."

# 按召回率排序并显示结果
echo "结果按召回率排序:"
sort -t, -k3,3nr ivf_results.csv | head -n 20

# 按延迟时间排序并显示结果
echo "结果按延迟时间排序:"
sort -t, -k4,4n ivf_results.csv | head -n 20

echo "测试完成，详细结果保存在 ivf_results.csv 文件中" 

# 生成一个简单的分析摘要
echo "IVF性能和精度平衡的最佳配置建议:" > ivf_performance_summary.txt
echo "1. 最高召回率配置:" >> ivf_performance_summary.txt
sort -t, -k3,3nr ivf_results.csv | head -n 1 | tee -a ivf_performance_summary.txt

echo "2. 最低延迟配置:" >> ivf_performance_summary.txt
sort -t, -k4,4n ivf_results.csv | head -n 1 | tee -a ivf_performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> ivf_performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$3/$4}' ivf_results.csv | sort -t, -k5,5nr | head -n 1 | tee -a ivf_performance_summary.txt

echo "IVF性能分析摘要已保存到 ivf_performance_summary.txt" 