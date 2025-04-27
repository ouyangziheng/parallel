#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p perf_results

# 首先确保main.cc中不会构建重复索引
# 修改main.cc文件
sed -i 's/build_pq_index(base, base_number, vecdim);/\/\/ build_pq_index(base, base_number, vecdim);/' main.cc

# 创建结果文件
echo "M,K,Multiplier,Recall,Latency_us" > results.csv

# M值列表 (子空间数量)
M_VALUES=(4)

# K值列表 (聚类中心数量)
K_VALUES=(64)

# 乘数列表 (用于调整召回率)
MULTIPLIER_VALUES=(20)

# 对每个M、K和乘数组合进行测试
for m in "${M_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    for multiplier in "${MULTIPLIER_VALUES[@]}"; do
      echo "测试参数: M=$m, K=$k, 乘数=$multiplier"
      
      # 修改flat_scan_with_pq_njjl.h中的M和K值
      sed -i "s/size_t M = [0-9]*;.*\/\/ 使用.*子空间.*/size_t M = $m;   \/\/ 使用$m子空间作为精度和速度的平衡点/" flat_scan_with_pq_njjl.h
      sed -i "s/size_t K = [0-9]*;.*\/\/ 每个子空间的聚类中心数量/size_t K = $k;  \/\/ 每个子空间的聚类中心数量/" flat_scan_with_pq_njjl.h
      
      # 修改PQIndex构造函数中的默认M值
      sed -i "s/PQIndex(size_t d, size_t M = [0-9]*, size_t K = [0-9]*)/PQIndex(size_t d, size_t M = $m, size_t K = $k)/" flat_scan_with_pq_njjl.h
      
      # 修改乘数值来调整召回率
      sed -i "s/if (pq_results.size() < k \* [0-9]*) {/if (pq_results.size() < k \* $multiplier) {/" flat_scan_with_pq_njjl.h
      
      # 删除旧索引文件，确保重新构建
      rm -f files/pq_Inner_Product.index files/pq_Inner_Product.index.pq
      
      # 获取主要输出
      output=$(bash test.sh 2 1)

      echo "$output" > perf_results/full_output_M${m}_K${k}_multiplier${multiplier}.txt
      
      # 提取召回率和延迟时间
      recall=$(echo "$output" | grep "average recall" | awk '{print $3}')
      latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
      
      # 保存结果
      echo "$m,$k,$multiplier,$recall,$latency" >> results.csv
      
      # 打印当前结果
      echo "M=$m K=$k 乘数=$multiplier 召回率: $recall 延迟: $latency 微秒"
      echo "----------------------------------------"
    done
  done
done

# 分析结果并创建简单报告
echo "生成结果分析..."

# 按召回率排序并显示结果
echo "结果按召回率排序:"
sort -t, -k4,4nr results.csv | head -n 20

# 按延迟时间排序并显示结果
echo "结果按延迟时间排序:"
sort -t, -k5,5n results.csv | head -n 20

echo "测试完成，详细结果保存在 results.csv 文件中" 

# 生成一个简单的分析摘要
echo "性能和精度平衡的最佳配置建议:" > performance_summary.txt
echo "1. 最高召回率配置:" >> performance_summary.txt
sort -t, -k4,4nr results.csv | head -n 1 | tee -a performance_summary.txt

echo "2. 最低延迟配置:" >> performance_summary.txt
sort -t, -k5,5n results.csv | head -n 1 | tee -a performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$4/$5}' results.csv | sort -t, -k6,6nr | head -n 1 | tee -a performance_summary.txt

echo "性能分析摘要已保存到 performance_summary.txt" 