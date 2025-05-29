#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p perf_results

# 创建结果文件
echo "M,EF_Construction,EF_Search,Recall,Latency_us" > hnsw_results.csv

# M值列表 (HNSW图中每个节点的最大连接数)
M_VALUES=(4 6 8 16 24) 

# EF_Construction值列表 (构建索引时使用的ef参数)
EF_CONSTRUCTION_VALUES=(100 150 200)

# EF_Search值列表 (搜索时使用的ef参数)
EF_SEARCH_VALUES=(50 100 200)

# 对每个参数组合进行测试
for m in "${M_VALUES[@]}"; do
  for ef_construction in "${EF_CONSTRUCTION_VALUES[@]}"; do
    for ef_search in "${EF_SEARCH_VALUES[@]}"; do
      echo "测试参数: M=$m, EF_Construction=$ef_construction, EF_Search=$ef_search"
      
      # 删除旧索引文件，确保重新构建
      rm -f files/hnsw.index
      
      # 修改hnsw_search.h中的ef参数
      sed -i "s/const int efConstruction = [0-9]*;/const int efConstruction = $ef_construction;/" hnsw_search.h
      sed -i "s/const int M = [0-9]*;/const int M = $m;/" hnsw_search.h
      
      # 修改main.cc中的ef_search参数
      sed -i "s/size_t ef_search = [0-9]*;/size_t ef_search = $ef_search;/" main.cc
      echo "test.sh ann 2 1"
      output=$(bash test.sh ann 2 1)
      
      # 保存完整输出
      echo "$output" > perf_results/hnsw_M${m}_EFC${ef_construction}_EFS${ef_search}.txt
      
      # 提取召回率和延迟时间
      recall=$(echo "$output" | grep "平均召回率" | awk '{print $2}')
      latency=$(echo "$output" | grep "平均查询延迟" | awk '{print $3}')
      
      # 保存结果
      echo "$m,$ef_construction,$ef_search,$recall,$latency" >> hnsw_results.csv
      
      # 打印当前结果
      echo "M=$m EF_Construction=$ef_construction EF_Search=$ef_search 召回率: $recall 延迟: $latency 微秒"
      echo "----------------------------------------"
    done
  done
done

# 分析结果并创建简单报告
echo "生成HNSW结果分析..."

# 按召回率排序并显示结果
echo "结果按召回率排序:"
sort -t, -k4,4nr hnsw_results.csv | head -n 10

# 按延迟时间排序并显示结果
echo "结果按延迟时间排序:"
sort -t, -k5,5n hnsw_results.csv | head -n 10

echo "测试完成，详细结果保存在 hnsw_results.csv 文件中"

# 生成一个简单的分析摘要
echo "HNSW性能和精度平衡的最佳配置建议:" > hnsw_performance_summary.txt
echo "1. 最高召回率配置:" >> hnsw_performance_summary.txt
sort -t, -k4,4nr hnsw_results.csv | head -n 1 | tee -a hnsw_performance_summary.txt

echo "2. 最低延迟配置:" >> hnsw_performance_summary.txt
sort -t, -k5,5n hnsw_results.csv | head -n 1 | tee -a hnsw_performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> hnsw_performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$4/$5}' hnsw_results.csv | sort -t, -k6,6nr | head -n 1 | tee -a hnsw_performance_summary.txt

echo "HNSW性能分析摘要已保存到 hnsw_performance_summary.txt" 