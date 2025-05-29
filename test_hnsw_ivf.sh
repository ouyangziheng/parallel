#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p perf_results

# 创建结果文件
echo "Nlist,Nprobe,EF_Construction,M,EF_Search,Recall,Latency_us" > ivf_hnsw_results.csv

# 参数列表
NLIST_VALUES=(128 256 512) # 簇的个数
NPROBE_VALUES=(16 32 64) # 搜索时检查的簇数量
EF_CONSTRUCTION_VALUES=(100 150 200) # 构建HNSW索引时使用的ef参数
M_VALUES=(8 16 24) # HNSW图中每个节点的最大连接数
EF_SEARCH_VALUES=(100 200 300) # 搜索时使用的ef参数

# 对每个参数组合进行测试
for nlist in "${NLIST_VALUES[@]}"; do
  for nprobe in "${NPROBE_VALUES[@]}"; do
    for ef_construction in "${EF_CONSTRUCTION_VALUES[@]}"; do
      for m in "${M_VALUES[@]}"; do
        for ef_search in "${EF_SEARCH_VALUES[@]}"; do
          echo "测试参数: Nlist=$nlist, Nprobe=$nprobe, EF_Construction=$ef_construction, M=$m, EF_Search=$ef_search"
          
          # 删除旧索引文件，确保重新构建
          rm -f files/*
          
          # 修改ivf_hnsw_search.h中的参数
          sed -i "s/const int efConstruction = [0-9]*;/const int efConstruction = $ef_construction;/" ivf_hnsw_search.h
          sed -i "s/const int M = [0-9]*;/const int M = $m;/" ivf_hnsw_search.h
          
          # 修改main.cc中的参数
          sed -i "s/const size_t nlist = [0-9]*;/const size_t nlist = $nlist;/" main.cc
          sed -i "s/const size_t nprobe = [0-9]*;/const size_t nprobe = $nprobe;/" main.cc
          sed -i "s/const size_t ef_search = [0-9]*;/const size_t ef_search = $ef_search;/" main.cc
          
          # 运行测试
          echo "执行测试: bash test.sh ann 2 1"
          output=$(bash test.sh ann 2 1)
          
          # 保存完整输出
          echo "$output" > perf_results/ivf_hnsw_N${nlist}_NP${nprobe}_EFC${ef_construction}_M${m}_EFS${ef_search}.txt
          
          # 提取召回率和延迟时间
          recall=$(echo "$output" | grep "平均召回率" | awk '{print $2}')
          latency=$(echo "$output" | grep "平均查询延迟" | awk '{print $3}')
          
          # 保存结果
          echo "$nlist,$nprobe,$ef_construction,$m,$ef_search,$recall,$latency" >> ivf_hnsw_results.csv
          
          # 打印当前结果
          echo "Nlist=$nlist Nprobe=$nprobe EF_Construction=$ef_construction M=$m EF_Search=$ef_search 召回率: $recall 延迟: $latency 微秒"
          echo "----------------------------------------"
        done
      done
    done
  done
done

# 分析结果并创建简单报告
echo "生成IVF+HNSW结果分析..."

# 按召回率排序并显示前10个结果
echo "结果按召回率排序:"
sort -t, -k6,6nr ivf_hnsw_results.csv | head -n 10

# 按延迟时间排序并显示前10个结果
echo "结果按延迟时间排序:"
sort -t, -k7,7n ivf_hnsw_results.csv | head -n 10

echo "测试完成，详细结果保存在 ivf_hnsw_results.csv 文件中"

# 生成一个简单的分析摘要
echo "IVF+HNSW性能和精度平衡的最佳配置建议:" > ivf_hnsw_performance_summary.txt
echo "1. 最高召回率配置:" >> ivf_hnsw_performance_summary.txt
sort -t, -k6,6nr ivf_hnsw_results.csv | head -n 1 | tee -a ivf_hnsw_performance_summary.txt

echo "2. 最低延迟配置:" >> ivf_hnsw_performance_summary.txt
sort -t, -k7,7n ivf_hnsw_results.csv | head -n 1 | tee -a ivf_hnsw_performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> ivf_hnsw_performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$6/$7}' ivf_hnsw_results.csv | sort -t, -k8,8nr | head -n 1 | tee -a ivf_hnsw_performance_summary.txt

echo "4. 高召回率(>0.95)中延迟最低的配置:" >> ivf_hnsw_performance_summary.txt
awk -F, 'NR>1 && $6>0.95 {print $0}' ivf_hnsw_results.csv | sort -t, -k7,7n | head -n 1 | tee -a ivf_hnsw_performance_summary.txt

echo "IVF+HNSW性能分析摘要已保存到 ivf_hnsw_performance_summary.txt"

