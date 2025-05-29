#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p perf_results

# 首先确保main.cc中不会构建重复索引
# 修改main.cc文件
sed -i 's/build_ivfpq_pq_then_ivf_index(base, base_number, vecdim, nlist, nprobe, m, k_codebook);/\/\/ build_ivfpq_pq_then_ivf_index(base, base_number, vecdim, nlist, nprobe, m, k_codebook);/' main.cc
sed -i 's/build_ivfpq_ivf_then_pq_index(base, base_number, vecdim, nlist, nprobe, m, k_codebook);/\/\/ build_ivfpq_ivf_then_pq_index(base, base_number, vecdim, nlist, nprobe, m, k_codebook);/' main.cc

# 创建结果文件
echo "Strategy,Nlist,Nprobe,M,K,Recall,Latency_us" > ivfpq_results.csv

# 策略列表（先PQ后IVF或先IVF后PQ）
STRATEGIES=("PQ_THEN_IVF" "IVF_THEN_PQ")

# Nlist值列表 (聚类中心/簇的数量)
NLIST_VALUES=(100 200)

# Nprobe值列表 (搜索时检查的簇数量)
NPROBE_VALUES=(16 32)

# M值列表 (子空间数量)
M_VALUES=(4 8)

# K值列表 (每个子空间的聚类中心数量)
K_VALUES=(256)

# 对每种参数组合进行测试
for strategy in "${STRATEGIES[@]}"; do
  for nlist in "${NLIST_VALUES[@]}"; do
    for nprobe in "${NPROBE_VALUES[@]}"; do
      for m in "${M_VALUES[@]}"; do
        for k in "${K_VALUES[@]}"; do
          echo "测试参数: 策略=$strategy, Nlist=$nlist, Nprobe=$nprobe, M=$m, K=$k"
          
          # 修改main.cc中的参数值
          sed -i "s/const size_t nlist = [0-9]*;/const size_t nlist = $nlist;/" main.cc
          sed -i "s/const size_t nprobe = [0-9]*;/const size_t nprobe = $nprobe;/" main.cc
          sed -i "s/const size_t m = [0-9]*;/const size_t m = $m;/" main.cc
          sed -i "s/const size_t k_codebook = [0-9]*;/const size_t k_codebook = $k;/" main.cc
          
          # 确保使用IVFPQ索引而不是其他索引
          sed -i 's/auto search_result = flat_search_with_ivf_Inner_Product/\/\/ auto search_result = flat_search_with_ivf_Inner_Product/' main.cc
          sed -i 's/\/\/ auto search_result = flat_search_with_ivfpq_Inner_Product/auto search_result = flat_search_with_ivfpq_Inner_Product/' main.cc
          
          # 根据策略设置use_pq_then_ivf参数
          if [ "$strategy" = "PQ_THEN_IVF" ]; then
            sed -i 's/auto search_result = flat_search_with_ivfpq_Inner_Product(base, test_query + i\*vecdim, base_number, vecdim, k, nprobe, [a-z]*)/auto search_result = flat_search_with_ivfpq_Inner_Product(base, test_query + i\*vecdim, base_number, vecdim, k, nprobe, true)/' main.cc
            
            # 删除旧索引文件，确保重新构建
            rm -f files/ivfpq_pq_then_ivf.index
            
            # 修改main.cc，设置正确的索引构建函数
            sed -i 's/\/\/ build_ivfpq_pq_then_ivf_index/build_ivfpq_pq_then_ivf_index/' main.cc
          else  # IVF_THEN_PQ
            sed -i 's/auto search_result = flat_search_with_ivfpq_Inner_Product(base, test_query + i\*vecdim, base_number, vecdim, k, nprobe, [a-z]*)/auto search_result = flat_search_with_ivfpq_Inner_Product(base, test_query + i\*vecdim, base_number, vecdim, k, nprobe, false)/' main.cc
            
            # 删除旧索引文件，确保重新构建
            rm -f files/ivfpq_ivf_then_pq.index
            
            # 修改main.cc，设置正确的索引构建函数
            sed -i 's/\/\/ build_ivfpq_ivf_then_pq_index/build_ivfpq_ivf_then_pq_index/' main.cc
          fi
          
          # 获取主要输出
          output=$(bash test.sh ann 2 1)

          echo "$output" > perf_results/ivfpq_${strategy}_nlist${nlist}_nprobe${nprobe}_M${m}_K${k}.txt
          
          # 提取召回率和延迟时间
          recall=$(echo "$output" | grep "average recall" | awk '{print $3}')
          latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
          
          # 保存结果
          echo "$strategy,$nlist,$nprobe,$m,$k,$recall,$latency" >> ivfpq_results.csv
          
          # 打印当前结果
          echo "策略=$strategy Nlist=$nlist Nprobe=$nprobe M=$m K=$k 召回率: $recall 延迟: $latency 微秒"
          echo "----------------------------------------"
          
          # 注释掉构建索引部分，避免后续重复构建
          if [ "$strategy" = "PQ_THEN_IVF" ]; then
            sed -i 's/build_ivfpq_pq_then_ivf_index/\/\/ build_ivfpq_pq_then_ivf_index/' main.cc
          else
            sed -i 's/build_ivfpq_ivf_then_pq_index/\/\/ build_ivfpq_ivf_then_pq_index/' main.cc
          fi
        done
      done
    done
  done
done

# 分析结果并创建简单报告
echo "生成结果分析..."

# 按召回率排序并显示结果
echo "结果按召回率排序:"
sort -t, -k6,6nr ivfpq_results.csv | head -n 20

# 按延迟时间排序并显示结果
echo "结果按延迟时间排序:"
sort -t, -k7,7n ivfpq_results.csv | head -n 20

echo "测试完成，详细结果保存在 ivfpq_results.csv 文件中" 

# 生成一个简单的分析摘要
echo "IVFPQ性能和精度平衡的最佳配置建议:" > ivfpq_performance_summary.txt

echo "1. 最高召回率配置:" >> ivfpq_performance_summary.txt
sort -t, -k6,6nr ivfpq_results.csv | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "2. 最低延迟配置:" >> ivfpq_performance_summary.txt
sort -t, -k7,7n ivfpq_results.csv | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> ivfpq_performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$6/$7}' ivfpq_results.csv | sort -t, -k8,8nr | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "4. 各策略最佳配置:" >> ivfpq_performance_summary.txt
echo "   - PQ_THEN_IVF 策略最佳配置:" >> ivfpq_performance_summary.txt
grep "PQ_THEN_IVF" ivfpq_results.csv | awk -F, '{print $0","$6/$7}' | sort -t, -k8,8nr | head -n 1 | tee -a ivfpq_performance_summary.txt
echo "   - IVF_THEN_PQ 策略最佳配置:" >> ivfpq_performance_summary.txt
grep "IVF_THEN_PQ" ivfpq_results.csv | awk -F, '{print $0","$6/$7}' | sort -t, -k8,8nr | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "IVFPQ性能分析摘要已保存到 ivfpq_performance_summary.txt" 