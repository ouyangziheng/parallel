#!/bin/bash

# 创建必要的目录
mkdir -p files
mkdir -p record

echo "========== IVF+PQ算法参数测试 ==========="

# 清空历史记录
rm -rf record/ivfpq_*

# 创建CSV结果文件
echo "Nlist,Nprobe,M,K,Strategy,Recall,Latency_us" > ivfpq_results.csv

# 参数列表
NLIST_VALUES=(32 64 128 256)            # 簇的数量
NPROBE_VALUES=(1 2 4 8 16 32 64)           # 搜索时检查的簇数量
M_VALUES=(1 2 4 8 16 32 64)                 # PQ子空间数量
K_VALUES=(1 2 4 8 16 32 64)                    # 每个子空间的聚类中心数量
STRATEGIES=("pq_first" "ivf_first") # 策略：先PQ后IVF或先IVF后PQ

# 测试所有参数组合
for nlist in "${NLIST_VALUES[@]}"; do
  for nprobe in "${NPROBE_VALUES[@]}"; do
    for m in "${M_VALUES[@]}"; do
      for k in "${K_VALUES[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
          echo "========================================================="
          echo "测试参数: Nlist=$nlist, Nprobe=$nprobe, M=$m, K=$k, 策略=$strategy"
          echo "========================================================="
          
          # 修改main.cc中的参数
          sed -i "s/const size_t nlist = [0-9]*;/const size_t nlist = $nlist;/" main.cc
          sed -i "s/const size_t nprobe = [0-9]*;/const size_t nprobe = $nprobe;/" main.cc
          sed -i "s/const size_t M = [0-9]*;/const size_t M = $m;/" main.cc
          sed -i "s/const size_t K = [0-9]*;/const size_t K = $k;/" main.cc
          
          # 设置策略
          if [ "$strategy" == "pq_first" ]; then
            sed -i "s/const bool use_pq_then_ivf = [a-z]*;/const bool use_pq_then_ivf = true;/" main.cc
            strat_name="先PQ后IVF"
          else
            sed -i "s/const bool use_pq_then_ivf = [a-z]*;/const bool use_pq_then_ivf = false;/" main.cc
            strat_name="先IVF后PQ"
          fi
          
          # 删除旧索引文件
          echo "删除旧索引文件..."
          rm -f files/*

          
          # 使用test.sh脚本执行测试，与test_ivf.sh保持一致
          result_file="record/ivfpq_nlist${nlist}_nprobe${nprobe}_M${m}_K${k}_${strategy}.txt"
          echo "运行程序，结果保存到 $result_file"
          
          # 执行测试脚本，参数与test_ivf.sh相同
          output=$(bash test.sh ann 2 1)
          echo "$output" > "$result_file"
          
          # 提取召回率和延迟时间
          recall=$(echo "$output" | grep "average recall" | awk '{print $3}')
          latency=$(echo "$output" | grep "average latency" | awk '{print $4}')
          
          # 检查是否成功获取到结果
          if [ -z "$recall" ] || [ -z "$latency" ]; then
            echo "警告: 未能从输出中提取结果，可能测试失败"
            echo "输出内容:"
            echo "$output"
            continue
          fi
          
          # 保存结果
          echo "$nlist,$nprobe,$m,$k,$strat_name,$recall,$latency" >> ivfpq_results.csv
          
          # 打印当前结果
          echo "Nlist=$nlist Nprobe=$nprobe M=$m K=$k 策略=$strat_name"
          echo "召回率: $recall 延迟: $latency 微秒"
          echo "---------------------------------------------------------"
        done
      done
    done
  done
done

# 分析结果并创建简单报告
echo "生成结果分析..."

# 按召回率排序并显示结果
echo "结果按召回率排序:"
sort -t, -k6,6nr ivfpq_results.csv | head -n 10

# 按延迟时间排序并显示结果
echo "结果按延迟时间排序:"
sort -t, -k7,7n ivfpq_results.csv | head -n 10

echo "测试完成，详细结果保存在 ivfpq_results.csv 文件中"

# 生成一个简单的分析摘要
echo "IVF+PQ性能和精度平衡的最佳配置建议:" > ivfpq_performance_summary.txt
echo "1. 最高召回率配置:" >> ivfpq_performance_summary.txt
sort -t, -k6,6nr ivfpq_results.csv | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "2. 最低延迟配置:" >> ivfpq_performance_summary.txt
sort -t, -k7,7n ivfpq_results.csv | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "3. 召回率-延迟平衡配置 (按召回率/延迟比值排序):" >> ivfpq_performance_summary.txt
# 创建临时文件计算比值
awk -F, 'NR>1 {print $0","$6/$7}' ivfpq_results.csv | sort -t, -k8,8nr | head -n 1 | tee -a ivfpq_performance_summary.txt

echo "IVF+PQ性能分析摘要已保存到 ivfpq_performance_summary.txt"

# 比较IVF和IVFPQ的性能
echo "比较IVF和IVFPQ性能..."

# 创建比较报告
echo "IVF vs IVF+PQ性能比较:" > ivf_vs_ivfpq.txt
echo "1. IVF基准 (nlist=256, nprobe=32):" >> ivf_vs_ivfpq.txt
echo "   召回率: $ivf_recall, 延迟: $ivf_latency 微秒" >> ivf_vs_ivfpq.txt
echo "2. 最佳IVFPQ召回率配置:" >> ivf_vs_ivfpq.txt
echo "   $best_ivfpq_recall" >> ivf_vs_ivfpq.txt
echo "3. 最佳IVFPQ延迟配置:" >> ivf_vs_ivfpq.txt
echo "   $best_ivfpq_latency" >> ivf_vs_ivfpq.txt
echo "4. 最佳IVFPQ平衡配置:" >> ivf_vs_ivfpq.txt
echo "   $best_ivfpq_balance" >> ivf_vs_ivfpq.txt

# 显示总结
echo "IVF vs IVF+PQ性能比较已保存到 ivf_vs_ivfpq.txt"
cat ivf_vs_ivfpq.txt 