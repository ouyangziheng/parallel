#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <thread>
#include <future>
#include "hnswlib/hnswlib/hnswlib.h"
#include "hnsw_search.h" // 引入hnsw_search.h来使用其中的函数和变量

// 删除重复定义的全局变量
// 删除重复定义的prepare_hnsw_index函数
// 删除重复定义的release_hnsw_index函数

/**
 * 简化版并行HNSW搜索算法实现
 * 这个版本采用更简单的并行策略，确保正确性
 */
std::priority_queue<std::pair<float, uint32_t>> hnsw_parallel_search(
    float* base,                // 基础数据集
    float* query,               // 查询向量
    size_t base_number,         // 数据集大小
    size_t vecdim,              // 向量维度
    size_t k,                   // 返回k个最近邻
    size_t ef = 100,            // ef参数，控制搜索精度
    size_t num_threads = 4      // 使用的线程数
) {
    // 检查索引是否已经加载
    if (g_hnsw_index == nullptr) {
        // 索引未加载，先加载索引
        prepare_hnsw_index(base, base_number, vecdim, ef);
    }
    
    // 先使用原始HNSW算法找到候选集合
    // 这一步确保我们有一组正确的候选节点
    auto initial_results = g_hnsw_index->searchKnn(query, std::max(ef, k*2));
    
    // 转换成向量方便并行处理
    std::vector<std::pair<float, uint32_t>> candidates;
    while (!initial_results.empty()) {
        candidates.push_back(initial_results.top());
        initial_results.pop();
    }
    
    // 如果候选数少于k，直接返回结果
    if (candidates.size() <= k) {
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (auto& pair : candidates) {
            result.push(pair);
        }
        return result;
    }
    
    // 并行重新计算距离，并选择最近的k个点
    std::vector<float> distances(candidates.size());
    
    // 计算每个线程处理的候选点数量
    size_t candidates_per_thread = (candidates.size() + num_threads - 1) / num_threads;
    
    // 创建线程向量
    std::vector<std::thread> threads;
    
    // 启动多个线程并行计算距离
    for (size_t t = 0; t < num_threads; t++) {
        size_t start_idx = t * candidates_per_thread;
        size_t end_idx = std::min(start_idx + candidates_per_thread, candidates.size());
        
        // 如果没有元素需要处理，跳过此线程
        if (start_idx >= end_idx) continue;
        
        threads.emplace_back([&, start_idx, end_idx]() {
            for (size_t i = start_idx; i < end_idx; i++) {
                // 获取候选点ID
                uint32_t candidate_id = candidates[i].second;
                
                // 获取候选点数据
                const void* candidate_data = g_hnsw_index->getDataByInternalId(candidate_id);
                
                // 计算距离
                distances[i] = g_hnsw_index->fstdistfunc_(
                    query,
                    candidate_data,
                    g_hnsw_index->dist_func_param_
                );
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // 创建候选点ID、距离的索引对，用于排序
    std::vector<std::pair<size_t, float>> index_dist_pairs(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++) {
        index_dist_pairs[i] = {i, distances[i]};
    }
    
    // 根据距离对候选点进行排序（升序，最近的在前面）
    std::sort(index_dist_pairs.begin(), index_dist_pairs.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });
    
    // 取前k个最近的点构建结果
    std::priority_queue<std::pair<float, uint32_t>> result;
    for (size_t i = 0; i < std::min(k, index_dist_pairs.size()); i++) {
        size_t idx = index_dist_pairs[i].first;
        result.emplace(distances[idx], candidates[idx].second);
    }
    
    return result;
} 