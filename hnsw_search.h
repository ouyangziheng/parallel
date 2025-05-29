#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include "hnswlib/hnswlib/hnswlib.h"

// 声明HNSW索引的全局变量
static hnswlib::HierarchicalNSW<float>* g_hnsw_index = nullptr;

/**
 * 构建或加载HNSW索引
 */
void prepare_hnsw_index(
    float* base,               // 基础数据集
    size_t base_number,        // 数据集大小
    size_t vecdim,             // 向量维度
    size_t ef = 100            // ef参数，控制搜索精度
) {
    const std::string index_path = "files/hnsw.index";
    
    // 检查索引是否已经加载
    if (g_hnsw_index != nullptr) {
        // 已经加载过索引，直接设置ef参数
        g_hnsw_index->setEf(ef);
        return;
    }
    
    // 检查索引文件是否存在
    bool index_exists = false;
    {
        std::ifstream f(index_path);
        index_exists = f.good();
    }
    
    // 创建内积空间，用于计算向量相似度
    hnswlib::InnerProductSpace* ipspace = new hnswlib::InnerProductSpace(vecdim);
    
    try {
        if (index_exists) {
            // 如果索引存在，加载索引
            std::cout << "加载HNSW索引: " << index_path << std::endl;
            g_hnsw_index = new hnswlib::HierarchicalNSW<float>(ipspace, index_path);
        } else {
            // 索引不存在，创建并构建索引
            std::cout << "构建HNSW索引..." << std::endl;
            
            // 设置HNSW参数
            const int efConstruction = 200; // 构建索引时使用的ef参数
            const int M = 24;  // 每个节点的最大连接数
            
            // 创建索引对象
            g_hnsw_index = new hnswlib::HierarchicalNSW<float>(ipspace, base_number, M, efConstruction);
            
            // 添加所有数据点到索引
            std::cout << "添加数据点到索引中..." << std::endl;
            g_hnsw_index->addPoint(base, 0);
            
            #pragma omp parallel for
            for(int i = 1; i < base_number; ++i) {
                g_hnsw_index->addPoint(base + 1ll*vecdim*i, i);
            }
            
            // 保存索引
            std::cout << "保存HNSW索引到: " << index_path << std::endl;
            g_hnsw_index->saveIndex(index_path);
        }
        
        // 设置搜索参数ef
        g_hnsw_index->setEf(ef);
    } catch (const std::exception& e) {
        std::cerr << "HNSW索引加载/构建错误: " << e.what() << std::endl;
        if (g_hnsw_index) {
            delete g_hnsw_index;
            g_hnsw_index = nullptr;
        }
    }
}

/**
 * 释放HNSW索引
 */
void release_hnsw_index() {
    if (g_hnsw_index != nullptr) {
        delete g_hnsw_index;
        g_hnsw_index = nullptr;
    }
}

/**
 * HNSW搜索算法实现
 * 
 * 该函数只执行搜索，不会加载或构建索引
 */
std::priority_queue<std::pair<float, uint32_t>> hnsw_search(
    float* base,                // 基础数据集
    float* query,               // 查询向量
    size_t base_number,         // 数据集大小
    size_t vecdim,              // 向量维度
    size_t k,                   // 返回k个最近邻
    size_t ef = 100             // ef参数，控制搜索精度
) {
    // 检查索引是否已经加载
    if (g_hnsw_index == nullptr) {
        // 索引未加载，先加载索引
        prepare_hnsw_index(base, base_number, vecdim, ef);
    }
    
    // 创建空结果，以防索引未能加载
    std::priority_queue<std::pair<float, uint32_t>> search_result;
    
    try {
        // 执行KNN搜索
        auto result = g_hnsw_index->searchKnn(query, k);
        
        // 转换结果格式
        while (!result.empty()) {
            auto& pair = result.top();
            search_result.emplace(pair.first, pair.second);
            result.pop();
        }
    } catch (const std::exception& e) {
        std::cerr << "HNSW搜索发生错误: " << e.what() << std::endl;
    }
    
    return search_result;
} 