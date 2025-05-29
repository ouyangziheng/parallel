#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <omp.h>
#include <fstream>
#include "hnswlib/hnswlib/hnswlib.h"

/**
 * IVF-HNSW索引实现
 * 
 * 该索引将数据点分配到不同的簇中，每个簇使用一个HNSW图索引
 * 查询时只搜索最接近的几个簇，从而减少需要比较的向量数量
 */
class IVFHNSWIndex {
public:
    // 聚类中心数量(簇的个数)
    size_t nlist;
    // 查询时搜索的簇数量
    size_t nprobe;
    // 向量维度
    size_t d;
    // 簇中心向量，大小为 nlist * d
    std::vector<float> centroids;
    // 每个簇的HNSW索引
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_indices;
    // 每个簇中的向量ID到全局ID的映射
    std::vector<std::vector<uint32_t>> id_mappings;
    // 用于存储所有基础向量的引用
    float* base_data;
    size_t base_number;
    // 搜索参数
    size_t ef_search;
    
    /**
     * 构造函数
     * @param dimension 向量维度
     * @param n_list 聚类中心数量(簇的个数)
     * @param n_probe 查询时搜索的簇数量
     * @param ef 搜索时的ef参数
     */
    IVFHNSWIndex(size_t dimension, size_t n_list = 100, size_t n_probe = 10, size_t ef = 100)
        : d(dimension), nlist(n_list), nprobe(n_probe), ef_search(ef) {}
    
    /**
     * 析构函数
     */
    ~IVFHNSWIndex() {
        // 释放所有HNSW索引
        for (auto* index : hnsw_indices) {
            if (index) delete index;
        }
    }
    
    /**
     * 使用k-means聚类构建索引
     * @param base 基础向量数据
     * @param n 基础向量数量
     */
    void build(float* base, size_t n) {
        base_data = base;
        base_number = n;
        
        std::cout << "构建IVF-HNSW索引，簇数量: " << nlist << std::endl;
        
        // 初始化簇中心
        init_centroids(base, n);
        
        // 为每个簇的向量ID列表分配空间
        id_mappings.resize(nlist);
        
        // 分配向量到最近的簇
        assign_vectors_to_clusters(base, n);
        
        // 为每个簇构建HNSW索引
        build_hnsw_per_cluster(base, n);
        
        std::cout << "IVF-HNSW索引构建完成" << std::endl;
    }
    
    /**
     * 初始化聚类中心
     * 使用k-means++方法初始化聚类中心
     */
    void init_centroids(float* base, size_t n) {
        std::cout << "初始化聚类中心..." << std::endl;
        
        centroids.resize(nlist * d);
        
        // 随机选择第一个中心点
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        size_t first_center = dis(gen);
        for (size_t j = 0; j < d; j++) {
            centroids[j] = base[first_center * d + j];
        }
        
        // 选择剩余的中心点
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        for (size_t i = 1; i < nlist; i++) {
            // 计算每个点到最近中心的距离
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                float min_dist = std::numeric_limits<float>::max();
                for (size_t c = 0; c < i; c++) {
                    // 计算内积距离(1 - 内积)
                    float ip = 0;
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        ip += centroids[c * d + d_idx] * base[j * d + d_idx];
                    }
                    float dist = 1 - ip;
                    min_dist = std::min(min_dist, dist);
                }
                min_distances[j] = min_dist;
            }
            
            // 按距离平方的概率选择新中心
            std::discrete_distribution<> weighted_dis(min_distances.begin(), min_distances.end());
            size_t new_center = weighted_dis(gen);
            
            for (size_t j = 0; j < d; j++) {
                centroids[i * d + j] = base[new_center * d + j];
            }
        }
    }
    
    /**
     * 将向量分配到最近的簇
     */
    void assign_vectors_to_clusters(float* base, size_t n) {
        std::cout << "将向量分配到簇..." << std::endl;
        
        // 清空已有的分配
        for (auto& cluster : id_mappings) {
            cluster.clear();
        }
        
        // 估计每个簇的大小并预分配内存
        size_t estimated_cluster_size = n / nlist * 2; // 乘以2是为了预留空间
        for (auto& cluster : id_mappings) {
            cluster.reserve(estimated_cluster_size);
        }
        
        // 分配向量到最近的簇
        #pragma omp parallel
        {
            // 每个线程维护本地的分配结果
            std::vector<std::vector<uint32_t>> local_assignments(nlist);
            for (auto& cluster : local_assignments) {
                cluster.reserve(estimated_cluster_size / omp_get_num_threads());
            }
            
            #pragma omp for
            for (int i = 0; i < n; i++) {
                float min_dist = std::numeric_limits<float>::max();
                size_t nearest_cluster = 0;
                
                for (size_t j = 0; j < nlist; j++) {
                    // 计算内积距离(1 - 内积)
                    float ip = 0;
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        ip += centroids[j * d + d_idx] * base[i * d + d_idx];
                    }
                    float dist = 1 - ip;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_cluster = j;
                    }
                }
                
                local_assignments[nearest_cluster].push_back(i);
            }
            
            // 合并本地结果到全局结果
            #pragma omp critical
            {
                for (size_t i = 0; i < nlist; i++) {
                    id_mappings[i].insert(id_mappings[i].end(), 
                                        local_assignments[i].begin(), 
                                        local_assignments[i].end());
                }
            }
        }
        
        // 打印每个簇的大小
        std::cout << "簇大小统计:" << std::endl;
        for (size_t i = 0; i < nlist; i++) {
            std::cout << "簇 " << i << ": " << id_mappings[i].size() << " 个向量" << std::endl;
        }
    }
    
    /**
     * 为每个簇构建HNSW索引
     */
    void build_hnsw_per_cluster(float* base, size_t n) {
        std::cout << "为每个簇构建HNSW索引..." << std::endl;
        
        // 设置HNSW参数
        const int efConstruction = 200; // 构建索引时使用的ef参数
        const int M = 24;  // 每个节点的最大连接数
        
        // 初始化索引向量
        hnsw_indices.resize(nlist, nullptr);
        
        // 为每个簇构建HNSW索引
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nlist; i++) {
            // 只为非空簇构建索引
            if (id_mappings[i].empty()) {
                continue;
            }
            
            size_t cluster_size = id_mappings[i].size();
            
            // 创建内积空间
            hnswlib::InnerProductSpace* space = new hnswlib::InnerProductSpace(d);
            
            // 创建HNSW索引
            hnswlib::HierarchicalNSW<float>* hnsw_index = 
                new hnswlib::HierarchicalNSW<float>(space, cluster_size, M, efConstruction);
            
            // 将簇中的向量添加到索引
            std::vector<float> cluster_vectors(cluster_size * d);
            
            for (size_t j = 0; j < cluster_size; j++) {
                uint32_t global_id = id_mappings[i][j];
                memcpy(cluster_vectors.data() + j * d, base + global_id * d, d * sizeof(float));
                
                // 添加到HNSW索引，局部ID为j
                hnsw_index->addPoint(cluster_vectors.data() + j * d, j);
            }
            
            // 设置搜索参数
            hnsw_index->setEf(ef_search);
            
            // 保存到索引向量
            hnsw_indices[i] = hnsw_index;
            
            std::cout << "完成簇 " << i << " 的HNSW索引构建" << std::endl;
        }
    }
    
    /**
     * 保存索引到文件
     */
    void save(const std::string& filename) {
        std::cout << "保存IVF-HNSW索引到: " << filename << std::endl;
        
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }
        
        // 保存基本参数
        out.write(reinterpret_cast<char*>(&nlist), sizeof(nlist));
        out.write(reinterpret_cast<char*>(&nprobe), sizeof(nprobe));
        out.write(reinterpret_cast<char*>(&d), sizeof(d));
        out.write(reinterpret_cast<char*>(&ef_search), sizeof(ef_search));
        
        // 保存簇中心
        out.write(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
        
        // 保存每个簇的向量ID映射
        for (size_t i = 0; i < nlist; i++) {
            size_t size = id_mappings[i].size();
            out.write(reinterpret_cast<char*>(&size), sizeof(size));
            
            if (size > 0) {
                out.write(reinterpret_cast<char*>(id_mappings[i].data()), 
                         size * sizeof(uint32_t));
            }
        }
        
        out.close();
        
        // 分别保存每个簇的HNSW索引
        for (size_t i = 0; i < nlist; i++) {
            if (hnsw_indices[i]) {
                std::string hnsw_filename = filename + ".cluster_" + std::to_string(i);
                hnsw_indices[i]->saveIndex(hnsw_filename);
            }
        }
    }
    
    /**
     * 从文件加载索引
     */
    void load(const std::string& filename) {
        std::cout << "从文件加载IVF-HNSW索引: " << filename << std::endl;
        
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("无法打开文件: " + filename);
        }
        
        // 释放已有资源
        for (auto* index : hnsw_indices) {
            if (index) delete index;
        }
        hnsw_indices.clear();
        id_mappings.clear();
        centroids.clear();
        
        // 加载基本参数
        in.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
        in.read(reinterpret_cast<char*>(&nprobe), sizeof(nprobe));
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        in.read(reinterpret_cast<char*>(&ef_search), sizeof(ef_search));
        
        // 加载簇中心
        centroids.resize(nlist * d);
        in.read(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
        
        // 加载每个簇的向量ID映射
        id_mappings.resize(nlist);
        for (size_t i = 0; i < nlist; i++) {
            size_t size;
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            
            if (size > 0) {
                id_mappings[i].resize(size);
                in.read(reinterpret_cast<char*>(id_mappings[i].data()), 
                       size * sizeof(uint32_t));
            }
        }
        
        in.close();
        
        // 初始化HNSW索引向量
        hnsw_indices.resize(nlist, nullptr);
        
        // 加载每个簇的HNSW索引
        for (size_t i = 0; i < nlist; i++) {
            std::string hnsw_filename = filename + ".cluster_" + std::to_string(i);
            
            // 检查文件是否存在
            std::ifstream test(hnsw_filename);
            if (test.good() && !id_mappings[i].empty()) {
                test.close();
                
                try {
                    // 创建内积空间
                    hnswlib::InnerProductSpace* space = new hnswlib::InnerProductSpace(d);
                    
                    // 加载HNSW索引
                    hnsw_indices[i] = new hnswlib::HierarchicalNSW<float>(space, hnsw_filename);
                    
                    // 设置搜索参数
                    hnsw_indices[i]->setEf(ef_search);
                } catch (const std::exception& e) {
                    std::cerr << "加载簇 " << i << " 的HNSW索引失败: " << e.what() << std::endl;
                }
            }
        }
    }
    
    /**
     * 设置查询时搜索的簇数量
     */
    void set_nprobe(size_t n_probe) {
        nprobe = n_probe;
    }
    
    /**
     * 设置搜索时的ef参数
     */
    void set_ef(size_t ef) {
        ef_search = ef;
        for (auto* index : hnsw_indices) {
            if (index) index->setEf(ef);
        }
    }
    
    /**
     * 查询最近邻
     * @param query 查询向量
     * @param k 返回的最近邻数量
     * @return 返回最近的k个向量ID及其距离
     */
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, size_t k) const {
        // 计算查询向量到每个簇中心的距离
        std::vector<std::pair<float, uint32_t>> centroid_distances(nlist);
        
        #pragma omp parallel for
        for (int i = 0; i < nlist; i++) {
            // 计算内积距离(1 - 内积)
            float ip = 0;
            for (size_t j = 0; j < d; j++) {
                ip += centroids[i * d + j] * query[j];
            }
            centroid_distances[i] = {1 - ip, i};
        }
        
        // 选择距离最近的nprobe个簇
        std::partial_sort(centroid_distances.begin(), 
                         centroid_distances.begin() + std::min(nprobe, nlist), 
                         centroid_distances.end());
        
        // 每个线程处理一部分簇，将结果合并
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> thread_results;
        
        #pragma omp parallel
        {
            // 每个线程的局部结果
            std::priority_queue<std::pair<float, uint32_t>> local_result;
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < std::min(nprobe, nlist); i++) {
                uint32_t cluster_id = centroid_distances[i].second;
                
                // 跳过没有索引的簇
                if (!hnsw_indices[cluster_id] || id_mappings[cluster_id].empty()) {
                    continue;
                }
                
                // 使用HNSW索引搜索
                auto cluster_results = hnsw_indices[cluster_id]->searchKnn(query, k);
                
                // 将局部ID转换为全局ID，并添加到结果中
                while (!cluster_results.empty()) {
                    auto& pair = cluster_results.top();
                    uint32_t local_id = pair.second;
                    uint32_t global_id = id_mappings[cluster_id][local_id];
                    
                    local_result.emplace(pair.first, global_id);
                    cluster_results.pop();
                }
            }
            
            #pragma omp critical
            {
                thread_results.push_back(std::move(local_result));
            }
        }
        
        // 合并所有线程的结果
        std::priority_queue<std::pair<float, uint32_t>> result;
        for (auto& thread_result : thread_results) {
            while (!thread_result.empty()) {
                if (result.size() < k) {
                    result.push(thread_result.top());
                } else if (thread_result.top().first < result.top().first) {
                    result.push(thread_result.top());
                    result.pop();
                }
                thread_result.pop();
            }
        }
        
        return result;
    }
};

// 全局变量用于缓存索引
namespace {
    IVFHNSWIndex* g_ivf_hnsw_index = nullptr;
    bool g_ivf_hnsw_initialized = false;
}

/**
 * 构建IVF-HNSW索引
 */
void build_ivf_hnsw_index(
    float* base, 
    size_t base_number, 
    size_t vecdim, 
    size_t nlist = 256, 
    size_t nprobe = 64,
    size_t ef_search = 200
) {
    std::cout << "构建IVF-HNSW索引，簇数量: " << nlist << "，搜索簇数量: " << nprobe << std::endl;
    
    // 创建索引
    IVFHNSWIndex* ivf_hnsw_index = new IVFHNSWIndex(vecdim, nlist, nprobe, ef_search);
    
    // 构建索引
    ivf_hnsw_index->build(base, base_number);
    
    // 保存索引
    ivf_hnsw_index->save("files/ivf_hnsw.index");
    
    // 释放内存
    delete ivf_hnsw_index;
}

/**
 * 确保IVF-HNSW索引已初始化
 */
void ensure_ivf_hnsw_index_initialized(
    const std::string& index_path = "files/ivf_hnsw.index",
    size_t ef_search = 200
) {
    // 清理之前的缓存索引（确保每次使用的都是最新的索引）
    if (g_ivf_hnsw_index != nullptr) {
        delete g_ivf_hnsw_index;
        g_ivf_hnsw_index = nullptr;
        g_ivf_hnsw_initialized = false;
    }
    
    if (g_ivf_hnsw_initialized) {
        return;
    }
    
    // 检查索引文件是否存在
    std::ifstream file(index_path);
    if (!file.is_open()) {
        std::cerr << "IVF-HNSW索引文件不存在: " << index_path << std::endl;
        std::cerr << "请先调用 build_ivf_hnsw_index 构建索引" << std::endl;
        return;
    }
    file.close();
    
    // 加载索引
    g_ivf_hnsw_index = new IVFHNSWIndex(0);
    
    try {
        g_ivf_hnsw_index->load(index_path);
        g_ivf_hnsw_index->set_ef(ef_search);
        g_ivf_hnsw_initialized = true;
    } catch (const std::exception& e) {
        std::cerr << "IVF-HNSW索引加载错误: " << e.what() << std::endl;
        delete g_ivf_hnsw_index;
        g_ivf_hnsw_index = nullptr;
    }
}

/**
 * 释放IVF-HNSW索引
 */
void release_ivf_hnsw_index() {
    if (g_ivf_hnsw_index != nullptr) {
        delete g_ivf_hnsw_index;
        g_ivf_hnsw_index = nullptr;
        g_ivf_hnsw_initialized = false;
    }
}

/**
 * 基于IVF-HNSW实现的最近邻查询
 * 
 * 该函数先查找最相似的nprobe个簇，然后对每个簇使用HNSW搜索
 * 
 * @param base 基础数据集
 * @param query 查询向量
 * @param base_number 数据集大小
 * @param vecdim 向量维度
 * @param k 返回k个最近邻
 * @param nprobe 搜索的簇数量
 * @param ef_search 搜索时的ef参数
 * @return 返回最近的k个向量ID及其距离
 */
std::priority_queue<std::pair<float, uint32_t>> ivf_hnsw_search(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe = 64,
    size_t ef_search = 200
) {
    // 假设索引已经初始化好了，直接使用
    if (!g_ivf_hnsw_initialized) {
        // 只在必要时进行初始化
        ensure_ivf_hnsw_index_initialized("files/ivf_hnsw.index", ef_search);
    }
    
    // 然后使用已加载的索引进行搜索，不要重新初始化
    return g_ivf_hnsw_index->search(query, k);
} 