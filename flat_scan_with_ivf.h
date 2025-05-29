#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <unordered_map>
#include <iostream>

#include "simd_utils.h"

/**
 * IVF (Inverted File Index) 索引实现
 * 
 * IVF索引将数据点分配到不同的簇中，查询时只搜索最接近的几个簇，
 * 从而减少需要比较的向量数量，提高查询效率。
 */
class IVFIndex {
public:
    // 聚类中心数量(簇的个数)
    size_t nlist;
    // 查询时搜索的簇数量
    size_t nprobe;
    // 向量维度
    size_t d;
    // 簇中心向量，大小为 nlist * d
    std::vector<float> centroids;
    // 每个簇中的向量ID列表
    std::vector<std::vector<uint32_t>> inverted_lists;
    // 用于存储所有基础向量的引用
    float* base_data;
    size_t base_number;
    // 是否进行了内存重排
    bool is_memory_rearranged;
    // 如果进行了内存重排，这里存储重排后的数据
    std::vector<float> rearranged_data;
    
    /**
     * 构造函数
     * @param dimension 向量维度
     * @param n_list 聚类中心数量(簇的个数)
     * @param n_probe 查询时搜索的簇数量
     */
    IVFIndex(size_t dimension, size_t n_list = 100, size_t n_probe = 10)
        : d(dimension), nlist(n_list), nprobe(n_probe), is_memory_rearranged(false) {}
    
    /**
     * 使用k-means聚类构建索引
     * @param base 基础向量数据
     * @param n 基础向量数量
     */
    void build(float* base, size_t n) {
        base_data = base;
        base_number = n;
        
        std::cout << "构建IVF索引，簇数量: " << nlist << std::endl;
        
        // 初始化簇中心
        init_centroids(base, n);
        
        // 为每个簇的向量ID列表分配空间
        inverted_lists.resize(nlist);
        
        // 分配向量到最近的簇
        assign_vectors_to_clusters(base, n);
        
        std::cout << "IVF索引构建完成，共 " << nlist << " 个簇" << std::endl;
        
        // 打印每个簇的大小统计信息
        print_cluster_stats();
    }
    
    /**
     * 内存重排：将相同簇的向量存储在一起，降低查询时的cache miss率
     */
    void rearrange_memory() {
        if (is_memory_rearranged) {
            std::cout << "内存已经重排过，跳过重排" << std::endl;
            return;
        }
        
        std::cout << "进行内存重排..." << std::endl;
        
        // 为重排后的数据分配空间
        rearranged_data.resize(base_number * d);
        
        // 创建原始ID到新位置的映射
        std::vector<uint32_t> id_mapping(base_number);
        
        // 重排数据
        size_t current_idx = 0;
        for (size_t i = 0; i < nlist; i++) {
            const auto& cluster = inverted_lists[i];
            for (uint32_t vec_id : cluster) {
                // 复制向量数据
                std::copy(base_data + vec_id * d, 
                          base_data + (vec_id + 1) * d, 
                          rearranged_data.data() + current_idx * d);
                
                // 更新ID映射
                id_mapping[vec_id] = current_idx;
                current_idx++;
            }
        }
        
        // 更新每个簇的向量ID
        for (size_t i = 0; i < nlist; i++) {
            auto& cluster = inverted_lists[i];
            for (size_t j = 0; j < cluster.size(); j++) {
                cluster[j] = id_mapping[cluster[j]];
            }
        }
        
        // 标记内存已重排
        is_memory_rearranged = true;
        
        std::cout << "内存重排完成" << std::endl;
    }
    
    /**
     * 保存索引到文件
     * @param filename 文件名
     */
    void save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "无法打开文件 " << filename << " 进行写入" << std::endl;
            return;
        }
        
        // 保存基本参数
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(&nlist), sizeof(nlist));
        out.write(reinterpret_cast<const char*>(&nprobe), sizeof(nprobe));
        out.write(reinterpret_cast<const char*>(&is_memory_rearranged), sizeof(is_memory_rearranged));
        
        // 保存簇中心
        size_t centroids_size = centroids.size();
        out.write(reinterpret_cast<const char*>(&centroids_size), sizeof(centroids_size));
        out.write(reinterpret_cast<const char*>(centroids.data()), centroids_size * sizeof(float));
        
        // 保存倒排列表
        for (size_t i = 0; i < nlist; i++) {
            size_t list_size = inverted_lists[i].size();
            out.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
            out.write(reinterpret_cast<const char*>(inverted_lists[i].data()), 
                     list_size * sizeof(uint32_t));
        }
        
        // 如果内存已重排，保存重排后的数据
        if (is_memory_rearranged) {
            size_t data_size = rearranged_data.size();
            out.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
            out.write(reinterpret_cast<const char*>(rearranged_data.data()), 
                     data_size * sizeof(float));
        }
        
        out.close();
        std::cout << "索引已保存到 " << filename << std::endl;
    }
    
    /**
     * 从文件加载索引
     * @param filename 文件名
     */
    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "无法打开文件 " << filename << " 进行读取" << std::endl;
            return;
        }
        
        // 读取基本参数
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        in.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
        in.read(reinterpret_cast<char*>(&nprobe), sizeof(nprobe));
        in.read(reinterpret_cast<char*>(&is_memory_rearranged), sizeof(is_memory_rearranged));
        
        // 读取簇中心
        size_t centroids_size;
        in.read(reinterpret_cast<char*>(&centroids_size), sizeof(centroids_size));
        centroids.resize(centroids_size);
        in.read(reinterpret_cast<char*>(centroids.data()), centroids_size * sizeof(float));
        
        // 读取倒排列表
        inverted_lists.resize(nlist);
        for (size_t i = 0; i < nlist; i++) {
            size_t list_size;
            in.read(reinterpret_cast<char*>(&list_size), sizeof(list_size));
            inverted_lists[i].resize(list_size);
            in.read(reinterpret_cast<char*>(inverted_lists[i].data()), 
                   list_size * sizeof(uint32_t));
        }
        
        // 如果内存已重排，读取重排后的数据
        if (is_memory_rearranged) {
            size_t data_size;
            in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            rearranged_data.resize(data_size);
            in.read(reinterpret_cast<char*>(rearranged_data.data()), 
                   data_size * sizeof(float));
        }
        
        in.close();
        std::cout << "索引已从 " << filename << " 加载" << std::endl;
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
        
        // 结果堆
        std::priority_queue<std::pair<float, uint32_t>> result;
        
        // 设置查询的数据源
        const float* data_ptr = is_memory_rearranged ? rearranged_data.data() : base_data;
        
        // 在选定的簇中搜索最近邻
        #pragma omp parallel
        {
            // 每个线程的局部结果
            std::priority_queue<std::pair<float, uint32_t>> local_result;
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < std::min(nprobe, nlist); i++) {
                uint32_t cluster_id = centroid_distances[i].second;
                const auto& cluster = inverted_lists[cluster_id];
                
                // 搜索当前簇中的所有向量
                for (uint32_t vec_id : cluster) {
                    float ip = 0;
                    const float* vec = data_ptr + vec_id * d;
                    
                    // 使用SIMD优化内积计算
                    ip = simd_inner_product(vec, query, d);
                    
                    // 使用1 - 内积作为距离
                    float dist = 1 - ip;
                    
                    if (local_result.size() < k) {
                        local_result.push({dist, vec_id});
                    } else if (dist < local_result.top().first) {
                        local_result.push({dist, vec_id});
                        local_result.pop();
                    }
                }
            }
            
            // 合并局部结果
            #pragma omp critical
            {
                while (!local_result.empty()) {
                    if (result.size() < k) {
                        result.push(local_result.top());
                    } else if (local_result.top().first < result.top().first) {
                        result.push(local_result.top());
                        result.pop();
                    }
                    local_result.pop();
                }
            }
        }
        
        return result;
    }
    
    /**
     * 设置查询时搜索的簇数量
     * @param n_probe 新的nprobe值
     */
    void set_nprobe(size_t n_probe) {
        nprobe = n_probe;
    }

private:
    /**
     * 初始化簇中心
     * 使用k-means++算法选择初始簇中心
     */
    void init_centroids(const float* base, size_t n) {
        centroids.resize(nlist * d);
        
        // 使用随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dist(0, n - 1);
        std::uniform_real_distribution<float> dist_float(0, 1);
        
        // 选择第一个簇中心（随机选择）
        uint32_t first_center = dist(gen);
        std::copy(base + first_center * d, base + (first_center + 1) * d, centroids.data());
        
        // 存储每个点到最近簇中心的距离
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        // 选择剩余的簇中心
        for (size_t i = 1; i < nlist; i++) {
            // 更新每个点到最近簇中心的距离
            #pragma omp parallel for
            for (int j = 0; j < n; j++) {
                float dist = 0;
                // 计算点j到最新簇中心的距离
                for (size_t k = 0; k < d; k++) {
                    float diff = base[j * d + k] - centroids[(i - 1) * d + k];
                    dist += diff * diff;
                }
                
                // 更新最小距离
                if (dist < min_distances[j]) {
                    min_distances[j] = dist;
                }
            }
            
            // 计算距离总和
            float sum_distances = 0;
            for (size_t j = 0; j < n; j++) {
                sum_distances += min_distances[j];
            }
            
            // 按照距离的平方作为概率选择下一个簇中心
            float rand_val = dist_float(gen) * sum_distances;
            float cum_sum = 0;
            uint32_t next_center = 0;
            
            for (size_t j = 0; j < n; j++) {
                cum_sum += min_distances[j];
                if (cum_sum >= rand_val) {
                    next_center = j;
                    break;
                }
            }
            
            // 复制新的簇中心
            std::copy(base + next_center * d, base + (next_center + 1) * d, 
                     centroids.data() + i * d);
        }
    }
    
    /**
     * 将向量分配到最近的簇
     */
    void assign_vectors_to_clusters(const float* base, size_t n) {
        // 清空倒排列表
        for (auto& list : inverted_lists) {
            list.clear();
        }
        
        // 为每个向量计算最近的簇
        #pragma omp parallel
        {
            // 每个线程的局部倒排列表
            std::vector<std::vector<uint32_t>> local_lists(nlist);
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                uint32_t nearest_cluster = 0;
                float min_dist = std::numeric_limits<float>::max();
                
                // 计算到每个簇中心的距离
                for (size_t j = 0; j < nlist; j++) {
                    float ip = 0;
                    for (size_t k = 0; k < d; k++) {
                        ip += centroids[j * d + k] * base[i * d + k];
                    }
                    
                    // 使用1 - 内积作为距离
                    float dist = 1 - ip;
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_cluster = j;
                    }
                }
                
                // 将向量ID添加到局部列表
                local_lists[nearest_cluster].push_back(i);
            }
            
            // 合并局部列表
            #pragma omp critical
            {
                for (size_t i = 0; i < nlist; i++) {
                    inverted_lists[i].insert(inverted_lists[i].end(), 
                                           local_lists[i].begin(), 
                                           local_lists[i].end());
                }
            }
        }
    }
    
    /**
     * 打印簇的统计信息
     */
    void print_cluster_stats() const {
        std::vector<size_t> sizes;
        for (const auto& list : inverted_lists) {
            sizes.push_back(list.size());
        }
        
        // 计算最大、最小和平均簇大小
        size_t min_size = *std::min_element(sizes.begin(), sizes.end());
        size_t max_size = *std::max_element(sizes.begin(), sizes.end());
        
        double avg_size = 0;
        for (size_t s : sizes) {
            avg_size += s;
        }
        avg_size /= sizes.size();
        
        std::cout << "簇统计信息：" << std::endl;
        std::cout << "- 最小簇大小: " << min_size << std::endl;
        std::cout << "- 最大簇大小: " << max_size << std::endl;
        std::cout << "- 平均簇大小: " << avg_size << std::endl;
        
        // 计算空簇的数量
        size_t empty_clusters = 0;
        for (const auto& list : inverted_lists) {
            if (list.empty()) {
                empty_clusters++;
            }
        }
        
        if (empty_clusters > 0) {
            std::cout << "- 空簇数量: " << empty_clusters << " (" 
                     << (100.0 * empty_clusters / nlist) << "%)" << std::endl;
        }
    }
};

/**
 * 构建IVF索引
 * @param base 基础向量数据
 * @param base_number 基础向量数量
 * @param vecdim 向量维度
 * @param nlist 聚类中心数量(簇的个数)
 * @param nprobe 查询时搜索的簇数量
 * @param rearrange_memory 是否进行内存重排
 */
void build_ivf_index(float* base, size_t base_number, size_t vecdim, 
                    size_t nlist = 100, size_t nprobe = 10, 
                    bool rearrange_memory = true) {
    // 创建IVF索引
    IVFIndex* ivf_index = new IVFIndex(vecdim, nlist, nprobe);
    
    // 构建索引
    ivf_index->build(base, base_number);
    
    // 如果需要，进行内存重排
    if (rearrange_memory) {
        ivf_index->rearrange_memory();
    }
    
    // 保存索引
    ivf_index->save("files/ivf.index");
    
    // 释放内存
    delete ivf_index;
}

// 全局变量用于缓存索引
namespace {
    IVFIndex* g_ivf_index = nullptr;
    bool g_ivf_initialized = false;
}

/**
 * 确保IVF索引已初始化
 */
void ensure_ivf_index_initialized(const std::string& index_path = "files/ivf.index") {
    // 清理之前的缓存索引（确保每次使用的都是最新的索引）
    if (g_ivf_index != nullptr) {
        delete g_ivf_index;
        g_ivf_index = nullptr;
        g_ivf_initialized = false;
    }
    
    if (g_ivf_initialized) {
        return;
    }
    
    // 检查索引文件是否存在
    std::ifstream file(index_path);
    if (!file.is_open()) {
        std::cerr << "IVF索引文件不存在: " << index_path << std::endl;
        std::cerr << "请先调用 build_ivf_index 构建索引" << std::endl;
        return;
    }
    file.close();
    
    // 加载索引
    g_ivf_index = new IVFIndex(0);
    g_ivf_index->load(index_path);
    g_ivf_initialized = true;
}

/**
 * 基于IVF实现的最近邻查询，使用内积距离(Inner Product)作为相似度指标
 * 
 * @param base 基础数据集合
 * @param query 查询向量
 * @param base_number 向量库中base向量的总数
 * @param vecdim 向量维度
 * @param k 需要返回的最相似向量个数
 * @param nprobe 搜索的簇数量（可选，默认为10）
 * 
 * @return 返回一个最大堆，堆中存放的是 {距离, 向量编号}，用于表示距离query最近的k个向量
 */
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_ivf_Inner_Product(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe = 10) {
    
    // 确保索引已初始化
    ensure_ivf_index_initialized();
    
    // 如果索引未成功初始化，回退到暴力搜索
    if (!g_ivf_initialized) {
        std::cerr << "IVF索引未初始化，回退到暴力搜索" << std::endl;
        
        std::priority_queue<std::pair<float, uint32_t>> q;
        
        for (int i = 0; i < base_number; ++i) {
            float dis = 0;
            
            for (int d = 0; d < vecdim; ++d) {
                dis += base[d + i*vecdim] * query[d];
            }
            dis = 1 - dis;
            
            if (q.size() < k) {
                q.push({dis, i});
            } else {
                if (dis < q.top().first) {
                    q.push({dis, i});
                    q.pop();
                }
            }
        }
        return q;
    }
    
    // 更新nprobe值
    g_ivf_index->set_nprobe(nprobe);
    
    // 使用IVF索引搜索
    return g_ivf_index->search(query, k);
} 