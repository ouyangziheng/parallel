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
#include <arm_neon.h>

#include "simd_utils.h"

// 前向声明
class IVFPQIndex;

// 定义量化器接口
class IVFPQQuantizer {
public:
    // 子空间数量
    size_t M;
    // 每个子空间的聚类中心数量
    size_t K;
    // 原始向量维度
    size_t d;
    // 每个子空间的维度
    size_t dsub;
    // 码本，存储每个子空间的聚类中心
    // 大小为 M * K * dsub
    std::vector<float> codebooks;
    // 每个子空间的距离表
    // 用于加速查询时的距离计算
    std::vector<float> distance_tables;
    
    /**
     * 构造函数
     * @param dimension 向量维度
     * @param m 子空间数量
     * @param k 每个子空间的聚类中心数量
     */
    IVFPQQuantizer(size_t dimension, size_t m = 8, size_t k = 256)
        : d(dimension), M(m), K(k) {
        // 确保每个子空间的维度是整数
        dsub = d / M;
        if (d % M != 0) {
            std::cerr << "警告: 向量维度 " << d << " 不能被子空间数量 " << M << " 整除" << std::endl;
            std::cerr << "调整子空间数量为 " << d / dsub << std::endl;
            M = d / dsub;
        }
        
        // 分配码本空间
        codebooks.resize(M * K * dsub, 0);
    }
    
    /**
     * 训练量化器 - 使用K-means在每个子空间上进行聚类
     * @param data 训练数据
     * @param n 训练数据数量
     */
    void train(const float* data, size_t n) {
        std::cout << "训练PQ量化器，子空间数量: " << M << ", 码本大小: " << K << std::endl;
        
        // 为每个子空间准备训练数据
        std::vector<std::vector<float>> sub_data(M);
        for (size_t m = 0; m < M; m++) {
            sub_data[m].resize(n * dsub);
            
            // 提取每个子空间的数据
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < dsub; j++) {
                    sub_data[m][i * dsub + j] = data[i * d + m * dsub + j];
                }
            }
        }
        
        // 为每个子空间训练K-means
        #pragma omp parallel for
        for (int m = 0; m < M; m++) {
            train_subspace(m, sub_data[m].data(), n);
        }
        
        std::cout << "PQ量化器训练完成" << std::endl;
    }
    
    /**
     * 量化向量 - 将向量编码为PQ码
     * @param x 原始向量
     * @param code 输出的PQ码
     */
    void compute_code(const float* x, uint8_t* code) const {
        // 标准化向量
        std::vector<float> x_norm(d);
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += x[j] * x[j];
        }
        norm = std::sqrt(norm) + 1e-8f; // 避免除零
        
        for (size_t j = 0; j < d; j++) {
            x_norm[j] = x[j] / norm;
        }
        
        // 为每个子空间找到最近的码字
        for (size_t m = 0; m < M; m++) {
            float min_dist = std::numeric_limits<float>::max();
            int min_idx = 0;
            
            // 计算到每个码字的距离
            for (size_t k = 0; k < K; k++) {
                float ip = 0; // 使用内积
                
                // 计算内积
                for (size_t j = 0; j < dsub; j++) {
                    ip += x_norm[m * dsub + j] * codebooks[(m * K + k) * dsub + j];
                }
                
                // 转换为距离：1 - 内积
                float dist = 1.0f - ip;
                
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = k;
                }
            }
            
            // 存储码字索引
            code[m] = static_cast<uint8_t>(min_idx);
        }
    }
    
    /**
     * 计算查询向量与所有码本的距离表
     * @param x 查询向量
     * @param distances 输出的距离表
     */
    void compute_distance_tables(const float* x, float* distances) const {
        // 标准化查询向量
        std::vector<float> x_norm(d);
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += x[j] * x[j];
        }
        norm = std::sqrt(norm) + 1e-8f; // 避免除零
        
        for (size_t j = 0; j < d; j++) {
            x_norm[j] = x[j] / norm;
        }
        
        // 计算查询向量与每个子空间中所有码字的距离
        for (size_t m = 0; m < M; m++) {
            for (size_t k = 0; k < K; k++) {
                float ip = 0; // 使用内积
                
                // 计算内积
                for (size_t j = 0; j < dsub; j++) {
                    ip += x_norm[m * dsub + j] * codebooks[(m * K + k) * dsub + j];
                }
                
                // 使用1 - 内积作为距离
                float dist = 1.0f - ip;
                
                // 存储距离
                distances[m * K + k] = dist;
            }
        }
    }
    
    /**
     * 从PQ码中重构原始向量的近似值
     * @param code PQ码
     * @param reconstructed 重构的向量
     */
    void decode(const uint8_t* code, float* reconstructed) const {
        // 对每个子空间，从码本中查找对应的码字
        for (size_t m = 0; m < M; m++) {
            uint8_t idx = code[m];
            
            // 复制码字到重构向量
            for (size_t j = 0; j < dsub; j++) {
                reconstructed[m * dsub + j] = codebooks[(m * K + idx) * dsub + j];
            }
        }
    }
    
    /**
     * 使用距离表计算查询向量与PQ码的距离
     * @param code PQ码
     * @param distances 距离表
     * @return 距离
     */
    float compute_distance_from_tables(const uint8_t* code, const float* distances) const {
        float dist = 0;
        
        // 从距离表中查找每个子空间的距离并累加
        for (size_t m = 0; m < M; m++) {
            uint8_t idx = code[m];
            dist += distances[m * K + idx];
        }
        
        return dist;
    }
    
private:
    /**
     * 训练单个子空间的K-means
     * @param m 子空间索引
     * @param data 子空间数据
     * @param n 数据数量
     */
    void train_subspace(size_t m, const float* data, size_t n) {
        // 标准化子空间数据
        std::vector<float> normalized_data(n * dsub);
        
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            // 计算子向量的L2范数
            float norm = 0.0f;
            for (size_t j = 0; j < dsub; j++) {
                norm += data[i * dsub + j] * data[i * dsub + j];
            }
            norm = std::sqrt(norm) + 1e-8f;
            
            // 标准化子向量
            for (size_t j = 0; j < dsub; j++) {
                normalized_data[i * dsub + j] = data[i * dsub + j] / norm;
            }
        }
        
        // K-means++初始化
        std::vector<float> centroids(K * dsub);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        std::uniform_real_distribution<float> dist_float(0, 1);
        
        // 随机选择第一个中心点
        size_t first_center = dist(gen);
        for (size_t j = 0; j < dsub; j++) {
            centroids[j] = normalized_data[first_center * dsub + j];
        }
        
        // 标准化第一个中心点
        float c_norm = 0.0f;
        for (size_t j = 0; j < dsub; j++) {
            c_norm += centroids[j] * centroids[j];
        }
        c_norm = std::sqrt(c_norm) + 1e-8f;
        
        for (size_t j = 0; j < dsub; j++) {
            centroids[j] /= c_norm;
        }
        
        // K-means++初始化剩余的中心点
        std::vector<float> min_dists(n, std::numeric_limits<float>::max());
        
        for (size_t c = 1; c < K; c++) {
            // 更新到最近中心点的距离
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                // 计算与当前中心的内积
                float ip = 0.0f;
                for (size_t j = 0; j < dsub; j++) {
                    ip += normalized_data[i * dsub + j] * centroids[(c-1) * dsub + j];
                }
                
                // 使用1-内积作为距离
                float dist = 1.0f - ip;
                
                if (dist < min_dists[i]) {
                    min_dists[i] = dist;
                }
            }
            
            // 计算概率分布
            float sum_dists = 0;
            for (size_t i = 0; i < n; i++) {
                sum_dists += min_dists[i];
            }
            
            // 按概率选择下一个中心点
            float rand_val = dist_float(gen) * sum_dists;
            float cumulative = 0;
            size_t next_center = 0;
            
            for (size_t i = 0; i < n; i++) {
                cumulative += min_dists[i];
                if (cumulative >= rand_val) {
                    next_center = i;
                    break;
                }
            }
            
            // 设置新的中心点
            for (size_t j = 0; j < dsub; j++) {
                centroids[c * dsub + j] = normalized_data[next_center * dsub + j];
            }
            
            // 标准化新中心点
            c_norm = 0.0f;
            for (size_t j = 0; j < dsub; j++) {
                c_norm += centroids[c * dsub + j] * centroids[c * dsub + j];
            }
            c_norm = std::sqrt(c_norm) + 1e-8f;
            
            for (size_t j = 0; j < dsub; j++) {
                centroids[c * dsub + j] /= c_norm;
            }
        }
        
        // K-means迭代
        const int max_iter = 30; // 增加迭代次数提高精度
        std::vector<size_t> assignments(n);
        std::vector<size_t> counts(K);
        std::vector<float> new_centroids(K * dsub);
        bool changed = true;
        
        for (int iter = 0; iter < max_iter && changed; iter++) {
            // 重置状态
            changed = false;
            std::fill(new_centroids.begin(), new_centroids.end(), 0);
            std::fill(counts.begin(), counts.end(), 0);
            
            // 分配点到最近的中心 - 使用内积距离
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                float max_ip = -std::numeric_limits<float>::max();
                size_t best_idx = 0;
                
                // 找到内积最大的中心点
                for (size_t k = 0; k < K; k++) {
                    float ip = 0.0f;
                    for (size_t j = 0; j < dsub; j++) {
                        ip += normalized_data[i * dsub + j] * centroids[k * dsub + j];
                    }
                    
                    if (ip > max_ip) {
                        max_ip = ip;
                        best_idx = k;
                    }
                }
                
                // 检查分配是否改变
                if (assignments[i] != best_idx) {
                    assignments[i] = best_idx;
                    changed = true;
                }
                
                // 累加点到新的中心
                #pragma omp atomic
                counts[best_idx]++;
                
                #pragma omp critical
                {
                    for (size_t j = 0; j < dsub; j++) {
                        new_centroids[best_idx * dsub + j] += normalized_data[i * dsub + j];
                    }
                }
            }
            
            // 计算新的中心点
            for (size_t k = 0; k < K; k++) {
                if (counts[k] > 0) {
                    // 平均化
                    for (size_t j = 0; j < dsub; j++) {
                        new_centroids[k * dsub + j] /= counts[k];
                    }
                    
                    // 标准化
                    float c_norm = 0.0f;
                    for (size_t j = 0; j < dsub; j++) {
                        c_norm += new_centroids[k * dsub + j] * new_centroids[k * dsub + j];
                    }
                    c_norm = std::sqrt(c_norm) + 1e-8f;
                    
                    for (size_t j = 0; j < dsub; j++) {
                        centroids[k * dsub + j] = new_centroids[k * dsub + j] / c_norm;
                    }
                } else {
                    // 处理空簇 - 选择一个随机点
                    size_t random_point = dist(gen);
                    for (size_t j = 0; j < dsub; j++) {
                        centroids[k * dsub + j] = normalized_data[random_point * dsub + j];
                    }
                }
            }
        }
        
        // 将训练好的码本复制到全局码本
        for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < dsub; j++) {
                codebooks[(m * K + k) * dsub + j] = centroids[k * dsub + j];
            }
        }
    }
};

/**
 * IVF-PQ混合索引实现
 * 
 * 结合了IVF的聚类索引和PQ的量化压缩技术，提供两种策略：
 * 1. 先对所有base data进行PQ，再构建IVF索引 (PQ_then_IVF)
 * 2. 先构建IVF索引，再在每个簇中分别进行PQ (IVF_then_PQ)
 */
class IVFPQIndex {
public:
    // 聚类中心数量(簇的个数)
    size_t nlist;
    // 查询时搜索的簇数量
    size_t nprobe;
    // 向量维度
    size_t d;
    // 簇中心向量，大小为 nlist * d
    std::vector<float> centroids;
    // 产品量化器
    std::shared_ptr<IVFPQQuantizer> pq;
    // 每个簇中的向量ID列表
    std::vector<std::vector<uint32_t>> inverted_lists;
    // 每个簇中向量的PQ编码
    std::vector<std::vector<uint8_t>> codes;
    // 用于存储所有基础向量的引用
    float* base_data;
    size_t base_number;
    // 标记是使用哪种索引策略
    bool pq_then_ivf;
    
    /**
     * 构造函数
     * @param dimension 向量维度
     * @param n_list 聚类中心数量(簇的个数)
     * @param n_probe 查询时搜索的簇数量
     * @param use_pq_then_ivf 是否使用"先PQ后IVF"的策略
     * @param m 子空间数量
     * @param k 每个子空间的聚类中心数量
     */
    IVFPQIndex(size_t dimension, size_t n_list = 100, size_t n_probe = 10,
               bool use_pq_then_ivf = false, size_t m = 8, size_t k = 256)
        : d(dimension), nlist(n_list), nprobe(n_probe), pq_then_ivf(use_pq_then_ivf), base_data(nullptr), base_number(0) {
        // 创建产品量化器
        pq = std::make_shared<IVFPQQuantizer>(dimension, m, k);
    }
                
    /**
     * 构建索引
     * @param base 基础向量数据
     * @param n 基础向量数量
     */
    void build(float* base, size_t n) {
        base_data = base;
        base_number = n;
        
        std::cout << "构建IVFPQ索引，策略: " << (pq_then_ivf ? "先PQ后IVF" : "先IVF后PQ") 
                 << ", 簇数量: " << nlist << std::endl;
        
        // 根据策略选择构建方法
        if (pq_then_ivf) {
            build_pq_then_ivf(base, n);
        } else {
            build_ivf_then_pq(base, n);
        }
        
        std::cout << "IVFPQ索引构建完成" << std::endl;
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
        out.write(reinterpret_cast<const char*>(&pq_then_ivf), sizeof(pq_then_ivf));
        
        // 保存PQ参数
        size_t m = pq->M;
        size_t k = pq->K;
        out.write(reinterpret_cast<const char*>(&m), sizeof(m));
        out.write(reinterpret_cast<const char*>(&k), sizeof(k));
        
        // 保存码本
        size_t codebook_size = pq->codebooks.size();
        out.write(reinterpret_cast<const char*>(&codebook_size), sizeof(codebook_size));
        out.write(reinterpret_cast<const char*>(pq->codebooks.data()), codebook_size * sizeof(float));
        
        // 保存簇中心
        size_t centroids_size = centroids.size();
        out.write(reinterpret_cast<const char*>(&centroids_size), sizeof(centroids_size));
        out.write(reinterpret_cast<const char*>(centroids.data()), centroids_size * sizeof(float));
        
        // 保存倒排列表和编码
        for (size_t i = 0; i < nlist; i++) {
            // 保存列表大小
            size_t list_size = inverted_lists[i].size();
            out.write(reinterpret_cast<const char*>(&list_size), sizeof(list_size));
            
            // 保存向量ID
            if (list_size > 0) {
                out.write(reinterpret_cast<const char*>(inverted_lists[i].data()), 
                         list_size * sizeof(uint32_t));
                
                // 保存PQ编码
                out.write(reinterpret_cast<const char*>(codes[i].data()), 
                         list_size * pq->M * sizeof(uint8_t));
            }
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
        in.read(reinterpret_cast<char*>(&pq_then_ivf), sizeof(pq_then_ivf));
        
        // 读取PQ参数
        size_t m, k;
        in.read(reinterpret_cast<char*>(&m), sizeof(m));
        in.read(reinterpret_cast<char*>(&k), sizeof(k));
        
        // 创建新的量化器
        pq = std::make_shared<IVFPQQuantizer>(d, m, k);
        
        // 读取码本
        size_t codebook_size;
        in.read(reinterpret_cast<char*>(&codebook_size), sizeof(codebook_size));
        pq->codebooks.resize(codebook_size);
        in.read(reinterpret_cast<char*>(pq->codebooks.data()), codebook_size * sizeof(float));
        
        // 读取簇中心
        size_t centroids_size;
        in.read(reinterpret_cast<char*>(&centroids_size), sizeof(centroids_size));
        centroids.resize(centroids_size);
        in.read(reinterpret_cast<char*>(centroids.data()), centroids_size * sizeof(float));
        
        // 读取倒排列表和编码
        inverted_lists.resize(nlist);
        codes.resize(nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            // 读取列表大小
            size_t list_size;
            in.read(reinterpret_cast<char*>(&list_size), sizeof(list_size));
            
            // 读取向量ID
            if (list_size > 0) {
                inverted_lists[i].resize(list_size);
                in.read(reinterpret_cast<char*>(inverted_lists[i].data()), 
                       list_size * sizeof(uint32_t));
                
                // 读取PQ编码
                codes[i].resize(list_size * pq->M);
                in.read(reinterpret_cast<char*>(codes[i].data()), 
                       list_size * pq->M * sizeof(uint8_t));
            }
        }
        
        // 初始化base_data为nullptr，防止在search中误用
        base_data = nullptr;
        
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
        // 根据索引策略选择搜索方法
        if (pq_then_ivf) {
            return search_pq_then_ivf(query, k);
        } else {
            return search_ivf_then_pq(query, k);
        }
    }
    
    /**
     * 设置查询时搜索的簇数量
     * @param n_probe 新的nprobe值
     */
    void set_nprobe(size_t n_probe) {
        nprobe = n_probe;
    }

private:
    // 先PQ后IVF的实现
    void build_pq_then_ivf(float* base, size_t n) {
        std::cout << "使用 先PQ后IVF 策略构建索引" << std::endl;
        
        // 1. 首先对所有数据进行PQ训练
        pq->train(base, n);
        
        // 2. 然后使用K-means++算法初始化簇中心
        init_centroids(base, n);
        
        // 3. 为每个簇的向量ID列表和编码分配空间
        inverted_lists.resize(nlist);
        codes.resize(nlist);
        
        // 4. 为每个向量计算PQ编码，并分配到最近的簇
        std::vector<uint8_t> all_codes(n * pq->M);
        
        #pragma omp parallel
        {
            // 每个线程的局部倒排列表和编码
            std::vector<std::vector<uint32_t>> local_lists(nlist);
            std::vector<std::vector<uint8_t>> local_codes(nlist);
            
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) {
                // 计算PQ编码
                std::vector<uint8_t> code(pq->M);
                pq->compute_code(base + i * d, code.data());
                
                // 复制到全局编码
                for (size_t j = 0; j < pq->M; j++) {
                    all_codes[i * pq->M + j] = code[j];
                }
                
                // 找到最近的簇
                uint32_t nearest_cluster = 0;
                float min_dist = std::numeric_limits<float>::max();
                
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
                
                // 将PQ编码添加到局部编码
                for (size_t j = 0; j < pq->M; j++) {
                    local_codes[nearest_cluster].push_back(code[j]);
                }
            }
            
            // 合并局部列表和编码
            #pragma omp critical
            {
                for (size_t i = 0; i < nlist; i++) {
                    // 将局部列表添加到全局列表
                    inverted_lists[i].insert(inverted_lists[i].end(), 
                                           local_lists[i].begin(), 
                                           local_lists[i].end());
                    
                    // 将局部编码添加到全局编码
                    codes[i].insert(codes[i].end(), 
                                   local_codes[i].begin(), 
                                   local_codes[i].end());
                }
            }
        }
        
        // 打印簇的统计信息
        print_cluster_stats();
    }
    
    // 先IVF后PQ的实现
    void build_ivf_then_pq(float* base, size_t n) {
        std::cout << "使用 先IVF后PQ 策略构建索引" << std::endl;
        
        // 1. 使用K-means++算法初始化簇中心
        init_centroids(base, n);
        
        // 2. 为每个簇的向量ID列表分配空间
        inverted_lists.resize(nlist);
        
        // 3. 将向量分配到最近的簇
        assign_vectors_to_clusters(base, n);
        
        // 4. 为每个簇的编码分配空间
        codes.resize(nlist);
        
        // 5. 对每个簇分别训练PQ量化器并编码
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nlist; i++) {
            const auto& cluster = inverted_lists[i];
            
            // 跳过空簇
            if (cluster.empty()) {
                continue;
            }
            
            // 为当前簇创建训练数据
            std::vector<float> cluster_data(cluster.size() * d);
            for (size_t j = 0; j < cluster.size(); j++) {
                std::copy(base + cluster[j] * d, 
                         base + (cluster[j] + 1) * d, 
                         cluster_data.data() + j * d);
            }
            
            // 创建该簇的PQ量化器
            IVFPQQuantizer local_pq(d, pq->M, pq->K);
            
            // 训练量化器
            local_pq.train(cluster_data.data(), cluster.size());
            
            // 计算每个向量的PQ编码
            codes[i].resize(cluster.size() * pq->M);
            for (size_t j = 0; j < cluster.size(); j++) {
                local_pq.compute_code(cluster_data.data() + j * d, 
                                    codes[i].data() + j * pq->M);
            }
            
            // 复制码本到全局码本（只保存最后一个簇的码本，用于示例）
            // 注意：实际应用中，你可能需要为每个簇保存一个独立的码本
            #pragma omp critical
            {
                std::copy(local_pq.codebooks.begin(), 
                         local_pq.codebooks.end(), 
                         pq->codebooks.begin());
            }
        }
        
        // 打印簇的统计信息
        print_cluster_stats();
    }
    
    // 查询实现 - 先PQ后IVF
    std::priority_queue<std::pair<float, uint32_t>> search_pq_then_ivf(const float* query, size_t k) const {
        // 标准化查询向量
        std::vector<float> query_norm(d);
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += query[j] * query[j];
        }
        norm = std::sqrt(norm) + 1e-8f;
        
        for (size_t j = 0; j < d; j++) {
            query_norm[j] = query[j] / norm;
        }
        
        // 计算查询向量到每个簇中心的距离
        std::vector<std::pair<float, uint32_t>> centroid_distances(nlist);
        
        #pragma omp parallel for
        for (int i = 0; i < nlist; i++) {
            // 计算内积距离(1 - 内积)
            float ip = 0;
            float centroid_norm = 0.0f;
            
            // 计算簇中心的L2范数
            for (size_t j = 0; j < d; j++) {
                centroid_norm += centroids[i * d + j] * centroids[i * d + j];
            }
            centroid_norm = std::sqrt(centroid_norm) + 1e-8f;
            
            // 计算标准化后的内积
            for (size_t j = 0; j < d; j++) {
                ip += query_norm[j] * (centroids[i * d + j] / centroid_norm);
            }
            centroid_distances[i] = {1 - ip, i};
        }
        
        // 选择距离最近的nprobe个簇
        std::partial_sort(centroid_distances.begin(), 
                         centroid_distances.begin() + std::min(nprobe, nlist), 
                         centroid_distances.end());
        
        // 为查询向量预计算与码本的距离表
        std::vector<float> distance_tables(pq->M * pq->K);
        pq->compute_distance_tables(query_norm.data(), distance_tables.data());
        
        // 结果堆 - 为了重排，获取更多的候选
        const size_t ef = k * 10; // 获取10倍的候选进行重排
        std::priority_queue<std::pair<float, uint32_t>> candidates;
        
        // 在选定的簇中搜索最近邻
        #pragma omp parallel
        {
            // 每个线程的局部结果
            std::priority_queue<std::pair<float, uint32_t>> local_result;
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < std::min(nprobe, nlist); i++) {
                uint32_t cluster_id = centroid_distances[i].second;
                const auto& cluster = inverted_lists[cluster_id];
                const auto& cluster_codes = codes[cluster_id];
                
                // 搜索当前簇中的所有向量
                for (size_t j = 0; j < cluster.size(); j++) {
                    uint32_t vec_id = cluster[j];
                    
                    // 使用距离表计算PQ码的距离
                    float dist = pq->compute_distance_from_tables(
                        cluster_codes.data() + j * pq->M, distance_tables.data());
                    
                    if (local_result.size() < ef) {
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
                    if (candidates.size() < ef) {
                        candidates.push(local_result.top());
                    } else if (local_result.top().first < candidates.top().first) {
                        candidates.push(local_result.top());
                        candidates.pop();
                    }
                    local_result.pop();
                }
            }
        }
        
        // 准备结果集
        std::priority_queue<std::pair<float, uint32_t>> result;
        std::vector<uint32_t> candidate_ids;
        
        // 提取候选ID
        while (!candidates.empty()) {
            candidate_ids.push_back(candidates.top().second);
            candidates.pop();
        }
        
        // 如果原始数据指针可用，则进行精确重排序
        if (base_data != nullptr && base_number > 0) {
            // 精确计算内积距离并重排
            #pragma omp parallel
            {
                std::priority_queue<std::pair<float, uint32_t>> local_result;
                
                #pragma omp for schedule(static)
                for (int i = 0; i < candidate_ids.size(); i++) {
                    uint32_t id = candidate_ids[i];
                    // 确保ID在有效范围内
                    if (id >= base_number) continue;
                    
                    float ip = 0;
                    float vec_norm = 0.0f;
                    
                    // 计算向量的L2范数
                    for (size_t j = 0; j < d; j++) {
                        vec_norm += base_data[id * d + j] * base_data[id * d + j];
                    }
                    vec_norm = std::sqrt(vec_norm) + 1e-8f;
                    
                    // 计算标准化后的内积
                    for (size_t j = 0; j < d; j++) {
                        ip += query_norm[j] * (base_data[id * d + j] / vec_norm);
                    }
                    
                    // 1 - 内积作为距离
                    float dist = 1 - ip;
                    
                    if (local_result.size() < k) {
                        local_result.push({dist, id});
                    } else if (dist < local_result.top().first) {
                        local_result.push({dist, id});
                        local_result.pop();
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
        } else {
            // 如果原始数据不可用，直接使用PQ距离
            for (uint32_t vec_id : candidate_ids) {
                // 计算当前候选向量所在的簇和在簇内的位置
                uint32_t cluster_id = 0;
                size_t pos_in_cluster = 0;
                bool found = false;
                
                for (size_t i = 0; i < nlist && !found; i++) {
                    for (size_t j = 0; j < inverted_lists[i].size(); j++) {
                        if (inverted_lists[i][j] == vec_id) {
                            cluster_id = i;
                            pos_in_cluster = j;
                            found = true;
                            break;
                        }
                    }
                }
                
                if (found) {
                    // 计算PQ距离
                    float dist = pq->compute_distance_from_tables(
                        codes[cluster_id].data() + pos_in_cluster * pq->M, 
                        distance_tables.data());
                    
                    if (result.size() < k) {
                        result.push({dist, vec_id});
                    } else if (dist < result.top().first) {
                        result.push({dist, vec_id});
                        result.pop();
                    }
                }
            }
        }
        
        return result;
    }
    
    // 查询实现 - 先IVF后PQ
    std::priority_queue<std::pair<float, uint32_t>> search_ivf_then_pq(const float* query, size_t k) const {
        // 标准化查询向量
        std::vector<float> query_norm(d);
        float norm = 0.0f;
        for (size_t j = 0; j < d; j++) {
            norm += query[j] * query[j];
        }
        norm = std::sqrt(norm) + 1e-8f;
        
        for (size_t j = 0; j < d; j++) {
            query_norm[j] = query[j] / norm;
        }
        
        // 计算查询向量到每个簇中心的距离
        std::vector<std::pair<float, uint32_t>> centroid_distances(nlist);
        
        #pragma omp parallel for
        for (int i = 0; i < nlist; i++) {
            // 计算内积距离(1 - 内积)
            float ip = 0;
            float centroid_norm = 0.0f;
            
            // 计算簇中心的L2范数
            for (size_t j = 0; j < d; j++) {
                centroid_norm += centroids[i * d + j] * centroids[i * d + j];
            }
            centroid_norm = std::sqrt(centroid_norm) + 1e-8f;
            
            // 计算标准化后的内积
            for (size_t j = 0; j < d; j++) {
                ip += query_norm[j] * (centroids[i * d + j] / centroid_norm);
            }
            centroid_distances[i] = {1 - ip, i};
        }
        
        // 选择距离最近的nprobe个簇
        std::partial_sort(centroid_distances.begin(), 
                         centroid_distances.begin() + std::min(nprobe, nlist), 
                         centroid_distances.end());
        
        // 结果堆 - 为了重排，获取更多的候选
        const size_t ef = k * 10; // 获取10倍的候选进行重排
        std::priority_queue<std::pair<float, uint32_t>> candidates;
        
        // 为查询向量预计算与码本的距离表
        std::vector<float> distance_tables(pq->M * pq->K);
        pq->compute_distance_tables(query_norm.data(), distance_tables.data());
        
        // 在选定的簇中搜索最近邻
        #pragma omp parallel
        {
            // 每个线程的局部结果
            std::priority_queue<std::pair<float, uint32_t>> local_result;
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < std::min(nprobe, nlist); i++) {
                uint32_t cluster_id = centroid_distances[i].second;
                const auto& cluster = inverted_lists[cluster_id];
                const auto& cluster_codes = codes[cluster_id];
                
                // 搜索当前簇中的所有向量
                for (size_t j = 0; j < cluster.size(); j++) {
                    uint32_t vec_id = cluster[j];
                    
                    // 使用距离表计算PQ码的距离
                    float dist = pq->compute_distance_from_tables(
                        cluster_codes.data() + j * pq->M, distance_tables.data());
                    
                    if (local_result.size() < ef) {
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
                    if (candidates.size() < ef) {
                        candidates.push(local_result.top());
                    } else if (local_result.top().first < candidates.top().first) {
                        candidates.push(local_result.top());
                        candidates.pop();
                    }
                    local_result.pop();
                }
            }
        }
        
        // 准备结果集
        std::priority_queue<std::pair<float, uint32_t>> result;
        std::vector<uint32_t> candidate_ids;
        
        // 提取候选ID
        while (!candidates.empty()) {
            candidate_ids.push_back(candidates.top().second);
            candidates.pop();
        }
        
        // 如果原始数据指针可用，则进行精确重排序
        if (base_data != nullptr && base_number > 0) {
            // 精确计算内积距离并重排
            #pragma omp parallel
            {
                std::priority_queue<std::pair<float, uint32_t>> local_result;
                
                #pragma omp for schedule(static)
                for (int i = 0; i < candidate_ids.size(); i++) {
                    uint32_t id = candidate_ids[i];
                    // 确保ID在有效范围内
                    if (id >= base_number) continue;
                    
                    float ip = 0;
                    float vec_norm = 0.0f;
                    
                    // 计算向量的L2范数
                    for (size_t j = 0; j < d; j++) {
                        vec_norm += base_data[id * d + j] * base_data[id * d + j];
                    }
                    vec_norm = std::sqrt(vec_norm) + 1e-8f;
                    
                    // 计算标准化后的内积
                    for (size_t j = 0; j < d; j++) {
                        ip += query_norm[j] * (base_data[id * d + j] / vec_norm);
                    }
                    
                    // 1 - 内积作为距离
                    float dist = 1 - ip;
                    
                    if (local_result.size() < k) {
                        local_result.push({dist, id});
                    } else if (dist < local_result.top().first) {
                        local_result.push({dist, id});
                        local_result.pop();
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
        } else {
            // 如果原始数据不可用，直接使用PQ距离
            for (uint32_t vec_id : candidate_ids) {
                // 计算当前候选向量所在的簇和在簇内的位置
                uint32_t cluster_id = 0;
                size_t pos_in_cluster = 0;
                bool found = false;
                
                for (size_t i = 0; i < nlist && !found; i++) {
                    for (size_t j = 0; j < inverted_lists[i].size(); j++) {
                        if (inverted_lists[i][j] == vec_id) {
                            cluster_id = i;
                            pos_in_cluster = j;
                            found = true;
                            break;
                        }
                    }
                }
                
                if (found) {
                    // 计算PQ距离
                    float dist = pq->compute_distance_from_tables(
                        codes[cluster_id].data() + pos_in_cluster * pq->M, 
                        distance_tables.data());
                    
                    if (result.size() < k) {
                        result.push({dist, vec_id});
                    } else if (dist < result.top().first) {
                        result.push({dist, vec_id});
                        result.pop();
                    }
                }
            }
        }
        
        return result;
    }
    
    // 初始化聚类中心 - 使用k-means++算法
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
    
    // 将向量分配到最近的簇
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
    
    // 打印簇的统计信息
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

// 全局变量用于缓存索引
namespace {
    IVFPQIndex* g_ivfpq_index = nullptr;
    bool g_ivfpq_initialized = false;
    bool g_use_pq_then_ivf = false;
}

/**
 * 构建IVF-PQ索引 - 先PQ后IVF
 */
void build_ivfpq_pq_then_ivf_index(float* base, size_t base_number, size_t vecdim, 
                           size_t nlist, size_t nprobe,
                           size_t m, size_t k) {
    // 创建IVF-PQ索引
    IVFPQIndex* ivfpq_index = new IVFPQIndex(vecdim, nlist, nprobe, true, m, k);
    
    // 构建索引
    ivfpq_index->build(base, base_number);
    
    // 保存索引
    ivfpq_index->save("files/ivfpq_pq_then_ivf.index");
    
    // 释放内存
    delete ivfpq_index;
}

/**
 * 构建IVF-PQ索引 - 先IVF后PQ
 */
void build_ivfpq_ivf_then_pq_index(float* base, size_t base_number, size_t vecdim, 
                           size_t nlist, size_t nprobe,
                           size_t m, size_t k) {
    // 创建IVF-PQ索引
    IVFPQIndex* ivfpq_index = new IVFPQIndex(vecdim, nlist, nprobe, false, m, k);
    
    // 构建索引
    ivfpq_index->build(base, base_number);
    
    // 保存索引
    ivfpq_index->save("files/ivfpq_ivf_then_pq.index");
    
    // 释放内存
    delete ivfpq_index;
}

/**
 * 确保IVF-PQ索引已初始化
 */
void ensure_ivfpq_index_initialized(const std::string& index_path, bool use_pq_then_ivf) {
    if (g_ivfpq_initialized && g_use_pq_then_ivf == use_pq_then_ivf) {
        return;
    }
    
    // 清理之前的索引（如果有）
    if (g_ivfpq_index) {
        delete g_ivfpq_index;
        g_ivfpq_index = nullptr;
        g_ivfpq_initialized = false;
    }
    
    // 检查索引文件是否存在
    std::ifstream file(index_path);
    if (!file.is_open()) {
        std::cerr << "IVF-PQ索引文件不存在: " << index_path << std::endl;
        std::cerr << "请先调用 build_ivfpq_" << (use_pq_then_ivf ? "pq_then_ivf" : "ivf_then_pq") 
                 << "_index 构建索引" << std::endl;
        return;
    }
    file.close();
    
    // 加载索引
    g_ivfpq_index = new IVFPQIndex(0);
    g_ivfpq_index->load(index_path);
    g_ivfpq_initialized = true;
    g_use_pq_then_ivf = use_pq_then_ivf;
}

/**
 * 基于IVF-PQ实现的最近邻查询，使用内积距离作为相似度指标
 */
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_ivfpq_Inner_Product(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, 
    size_t nprobe, bool use_pq_then_ivf) {
    
    // 确定索引文件路径
    std::string index_path = use_pq_then_ivf ? 
                           "files/ivfpq_pq_then_ivf.index" : 
                           "files/ivfpq_ivf_then_pq.index";
    
    // 确保索引已初始化
    ensure_ivfpq_index_initialized(index_path, use_pq_then_ivf);
    
    // 如果索引未成功初始化，回退到暴力搜索
    if (!g_ivfpq_initialized) {
        std::cerr << "IVF-PQ索引未初始化，回退到暴力搜索" << std::endl;
        
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
    g_ivfpq_index->set_nprobe(nprobe);
    
    // 设置原始数据指针，用于重排序
    g_ivfpq_index->base_data = base;
    
    // 使用IVF-PQ索引搜索
    return g_ivfpq_index->search(query, k);
}

// 全局变量用于缓存索引
namespace {
    IVFPQIndex* g_ivfpq_index_fixed = nullptr;
    bool g_ivfpq_initialized_fixed = false;
    bool g_use_pq_then_ivf_fixed = false;
}

/**
 * 确保IVF-PQ索引已初始化
 */
void ensure_ivfpq_index_initialized_fixed(const std::string& index_path, bool use_pq_then_ivf) {
    if (g_ivfpq_initialized_fixed && g_use_pq_then_ivf_fixed == use_pq_then_ivf) {
        return;
    }
    
    // 清理之前的索引（如果有）
    if (g_ivfpq_index_fixed) {
        delete g_ivfpq_index_fixed;
        g_ivfpq_index_fixed = nullptr;
        g_ivfpq_initialized_fixed = false;
    }
    
    // 检查索引文件是否存在
    std::ifstream file(index_path);
    if (!file.is_open()) {
        std::cerr << "IVF-PQ索引文件不存在: " << index_path << std::endl;
        std::cerr << "请先调用 build_ivfpq_" << (use_pq_then_ivf ? "pq_then_ivf" : "ivf_then_pq") 
                 << "_index_fixed 构建索引" << std::endl;
        return;
    }
    file.close();
    
    // 加载索引
    g_ivfpq_index_fixed = new IVFPQIndex(0);
    g_ivfpq_index_fixed->load(index_path);
    g_ivfpq_initialized_fixed = true;
    g_use_pq_then_ivf_fixed = use_pq_then_ivf;
}

/**
 * 构建IVF-PQ索引 - 先PQ后IVF (修复版)
 */
void build_ivfpq_pq_then_ivf_index_fixed(float* base, size_t base_number, size_t vecdim, 
                                  size_t nlist, size_t nprobe,
                                  size_t m, size_t k) {
    // 创建IVF-PQ索引
    IVFPQIndex* ivfpq_index = new IVFPQIndex(vecdim, nlist, nprobe, true, m, k);
    
    // 构建索引
    ivfpq_index->build(base, base_number);
    
    // 保存索引
    ivfpq_index->save("files/ivfpq_pq_then_ivf_fixed.index");
    
    // 释放内存
    delete ivfpq_index;
}

/**
 * 构建IVF-PQ索引 - 先IVF后PQ (修复版)
 */
void build_ivfpq_ivf_then_pq_index_fixed(float* base, size_t base_number, size_t vecdim, 
                                  size_t nlist, size_t nprobe,
                                  size_t m, size_t k) {
    // 创建IVF-PQ索引
    IVFPQIndex* ivfpq_index = new IVFPQIndex(vecdim, nlist, nprobe, false, m, k);
    
    // 构建索引
    ivfpq_index->build(base, base_number);
    
    // 保存索引
    ivfpq_index->save("files/ivfpq_ivf_then_pq_fixed.index");
    
    // 释放内存
    delete ivfpq_index;
}

/**
 * 基于IVF-PQ实现的最近邻查询，使用内积距离作为相似度指标 (修复版)
 */
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_ivfpq_Inner_Product_fixed(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, 
    size_t nprobe, bool use_pq_then_ivf) {
    
    // 确定索引文件路径
    std::string index_path = use_pq_then_ivf ? 
                           "files/ivfpq_pq_then_ivf_fixed.index" : 
                           "files/ivfpq_ivf_then_pq_fixed.index";
    
    // 确保索引已初始化
    ensure_ivfpq_index_initialized_fixed(index_path, use_pq_then_ivf);
    
    // 如果索引未成功初始化，回退到暴力搜索
    if (!g_ivfpq_initialized_fixed) {
        std::cerr << "IVF-PQ索引未初始化，回退到暴力搜索" << std::endl;
        
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
    g_ivfpq_index_fixed->set_nprobe(nprobe);
    
    // 设置原始数据指针，用于重排序
    g_ivfpq_index_fixed->base_data = base;
    
    // 使用IVF-PQ索引搜索
    return g_ivfpq_index_fixed->search(query, k);
} 