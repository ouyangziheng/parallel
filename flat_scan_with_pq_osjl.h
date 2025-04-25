#pragma once
#include <arm_neon.h>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <functional>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <limits>
#include <cassert>

#include "simd_utils.h"

// 产品量化（PQ）算法实现 - 使用欧氏距离
class ProductQuantizer {
public:
    // 构造函数
    ProductQuantizer(size_t d, size_t M = 4, size_t K = 256) 
    : dim(d), M(M), K(K) {
        // 确保维度可以被M整除
        if (dim % M != 0) {
            throw std::runtime_error("维度必须被子空间数量整除");
        }
        
        // 每个子向量的维度
        d_sub = dim / M;
        
        // 初始化编码表 (M个子空间，每个子空间K个类中心)
        codebooks.resize(M);
        for (size_t m = 0; m < M; m++) {
            codebooks[m].resize(K * d_sub);
        }
        
        // 初始化距离表
        distance_tables.resize(M);
        for (size_t m = 0; m < M; m++) {
            distance_tables[m].resize(K);
        }
    }
    
    // 训练量化器
    void train(const std::vector<float>& data, size_t n) {
        // 随机数生成器
        std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        
        // 数据的维度检查
        if (data.size() != n * dim) {
            throw std::runtime_error("数据大小与指定的样本数和维度不匹配");
        }
        
        // 对每个子空间单独进行K-means聚类
        #pragma omp parallel for
        for (size_t m = 0; m < M; m++) {
            // 提取当前子空间的数据
            std::vector<float> sub_data(n * d_sub);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < d_sub; j++) {
                    sub_data[i * d_sub + j] = data[i * dim + m * d_sub + j];
                }
            }
            
            // K-means聚类
            k_means(sub_data.data(), n, d_sub, K, codebooks[m].data(), rng);
        }
    }
    
    // 编码单个向量 - 使用欧氏距离
    void encode(const float* x, uint8_t* code) const {
        for (size_t m = 0; m < M; m++) {
            float min_dist = std::numeric_limits<float>::max();
            uint8_t best_idx = 0;
            
            // 计算当前子向量到所有聚类中心的欧式距离
            for (size_t k = 0; k < K; k++) {
                float dist = 0.0f;
                
                // 使用SIMD优化欧氏距离计算
                if (d_sub >= 4) {
                    size_t j = 0;
                    float32x4_t sum_vec = vdupq_n_f32(0);
                    
                    for (; j + 3 < d_sub; j += 4) {
                        float32x4_t x_vec = vld1q_f32(x + m * d_sub + j);
                        float32x4_t c_vec = vld1q_f32(&codebooks[m][k * d_sub + j]);
                        float32x4_t diff = vsubq_f32(x_vec, c_vec);
                        // 平方和累加
                        sum_vec = vmlaq_f32(sum_vec, diff, diff);
                    }
                    
                    // 合并部分和
                    float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                    float32x2_t sum1 = vpadd_f32(sum2, sum2);
                    dist = vget_lane_f32(sum1, 0);
                    
                    // 处理剩余元素
                    for (; j < d_sub; j++) {
                        float diff = x[m * d_sub + j] - codebooks[m][k * d_sub + j];
                        dist += diff * diff;
                    }
                } else {
                    // 标量计算
                    for (size_t j = 0; j < d_sub; j++) {
                        float diff = x[m * d_sub + j] - codebooks[m][k * d_sub + j];
                        dist += diff * diff;
                    }
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = k;
                }
            }
            
            // 存储最佳匹配的聚类中心索引
            code[m] = best_idx;
        }
    }
    
    // 批量编码向量
    void encode_dataset(const float* dataset, size_t n, uint8_t* codes) const {
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            encode(dataset + i * dim, codes + i * M);
        }
    }
    
    // 计算查询向量与所有子空间的聚类中心的距离表 - 使用欧氏距离
    void compute_distance_tables(const float* query) const {
        for (size_t m = 0; m < M; m++) {
            for (size_t k = 0; k < K; k++) {
                float dist = 0.0f;
                
                // 使用SIMD优化欧氏距离计算
                if (d_sub >= 4) {
                    size_t j = 0;
                    float32x4_t sum_vec = vdupq_n_f32(0);
                    
                    for (; j + 3 < d_sub; j += 4) {
                        float32x4_t q_vec = vld1q_f32(query + m * d_sub + j);
                        float32x4_t c_vec = vld1q_f32(&codebooks[m][k * d_sub + j]);
                        float32x4_t diff = vsubq_f32(q_vec, c_vec);
                        // 平方和累加
                        sum_vec = vmlaq_f32(sum_vec, diff, diff);
                    }
                    
                    // 合并部分和
                    float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                    float32x2_t sum1 = vpadd_f32(sum2, sum2);
                    dist = vget_lane_f32(sum1, 0);
                    
                    // 处理剩余元素
                    for (; j < d_sub; j++) {
                        float diff = query[m * d_sub + j] - codebooks[m][k * d_sub + j];
                        dist += diff * diff;
                    }
                } else {
                    // 对于较小的维度，直接计算
                    for (size_t j = 0; j < d_sub; j++) {
                        float diff = query[m * d_sub + j] - codebooks[m][k * d_sub + j];
                        dist += diff * diff;
                    }
                }
                
                const_cast<float&>(distance_tables[m][k]) = dist;
            }
        }
    }
    
    // 使用预计算的距离表计算查询向量与PQ编码向量之间的近似距离
    float compute_distance(const uint8_t* code) const {
        float dist = 0;
        for (size_t m = 0; m < M; m++) {
            dist += distance_tables[m][code[m]];
        }
        return dist;
    }
    
    // 保存量化器到文件
    void save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        if (!fout) {
            throw std::runtime_error("无法打开文件进行写入");
        }
        
        // 写入维度和参数
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        fout.write(reinterpret_cast<const char*>(&M), sizeof(M));
        fout.write(reinterpret_cast<const char*>(&K), sizeof(K));
        
        // 写入编码本
        for (size_t m = 0; m < M; m++) {
            fout.write(reinterpret_cast<const char*>(codebooks[m].data()), 
                       codebooks[m].size() * sizeof(float));
        }
        
        fout.close();
    }
    
    // 从文件加载量化器
    void load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("无法打开文件进行读取");
        }
        
        // 读取维度和参数
        fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        fin.read(reinterpret_cast<char*>(&M), sizeof(M));
        fin.read(reinterpret_cast<char*>(&K), sizeof(K));
        
        // 计算子向量维度
        d_sub = dim / M;
        
        // 调整编码本和距离表大小
        codebooks.resize(M);
        for (size_t m = 0; m < M; m++) {
            codebooks[m].resize(K * d_sub);
        }
        
        distance_tables.resize(M);
        for (size_t m = 0; m < M; m++) {
            distance_tables[m].resize(K);
        }
        
        // 读取编码本
        for (size_t m = 0; m < M; m++) {
            fin.read(reinterpret_cast<char*>(codebooks[m].data()), 
                     codebooks[m].size() * sizeof(float));
        }
        
        fin.close();
    }

private:
    void k_means(const float* data, size_t n, size_t d, size_t k, 
                float* centroids, std::mt19937& rng) {
        // 随机选择k个初始聚类中心
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; i++) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // 初始化聚类中心（直接使用随机选择的数据点）
        for (size_t i = 0; i < k && i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                centroids[i * d + j] = data[indices[i] * d + j];
            }
        }
        
        // 分配数组
        std::vector<size_t> assignments(n);
        std::vector<size_t> counts(k);
        std::vector<float> new_centroids(k * d);
        
        // 最大迭代次数
        const size_t max_iter = 20;
        const float threshold = 1e-4f;  // 收敛阈值
        
        for (size_t iter = 0; iter < max_iter; iter++) {
            // 清零
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(new_centroids.begin(), new_centroids.end(), 0);
            
            // 分配步骤 - 为每个数据点找到最近的中心
            for (size_t i = 0; i < n; i++) {
                float min_dist = std::numeric_limits<float>::max();
                size_t best_centroid = 0;
                
                // 找到最近的聚类中心
                for (size_t j = 0; j < k; j++) {
                    float dist = 0;
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        float diff = data[i * d + d_idx] - centroids[j * d + d_idx];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_centroid = j;
                    }
                }
                
                // 分配数据点
                assignments[i] = best_centroid;
                counts[best_centroid]++;
                
                // 累加数据点到新中心
                for (size_t d_idx = 0; d_idx < d; d_idx++) {
                    new_centroids[best_centroid * d + d_idx] += data[i * d + d_idx];
                }
            }
            
            // 更新步骤 - 计算新的中心
            bool changed = false;
            for (size_t j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    float max_change = 0.0f;
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        float old_val = centroids[j * d + d_idx];
                        float new_val = new_centroids[j * d + d_idx] / counts[j];
                        float change = std::abs(new_val - old_val);
                        max_change = std::max(max_change, change);
                        centroids[j * d + d_idx] = new_val;
                    }
                    if (max_change > threshold) {
                        changed = true;
                    }
                }
            }
            
            // 如果聚类中心稳定，提前退出
            if (!changed) {
                break;
            }
        }
        
        // 处理空聚类
        for (size_t j = 0; j < k; j++) {
            if (counts[j] == 0) {
                // 找到最大数据点数的聚类
                size_t max_count_idx = 0;
                for (size_t i = 1; i < k; i++) {
                    if (counts[i] > counts[max_count_idx]) {
                        max_count_idx = i;
                    }
                }
                
                // 分割最大聚类的数据点
                if (counts[max_count_idx] > 1) {
                    counts[j] = counts[max_count_idx] / 2;
                    counts[max_count_idx] -= counts[j];
                    
                    // 复制中心并添加一些随机扰动
                    for (size_t d_idx = 0; d_idx < d; d_idx++) {
                        float noise = (rng() % 1000) / 10000.0f - 0.05f; // 添加 -0.05 到 0.05 之间的噪声
                        centroids[j * d + d_idx] = centroids[max_count_idx * d + d_idx] + noise;
                    }
                }
            }
        }
    }
    
    size_t dim;      // 向量维度
    size_t M;        // 子空间数量
    size_t K;        // 每个子空间的聚类中心数量
    size_t d_sub;    // 每个子向量的维度
    
    // 编码本 [M][K * d_sub]
    std::vector<std::vector<float>> codebooks;
    
    // 距离表 [M][K]
    mutable std::vector<std::vector<float>> distance_tables;
};

// PQ索引类，使用欧氏距离的PQ算法实现
class PQIndex {
public:
    PQIndex(size_t d, size_t M = 4, size_t K = 256) 
    : dim(d), pq(d, M, K), M(M) {}
    
    // 构建索引
    void build(const float* data, size_t n) {
        std::cout << "开始构建PQ索引，数据点数: " << n << ", 维度: " << dim << std::endl;
        
        // 保存原始数据向量的数量
        n_data = n;
        
        // 将输入数据转换为vector进行训练
        std::vector<float> train_data(data, data + n * dim);
        

        // 这里用简单的线性变换将数据缩放
        preprocess_data(train_data.data(), n);
        
        // 训练PQ量化器
        std::cout << "训练PQ量化器..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        pq.train(train_data, n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "训练完成，耗时: " << duration.count() << " 毫秒" << std::endl;
        
        // 编码所有数据
        codes.resize(n * M);
        std::cout << "编码数据..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        pq.encode_dataset(train_data.data(), n, codes.data());
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "编码完成，耗时: " << duration.count() << " 毫秒" << std::endl;
        
        // 为两阶段检索保存原始数据
        original_data.resize(n * dim);
        std::copy(data, data + n * dim, original_data.begin());
        
        std::cout << "PQ索引构建完成" << std::endl;
    }
    
    // 预处理数据
    void preprocess_data(float* data, size_t n) {
        // 计算均值
        std::vector<float> mean(dim, 0);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < dim; j++) {
                mean[j] += data[i * dim + j];
            }
        }
        
        for (size_t j = 0; j < dim; j++) {
            mean[j] /= n;
        }
        
        // 均值中心化，统一处理数据
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < dim; j++) {
                data[i * dim + j] -= mean[j];
            }
        }
    }
    
    // 搜索最近邻 - 使用欧氏距离
    std::priority_queue<std::pair<float, uint32_t>> search(const float* query, size_t k) const {
        // 最终结果
        std::priority_queue<std::pair<float, uint32_t>> results;
        
        // 预处理查询向量 (同样的预处理方式)
        std::vector<float> query_processed(dim);
        for (size_t i = 0; i < dim; i++) {
            query_processed[i] = query[i];
        }
        
        // 计算距离表
        pq.compute_distance_tables(query_processed.data());
        
        // 当数据量小时直接计算精确距离
        if (n_data <= 10000) {
            // 精确搜索
            for (size_t i = 0; i < n_data; i++) {
                // 计算欧氏距离
                float dist = 0.0f;
                for (size_t j = 0; j < dim; j++) {
                    float diff = query[j] - original_data[i * dim + j];
                    dist += diff * diff;
                }
                
                if (results.size() < k) {
                    results.push({dist, i});
                } else if (dist < results.top().first) {
                    results.pop();
                    results.push({dist, i});
                }
            }
        } else {
            // 两阶段检索 
            // 第一阶段：PQ近似搜索
            const size_t candidate_count = k * 10;  // 选取更多候选进行重排序
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(candidate_count);
            
            for (size_t i = 0; i < n_data; i++) {
                float dist = pq.compute_distance(codes.data() + i * M);
                
                if (candidates.size() < candidate_count) {
                    candidates.push_back({dist, i});
                    if (candidates.size() == candidate_count) {
                        std::make_heap(candidates.begin(), candidates.end());
                    }
                } else if (dist < candidates.front().first) {
                    std::pop_heap(candidates.begin(), candidates.end());
                    candidates.back() = {dist, i};
                    std::push_heap(candidates.begin(), candidates.end());
                }
            }
            
            // 第二阶段：精确计算欧氏距离
            for (const auto& candidate : candidates) {
                uint32_t idx = candidate.second;
                
                // 计算精确欧氏距离
                float dist = 0.0f;
                for (size_t j = 0; j < dim; j++) {
                    float diff = query[j] - original_data[idx * dim + j];
                    dist += diff * diff;
                }
                
                if (results.size() < k) {
                    results.push({dist, idx});
                } else if (dist < results.top().first) {
                    results.pop();
                    results.push({dist, idx});
                }
            }
        }
        
        return results;
    }
    
    // 保存索引到文件
    void save(const std::string& filename) const {
        std::ofstream fout(filename, std::ios::binary);
        if (!fout) {
            throw std::runtime_error("无法打开文件进行写入");
        }
        
        // 写入索引参数
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        fout.write(reinterpret_cast<const char*>(&M), sizeof(M));
        fout.write(reinterpret_cast<const char*>(&n_data), sizeof(n_data));
        
        // 写入编码
        fout.write(reinterpret_cast<const char*>(codes.data()), codes.size() * sizeof(uint8_t));
        
        // 写入原始数据 (可能会很大)
        fout.write(reinterpret_cast<const char*>(original_data.data()), original_data.size() * sizeof(float));
        
        fout.close();
        
        // 保存PQ量化器
        pq.save(filename + ".pq");
        
        std::cout << "索引已保存到 " << filename << std::endl;
    }
    
    // 从文件加载索引
    void load(const std::string& filename) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("无法打开文件进行读取");
        }
        
        // 读取索引参数
        fin.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        fin.read(reinterpret_cast<char*>(&M), sizeof(M));
        fin.read(reinterpret_cast<char*>(&n_data), sizeof(n_data));
        
        // 调整编码数组大小
        codes.resize(n_data * M);
        
        // 读取编码
        fin.read(reinterpret_cast<char*>(codes.data()), codes.size() * sizeof(uint8_t));
        
        // 读取原始数据
        original_data.resize(n_data * dim);
        fin.read(reinterpret_cast<char*>(original_data.data()), original_data.size() * sizeof(float));
        
        fin.close();
        
        // 加载PQ量化器
        pq.load(filename + ".pq");
        
        std::cout << "索引已从 " << filename << " 加载" << std::endl;
    }
    
private:
    size_t dim;           // 向量维度
    size_t M;             // 子空间数量
    size_t n_data;        // 数据点数量
    ProductQuantizer pq;  // 产品量化器
    std::vector<uint8_t> codes;  // 编码数据
    std::vector<float> original_data;  // 原始数据用于精确搜索
};

// 全局PQ索引指针
PQIndex* global_pq_index = nullptr;

// 使用PQ在基础向量集上搜索查询向量的k个最近邻 - 欧氏距离版本
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_pq(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    
    // 检查数据有效性
    assert(base != nullptr);
    assert(query != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(k > 0);
    
    // 如果PQ索引尚未初始化，则创建并构建索引
    if (global_pq_index == nullptr) {
        // 为典型的向量维度设置合理的PQ参数
        // vecdim=128时，设置M=4，每段32维
        size_t M = 4;  // 子空间数量
        size_t K = 256; // 每个子空间的聚类中心数量
        
        std::cout << "创建新的PQ索引，M=" << M << ", K=" << K << std::endl;
        
        // 创建PQ索引
        global_pq_index = new PQIndex(vecdim, M, K);
        
        // 构建索引
        global_pq_index->build(base, base_number);
        
        // 保存索引以便后续使用
        global_pq_index->save("files/pq.index");
    }
    
    // 使用PQ索引搜索k个最近邻
    return global_pq_index->search(query, k);
}