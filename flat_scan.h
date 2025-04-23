// 禁止修改该文件
#pragma once
#include <queue>

/**
 * @brief 基于暴力搜索实现的最近邻查询，使用内积距离（IP）作为相似度指标。
 *        在 DEEP100K 数据集上，使用 1 - inner_product 作为距离衡量标准。
 * 
 * @param base        基础数据集合
 * 长度为 base_number * vecdim，按行排列（每个向量连续存储）
 * @param query       查询向量，长度为 vecdim
 * @param base_number 向量库中 base 向量的总数
 * @param vecdim      向量维度（每个向量的长度）
 * @param k           需要返回的最相似向量个数（Top-k）
 * 
 * @return std::priority_queue<std::pair<float, uint32_t>> 
 *         返回一个最大堆，堆中存放的是 {距离, 向量编号}，用于表示距离 query 最近的 k 个向量
 */
std::priority_queue<std::pair<float, uint32_t> > flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i) {
        float dis = 0;

        // DEEP100K数据集使用ip距离 
        // 计算内积距离 base[i] 和 query
        // 实际上 d+i*vecdim 是 base[i][d]
        // base[i][d] * query[d] 代表内积
        for(int d = 0; d < vecdim; ++d) {
            dis += base[d + i*vecdim]*query[d];
        }
        dis = 1 - dis;

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

