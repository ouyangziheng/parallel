#include<arm_neon.h>
#include"simd_utils.h"
#include<queue>
#pragma once

std::priority_queue<std::pair<float, uint32_t>> flat_search_with_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (int i = 0; i < base_number; ++i) {
        // 使用 SIMD 加速的内积计算
        float dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);  // 调用 SIMD 内积函数

        // 保持 top-k 最小距离
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
