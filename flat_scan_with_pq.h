#pragma once
#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <queue>

#include "simd_utils.h"

template <typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

// 标量量化函数 - 将浮点数向量量化为8位整数
void quantize_vector(const float* src, uint8_t* dst, float& scale,
                     float& offset, size_t vecdim) {
    // 找出最大值和最小值
    float min_val = src[0];
    float max_val = src[0];

    for (size_t i = 1; i < vecdim; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }

    // 计算缩放因子和偏移量
    scale = (max_val - min_val) / 255.0f;
    offset = min_val;

    if (min_val == max_val) {
        scale = 1.0f;
        std::fill(dst, dst + vecdim, 0);  // 量化结果为0
        offset = min_val;
        return;
    }

    // 量化
    for (size_t i = 0; i < vecdim; i++) {
        float normalized = (src[i] - offset) / scale;
        dst[i] = static_cast<uint8_t>(clamp(std::round(normalized), 0.0f, 255.0f));
    }
}


// 使用量化计算内积距离
float inner_product_quantized(const uint8_t* b1, const uint8_t* b2,
                              const float scale1, const float offset1,
                              const float scale2, const float offset2,
                              size_t vecdim) {
    int32_t sum = 0;

    // SIMD优化的8位整数内积计算
    size_t i = 0;

    // 使用NEON进行SIMD加速
    int32x4_t sum_vec = vdupq_n_s32(0);

    for (; i + 15 < vecdim; i += 16) {
        // 128 = 8 * 16 每个点均为 8 位整数
        uint8x16_t v1 = vld1q_u8(b1 + i);
        uint8x16_t v2 = vld1q_u8(b2 + i);

        // 将8位无符号整数转换为16位有符号整数
        // 防止超出范围
        // vget_low_u8(v1) 用于加载前 8 个
        int16x8_t low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v1)));
        int16x8_t high1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v1)));
        int16x8_t low2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v2)));
        int16x8_t high2 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v2)));

        // 计算乘积并累加
        sum_vec = vmlal_s16(sum_vec, vget_low_s16(low1), vget_low_s16(low2));
        sum_vec = vmlal_s16(sum_vec, vget_high_s16(low1), vget_high_s16(low2));
        sum_vec = vmlal_s16(sum_vec, vget_low_s16(high1), vget_low_s16(high2));
        sum_vec = vmlal_s16(sum_vec, vget_high_s16(high1), vget_high_s16(high2));
    }

    // 将向量中的元素相加
    sum = vaddvq_s32(sum_vec); 


    // 处理剩余元素
    for (; i < vecdim; i++) {
        sum += static_cast<int32_t>(b1[i]) * static_cast<int32_t>(b2[i]);
    }

    // 反量化结果
    float result = sum * scale1 * scale2 + vecdim * offset1 * offset2;
    return 1.0f - result;  // 返回内积距离
}


// 使用标量量化的扁平扫描搜索函数
std::priority_queue<std::pair<float, uint32_t>> flat_search_with_pq(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
      
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 量化查询向量
    uint8_t* query_quantized = new uint8_t[vecdim];
    float query_scale, query_offset;
    quantize_vector(query, query_quantized, query_scale, query_offset, vecdim);

    // 为所有基础向量分配量化内存
    uint8_t* base_quantized = new uint8_t[base_number * vecdim];
    float* scales = new float[base_number];
    float* offsets = new float[base_number];

// 使用多线程进行量化
#pragma omp parallel for 
    for (int i = 0; i < base_number; ++i) {
        quantize_vector(base + i * vecdim, base_quantized + i * vecdim,
                        scales[i], offsets[i], vecdim);
    }

    // 计算量化后的距离
    for (int i = 0; i < base_number; ++i) {
        float dis = inner_product_quantized(
            base_quantized + i * vecdim, query_quantized, scales[i], offsets[i],
            query_scale, query_offset, vecdim);

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

    // 释放内存
    delete[] query_quantized;
    delete[] base_quantized;
    delete[] scales;
    delete[] offsets;

    return q;
}