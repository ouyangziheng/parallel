# ANN 搜索算法实现与优化

这个项目实现了多种高效的ANN搜索算法，包括HNSW、IVF、IVF+HNSW、IVF+PQ等，主要用于大规模向量检索。

## 项目概述

主要实现的算法包括：

- **HNSW**：基于图的索引方法，利用分层结构快速搜索
- **IVF**：基于聚类的索引方法，将数据空间分割为多个簇
- **IVF+HNSW**：结合IVF和HNSW的优势，在每个簇内构建HNSW图
- **IVF+PQ**：在IVF基础上增加了乘积量化，进一步压缩内存


## 目录结构

- `main.cc`：主程序入口，包含数据加载和性能测试代码
- `*.h`：各种算法的头文件实现
  - `hnsw_search.h`：HNSW算法实现
  - `ivf_hnsw_search.h`：IVF+HNSW算法实现
  - `flat_scan_with_ivf.h`：IVF算法实现
  - `flat_scan_with_ivf_pq.h`：IVFPQ算法实现
  - `simd_utils.h`：SIMD加速工具函数
- `test_*.sh`：各种算法的测试脚本
  - `test_hnsw.sh`：HNSW算法测试
  - `test_hnsw_ivf.sh`：IVF+HNSW算法测试
  - `test_ivf.sh`：IVF算法测试
  - `test_ivf_pq.sh`：IVFPQ算法测试
  - `test.sh`：通用测试脚本

## 算法说明

### HNSW
HNSW 通过构建多层导航图来加速搜索。关键参数包括：
- `M`：每个节点的最大连接数
- `efConstruction`：构建索引时的搜索深度
- `efSearch`：查询时的搜索深度

### IVF
IVF 通过将数据点分配到不同的簇中，查询时只搜索最接近的几个簇，从而减少需要比较的向量数量。关键参数包括：
- `nlist`：聚类中心数量（簇的个数）
- `nprobe`：查询时搜索的簇数量

### IVF+HNSW
先通过IVF找簇，簇内使用HNSW搜索。

### IVF+PQ
IVF+PQ，获得速度和正确率的全面提高。

## 使用方法

### 运行测试

```bash
# 运行HNSW测试
bash test_hnsw.sh

# 运行IVF+HNSW测试
bash test_hnsw_ivf.sh

# 运行IVFPQ测试
bash test_ivf_pq.sh

# 运行所有测试
bash test_all.sh
```

### 自定义测试

修改各个测试脚本中的参数，进行自定义测试：

```bash
# 例如，修改test_hnsw_ivf.sh中的参数
NLIST_VALUES=(128 256 512)  # 簇的个数
NPROBE_VALUES=(16 32 64)    # 搜索时检查的簇数量
EF_CONSTRUCTION_VALUES=(100 150 200)  # 构建HNSW索引时使用的ef参数
M_VALUES=(8 16 24)          # HNSW图中每个节点的最大连接数
EF_SEARCH_VALUES=(100 200 300)  # 搜索时使用的ef参数
```

## 性能评测

项目包含完整的性能评测框架，可以自动测试不同参数组合下的召回率和查询延迟：

1. 测试脚本自动修改相关代码中的参数
2. 运行算法并收集召回率和延迟数据
3. 生成CSV格式的详细结果和TXT格式的性能摘要
4. 提供多种排序方式展示最优配置

