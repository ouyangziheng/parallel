#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
// #include "flat_scan.h"
// #include "flat_scan_with_simd.h"
// #include "flat_scan_with_pq.h"
// #include "flat_scan_with_pq_njjl.h"
#include "flat_scan_with_ivf.h"
#include "flat_scan_with_ivf_pq.h"
#include "hnsw_search.h" // 引入HNSW搜索头文件
#include "ivf_hnsw_search.h" // 引入IVF+HNSW搜索头文件
#include <fstream>

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.is_open();
}
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 注释掉未使用的PQ索引构建函数，避免编译错误
/*
void build_pq_index(float* base, size_t base_number, size_t vecdim)
{
    // 对于128维向量，设置为4个子空间，每个子空间32维
    size_t M = 4;  // 子空间数量
    size_t K = 256; // 每个子空间的聚类中心数量
    
    std::cout << "构建PQ索引，子空间数量：" << M << "，类中心数量：" << K << std::endl;
    
    // 创建PQ索引
    PQIndex* pq_index = new PQIndex(vecdim, M, K);
    
    // 构建索引
    pq_index->build(base, base_number);
    
    // 保存索引以便后续使用
    pq_index->save("files/pq.index");
    
    std::cout << "PQ索引构建完成并保存到 files/pq.index" << std::endl;
    
    // 释放内存
    delete pq_index;
}
*/

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);
    
    // 构建PQ索引
    // build_pq_index(base, base_number, vecdim);
    
    // // 构建IVF索引
    // const size_t nlist = 512; // 簇的个数
    // const size_t nprobe = 64; // 搜索时检查的簇数量
    
    // // IVF+PQ参数
    // const size_t M = 64;       // PQ子空间数量
    // const size_t K = 256;     // 每个子空间的聚类中心数量
    // const bool use_pq_then_ivf = false; // true表示先PQ后IVF，false表示先IVF后PQ
    
    // 确保目录存在
    std::cout << "确保files目录存在..." << std::endl;
    system("mkdir -p files");
    
    // IVF+HNSW的参数设置
    const size_t nlist = 512;   // 簇的个数
    const size_t nprobe = 64;   // 搜索时检查的簇数量
    const size_t ef_search = 300; // HNSW搜索参数
    
    // 检查IVF+HNSW索引是否存在，如果不存在则构建
    std::cout << "检查IVF+HNSW索引是否存在..." << std::endl;
    if (!fileExists("files/ivf_hnsw.index")) {
        std::cout << "构建IVF+HNSW索引..." << std::endl;
        build_ivf_hnsw_index(base, base_number, vecdim, nlist, nprobe, ef_search);
    }
    
    // 预先加载索引
    std::cout << "预加载IVF+HNSW索引..." << std::endl;
    ensure_ivf_hnsw_index_initialized("files/ivf_hnsw.index", ef_search);
    
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 使用IVF+HNSW搜索
        auto search_result = ivf_hnsw_search(base, test_query + i*vecdim, base_number, vecdim, k, nprobe, ef_search);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (search_result.size()) {   
            int x = search_result.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            search_result.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "平均召回率: " << avg_recall / test_number << "\n";
    std::cout << "平均查询延迟 (微秒): " << avg_latency / test_number << "\n";
    
    // 释放索引资源
    release_ivf_hnsw_index();
    
    return 0;
}
