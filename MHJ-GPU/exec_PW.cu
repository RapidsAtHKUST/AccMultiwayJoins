//
// Created by Bryan on 17/10/2019.
//

#include "HashJoinEPFL/hashJoinEPFL.cuh"
#include "QueryProcessor.cuh"

template<typename DataType, typename CntType>
void test_general(string dir_name) {
    CUDAMemStat memstat;
    CUDATimeStat timing;
    QueryProcessor<DataType, CntType> processor;
    Relation<DataType,CntType> *relations = nullptr;
    uint32_t num_tables;
    uint32_t gpu_time_idx;
    Timer t;

    /*load data from disk*/
    processor.load_multi_rels(dir_name, relations, num_tables, &memstat);
    int num_attrs_in_res = 0; //number of output attrs
    bool attr_referred[MAX_NUM_RES_ATTRS] = {false};
    for(auto i = 0; i < num_tables; i++) { /*compute used_for_compare array*/
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            if (!attr_referred[relations[i].attr_list[a]]) {//this attr has not shown in previous relations
                attr_referred[relations[i].attr_list[a]] = true;
                num_attrs_in_res++;
            }
        }
    }
    cudaDeviceSynchronize();

    /*prefetch relation data*/
    for(auto i = 0; i < num_tables; i++) {
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            checkCudaErrors(cudaMemPrefetchAsync(relations[i].data[a], sizeof(DataType)*relations[i].length, DEVICE_ID));
            log_info("Prefetch col %d of table %d", a, i);
        }
    }
    cudaDeviceSynchronize();

    t.reset();
    DataType **res;
    gpu_time_idx = timing.get_idx();
    auto num_res = PW(relations, num_tables, res, num_attrs_in_res, &memstat, &timing);

    log_info("---------------------------------");
    log_info("PW kernel time: %.2f ms", timing.diff_time(gpu_time_idx));
    log_info("Num results: %llu", num_res);
    log_info("Total CPU execution time: %.2f s", t.elapsed());
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/* Usage:
 *     ./cuda-pw DATA_DIR
 * */
int main(int argc, char *argv[]) {
    assert(2 == argc);
    cudaSetDevice(DEVICE_ID);
    test_general<KeyType,CarType>(string(argv[1]));
    return 0;
}