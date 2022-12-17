#include "ooc.cuh"
#include "QueryProcessor.cuh"

template<typename DataType, typename CntType>
void test_general(string dir_name, bool ooc) {
    CUDAMemStat memstat;
    CUDATimeStat timing;
    QueryProcessor<DataType, CntType> processor;
    Relation<DataType,CntType> *relations = nullptr;
    HashTable<DataType,CntType> *hash_tables = nullptr;
    CntType *bucket_vec = nullptr;
    uint32_t num_tables;
    uint32_t gpu_time_idx;
    float build_kernel_time, probe_kernel_time;
    Timer t;

    /*load data from disk*/
    processor.load_multi_rels(dir_name, relations, num_tables, &memstat);
    uint32_t num_hash_tables = num_tables - 1;

    /*analyze the query datasets*/
    int num_attrs_in_res = 0; //number of output attrs
    int num_attr_idxes_in_iRes=0; //number of output attrs in iRes
    AttrType *attr_idxes_in_iRes = nullptr;
    CUDA_MALLOC(&attr_idxes_in_iRes, sizeof(AttrType)*MAX_NUM_RES_ATTRS, &memstat);

    /*compute used_for_compare*/
    bool **used_for_compare = nullptr;
    bool attr_referred[MAX_NUM_RES_ATTRS] = {false};
    CUDA_MALLOC(&used_for_compare, sizeof(bool*)*num_hash_tables, &memstat);
    for(auto i = 0; i < num_tables; i++) { /*compute used_for_compare array*/
        if (0 != i)
            CUDA_MALLOC(&used_for_compare[i-1], sizeof(bool)*relations[i].num_attrs, &memstat);
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            if (!attr_referred[relations[i].attr_list[a]]) {//this attr has not shown in previous relations
                attr_referred[relations[i].attr_list[a]] = true;
                num_attrs_in_res++;
                if (0 != i) used_for_compare[i-1][a] = false;
                if (i != num_tables -1) attr_idxes_in_iRes[num_attr_idxes_in_iRes++] = relations[i].attr_list[a];
            }
            else if (0 != i) used_for_compare[i-1][a] = true;
        }
    }

    /*prefetch relation data*/
    cudaDeviceSynchronize();
    t.reset();
    for(auto i = 0; i < num_tables; i++) {
        if ((0 == i) && (relations[i].length > MAX_PROBE_PREFETCH_THRES)) { //do not prefetch the probe table
            log_info("Skip prefetch of table 0");
            continue;
        }
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            checkCudaErrors(cudaMemPrefetchAsync(relations[i].data[a], sizeof(DataType)*relations[i].length, DEVICE_ID));
            log_info("Prefetch col %d of table %d", a, i);
        }
    }

    /*build phase*/
    gpu_time_idx = timing.get_idx();
    processor.build_hash_multiway(&relations[1], hash_tables, bucket_vec, num_tables - 1, MHJ_BUC_RATIO, &memstat, &timing);
    build_kernel_time = timing.diff_time(gpu_time_idx);

    size_t free_byte, total_byte;
    auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);//show memory usage of GPU
    if (cudaSuccess != cuda_status){
        log_error("cudaMemGetInfo fails, %s", cudaGetErrorString(cuda_status));
        exit(1);
    }
    auto used_byte = total_byte - free_byte;
    log_info("GPU memory usage: used = %.1f MB, free = %.1f MB, total = %.1f MB",
             1.0*used_byte/1024.0/1024.0, 1.0*free_byte/1024.0/1024.0, 1.0*total_byte/1024.0/1024.0);
    auto total_budget = (bsize_t)free_byte; //double buffer
    log_info("Initialize OOCWrapper with budget: %.1f MB", 1.0*total_budget/1024.0/1024.0);

    OOC<DataType,CntType, false, TYPE_MHJ> ooc_wrapper(1.0*total_budget/1024.0/1024.0);

    /*probe phase*/
    DataType **res_dummy = nullptr;
    gpu_time_idx = timing.get_idx();
    ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                        bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                        num_attr_idxes_in_iRes, ooc, res_dummy, &memstat, &timing);
    probe_kernel_time = timing.diff_time(gpu_time_idx);

    log_info("---------------------------------");
    log_info("Build kernel time: %.2f ms", build_kernel_time);
    log_info("Probe kernel time: %.2f ms", probe_kernel_time);
    log_info("Total kernel time: %.0f ms", build_kernel_time+probe_kernel_time);
    log_info("Total CPU execution time: %.2f ms", t.elapsed()*1000);
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/*
 * ./cuda-mwtbj DATA_DIR OOC
 * */
int main(int argc, char *argv[]) {
    cudaSetDevice(DEVICE_ID);

    if (argc == 4) {
        FILE *fp;
        fp = fopen(argv[3], "a+");
        if (fp == NULL) {
            cout<<"wrong file fp"<<endl;
            exit(1);
        }
        log_set_fp(fp);
    }

    test_general<KeyType,CarType>(string(argv[1]), (bool)stoi(argv[2]));
    return 0;
}