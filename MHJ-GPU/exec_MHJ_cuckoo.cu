#include "ooc.cuh"
#include "Indexing/cuckoo_radix.cuh"
#include "Joins/MHJ_cuckoo.cuh"
#include "QueryProcessor.cuh"

template<typename DataType, typename CntType>
void test_general(string dir_name) {
    CUDAMemStat memstat;
    CUDATimeStat timing;
    QueryProcessor<DataType, CntType> processor;
    Relation<DataType,CntType> *relations = nullptr;
    CuckooHashTableRadix<DataType,CntType,CntType> *hash_tables = nullptr;
    uint32_t num_tables;
    uint32_t gpu_time_idx;
    float build_kernel_time = 0, probe_kernel_time = 0;
    Timer t, t_total;

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

    DataType **res_gpu = nullptr;

    /*prefetch relation data*/
    cudaDeviceSynchronize();
//    t_total.reset();
    for(auto i = 0; i < num_tables; i++) {
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            checkCudaErrors(cudaMemPrefetchAsync(relations[i].data[a], sizeof(DataType)*relations[i].length, DEVICE_ID));
            log_info("Prefetch col %d of table %d", a, i);
        }
    }
    cudaDeviceSynchronize();

    /*build phase*/
    t_total.reset();
    CUDA_MALLOC(&hash_tables, sizeof(CuckooHashTableRadix<DataType,CntType,CntType>)*num_hash_tables, &memstat);
    for(auto i = 0; i < num_hash_tables; i++) {
        int evict_bound = (int)(7 * log(relations[i+1].length));
        t.reset();
        hash_tables[i].init(relations[i+1].length, CUCKOO_HT_RATIO,
                            evict_bound, &relations[i+1], relations[i+1].attr_list[0],
                            &memstat, &timing);
        int rehash_cnt;
        int hash_cnt = 0;
        do {
            if (hash_cnt > 0) log_debug("rehashing");
            gpu_time_idx = timing.get_idx();
            rehash_cnt = hash_tables[i].insert_ki(relations[i + 1].data[0], relations[i + 1].length);
            hash_cnt++;
        } while (rehash_cnt != 0);

        log_info("Build cuckoo hash table %d CPU time: %.2f ms", i, t.elapsed()*1000);
        build_kernel_time += timing.diff_time(gpu_time_idx);
    }

    /*probe phase*/
    gpu_time_idx = timing.get_idx();
    MHJ_cuckoo_hashing<DataType, CntType, true>(
            relations[0], hash_tables, num_hash_tables, used_for_compare,
            attr_idxes_in_iRes, num_attr_idxes_in_iRes, res_gpu, num_attrs_in_res,
            &memstat, &timing);
    probe_kernel_time = timing.diff_time(gpu_time_idx);

    log_info("---------------------------------");
    log_info("Build kernel time: %.2f ms", build_kernel_time);
    log_info("Probe kernel time: %.2f ms", probe_kernel_time);
    log_info("Total kernel time: %.0f ms", build_kernel_time+probe_kernel_time);
    log_info("Total CPU execution time: %.2f ms", t_total.elapsed()*1000);
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/*
 * ./cuda-tbj-cuckoo DATA_DIR
 * */
int main(int argc, char *argv[]) {
    cudaSetDevice(DEVICE_ID);
    srand(time(nullptr)); //reset the seed

    if (argc == 3) {
        FILE *fp;
        fp = fopen(argv[2], "a+");
        if (fp == NULL) {
            cout<<"wrong file fp"<<endl;
            exit(1);
        }
        log_set_fp(fp);
    }

    test_general<KeyType,CarType>(string(argv[1]));
    return 0;
}