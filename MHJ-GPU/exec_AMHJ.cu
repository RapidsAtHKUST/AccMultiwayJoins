#include "ooc.cuh"
#include "QueryProcessor.cuh"
#include "res_stat.h"

enum AMHJQueryType {
    NORMAL, TRIANGLE, FOUR_CLK
};

template<typename DataType, typename CntType, AMHJQueryType query_type>
void test_general(string dir_file_name, bool ooc, int ws, int sbp, int fib) {
    CUDAMemStat memstat;
    CUDATimeStat timing;
    QueryProcessor<DataType, CntType> processor;
    Relation<DataType,CntType> *relations = nullptr;
    HashTable<DataType,CntType> *hash_tables = nullptr;
    CntType *bucket_vec = nullptr;
    uint32_t num_tables;
    uint32_t gpu_time_idx;
    float build_kernel_time, probe_kernel_time;
    uint32_t num_hash_tables;
    Timer t;

    /*auxiliary data structures*/
    int num_attrs_in_res = 0; //number of output attrs
    int num_attr_idxes_in_iRes=0; //number of output attrs in iRes
    AttrType *attr_idxes_in_iRes = nullptr;
    bool **used_for_compare = nullptr;
    bool attr_referred[MAX_NUM_RES_ATTRS] = {false};
    CUDA_MALLOC(&attr_idxes_in_iRes, sizeof(AttrType)*MAX_NUM_RES_ATTRS, &memstat);

    /*load data from disk and set the auxiliary structures according to the query type*/
    if (query_type == TRIANGLE) { //triangle query
        processor.load_single_rel(dir_file_name, relations, &memstat);
        num_tables = 3;
        num_hash_tables = 2;
        int attrs_per_table = 2;
        /*hard code used_for_compare*/
        CUDA_MALLOC(&used_for_compare, sizeof(bool*)*num_hash_tables, &memstat);
        for(auto i = 0; i < num_hash_tables; i++) {
            CUDA_MALLOC(&used_for_compare[i], sizeof(bool)*attrs_per_table, &memstat);
        }
        /*A(0,1),B(1,2),C(0,2)*/
        used_for_compare[0][0] = 1; used_for_compare[0][1] = 0; //iRes(0,1), B(1,2)
        used_for_compare[1][0] = 1; used_for_compare[1][1] = 1; //iRes(0,1,2), C(0,2)
        num_attrs_in_res = 3; num_attr_idxes_in_iRes = 3;
        for(auto i = 0; i < num_attr_idxes_in_iRes; i++) {
            attr_idxes_in_iRes[i] = i;
        }
    }
    else if (query_type == FOUR_CLK) { //four clique
        processor.load_single_rel(dir_file_name, relations, &memstat);
        num_tables = 6;
        num_hash_tables = 5;
        int attrs_per_table = 2;
        CUDA_MALLOC(&used_for_compare, sizeof(bool*)*num_hash_tables, &memstat);
        for(auto i = 0; i < num_hash_tables; i++) {
            CUDA_MALLOC(&used_for_compare[i], sizeof(bool)*attrs_per_table, &memstat);
        }
        /*A(0,1),B(1,2),C(2,3),D(0,3),E(0,2),F(1,3)*/
        used_for_compare[0][0] = 1; used_for_compare[0][1] = 0; //iRes(0,1), B(1,2)
        used_for_compare[1][0] = 1; used_for_compare[1][1] = 0; //iRes(0,1,2), C(2,3)
        used_for_compare[2][0] = 1; used_for_compare[2][1] = 1; //iRes(0,1,2,3), D(0,3)
        used_for_compare[3][0] = 1; used_for_compare[3][1] = 1; //iRes(0,1,2,3), E(0,2)
        used_for_compare[4][0] = 1; used_for_compare[4][1] = 1; //iRes(0,1,2,3), F(1,3)
        num_attrs_in_res = 4; num_attr_idxes_in_iRes = 4;
        for(auto i = 0; i < num_attr_idxes_in_iRes; i++) {
            attr_idxes_in_iRes[i] = i;
        }
    }
    else { //normal multiway joins
        processor.load_multi_rels(dir_file_name, relations, num_tables, &memstat);
        num_hash_tables = num_tables - 1;

        /*compute used_for_compare*/
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
    }

//    for(auto i = 0; i < 10; i++) {
//        for(auto a = 0; a < relations[0].num_attrs; a++) {
//            cout<<relations[0].data[a][i]<<' ';
//        }
//        cout<<endl;
//    }

    /*prefetch relation data and build hash tables*/
    cudaDeviceSynchronize();
    t.reset();
    if (query_type == NORMAL) {
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
        processor.build_hash_multiway(&relations[1], hash_tables, bucket_vec, num_tables - 1, AMHJ_BUC_RATIO, &memstat, &timing);
        build_kernel_time = timing.diff_time(gpu_time_idx);
    }
    else { //only prefetch a single relation
        for(auto a = 0; a < relations[0].num_attrs; a++) {
            checkCudaErrors(cudaMemPrefetchAsync(relations[0].data[a], sizeof(DataType)*relations[0].length, DEVICE_ID));
            log_info("Prefetch col %d of the table", a);
        }
        cudaDeviceSynchronize();

        vector<pair<AttrType,AttrType>> input_attrs;
        if (query_type == TRIANGLE) { /*A(0,1),B(1,2),C(0,2)*/
            input_attrs.emplace_back(make_pair(0, 1));
            input_attrs.emplace_back(make_pair(1, 2));
            input_attrs.emplace_back(make_pair(0, 2));
        }
        else { /*A(0,1),B(1,2),C(2,3),D(0,3),E(0,2),F(1,3)*/
            input_attrs.emplace_back(make_pair(0, 1));
            input_attrs.emplace_back(make_pair(1, 2));
            input_attrs.emplace_back(make_pair(2, 3));
            input_attrs.emplace_back(make_pair(0, 3));
            input_attrs.emplace_back(make_pair(0, 2));
            input_attrs.emplace_back(make_pair(1, 3));
        }

        /*build phase*/
        gpu_time_idx = timing.get_idx();
        processor.build_hash_single(relations[0], hash_tables, bucket_vec, num_tables - 1, AMHJ_BUC_RATIO,
                                    input_attrs, &memstat, &timing);
        build_kernel_time = timing.diff_time(gpu_time_idx);
    }

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

    DataType **res_data = nullptr;
    CntType res_num = 0;
    gpu_time_idx = timing.get_idx();

    if (sbp == 0) {
        if(ws == 0) {
            OOC<DataType,CntType, true, TYPE_AMHJ> ooc_wrapper(1.0*total_budget/1024.0/1024.0); //AMHJ
            res_num = ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                                bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                                num_attr_idxes_in_iRes, ooc, res_data, &memstat, &timing);
        }
        else {
            if (fib) {
                OOC<DataType,CntType, true, TYPE_AMHJ_FIB> ooc_wrapper(1.0*total_budget/1024.0/1024.0); //WS+FIB
                res_num = ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                                    bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                                    num_attr_idxes_in_iRes, ooc, res_data, &memstat, &timing);
            }
            else {
                OOC<DataType,CntType, true, TYPE_AMHJ_WS> ooc_wrapper(1.0*total_budget/1024.0/1024.0); //WS
                res_num = ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                                    bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                                    num_attr_idxes_in_iRes, ooc, res_data, &memstat, &timing);
            }
        }
    }
    else {
        if (ws == 0) {
            OOC<DataType,CntType, true, TYPE_AMHJ_DRO> ooc_wrapper(1.0*total_budget/1024.0/1024.0);
            res_num = ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                                bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                                num_attr_idxes_in_iRes, ooc, res_data, &memstat, &timing);
        }
        else {
            OOC<DataType,CntType, true, TYPE_AMHJ_WS_DRO> ooc_wrapper(1.0*total_budget/1024.0/1024.0);
            res_num = ooc_wrapper.execute(relations[0], hash_tables, num_hash_tables, used_for_compare,
                                bucket_vec, num_attrs_in_res, attr_idxes_in_iRes,
                                num_attr_idxes_in_iRes, ooc, res_data, &memstat, &timing);
        }
    }
    probe_kernel_time = timing.diff_time(gpu_time_idx);

    log_info("---------------------------------");
    log_info("Result checking...");
    res_stat(res_data, res_num, num_attrs_in_res, false);

    log_info("---------------------------------");
    log_info("Time and Mem stat...");
    log_info("Build kernel time: %.2f ms", build_kernel_time);
    log_info("Probe kernel time: %.2f ms", probe_kernel_time);
    log_info("Total kernel time: %.0f ms", build_kernel_time+probe_kernel_time);
    log_info("Total CPU time: %.2f ms", t.elapsed()*1000);
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/*
 * ./exec-AMHJ [NORMAL|TRI|FOUR] DATA_DIR OOC WS DRO FIB
 * */
int main(int argc, char *argv[]) {

    cudaSetDevice(DEVICE_ID);

//    if (argc == 7) {
//        FILE *fp;
//        fp = fopen(argv[6], "a+");
//        if (fp == NULL) {
//            cout<<"wrong file fp"<<endl;
//            exit(1);
//        }
//        log_set_fp(fp);
//    }

    string query_type = string(argv[1]);
    if (query_type == "NORMAL") {
        test_general<KeyType,CarType,NORMAL>(string(argv[2]), (bool)stoi(argv[3]), stoi(argv[4]), stoi(argv[5]), stoi(argv[6]));
    }
    else if (query_type == "TRI") {
        test_general<KeyType,CarType,TRIANGLE>(string(argv[2]), (bool)stoi(argv[3]), stoi(argv[4]), stoi(argv[5]), stoi(argv[6]));
    }
    else if (query_type == "FOUR") {
        test_general<KeyType,CarType,FOUR_CLK>(string(argv[2]), (bool)stoi(argv[3]), stoi(argv[4]), stoi(argv[5]), stoi(argv[6]));
    }
    else {
        log_error("Wrong query type");
    }
    return 0;
}