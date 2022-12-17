//
// Created by Bryan on 26/11/2019.
//

#include "types.h"
#include "LFTJ_BFS.cuh"
#include "generator_cuda.cuh"
#include "../common-utils/file_io.h"
#include "multi_partitioning.cuh"
#include "ooc_wrapper.cuh"
#include <tuple>
using namespace std;

using DataType = int; //data type
using CntType = unsigned long long int;  //count type

#define FIXED_SEED      (1234ull)

string trie_prefix = ".tr";
string rel_prefix = ".db";

enum DatasetType {
    MULTI_WAY, SELF
};

/*
 * LFTJ_GPU, only count the number of output reselts
 * CPU time: prefetch + sort (build Trie) + LFTJ
 * */
template<typename DataType, typename CntType>
CntType LFTJ_exec(
        IndexedTrie<DataType, CntType> *Tries, uint32_t num_tables, uint32_t num_attrs, uint32_t *attr_order,
        LFTJ_type algo_type, bool ooc, bool work_sharing, CUDAMemStat *memstat, CUDATimeStat *timing) {
    Timer t;
    auto gpu_time_idx = timing->get_idx();
    OOCWrapper<DataType,CntType> wrapper;
    CntType res;

    t.reset();
    gpu_time_idx = timing->get_idx();
    if (algo_type == TYPE_BFS)
        res = wrapper.execute_with_BFS(Tries, num_tables, num_attrs, attr_order, ooc, work_sharing, memstat, timing);
    else if (algo_type == TYPE_DFS)
        res = wrapper.execute_with_DFS(Tries, num_tables, num_attrs, attr_order, ooc, work_sharing, memstat, timing);
    else {
        log_error("Unsupported algorithm type");
        return 0;
    }

    log_info("LFTJ Total kernel time: %.2f ms", timing->diff_time(gpu_time_idx));
    log_info("LFTJ Total CPU time: %.2f ms", t.elapsed()*1000);

    return res;
}

/*
 * To specify queries in self-join case:
 * 1. specficy the num_tables, num_attrs, num_cols;
 * 2. specify the attr_list
 * */
template<typename DataType, typename CntType>
void query_processing(char *dir_name, LFTJ_type algo_type, bool ooc, bool work_sharing) {
    Timer t;
    CUDAMemStat memstat;
    CUDATimeStat timing;
    vector<string> file_names;  //record the dataset names
    vector<bool> is_trie;       //whether a dataset is a Trie
    DatasetType d_type;
    uint32_t num_tables, num_attrs, num_columns, temp_idx;
    IndexedTrie<DataType,CntType> *Tries = nullptr;
    auto gpu_time_idx = timing.get_idx();
    uint32_t *attr_order = nullptr;
    DataType **data_group = nullptr;

    auto file_list = list_files(dir_name);
    is_trie.resize(file_list.size(), false);
    for(auto i = 0; i < file_list.size(); i++) {
        if ((file_list[i].find(trie_prefix)==string::npos) &&
            (file_list[i].find(rel_prefix)==string::npos))
            continue; //ignore other files
        string cur_name = file_list[i].substr(0, file_list[i].length()-3); //remove suffix
        auto it = std::find(file_names.begin(), file_names.end(), cur_name);
        if (it == file_names.end()) { //not found
            file_names.emplace_back(cur_name);
            it = std::find(file_names.begin(), file_names.end(), cur_name);
        }
        if (file_list[i].find(trie_prefix)!=string::npos) { //is a Trie index
            auto offset = distance(file_names.begin(), it);
            is_trie[offset] = true;
        }
    }
    if (0 == file_names.size() || (2 == file_names.size())) {
        log_error("Wrong number of datasets: %d", file_names.size());
        exit(1);
    }
    else if(1 == file_names.size()) { //single input table, self-join
        d_type = SELF;

        /*Triangle query Q(A,B,C) :- R(A,B),S(B,C),T(A,C)*/
        num_tables = num_attrs = 3;
        num_columns = 6;
    }
    else { //multiple input tables
        d_type = MULTI_WAY;
        num_attrs = num_columns = 0;
        num_tables = (uint32_t)file_names.size();
    }

    CUDA_MALLOC(&Tries, sizeof(IndexedTrie<DataType,CntType>)*num_tables, &memstat);
    vector<bool> attrs(4*num_tables, false);
    for(auto i = 0; i < num_tables; i++) {
        vector<string> key_attr_strs;
        vector<AttrType> key_attrs;
        int cur_num_cols;

        if ((0 == i) || (MULTI_WAY == d_type)) {
            split_string(file_names[i], key_attr_strs, "_");
            CntType cnt = stoull(key_attr_strs[0].c_str());

            /*extract the column attrs*/
            cur_num_cols = (int)key_attr_strs.size()-1;
            if (MULTI_WAY == d_type) {
                num_columns += cur_num_cols;
            }
            for(auto j = 0; j < cur_num_cols; j++) {
                auto cur_attr = (AttrType)stoi(key_attr_strs[1+j].c_str());
                key_attrs.emplace_back(cur_attr);
                attrs[cur_attr] = true;
            }
            auto file_name_conc = string(dir_name) + "/" + file_names[i] + ((is_trie[i])?trie_prefix:rel_prefix);
            if (is_trie[i]) { //load the Trie index
                log_info("Read from Trie file:");
                Tries[i].init_with_deserialization(file_name_conc.c_str(), &memstat);
                for(auto j = 0; j < cur_num_cols; j++) {
                    Tries[i].attr_list[j] = key_attrs[j];
                }
            }
            else { //load the rel
                log_info("Read from raw file: %s", file_name_conc.c_str());
                log_debug("cur_num_cols=%d", cur_num_cols);
                Tries[i].init(cur_num_cols, &memstat);
                auto data_out = read_rel_cols_mmap<DataType, CntType>(file_name_conc.c_str(), cur_num_cols, cnt);
                CUDA_MALLOC(&data_group, sizeof(DataType*)*cur_num_cols, &memstat);

                map<AttrType,DataType*> attr_data; //data in map is auto sorted w.r.t. the keys
                for(auto j = 0; j < cur_num_cols; j++) {
                    attr_data.insert(std::make_pair(key_attrs[j], data_out[j]));
                }

                t.reset();
                temp_idx = 0;
                for(auto &it : attr_data) {
                    if (MULTI_WAY == d_type) Tries[i].attr_list[temp_idx] = it.first; //init attr_list for Tries[i]
                    data_group[temp_idx++] = it.second;
                    checkCudaErrors(cudaMemPrefetchAsync(it.second, sizeof(DataType)*cnt, DEVICE_ID));
                }
                switch(cur_num_cols) {
                    case 2: {
                        constructSortedTrie<DataType, CntType, 2>(
                                data_group, cnt,
                                Tries[i].data, Tries[i].offsets, Tries[i].data_len,
                                &memstat, &timing); //compute data, offsets and data_len for Tries[i]
                        break;
                    }
                    case 3: {
                        constructSortedTrie<DataType, CntType, 3>(
                                data_group, cnt,
                                Tries[i].data, Tries[i].offsets, Tries[i].data_len,
                                &memstat, &timing); //compute data, offsets and data_len for Tries[i]
                        break;
                    }
                    case 4: {
                        constructSortedTrie<DataType, CntType, 4>(
                                data_group, cnt,
                                Tries[i].data, Tries[i].offsets, Tries[i].data_len,
                                &memstat, &timing); //compute data, offsets and data_len for Tries[i]
                        break;
                    }
                    case 5: {
                        constructSortedTrie<DataType, CntType, 5>(
                                data_group, cnt,
                                Tries[i].data, Tries[i].offsets, Tries[i].data_len,
                                &memstat, &timing); //compute data, offsets and data_len for Tries[i]
                        break;
                    }
                    default: {
                        log_error("Unsupported number of columns");
                        exit(1);
                    }
                }
                auto sort_time = t.elapsed();
                log_debug("Trie Construction CPU time: %.2f ms", sort_time * 1000);
                CUDA_FREE(data_group, &memstat);
            }
            auto trie_size = Tries[i].get_disk_size();
            log_debug("Trie size: %.2f GB", trie_size*1.0/1024/1024/1024);
        }
        else { //other Tries in SELF case, copy some structures to new Tries
            log_info("Copy Trie[0] data to Trie[%d]", i);
            Tries[i].init(cur_num_cols, &memstat);
            for(auto j = 0; j < cur_num_cols; j++) {
                Tries[i].data[j] = Tries[0].data[j];
                Tries[i].data_len[j] = Tries[0].data_len[j];
                Tries[i].attr_list[j] = Tries[0].attr_list[j];
                Tries[i].trie_offsets[j] = Tries[0].trie_offsets[j];
            }
            for(auto j = 0; j < cur_num_cols-1; j++) {
                Tries[i].offsets[j] = Tries[0].offsets[j];
            }
        }
//        cout<<"Trie "<<i<<endl;
//        Tries[i].print();
    }
    for(auto i = 0; i < 4*num_tables; i++) {
        if ((attrs[i]) && (d_type == MULTI_WAY)) num_attrs++;
    }

    if (SELF == d_type) { /*attr_list setting for triangle query*/
        Tries[0].attr_list[0] = 0; Tries[0].attr_list[1] = 1;
        Tries[1].attr_list[0] = 1; Tries[1].attr_list[1] = 2;
        Tries[2].attr_list[0] = 0; Tries[2].attr_list[1] = 2;
    }
    for(auto i = 0; i < num_tables; i++) { //check whether the Trie has been set completely
        if (!Tries[i].check()) {
            log_error("Trie %d is not properly set", i);
            exit(1);
        }
    }

    CUDA_MALLOC(&attr_order, sizeof(uint32_t)*num_attrs, &memstat);
    for(auto i = 0; i < num_attrs; i++) attr_order[i] = i; //todo: attr order is always 0, 1, 2, 3...

    log_info("Read file & Trie construction kernel time: %.2f ms", timing.diff_time(gpu_time_idx));
    log_info("Read file & Trie construction CPU time: %.2f ms", t.elapsed()*1000);

    /*Query information*/
    log_info("Query: #datasets=%d, #join tables=%d, #columns=%d, #join attrs=%d",
             file_list.size(), num_tables, num_columns, num_attrs);

    LFTJ_exec(Tries, num_tables, num_attrs, attr_order, algo_type, ooc, work_sharing, &memstat, &timing);

#ifdef FREE_DATA
    Tries[0].clear();
    for(auto i = 1; i < num_tables; i++) {
        if (MULTI_WAY == d_type) Tries[i].clear();
        else                     Tries[i].clear(false);
    }

    CUDA_FREE(Tries, &memstat);
    CUDA_FREE(attr_order, &memstat);
#endif
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/* Usage:
 *      ./cuda-mwgeneral [data_address] [algo_type] [is_occ] [is_WS]
 * algo_type: 0-BFS, 1-DFS
 * is_occ: 0-no ooc, 1-use ooc
 * */
int main(int argc, char *argv[]) {
    Timer t;
    assert(argc == 5);
    cudaSetDevice(DEVICE_ID);

    query_processing<DataType, CntType>(argv[1], (LFTJ_type)(atoi(argv[2])), (bool)stoi(argv[3]), (bool)stoi(argv[4]));
    log_info("Total wall time: %.2f s", t.elapsed());
    return 0;
}