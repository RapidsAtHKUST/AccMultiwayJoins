#include "Joins/LFTJ.cuh"
#include "ooc.cuh"
#include "QueryProcessor.cuh"
using namespace std;

/*CPU time: prefetch + sort (build Trie) + LFTJ*/
template<typename DataType, typename CntType>
void test_general(string dir_name) { //todo: all build tables have the same #buckets
    CUDAMemStat memstat;
    CUDATimeStat timing;
    QueryProcessor<DataType, CntType> processor;
    Relation<DataType,CntType> *relations = nullptr;
    uint32_t num_tables;
    uint32_t gpu_time_idx;
    float build_trie_time = 0, lftj_time = 0;
    Timer t;

    /*load data from disk*/
    processor.load_multi_rels(dir_name, relations, num_tables, &memstat);

    /*prefetch relation data*/
    cudaDeviceSynchronize();
    t.reset();
    for(auto i = 0; i < num_tables; i++) {
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            checkCudaErrors(cudaMemPrefetchAsync(relations[i].data[a], sizeof(DataType)*relations[i].length, DEVICE_ID));
            log_info("Prefetch col %d of table %d", a, i);
        }
    }

    IndexedTrie<DataType,CntType> *Tries = nullptr;
    CUDA_MALLOC(&Tries, sizeof(IndexedTrie<DataType,CntType>)*num_tables, &memstat);

    gpu_time_idx = timing.get_idx();
    for(auto i = 0; i < num_tables; i++) { /*build the Tries according to the given attr_order*/
        Tries[i].init(relations[i].num_attrs, &memstat);
        vector<pair<AttrType, uint32_t>> reorder_vec;
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            reorder_vec.emplace_back(make_pair(relations[i].attr_list[a], a));
        }
        sort(reorder_vec.begin(), reorder_vec.end(), comp);

        /*rearrange the Trie data and attr_list*/
        DataType **rearranged_data = nullptr;
        CUDA_MALLOC(&rearranged_data, sizeof(DataType *)*relations[i].num_attrs, &memstat);
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            Tries[i].attr_list[a] = reorder_vec[a].first;
            rearranged_data[reorder_vec[a].second] = relations[i].data[a];
        }
        /*for Q3 and other chain query*/
        constructSortedTrie<DataType,CntType,2>(rearranged_data, relations[i].length, Tries[i].data, Tries[i].offsets, Tries[i].data_len, &memstat, &timing);

        /*for Q8_sp*/
//        if (4 == i) { //for lineitem
//            constructSortedTrie<DataType,CntType,3>(rearranged_data, relations[i].length, Tries[i].data, Tries[i].offsets, Tries[i].data_len, &memstat, &timing);
//        }
//        else if ((0==i) || (7==i)) { //for region and nation_1
//            constructSortedTrie<DataType,CntType,1>(rearranged_data, relations[i].length, Tries[i].data, Tries[i].offsets, Tries[i].data_len, &memstat, &timing);
//        }
//        else { //for other tables
//            constructSortedTrie<DataType,CntType,2>(rearranged_data, relations[i].length, Tries[i].data, Tries[i].offsets, Tries[i].data_len, &memstat, &timing);
//        }

        for(auto x = 0; x < relations[i].num_attrs; x++) { //release the original unsorted data to avoid memory overflow
            CUDA_FREE(relations[i].data[x], &memstat);
        }
    }

    build_trie_time += timing.diff_time(gpu_time_idx);
    gpu_time_idx = timing.get_idx();
    LFTJ<DataType,CntType>(Tries, num_tables, &memstat, &timing); //todo: LFTJ cannot run Q8 yet
    lftj_time += timing.diff_time(gpu_time_idx);

    log_info("---------------------------------");
    log_info("Build Trie kernel time: %.2f ms", build_trie_time);
    log_info("LFTJ kernel time: %.2f ms", lftj_time);
    log_info("Total kernel time: %.0f ms", build_trie_time+lftj_time);
    log_info("Total CPU execution time: %.2f s", t.elapsed());
    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/*
 * ./cuda-lftj DATA_DIR
 * */
int main(int argc, char *argv[]) {
    assert(2 == argc);
    cudaSetDevice(DEVICE_ID);
    test_general<KeyType,CarType>(string(argv[1]));

    return 0;
}