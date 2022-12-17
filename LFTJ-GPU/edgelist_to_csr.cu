//
// Created by Bryan on 8/8/2020.
//
#include "types.h"
#include "IndexedTrie.cuh"
#include "../common-utils/file_io.h"
#include "multi_partitioning.cuh"
using namespace std;

#define SEP  (" |\t")

using DataType = int; //data type
using CntType = unsigned long long int;  //count type

/*
 * transferring a .db file to a CSR file
 * */
template<typename DataType, typename CntType>
void edgelist_to_csr(string file_name) {
    Timer t;
    CUDAMemStat memstat;
    CUDATimeStat timing;
    string output_file_name = "output.tr";
    IndexedTrie<DataType,CntType> *Trie = nullptr;
    DataType **data_group;
    vector<string> key_attr_strs;
    FILE *fp;
    char *line = nullptr;
    size_t len = 0;

    split_string(get_file_name_from_addr(file_name), key_attr_strs, "_");
    CntType cnt = stoull(key_attr_strs[0].c_str()); //the cardinality, avoid wc -l
    log_info("Cardinality: %llu", cnt);

    /*allocate the memory objects for data*/
    CUDA_MALLOC(&data_group, sizeof(DataType*)*2, &memstat);
    for(auto i = 0; i < 2; i++) {
        CUDA_MALLOC(&data_group[i], sizeof(DataType*)*cnt, &memstat);
    }
    CUDA_MALLOC(&Trie, sizeof(IndexedTrie<DataType,CntType>), &memstat);
    Trie->init(2, &memstat); //2 columns

    /*read from file*/
    t.reset();
    log_info("Read from raw file: %s", file_name.c_str());
    if ((fp = fopen(file_name.c_str(), "r")) == nullptr) {
        log_error("File %s not exist.", file_name.c_str());
        exit(1);
    }
    CntType row_idx = 0;
    while (-1 != getline(&line, &len, fp)) {
        char *stringRet;
        int attr_idx = 0;
        stringRet = strtok(line, SEP); //partition
        while (stringRet != nullptr) {
            data_group[attr_idx][row_idx] = stoul(stringRet);
            stringRet = strtok(nullptr, SEP);
            attr_idx++;
        }
        row_idx++;
    }
    if(line) free(line);
    fclose(fp);
    log_info("Read file finished, time: %.2f s", t.elapsed());

    t.reset();
    log_info("Begin Trie construction");
    checkCudaErrors(cudaMemPrefetchAsync(data_group[0], sizeof(DataType)*cnt, DEVICE_ID));
    checkCudaErrors(cudaMemPrefetchAsync(data_group[1], sizeof(DataType)*cnt, DEVICE_ID));
    constructSortedTrie<DataType, CntType, 2>(data_group, cnt, Trie->data, Trie->offsets, Trie->data_len, &memstat, &timing);
    log_info("Trie construction finished, time: %.2f s", t.elapsed());
    auto trie_size = Trie->get_disk_size();
    log_debug("Trie size: %.2f GB", trie_size*1.0/1024/1024/1024);

    log_info("Begin writing Trie to disk.");
    t.reset();
    Trie->serialization(output_file_name.c_str());
    log_info("Trie write finished, time: %.2f s", t.elapsed());

    Trie->clear();
    CUDA_FREE(Trie, &memstat);
    CUDA_FREE(data_group, &memstat);

    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
}

/* Usage:
 *    ./edgelist_to_csr [data_address]
 * */
int main(int argc, char *argv[]) {
    Timer t;
    assert(argc == 2);
    cudaSetDevice(DEVICE_ID);
    edgelist_to_csr<DataType, CntType>(string(argv[1]));
    log_info("Total wall time: %.2f s", t.elapsed());
    return 0;
}