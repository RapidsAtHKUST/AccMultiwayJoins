//
// Created by Bryan on 10/12/2019.
//
/*write data to files*/
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include "generator_cuda.cuh"
#include "../common-utils/file_io.h"
#include "pretty_print.h"
#include "IndexedTrie.cuh"
#include "multi_partitioning.cuh"

#define FIXED_SEED (1234)
using DType = int; //data is int
using CType = unsigned long long int; //count is unsigned long long int

uint32_t T_A[3][6] = {{1,1,1,2,2,9},{2,2,3,3,5,12},{1,1,2,2,2,22}};
uint32_t T_B[3][6] = {{2,3,5,2,5,10},{3,5,2,5,5,17},{2,3,2,3,5,98}};

enum WRITE_TYPE {
    DATA_ONLY, TRI_ONLY, DATA_AND_TRI
};

template<typename DataType, typename CntType>
void dataInitSyntheticTriangleSpecific(
        DataType* keys1, DataType* keys2, CntType cnts) {
    static int table = 0;
    /*init the tables*/
    for(auto j = 0; j < cnts; j++) {
        keys1[j] = T_A[table][j];
        keys2[j] = T_B[table][j];
    }
    table++;
}

/*
 * Produce uniformly distributed datasets with unique tuples
 * */
template<typename DataType, typename CntType, typename RangeType>
void writeUniformTriangle(CntType cnt, RangeType range, string dir_addr, WRITE_TYPE type, uint32_t *attr_order, CUDAMemStat *memstat) {
    int num_tables = 3; //for triangle
    int num_attrs = 3;
    int num_attrs_per_table = 2;
    uint32_t key1_attrs[3], key2_attrs[3];

    string dir_name = dir_addr + "/" + "tri_" + to_string(cnt) + "_" + to_string(range) + "_uu";
    auto is_create = mkdir(dir_name.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if (is_create) {
        printf("create directory failed! error code : %d\n", is_create);
        exit(1);
    }

    /*init the tables*/
    for(auto i = 0; i < num_tables; i++) {
        log_info("------------------");
        vector<DataType*> table;
        for(auto t = 0; t < num_attrs_per_table; t++) {
            DataType *temp = nullptr;
            CUDA_MALLOC(&temp, sizeof(DataType)*cnt, memstat);
            table.emplace_back(temp);
        }
        uniformUniquePairGenerator(table[0], table[1], cnt, 0, range, 0, range);
//        dataInitSyntheticTriangleSpecific(table[0], table[1], 6);

        key1_attrs[i] = i;
        key2_attrs[i] = (key1_attrs[i]+1)%num_tables;
#ifndef UNSORTED_DATA
        if (key1_attrs[i] > key2_attrs[i]) { //todo: force specific ordering since we can only make it sorted w.r.t key1,key2
            auto temp = key1_attrs[i];
            key1_attrs[i] = key2_attrs[i];
            key2_attrs[i] = temp;
        }
#endif
        if (type != TRI_ONLY) {
            string file_name =  dir_name + "/" + to_string(cnt) + "_" + to_string(key1_attrs[i]) + "_" + to_string(key2_attrs[i]) + ".db";
            log_info("File name: %s", file_name.c_str());
            write_rel_cols_mmap(file_name.c_str(), table, cnt);
            log_info("Raw table %d(%d,%d) written to disk, count: %llu, range: 0-%d", i, key1_attrs[i], key2_attrs[i], cnt, range);
        }
        if (type != DATA_ONLY) { //construct the Trie and write to disk, need to provide attr_order
            string file_name =  dir_name + "/" + to_string(cnt) + "_" + to_string(key1_attrs[i]) + "_" + to_string(key2_attrs[i]) + ".tr";
            log_info("File name: %s", file_name.c_str());

            IndexedTrie<DataType,CntType> *cur_trie= nullptr;
            CUDA_MALLOC(&cur_trie, sizeof(IndexedTrie<DataType,CntType>), memstat);
            cur_trie->init(num_attrs_per_table, memstat);
            DataType **data_group;
            CUDA_MALLOC(&data_group, sizeof(DataType*)*num_attrs_per_table, memstat);
            if (precede(attr_order, num_attrs, key1_attrs[i], key2_attrs[i])) {
                data_group[0] = table[0];
                data_group[1] = table[1];
                cur_trie->attr_list[0] = key1_attrs[i];
                cur_trie->attr_list[1] = key2_attrs[i];
            }
            else {
                log_info("Switch the table attrs according to the attr order");
                data_group[1] = table[0];
                data_group[0] = table[1];
                cur_trie->attr_list[0] = key2_attrs[i];
                cur_trie->attr_list[1] = key1_attrs[i];
            }
            constructSortedTrie<DataType, CntType, 2>(
                data_group, cnt, cur_trie->data, cur_trie->offsets,
                cur_trie->data_len, memstat, nullptr);

//            /*check the trie*/
//            auto data_len_0 = cur_trie->data_len[0];
//            printf("data_len[0]=%llu, data_len[1]=%llu\n", data_len_0, cur_trie->data_len[1]);
//
//            printf("last 11 data[0]: ");
//            for(CntType j = data_len_0 - 10; j < data_len_0; j++) {
//                printf("%d ", cur_trie->data[0][j]);
//            }
//            printf("\n");
//            printf("last 11 offsets[0]: ");
//            for(CntType j = data_len_0 - 10; j <= data_len_0; j++) {
//                printf("%llu ", cur_trie->offsets[0][j]);
//            }
//            printf("\n\n");
//
//            exit(1);

            cur_trie->serialization(file_name.c_str());
            log_info("Trie of raw table %d(%d,%d) written to disk",i, key1_attrs[i], key2_attrs[i]);

            cur_trie->clear();
            CUDA_FREE(cur_trie, memstat);
            CUDA_FREE(data_group, memstat);
        }
        /*free the memory used*/
        for(auto t = 0; t < table.size(); t++)
            CUDA_FREE(table[t], memstat);
    }
}

int main(int argc, char* argv[]) {
    assert(argc == 4);
    CType cnt = stoull(argv[1]);
    DType range = stoi(argv[2]);
    string base_addr = string(argv[3]);

    uint32_t order[3] = {0,1,2};
    CUDAMemStat memstat;
    writeUniformTriangle<DType, CType, DType>(cnt, range, base_addr, DATA_ONLY, order, &memstat);

    log_info("Maximal device mem demanded: %ld bytes.", memstat.get_max_use());
    log_info("Unfreed mem size: %ld bytes.", memstat.get_cur_use());
    return 0;
}
