//
// Created by Bryan on 10/12/2019.
//
/*write data to files*/
#include "generator_cuda.cuh"
#include <iostream>
#include "../common-utils/file_io.h"
#include "pretty_print.h"
#include "helper.h"

#define FIXED_SEED (1234)
using DType = int; //data is int
using CType = unsigned long long int; //count is unsigned long long int

/*
 * Produce skewed datasets for the uniform-skew case
 * i.e., R(A[u],B[u]) join S(B[s],C[u]) join T(A[u],C[u])
 * */
template<typename DataType, typename CntType, typename RangeType>
void writeSkewUSUniqueTriangle(
        CntType cnt, RangeType range, double z,
        Dis_type key0_type, Dis_type key1_type, string dir_addr) {
    int num_tables = 3; //for triangle
    int key1_attrs[3], key2_attrs[3];

    /*todo: only support a single skewed column*/
    assert((int)(key0_type) * (int)(key1_type) == 0);
    assert((int)(key0_type) + (int)(key1_type) == 1);
    char skewed_col = 0;
    if (key1_type == SKEWED) skewed_col = 1;

    string dir_name = dir_addr + "/" + "tri_" + to_string(cnt) + "_" + to_string(range) + "_";
    if (skewed_col == 0) dir_name += "su_";
    else                 dir_name += "us_";
    dir_name += double_to_string(z);
    auto is_create = mkdir(dir_name.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    if (is_create) {
        printf("create directory failed! error code : %d\n", is_create);
        exit(1);
    }

    /*init the tables*/
    for(auto i = 0; i < num_tables; i++) {
        vector<DataType*> table;
        for(auto t = 0; t < 2; t++) {
            DataType *temp = nullptr;
            CUDA_MALLOC(&temp, sizeof(DataType)*cnt, nullptr);
            table.emplace_back(temp);
        }
        zipfUniquePairGenerator(table[0], table[1], cnt, 0u, range, 0u, range, FIXED_SEED, skewed_col, z);
        key1_attrs[i] = i;
        key2_attrs[i] = (key1_attrs[i]+1)%num_tables;
        string file_name =  dir_name + "/" + to_string(cnt) + "_" + to_string(key1_attrs[i]) + "_" + to_string(key2_attrs[i]) + ".db";
        log_info("File name: %s.", file_name.c_str());
        write_rel_cols_mmap(file_name.c_str(), table, cnt);
        log_info("Table %d written to disk, count: %u, range: 0-%d, key idx: %d, value idx: %d.", i, cnt, range, key1_attrs[i], key2_attrs[i]);
    }
}

int main(int argc, char* argv[]) {
    assert(argc == 7);
    CType cnt = stoull(argv[1]);
    CType range = stoull(argv[2]);
    double z = stod(argv[3]);
    Dis_type key0_type = (Dis_type)(stoi(argv[4]));
    Dis_type key1_type = (Dis_type)(stoi(argv[5]));
    string base_addr = string(argv[6]);
    writeSkewUSUniqueTriangle<DType, CType, CType>(cnt, range, z, key0_type, key1_type, base_addr);
    return 0;
}
