//
// Created by Bryan on 24/3/2020.
//

#include <iostream>
#include <sys/stat.h>
#include "../common-utils/file_io.h"
#include "pretty_print.h"
#include "data_generator.cuh"
#include "types.h"

/*
 * Produce zipf distributed dataset
 * */
template<typename DataType, typename CntType, typename RangeType>
void skewed_writer(
        CntType cnt, vector<RangeType> ranges, vector<double> alphas,
        vector<AttrType> attrs, string base_addr, CntType **mappings) {
    int num_attrs = (int)ranges.size();
    string file_name = base_addr + "/" + to_string(cnt);

    vector<DataType*> table;
    for(auto i = 0; i < num_attrs; i++) {
        file_name += ("_" + to_string(attrs[i]));
        DataType *temp = nullptr;
        checkCudaErrors(cudaMallocManaged((void**)&temp, sizeof(DataType)*cnt));
        if(0 == alphas[i])      { //uniform with unique keys
            for(auto j = 0; j < cnt; j++) temp[j] = j % ranges[i];
            for(auto j = cnt-1; j > 0; j--) { //shuffle
                auto x = rand() % j;
                std::swap(temp[j], temp[x]);
            }
//            uniform_generator(temp, cnt, 0, ranges[i]);
        }
        else if (alphas[i]>0)   zipf_generator_GPU(temp, cnt, 0, ranges[i], alphas[i], mappings[i]);
        table.emplace_back(temp);
    }
    file_name += ".db";
    log_trace("Data generated");
    write_rel_cols_mmap(file_name.c_str(), table, cnt);
    log_info("Generated file: %s", file_name.c_str());

    for(auto t = 0; t < table.size(); t++) checkCudaErrors(cudaFree(table[t]));
}

/*
 * Write a single uniformly-distributed table with specifed columns and data range
 * Usage
 *   ./skewed_writter   TABLE_CARD NUM_COLS
 *                      COL0_RANGE COL0_ALPHA COL0_ATTR,
 *                      [COL1_RANGE][COL1_ALPHA][COL1_ATTR],...,
 *                      BASE_ADDR SHUFFLE
 * */
int main(int argc, char* argv[]) {
    srand(time(nullptr));
    CarType cnt = (CarType)stoull(argv[1]);
    int num_cols = stoi(argv[2]);
    if (argc != num_cols * 3 + 5) {
        log_error("wrong parameters");
        exit(1);
    }

    vector<KeyType> ranges;
    vector<AttrType> attrs;
    vector<double> alphas;
    for(auto i = 0; i < num_cols; i++) {
        ranges.emplace_back((KeyType)stoi(argv[3+i*3]));
        alphas.emplace_back(stod(argv[4+i*3]));
        attrs.emplace_back((KeyType)stoi(argv[5+i*3]));
    }
    string base_addr = string(argv[argc-2]);
    bool is_shuffle = (bool)stoi(argv[argc-1]);

    /*generate mapping*/
    CarType **mappings = nullptr;
    CUDA_MALLOC(&mappings, sizeof(CarType*)*num_cols, nullptr);
    for(auto i = 0; i < num_cols; i++) mappings[i] = nullptr;

    if(!is_shuffle) {
        for(auto i = 0; i < num_cols; i++) { //set acescending mapping to disable shuffling
            CUDA_MALLOC(&mappings[i], sizeof(CarType)*ranges[i], nullptr);
#pragma omp parallel for
            for(auto j = 0; j < ranges[i]; j++) mappings[i][j] = j;
        }
    }
    skewed_writer<KeyType, CarType, KeyType>(cnt, ranges, alphas, attrs, base_addr, mappings);

    return 0;
}