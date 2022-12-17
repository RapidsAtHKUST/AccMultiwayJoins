//
// Created by Bryan on 13/7/2020.
//

#pragma once
#include "types.h"
#include <string>
#include "helper.h"

template<typename DataType, typename CntType>
void load_single_rel(string full_file_name, Relation<DataType, CntType> *&relation, CUDAMemStat *memstat) {
    CUDA_MALLOC(&relation, sizeof(Relation<DataType,CntType>), memstat);
    int num_columns = 2;
    vector<string> properties;
    auto file_name = get_file_name_from_addr(full_file_name);
    split_string(file_name, properties, "_");
    auto file_rows = stoull(properties[0]);
    assert(file_rows > 0);

    auto data_from_file = read_rel_cols_mmap<DataType,CntType>(
            full_file_name.c_str(), num_columns, file_rows);
    relation[0].init(num_columns, file_rows, memstat);
    for(auto a = 0; a < num_columns; a++) {
        relation[0].data[a] = data_from_file[a];
    }
}

/*
 * Load all the relation files in the directory
 * */
template<typename DataType, typename CntType>
void load_multi_rels(
        string dir_name, Relation<DataType, CntType> *&relations,
        uint32_t &num_tables, CUDAMemStat *memstat) {
    auto file_list = list_files(dir_name.c_str());
    if (0 == file_list.size()) {
        log_error("The directory is empty");
        exit(1);
    }
    num_tables = (uint32_t)file_list.size();

    /*read the relations from files*/
    CUDA_MALLOC(&relations, sizeof(Relation<DataType,CntType>)*num_tables, memstat);
    for(auto i = 0; i < num_tables; i++) {
        string cur_file_name = file_list[i].substr(0,file_list[i].length()-rel_prefix.size());
        vector<string> properties;
        split_string(cur_file_name, properties, "_");
        auto data_from_file = read_rel_cols_mmap<DataType,CntType>(
                (dir_name+"/"+file_list[i]).c_str(),
                (int)properties.size() - 1, stoull(properties[0]));

        int cur_idx;
        log_no_newline("Enter the table order: ");
//            cin>>cur_idx; //input table order
        cur_idx = i;
        assert(cur_idx < num_tables);

        relations[cur_idx].init((uint32_t)properties.size() - 1, stoull(properties[0]), memstat);
        for(auto a = 0; a < relations[cur_idx].num_attrs; a++) {
            relations[cur_idx].attr_list[a] = (AttrType)stoi(properties[a+1]);
            relations[cur_idx].data[a] = data_from_file[a];
        }
    }
}