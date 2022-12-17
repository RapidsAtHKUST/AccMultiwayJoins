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
 * Produce uniformly distributed dataset
 * */
//template<typename DataType, typename CntType, typename RangeType>
//void uniform(CntType cnt, vector<RangeType> ranges, vector<AttrType> attrs, string base_addr) {
//    log_info("In function %s", __FUNCTION__);
//    int num_attrs = (int)ranges.size();
//    string file_name = base_addr + "/" + to_string(cnt);
//
//    vector<DataType*> table;
//    for(auto i = 0; i < num_attrs; i++) {
//        file_name += ("_" + to_string(attrs[i]));
//        DataType *temp = new DataType[cnt];
//        uniform_generator(temp, cnt, 0, ranges[i]); //no unique primary keys
//        table.emplace_back(temp);
//    }
//    file_name += ".db";
//    log_trace("Data generated");
//    write_rel_cols_mmap(file_name.c_str(), table, cnt);
//    log_info("Generated file: %s", file_name.c_str());
//
//    for(auto t = 0; t < table.size(); t++) delete[] table[t];
//}
//
///* the keys are unique */
//template<typename DataType, typename CntType, typename RangeType>
//void uniform_key_unique(CntType cnt, vector<RangeType> ranges, vector<AttrType> attrs, string base_addr) {
//    log_info("In function %s", __FUNCTION__);
//    int num_attrs = (int)ranges.size();
//    string file_name = base_addr + "/" + to_string(cnt);
//
//    vector<DataType*> table;
//    for(auto i = 0; i < num_attrs; i++) {
//        file_name += ("_" + to_string(attrs[i]));
//        DataType *temp = new DataType[cnt];
////        if (i == 0) { //generate unique keys
//            for(auto j = 0; j < cnt; j++) temp[j] = j % ranges[i];
//            for(auto j = cnt-1; j > 0; j--) { //shuffle
//                auto x = rand() % j;
//                std::swap(temp[j], temp[x]);
//            }
////        }
////        else uniform_generator(temp, cnt, 0, ranges[i]); //generate non-unique values
//        table.emplace_back(temp);
//    }
//    file_name += ".db";
//    log_trace("Data generated");
//    write_rel_cols_mmap(file_name.c_str(), table, cnt);
//    log_info("Generated file: %s", file_name.c_str());
//
//    for(auto t = 0; t < table.size(); t++) delete[] table[t];
//}

template<typename DataType, typename CntType, typename RangeType>
void uniform_column_generator(
        CntType cnt, vector<RangeType> ranges, vector<AttrType> attrs,
        vector<int> gen_types, string base_addr) {
    log_info("In function %s", __FUNCTION__);
    int num_attrs = (int)ranges.size();
    string file_name = base_addr + "/" + to_string(cnt);

    vector<DataType*> table;
    for(auto i = 0; i < num_attrs; i++) {
        file_name += ("_" + to_string(attrs[i]));
        DataType *temp = new DataType[cnt];
        if (gen_types[i] == 0) { //uniform
            uniform_generator(temp, cnt, 0, ranges[i]); //generate non-unique values
        }
        else if (gen_types[i] == 1) { //uniform with unique values
            for(auto j = 0; j < cnt; j++) temp[j] = j % ranges[i];
            for(auto j = cnt-1; j > 0; j--) { //shuffle
                auto x = rand() % j;
                std::swap(temp[j], temp[x]);
            }
        }
        else {
            log_error("Wrong gen-type");
            exit(1);
        }
        table.emplace_back(temp);
        log_info("Generate column %i, gen_type=%d", i, gen_types[i]);
    }
    file_name += ".db";
    log_trace("Data generated");
    write_rel_cols_mmap(file_name.c_str(), table, cnt);
    log_info("Generated file: %s", file_name.c_str());

    for(auto t = 0; t < table.size(); t++) delete[] table[t];
};

/*
 * Write a single uniformly-distributed table with specifed columns and data range
 * Usage
 *   ./uniform_writter  TABLE_CARD NUM_COLS
 *                      COL0_RANGE COL0_ATTR, COL0_GEN_TYPE
 *                      [COL1_RANGE][COL1_ATTR][COL1_GEN_TYPE],
 *                      ...,
 *                      BASE_ADDR
 *   GEN_TYPE = 0: uniform
 *   GEN_TYPE = 1: uniform with unique keys
 * */
int main(int argc, char* argv[]) {
    srand(time(nullptr));
    CarType cnt = (CarType)stoull(argv[1]);
    int num_cols = stoi(argv[2]);
    if (argc != num_cols * 3 + 4) {
        log_error("wrong parameters");
        exit(1);
    }

    vector<KeyType> ranges;
    vector<AttrType> attrs;
    vector<int> gen_type;
    for(auto i = 0; i < num_cols; i++) {
        ranges.emplace_back((KeyType)stoi(argv[3+i*3]));
        attrs.emplace_back((KeyType)stoi(argv[4+i*3]));
        gen_type.emplace_back((KeyType)stoi(argv[5+i*3]));
    }
    string base_addr = string(argv[argc-1]);
    uniform_column_generator<KeyType, CarType, KeyType>(cnt, ranges, attrs, gen_type, base_addr);
    return 0;
}