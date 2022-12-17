//
// Created by Bryan on 14/4/2020.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "log.h"
#include "timer.h"
#include "pretty_print.h"
#include "types.h"
#include "file_io.h"
using namespace std;

#define SEP  ("\t|\n")

/*
 * Read selective attrs of a .tbl file from TPC-H benchmark and write to a .db file with mmap
 * retrieve_attr_idxes: retrieve these attrs from the tbl file and rename the attr as 0, 1,...
 * */
template<typename DataType, typename CntType>
void tbl_2_db(string tbl_addr, vector<AttrType> retrieve_attr_idxes, vector<AttrType> new_attr_idxes) {
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;
    CntType table_cardinality = 0;
    string tbl_prefix = ".tbl";
    string db_prefix = ".db";
    vector<DataType*> retrieved_data;

    if ((fp = fopen(tbl_addr.c_str(), "r")) == nullptr) {
        log_error("File %s not exist.", tbl_addr.c_str());
        exit(1);
    }
    log_info("Read file from %s", tbl_addr.c_str());

    /*use linux "wc -l" inst to get the number of rows (table_cardinality)*/
    string command = string("wc -l ")+tbl_addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr) {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');
    table_cardinality = stoull(cnt_token);
//    table_cardinality = 1500000000;
    log_info("Number of rows: %llu", table_cardinality);

    /*allocate the memory objects for data*/
    retrieved_data.reserve(retrieve_attr_idxes.size());
    for(auto i = 0; i < retrieve_attr_idxes.size(); i++) {
        DataType *temp = (DataType*)malloc(sizeof(DataType)*table_cardinality);
        retrieved_data.emplace_back(temp);
    }

    /*sort the indexes according to retrieve_attr_idxes*/
    vector<pair<AttrType,AttrType>> retrieve_atrr_idxes_pair;
    for(auto i = 0; i < retrieve_attr_idxes.size(); i++) {
        retrieve_atrr_idxes_pair.emplace_back(make_pair(retrieve_attr_idxes[i], i));
    }
    sort(retrieve_atrr_idxes_pair.begin(), retrieve_atrr_idxes_pair.end());

    /*make it out of order*/
//    CntType *mapping = new CntType[table_cardinality];
//    for(auto i = 0; i < table_cardinality; i++) mapping[i] = i;
//    for(auto x = table_cardinality-1; x > 0; x--) {
//        auto target_idx = rand() % x;
//        std::swap(mapping[x], mapping[target_idx]);
//    }

    /*read from file*/
    CntType row_idx = 0;
    while (-1 != getline(&line, &len, fp)) {
        char *stringRet;
        int attr_idx = 0, read_idx = 0;
        stringRet = strtok(line, SEP); //partition
        while (stringRet != nullptr) {
            if (read_idx == retrieve_atrr_idxes_pair[attr_idx].first) {
                auto cur_attr_pos = retrieve_atrr_idxes_pair[attr_idx].second;
//                retrieved_data[cur_attr_pos][mapping[row_idx]] = stoul(stringRet); //reordering
                retrieved_data[cur_attr_pos][row_idx] = stoul(stringRet);
                attr_idx++;
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
        row_idx++;
    }
    if(line) free(line);
    fclose(fp);

    log_info("Finish reading the .tbl file");
    auto myTime = t.elapsed();
    log_info("%llu rows are read in %.2f s, throughput: %.2f MB/s.", table_cardinality, myTime,
             1.0*sizeof(DataType)*retrieve_attr_idxes.size()*table_cardinality/1024/1024/myTime);

    string db_addr = to_string(table_cardinality);
    for(auto i = 0; i < new_attr_idxes.size(); i++) {
        db_addr = db_addr + "_" + to_string(new_attr_idxes[i]);
    }
    db_addr += db_prefix;
    log_info("Write file %s", db_addr.c_str());
    t.reset();
    write_rel_cols_mmap(db_addr.c_str(), retrieved_data, table_cardinality);
    myTime = t.elapsed();
    log_info("Finish writing the %s file", db_prefix.c_str());
    log_info("%llu rows are written in %.2f s, throughput: %.2f MB/s.", table_cardinality, myTime,
             1.0*sizeof(DataType)*retrieve_attr_idxes.size()*table_cardinality/1024/1024/myTime);
}

/*
 * usage:
 *  ./tbl_2_db TBL_FILE_NAME RETRIEVE_ATTR_IDX0, RETRIEVE_ATTR_IDX1,..., NEW_ATTR_IDX0, NEW_ATTR_IDX1,...
 * */
int main(int argc, char *argv[]) {
    assert((argc % 2) == 0);
    auto num_attrs = (argc-2)/2;
    string tbl_addr = string(argv[1]);
    vector<AttrType> retrieve_attr_idxes, new_attr_idxes;
    int i = 2;
    for(; i < num_attrs+2; i++) retrieve_attr_idxes.emplace_back((AttrType)stoi(argv[i]));
    for(; i < 2*num_attrs+2; i++) new_attr_idxes.emplace_back((AttrType)stoi(argv[i]));
    tbl_2_db<KeyType,CarType>(tbl_addr, retrieve_attr_idxes, new_attr_idxes);
    return 0;
}