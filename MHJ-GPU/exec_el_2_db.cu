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

#define SEP  (" |\t")

template<typename DataType, typename CntType>
void edgelist_2_db(string edgelist_addr) {
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;
    CntType table_cardinality = 0;
    string db_prefix = ".db";
    vector<DataType*> retrieved_data;

    if ((fp = fopen(edgelist_addr.c_str(), "r")) == nullptr) {
        log_error("File %s not exist.", edgelist_addr.c_str());
        exit(1);
    }
    log_info("Read file from %s", edgelist_addr.c_str());

    /*use linux "wc -l" inst to get the number of rows (table_cardinality)*/
    string command = string("wc -l ")+edgelist_addr;
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
    log_info("Number of rows: %llu", table_cardinality);

    /*allocate the memory objects for data*/
    for(auto i = 0; i < 2; i++) {
        DataType *temp = (DataType*)malloc(sizeof(DataType)*table_cardinality);
        retrieved_data.emplace_back(temp);
    }

    /*read from file*/
    CntType row_idx = 0;
    while (-1 != getline(&line, &len, fp)) {
        char *stringRet;
        int attr_idx = 0;
        stringRet = strtok(line, SEP); //partition
        while (stringRet != nullptr) {
            retrieved_data[attr_idx][row_idx] = stoul(stringRet);
            stringRet = strtok(nullptr, SEP);
            attr_idx++;
        }
        row_idx++;
    }
    if(line) free(line);
    fclose(fp);

    log_info("Finish reading the edgelist file");
    auto myTime = t.elapsed();
    log_info("%llu rows are read in %.2f s, throughput: %.2f MB/s.", table_cardinality, myTime,
             1.0*sizeof(DataType)*2*table_cardinality/1024/1024/myTime);

    string db_addr = edgelist_addr + db_prefix;
    log_info("Write file %s", db_addr.c_str());
    t.reset();
    write_rel_cols_mmap(db_addr.c_str(), retrieved_data, table_cardinality);
    myTime = t.elapsed();
    log_info("Finish writing the %s file", db_prefix.c_str());
    log_info("%llu rows are written in %.2f s, throughput: %.2f MB/s.", table_cardinality, myTime,
             1.0*sizeof(DataType)*2*table_cardinality/1024/1024/myTime);
}

/*
 * usage:
 *  ./edgelist_2_db edge_list_file
 * */
int main(int argc, char *argv[]) {
    assert(argc == 2);
    edgelist_2_db<KeyType,CarType>(string(argv[1]));
    return 0;
}