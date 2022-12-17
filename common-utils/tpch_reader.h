//
// Created by Bryan on 7/9/2019.
//

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "cuda/CUDAStat.cuh"
#include "log.h"
#include "timer.h"
#include "pretty_print.h"
#include "../hash-chain-join-gpu/TPCH-Q3/tpch_class.h"
using namespace std;

#ifndef DATA_ADDR
#define DATA_ADDR ""
#endif

#define SEP  ("\t|\n")

template<typename DataType, typename CntType>
void readTPCHTable(
        string tableName, const int sf, vector<int> attrIdxes, //attrIdxes in ascending order
        vector<DataType*> &data, CntType &cnt,
        CUDAMemStat *memstat)
{
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;

    /*construct the file dir_addr*/
    auto *addr = (char *)malloc(sizeof(char)*1000);
//    strcpy(dir_addr, DATA_ADDR);
    strcpy(addr, "/disk/zlai/datasets/TPC-H/tpch_s_");
    strcat(addr, to_string(sf).c_str());
    strcat(addr, "/");
    strcat(addr, tableName.c_str());
    strcat(addr, ".tbl");

    if ((fp = fopen(addr, "r")) == nullptr)
    {
        log_error("File %s not exist.", addr);
        exit(1);
    }
    log_info("Read file from %s", addr);

    /*use linux "wc -l" inst to get the number of rows*/
    string command = string("wc -l ")+addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr)
    {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');
    cnt = stoul(cnt_token);

    /*allocate the memory objects for data*/
    data.reserve(attrIdxes.size());
    for(auto i = 0; i < attrIdxes.size(); i++)
    {
        DataType *temp = nullptr;
        CUDA_MALLOC(&temp, sizeof(DataType)*cnt, memstat);
        data.emplace_back(temp);
    }

    /*read from file*/
    CntType row_idx = 0;
    while (-1 != getline(&line, &len, fp))
    {
        char *stringRet;
        int attr_idx = 0;
        int read_idx = 0;

        /*partition*/
        stringRet = strtok(line, SEP);
        while (stringRet != nullptr)
        {
            if (read_idx == attrIdxes[attr_idx])
            {
                data[attr_idx][row_idx] = stoul(stringRet);
                attr_idx++;
//                if (attr_idx == attrIdxes.size()) break; //have read the last needed attr
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
        row_idx++;
    }

    if(line) free(line);
    free(addr);
    fclose(fp);

    auto myTime = t.elapsed();
    log_info("Table %s: %lu rows are read in %.2f s, throughput: %.2f MB/s.", tableName.c_str(), cnt, myTime,
             1.0*sizeof(DataType)*attrIdxes.size()*cnt/1024/1024/myTime);
}

template<typename DataType, typename CntType>
void readGeneralTable(
        char* addr, vector<int> attrIdxes, //attrIdxes in ascending order
        vector<DataType*> &data, CntType &cnt,
        CUDAMemStat *memstat)
{
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;

    /*construct the file dir_addr*/

    if ((fp = fopen(addr, "r")) == nullptr)
    {
        log_error("File not exist.");
        exit(1);
    }
    log_info("Read file from %s", addr);

    /*use linux "wc -l" inst to get the number of rows*/
    string command = string("wc -l ")+addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr)
    {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');
    cnt = stoul(cnt_token);
    log_info("%d lines are detected.", cnt);

    /*allocate the memory objects for data*/
    data.reserve(attrIdxes.size());

    for(auto i = 0; i < attrIdxes.size(); i++)
    {
        DataType *temp = nullptr;
        CUDA_MALLOC(&temp, sizeof(DataType)*cnt, memstat);
        data.emplace_back(temp);
    }
    int numSelfLoops = 0;

    /*read from file*/
    CntType row_idx = 0;
    while (-1 != getline(&line, &len, fp))
    {
        char *stringRet;
        int attr_idx = 0;
        int read_idx = 0;

        /*partition*/
        stringRet = strtok(line, SEP);
        while (stringRet != nullptr)
        {
            if (read_idx == attrIdxes[attr_idx])
            {
                data[attr_idx][row_idx] = stoul(stringRet);
                attr_idx++;
//                if (attr_idx == attrIdxes.size()) break; //have read the last needed attr
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
        if (data[0][row_idx] != data[1][row_idx]) //eliminate self loop
            row_idx++;
        else
            numSelfLoops++;
    }

    cnt = row_idx;
    log_info("#edges without self loop: %d.", cnt);

    if(line) free(line);
    fclose(fp);

    log_info("Eliminated %d self loops in read function.", numSelfLoops);

    auto myTime = t.elapsed();
    log_info("%lu rows are read in %.2f s, throughput: %.2f MB/s.", cnt, myTime,
             1.0*sizeof(DataType)*attrIdxes.size()*cnt/1024/1024/myTime);
}

void readCustomerWithFilter(
        const char *addr,
        Customer &customer,
        char *SEG,
        CUDAMemStat *memstat)
{
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;

    /*construct the file dir_addr*/
    if ((fp = fopen(addr, "r")) == nullptr)
    {
        log_error("File not exist.");
        exit(1);
    }
    log_info("Read file from %s", addr);

    /*use linux "wc -l" inst to get the number of rows*/
    string command = string("wc -l ")+addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr)
    {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');

    auto cnt = stoul(cnt_token);
    log_info("%d lines are detected.", cnt);

    /*allocate the memory objects for data*/
    CUDA_MALLOC(&customer.c_custkey_bitmap, sizeof(bool)*cnt, memstat);
    checkCudaErrors(cudaMemset(customer.c_custkey_bitmap, 0, sizeof(bool)*cnt));

    /*read from file*/
    int finalCnt = 0;
    while (-1 != getline(&line, &len, fp))
    {
        char *stringRet;
        int read_idx = 0;
        int32_t cur_custkey = -1;

        /*partition*/
        stringRet = strtok(line, SEP);
        while (stringRet != nullptr)
        {
            if (read_idx == 0) //custkey
                cur_custkey = stoul(stringRet);
            else if (read_idx == 6) //mktsegment: 6 in tbl, 1 in txt
            {
                if (!strcmp(stringRet, SEG))
                {
                    customer.c_custkey_bitmap[cur_custkey-1] = 1;
                    finalCnt ++;
                }
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
    }
    customer.cardinality = finalCnt;

    /*check*/
//    long total = 0;
//    for(auto i = 0; i < cnt; i++)
//    {
//        if (customer.c_custkey_bitmap[i] == 1) total++;
//    }
//    log_info("Total number of set bits: %d.", total);

    if(line) free(line);
    fclose(fp);

    auto myTime = t.elapsed();
    log_trace("Customer read finished, %d items after filter.", finalCnt);
}

//e.g. 1996-01-02 -> 19960102
int date2int(char *date)
{
    char year[5], month[3], day[3];
    strncpy(year, date, 4); year[4] = '\0';
    strncpy(month, &date[5], 2); month[2] = '\0';
    strncpy(day, &date[8], 2); day[2] = '\0';

    return stoi(year) * 10000 + stoi(month) * 100 + stoi(day);
}

void readOrdersWithFilter(
        const char *addr,
        Orders &orders,
        char *ORDER_DATE_THRES,
        CUDAMemStat *memstat)
{
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;

    /*construct the file dir_addr*/
    if ((fp = fopen(addr, "r")) == nullptr)
    {
        log_error("File not exist.");
        exit(1);
    }
    log_info("Read file from %s", addr);

    /*use linux "wc -l" inst to get the number of rows*/
    string command = string("wc -l ")+addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr)
    {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');

    auto cnt = stoul(cnt_token);
    log_info("%d lines are detected.", cnt);

    /*allocate the memory objects for data*/
    CUDA_MALLOC(&orders.o_orderkey, sizeof(int32_t)*cnt, memstat);
    CUDA_MALLOC(&orders.o_custkey, sizeof(int32_t)*cnt, memstat);
    CUDA_MALLOC(&orders.o_orderdate, sizeof(int32_t)*cnt, memstat);

    int ORDER_DATE_VAL = date2int(ORDER_DATE_THRES); //switch to int

    /*read from file*/
    int finalCnt = 0;
    while (-1 != getline(&line, &len, fp))
    {
        char *stringRet;
        int read_idx = 0;
        int32_t cur_orderkey = -1;
        int32_t cur_custKey = -1;

        /*partition*/
        stringRet = strtok(line, SEP);
        while (stringRet != nullptr)
        {
            if (read_idx == 0) //orderkey
                cur_orderkey = stoul(stringRet);
            else if (read_idx == 1) //custkey
                cur_custKey = stoul(stringRet);
            else if (read_idx == 4) //orderdate: 4 in tbl, 2 in txt
            {
                int32_t cur_orderdate = date2int(stringRet);
                if (cur_orderdate < ORDER_DATE_VAL) {
                    orders.o_orderkey[finalCnt] = cur_orderkey;
                    orders.o_custkey[finalCnt] = cur_custKey;
                    orders.o_orderdate[finalCnt] = cur_orderdate;
                    finalCnt ++;
                }
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
    }
    orders.cardinality = finalCnt;

    if(line) free(line);
    fclose(fp);

    auto myTime = t.elapsed();
    log_trace("Orders read finished, %d items after filter.", finalCnt);
}

void readLineitemWithFilter(
        const char *addr,
        Lineitem &lineitem,
        char *SHIP_DATE_THRES,
        CUDAMemStat *memstat)
{
    Timer t;
    FILE *fp, *fc;
    char *line = nullptr;
    size_t len = 0;

    /*construct the file dir_addr*/
    if ((fp = fopen(addr, "r")) == nullptr)
    {
        log_error("File not exist.");
        exit(1);
    }
    log_info("Read file from %s", addr);

    /*use linux "wc -l" inst to get the number of rows*/
    string command = string("wc -l ")+addr;
    fc = popen(command.c_str(), "r");
    if (fc == nullptr)
    {
        log_error("popen error.");
        exit(1);
    }
    char popen_data[500];
    fgets(popen_data, sizeof(popen_data), fc);
    stringstream ss(popen_data);
    string cnt_token;
    getline(ss, cnt_token, ' ');

    auto cnt = stoul(cnt_token);
    log_info("%d lines are detected.", cnt);

    /*allocate the memory objects for data*/
    CUDA_MALLOC(&lineitem.l_orderkey, sizeof(int32_t)*cnt, memstat);
    CUDA_MALLOC(&lineitem.l_expendedprice, sizeof(double)*cnt, memstat);

    int SHIP_DATE_VAL = date2int(SHIP_DATE_THRES); //switch to int

    /*read from file*/
    int finalCnt = 0;
    while (-1 != getline(&line, &len, fp))
    {
        char *stringRet;
        int read_idx = 0;
        int32_t cur_orderkey = -1;
        double cur_expendedprice = -1;

        /*partition*/
        stringRet = strtok(line, SEP);
        while (stringRet != nullptr)
        {
            if (read_idx == 0) //orderkey
                cur_orderkey = stoul(stringRet);
            else if (read_idx == 5) //expendedprice: 5 in tbl, 1 in txt
                cur_expendedprice = stod(stringRet);
            else if (read_idx == 10) //shipdate: 10 in tbl, 2 in txt
            {
                int cur_shipdate = date2int(stringRet);
                if (cur_shipdate > SHIP_DATE_VAL) {
                    lineitem.l_orderkey[finalCnt] = cur_orderkey;
                    lineitem.l_expendedprice[finalCnt] = cur_expendedprice;
                    finalCnt ++;
                }
            }
            stringRet = strtok(nullptr, SEP);
            read_idx++;
        }
    }
    lineitem.cardinality = finalCnt;

    if(line) free(line);
    fclose(fp);

    auto myTime = t.elapsed();
    log_trace("Lineitem read finished, %d items after filter.", finalCnt);
}