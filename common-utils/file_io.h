//
// Created by Bryan on 10/12/2019.
//

#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda/cuda_fake/fake.h"
#include "openmp_fake.h"
#endif

#include <iostream>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <vector>
#include <cassert>
#include "timer.h"
#include "log.h"

/*write columns of data using ostream*/
template<typename DataType, typename CntType>
void write_rel_cols(const char *file_name, std::vector<DataType*> data, CntType len) {
    Timer write_time;
    auto num_cols = data.size();
    std::ofstream os(file_name, std::ios::out);
    if (!os) {
        log_error("File does not exist.");
        exit(1);
    }
    for(CntType i = 0; i < len; i++) {
        for(auto c = 0; c < num_cols; c++) {
            os << data[c][i];
            if (c != num_cols - 1) os<<' ';
        }
        os<<'\n';
    }
    os.close();
    log_info("write_data time: %.2f s.", write_time.elapsed());
}

template<typename DataType, typename CntType>
void write_rel_cols_mmap(const char *file_name, std::vector<DataType*> data, CntType len) {
    Timer write_time;
    auto num_cols = data.size();
    auto fd = open(file_name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    auto total_len = sizeof(DataType)*len*num_cols;
    log_info("total_len: %llu", total_len);
    auto ret = ftruncate(fd, total_len);
    DataType *buf = (DataType*)mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (buf == MAP_FAILED) {
        log_error("mmap (for write) gets wrong.");
        exit(1);
    }
    assert(ret == 0);
    CntType idx = 0;
    for(auto c = 0; c < num_cols; c++) {
        memcpy(buf+idx, data[c], sizeof(DataType)*len);
        idx += len;
    }
    close(fd);
    munmap(buf, total_len);
    log_info("write_data time: %.2f s.", write_time.elapsed());
}

template<typename DataType, typename CntType>
void write_rel_cols_csv(const char *file_name, std::vector<DataType*> data, CntType len) {
    Timer write_time;
    auto num_cols = data.size();
    std::ofstream outfile;
    outfile.open(file_name);
    for(auto i = 0; i < len; i++) {
        for(auto j = 0; j < num_cols; j++) {
            outfile << data[j][i];
            if (j != num_cols-1) outfile<<'\t';
            else                 outfile<<std::endl;
        }
    }
    outfile.close();
    log_info("write_data time: %.2f s.", write_time.elapsed());
}

/*
 * todo: if len set to 0, call wc -l to derive the number of items
 * */
template<typename DataType, typename CntType>
std::vector<DataType*> read_rel_cols_mmap(const char *file_name, int num_cols, CntType len) {
    log_debug("len in read func: %d", len);
    Timer read_time;
    auto fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        log_error("Faile to open file: %s", file_name);
        exit(1);
    }
    long long total_size = lseek64(fd,0,SEEK_END);
    auto *buf = (DataType*)mmap(NULL, total_size, PROT_READ, MAP_SHARED, fd, 0);
    if (buf == MAP_FAILED) {
        log_error("mmap (for read) gets wrong.");
        exit(1);
    }

    std::vector<DataType*> data_out;
    data_out.reserve(num_cols);
    for(auto i = 0; i < num_cols; i++) {
        DataType *data_temp = nullptr;
        cudaMallocManaged((void**)&data_temp, sizeof(DataType)*len); //UM tractable memory
        data_out.emplace_back(data_temp);
    }
    CntType idx = 0;
    for(auto c = 0; c < num_cols; c++) {
        memcpy(data_out[c], buf+idx, sizeof(DataType)*len);
        idx += len;
    }
    close(fd);
    munmap(buf, total_size);
    log_info("Read file: %s, size: %.1f MB, time: %.2f s.", file_name, 1.0f*total_size/1024/1024, read_time.elapsed());
    return data_out;
}

template<typename DataType, typename CntType>
std::vector<DataType*> read_rel_cols_mmap_CPU(const char *file_name, int num_cols, CntType len) {
    Timer read_time;
    auto fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        log_error("Faile to open file: %s", file_name);
        exit(1);
    }
    long long total_size = lseek64(fd,0,SEEK_END);
    auto *buf = (DataType*)mmap(NULL, total_size, PROT_READ, MAP_SHARED, fd, 0);
    if (buf == MAP_FAILED) {
        log_error("mmap (for read) gets wrong.");
        exit(1);
    }

    std::vector<DataType*> data_out;
    data_out.reserve(num_cols);
    for(auto i = 0; i < num_cols; i++) {
        DataType *data_temp = (DataType *)malloc(sizeof(DataType)*len); //UM tractable memory
        data_out.emplace_back(data_temp);
    }
    CntType idx = 0;
    for(auto c = 0; c < num_cols; c++) {
        memcpy(data_out[c], buf+idx, sizeof(DataType)*len);
        idx += len;
    }
    close(fd);
    munmap(buf, total_size);
    log_info("Read file: %s, size: %.1f MB, time: %.2f s.", file_name, 1.0f*total_size/1024/1024, read_time.elapsed());
    return data_out;
}