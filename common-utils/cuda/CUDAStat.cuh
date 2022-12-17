//
// Created by Bryan on 10/7/2019.
//
/*CUDA memory stat class*/
#pragma once

#include <map>
#include <vector>
#include <cassert>
#include <cstring>

#include "cuda_base.cuh"

/*memory statistics*/
class CUDAMemStat {
private:
    long cur_use; //current use of memory in bytes
    long max_use; //maximum use of memory in bytes
    long acc_use; //accumulated memory used in bytes
    bool valid;   //whether the MemStat is valid

    std::map<unsigned long, long> addr_bytes; //store the allocated address and sizes
public:
    CUDAMemStat() {
        cur_use = 0;
        max_use = 0;
        acc_use = 0;
        valid = true;
    }
    void malloc_mem_stat(unsigned long addr, long add_bytes) {
        if (!valid) return;

        cur_use += add_bytes;
        acc_use += add_bytes;
        if (cur_use > max_use) {
            max_use = cur_use;
        }

        /*record the allocated address and memory size*/
        addr_bytes.insert(std::make_pair(addr, add_bytes));
    }
    void delete_mem_stat(unsigned long addr) {
        if (!valid) return;

        auto iter = addr_bytes.find(addr);
        if (iter == addr_bytes.end()) {
            log_warn("MemStat does not match. The Mem stat becomes invalid.");
            valid = false;
            return;
        }

        long delete_bytes = iter->second;
        addr_bytes.erase(iter); //remove the address
        cur_use -= delete_bytes;
        assert(cur_use >= 0);
    }
    long get_max_use() {
        return max_use;
    }
    long get_acc_use() {
        return acc_use;
    }
    long get_cur_use() {
        return cur_use;
    }
    void reset() {
        cur_use = 0;
        max_use = 0;
        acc_use = 0;

        addr_bytes.clear();
        valid = true;
    }
};

/*time statistics*/
class CUDATimeStat {
private:
    uint32_t idx;                               //current start idx of the kernel interested
    std::vector<std::string> file_name;         //name of the file the kernel is invoked in
    std::vector<std::string> host_func_name;   //name of the host function the kernel is invoked in
    std::vector<std::string> kernel_name;       //name of the kernel invoked
    std::vector<float> kernel_time;              //kernel time

public:
    CUDATimeStat() {
        idx = 0;
    }

    uint32_t get_idx() {
        return kernel_time.size();
    }

    void reset() {
        idx = (uint32_t)kernel_time.size();
    }

    float elapsed() {
        auto end = (uint32_t)kernel_time.size();
        float res = 0.0;
        for(auto i = idx; i < end; i++)
            res += this->kernel_time[i];
        return res;
    }

    float diff_time(uint32_t start_idx) {
        float res = 0.0;
        for(auto i = start_idx; i < (uint32_t)kernel_time.size(); i++)
            res += this->kernel_time[i];
        return res;
    }

    void insert_record( std::string file,
                        std::string host_func,
                        std::string kernel,
                        float ker_time) {
        file_name.emplace_back(file);
        host_func_name.emplace_back(host_func);
        kernel_name.emplace_back(kernel);
        kernel_time.emplace_back(ker_time);
    }
};

/*Wrapper for memory malloc, need to touch the malloced memory once before using it*/
template <typename T>
void CUDA_MALLOC(T **addr, size_t malloc_bytes, CUDAMemStat *stat, cudaStream_t stream = 0) {
    assert(malloc_bytes > 0);
    checkCudaErrors(cudaMallocManaged((void**)addr, malloc_bytes));
    checkCudaErrors(cudaMemPrefetchAsync(*addr,malloc_bytes, DEVICE_ID, stream));
    if (stat)
        stat->malloc_mem_stat((unsigned long)*addr, malloc_bytes);
}

template <typename T>
void CUDA_MALLOC_NO_PREFETCH(T **addr, size_t malloc_bytes, CUDAMemStat *stat) {
    assert(malloc_bytes > 0);
    checkCudaErrors(cudaMallocManaged((void**)addr, malloc_bytes));
    if (stat)
        stat->malloc_mem_stat((unsigned long)*addr, malloc_bytes);
}

template <typename T>
void CUDA_FREE(T *addr, CUDAMemStat *stat) {
    checkCudaErrors(cudaFree(addr));
    if (stat)
        stat->delete_mem_stat((unsigned long)addr);
}
