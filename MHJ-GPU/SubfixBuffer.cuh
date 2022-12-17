//
// Created by Bryan on 15/9/2020.
//
#pragma once

#include "cuda/CUDAStat.cuh"
#include "../../dependencies/topkapi/topkapi.h"
#include "pretty_print.h"
#include "Indexing/radix_partitioning.cuh"

#define FIB_BUFFER_MAX_LEN  (256) /*each attribute stores BUFFER_LEN frequently used items*/
#define FIB_BUFFER_BUCKETS  (256)

enum sb_state {
    NOT_PROCESSED, //0
    UNDER_PROCESS, //1
    PROCESSED      //2
};

template<typename DataType, typename CntType>
struct SubfixBuffer {
    DataType **m_items;
    volatile CntType **m_cnts;   //#results of each SB item
    volatile CntType **m_starts; //start location of results of each SB item in output table
    sb_state **m_states;         //states of each sb item
    DataType **m_ht;             //hash table for SB items
    int **m_ht_bucptrs;          //bucket ptrs for hash table

    int *m_num_items;               //#SB items for each data column
    int m_num_cols;                 //#data columns recorded
    uint32_t **m_num_tasks_related; //#WS tasks related to each SB item

    CUDAMemStat *m_memstat;
    CUDATimeStat *m_timing;

    void init(int num_cols, CUDAMemStat *memstat, CUDATimeStat *timing) {
        this->m_num_cols = num_cols;
        this->m_memstat = memstat;
        this->m_timing = timing;

        CUDA_MALLOC(&this->m_items, sizeof(DataType *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_cnts, sizeof(CntType *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_starts, sizeof(CntType *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_states, sizeof(sb_state *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_ht, sizeof(DataType *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_ht_bucptrs, sizeof(int *) * this->m_num_cols, this->m_memstat);
        CUDA_MALLOC(&this->m_num_tasks_related, sizeof(uint32_t *) * this->m_num_cols, this->m_memstat);

        for (auto i = 0; i < m_num_cols; i++) {
            CUDA_MALLOC(&this->m_items[i], sizeof(DataType) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            CUDA_MALLOC((void **) &this->m_cnts[i], sizeof(CntType) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            CUDA_MALLOC((void **) &this->m_starts[i], sizeof(CntType) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            CUDA_MALLOC(&this->m_states[i], sizeof(sb_state) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            CUDA_MALLOC(&this->m_ht[i], sizeof(DataType) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            CUDA_MALLOC(&this->m_ht_bucptrs[i], sizeof(int) * (FIB_BUFFER_BUCKETS + 1), this->m_memstat);
            CUDA_MALLOC(&this->m_num_tasks_related[i], sizeof(uint32_t) * FIB_BUFFER_MAX_LEN, this->m_memstat);
            checkCudaErrors(cudaMemset((CntType *) this->m_cnts[i], 0, sizeof(CntType) * FIB_BUFFER_MAX_LEN));
            checkCudaErrors(cudaMemset(this->m_num_tasks_related[i], 0, sizeof(uint32_t) * FIB_BUFFER_MAX_LEN));
        }
        CUDA_MALLOC(&this->m_num_items, sizeof(int) * this->m_num_cols, this->m_memstat);
    }

    /*compute the top-k frequent items*/
    template<bool print_res = false>
    void compute_topk(DataType **data, CntType *cnts) {
        for (auto i = 0; i < this->m_num_cols; i++) {
            topkapi(data[i], cnts[i], this->m_items[i], nullptr, m_num_items[i], FIB_BUFFER_MAX_LEN);
            log_info("Actual buffer length for col %d: %d.", i, m_num_items[i]);
            if (m_num_items[i] > 0) {
                RadixPartitioner<DataType, DataType, int> rp(m_num_items[i], 1, FIB_BUFFER_BUCKETS, this->m_memstat,
                                                             this->m_timing);
                rp.splitK(this->m_items[i], this->m_ht[i], this->m_ht_bucptrs[i]);
                if (print_res) {
                    cout << "BufferItems " << i << ": " << pretty_print_array(this->m_items[i], this->m_num_items[i])
                         << endl;
                    cout << "Buffer hash table: " << pretty_print_array(this->m_ht[i], this->m_num_items[i]) << endl;
                    cout << "Buffer hash table ptr: "
                         << pretty_print_array(this->m_ht_bucptrs[i], (FIB_BUFFER_BUCKETS + 1)) << endl;

                    for (auto j = 0; j < FIB_BUFFER_BUCKETS; j++) {
                        cout << "Bucket " << j << ": ";
                        for (auto k = this->m_ht_bucptrs[i][j]; k < this->m_ht_bucptrs[i][j + 1]; k++) {
                            cout << this->m_ht[i][k] << ' ';
                        }
                        cout << endl;
                    }
                    std::cout << "-------------------------------" << std::endl;
                    cudaMemPrefetchAsync(this->m_ht[i], sizeof(DataType) * FIB_BUFFER_MAX_LEN, DEVICE_ID);
                    cudaMemPrefetchAsync(this->m_ht_bucptrs[i], sizeof(int) * (FIB_BUFFER_BUCKETS + 1), DEVICE_ID);
                }
            }
        }

        /*reset m_states to 0 (not processed)*/
        for (int i = 0; i < this->m_num_cols; i++) {
            checkCudaErrors(cudaMemset(this->m_states[i], 0, sizeof(sb_state) * FIB_BUFFER_MAX_LEN));
        }
    }
};