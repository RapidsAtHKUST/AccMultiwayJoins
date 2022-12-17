//
// Created by Bryan on 18/4/2020.
//

#pragma once

#include "types.h"
#include "SubfixBuffer.cuh"

/*todo: the fib part should be put in a separate child class*/
/*recording all the tasks existed*/
template<typename DataType, typename CntType>
struct TaskBook {
    volatile DataType *m_iRes;    //intermediate results
    int m_num_attrs_in_iRes;       //length of each iRes
    volatile CntType *m_buc_starts; //bucket starts
    volatile CntType *m_buc_ends;   //bucket ends
    volatile char *m_cur_tables;    //current table the probe item (data) is in

    bool *m_is_fib_task;        //whether the task is related to a fib item
    int *m_fib_col_id;         //which column the fib item is in
    int *m_fib_item_id;         //item id of the fib item in the column

    CntType *m_cnt;
    CntType m_capacity;
    CUDAMemStat *m_memstat;

    void init(CntType cap, int num_attrs_in_iRes, CUDAMemStat *memStat) {
        this->m_capacity = cap;
        this->m_num_attrs_in_iRes = num_attrs_in_iRes;
        this->m_memstat = memStat;

        CUDA_MALLOC((void**)&m_iRes, sizeof(DataType)*m_num_attrs_in_iRes*m_capacity, memStat);
        CUDA_MALLOC((void**)&m_buc_starts, sizeof(CntType)*m_capacity, memStat);
        CUDA_MALLOC((void**)&m_buc_ends, sizeof(CntType)*m_capacity, memStat);
        CUDA_MALLOC((void**)&m_cur_tables, sizeof(char)*m_capacity, memStat);
        CUDA_MALLOC(&m_cnt, sizeof(CntType), memStat);
        checkCudaErrors(cudaMemset(m_cnt, 0, sizeof(CntType)));

        CUDA_MALLOC((void**)&this->m_is_fib_task, sizeof(bool)*cap, memStat);
        checkCudaErrors(cudaMemset(this->m_is_fib_task, false, sizeof(bool)*cap));

        CUDA_MALLOC((void**)&this->m_fib_col_id, sizeof(int)*cap, memStat);
        CUDA_MALLOC((void**)&this->m_fib_item_id, sizeof(int)*cap, memStat);
    }

    /*return the task id (position)*/
    __device__ CntType push_task(DataType *iRes, CntType bs, CntType bn, char curTable) {
        auto cur = atomicAdd(m_cnt, 1);
        /*todo: number of tasks may be larger than the capacity of the taskbook*/
        auto offset = cur*m_num_attrs_in_iRes;
        for(auto i = 0; i < m_num_attrs_in_iRes; i++) m_iRes[offset + i] = iRes[i];
        m_buc_starts[cur] = bs;
        m_buc_ends[cur] = bn;
        m_cur_tables[cur] = curTable;
        return cur;
    }

    __device__ CntType push_task_fib(DataType *iRes, CntType bs, CntType bn, char curTable,
                                     int fib_col, int fib_item) {
        auto cur = atomicAdd(m_cnt, 1);
        auto offset = cur*m_num_attrs_in_iRes;
        for(auto i = 0; i < m_num_attrs_in_iRes; i++) m_iRes[offset + i] = iRes[i];
        m_buc_starts[cur] = bs;
        m_buc_ends[cur] = bn;
        m_cur_tables[cur] = curTable;

        m_is_fib_task[cur] = true;
        m_fib_col_id[cur] = fib_col;
        m_fib_item_id[cur] = fib_item;
        return cur;
    }

    void reset(cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync(m_cnt, 0, sizeof(CntType), stream));
    }

    size_t get_size() {
        return sizeof(DataType)*m_num_attrs_in_iRes*m_capacity + sizeof(CntType)*4 + sizeof(char) + sizeof(int) + sizeof(CUDAMemStat*);
    }

    void clear() {
        CUDA_FREE((void*)m_iRes, m_memstat);
        CUDA_FREE((void*)m_buc_starts, m_memstat);
        CUDA_FREE((void*)m_buc_ends, m_memstat);
        CUDA_FREE((void*)m_cur_tables, m_memstat);
        CUDA_FREE((void*)m_cnt, m_memstat);

        CUDA_FREE((void*)m_is_fib_task, m_memstat);
        CUDA_FREE((void*)m_fib_col_id, m_memstat);
        CUDA_FREE((void*)m_fib_item_id, m_memstat);
    }

    /*combine another task book into this task book*/
    void combine(const TaskBook<DataType,CntType> *other, cudaStream_t stream=0) {
        if (other->m_cnt[0] == 0) return;
        assert(m_cnt[0] + other->m_cnt[0] <= m_capacity);
        assert(m_num_attrs_in_iRes == other->m_num_attrs_in_iRes);
        checkCudaErrors(cudaMemcpyAsync((void*)(m_iRes+m_num_attrs_in_iRes*m_cnt[0]), (void*)other->m_iRes,
                                        sizeof(DataType)*m_num_attrs_in_iRes*other->m_cnt[0], cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync((void*)(m_buc_starts+m_cnt[0]), (void*)other->m_buc_starts,
                                        sizeof(CntType)*other->m_cnt[0], cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync((void*)(m_buc_ends+m_cnt[0]), (void*)other->m_buc_ends,
                                        sizeof(CntType)*other->m_cnt[0], cudaMemcpyDeviceToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync((void*)(m_cur_tables+m_cnt[0]), (void*)other->m_cur_tables,
                                        sizeof(char)*other->m_cnt[0], cudaMemcpyDeviceToDevice, stream));
        cudaStreamSynchronize(stream);
        m_cnt[0] += other->m_cnt[0];
    }
};