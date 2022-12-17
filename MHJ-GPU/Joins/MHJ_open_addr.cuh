//
// Created by Bryan on 15/8/2019.
//
#pragma once

#include "cuda/primitives.cuh"
#include "timer.h"
#include "../types.h"
#include "../common_kernels.cuh"
using namespace std;

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key>
__global__
void MHJ_count_open_addr(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *res_cnts) {
    CntType iterators[MAX_NUM_BUILDTABLES];
    CntType buc_start[MAX_NUM_BUILDTABLES];
    DataType iRes[MAX_NUM_RES_ATTRS];

    __shared__ CntType l_cnt;
    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto gtid = (CntType)(tid + blockDim.x * bid); /*will change*/
    auto gtnum = (CntType)(blockDim.x * gridDim.x);
    CntType p_cnt = 0;
    char cur_table = 0; //begin from the first attr
    auto probe_cnt = probe_table.length;

    /*init the data structures*/
    buc_start[0] = INVALID_BUC_START;
    if (0 == tid) l_cnt = 0;
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if ((0 == cur_table) && (buc_start[0] == INVALID_BUC_START)) { //get new probe item
            auto probe_iter = gtid;
            gtid += gtnum;
            if (probe_iter >= probe_cnt) break; //return
            for(auto i = 0; i < probe_table.num_attrs; i++) { //update iRes
                iRes[probe_table.attr_list[i]] = probe_table.data[i][probe_iter];
            }
            auto hash_val = murmur_hash(iRes[hash_tables[0].hash_attr], hash_tables[0].capacity);
            buc_start[0] = hash_val;
        }
        if (cur_table == num_hash_tables - 1) { //reach the last table
            for(auto j = buc_start[cur_table] & (hash_tables[cur_table].capacity - 1); ;
                j =  (j + 1) & (hash_tables[cur_table].capacity - 1)) {
                if (hash_tables[cur_table].hash_keys[j] == EMPTY_HASH_SLOT) break;
                bool is_chosen = true;
                if (!single_join_key) {
                    for(auto a = 0; a < s_hash_table_num_attrs[cur_table]; a++) {
                        if (s_used_for_compare[cur_table][a]) {
                            auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                            if (iRes[hash_tables[cur_table].attr_list[a]]
                                != hash_tables[cur_table].data[a][origin_idx]) {
                                is_chosen = false;
                                break;
                            }
                        }
                    }
                }
                if (((!single_join_key) && (is_chosen)) ||
                    ((single_join_key) && (iRes[hash_tables[cur_table].hash_attr] == hash_tables[cur_table].hash_keys[j]))) {
                    p_cnt++;
                }
            }
            cur_table--;
            continue;
        }
        else {
            auto found = probe_tbj_open_addr<DataType,CntType,single_join_key>(
                    iRes, hash_tables[cur_table], buc_start[cur_table],
                    s_used_for_compare[cur_table], iterators[cur_table]);
            if (!found) { //no match is found
                if (cur_table > 0)  cur_table--;  //backtrack to the last attribute
                else                buc_start[0] = INVALID_BUC_START; //finish this probe item
                continue;
            }
        }

        /*write iRes*/
        auto curIter = iterators[cur_table]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        for(auto i = 0; i < s_hash_table_num_attrs[cur_table]; i++) {
            if (!s_used_for_compare[cur_table][i]) {
                iRes[hash_tables[cur_table].attr_list[i]] = hash_tables[cur_table].data[i][idx_in_origin_table];
            }
        }

        /*update the start of next attr*/
        auto hash_val = murmur_hash(iRes[hash_tables[cur_table+1].hash_attr], hash_tables[cur_table+1].capacity);
        buc_start[cur_table+1] = hash_val;
        cur_table++; //advance to the next attr
    }
    atomicAdd(&l_cnt, p_cnt);
    __syncthreads();

    if (0 == tid) res_cnts[bid] = l_cnt;
}

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key>
__global__
void MHJ_write_open_addr(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare,
        CntType *res_cnts_scanned, DataType **res, CntType max_writable,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes) {
    CntType iterators[MAX_NUM_BUILDTABLES];
    CntType buc_start[MAX_NUM_BUILDTABLES];
    DataType iRes[MAX_NUM_RES_ATTRS];

    __shared__ CntType l_cnt;
    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto gtid = (CntType)(tid + blockDim.x * bid); /*will change*/
    auto gtnum = (CntType)(blockDim.x * gridDim.x);
    char cur_table = 0; //begin from the first attr
    auto probe_cnt = probe_table.length;

    /*init the data structures*/
    buc_start[0] = INVALID_BUC_START;
    if (0 == tid) l_cnt = res_cnts_scanned[bid];
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if ((0 == cur_table) && (buc_start[0] == INVALID_BUC_START)) { //get new probe item
            auto probe_iter = gtid;
            gtid += gtnum;
            if (probe_iter >= probe_cnt) break; //return
            for(auto i = 0; i < probe_table.num_attrs; i++) { //update iRes
                iRes[probe_table.attr_list[i]] = probe_table.data[i][probe_iter];
            }
            auto hash_val = murmur_hash(iRes[hash_tables[0].hash_attr], hash_tables[0].capacity);
            buc_start[0] = hash_val;
        }
        if (cur_table == num_hash_tables - 1) { //reach the last table
            for(auto j = buc_start[cur_table] & (hash_tables[cur_table].capacity - 1); ;
                j =  (j + 1) & (hash_tables[cur_table].capacity - 1)) {
                if (hash_tables[cur_table].hash_keys[j] == EMPTY_HASH_SLOT) break;
                bool is_chosen = true;
                if (!single_join_key) {
                    for(auto a = 0; a < s_hash_table_num_attrs[cur_table]; a++) {
                        if (s_used_for_compare[cur_table][a]) {
                            auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                            if (iRes[hash_tables[cur_table].attr_list[a]]
                                != hash_tables[cur_table].data[a][origin_idx]) {
                                is_chosen = false;
                                break;
                            }
                        }
                    }
                }
                if (((!single_join_key) && (is_chosen)) ||
                    ((single_join_key) && (iRes[hash_tables[cur_table].hash_attr] == hash_tables[cur_table].hash_keys[j]))) {
                    CntType writePos = atomicAdd(&l_cnt, 1) % max_writable;
                    auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                    #pragma unroll
                    for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                        res[attr_idxes_in_iRes[p]][writePos] = iRes[attr_idxes_in_iRes[p]];
                    for(auto p = 0; p < s_hash_table_num_attrs[cur_table]; p++)
                        if (!s_used_for_compare[cur_table][p]) //this attr only appears in the last ht
                            res[hash_tables[cur_table].attr_list[p]][writePos] = hash_tables[cur_table].data[p][origin_idx];
                }
            }
            cur_table--;
            continue;
        }
        else {
            auto found = probe_tbj_open_addr<DataType,CntType,single_join_key>(
                    iRes, hash_tables[cur_table], buc_start[cur_table],
                    s_used_for_compare[cur_table], iterators[cur_table]);
            if (!found) { //no match is found
                if (cur_table > 0)  cur_table--;  //backtrack to the last attribute
                else                buc_start[0] = INVALID_BUC_START; //finish this probe item
                continue;
            }
        }

        /*write iRes*/
        auto curIter = iterators[cur_table]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        for(auto i = 0; i < s_hash_table_num_attrs[cur_table]; i++) {
            if (!s_used_for_compare[cur_table][i]) {
                iRes[hash_tables[cur_table].attr_list[i]] = hash_tables[cur_table].data[i][idx_in_origin_table];
            }
        }

        /*update the start of next attr*/
        auto hash_val = murmur_hash(iRes[hash_tables[cur_table+1].hash_attr], hash_tables[cur_table+1].capacity);
        buc_start[cur_table+1] = hash_val;
        cur_table++; //advance to the next attr
    }
}

/*----------------- host code -------------------*/

template<typename DataType, typename CntType, bool single_join_key>
CntType MHJ_open_addr(const Relation<DataType, CntType> probe_table,
                      HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
                      bool **used_for_compare, AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
                      DataType **&res, int num_res_attrs, CntType max_writable,
                      CUDAMemStat *memstat, CUDATimeStat *timing, cudaStream_t stream = 0) {
    log_trace("Function: %s", __FUNCTION__);

    int block_size = BLOCK_SIZE;
    int grid_size = (probe_table.length + block_size - 1)/block_size;
    log_info("grisSize = %d, block_size = %d.", grid_size, block_size);

    /*record the matching count of each probe item*/
    CntType *res_cnts = nullptr;
    CUDA_MALLOC(&res_cnts, sizeof(CntType)*grid_size, memstat, stream);

    /*1.count*/
    int accBlocksPerSM;
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, MHJ_count_open_addr<DataType, CntType,single_join_key>, block_size, 0));
    log_info("Kernel: MHJ_count_open_addr, occupancy: %d/%d.",accBlocksPerSM, 2048/block_size);

    timing->reset();
    execKernel((MHJ_count_open_addr<DataType,CntType,single_join_key>), grid_size, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, res_cnts);
    log_info("Probe-count time: %.2f ms.", timing->elapsed());

    /*2.global scan & output page generation*/
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    auto total_cnt = res_cnts[grid_size-1];
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, res_cnts, res_cnts, grid_size, stream);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat, stream);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, res_cnts, res_cnts, grid_size, stream);
    cudaStreamSynchronize(stream);
    total_cnt += res_cnts[grid_size-1];
    log_info("Output count: %llu", total_cnt);

    CntType final_num_output;
    if (total_cnt > max_writable) {
        log_warn("Output exceeded the max limit, will write circle");
        final_num_output = max_writable;
    }
    else final_num_output = total_cnt;

    CUDA_FREE(d_temp_storage, memstat);

    /*3.materialization*/
    /*allocate res if it is empty (when ooc is disabled)*/
    if (res == nullptr) {
        log_info("Allocate space for res");
        CUDA_MALLOC(&res, sizeof(DataType*)*num_res_attrs, memstat);
        for(auto i = 0; i < num_res_attrs; i++)
            CUDA_MALLOC(&res[i], sizeof(DataType)*final_num_output, memstat);
    }

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, MHJ_write_open_addr<DataType,CntType,single_join_key>, block_size, 0));
    log_info("Kernel: MHJ_write_open_addr, occupancy: %d/%d.",accBlocksPerSM, 2048/block_size);

    timing->reset();
    execKernel((MHJ_write_open_addr<DataType,CntType,single_join_key>), grid_size, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, res_cnts, res, max_writable, attr_idxes_in_iRes, num_attr_idxes_in_iRes);
    log_info("Probe-write time: %.2f ms.", timing->elapsed());

    return total_cnt;
}