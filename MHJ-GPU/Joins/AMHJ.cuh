//
// Created by Bryan on 16/8/2019.
//

#pragma once

#include "cuda/primitives.cuh"
#include "timer.h"
#include "../types.h"
#include "../common_kernels.cuh"
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups;

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_count(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec, CntType *res_cnts) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ CntType auc[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType l_cnt;

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    __shared__ AttrType hash_attrs[MAX_NUM_BUILDTABLES];
    __shared__ CntType capacities[MAX_NUM_BUILDTABLES];

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType gwarpId = (tid + blockDim.x * blockIdx.x) >> WARP_BITS;
    CntType gwarpNum = (blockDim.x * gridDim.x) >> WARP_BITS;
    CntType p_cnt = 0;
    char cur_table = 0; //begin from the first attr
    auto probe_cnt = probe_table.length;
    CntType probe_iter;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) msIdx[lwarpId][lane] = 0;
    if (0 == lane) { //to ensure 1st probe item can be fetched
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0;
    }
    if (0 == tid) l_cnt = 0;
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    if (tid < num_hash_tables) {
        hash_attrs[tid] = hash_tables[tid].hash_attr;
        capacities[tid] = bucketVec[tid];
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((0 == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                probe_iter = gwarpId;
                gwarpId += gwarpNum;
                if (probe_iter >= probe_cnt) goto LOOP_END; //flush and return
                if (lane < probe_table.num_attrs) //init iRes with the probe item
                    iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iter];
                __syncwarp();

                if (0 == lane) {
                    auto hash_val = iRes[lwarpId][hash_attrs[0]] & (capacities[0] - 1); //todo: opt bucketVec
                    buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val]; //update buc_start and buc_end
                    buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                }
                __syncwarp();
            }
            if (cur_table == num_hash_tables - 1) { //reach the last table
                for(auto j = buc_start[lwarpId][cur_table] + lane; j < buc_end[lwarpId][cur_table]; j += WARP_SIZE) {
                    bool is_chosen = true;
                    if (!single_join_key) {
                        for(auto a = 0; a < s_hash_table_num_attrs[cur_table]; a++) {
                            if (s_used_for_compare[cur_table][a]) {
                                auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                                if (iRes[lwarpId][hash_tables[cur_table].attr_list[a]]
                                    != hash_tables[cur_table].data[a][origin_idx]) {
                                    is_chosen = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (((!single_join_key) && (is_chosen)) ||
                        ((single_join_key) && (iRes[lwarpId][hash_attrs[cur_table]] == hash_tables[cur_table].hash_keys[j]))) {
                        p_cnt++;
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                auto found = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc[lwarpId], lane);

                if (!found) { //no match is found
                    if (cur_table > 0) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][0] = buc_end[lwarpId][0]; //finish this probe item
                    __syncwarp();
                    continue;
                }
            }
        }
        else if (0 == lane) msIdx[lwarpId][cur_table]--;
        __syncwarp();

        /*write iRes*/
        auto cur_ms = msIdx[lwarpId][cur_table];
        auto curIter = iterators[lwarpId][cur_table][cur_ms]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        if((lane < s_hash_table_num_attrs[cur_table]) && (!s_used_for_compare[cur_table][lane]))
            iRes[lwarpId][hash_tables[cur_table].attr_list[lane]] = hash_tables[cur_table].data[lane][idx_in_origin_table];
        __syncwarp();

        if (0 == lane) { //update the start and end of next attr
            auto hash_val = iRes[lwarpId][hash_attrs[cur_table+1]] & (capacities[cur_table+1] - 1);
            buc_start[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val];
            buc_end[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val+1];
            msIdx[lwarpId][cur_table+1] = 0; //init the msIdx for the next attr
        }
        __syncwarp();
        cur_table++; //advance to the next attr
    }
    LOOP_END:
    __syncwarp();

    WARP_REDUCE(p_cnt);
    if (lane == 0) atomicAdd(&l_cnt, p_cnt);
    __syncthreads();

    if (0 == tid) res_cnts[blockIdx.x] = l_cnt;
}

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *res_cnts_scanned, DataType **res, CntType max_writable,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ CntType auc[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType l_cnt;

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType gwarpId = (tid + blockDim.x * blockIdx.x) >> WARP_BITS;
    CntType gwarpNum = (blockDim.x * gridDim.x) >> WARP_BITS;
    char cur_table = 0; //begin from the first attr
    auto probe_cnt = probe_table.length;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) msIdx[lwarpId][lane] = 0;
    if (0 == lane) { //to ensure 1st probe item can be fetched
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0;
    }
    if (0 == tid) l_cnt = res_cnts_scanned[blockIdx.x];
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((0 == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                auto probe_iter = gwarpId;
                gwarpId += gwarpNum;
                if (probe_iter >= probe_cnt) return; //return

                if (lane < probe_table.num_attrs) //init iRes with the probe item
                    iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iter];
                __syncwarp();

                if (0 == lane) {
                    auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1); //todo: opt bucketVec
                    buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val]; //update buc_start and buc_end
                    buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                }
                __syncwarp();
            }
            if (cur_table == num_hash_tables - 1) { //reach the last table
                for(auto j = buc_start[lwarpId][cur_table] + lane; j < buc_end[lwarpId][cur_table]; j += WARP_SIZE) {
                    bool is_chosen = true;
                    if (!single_join_key) {
                        for(auto a = 0; a < s_hash_table_num_attrs[cur_table]; a++) {
                            if (s_used_for_compare[cur_table][a]) {
                                auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                                if (iRes[lwarpId][hash_tables[cur_table].attr_list[a]]
                                    != hash_tables[cur_table].data[a][origin_idx]) {
                                    is_chosen = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (((!single_join_key) && (is_chosen)) ||
                        ((single_join_key) && (iRes[lwarpId][hash_tables[cur_table].hash_attr] == hash_tables[cur_table].hash_keys[j]))) {
                        CntType writePos = atomicAdd(&l_cnt, 1) % max_writable;
                        auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                        #pragma unroll
                        for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                            res[attr_idxes_in_iRes[p]][writePos] = iRes[lwarpId][attr_idxes_in_iRes[p]];
                        for(auto p = 0; p < s_hash_table_num_attrs[cur_table]; p++) { //this attr only appears in the last ht
                            if (!s_used_for_compare[cur_table][p]) {
                                auto d = hash_tables[cur_table].data[p][origin_idx];
                                res[hash_tables[cur_table].attr_list[p]][writePos] = d;
                            }
                        }
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                auto found = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc[lwarpId], lane);
                if (!found) { //no match is found
                    if (cur_table > 0) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][0] = buc_end[lwarpId][0]; //finish this probe item
                    __syncwarp();
                    continue;
                }
            }
        }
        else if (0 == lane) msIdx[lwarpId][cur_table]--;
        __syncwarp();

        /*write iRes*/
        auto cur_ms = msIdx[lwarpId][cur_table];
        auto curIter = iterators[lwarpId][cur_table][cur_ms]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        if((lane < s_hash_table_num_attrs[cur_table]) && (!s_used_for_compare[cur_table][lane]))
            iRes[lwarpId][hash_tables[cur_table].attr_list[lane]] = hash_tables[cur_table].data[lane][idx_in_origin_table];
        __syncwarp();

        if (0 == lane) { //update the start and end of next attr
            auto hash_val = iRes[lwarpId][hash_tables[cur_table+1].hash_attr] & (bucketVec[cur_table+1] - 1);
            buc_start[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val];
            buc_end[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val+1];
            msIdx[lwarpId][cur_table+1] = 0; //init the msIdx for the next attr
        }
        __syncwarp();
        cur_table++; //advance to the next attr
    }
}

/*----------------- host code -------------------*/

template<typename DataType, typename CntType, bool single_join_key>
class AMHJ {
    int block_size;
    int grid_size;
    CntType *histograms;
    int num_res_attrs;
    CUDAMemStat *memstat;
    CUDATimeStat *timing;
public:
    AMHJ(int num_res_attrs, CntType max_probe_length, CUDAMemStat *memstat, CUDATimeStat *timing): num_res_attrs(num_res_attrs), memstat(memstat), timing(timing) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        auto maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        block_size = BLOCK_SIZE;
        grid_size = (max_probe_length + block_size - 1)/block_size;
        log_info("grisSize = %d, block_size = %d.", grid_size, block_size);

        CUDA_MALLOC(&histograms, sizeof(CntType)*grid_size, memstat); //init histograms

        int accBlocksPerSM;
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM,
                                                                      AMHJ_count<DataType, CntType, single_join_key>,
                                                                      block_size, 0));
        log_info("Kernel: AMHJ_count, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);

        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM,
                                                                      AMHJ_write<DataType, CntType, single_join_key>,
                                                                      block_size, 0));
        log_info("Kernel: AMHJ_write, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
    }
    CntType AMHJ_evaluate(const Relation<DataType, CntType> probe_table,
                          HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
                          bool **used_for_compare, CntType *bucketVec,
                          AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
                          DataType **&res, CntType max_writable,
                          cudaStream_t stream = 0) {
        log_trace("Function: %s", __FUNCTION__);
        uint32_t gpu_time_stamp;

        /*1.count*/
        if(0 == stream) {
            gpu_time_stamp = timing->get_idx();
            execKernel((AMHJ_count<DataType,CntType,single_join_key>), grid_size, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, histograms);
            log_info("Probe-count time: %.2f ms.", timing->diff_time(gpu_time_stamp));
        }
        else { //pipelining
            AMHJ_count<DataType, CntType, single_join_key> <<<grid_size,block_size,0,stream>>>
                    (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, histograms);
            cudaStreamSynchronize(stream);
        }

        /*2.global scan & output page generation*/
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        auto total_cnt = histograms[grid_size-1]; //todo: is it because the CPU read causes gap in pipeline?
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                      histograms, histograms, grid_size, stream);
        CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat, stream);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                      histograms, histograms, grid_size, stream);
        cudaStreamSynchronize(stream);
        total_cnt += histograms[grid_size-1];
        log_info("Output count: %llu", total_cnt);

        CntType final_num_output;
        if (total_cnt > max_writable) {
            log_warn("Output exceeded the max limit, will write circle");
            final_num_output = max_writable;
        }
        else final_num_output = total_cnt;

//    CUDA_FREE(d_temp_storage, memstat);

        /*3.materialization*/
        /*allocate res if it is empty (when ooc is disabled)*/
        if (res == nullptr) {
            log_info("Allocate space for res");
            CUDA_MALLOC(&res, sizeof(DataType*)*num_res_attrs, memstat, stream);
            for(auto i = 0; i < num_res_attrs; i++)
                CUDA_MALLOC(&res[i], sizeof(DataType)*final_num_output, memstat, stream);
        }

        if (0 == stream) {
            gpu_time_stamp = timing->get_idx();
            execKernel((AMHJ_write<DataType,CntType,single_join_key>), grid_size, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, histograms, res, max_writable, attr_idxes_in_iRes, num_attr_idxes_in_iRes);
            log_info("Probe-write time: %.2f ms.", timing->diff_time(gpu_time_stamp));
        }
        else {
            AMHJ_write<DataType, CntType, single_join_key> <<<grid_size,block_size,0,stream>>>(probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, histograms, res, max_writable, attr_idxes_in_iRes, num_attr_idxes_in_iRes);
            cudaStreamSynchronize(stream);
        }
        return total_cnt;
    }
};

