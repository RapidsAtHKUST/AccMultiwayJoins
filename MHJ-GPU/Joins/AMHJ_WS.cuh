//
// Created by Bryan on 16/8/2019.
//
/*
 * WWJ with Work Sharing (WS)
 *
 * */
#pragma once

#include "../conf.h"
#include "cuda/primitives.cuh"
#include "cuda/GCQueue.cuh"
#include "timer.h"
#include "../types.h"
#include "../common_kernels.cuh"
#include "../TaskBook.cuh"
#include "../CBarrier.cuh"

#include <cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

#define BUC_SIZE_THRESHOLD   (1024)      //When tempCounter reaches this and rest bucket size is also larger than that, split the tasks
#define TEMP_CNT_THRESHOLD_WS   (16)

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_WS_count(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec, CntType *res_cnt, CntType *probe_iter,
        int num_res_attrs, GCQueue<DataType,uint32_t> *cq, TaskBook<DataType,CntType> *tb, CBarrier *br) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK];
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];

    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType probe_iterator[WARPS_PER_BLOCK];
    __shared__ CntType l_cnt;

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    __shared__ int tempCounter[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in stealing mode
    __shared__ bool triggerWS[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType p_cnt = 0, probe_cnt = probe_table.length;
    char cur_table = 0, start_table = 0; //begin from the start_table
    bool found;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        tempCounter[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
        probe_iterator[tid] = 0;
    }
    SIN_L(
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0);
    if (0 == tid) l_cnt = 0;
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) && (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item
                if (work_sharing && triggerWS[lwarpId])
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) {
                    SIN_L(probe_iterator[lwarpId] = atomicAdd(probe_iter,1));
                    if (probe_iterator[lwarpId] >= probe_cnt) {
                        if (!work_sharing) goto LOOP_END;
                        SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]);
                                      triggerWS[lwarpId] = true);
                        while (true) {
                            if (sharing[lwarpId]) break;
                            SIN_L(br->setActive(false));
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(br->setActive(true);
                                                  sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                                    if (!sharing[lwarpId]) SIN_L(br->setActive(false));
                                }
                                if (br->isTerminated()) {
                                    goto LOOP_END;//all warps reach this barrier, exit
                                }
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) {
                    start_table = cur_table = tb->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb->m_iRes[auc[lwarpId]*num_res_attrs+lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb->m_buc_ends[auc[lwarpId]];
                    }
                }
                else { //init data for original tasks
                    cur_table = start_table = 0;
                    if (lane < probe_table.num_attrs) //init iRes with the probe item
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iterator[lwarpId]];
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1); //todo: opt bucketVec
                        buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val];
                        buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                    }
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
                        p_cnt++;
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                found = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(if(work_sharing && found) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table]+1);
                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > TEMP_CNT_THRESHOLD_WS) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD)) {
                    if (0 == lane) triggerWS[lwarpId] = true;

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD;
                        auto taskId = tb->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                        cq->enqueue(taskId);
                    }
                    __syncwarp();

                    /*no need to probe the rest*/
                    SIN_L (
                            tempCounter[lwarpId][cur_table] = 0;
                            buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table]);
                }
                __syncwarp();

                if (!found) { //no match is found
                    if(0 == lane)  tempCounter[lwarpId][cur_table] = 0;
                    if (cur_table > start_table) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table]; //finish this probe item
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

        SIN_L(
            auto hash_val = iRes[lwarpId][hash_tables[cur_table+1].hash_attr] & (bucketVec[cur_table+1] - 1);
            buc_start[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val];
            buc_end[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val+1];
            msIdx[lwarpId][cur_table+1] = 0); //init the msIdx for the next attr
        cur_table++; //advance to the next attr
    }
    LOOP_END:
    __syncwarp();

    WARP_REDUCE(p_cnt);
    if (lane == 0) atomicAdd(&l_cnt, p_cnt);
    __syncthreads();

    if (0 == tid) atomicAdd(res_cnt,l_cnt);
}

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_WS_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *res_cnt, CntType *probe_iter,DataType **res, int num_res_attrs, CntType max_writable,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        GCQueue<DataType,uint32_t> *cq, TaskBook<DataType,CntType> *tb, CBarrier *br) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK];
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType probe_iterator[WARPS_PER_BLOCK];

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    __shared__ int tempCounter[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in stealing mode
    __shared__ bool triggerWS[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType probe_cnt = probe_table.length;
    char cur_table = 0, start_table = 0; //begin from the start_table
    bool found;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        tempCounter[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
        probe_iterator[tid] = 0;
    }
    SIN_L(
            buc_start[lwarpId][0] = 0;
            buc_end[lwarpId][0] = 0);
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) && (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item
                if (work_sharing && triggerWS[lwarpId])
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) {
                    SIN_L(probe_iterator[lwarpId] = atomicAdd(probe_iter,1));
                    if (probe_iterator[lwarpId] >= probe_cnt) {
                        if (!work_sharing) return;
                        SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]);triggerWS[lwarpId] = true);
                        while (true) {
                            if (sharing[lwarpId]) break;
                            SIN_L(br->setActive(false));
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(br->setActive(true);sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                                    if (!sharing[lwarpId]) SIN_L(br->setActive(false));
                                }
                                if (br->isTerminated())    return;
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) {
                    start_table = cur_table = tb->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb->m_iRes[auc[lwarpId]*num_res_attrs+lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb->m_buc_ends[auc[lwarpId]];
                    }
                }
                else { //init data for original tasks
                    cur_table = start_table = 0;
                    if (lane < probe_table.num_attrs) //init iRes with the probe item
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iterator[lwarpId]];
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1); //todo: opt bucketVec
                        buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val];
                        buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                    }
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
                        CntType writePos = atomicAdd(res_cnt, 1) % max_writable;
                        auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                        #pragma unroll
                        for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                            res[attr_idxes_in_iRes[p]][writePos] = iRes[lwarpId][attr_idxes_in_iRes[p]];
                        for(auto p = 0; p < s_hash_table_num_attrs[cur_table]; p++)
                            if (!s_used_for_compare[cur_table][p]) //this attr only appears in the last ht
                                res[hash_tables[cur_table].attr_list[p]][writePos] = hash_tables[cur_table].data[p][origin_idx];
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                found = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(if(work_sharing && found) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table]+1);
                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > TEMP_CNT_THRESHOLD_WS) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD)) {
                    if (0 == lane) triggerWS[lwarpId] = true;

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD;
                        auto taskId = tb->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                        cq->enqueue(taskId);
                    }
                    __syncwarp();

                    /*no need to probe the rest*/
                    SIN_L (
                            tempCounter[lwarpId][cur_table] = 0;
                            buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table]);
                }
                __syncwarp();

                if (!found) { //no match is found
                    if(work_sharing && (0 == lane))  tempCounter[lwarpId][cur_table] = 0;
                    if (cur_table > start_table) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table]; //finish this probe item
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

        SIN_L(
                auto hash_val = iRes[lwarpId][hash_tables[cur_table+1].hash_attr] & (bucketVec[cur_table+1] - 1);
                buc_start[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val];
                buc_end[lwarpId][cur_table+1] = hash_tables[cur_table+1].buc_ptrs[hash_val+1];
                msIdx[lwarpId][cur_table+1] = 0); //init the msIdx for the next attr
        cur_table++; //advance to the next attr
    }
}

/*----------------- host code -------------------*/

template<typename DataType, typename CntType, bool single_join_key, bool work_sharing>
class AMHJ_WS {
    GCQueue<DataType,uint32_t> *cq; //queueing data is CntType
    TaskBook<DataType,CntType> *tb;
    CBarrier *br;
    int num_res_attrs;
    CntType *res_cnt;
    CntType *probe_iter;

    int block_size;
    int grid_size_probe_count;
    int grid_size_probe_write;

    CUDAMemStat *memstat;
    CUDATimeStat *timing;
public:
    AMHJ_WS(int num_res_attrs, CUDAMemStat *memstat, CUDATimeStat *timing): num_res_attrs(num_res_attrs), memstat(memstat), timing(timing) {
        CUDA_MALLOC(&this->cq, sizeof(GCQueue<DataType,uint32_t>), this->memstat); //CntType can only be 32-bit values
        this->cq->init(750000, this->memstat);
        CUDA_MALLOC(&this->tb, sizeof(TaskBook<DataType,CntType>), this->memstat); //tb
        this->tb->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->br, sizeof(CBarrier), this->memstat); //br
        this->br->initWithWarps(0, this->memstat);

        CUDA_MALLOC(&res_cnt, sizeof(CntType), this->memstat);
        CUDA_MALLOC(&probe_iter, sizeof(CntType), this->memstat);
        checkCudaErrors(cudaMemsetAsync(res_cnt, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType)));

        block_size = BLOCK_SIZE;

        /*setting for persistent warps*/
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        auto maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        auto numSM = prop.multiProcessorCount;

        /*set the grid size according to occupancy*/
        /*probe-count*/
        int accBlocksPerSM;
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_WS_count<DataType,CntType,single_join_key,work_sharing>, block_size, 0));
        log_info("Kernel: AMHJ_WS_count, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_probe_count = numSM * accBlocksPerSM;

        /*probe-write*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_WS_write<DataType,CntType,single_join_key,work_sharing>, block_size, 0));
        grid_size_probe_write = numSM * accBlocksPerSM; //change the gridSize since the occupancy may change

        this->br->reset_with_warps(grid_size_probe_count*block_size/WARP_SIZE);
    }

    size_t required_size() { //return the size of intermediate data structure used for evaluation
        return this->cq->get_size() + this->tb->get_size() + this->br->get_size();
    }

    CntType AMHJ_WS_evaluate(const Relation<DataType, CntType> probe_table,
                             HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
                             bool **used_for_compare, CntType *bucketVec,
                             AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
                             DataType **&res, CntType max_writable,
                             cudaStream_t stream = 0) {
        log_trace("Function: %s", __FUNCTION__);

        /*1.count*/
        log_info("Probe-count: grid_size = %d, block_size = %d.", grid_size_probe_count, block_size);
        if (stream == 0) { //timing
            auto timestamp = timing->get_idx();
            execKernel((AMHJ_WS_count<DataType,CntType,single_join_key,work_sharing>), grid_size_probe_count,
                       block_size, timing, true,
                       probe_table, hash_tables, num_hash_tables, used_for_compare,
                       bucketVec, res_cnt, probe_iter, num_res_attrs, cq, tb, br);
            log_info("AMHJ_WS_count time: %.2f ms", timing->diff_time(timestamp));
        }
        else { //streaming, no timing
            AMHJ_WS_count<DataType,CntType,single_join_key,work_sharing>
                    <<<grid_size_probe_count, block_size, 0, stream>>>(
                    probe_table, hash_tables, num_hash_tables, used_for_compare,
                            bucketVec, res_cnt, probe_iter, num_res_attrs, cq, tb, br);
            cudaStreamSynchronize(stream);
        }

        CntType total_cnt = res_cnt[0];
        log_info("Output count: %u.", total_cnt);

        /*2.global scan & output page generation*/
        log_debug("GCQueue: qHead=%d, qRear=%d.", cq->qHead[0], cq->qRear[0]);
        log_debug("Num tasks: %d", tb->m_cnt[0]);

        /*reset the data structures for probe-write*/
        this->cq->reset(stream); this->tb->reset(stream);
        checkCudaErrors(cudaMemsetAsync(res_cnt, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));
        br->reset_with_warps(grid_size_probe_write*block_size/WARP_SIZE, stream);

        /*3.materialization*/
        CntType final_num_output;
        if (total_cnt > max_writable) {
            log_warn("Output exceeded the max limit, will write circle");
            final_num_output = max_writable;
        }
        else final_num_output = total_cnt;
        if (res == nullptr) {
            log_info("Allocate space for res");
            CUDA_MALLOC(&res, sizeof(DataType*)*num_res_attrs, memstat);
            for(auto i = 0; i < num_res_attrs; i++)
                CUDA_MALLOC(&res[i], sizeof(DataType)*final_num_output, memstat);
        }

        if (stream == 0) { //timing
            auto timestamp = timing->get_idx();
            execKernel((AMHJ_WS_write<DataType,CntType,single_join_key,work_sharing>), grid_size_probe_write,
                       block_size, timing, true,
                       probe_table, hash_tables, num_hash_tables, used_for_compare,
                       bucketVec, res_cnt, probe_iter, res, num_res_attrs, max_writable,
                       attr_idxes_in_iRes, num_attr_idxes_in_iRes, cq, tb, br);
            log_info("AMHJ_WS_write time: %.2f ms", timing->diff_time(timestamp));
        }
        else {
            AMHJ_WS_write<DataType,CntType,single_join_key,work_sharing>
                    <<<grid_size_probe_write, block_size, 0, stream>>>(
                    probe_table, hash_tables, num_hash_tables, used_for_compare,
                            bucketVec, res_cnt, probe_iter, res, num_res_attrs, max_writable,
                            attr_idxes_in_iRes, num_attr_idxes_in_iRes, cq, tb, br);
            cudaStreamSynchronize(stream);
        }

        /*reset for the next iteration*/
        checkCudaErrors(cudaMemsetAsync(res_cnt, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));
        this->cq->reset(stream); this->tb->reset(stream);
        this->br->reset_with_warps(grid_size_probe_count*block_size/WARP_SIZE);

        return total_cnt;
    }
};

#undef BUC_SIZE_THRESHOLD