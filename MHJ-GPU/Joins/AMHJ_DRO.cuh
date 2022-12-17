//
// Created by Bryan on 20/9/2019.
//
#pragma once

#include "timer.h"
#include "../Indexing/radix_partitioning.cuh"
#include "../common_kernels.cuh"
#include "../types.h"

#include <cooperative_groups.h>
#include <set>
using namespace std;
using namespace cooperative_groups;

template<typename DataType, typename CntType, bool single_join_key>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_DRO_sampling(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *sample_list, CntType *res_cnts) {
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
    CntType p_cnt = 0;
    char cur_table = 0; //begin from the first attr
    auto probe_iter = sample_list[gwarpId];
    bool finish = false;

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
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((0 == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                if (finish) goto LOOP_END; //only process a single probe item
                finish = true;

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
            auto hash_val = iRes[lwarpId][hash_tables[cur_table+1].hash_attr] & (bucketVec[cur_table+1] - 1);
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
void AMHJ_DRO_main_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *g_mains, CntType *g_residuals, DataType **res,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        CntType num_res_allocated, CntType *g_chunk_head,
        CntType *bp_head, CntType *bp_iters, DataType *bp_iRes, int chunk_size,
        CntType *probe_iter, bool *count_only, CntType *min_count_idx, CntType max_writable) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ CntType auc[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx_max[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType s_probe_iter[WARPS_PER_BLOCK];

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];
    __shared__ int lock;

    __shared__ CntType s_mains, s_residuals;   //number of main and residuals counted by this block
    __shared__ bool warp_full[WARPS_PER_BLOCK]; //record whether the chunks for the warps are full
    __shared__ CntType chunk_start, chunk_end; //recording the global start and end write pos for each block's chunk

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    auto bid = blockIdx.x;
    char cur_table = 0; //begin from the first attr
    auto probe_cnt = probe_table.length;
    CntType l_wrt_pos; //write position

    /*init the data structures*/
    if (1 == tid) {
        s_mains = 0;
        s_residuals = 0;
    }
    if (tid < WARPS_PER_BLOCK) warp_full[tid] = false;
    if (lane < MAX_NUM_BUILDTABLES)
        msIdx[lwarpId][lane] = msIdx_max[lwarpId][lane] = 0;
    if (0 == lane) { //to ensure 1st probe item can be fetched
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0;
    }
    if (0 == tid) {
        chunk_start = bid * chunk_size;
        chunk_end = (bid+1)*chunk_size;
        lock = 0;
    }
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if (msIdx_max[lwarpId][cur_table] == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((0 == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                SIN_L(s_probe_iter[lwarpId] = atomicAdd(probe_iter, 1)); //atomic fetch is faster than skip fetch
                if (s_probe_iter[lwarpId] >= probe_cnt) goto LOOP_END; //exit

                SIN_L(if(warp_full[lwarpId]) count_only[s_probe_iter[lwarpId]] = true);//record count-only probe items
                if (lane < probe_table.num_attrs) //init iRes with the probe item
                    iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][s_probe_iter[lwarpId]];
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
                    auto active_in_for_loop = coalesced_threads();
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
                        auto active_ths = coalesced_threads();
                        auto active_rank = active_ths.thread_rank();
                        auto active_size = active_ths.size();

                        if (warp_full[lwarpId]) { //no chunks available, just count
                            if (0 == active_rank) atomicAdd(&s_residuals, active_size);
                        }
                        else {
                            if (0 == active_rank) { //critical section
                                while (atomicCAS(&lock, 0, 1)); //get lock: 0->1
                                if (chunk_start + active_size < chunk_end) { //have slots
                                    l_wrt_pos = chunk_start;
                                    chunk_start += active_size;
                                }
                                else { //no slot available
                                    CntType new_chunk_start;
                                    new_chunk_start = atomicAdd(g_chunk_head, (CntType)chunk_size);
                                    if (new_chunk_start >= num_res_allocated) { //no available chunks
                                        warp_full[lwarpId] = true;
                                    }
                                    else { //has available chunk
                                        l_wrt_pos = new_chunk_start;
                                        chunk_start = new_chunk_start + active_size;
                                        chunk_end = new_chunk_start + chunk_size;
                                    }
                                }
                                lock = 0; //release lock: 1->0
                            }
                            active_ths.sync();

                            if (warp_full[lwarpId]) { //write the bp info
                                CntType cur_head;
                                if (0 == active_rank) {
                                    atomicAdd(&s_residuals, active_size);
                                    cur_head = atomicAdd(bp_head, 1);
                                }
                                cur_head = active_ths.shfl(cur_head, 0);
                                auto bp_pos = cur_head * num_attr_idxes_in_iRes;
                                for(auto xp = active_rank; xp < num_attr_idxes_in_iRes; xp+=active_size)
                                    bp_iRes[bp_pos+xp] = iRes[lwarpId][xp]; //write bp_iRes
                                bp_pos = cur_head * (1+num_hash_tables);
                                for(auto xp = active_rank; xp < num_hash_tables-1; xp+=active_size)
                                    bp_iters[bp_pos+xp+1] = iterators[lwarpId][xp][msIdx[lwarpId][xp]];
                                if (0 == active_rank) {
                                    bp_iters[bp_pos] = s_probe_iter[lwarpId];
                                    bp_iters[bp_pos+num_hash_tables] = j;
                                    atomicMin(min_count_idx, s_probe_iter[lwarpId]);
                                }
                            }
                            else { //write results
                                auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                                l_wrt_pos = (active_ths.shfl(l_wrt_pos, 0) + active_rank) % max_writable;

                                #pragma unroll
                                for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                                    res[attr_idxes_in_iRes[p]][l_wrt_pos] = iRes[lwarpId][attr_idxes_in_iRes[p]];
                                for(auto p = 0; p < s_hash_table_num_attrs[cur_table]; p++) {
                                    if (!s_used_for_compare[cur_table][p]) { //this attr only appears in the last ht
                                        auto d = hash_tables[cur_table].data[p][origin_idx];
                                        res[hash_tables[cur_table].attr_list[p]][l_wrt_pos] = d;
                                    }
                                }
                                active_ths.sync();
                                if (0 == active_rank) atomicAdd(&s_mains, active_size); //update s_mains
                            }
                        }
                        active_ths.sync();
                    }
                    active_in_for_loop.sync();
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                auto found = probe_sbp<DataType, CntType, single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table], s_used_for_compare[cur_table],
                        msIdx[lwarpId][cur_table], msIdx_max[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc[lwarpId], lane);
                if (!found) { //no match is found
                    if (cur_table > 0) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][0] = buc_end[lwarpId][0]; //finish this probe item
                    __syncwarp();
                    continue;
                }
            }
        }
        else if (0 == lane) msIdx[lwarpId][cur_table]++; //increasing order
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
            msIdx[lwarpId][cur_table+1] = msIdx_max[lwarpId][cur_table+1] = 0; //init the msIdx for the next attr
        }
        __syncwarp();
        cur_table++; //advance to the next attr
    }
    LOOP_END:
    __syncthreads();
    if (0 == tid) {
        atomicAdd(g_residuals,s_residuals);
        atomicAdd(g_mains, s_mains);
    }
}

/*
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * */
template<typename DataType, typename CntType, bool single_join_key>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_DRO_residual_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *res_iter, DataType **res,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        CntType *bp_head, CntType *bp_iters, DataType *bp_iRes,
        CntType *probe_iter, bool *count_only, CntType max_writable) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ CntType auc[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];
    __shared__ CntType s_probe_iter[WARPS_PER_BLOCK];

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType gwarpId = (tid + blockDim.x * blockIdx.x) >> WARP_BITS;
    char cur_table = 0;
    auto probe_cnt = probe_table.length;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) msIdx[lwarpId][lane] = 0;
    if (0 == lane) { //to ensure 1st probe item can be fetched
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0;
    }
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    if (gwarpId < bp_head[0]) { //restore the scene
        auto bp_pos = gwarpId * (1+num_hash_tables);
        for(auto xp = lane; xp < num_hash_tables-1; xp += WARP_SIZE) {
            msIdx[lwarpId][xp] = 0;
            iterators[lwarpId][xp][0] = bp_iters[bp_pos+1+xp];//restore iterators
        }
        if (0 == lane) s_probe_iter[lwarpId] = bp_iters[bp_pos];
        bp_pos = gwarpId * num_attr_idxes_in_iRes;
        for(auto xp = lane; xp < num_attr_idxes_in_iRes; xp += WARP_SIZE)
            iRes[lwarpId][xp] = bp_iRes[bp_pos+xp];//restore iRes
        for(auto xp = lane; xp < num_hash_tables-1; xp += WARP_SIZE) {
            buc_start[lwarpId][xp] = iterators[lwarpId][xp][0] + 1; //restore buc_start
        }
        if (0 == lane) buc_start[lwarpId][num_hash_tables-1] = bp_iters[bp_pos+num_hash_tables];
        for(auto xp = lane; xp < num_hash_tables; xp += WARP_SIZE) {
            auto hash_val = iRes[lwarpId][hash_tables[xp].hash_attr] & (bucketVec[xp] - 1);
            buc_end[lwarpId][xp] = hash_tables[xp].buc_ptrs[hash_val+1];//restore buc_end
        }
        cur_table = (char)num_hash_tables - 1;
    }

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((0 == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                SIN_L(s_probe_iter[lwarpId] = atomicAdd(probe_iter, 1));
                if (s_probe_iter[lwarpId] >= probe_cnt) return; //return
                if (!count_only[s_probe_iter[lwarpId]]) continue;//the item has been processed

                if (lane < probe_table.num_attrs) //init iRes with the probe item
                    iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][s_probe_iter[lwarpId]];
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
                        CntType writePos = (atomicAdd(res_iter, 1)) % max_writable;
                        auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                        #pragma unroll
                        for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                            res[attr_idxes_in_iRes[p]][writePos] = iRes[lwarpId][attr_idxes_in_iRes[p]];
                        for(auto p = 0; p < s_hash_table_num_attrs[cur_table]; p++)
                            if (!s_used_for_compare[cur_table][p]) //this attr only appears in the last ht
                                res[hash_tables[cur_table].attr_list[p]][writePos] =
                                        hash_tables[cur_table].data[p][origin_idx];
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

template<typename DataType, typename CntType, bool single_join_key>
class AMHJ_DRO {
    int block_size;
    int grid_size_sampling;
    int grid_size_main_writing;
    int grid_size_residual_writing;
    uint32_t num_persistent_warps_sampling;
    uint32_t num_persistent_warps_main_writing;
    int chunk_size;
    int num_res_attrs;
    CntType min_probe_length;

    /*sampling*/
    CntType *samples;
    CntType *sample_res_cnts;

    /*bp information*/
    CntType *bp_iters;
    CntType *bp_head;
    DataType *bp_iRes;

    /*iterators*/
    CntType *probe_iter;
    CntType *min_count_idx;
    CntType *chunk_head;
    CntType *g_mains;
    CntType *g_residuals;
    CntType *residual_cnt;

    bool *count_only;
    CUDAMemStat *memstat;
    CUDATimeStat *timing;
public:
    AMHJ_DRO(int num_hash_tables, int num_res_attrs, CntType min_probe_length, CntType max_probe_length,
            CUDAMemStat *memstat, CUDATimeStat *timing): num_res_attrs(num_res_attrs), min_probe_length(min_probe_length), memstat(memstat), timing(timing) {
        /*setting for persistent warps*/
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        auto maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        auto numSM = prop.multiProcessorCount;
        block_size = BLOCK_SIZE;
        int accBlocksPerSM;
        chunk_size = 256;
        log_info("chunk_size=%llu", chunk_size);

        /*profile sampling kernel*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_DRO_sampling<DataType, CntType, single_join_key>, block_size, 0));
        log_info("Kernel: AMHJ_DRO_sampling, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_sampling = numSM * accBlocksPerSM;
        num_persistent_warps_sampling = block_size * grid_size_sampling / WARP_SIZE;
        log_info("Sampling: gris_size=%llu, block_size=%llu, persistent warps=%llu", grid_size_sampling, block_size, num_persistent_warps_sampling);

        /*profile main writing kernel*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_DRO_main_write<DataType, CntType, single_join_key>, block_size, 0));
        log_info("Kernel: AMHJ_DRO_main_write, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_main_writing = numSM * accBlocksPerSM;
        grid_size_residual_writing = grid_size_main_writing;
        num_persistent_warps_main_writing = block_size * grid_size_main_writing / WARP_SIZE;
        log_info("Main-writing: gris_size=%llu, block_size=%llu, persistent warps=%llu", grid_size_main_writing, block_size, num_persistent_warps_main_writing);

        /*profile residual writing kernel*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_DRO_residual_write<DataType, CntType, single_join_key>, block_size, 0));
        log_info("Kernel: AMHJ_DRO_residual_write, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);

        /*init sampling related*/
        CUDA_MALLOC(&samples, sizeof(CntType)*num_persistent_warps_sampling, memstat);
        CUDA_MALLOC(&sample_res_cnts, sizeof(CntType)*grid_size_sampling, memstat);

        /*init the bp information*/
        CUDA_MALLOC(&bp_iters, sizeof(CntType)*num_persistent_warps_main_writing*(1+num_hash_tables), memstat);
        CUDA_MALLOC(&bp_iRes, sizeof(DataType)*num_persistent_warps_main_writing*num_res_attrs, memstat);
        CUDA_MALLOC(&bp_head, sizeof(CntType), memstat);
        checkCudaErrors(cudaMemset(bp_head, 0, sizeof(CntType)));

        /*init the iterators*/
        CUDA_MALLOC(&probe_iter, sizeof(CntType), memstat);
        CUDA_MALLOC(&min_count_idx, sizeof(CntType), memstat);
        CUDA_MALLOC(&g_mains, sizeof(CntType), memstat);
        CUDA_MALLOC(&g_residuals, sizeof(CntType), memstat);
        CUDA_MALLOC(&chunk_head, sizeof(CntType), memstat);
        CUDA_MALLOC(&residual_cnt, sizeof(CntType), memstat);
        CUDA_MALLOC(&count_only, sizeof(bool)*max_probe_length, memstat);
        checkCudaErrors(cudaMemset(probe_iter, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemset(g_mains, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemset(g_residuals, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemset(residual_cnt, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemset(count_only, false, sizeof(bool)*max_probe_length));
        chunk_head[0] = grid_size_main_writing*chunk_size; //pointed to next usable chunk
        min_count_idx[0] = INT32_MAX;

        /*1.Sampling, do only once*/
        CntType written_sample_cnt = 0;
        set<CntType> chosen_samples;
        while (written_sample_cnt < num_persistent_warps_sampling) { //compute sample set
            auto cur = rand() % min_probe_length;
            if (chosen_samples.find(cur) == chosen_samples.end()) {
                chosen_samples.insert(cur);
                samples[written_sample_cnt++] = cur;
            }
        }
        checkCudaErrors(cudaMemPrefetchAsync(samples, sizeof(CntType)*num_persistent_warps_sampling, DEVICE_ID));
    }

    CntType AMHJ_DRO_evaluate(const Relation<DataType, CntType> probe_table,
                              HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
                              bool **used_for_compare, CntType *bucketVec,
                              AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
                              DataType **&res, CntType max_writable,
                              cudaStream_t stream = 0) {
        log_trace("Function: %s", __FUNCTION__);
        uint32_t gpu_time_stamp;
        CntType min_count_idx_fixed = 0;

        if (0 == stream) {
            gpu_time_stamp = timing->get_idx();
            execKernel((AMHJ_DRO_sampling<DataType,CntType,single_join_key>), grid_size_sampling, block_size, timing, true, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, samples, sample_res_cnts);
            log_info("Sampling time: %.2f ms.", timing->diff_time(gpu_time_stamp));
        }
        else {
            AMHJ_DRO_sampling<DataType,CntType,single_join_key><<<grid_size_sampling,block_size,0,stream>>>(probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, samples, sample_res_cnts);
            cudaStreamSynchronize(stream);
        }

        /*average estimation*/
        float ave = 0;
        for (auto i = 0; i < grid_size_sampling; i++)
            ave += sample_res_cnts[i];
        ave /= num_persistent_warps_sampling;
        log_debug("Output ave: %.2f output tuples/probe item, total estimated: %.0f.", ave, ave * probe_table.length);
        ave *= probe_table.length;
        CntType estimated_num_res = ((CntType)ave + chunk_size - 1) / chunk_size * chunk_size; //a multiple of chunk_size
        log_debug("Final estimated #res(adjusted): %llu (%llu chunks)", estimated_num_res, estimated_num_res/chunk_size);

#ifdef FREE_DATA
        CUDA_FREE(samples, memstat);
        CUDA_FREE(sample_res_cnts, memstat);
#endif

        CntType final_num_output;
        if (estimated_num_res > max_writable) {
            log_warn("Output exceeded the max limit, will write circle");
            final_num_output = max_writable;
        }
        else final_num_output = estimated_num_res;

        /*2.Main writing*/
        if (res == nullptr) {
            log_info("Allocate space for res");
            CUDA_MALLOC(&res, sizeof(DataType*)*num_res_attrs, memstat, stream);
            for(auto i = 0; i < num_res_attrs; i++)
                CUDA_MALLOC(&res[i], sizeof(DataType)*final_num_output, memstat, stream);
        }

        if (0 == stream) {
            gpu_time_stamp = timing->get_idx();
            execKernel((AMHJ_DRO_main_write<DataType,CntType,single_join_key>), grid_size_main_writing, block_size, timing, true, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, g_mains, g_residuals, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, estimated_num_res, chunk_head, bp_head, bp_iters, bp_iRes, chunk_size, probe_iter, count_only, min_count_idx, max_writable);
            log_info("Main-write time: %.2f ms.", timing->diff_time(gpu_time_stamp));
        }
        else {
            AMHJ_DRO_main_write<DataType,CntType,single_join_key><<<grid_size_main_writing,block_size,0,stream>>>(probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, g_mains, g_residuals, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, estimated_num_res, chunk_head, bp_head, bp_iters, bp_iRes, chunk_size, probe_iter, count_only, min_count_idx, max_writable);
            cudaStreamSynchronize(stream);
        }
        min_count_idx_fixed = min_count_idx[0];

        /*3.Redidual writing*/
        auto mains_res_cnt = g_mains[0];
        auto residual_res_cnt = g_residuals[0];
        log_info("mains = %llu, residuals = %llu", mains_res_cnt, residual_res_cnt);
        if (residual_res_cnt > 0) { //todo: currently the main and residual space are together
            if (0 == stream) {
                gpu_time_stamp = timing->get_idx();
                execKernel((AMHJ_DRO_residual_write<DataType,CntType,single_join_key>), grid_size_residual_writing, block_size, timing, true, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, residual_cnt, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, bp_head, bp_iters, bp_iRes, min_count_idx, count_only, max_writable);
                log_info("Residual-write time: %.2f ms.", timing->diff_time(gpu_time_stamp));
            }
            else {
                AMHJ_DRO_residual_write<DataType,CntType,single_join_key><<<grid_size_residual_writing,block_size,0,stream>>>(probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, residual_cnt, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, bp_head, bp_iters, bp_iRes, min_count_idx, count_only, max_writable);
                cudaStreamSynchronize(stream);
            }
        }

        /*reset for the next iteration*/
        checkCudaErrors(cudaMemsetAsync(bp_head, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(g_mains, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(g_residuals, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(residual_cnt, 0, sizeof(CntType), stream));
        if (min_count_idx_fixed != INT32_MAX) { //=INT32_MAX means no residual write occurs, no need to memset
            checkCudaErrors(cudaMemsetAsync(count_only+min_count_idx_fixed, false,
                                            sizeof(bool)*(probe_table.length-min_count_idx_fixed), stream));
        }
        min_count_idx[0] = INT32_MAX;
        chunk_head[0] = grid_size_main_writing*chunk_size;
        cudaStreamSynchronize(stream);

        /*check main res*/
//    CntType main_res_cnt_check = 0;
//    for(auto i = 0; i < estimated_num_res; i++)
//        if (main_res[0][i] != -1) main_res_cnt_check++;
//    cout<<"main_res_check="<<main_res_cnt_check<<endl;

        return mains_res_cnt + residual_res_cnt;
    }
};



