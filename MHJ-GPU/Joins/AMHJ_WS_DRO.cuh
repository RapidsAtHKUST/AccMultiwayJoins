//
// Created by Bryan on 20/9/2019.
//
#pragma once

#include "timer.h"
#include "../Indexing/radix_partitioning.cuh"
#include "../common_kernels.cuh"
#include "../types.h"

#include "../CBarrier.cuh"
#include "../TaskBook.cuh"
#include "cuda/GCQueue.cuh"
#include "../BreakPoints.cuh"

#include <cooperative_groups.h>
#include <set>
using namespace std;
using namespace cooperative_groups;

#define BUC_SIZE_THRESHOLD_WS_DRO   (1024)
#define TEMP_CNT_THRESHOLD_WS_DRO   (4)

/*three types of main-write kernels*/
enum MainWriteType {
    MW_READ_FROM_TABLE_WRITE,  //process items in probe tables, will write results
    MW_READ_FROM_TB_WRITE,     //process items in task book (TB), will write results
    MW_READ_FROM_TB_COUNT      //process items in task book (TB), only count results
};

enum ResidualWriteType {
    RW_READ_FROM_TABLE,        //breakpoints w.r.t. probe tables
    RW_READ_FROM_TB,           //breakpoints w.r.t. task book,
    RW_READ_FROM_TB_NO_CO          //breakpoints w.r.t. task book, do not consider count_only
};

template<typename DataType, typename CntType, bool single_join_key, bool work_sharing>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_WS_DRO_sampling(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *sample_list, CntType *res_cnts,
        int num_res_attrs, GCQueue<DataType,uint32_t> *cq, TaskBook<DataType,CntType> *tb, CBarrier *br) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK];
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType l_cnt;

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];

    /*work-sharing related*/
    __shared__ int tempCounter[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in stealing mode
    __shared__ bool triggerWS[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType gwarpId = (tid + blockDim.x * blockIdx.x) >> WARP_BITS;
    CntType p_cnt = 0;
    char cur_table = 0, start_table = 0; //begin from the first attr
    auto probe_iter = sample_list[gwarpId];
    bool finish = false;
    bool found;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        tempCounter[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
    }
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
            if ((start_table == cur_table) && (buc_start[lwarpId][0] >= buc_end[lwarpId][0])) { //get new probe item
                if (work_sharing && triggerWS[lwarpId])
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) {
                    if (finish) {
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
                                if (br->isTerminated())    goto LOOP_END;//all warps reach this barrier, exit
                            }
                        }
                    }
                    finish = true; //only process a single probe item
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
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iter];
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
                auto found = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(if(work_sharing && found) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table]+1);
                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > BUC_SIZE_THRESHOLD_WS_DRO) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD_WS_DRO)) {
                    if (0 == lane) triggerWS[lwarpId] = true;

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD_WS_DRO;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD_WS_DRO * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD_WS_DRO > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD_WS_DRO;
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
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, MainWriteType wt>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_WS_DRO_main_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *g_mains, CntType *g_residuals, DataType **res,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        CntType num_res_allocated, CntType *g_chunk_head,
        CntType *bp_head, CntType *bp_iters,
        CntType *bp_buc_ends,
        DataType *bp_iRes, char *bp_start_tables,
        int chunk_size, CntType *probe_iter, bool *count_only, CntType *min_count_idx, CntType max_writable,
        int num_res_attrs,
        TaskBook<DataType,CntType> *tb_in,
        TaskBook<DataType,CntType> *tb_out_before_bp, TaskBook<DataType,CntType> *tb_out_after_bp) {
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

    __shared__ int tempCounter[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    auto bid = blockIdx.x;
    char cur_table = 0, start_table = 0;

    CntType probe_cnt;
    CntType l_wrt_pos; //write position

    if (wt == MW_READ_FROM_TABLE_WRITE) probe_cnt = probe_table.length;
    else                                probe_cnt = tb_in->m_cnt[0];

    /*init the data structures*/
    if (1 == tid) {
        s_mains = 0;
        s_residuals = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        if (wt == MW_READ_FROM_TB_COUNT)    warp_full[tid] = true;  //only count
        else                                warp_full[tid] = false; //will write
    }
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
            if ((start_table == cur_table) && (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item
                SIN_L(s_probe_iter[lwarpId] = atomicAdd(probe_iter, 1)); //atomic fetch is faster than skip fetch
                if (s_probe_iter[lwarpId] >= probe_cnt) goto LOOP_END; //todo: change to break

                if(wt != MW_READ_FROM_TB_COUNT) {
                    SIN_L(
                            if(warp_full[lwarpId]) count_only[s_probe_iter[lwarpId]] = true
                    ); //record count-only probe items
                }
                if (wt == MW_READ_FROM_TABLE_WRITE) { //fetch an item in probe table
                    if (lane < probe_table.num_attrs) //init iRes with the probe item
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][s_probe_iter[lwarpId]];
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1); //todo: opt bucketVec
                        buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val]; //update buc_start and buc_end
                        buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                    }
                }
                else { //fetch an item in tb_in
                    start_table = cur_table = tb_in->m_cur_tables[s_probe_iter[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb_in->m_iRes[s_probe_iter[lwarpId]*num_res_attrs+lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb_in->m_buc_starts[s_probe_iter[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb_in->m_buc_ends[s_probe_iter[lwarpId]];
                    }
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
                                    bp_start_tables[cur_head] = start_table;
                                }

                                /*todo: buc_end*/
                                bp_pos = cur_head * num_hash_tables;
                                for(auto xp = active_rank; xp < num_hash_tables; xp+=active_size)
                                    bp_buc_ends[bp_pos+xp] = buc_end[lwarpId][xp];
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
                    active_in_for_loop.sync(); //necessary to maintain synchronized execution for each warp
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                auto found_res = probe_sbp<DataType, CntType, single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table], s_used_for_compare[cur_table],
                        msIdx[lwarpId][cur_table], msIdx_max[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(if(work_sharing && found_res) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table]+1);
                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > TEMP_CNT_THRESHOLD_WS_DRO) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD_WS_DRO)) {

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD_WS_DRO;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD_WS_DRO * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD_WS_DRO > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD_WS_DRO;
                        if (warp_full[lwarpId]) tb_out_after_bp->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                        else                    tb_out_before_bp->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                    }
                    __syncwarp();

                    /*no need to probe the rest*/
                    SIN_L (
                            tempCounter[lwarpId][cur_table] = 0;
                            buc_end[lwarpId][cur_table] = buc_start[lwarpId][cur_table]); //let end = start
                }
                __syncwarp();

                if (!found_res) { //no match is found
                    if(0 == lane)  tempCounter[lwarpId][cur_table] = 0;
                    if (cur_table > start_table) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table]; //finish this probe item
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
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, ResidualWriteType rwt>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM/BLOCK_SIZE)
void AMHJ_WS_DRO_residual_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec,
        CntType *res_iter, DataType **res,
        AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        CntType *bp_head, CntType *bp_iters, CntType *bp_buc_ends, DataType *bp_iRes, char *bp_start_tables,
        CntType *probe_iter, bool *count_only, CntType max_writable, int num_res_attrs,
        GCQueue<DataType,uint32_t> *cq, TaskBook<DataType,CntType> *tb_in, TaskBook<DataType,CntType> *tb_out, CBarrier *br) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];

    __shared__ bool s_used_for_compare[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_hash_table_num_attrs[MAX_NUM_BUILDTABLES];
    __shared__ CntType s_probe_iter[WARPS_PER_BLOCK];

    __shared__ int tempCounter[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in stealing mode
    __shared__ bool triggerWS[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType gwarpId = (tid + blockDim.x * blockIdx.x) >> WARP_BITS;
    char cur_table = 0, start_table = 0;
    CntType probe_cnt;
    bool found;

    if (rwt == RW_READ_FROM_TABLE)  probe_cnt = probe_table.length;
    else                            probe_cnt = tb_in->m_cnt[0];

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        tempCounter[lwarpId][lane] = 0;
    }
    if (0 == lane) { //to ensure 1st probe item can be fetched
        buc_start[lwarpId][0] = 0;
        buc_end[lwarpId][0] = 0;
    }
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char)hash_tables[tid].num_attrs;
        for(auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
    }
    __syncthreads();

    if ((rwt != RW_READ_FROM_TB_NO_CO) && (gwarpId < bp_head[0])) { //restore the scene
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
        bp_pos = gwarpId * num_hash_tables;
        for(auto xp = lane; xp < num_hash_tables; xp += WARP_SIZE) { //restore buc_end
            buc_end[lwarpId][xp] = bp_buc_ends[bp_pos+xp];
        }
        /* restore the cur_table and start_table */
        cur_table = (char)num_hash_tables - 1;
        start_table = bp_start_tables[gwarpId];
    }

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) && (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item
                if (work_sharing && triggerWS[lwarpId])
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) {
                    SIN_L(s_probe_iter[lwarpId] = atomicAdd(probe_iter,1));
                    if ((rwt != RW_READ_FROM_TB_NO_CO)
                        && (s_probe_iter[lwarpId] < probe_cnt)
                        && (!count_only[s_probe_iter[lwarpId]]))
                        continue; //the item has been processed
                    if (s_probe_iter[lwarpId] >= probe_cnt) {
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
                    start_table = cur_table = tb_out->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb_out->m_iRes[auc[lwarpId]*num_res_attrs+lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb_out->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb_out->m_buc_ends[auc[lwarpId]];
                    }
                }
                else { //init data for original tasks
                    if (rwt == RW_READ_FROM_TABLE) {
                        cur_table = start_table = 0;
                        if (lane < probe_table.num_attrs) //init iRes with the probe item
                            iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][s_probe_iter[lwarpId]];
                        __syncwarp();

                        if (0 == lane) {
                            auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1);
                            buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val];
                            buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val+1];
                        }
                    }
                    else {
                        start_table = cur_table = tb_in->m_cur_tables[s_probe_iter[lwarpId]];
                        if (lane < num_res_attrs) //recover iRes
                            iRes[lwarpId][lane] = tb_in->m_iRes[s_probe_iter[lwarpId]*num_res_attrs+lane];
                        if (0 == lane) { //recover buc_start and buc_end
                            buc_start[lwarpId][cur_table] = tb_in->m_buc_starts[s_probe_iter[lwarpId]];
                            buc_end[lwarpId][cur_table] = tb_in->m_buc_ends[s_probe_iter[lwarpId]];
                        }
                    }
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
                    active_in_for_loop.sync();
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
                    active_in_for_loop.sync();
                }
                __syncwarp();
                cur_table--;
                continue;
            }
            else {
                auto found_res = probe<DataType,CntType,single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(if(work_sharing && found_res) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table]+1);
                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > TEMP_CNT_THRESHOLD_WS_DRO) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD_WS_DRO)) {
                    if (0 == lane) triggerWS[lwarpId] = true;

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD_WS_DRO;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD_WS_DRO * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD_WS_DRO > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD_WS_DRO;
                        auto taskId = tb_out->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                        cq->enqueue(taskId);
                    }
                    __syncwarp();

                    /*no need to probe the rest*/
                    SIN_L (
                            tempCounter[lwarpId][cur_table] = 0;
                            buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table]);
                }
                __syncwarp();

                if (!found_res) { //no match is found
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

template<typename DataType, typename CntType, bool single_join_key, bool work_sharing>
class AMHJ_WS_DRO {
    int block_size;
    int grid_size_sampling;
    int grid_size_main_writing;
    int grid_size_residual_writing;
    uint32_t num_persistent_warps_sampling;
    uint32_t num_persistent_warps_main_writing;
    uint32_t num_persistent_warps_residual_writing;
    int chunk_size;
    int num_res_attrs;
    CntType min_probe_length;

    /*sampling*/
    CntType *samples;
    CntType *sample_res_cnts;

    /*bp information*/
    BreakPoints<DataType,CntType> *bps;

    /*iterators*/
    CntType *probe_iter;
    CntType *min_count_idx;
    CntType *chunk_head;
    CntType *g_mains;
    CntType *g_residuals;
    CntType *residual_cnt;

    /*work-sharing*/
    GCQueue<DataType,uint32_t> *cq; //queueing data is CntType
    TaskBook<DataType,CntType> *tb_in;
    TaskBook<DataType,CntType> *tb_out_before_bp; //for double tb in phase 2
    TaskBook<DataType,CntType> *tb_out_after_bp; //for double tb in phase 2
    TaskBook<DataType,CntType> *tb_in_resis; //for storing a specific tb for case 2
    TaskBook<DataType,CntType> *tb_out_before_bp_resis; //for storing a specific tb for case 2
    CBarrier *br;

    bool *count_only;
    CUDAMemStat *memstat;
    CUDATimeStat *timing;
public:
    AMHJ_WS_DRO(int num_hash_tables, int num_res_attrs, CntType min_probe_length, CntType max_probe_length,
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
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_WS_DRO_sampling<DataType, CntType, single_join_key, work_sharing>, block_size, 0));
        log_info("Kernel: AMHJ_WS_DRO_sampling, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_sampling = numSM * accBlocksPerSM;
        num_persistent_warps_sampling = block_size * grid_size_sampling / WARP_SIZE;
        log_info("Sampling: gris_size=%llu, block_size=%llu, persistent warps=%llu", grid_size_sampling, block_size, num_persistent_warps_sampling);

        /*profile main writing kernel*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_WS_DRO_main_write<DataType, CntType, single_join_key, work_sharing, MW_READ_FROM_TABLE_WRITE>, block_size, 0));
        log_info("Kernel: AMHJ_WS_DRO_main_write, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_main_writing = numSM * accBlocksPerSM;
        num_persistent_warps_main_writing = block_size * grid_size_main_writing / WARP_SIZE;
        log_info("Main-writing: gris_size=%llu, block_size=%llu, persistent warps=%llu", grid_size_main_writing, block_size, num_persistent_warps_main_writing);

        /*profile residual writing kernel*/
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&accBlocksPerSM, AMHJ_WS_DRO_residual_write<DataType, CntType, single_join_key, work_sharing, RW_READ_FROM_TABLE>, block_size, 0));
        log_info("Kernel: AMHJ_WS_DRO_residual_write, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/block_size);
        grid_size_residual_writing = grid_size_main_writing;
        num_persistent_warps_residual_writing = block_size * grid_size_residual_writing / WARP_SIZE;
        log_info("AMHJ_WS_DRO_residual_write: gris_size=%llu, block_size=%llu, persistent warps=%llu", grid_size_residual_writing, block_size, num_persistent_warps_residual_writing);

        /*init sampling related*/
        CUDA_MALLOC(&samples, sizeof(CntType)*num_persistent_warps_sampling, memstat);
        CUDA_MALLOC(&sample_res_cnts, sizeof(CntType)*grid_size_sampling, memstat);

        /*init the bp information*/
        CUDA_MALLOC(&bps, sizeof(BreakPoints<DataType,CntType>), memstat);
        bps->init(num_persistent_warps_main_writing, 1+num_hash_tables, num_res_attrs, memstat);

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

        /*init the work-sharing related data structures*/
        CUDA_MALLOC(&this->cq, sizeof(GCQueue<DataType,uint32_t>), this->memstat); //CntType can only be 32-bit values
        this->cq->init(750000, this->memstat);

        /*init tbs*/
        CUDA_MALLOC(&this->tb_in, sizeof(TaskBook<DataType,CntType>), this->memstat);
        this->tb_in->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->tb_out_after_bp, sizeof(TaskBook<DataType,CntType>), this->memstat);
        this->tb_out_after_bp->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->tb_out_before_bp, sizeof(TaskBook<DataType,CntType>), this->memstat);
        this->tb_out_before_bp->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->tb_in_resis, sizeof(TaskBook<DataType,CntType>), this->memstat);
        this->tb_in_resis->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->tb_out_before_bp_resis, sizeof(TaskBook<DataType,CntType>), this->memstat);
        this->tb_out_before_bp_resis->init(750000, this->num_res_attrs, this->memstat);

        CUDA_MALLOC(&this->br, sizeof(CBarrier), this->memstat); //br
        this->br->initWithWarps(0, this->memstat);
    }

    size_t required_size() { //return the size of intermediate data structure used for evaluation
        return this->cq->get_size() + this->tb_in->get_size() + this->tb_in_resis->get_size() + this->tb_out_after_bp->get_size() + this->tb_out_before_bp->get_size() + this->tb_out_before_bp_resis->get_size() + this->br->get_size();
    }

    CntType AMHJ_WS_DRO_evaluate(
            const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
            uint32_t num_hash_tables,
            bool **used_for_compare, CntType *bucketVec, AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
            DataType **&res, CntType max_writable, cudaStream_t stream = 0) {
        log_trace("Function: %s", __FUNCTION__);
        uint32_t gpu_time_stamp;
        CntType min_count_idx_fixed = 0;

        if (0 == stream) {
            gpu_time_stamp = timing->get_idx();
            execKernel((AMHJ_WS_DRO_sampling<DataType,CntType,single_join_key,work_sharing>), grid_size_sampling, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, samples, sample_res_cnts, num_res_attrs, cq, tb_in, br);
            log_info("Sampling time: %.2f ms.", timing->diff_time(gpu_time_stamp));
        }
        else {
            AMHJ_WS_DRO_sampling<DataType,CntType,single_join_key,work_sharing><<<grid_size_sampling,block_size,0,stream>>>(probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, samples, sample_res_cnts, num_res_attrs, cq, tb_in, br);
            cudaStreamSynchronize(stream);
        }

        /*reset the data structures for AMHJ_WS_main_writing*/
        this->cq->reset(stream); this->tb_in->reset(stream);
        br->reset_with_warps(grid_size_main_writing*block_size/WARP_SIZE, stream);

        /*average estimation*/
        float ave = 0;
        for (auto i = 0; i < grid_size_sampling; i++)
            ave += sample_res_cnts[i];
        ave /= num_persistent_warps_sampling;
        log_debug("Output ave: %.2f output tuples/probe item, total estimated: %.0f.", ave, ave * probe_table.length);
        ave = ave * probe_table.length;
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
        int sub_write_times = 1;
        bool tb_out_before_bp_recorded = false;

        if (res == nullptr) { //allocate space for output
            CUDA_MALLOC(&res, sizeof(DataType *) * num_res_attrs, memstat, stream);
            for (auto i = 0; i < num_res_attrs; i++)
                CUDA_MALLOC(&res[i], sizeof(DataType) * final_num_output, memstat, stream);
        }

        gpu_time_stamp = timing->get_idx();
        execKernel((AMHJ_WS_DRO_main_write<DataType,CntType,single_join_key,work_sharing,MW_READ_FROM_TABLE_WRITE>), grid_size_main_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, g_mains, g_residuals, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, estimated_num_res, chunk_head, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, chunk_size, probe_iter, count_only, min_count_idx, max_writable, num_res_attrs, nullptr, tb_out_before_bp, tb_out_after_bp);
        log_info("Main-write 0, type=MW_READ_FROM_TABLE_WRITE, kernel time=%.2f ms.", timing->diff_time(gpu_time_stamp));
        log_info("Task generated: before BP=%llu, after BP=%llu", tb_out_before_bp->m_cnt[0], tb_out_after_bp->m_cnt[0]);

        //todo: with streaming the access of no_chunk should be sync
        if(tb_out_before_bp->m_cnt[0]+tb_out_after_bp->m_cnt[0] == 0) { //task book has no tasks
            if (g_residuals[0] > 0)     log_debug("Case 3");    //all chunks are used
            else                        log_debug("Case 0");    //have free chunks
        }
        else {
            if (g_residuals[0] > 0)    log_debug("Case 4");    //all chunks are used
            else                        log_debug("Case 2");    //have free chunks
        }

        while ((tb_out_before_bp->m_cnt[0] + tb_out_after_bp->m_cnt[0]) > 0) { //must be case 2 or 4
            checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));
            gpu_time_stamp = timing->get_idx();
            tb_out_after_bp->combine(tb_out_before_bp, stream);
            std::swap(tb_out_after_bp, tb_in);
            tb_out_after_bp->reset(stream);
            if (g_residuals[0] > 0) { //can be case 2 or 4,
                if (!tb_out_before_bp_recorded) {
                    if (tb_out_before_bp->m_cnt[0] > 0) { //swap, because before_bp is useless then
                        std::swap(tb_out_before_bp, tb_out_before_bp_resis);
                    }
                    tb_out_before_bp_recorded = true;
                }

                execKernel((AMHJ_WS_DRO_main_write<DataType,CntType,single_join_key,work_sharing,MW_READ_FROM_TB_COUNT>), grid_size_main_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, g_mains, g_residuals, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, estimated_num_res, chunk_head, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, chunk_size, probe_iter, nullptr, nullptr, max_writable, num_res_attrs, tb_in, nullptr, tb_out_after_bp);
                log_info("Main-write %d, type=MW_READ_FROM_TB_COUNT, kernel time=%.2f ms.", sub_write_times, timing->diff_time(gpu_time_stamp));
            }
            else { //must be case 2
                tb_out_before_bp->reset(stream);
                execKernel((AMHJ_WS_DRO_main_write<DataType,CntType,single_join_key,work_sharing,MW_READ_FROM_TB_WRITE>), grid_size_main_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, g_mains, g_residuals, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, estimated_num_res, chunk_head, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, chunk_size, probe_iter, count_only, min_count_idx, max_writable, num_res_attrs, tb_in, tb_out_before_bp, tb_out_after_bp);
                log_info("Main-write %d, type=MW_READ_FROM_TB_WRITE, kernel time=%.2f ms.", sub_write_times, timing->diff_time(gpu_time_stamp));
                if (g_residuals[0] > 0) { //if case 2 never goes into here, then it is finished with no residual writes
                    std::swap(tb_in, tb_in_resis); //store the current tb_in
                    if (tb_out_before_bp->m_cnt[0] > 0) { //combine instead of swap, since before_bp will be used then
                        tb_out_before_bp_resis->combine(tb_out_before_bp, stream);
                    }
                    tb_out_before_bp_recorded = true;
                }
            }
            cudaStreamSynchronize(stream);
            sub_write_times++;
            log_info("Task generated, before BP=%llu, after BP=%llu", tb_out_before_bp->m_cnt[0], tb_out_after_bp->m_cnt[0]);
        }
        min_count_idx_fixed = min_count_idx[0];

        /*3.Redidual writing*/
        auto mains_res_cnt = g_mains[0];
        auto residual_res_cnt = g_residuals[0];
        log_info("Mains=%llu, Residuals=%llu, Total=%llu", mains_res_cnt, residual_res_cnt, mains_res_cnt+residual_res_cnt);

        if (residual_res_cnt > 0) { //todo: currently the main and residual space are together
            this->cq->reset(stream); this->tb_out_after_bp->reset(stream);
            br->reset_with_warps(grid_size_residual_writing*block_size/WARP_SIZE, stream);
            gpu_time_stamp = timing->get_idx();

            if (tb_in_resis->m_cnt[0] == 0) { //case 3 or 4
                execKernel((AMHJ_WS_DRO_residual_write<DataType,CntType,single_join_key,work_sharing, RW_READ_FROM_TABLE>), grid_size_residual_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, residual_cnt, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, min_count_idx, count_only, max_writable, num_res_attrs, cq, nullptr, tb_out_after_bp, br);
                log_info("Residual-write, type=RW_READ_FROM_TABLE, kernel time=%.2f ms.", timing->diff_time(gpu_time_stamp));
            }
            else { //case 2
                execKernel((AMHJ_WS_DRO_residual_write<DataType,CntType,single_join_key,work_sharing, RW_READ_FROM_TB>), grid_size_residual_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, residual_cnt, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, min_count_idx, count_only, max_writable, num_res_attrs, cq, tb_in_resis, tb_out_after_bp, br);
                log_info("Residual-write, type=RW_READ_FROM_TB, kernel time=%.2f ms.", timing->diff_time(gpu_time_stamp));
            }

            if (tb_out_before_bp_resis->m_cnt[0] > 0) { //has tasks before bp
                checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));//start from the first item
                this->cq->reset(stream); this->tb_out_after_bp->reset(stream);
                br->reset_with_warps(grid_size_residual_writing*block_size/WARP_SIZE, stream);

                gpu_time_stamp = timing->get_idx();
                execKernel((AMHJ_WS_DRO_residual_write<DataType,CntType,single_join_key,work_sharing, RW_READ_FROM_TB_NO_CO>), grid_size_residual_writing, block_size, timing, false, probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, residual_cnt, res, attr_idxes_in_iRes, num_attr_idxes_in_iRes, bps->_head, bps->_iters, bps->_buc_ends, bps->_iRes, bps->_start_tables, probe_iter, count_only, max_writable, num_res_attrs, cq, tb_out_before_bp_resis, tb_out_after_bp, br);
                log_info("Residual-write, type=RW_READ_FROM_TB, kernel time=%.2f ms.", timing->diff_time(gpu_time_stamp));
            }
        }
        log_debug("Written residuals: %llu", *residual_cnt);

        /*reset for the next iteration*/
        this->cq->reset(stream); this->tb_in->reset(stream); this->bps->reset(stream);
        br->reset_with_warps(grid_size_sampling*block_size/WARP_SIZE, stream);
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