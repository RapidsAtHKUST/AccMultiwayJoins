//
// Created by Bryan on 16/8/2019.
//

#pragma once

#include "../conf.h"
#include "cuda/primitives.cuh"
#include "../types.h"
#include "cuda/GCQueue.cuh"
#include "../common_kernels.cuh"
#include "../TaskBook.cuh"
#include "../CBarrier.cuh"
#include "../SubfixBuffer.cuh"
#include "../../common-utils/pretty_print.h"

#include "timer.h"
#include <cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

/* When tempCounter reaches this and rest bucket size is
 * also larger than that, split the tasks*/
#define BUC_SIZE_THRESHOLD   (1024)
#define PROCESS_THRESHOLD    (4)

/*hash probe in the small FIB hash table*/ //todo: need to improve
template<typename DataType>
__device__
bool findItem(
        DataType item, DataType *hashTable, int *bucPtrs,
        int lane, int &idx_in_buffer) {
    auto hashValue = item & (FIB_BUFFER_BUCKETS - 1);
    SIN_L(idx_in_buffer = bucPtrs[hashValue + 1]);
    for (auto i = bucPtrs[hashValue] + lane; i < idx_in_buffer; i += WARP_SIZE) {
        auto active_group = coalesced_threads();
        if (item == hashTable[i]) { //find the match
            idx_in_buffer = i;
        }
        active_group.sync();
    }
    __syncwarp();

    /* return true if found, false otherwise */
    return (idx_in_buffer != bucPtrs[hashValue + 1]);
}

/*
 * Probe-count kernel of AMHJ+FIB with coarse-grained WS
 *  All the tasks (except the one that is processing FIB items) will be executed with WS
 *  The task dealing with FIB item will be executed without WS
 * */
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, bool fib>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM / BLOCK_SIZE)
void AMHJ_FIB_count(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **comp, CntType *bucketVec, CntType *res_cnt, CntType *probe_iter,
        int num_res_attrs, GCQueue<DataType, uint32_t> *cq, TaskBook<DataType, CntType> *tb, CBarrier *br,
        SubfixBuffer<DataType, CntType> *sb) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK];
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType s_prb_iter[WARPS_PER_BLOCK];
    __shared__ CntType l_cnt;
    __shared__ bool s_omp[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_ht_num_attrs[MAX_NUM_BUILDTABLES];

    __shared__ int acc_cnt[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; // #matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in sharing mode
    __shared__ bool ws_trigger[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    __shared__ int sb_item[WARPS_PER_BLOCK];
    __shared__ int sb_col[WARPS_PER_BLOCK];
    __shared__ uint32_t buf_len[WARPS_PER_BLOCK];
    __shared__ bool fibing[WARPS_PER_BLOCK]; //whether the warp is doing fib things
    __shared__ sb_state my_sb_state[WARPS_PER_BLOCK];
    __shared__ int num_sb_cols;

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType p_cnt = 0, p_cnt_fib = 0, probe_cnt = probe_table.length;
    char cur_table = 0, start_table = 0; //begin from the start_table
    bool found;

    /*init the data structures*/
    if (tid == 0) {
        num_sb_cols = sb->m_num_cols; //cache number of fib columns in shared mem
        l_cnt = 0;
    }
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        acc_cnt[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        ws_trigger[tid] = false;
        s_prb_iter[tid] = 0;
        fibing[tid] = false;
        buf_len[tid] = 0;
    }
    SIN_L(
            buc_start[lwarpId][0] = 0;
            buc_end[lwarpId][0] = 0;
    )
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_ht_num_attrs[tid] = (char) hash_tables[tid].num_attrs;
        for (auto i = 0; i < s_ht_num_attrs[tid]; i++) {
            s_omp[tid][i] = comp[tid][i];
        }
    }
    __syncthreads();

    while (true) {
        /* The FIB item has been processed when cur_table is smaller than chosen_sb_col
         * This can only process when cur_table > 0 (sub-joins except the first one)
         * */
        if (fib && fibing[lwarpId] && ((cur_table < sb_col[lwarpId]))) { /*update the last FIB buffer item*/
            if (p_cnt_fib > 0) {
                atomicAdd(&buf_len[lwarpId], p_cnt_fib);
            }
            __syncwarp();

            SIN_L(
                    sb->m_cnts[sb_col[lwarpId]][sb_item[lwarpId]] = buf_len[lwarpId];
                    sb->m_states[sb_col[lwarpId]][sb_item[lwarpId]] = PROCESSED;
                    buf_len[lwarpId] = 0;
                    fibing[lwarpId] = false; //reset fibing
            )
            p_cnt_fib = 0;
        }
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) &&
                (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item or tasks
                if (fib && fibing[lwarpId]) { /*update the last FIB buffer item for sub-join 0*/
                    if (p_cnt_fib > 0) {
                        atomicAdd(&buf_len[lwarpId], p_cnt_fib);
                    }
                    __syncwarp();

                    SIN_L(
                            sb->m_cnts[sb_col[lwarpId]][sb_item[lwarpId]] = buf_len[lwarpId];
                            sb->m_states[sb_col[lwarpId]][sb_item[lwarpId]] = PROCESSED;
                            buf_len[lwarpId] = 0;
                            fibing[lwarpId] = false; //reset fibing
                    )
                    p_cnt_fib = 0;
                }

                if (work_sharing && ws_trigger[lwarpId]) {
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]))
                }
                if (!sharing[lwarpId]) {
                    SIN_L(s_prb_iter[lwarpId] = atomicAdd(probe_iter, 1))
                    if (s_prb_iter[lwarpId] >= probe_cnt) {
                        if (!work_sharing) goto LOOP_END;
                        SIN_L(
                                sharing[lwarpId] = cq->dequeue(auc[lwarpId]);
                                ws_trigger[lwarpId] = true;
                        )
                        while (true) {
                            if (sharing[lwarpId]) break;
                            SIN_L(br->setActive(false))
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(
                                            br->setActive(true);
                                            sharing[lwarpId] = cq->dequeue(auc[lwarpId])
                                    )
                                    if (!sharing[lwarpId]) {
                                        SIN_L(br->setActive(false))
                                    }
                                }
                                if (br->isTerminated()) {
                                    goto LOOP_END; //all warps reach this barrier, exit
                                }
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) {
                    start_table = cur_table = tb->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) { //recover iRes
                        iRes[lwarpId][lane] = tb->m_iRes[auc[lwarpId] * num_res_attrs + lane];
                    }
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb->m_buc_ends[auc[lwarpId]];
                    }
                } else { //init data for original tasks
                    cur_table = start_table = 0;
                    if (lane < probe_table.num_attrs) {//init iRes with the probe item
                        auto probe_data = probe_table.data[lane][s_prb_iter[lwarpId]];
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_data;
                    }
                    __syncwarp();

                    /*check the buffers for iRes[lwarpId][hash_tables[0].hash_attr]
                     * if in the buffer, just append the existing result to interRes[lwarpId][0]
                     * and go to the next item
                     * */
                    /*chosen_sb_item is assigned in the findItem function*/
                    if (fib && (sb->m_num_items[0] > 0) &&
                        findItem(iRes[lwarpId][hash_tables[0].hash_attr], sb->m_ht[cur_table],
                                 sb->m_ht_bucptrs[cur_table], lane, sb_item[lwarpId])) {
                        auto state = sb->m_states[cur_table][sb_item[lwarpId]];
                        if (PROCESSED == state) { //processed sb item, reuse
                            SIN_L(p_cnt += sb->m_cnts[cur_table][sb_item[lwarpId]])
                            continue;
                        }
                        SIN_L(
                                my_sb_state[lwarpId] = (sb_state) atomicCAS(
                                        (int *) &sb->m_states[cur_table][sb_item[lwarpId]],
                                        NOT_PROCESSED, UNDER_PROCESS);
                        ) //try to lock the sb item
                        if ((NOT_PROCESSED == my_sb_state[lwarpId]) &&
                            (0 == lane)) { //successfully lock the sb item and process it
                            fibing[lwarpId] = true;
                            sb_col[lwarpId] = cur_table;
                        }
                    }
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1);
                        auto hb = hash_tables[0].buc_ptrs;
                        buc_start[lwarpId][0] = hb[hash_val];
                        buc_end[lwarpId][0] = hb[hash_val + 1];
                    }
                }
                __syncwarp();
            }
            if (cur_table == num_hash_tables - 1) { //reach the last table
                for (auto j = buc_start[lwarpId][cur_table] + lane;
                     j < buc_end[lwarpId][cur_table];
                     j += WARP_SIZE) {
                    bool is_chosen = true;
                    if (!single_join_key) {
                        for (auto a = 0; a < s_ht_num_attrs[cur_table]; a++) {
                            if (s_omp[cur_table][a]) {
                                auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                                if (iRes[lwarpId][hash_tables[cur_table].attr_list[a]]
                                    != hash_tables[cur_table].data[a][origin_idx]) {
                                    is_chosen = false;
                                    break;
                                }
                            }
                        }
                    }
                    auto res_comp = iRes[lwarpId][hash_tables[cur_table].hash_attr];
                    auto ht_comp = hash_tables[cur_table].hash_keys[j];
                    if (((!single_join_key) && (is_chosen)) ||
                        ((single_join_key) && (res_comp == ht_comp))) {
                        p_cnt++;
                        if (fib && fibing[lwarpId]) {
                            p_cnt_fib++;
                        }
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            } else {
                found = probe<DataType, CntType, single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_omp[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(
                        if (work_sharing && found) {
                            acc_cnt[lwarpId][cur_table] += (msIdx[lwarpId][cur_table] + 1);
                        }
                )

                /*skew detection*/
                if (work_sharing && (acc_cnt[lwarpId][cur_table] > PROCESS_THRESHOLD) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD)) {
                    if (!fibing[lwarpId]) {
                        if (0 == lane) ws_trigger[lwarpId] = true;

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
                                acc_cnt[lwarpId][cur_table] = 0;
                                buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table];
                        )
                    }
                }
                __syncwarp();

                if (!found) { //no match is found
                    if (0 == lane) {
                        acc_cnt[lwarpId][cur_table] = 0;
                    }
                    if (cur_table > start_table) {
                        cur_table--; //backtrack to the last attribute
                    } else if (0 == lane) {
                        buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table]; //finish this probe item
                    }
                    __syncwarp();
                    continue;
                }
            }
        } else if (0 == lane) msIdx[lwarpId][cur_table]--;
        __syncwarp();

        /*write iRes*/
        auto cur_ms = msIdx[lwarpId][cur_table];
        auto curIter = iterators[lwarpId][cur_table][cur_ms]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        if ((lane < s_ht_num_attrs[cur_table]) && (!s_omp[cur_table][lane])) {
            auto ht_data = hash_tables[cur_table].data[lane][idx_in_origin_table];
            iRes[lwarpId][hash_tables[cur_table].attr_list[lane]] = ht_data;
        }
        __syncwarp();

        SIN_L(
                auto ht = hash_tables[cur_table + 1];
                auto ires_val = iRes[lwarpId][ht.hash_attr];
                auto hash_val = ires_val & (bucketVec[cur_table + 1] - 1);
                buc_start[lwarpId][cur_table + 1] = ht.buc_ptrs[hash_val];
                buc_end[lwarpId][cur_table + 1] = ht.buc_ptrs[hash_val + 1];
                msIdx[lwarpId][cur_table + 1] = 0;
        ) //init the msIdx for the next attr
        cur_table++; //advance to the next attr

        /* Check for new FIB item
         * This can only process when cur_table > 0 (sub-joins except the first one)
         * */
        if (fib && (cur_table < num_sb_cols) && (sb->m_num_items[cur_table] > 0) &&
            findItem(iRes[lwarpId][hash_tables[cur_table].hash_attr], sb->m_ht[cur_table],
                     sb->m_ht_bucptrs[cur_table], lane, auc[lwarpId])) {
            if (PROCESSED == sb->m_states[cur_table][auc[lwarpId]]) { //processed sb item, reuse
                SIN_L(p_cnt += sb->m_cnts[cur_table][auc[lwarpId]])
                if (fibing[lwarpId]) {
                    SIN_L(p_cnt_fib += sb->m_cnts[cur_table][auc[lwarpId]])
                }
                cur_table--; //go back to the last sub-join
                continue;
            }
            if (!fibing[lwarpId]) {
                SIN_L(
                        sb_item[lwarpId] = auc[lwarpId];
                        my_sb_state[lwarpId] = (sb_state) atomicCAS(
                                (int *) &sb->m_states[cur_table][sb_item[lwarpId]],
                                NOT_PROCESSED, UNDER_PROCESS);
                ) //try to lock the sb item
                if (NOT_PROCESSED == my_sb_state[lwarpId]) { //successfully lock the sb item and process it
                    SIN_L(
                            fibing[lwarpId] = true;
                            sb_col[lwarpId] = cur_table;
                    )
                }
            }
            __syncwarp();
        }
    }
    LOOP_END:
    __syncwarp();

    WARP_REDUCE(p_cnt);
    if (lane == 0) atomicAdd(&l_cnt, p_cnt);
    __syncthreads();

    if (0 == tid) atomicAdd(res_cnt, l_cnt);
}

/*
 * Probe-write kernel of AMHJ+FIB with coarse-grained WS
 *  All the tasks (except the one that is processing FIB items) will be executed with WS
 *  The task dealing with FIB item will be executed without WS
 * */
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, bool fib>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM / BLOCK_SIZE)
void AMHJ_FIB_write(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **comp, CntType *bucketVec, CntType *res_cnt, CntType *probe_iter,
        DataType **res, int num_res_attrs, AttrType *attr_idxes_in_res,
        CntType max_writable, AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
        GCQueue<DataType, uint32_t> *cq, TaskBook<DataType, CntType> *tb, CBarrier *br,
        SubfixBuffer<DataType, CntType> *sb) {
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES][WARP_SIZE];
    __shared__ DataType auc[WARPS_PER_BLOCK];
    __shared__ CntType auc_probe[WARPS_PER_BLOCK]; /*replacing the break clause*/
    __shared__ CntType buc_start[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ CntType buc_end[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES];

    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_RES_ATTRS];
    __shared__ CntType s_prb_iter[WARPS_PER_BLOCK];

    __shared__ bool s_comp[MAX_NUM_BUILDTABLES][MAX_NUM_ATTRS_IN_BUILD_TABLE];
    __shared__ char s_ht_num_attrs[MAX_NUM_BUILDTABLES];

    __shared__ int temp_cnt[WARPS_PER_BLOCK][MAX_NUM_BUILDTABLES]; //recording the number of matches found in each build tables
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in sharing mode
    __shared__ bool ws_trigger[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    __shared__ int sb_item[WARPS_PER_BLOCK];
    __shared__ int sb_col[WARPS_PER_BLOCK];
    __shared__ bool fibing[WARPS_PER_BLOCK]; //whether the warp is doing fib things
    __shared__ sb_state my_sb_state[WARPS_PER_BLOCK];
    __shared__ int num_sb_cols;
    __shared__ CntType s_wrt_pos[WARPS_PER_BLOCK];

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType probe_cnt = probe_table.length;
    char cur_table = 0, start_table = 0; //begin from the start_table
    bool found;

    /*init the data structures*/
    if (tid == 0) {
        num_sb_cols = sb->m_num_cols; //cache number of fib columns in shared mem
    }
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        temp_cnt[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        ws_trigger[tid] = false;
        s_prb_iter[tid] = 0;
        fibing[tid] = false;
    }
    SIN_L(
            buc_start[lwarpId][0] = 0;
            buc_end[lwarpId][0] = 0);
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_ht_num_attrs[tid] = (char) hash_tables[tid].num_attrs;
        for (auto i = 0; i < s_ht_num_attrs[tid]; i++) {
            s_comp[tid][i] = comp[tid][i];
        }
    }
    __syncthreads();

    while (true) {
        /* The FIB item has been processed when cur_table is smaller than chosen_sb_col
         * This can only process when cur_table > 0 (sub-joins except the first one)
         * */
        if (fib && fibing[lwarpId] && ((cur_table < sb_col[lwarpId]))) { /*update the last FIB buffer item*/
            SIN_L(
                    sb->m_states[sb_col[lwarpId]][sb_item[lwarpId]] = PROCESSED;
                    fibing[lwarpId] = false; //reset fibing
            )
        }
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) &&
                (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item or tasks
                if (fib && fibing[lwarpId]) { /*update the last FIB buffer item for sub-join 0*/
                    SIN_L(
                            sb->m_states[sb_col[lwarpId]][sb_item[lwarpId]] = PROCESSED;
                            fibing[lwarpId] = false; //reset fibing
                    )
                }

                if (work_sharing && ws_trigger[lwarpId]) {
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]))
                }
                if (!sharing[lwarpId]) {
                    SIN_L(s_prb_iter[lwarpId] = atomicAdd(probe_iter, 1))
                    if (s_prb_iter[lwarpId] >= probe_cnt) {
                        if (!work_sharing) return;
                        SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]);
                                      ws_trigger[lwarpId] = true);
                        while (true) {
                            if (sharing[lwarpId]) break;
                            SIN_L(br->setActive(false));
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(br->setActive(true);
                                                  sharing[lwarpId] = cq->dequeue(auc[lwarpId]))
                                    if (!sharing[lwarpId]) SIN_L(br->setActive(false))
                                }
                                if (br->isTerminated()) return;//all warps reach this barrier, exit
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) {
                    start_table = cur_table = tb->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb->m_iRes[auc[lwarpId] * num_res_attrs + lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb->m_buc_ends[auc[lwarpId]];
                    }
                } else { //init data for original tasks
                    cur_table = start_table = 0;
                    if (lane < probe_table.num_attrs) { //init iRes with the probe item
                        auto probe_data = probe_table.data[lane][s_prb_iter[lwarpId]];
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_data;
                    }
                    __syncwarp();

                    /*check the buffers for iRes[lwarpId][hash_tables[0].hash_attr]
                     * if in the buffer, just append the existing result to interRes[lwarpId][0]
                     * and go to the next item
                     * */
                    /*chosen_sb_item is assigned in the findItem function*/
                    if (fib && (sb->m_num_items[0] > 0)
                        && findItem(iRes[lwarpId][hash_tables[0].hash_attr], sb->m_ht[cur_table],
                                    sb->m_ht_bucptrs[cur_table], lane, sb_item[lwarpId])) {
                        auto state = sb->m_states[cur_table][sb_item[lwarpId]];
                        if (PROCESSED == state) { //processed sb item, reuse
                            CntType base_wrt_pos;
                            SIN_L(
                                    base_wrt_pos = atomicAdd(res_cnt,
                                                             sb->m_cnts[cur_table][sb_item[lwarpId]]) % max_writable;
                            )
                            base_wrt_pos = __shfl_sync(0xffffffff, base_wrt_pos, 0);
                            auto base_rd_pos = sb->m_starts[cur_table][sb_item[lwarpId]] % max_writable;

                            for (auto p = 0; p < num_res_attrs; p++) {
                                for (auto off = lane;
                                     off < sb->m_cnts[cur_table][sb_item[lwarpId]];
                                     off += WARP_SIZE) {
                                    auto wrt_to_off = (base_wrt_pos + off) % max_writable;
                                    auto rd_from_off = (base_rd_pos + off) % max_writable;
                                    auto cur_attr = attr_idxes_in_res[p];
                                    if (p <= cur_table + 1) {
                                        res[cur_attr][wrt_to_off] = iRes[lwarpId][cur_attr];
                                    } else {
                                        res[cur_attr][wrt_to_off] = res[cur_attr][rd_from_off]; //res copy
                                    }
                                }
                            }
                            continue;
                        }
                        SIN_L(
                                my_sb_state[lwarpId] = (sb_state) atomicCAS(
                                        (int *) &sb->m_states[cur_table][sb_item[lwarpId]],
                                        NOT_PROCESSED, UNDER_PROCESS);
                        ) //try to lock the sb item

                        if (NOT_PROCESSED == my_sb_state[lwarpId]) { //successfully lock the sb item and process it
                            SIN_L(
                                    fibing[lwarpId] = true;
                                    sb_col[lwarpId] = cur_table;
                                    s_wrt_pos[lwarpId] = atomicAdd(res_cnt,
                                                                   sb->m_cnts[cur_table][sb_item[lwarpId]]);
                                    sb->m_starts[cur_table][sb_item[lwarpId]] = s_wrt_pos[lwarpId];
                            )
                        }

                    }
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1);
                        buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val];
                        buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val + 1];
                    }
                }
                __syncwarp();
            }
            if (cur_table == num_hash_tables - 1) { //reach the last table
                for (auto j = buc_start[lwarpId][cur_table] + lane; j < buc_end[lwarpId][cur_table]; j += WARP_SIZE) {
                    bool is_chosen = true;
                    if (!single_join_key) {
                        for (auto a = 0; a < s_ht_num_attrs[cur_table]; a++) {
                            if (s_comp[cur_table][a]) {
                                auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
                                if (iRes[lwarpId][hash_tables[cur_table].attr_list[a]]
                                    != hash_tables[cur_table].data[a][origin_idx]) {
                                    is_chosen = false;
                                    break;
                                }
                            }
                        }
                    }
                    auto ires = iRes[lwarpId][hash_tables[cur_table].hash_attr];
                    auto ht_data = hash_tables[cur_table].hash_keys[j];
                    if (((!single_join_key) && (is_chosen)) ||
                        ((single_join_key) && (ires == ht_data))) {
                        CntType writePos;
                        if (fib && fibing[lwarpId]) {
                            writePos = atomicAdd(&s_wrt_pos[lwarpId], 1) % max_writable;
                        } else {
                            writePos = atomicAdd(res_cnt, 1) % max_writable;
                        }

                        auto origin_idx = hash_tables[cur_table].idx_in_origin[j];
#pragma unroll
                        for (auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                            res[attr_idxes_in_iRes[p]][writePos] = iRes[lwarpId][attr_idxes_in_iRes[p]];
                        for (auto p = 0; p < s_ht_num_attrs[cur_table]; p++) {
                            if (!s_comp[cur_table][p]) { //this attr only appears in the last ht
                                auto ht_data = hash_tables[cur_table].data[p][origin_idx];
                                res[hash_tables[cur_table].attr_list[p]][writePos] = ht_data;
                            }
                        }
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            } else {
                found = probe<DataType, CntType, single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_comp[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(
                        if (work_sharing && found) temp_cnt[lwarpId][cur_table] += (msIdx[lwarpId][cur_table] + 1)
                )

                /*skew detection*/
                if (work_sharing && (temp_cnt[lwarpId][cur_table] > PROCESS_THRESHOLD) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD)) {
                    if (!fibing[lwarpId]) {
                        if (0 == lane) ws_trigger[lwarpId] = true;

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
                                temp_cnt[lwarpId][cur_table] = 0;
                                buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table]
                        )
                    }
                }
                __syncwarp();

                if (!found) { //no match is found
                    if (0 == lane) temp_cnt[lwarpId][cur_table] = 0;
                    if (cur_table > start_table) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane) { //finish this probe item
                        buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table];
                    }
                    __syncwarp();
                    continue;
                }
            }
        } else if (0 == lane) msIdx[lwarpId][cur_table]--;
        __syncwarp();

        /*write iRes*/
        auto cur_ms = msIdx[lwarpId][cur_table];
        auto curIter = iterators[lwarpId][cur_table][cur_ms]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        if ((lane < s_ht_num_attrs[cur_table]) && (!s_comp[cur_table][lane])) {
            auto ht_data = hash_tables[cur_table].data[lane][idx_in_origin_table];
            iRes[lwarpId][hash_tables[cur_table].attr_list[lane]] = ht_data;
        }
        __syncwarp();

        SIN_L(
                auto ht = hash_tables[cur_table + 1];
                auto hash_val = iRes[lwarpId][ht.hash_attr] & (bucketVec[cur_table + 1] - 1);
                buc_start[lwarpId][cur_table + 1] = ht.buc_ptrs[hash_val];
                buc_end[lwarpId][cur_table + 1] = ht.buc_ptrs[hash_val + 1];
                msIdx[lwarpId][cur_table + 1] = 0
        ); //init the msIdx for the next attr
        cur_table++; //advance to the next attr

        /* Check for new FIB item
         * This can only process when cur_table > 0 (sub-joins except the first one)
         * */
        if (fib && (cur_table < num_sb_cols) && (sb->m_num_items[cur_table] > 0)
            && findItem(iRes[lwarpId][hash_tables[cur_table].hash_attr], sb->m_ht[cur_table],
                        sb->m_ht_bucptrs[cur_table], lane, auc[lwarpId])) {
            auto state = sb->m_states[cur_table][auc[lwarpId]];
            if (PROCESSED == state) { //processed sb item, reuse
                CntType base_wrt_pos;
                if (fibing[lwarpId]) { //already in the fib mode
                    SIN_L(
                            base_wrt_pos =
                                    atomicAdd(&s_wrt_pos[lwarpId], sb->m_cnts[cur_table][auc[lwarpId]]) %
                                    max_writable
                    )
                } else {
                    SIN_L(
                            base_wrt_pos = atomicAdd(res_cnt, sb->m_cnts[cur_table][auc[lwarpId]]) %
                                           max_writable
                    )
                }
                base_wrt_pos = __shfl_sync(0xffffffff, base_wrt_pos, 0);
                auto base_rd_pos = sb->m_starts[cur_table][auc[lwarpId]] % max_writable;
                for (auto p = 0; p < num_res_attrs; p++) {
                    for (auto off = lane;
                         off < sb->m_cnts[cur_table][auc[lwarpId]];
                         off += WARP_SIZE) {
                        auto wrt_to_off = (base_wrt_pos + off) % max_writable;
                        auto rd_from_off = (base_rd_pos + off) % max_writable;
                        auto cur_attr = attr_idxes_in_res[p];
                        if (p <= cur_table + 1) {
                            res[cur_attr][wrt_to_off] = iRes[lwarpId][cur_attr];
                        } else {
                            res[cur_attr][wrt_to_off] = res[cur_attr][rd_from_off]; //res copy
                        }
                    }
                }
                cur_table--; //go back to the last sub-join`
                continue;
            }
            if (!fibing[lwarpId]) {
                SIN_L(
                        sb_item[lwarpId] = auc[lwarpId];
                        my_sb_state[lwarpId] = (sb_state) atomicCAS(
                                (int *) &sb->m_states[cur_table][sb_item[lwarpId]],
                                NOT_PROCESSED, UNDER_PROCESS);
                ) //try to lock the sb item

                if (NOT_PROCESSED == my_sb_state[lwarpId]) { //successfully lock the sb item and process it
                    SIN_L(
                            fibing[lwarpId] = true;
                            sb_col[lwarpId] = cur_table;
                            s_wrt_pos[lwarpId] = atomicAdd(res_cnt,
                                                           sb->m_cnts[cur_table][sb_item[lwarpId]]);
                            sb->m_starts[cur_table][sb_item[lwarpId]] = s_wrt_pos[lwarpId];
                    )
                }
            }
            __syncwarp();
        }
    }
}

template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, bool fib>
class AMHJ_WS_FIB {
    GCQueue<DataType, uint32_t> *cq; //queueing data is CntType
    TaskBook<DataType, CntType> *tb;
    CBarrier *br;
    int num_res_attrs;
    CntType *res_cnt;
    CntType *probe_iter;

    CUDAMemStat *memstat;
    CUDATimeStat *timing;
public:
    AMHJ_WS_FIB(int num_res_attrs, CUDAMemStat *memstat, CUDATimeStat *timing) : num_res_attrs(num_res_attrs),
                                                                                 memstat(memstat), timing(timing) {
        CUDA_MALLOC(&this->cq, sizeof(GCQueue<DataType, uint32_t>), this->memstat); //CntType can only be 32-bit values
        this->cq->init(750000, this->memstat);
        CUDA_MALLOC(&this->tb, sizeof(TaskBook<DataType, CntType>), this->memstat); //tb
        this->tb->init(750000, this->num_res_attrs, this->memstat);
        CUDA_MALLOC(&this->br, sizeof(CBarrier), this->memstat); //br
        this->br->initWithWarps(0, this->memstat);

        CUDA_MALLOC(&res_cnt, sizeof(CntType), this->memstat);
        CUDA_MALLOC(&probe_iter, sizeof(CntType), this->memstat);
    }

    size_t required_size() { //return the size of intermediate data structure used for evaluation
        return this->cq->get_size() + this->tb->get_size() + this->br->get_size();
    }

    CntType AMHJ_WS_FIB_evaluate(const Relation<DataType, CntType> probe_table,
                                 HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
                                 bool **used_for_compare, CntType *bucketVec,
                                 AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
                                 DataType **&res, CntType max_writable,
                                 cudaStream_t stream = 0) {
        log_trace("Function: %s", __FUNCTION__);

        /*compute the whole attr idxes in res*/
        assert(num_res_attrs == (num_attr_idxes_in_iRes + 1));
        AttrType *attr_idxes_in_res = nullptr;
        CUDA_MALLOC(&attr_idxes_in_res, sizeof(AttrType) * num_res_attrs, this->memstat);
        for (auto i = 0; i < num_attr_idxes_in_iRes; i++) {
            attr_idxes_in_res[i] = attr_idxes_in_iRes[i];
        }
        attr_idxes_in_res[num_res_attrs - 1] = hash_tables[num_hash_tables - 1].attr_list[1];

        /*setting for persistent warps*/
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        auto maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        auto numSM = prop.multiProcessorCount;

        int block_size = BLOCK_SIZE;
        checkCudaErrors(cudaMemsetAsync(res_cnt, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));
        this->cq->reset(stream);
        this->tb->reset(stream);

        /*subfix buffer for FIB*/
        SubfixBuffer<DataType, CntType> *sb;
        CUDA_MALLOC(&sb, sizeof(SubfixBuffer<DataType, CntType>), memstat);
        sb->init(num_hash_tables - 1, memstat, timing);

        /*grouping data*/
        DataType **temp_group_data;
        CntType *temp_group_cnts;
        CUDA_MALLOC(&temp_group_data, sizeof(DataType *) * (num_hash_tables - 1), memstat);
        CUDA_MALLOC(&temp_group_cnts, sizeof(CntType) * (num_hash_tables - 1), memstat);
        temp_group_data[0] = probe_table.data[1];
        temp_group_cnts[0] = probe_table.length;

        for (auto i = 0; i < num_hash_tables - 2; i++) {
            temp_group_data[i + 1] = hash_tables[i].data[1];
            temp_group_cnts[i + 1] = hash_tables[i].length;
        }
        sb->compute_topk(temp_group_data, temp_group_cnts);

        /*1.count*/
        int accBlocksPerSM;
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &accBlocksPerSM, AMHJ_FIB_count<DataType, CntType, single_join_key, work_sharing, fib>,
                block_size, 0));
        log_info("Kernel: AMHJ_FIB_count, occupancy: %d/%d.", accBlocksPerSM, maxThreadsPerSM / block_size);

        /*set gridSize according to the profiling result*/
        int grid_size = numSM * accBlocksPerSM;
        log_info("grid_size = %d, block_size = %d.", grid_size, block_size);
        br->reset_with_warps(grid_size * block_size / WARP_SIZE, stream);

        if (stream == 0) { //timing
            auto timestamp = timing->get_idx();
            execKernel((AMHJ_FIB_count<DataType, CntType, single_join_key, work_sharing, fib>), grid_size, block_size,
                       timing, true,
                       probe_table, hash_tables, num_hash_tables, used_for_compare,
                       bucketVec, res_cnt, probe_iter, this->num_res_attrs, cq, tb, br, sb);
            log_info("AMHJ_FIB_count time: %.2f ms", timing->diff_time(timestamp));
        } else { //streaming, no timing
            AMHJ_FIB_count<DataType, CntType, single_join_key, work_sharing, fib>
                    << < grid_size, block_size, 0, stream >> > (
                    probe_table, hash_tables, num_hash_tables, used_for_compare,
                            bucketVec, res_cnt, probe_iter, this->num_res_attrs, cq, tb, br, sb);
            cudaStreamSynchronize(stream);
        }

        CntType total_cnt = res_cnt[0];
        log_info("Output count: %llu.", total_cnt);

        /*2.global scan & output page generation*/
        log_debug("GCQueue: qHead=%d, qRear=%d.", cq->qHead[0], cq->qRear[0]);
        log_debug("Num tasks: %d", tb->m_cnt[0]);

        /*reset the cq, tb, sb data structures*/
        this->cq->reset(stream);
        this->tb->reset(stream);
        checkCudaErrors(cudaMemsetAsync(res_cnt, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync(probe_iter, 0, sizeof(CntType), stream));

        /*reset m_states to 0 (not processed)*/
        for (int i = 0; i < sb->m_num_cols; i++) {
            checkCudaErrors(cudaMemset(sb->m_states[i], 0, sizeof(sb_state) * FIB_BUFFER_MAX_LEN));
        }

        /*3.materialization*/
        grid_size = numSM * accBlocksPerSM; //change the gridSize since the occupancy may change
        br->reset_with_warps(grid_size * block_size / WARP_SIZE, stream);

        CntType final_num_output;
        if (total_cnt > max_writable) {
            log_warn("Output exceeded the max limit, will write circle");
            final_num_output = max_writable;
        } else final_num_output = total_cnt;
        if (res == nullptr) {
            log_info("Allocate space for res");
            CUDA_MALLOC(&res, sizeof(DataType *) * num_res_attrs, memstat);
            for (auto i = 0; i < num_res_attrs; i++)
                CUDA_MALLOC(&res[i], sizeof(DataType) * final_num_output, memstat);
        }

        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &accBlocksPerSM,
                AMHJ_FIB_write<DataType, CntType, single_join_key, work_sharing, fib>,
                block_size, 0));
        log_info("Kernel: AMHJ_WS_write, occupancy: %d/%d.", accBlocksPerSM, 2048 / block_size);

        if (stream == 0) { //timing
            auto timestamp = timing->get_idx();
            execKernel((AMHJ_FIB_write<DataType, CntType, single_join_key, work_sharing, fib>), grid_size, block_size,
                       timing, true,
                       probe_table, hash_tables, num_hash_tables, used_for_compare,
                       bucketVec, res_cnt, probe_iter, res, num_res_attrs, attr_idxes_in_res,
                       max_writable, attr_idxes_in_iRes, num_attr_idxes_in_iRes, cq, tb, br, sb);
            log_info("AMHJ_FIB_write time: %.2f ms", timing->diff_time(timestamp));
        } else {
            AMHJ_FIB_write<DataType, CntType, single_join_key, work_sharing, fib>
                    << < grid_size, block_size, 0, stream >> > (
                    probe_table, hash_tables, num_hash_tables, used_for_compare,
                            bucketVec, res_cnt, probe_iter, res, num_res_attrs, attr_idxes_in_res,
                            max_writable, attr_idxes_in_iRes, num_attr_idxes_in_iRes, cq, tb, br, sb);
            cudaStreamSynchronize(stream);
        }
        CUDA_FREE(attr_idxes_in_res, this->memstat);
        return total_cnt;
    }
};


/*
 * AMHJ+FIB with fine-grained WS, deprecated
 *  All the tasks (including the one that is processing FIB items) will be executed with WS
 *  Note: currently slower than the coarse-grained one above
 * */
template<typename DataType, typename CntType, bool single_join_key, bool work_sharing, bool fib>
__global__ __launch_bounds__(BLOCK_SIZE, MAX_THREADS_PER_SM / BLOCK_SIZE)
void AMHJ_FIB_count_fine_grained(
        const Relation<DataType, CntType> probe_table, HashTable<DataType, CntType> *hash_tables,
        uint32_t num_hash_tables, bool **used_for_compare, CntType *bucketVec, CntType *res_cnt, CntType *probe_iter,
        int num_res_attrs, GCQueue<DataType, uint32_t> *cq, TaskBook<DataType, CntType> *tb, CBarrier *br,
        SubfixBuffer<DataType, CntType> *sb,
        CntType *free_count_1st) {
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
    __shared__ bool sharing[WARPS_PER_BLOCK]; //whether the warps are in sharing mode
    __shared__ bool triggerWS[WARPS_PER_BLOCK]; //whether WS is triggered in this block

    __shared__ int chosen_sb_item[WARPS_PER_BLOCK];
    __shared__ int chosen_sb_col[WARPS_PER_BLOCK];
    __shared__ bool fibing[WARPS_PER_BLOCK]; //whether the warp is doing fib things
    __shared__ sb_state my_sb_state[WARPS_PER_BLOCK];

    auto tid = threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    CntType p_cnt = 0, probe_cnt = probe_table.length;
    char cur_table = 0, start_table = 0; //begin from the start_table
    bool found;
    CntType p_cnt_fib = 0;

    /*init the data structures*/
    if (lane < MAX_NUM_BUILDTABLES) {
        msIdx[lwarpId][lane] = 0;
        tempCounter[lwarpId][lane] = 0;
    }
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
        probe_iterator[tid] = 0;
        fibing[tid] = false;
    }
    SIN_L(
            buc_start[lwarpId][0] = 0;
            buc_end[lwarpId][0] = 0);
    if (0 == tid) l_cnt = 0;
    if (tid < num_hash_tables) { //move these two metadata into the shared memory
        s_hash_table_num_attrs[tid] = (char) hash_tables[tid].num_attrs;
        for (auto i = 0; i < s_hash_table_num_attrs[tid]; i++)
            s_used_for_compare[tid][i] = used_for_compare[tid][i];
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_table]) { //get the next matching item
            if ((start_table == cur_table) &&
                (buc_start[lwarpId][start_table] >= buc_end[lwarpId][start_table])) { //get new probe item or tasks
                if (fib && fibing[lwarpId]) { /*update the last FIB buffer item*/
                    if (p_cnt_fib > 0) {
                        atomicAdd((CntType *) &sb->m_cnts[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]], p_cnt_fib);
                    }
                    __syncwarp();

                    if (0 == lane) {
                        atomicSub(&sb->m_num_tasks_related[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]], 1);
                        if (0 ==
                            sb->m_num_tasks_related[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]]) { //all the tasks are finished
                            sb->m_states[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]] = PROCESSED;
                        }
                        fibing[lwarpId] = false; //reset fibing
                    }
                    p_cnt_fib = 0;
                    __syncwarp();
                }

                if (work_sharing && triggerWS[lwarpId]) {
                    SIN_L(sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                }
                if (!sharing[lwarpId]) {
                    SIN_L(probe_iterator[lwarpId] = atomicAdd(probe_iter, 1));
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
                                if (br->isTerminated()) goto LOOP_END;//all warps reach this barrier, exit
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) {
                    start_table = cur_table = tb->m_cur_tables[auc[lwarpId]];
                    if (lane < num_res_attrs) //recover iRes
                        iRes[lwarpId][lane] = tb->m_iRes[auc[lwarpId] * num_res_attrs + lane];
                    if (0 == lane) { //recover buc_start and buc_end
                        buc_start[lwarpId][cur_table] = tb->m_buc_starts[auc[lwarpId]];
                        buc_end[lwarpId][cur_table] = tb->m_buc_ends[auc[lwarpId]];
                        if (fib) { //recover the fib related things
                            fibing[lwarpId] = tb->m_is_fib_task[auc[lwarpId]];
                            if (fibing[lwarpId]) {
                                chosen_sb_col[lwarpId] = tb->m_fib_col_id[auc[lwarpId]];
                                chosen_sb_item[lwarpId] = tb->m_fib_item_id[auc[lwarpId]];
                            }
                        }
                    }
                } else { //init data for original tasks
                    cur_table = start_table = 0;
                    if (lane < probe_table.num_attrs) //init iRes with the probe item
                        iRes[lwarpId][probe_table.attr_list[lane]] = probe_table.data[lane][probe_iterator[lwarpId]];
                    __syncwarp();

                    /*check the buffers for iRes[lwarpId][hash_tables[0].hash_attr]
                     * if in the buffer, just append the existing result to interRes[lwarpId][0]
                     * and go to the next item
                     * */
                    /*chosen_sb_item is assigned in the findItem function*/
                    if (fib) {
                        auto found_in_buffer = findItem(iRes[lwarpId][hash_tables[0].hash_attr], sb->m_ht[cur_table],
                                                        sb->m_ht_bucptrs[cur_table], lane, chosen_sb_item[lwarpId]);
                        if (found_in_buffer) { //the item is found in cache
                            if (PROCESSED ==
                                sb->m_states[cur_table][chosen_sb_item[lwarpId]]) { //processed sb item, reuse
                                SIN_L(fibing[lwarpId] = false);
                                SIN_L(p_cnt += sb->m_cnts[cur_table][chosen_sb_item[lwarpId]]);
                                SIN_L(
                                        atomicAdd(free_count_1st, sb->m_cnts[cur_table][chosen_sb_item[lwarpId]]));
                                continue;
                            }
                            SIN_L(
                                    my_sb_state[lwarpId] = (sb_state) atomicCAS(
                                            (int *) &sb->m_states[cur_table][chosen_sb_item[lwarpId]],
                                            NOT_PROCESSED, UNDER_PROCESS);
                            ); //try to lock the sb item

                            if (NOT_PROCESSED == my_sb_state[lwarpId]) { //successfully lock the sb item and process it
                                SIN_L(
                                        fibing[lwarpId] = true;
                                        chosen_sb_col[lwarpId] = cur_table);
                            } else { //the sb item is under process, do it with normal WS
                                SIN_L(fibing[lwarpId] = false);
                            }
                        }
                    }
                    __syncwarp();

                    if (0 == lane) {
                        auto hash_val = iRes[lwarpId][hash_tables[0].hash_attr] & (bucketVec[0] - 1);
                        buc_start[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val];
                        buc_end[lwarpId][0] = hash_tables[0].buc_ptrs[hash_val + 1];
                    }
                }
                __syncwarp();
            }
            if (cur_table == num_hash_tables - 1) { //reach the last table
                for (auto j = buc_start[lwarpId][cur_table] + lane; j < buc_end[lwarpId][cur_table]; j += WARP_SIZE) {
                    bool is_chosen = true;
                    if (!single_join_key) {
                        for (auto a = 0; a < s_hash_table_num_attrs[cur_table]; a++) {
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
                        ((single_join_key) &&
                         (iRes[lwarpId][hash_tables[cur_table].hash_attr] == hash_tables[cur_table].hash_keys[j]))) {
                        p_cnt++;

                        if (fib && fibing[lwarpId]) {
                            p_cnt_fib++;
                        }
                    }
                }
                __syncwarp();
                cur_table--;
                continue;
            } else {
                found = probe<DataType, CntType, single_join_key>(
                        iRes[lwarpId], hash_tables[cur_table],
                        buc_start[lwarpId][cur_table], buc_end[lwarpId][cur_table],
                        s_used_for_compare[cur_table], msIdx[lwarpId][cur_table],
                        iterators[lwarpId][cur_table], auc_probe[lwarpId], lane);

                /*update tempCounter*/
                SIN_L(
                        if (work_sharing && found) tempCounter[lwarpId][cur_table] += msIdx[lwarpId][cur_table] + 1);

                /*skew detection*/
                if (work_sharing && (tempCounter[lwarpId][cur_table] > PROCESS_THRESHOLD) &&
                    (buc_end[lwarpId][cur_table] > buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD)) {

                    if (0 == lane) triggerWS[lwarpId] = true;

                    /*have probed BUC_SIZE_THRESHOLD matches, push the rest to task queue*/
                    for (auto l_st = buc_start[lwarpId][cur_table] + lane * BUC_SIZE_THRESHOLD;
                         l_st < buc_end[lwarpId][cur_table];
                         l_st += BUC_SIZE_THRESHOLD * WARP_SIZE) {
                        auto l_en = (l_st + BUC_SIZE_THRESHOLD > buc_end[lwarpId][cur_table]) ?
                                    buc_end[lwarpId][cur_table] : l_st + BUC_SIZE_THRESHOLD;
                        CntType taskId;
                        if (fibing[lwarpId]) {
                            taskId = tb->push_task_fib(iRes[lwarpId], l_st, l_en, cur_table, chosen_sb_col[lwarpId],
                                                       chosen_sb_item[lwarpId]);
                        } else {
                            taskId = tb->push_task(iRes[lwarpId], l_st, l_en, cur_table);
                        }
                        cq->enqueue(taskId);
                    }
                    __syncwarp();

                    /*update the #WS tasks related to the SB item*/
                    if (fibing[lwarpId]) {
                        auto num_new_tasks =
                                (buc_end[lwarpId][cur_table] - buc_start[lwarpId][cur_table] + BUC_SIZE_THRESHOLD - 1) /
                                BUC_SIZE_THRESHOLD;
                        if (sharing[lwarpId]) { //currently processing a WS task
                            atomicAdd(&sb->m_num_tasks_related[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]],
                                      num_new_tasks);
                            __threadfence_block();
                            atomicSub(&sb->m_num_tasks_related[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]],
                                      1); //the parent task
                        } else { //currently processing an original task
                            sb->m_num_tasks_related[chosen_sb_col[lwarpId]][chosen_sb_item[lwarpId]] =
                                    num_new_tasks + 1;//including the parent's work
                        }
                    }
                    __syncwarp();

                    /*no need to probe the rest*/
                    SIN_L (
                            tempCounter[lwarpId][cur_table] = 0;
                            buc_start[lwarpId][cur_table] = buc_end[lwarpId][cur_table]);

                }
                __syncwarp();

                if (!found) { //no match is found
                    if (0 == lane) tempCounter[lwarpId][cur_table] = 0;
                    if (cur_table > start_table) cur_table--;  //backtrack to the last attribute
                    else if (0 == lane)
                        buc_start[lwarpId][start_table] = buc_end[lwarpId][start_table]; //finish this probe item
                    __syncwarp();
                    continue;
                }
            }
        } else if (0 == lane) msIdx[lwarpId][cur_table]--;
        __syncwarp();

        /*write iRes*/
        auto cur_ms = msIdx[lwarpId][cur_table];
        auto curIter = iterators[lwarpId][cur_table][cur_ms]; //msIdx is used here
        auto idx_in_origin_table = hash_tables[cur_table].idx_in_origin[curIter];
        if ((lane < s_hash_table_num_attrs[cur_table]) && (!s_used_for_compare[cur_table][lane]))
            iRes[lwarpId][hash_tables[cur_table].attr_list[lane]] = hash_tables[cur_table].data[lane][idx_in_origin_table];
        __syncwarp();

        SIN_L(
                auto hash_val = iRes[lwarpId][hash_tables[cur_table + 1].hash_attr] & (bucketVec[cur_table + 1] - 1);
                buc_start[lwarpId][cur_table + 1] = hash_tables[cur_table + 1].buc_ptrs[hash_val];
                buc_end[lwarpId][cur_table + 1] = hash_tables[cur_table + 1].buc_ptrs[hash_val + 1];
                msIdx[lwarpId][cur_table + 1] = 0); //init the msIdx for the next attr
        cur_table++; //advance to the next attr
    }
    LOOP_END:
    __syncwarp();

    WARP_REDUCE(p_cnt);
    if (lane == 0) atomicAdd(&l_cnt, p_cnt);
    __syncthreads();

    if (0 == tid) atomicAdd(res_cnt, l_cnt);
}