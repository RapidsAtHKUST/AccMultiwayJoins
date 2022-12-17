//
// Created by Bryan on 19/4/2020.
//

#pragma once

#include "cuda/primitives.cuh"
#include "helper.h"
#include "timer.h"
#include "../Relation.cuh"
#include "pretty_print.h"
#include "../Indexing/build_Trie.cuh"
#include "../IndexedTrie.cuh"
#include "../common_kernels.cuh"

#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_load_balance.hxx>
using namespace std;
using namespace mgpu;

/*
 * matchNeedles[i] is 1 if needles[i] finds a match in haystack
 * Both the needles and haystack arrays should be sorted.
 * */
template<typename DataType, typename CntType>
void MGPUFindMatchBoolean(
        DataType *needles, CntType needle_num,
        DataType *haystack, CntType haystack_num,
        bool *matchNeedles, standard_context_t *context,
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    auto asc_begin = thrust::make_counting_iterator((CntType)0);
    auto asc_end = thrust::make_counting_iterator(needle_num);
    CntType *matchIdx = nullptr;
    CUDA_MALLOC(&matchIdx, sizeof(CntType)*needle_num, memstat);

    /*
     * Using MGPU sorted search to compute the lower bound index of each needle in haystack
     * */
    timingKernel(
            sorted_search<bounds_lower>(needles, needle_num, haystack, haystack_num, matchIdx, less_t<DataType>(), *context), timing);
    /*
     * Check whether the value in the lower bound indexed slot is equal to needle
     * */
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, matchNeedles, [=] __device__(CntType idx) {
            return (matchIdx[idx] < haystack_num) && (needles[idx] == haystack[matchIdx[idx]]);
    }), timing); //matchIdx[idx] < haystack_num is necessary
    CUDA_FREE(matchIdx, memstat);
}

/*
 * Joining two unprocessed tables, on the first attr
 * */
template<typename DataType,typename CntType>
void np_join_np(
        IndexedTrie<DataType,CntType> rel_np0, uint32_t rel_np0_index,
        IndexedTrie<DataType,CntType> rel_np1, uint32_t rel_np1_index,
        CntType **res, CntType &res_num,
        standard_context_t *context, CUDAMemStat *memstat, CUDATimeStat *timing) {
    Timer t;
    log_trace("In function: %s", __FUNCTION__);
    auto gpu_time_idx = timing->get_idx();
    bool *match_A = nullptr, *match_B = nullptr;
    CUDA_MALLOC(&match_A, sizeof(bool)*rel_np0.data_len[0], memstat);
    CUDA_MALLOC(&match_B, sizeof(bool)*rel_np1.data_len[0], memstat);

//    log_debug("rel_np0 len = %llu, rel_np1 len = %llu", rel_np0.data_len[0], rel_np1.data_len[0]);
//    log_debug("rel_np0 data range: [%d, %d]", rel_np0.data[0][0], rel_np0.data[0][rel_np0.data_len[0]-1]);
//    log_debug("rel_np1 data range: [%d, %d]", rel_np1.data[0][0], rel_np1.data[0][rel_np1.data_len[0]-1]);
    MGPUFindMatchBoolean(rel_np0.data[0], rel_np0.data_len[0], rel_np1.data[0], rel_np1.data_len[0], match_A, context, memstat, timing);
    MGPUFindMatchBoolean(rel_np1.data[0], rel_np1.data_len[0], rel_np0.data[0], rel_np0.data_len[0], match_B, context, memstat, timing);

    /*construct ascending arrays using Thrust*/
    CntType *asc_A = nullptr, *asc_B = nullptr;
    CUDA_MALLOC(&asc_A, sizeof(CntType)*rel_np0.data_len[0], memstat);
    CUDA_MALLOC(&asc_B, sizeof(CntType)*rel_np1.data_len[0], memstat);

    thrust::counting_iterator<CntType> iter(0);
    timingKernel(thrust::copy(iter, iter + rel_np0.data_len[0], asc_A), timing);
    timingKernel(thrust::copy(iter, iter + rel_np1.data_len[0], asc_B), timing);

    CntType *res_A = nullptr, *res_B = nullptr;
    CUDA_MALLOC(&res_A, sizeof(CntType)*rel_np0.data_len[0], memstat);
    CUDA_MALLOC(&res_B, sizeof(CntType)*rel_np1.data_len[0], memstat);

    auto resNum_A = CUBSelect(asc_A, res_A, match_A, rel_np0.data_len[0], memstat, timing);
    auto resNum_B = CUBSelect(asc_B, res_B, match_B, rel_np1.data_len[0], memstat, timing);
    assert(resNum_A == resNum_B);

#ifdef FREE_DATA
    CUDA_FREE(match_A, memstat);
    CUDA_FREE(match_B, memstat);
    CUDA_FREE(asc_A, memstat);
    CUDA_FREE(asc_B, memstat);
#endif
    res_num= resNum_A;
    res[rel_np0_index] = res_A;
    res[rel_np1_index] = res_B;
    log_info("%s kernel time: %.2f ms.", __FUNCTION__, timing->diff_time(gpu_time_idx));
    log_info("%s CPU time: %.2f ms.", __FUNCTION__, t.elapsed()*1000);
    log_info("New res num: %llu", res_num);
}

/*
     * Joining one processed table and an unprocessed table
     * CSR parent join children, items in the parent CSR are unique
     *
     * The order and order+1 column of res will be populated
     * rel_p: the table whose child column will be joined
     *      level: the offsets[level] of rel_p will be referred for expanding children
     * rel_np: the table whose parent column will be joined (keys are unique)
     * */
template<typename DataType,typename CntType>
void p_join_np(
        IndexedTrie<DataType,CntType> rel_p, uint32_t rel_p_index, int level,
        IndexedTrie<DataType,CntType> rel_np, uint32_t rel_np_index,
        vector<uint32_t> processed_rel_list,
        CntType **res, CntType &res_num,
        standard_context_t *context, CUDAMemStat *memstat, CUDATimeStat *timing) {
    Timer t;
    log_trace("In function: %s", __FUNCTION__);
    auto gpu_time_idx = timing->get_idx();
    CntType *col_parent_p = res[rel_p_index];
    auto num_pars_in_rel_p = res_num; //number of parent items in rel_p

    /*1st: get children count of rel_p*/
    int *children_cnts_in_rel_p;
    CUDA_MALLOC(&children_cnts_in_rel_p, sizeof(int)*num_pars_in_rel_p, memstat);

    auto asc_begin = thrust::make_counting_iterator((CntType)0);
    auto asc_end = thrust::make_counting_iterator(num_pars_in_rel_p);

    /*2nd: extract the number of children for each parent item in rel_p*/
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, children_cnts_in_rel_p, [=] __device__(CntType idx) {
            return rel_p.offsets[level][col_parent_p[idx]+1] - rel_p.offsets[level][col_parent_p[idx]];
    }), timing);

    unsigned long long sum_p = 0;
    float sum_p_in_GB;
    sum_p = CUBSum<int,unsigned long long,CntType>(children_cnts_in_rel_p, num_pars_in_rel_p, memstat, timing);
    sum_p_in_GB = sum_p * sizeof(int) * 1.0f / 1024/1024/1024;
    log_info("p expanded children num: %lu (%.2f GB)", sum_p, sum_p_in_GB);
    if (sum_p_in_GB > 32) {
        log_error("Counter overflow. Exit");
        exit(1);
    }
    CntType num_children_in_rel_p = CUBScanExclusive(children_cnts_in_rel_p, children_cnts_in_rel_p, num_pars_in_rel_p, memstat, timing);

    int *lbs = nullptr; //load-balancing search res, to store the expanded children
    CUDA_MALLOC(&lbs, sizeof(int)*num_children_in_rel_p, memstat);

    /*3rd: load balance search*/
    timingKernel(load_balance_search(num_children_in_rel_p, children_cnts_in_rel_p, num_pars_in_rel_p, lbs, *context), timing);
    log_trace("finish load_balance_search");

    /*4th: compute the filtered children values of relation a when intersecting with b*/
    DataType *expanded_children_val = nullptr;
    CntType *expanded_children_ptr = nullptr;
    CUDA_MALLOC(&expanded_children_val, sizeof(DataType)*num_children_in_rel_p, memstat);
    CUDA_MALLOC(&expanded_children_ptr, sizeof(CntType)*num_children_in_rel_p, memstat);

    /*populate the expanded children ptrs*/
    asc_end = thrust::make_counting_iterator(num_children_in_rel_p);
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_ptr, [=] __device__(CntType idx) {
            auto rank = idx - children_cnts_in_rel_p[lbs[idx]];
            auto startPos = rel_p.offsets[level][col_parent_p[lbs[idx]]];
            return rank + startPos;
    }), timing);

    /*populate the expanded children vals*/
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_val, [=] __device__(CntType idx) {
            return rel_p.data[level+1][expanded_children_ptr[idx]];
    }), timing);
#ifdef FREE_DATA
    CUDA_FREE(children_cnts_in_rel_p, memstat);
#endif

    /*5th: compute the bitmap and matching indexes using binary searches*/
    bool *bitmap = nullptr;
    CntType *match_idx = nullptr;
    CUDA_MALLOC(&bitmap, sizeof(bool)*num_children_in_rel_p, memstat);
    CUDA_MALLOC(&match_idx, sizeof(CntType)*num_children_in_rel_p, memstat);
    /*binary search solution*/

    log_debug("rel_np.data_len[0]=%llu", rel_np.data_len[0]);
    execKernel(binarySearchForMatchAndIndexes, GRID_SIZE, BLOCK_SIZE, timing, false, expanded_children_val, num_children_in_rel_p, rel_np.data[0], rel_np.data_len[0], bitmap, match_idx);
#ifdef FREE_DATA
    CUDA_FREE(expanded_children_val, memstat);
#endif
    /*compute the number of matches*/
    auto num_matches = CUBSum<bool,CntType,CntType>(bitmap, num_children_in_rel_p, memstat, timing);
    if (0 == num_matches)    log_info("No matches!");

    /*6th: update the res of each relevant relation*/
    /*1)update all the previous relations that are joined with the processed relation before
     * Approach: get the filtered parents (lbs+bitmap) of the processed relation and update the prev_res of those previous relations
     * */
    int *selected_lbs = nullptr;
    CUDA_MALLOC(&selected_lbs, sizeof(int)*num_matches, memstat);
    CUBSelect(lbs, selected_lbs, bitmap, num_children_in_rel_p, memstat, timing);
#ifdef FREE_DATA
    CUDA_FREE(lbs, memstat);
#endif
    /*construct new result and replace the old one for each previously processed rel*/
    for(auto i = 0; i < processed_rel_list.size(); i++) {
        auto processed_rel_index = processed_rel_list[i];
        CntType *new_prev = nullptr;
        CUDA_MALLOC(&new_prev, sizeof(CntType)*num_matches, memstat);
        execKernel(gather, GRID_SIZE, BLOCK_SIZE, timing, false, res[processed_rel_index], new_prev, selected_lbs, num_matches);
#ifdef FREE_DATA
        CUDA_FREE(res[processed_rel_index], memstat);
#endif
        res[processed_rel_index] = new_prev;
    }
#ifdef FREE_DATA
    CUDA_FREE(selected_lbs, memstat);
#endif
    /*2)update the rel_p relation in this function (res indicates its children)
     * Approach: select the children ptr, flag = bitmap
     * */
    CntType *new_col_for_C = nullptr;
    CUDA_MALLOC(&new_col_for_C, sizeof(CntType)*num_matches, memstat);
    CUBSelect(expanded_children_ptr, new_col_for_C, bitmap, num_children_in_rel_p, memstat, timing);
#ifdef FREE_DATA
    CUDA_FREE(expanded_children_ptr, memstat);
        CUDA_FREE(res[rel_p_index], memstat);
#endif
    res[rel_p_index] = new_col_for_C;

    /*3)update the rel_np relation in this function (res indicates its parents) and append it to the final result list
     * Approach: select the match indexes, flag = bitmap
     * */
    CntType *new_col_for_P = nullptr;
    CUDA_MALLOC(&new_col_for_P, sizeof(CntType)*num_matches, memstat);
    CUBSelect(match_idx, new_col_for_P, bitmap, num_children_in_rel_p, memstat, timing);
#ifdef FREE_DATA
    CUDA_FREE(match_idx, memstat);
        CUDA_FREE(bitmap, memstat);
#endif
    res[rel_np_index] = new_col_for_P;
    res_num = num_matches; //update the result count

    log_info("%s kernel time: %.2f ms.", __FUNCTION__, timing->diff_time(gpu_time_idx));
    log_info("%s CPU time: %.2f ms.", __FUNCTION__, t.elapsed()*1000);
    log_info("New res num: %llu", res_num);
}

template<typename DataType, typename CntType>
void expandEndPoint(
        IndexedTrie<DataType,CntType> trie, uint32_t num_tables,
        CntType &num_res, CntType **res, uint32_t trie_idx,
        standard_context_t *context, CUDAMemStat *memstat, CUDATimeStat *timing) {
    timing->reset();
    auto *res_prev = res[trie_idx];

    /*1st: get children count of relation p*/
    int *children_cnts = nullptr;
    CUDA_MALLOC(&children_cnts, sizeof(int)*num_res, memstat);

    auto asc_begin = thrust::make_counting_iterator((CntType)0);
    auto asc_end = thrust::make_counting_iterator(num_res);

    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, children_cnts, [=] __device__(CntType idx) {
            return trie.offsets[0][res_prev[idx]+1] - trie.offsets[0][res_prev[idx]]; //todo: offset[0]
    }), timing);

    auto total_num_children = CUBScanExclusive(children_cnts, children_cnts, num_res, memstat, timing);

    int *lbs = nullptr;
    CUDA_MALLOC(&lbs, sizeof(int)*total_num_children, memstat);

    timingKernel(
            load_balance_search(total_num_children, children_cnts, num_res, lbs, *context), timing);

    /*4th: compute the filtered children values of relation p when intersecting with relation np*/
    CntType *expanded_children_ptr = nullptr;
    CUDA_MALLOC(&expanded_children_ptr, sizeof(CntType)*total_num_children, memstat);

    asc_end = thrust::make_counting_iterator((CntType)total_num_children);
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_ptr, [=] __device__(CntType idx) {
            auto rank = idx - children_cnts[lbs[idx]];
            auto startPos = trie.offsets[0][res_prev[lbs[idx]]]; //todo: offset[0]
            return rank + startPos;
    }), timing);
    CUDA_FREE(children_cnts, memstat);

    /*5th update all the results of relations except itself*/
    for(int i = 0; i < num_tables; i++) {
        /*dont update itself*/
        if (i == trie_idx) continue;

        /*construct new result*/
        CntType *newCol = nullptr;
        CUDA_MALLOC(&newCol, sizeof(CntType)*total_num_children, memstat);

        auto *oldCol = res[i];
        execKernel(gather, GRID_SIZE, BLOCK_SIZE, timing, false, oldCol, newCol, lbs, total_num_children);
        CUDA_FREE(oldCol, memstat);
        res[i] = newCol;
    }
    CUDA_FREE(lbs, memstat);
    CUDA_FREE(res[trie_idx], memstat);
    res[trie_idx] = expanded_children_ptr;

    num_res = total_num_children;
    log_info("%s kernel time: %.2f ms.", __FUNCTION__, timing->elapsed());
}

/*for Q3*/
//#define NUM_OUTPUT_ATTR (4)
//AttrType attr_order[] = {1,2,3,0};

/*for Q8*/
#define NUM_OUTPUT_ATTR (8)
AttrType attr_order[] = {0,1,2,3,4,5,7,6}; //sp

bool comp(const pair<AttrType,uint32_t> &a, const pair<AttrType,uint32_t> &b) {
    return precede(attr_order, NUM_OUTPUT_ATTR, a.first, b.first);
}

/*todo: do not materialize the full results, only the indexes*/
template<typename DataType,typename CntType>
void LFTJ(IndexedTrie<DataType,CntType> *Tries, uint32_t num_tables,
          CUDAMemStat *memstat, CUDATimeStat *timing) {
    standard_context_t context(false);

    CntType res_num = 0;
    CntType **res_idxes;
    CUDA_MALLOC(&res_idxes, sizeof(CntType*)*num_tables, memstat);

    /*general chain join */
    np_join_np(Tries[0], 0, Tries[1], 1, res_idxes, res_num, &context, memstat, timing);
    if (0 == res_num) return;
    p_join_np(Tries[1], 1, 0, Tries[2], 2, {0}, res_idxes, res_num, &context, memstat, timing);
    if (0 == res_num) return;
    if (num_tables >= 4){
        p_join_np(Tries[2], 2, 0, Tries[3], 3, {0,1}, res_idxes, res_num, &context, memstat, timing);
        if (0 == res_num) return;
    }
    if (num_tables >= 5) {
        p_join_np(Tries[3], 3, 0, Tries[4], 4, {0,1,2}, res_idxes, res_num, &context, memstat, timing);
        if (0 == res_num) return;
    }
    if (num_tables >= 6) {
        p_join_np(Tries[4], 4, 0, Tries[5], 5, {0,1,2,3}, res_idxes, res_num, &context, memstat, timing);
        if (0 == res_num) return;
    }
    expandEndPoint(Tries[0], num_tables, res_num, res_idxes, 0, &context, memstat, timing);
    log_trace("Finished expandEndPoint 0.");
    expandEndPoint(Tries[num_tables-1], num_tables, res_num, res_idxes, num_tables-1, &context, memstat, timing);
    log_trace("Finished expandEndPoint %d.", num_tables-1);

    /*
     * Modifications for specific TPC-H query
     * 1. Revise the following execution scheme
     * 2. Revise the Trie construction function in cuda_lftj.cu
     * 3. Revise the attr_order in above
     * */

    /*BFS-LFTJ evaluation for Q3_sp and lp*/
//    log_trace("---------------- orders Join customer ----------------");
//    np_join_np(Tries[0], 0, Tries[1], 1, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- lineitem Join orders ----------------");
//    p_join_np(Tries[1], 1, 0, Tries[2], 2, {0}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    expandEndPoint(Tries[0], num_tables, res_num, res_idxes, 0, &context, memstat, timing);
//    log_trace("Finished expandEndPoint 0.");
//    expandEndPoint(Tries[num_tables-1], num_tables, res_num, res_idxes, num_tables-1, &context, memstat, timing);
//    log_trace("Finished expandEndPoint %d.", num_tables-1);

    /*BFS-LFTJ evaluation for Q8_lp*/
//    log_trace("---------------- Join part ----------------");
//    np_join_np(Tries[0], 0, Tries[1], 1, res_idxes, res_num, &context, memstat, timing); //Lineitem join Part
//    if (0 == res_num) return;
//    log_trace("---------------- Join order ----------------");
//    p_join_np(Tries[0], 0, 0, Tries[2], 2, {1}, res_idxes, res_num, &context, memstat, timing);//join order
//    if (0 == res_num) return;
//    log_trace("---------------- Join customer ----------------");
//    p_join_np(Tries[2], 2, 0, Tries[3], 3, {0,1}, res_idxes, res_num, &context, memstat, timing);//join customer
//    if (0 == res_num) return;
//    log_trace("---------------- Join nation_0 ----------------");
//    p_join_np(Tries[3], 3, 0, Tries[4], 4, {0,1,2}, res_idxes, res_num, &context, memstat, timing);//join nation
//    if (0 == res_num) return;
//    log_trace("---------------- Join region ----------------");
//    p_join_np(Tries[4], 4, 0, Tries[5], 5, {0,1,2,3}, res_idxes, res_num, &context, memstat, timing);//join region
//    if (0 == res_num) return;
//    log_trace("---------------- Join supplier ----------------");
//    p_join_np(Tries[0], 0, 1, Tries[6], 6, {1,2,3,4,5}, res_idxes, res_num, &context, memstat, timing);//join supplier
//    if (0 == res_num) return;
//    log_trace("---------------- Join nation_1 ----------------");
//    p_join_np(Tries[6], 6, 0, Tries[7], 7, {0,1,2,3,4,5}, res_idxes, res_num, &context, memstat, timing);//join nation
//    if (0 == res_num) return;
//    expandEndPoint(Tries[1], num_tables, res_num, res_idxes, 1, &context, memstat, timing);
//    log_trace("Finished expandEndPoint table 1.");

    /*BFS-LFTJ evaluation for Q8_sp*/
//    log_trace("---------------- region Join nation_0 ----------------");
//    np_join_np(Tries[0], 0, Tries[1], 1, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- nation_0 Join customer ----------------");
//    p_join_np(Tries[1], 1, 0, Tries[2], 2, {0}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- customer Join orders ----------------");
//    p_join_np(Tries[2], 2, 0, Tries[3], 3, {0,1}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- orders Join lineitem ----------------");
//    p_join_np(Tries[3], 3, 0, Tries[4], 4, {0,1,2}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- lineitem Join part ----------------");
//    p_join_np(Tries[4], 4, 0, Tries[5], 5, {0,1,2,3}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- lineitem Join supplier ----------------");
//    p_join_np(Tries[4], 4, 1, Tries[6], 6, {0,1,2,3,5}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    log_trace("---------------- supplier Join nation_1 ----------------");
//    p_join_np(Tries[6], 6, 0, Tries[7], 7, {0,1,2,3,4,5}, res_idxes, res_num, &context, memstat, timing);
//    if (0 == res_num) return;
//    expandEndPoint(Tries[5], num_tables, res_num, res_idxes, 5, &context, memstat, timing);
//    log_trace("Finished expandEndPoint table 5.");


    log_info("Output count: %llu", res_num);
}

#undef NUM_OUTPUT_ATTR