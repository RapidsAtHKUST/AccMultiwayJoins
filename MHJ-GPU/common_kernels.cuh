//
// Created by Bryan on 30/3/2020.
//

#pragma once

#include <iostream>
#include "Relation.cuh"

/* 32 bit Murmur3 hash */
template<typename KeyType, typename CapacityType>
__device__ KeyType murmur_hash(KeyType k, CapacityType hash_capacity) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % hash_capacity;
}

/*
 * With current iRes and iRes_set, probe hash_table to find a batch of matches
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * ms_idx is decreasing
 * */
template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe(
        DataType iRes[MAX_NUM_BUILDTABLES+2],
        HashTable<DataType,CntType> hash_table,
        CntType &buc_start, CntType buc_end,
        bool *used_for_compare,
        char &ms_idx, CntType *iterators,
        CntType &auc, int lane) {
    if (0 == lane) auc = buc_end;
    __syncwarp();

    /*update current iterator and point to the next batch of matches*/
    for(auto i = buc_start + lane; i < auc; i += WARP_SIZE) {
        auto active_come_in = coalesced_threads();
        bool is_chosen = true;
        if (!single_join_key) {
            for(auto a = 0; a < hash_table.num_attrs; a++) {
                if (used_for_compare[a]) {
                    auto origin_idx = hash_table.idx_in_origin[i];
                    if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                        is_chosen = false;
                        break;
                    }
                }
            }
        }
        if (((!single_join_key) && (is_chosen)) ||
            ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
            auto active = coalesced_threads();
            auto rank = active.thread_rank(); /*rank within the active group*/
            iterators[rank] = i; /*can write multiple iterators*/
            if (0 == rank) {
                ms_idx = (char)(active.size()-1);
                buc_start = i + WARP_SIZE - lane;
                auc = 0;
            }
        }
        active_come_in.sync();
    }
    __syncwarp();
    return (auc != buc_end); //a match is found
}

/* probe function where hash tables use open addressing with linear probe */
template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe_open_addr(
        DataType *iRes, HashTable<DataType, CntType> hash_table,
        CntType &buc_start, bool *used_for_compare, char &ms_idx,
        CntType *iterators, int &auc, int lane) {
    if (buc_start == INVALID_BUC_START) return false; //will not probe it buc_start is INVALID
    bool find_match = false;

    /*update current iterator and point to the next batch of matches*/
    auto capacity = hash_table.capacity;

    for(auto i = (buc_start + lane) & (capacity-1); ;
        i = (i + WARP_SIZE) & (capacity-1)) {
        if (hash_table.hash_keys[i] == EMPTY_HASH_SLOT) {
            auto empty_group = coalesced_threads();
            if (0 == empty_group.thread_rank()) { //smallest lane getting empty
                auc = lane;
                buc_start = INVALID_BUC_START; //reach end of this bucket
            }
        }
        __syncwarp();

        /* Four cases here:
         * Case 0:
         *      WARP_SIZE threads come in (auc = WARP_SIZE), but no match is found, continue the for loop
         * Case 1:
         *      WARP_SIZE threads come in (auc = WARP_SIZE), have match,
         *      set (buc_start = i + WARP_SIZE - lane),
         *      set (auc = 0) for break
         * Case 2:
         *      fewer than WARP_SIZE threads come in (auc < WARP_SIZE), no match is found,
         *      buc_start has been set to INVALID_BUC_START, break since (auc < WARP_SIZE)
         * Case 3:
         *      fewer than WARP_SIZE threads come in (auc < WARP_SIZE), have match,
         *      buc_start has been set to INVALID_BUC_START, break since (auc = 0)
         * */
        if (lane < auc) {
            bool is_chosen = true;
            if (!single_join_key) {
                for(auto a = 0; a < hash_table.num_attrs; a++) {
                    if (used_for_compare[a]) {
                        auto origin_idx = hash_table.idx_in_origin[i];
                        if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                            is_chosen = false;
                            break;
                        }
                    }
                }
            }
            if (((!single_join_key) && (is_chosen)) ||
                ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
                auto active = coalesced_threads();
                auto rank = active.thread_rank(); /*rank within the active group*/
                iterators[rank] = i; /*can write multiple iterators*/
                find_match = true;
                if (0 == rank) {
                    ms_idx = (char)(active.size()-1);
                    if (auc == WARP_SIZE) { //not reach the end
                        buc_start = (i + WARP_SIZE - lane) & (capacity-1);
                    }
                    auc = 0;
                }
            }
        }
        __syncwarp();

        if (auc < WARP_SIZE) break;
    }
    __syncwarp();

    if (0 == lane) auc = WARP_SIZE; //reset
    return (__any_sync(0xffffffff, find_match) != 0); //a match is found
}

/* probe function for MHJ */
template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe_tbj(
        DataType iRes[MAX_NUM_RES_ATTRS],
        HashTable<DataType,CntType> hash_table,
        CntType &buc_start, CntType buc_end,
        bool *used_for_compare, CntType &iterator) {
    for(auto i = buc_start; i < buc_end; i ++) {
        bool is_chosen = true;
        if (!single_join_key) {
            for(auto a = 0; a < hash_table.num_attrs; a++) {
                if (used_for_compare[a]) {
                    auto origin_idx = hash_table.idx_in_origin[i];
                    if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                        is_chosen = false;
                        break;
                    }
                }
            }
        }
        if (((!single_join_key) && (is_chosen)) ||
            ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
            iterator = i;
            buc_start = i+1;
            return true; //a match is found
        }
    }
    return false; //no match is found
}

/* probe function for MHJ */
template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe_tbj_open_addr(
        DataType iRes[MAX_NUM_RES_ATTRS],
        HashTable<DataType,CntType> hash_table,
        CntType &buc_start, bool *used_for_compare, CntType &iterator) {
    auto capacity = hash_table.capacity;
    for(auto i = buc_start % capacity; ; i = (i+1)% capacity) {
        if (hash_table.hash_keys[i] == EMPTY_HASH_SLOT) return false;
        bool is_chosen = true;
        if (!single_join_key) {
            for(auto a = 0; a < hash_table.num_attrs; a++) {
                if (used_for_compare[a]) {
                    auto origin_idx = hash_table.idx_in_origin[i];
                    if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                        is_chosen = false;
                        break;
                    }
                }
            }
        }
        if (((!single_join_key) && (is_chosen)) ||
            ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
            iterator = i;
            buc_start = (i+1) % capacity;
            return true; //a match is found
        }
    }
}

/*
 * With current iRes and iRes_set, probe hash_table to find a batch of matches
 * single_join_key: whether the hash table is attached to the iRes with a single join key
 * ms_idx is increasing
 * */
template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe_sbp(
        DataType *iRes,
        HashTable<DataType, CntType> hash_table,
        CntType &buc_start, CntType buc_end,
        bool *used_for_compare,
        char &ms_idx, char &ms_max,
        CntType *iterators,
        CntType &auc, int lane) {
    SIN_L(auc = buc_end);

    /*update current iterator and point to the next batch of matches*/
    for(auto i = buc_start + lane; i < auc; i += WARP_SIZE) {
        auto active_come_in = coalesced_threads();
        bool is_chosen = true;
        if (!single_join_key) {
            for(auto a = 0; a < hash_table.num_attrs; a++) {
                if (used_for_compare[a]) {
                    auto origin_idx = hash_table.idx_in_origin[i];
                    if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                        is_chosen = false;
                        break;
                    }
                }
            }
        }
        if (((!single_join_key) && (is_chosen)) ||
            ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
            auto active = coalesced_threads();
            auto rank = active.thread_rank(); /*rank within the active group*/
            iterators[rank] = i; /*can write multiple iterators*/
            if (0 == rank) {
                ms_max = (char)(active.size()-1);
                ms_idx = 0;
                buc_start = i + WARP_SIZE - lane;
                auc = 0;
            }
        }
        active_come_in.sync();
    }
    __syncwarp();
    return (auc != buc_end); //a match is found
}

template<typename DataType, typename CntType, bool single_join_key>
__device__ bool probe_sbp_open_addr(
        DataType *iRes, HashTable<DataType, CntType> hash_table,
        CntType &buc_start, bool *used_for_compare, char &ms_idx, char &ms_max,
        CntType *iterators, int &auc, int lane) {
    if (buc_start == INVALID_BUC_START) return false; //will not probe it buc_start is INVALID
    bool find_match = false;

    /*update current iterator and point to the next batch of matches*/
    auto capacity = hash_table.capacity;
    for(auto i = (buc_start + lane) & (capacity-1); ;
        i = (i + WARP_SIZE) & (capacity-1)) {
        if (hash_table.hash_keys[i] == EMPTY_HASH_SLOT) {
            auto empty_group = coalesced_threads();
            if (0 == empty_group.thread_rank()) { //smallest lane getting empty
                auc = lane;
                buc_start = INVALID_BUC_START; //reach end of this bucket
            }
        }
        __syncwarp();

        /* Four cases here:
         * Case 0:
         *      WARP_SIZE threads come in (auc = WARP_SIZE), but no match is found, continue the for loop
         * Case 1:
         *      WARP_SIZE threads come in (auc = WARP_SIZE), have match,
         *      set (buc_start = i + WARP_SIZE - lane),
         *      set (auc = 0) for break
         * Case 2:
         *      fewer than WARP_SIZE threads come in (auc < WARP_SIZE), no match is found,
         *      buc_start has been set to INVALID_BUC_START, break since (auc < WARP_SIZE)
         * Case 3:
         *      fewer than WARP_SIZE threads come in (auc < WARP_SIZE), have match,
         *      buc_start has been set to INVALID_BUC_START, break since (auc = 0)
         * */
        if (lane < auc) {
            bool is_chosen = true;
            if (!single_join_key) {
                for(auto a = 0; a < hash_table.num_attrs; a++) {
                    if (used_for_compare[a]) {
                        auto origin_idx = hash_table.idx_in_origin[i];
                        if (iRes[hash_table.attr_list[a]] != hash_table.data[a][origin_idx]) {
                            is_chosen = false;
                            break;
                        }
                    }
                }
            }
            if (((!single_join_key) && (is_chosen)) ||
                ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_keys[i]))) {
                auto active = coalesced_threads();
                auto rank = active.thread_rank(); /*rank within the active group*/
                iterators[rank] = i; /*can write multiple iterators*/
                find_match = true;
                if (0 == rank) {
                    ms_idx = 0;
                    ms_max = (char)(active.size()-1);
                    if (auc == WARP_SIZE) { //not reach the end
                        buc_start = (i + WARP_SIZE - lane) & (capacity-1);
                    }
                    auc = 0;
                }
            }
        }
        __syncwarp();

        if (auc < WARP_SIZE) break;
    }
    __syncwarp();

    if (0 == lane) auc = WARP_SIZE; //reset
    return (__any_sync(0xffffffff, find_match) != 0); //a match is found
}

/* parallel binary search for LFTJ*/
template<typename NeedleType, typename HaystackType, typename CntType>
__global__
void binarySearchForMatchAndIndexes(
        NeedleType *needles, CntType num_needles,        /*needles are unsorted*/
        HaystackType *haystacks, CntType num_haystacks,    /*haystacks are sorted*/
        bool *bitmaps, CntType *matchIdx)
{
    auto gtid = (CntType)(threadIdx.x + blockIdx.x * blockDim.x);
    auto gtnum = (CntType)(blockDim.x * gridDim.x);

    while (gtid < num_needles)
    {
        bool found = false;
        NeedleType needle = needles[gtid];
        int middle, begin = 0, end = num_haystacks-1;
        while (begin <= end)
        {
            middle = begin + (end - begin)/2;
            if (needle > haystacks[middle])
                begin = middle + 1;
            else if (needle < haystacks[middle])
                end = middle - 1;
            else /*found the match*/
            {
                found = true;
                break;
            }
        }
        bitmaps[gtid] = found;
        matchIdx[gtid] = (CntType)middle;
        gtid += gtnum;
    }
}