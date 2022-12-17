//
// Created by Bryan on 15/9/2019.
//

#pragma once
#include <cstdint>

//zlai: the log_parts1 and log_parts2 have to be adapted to the number of inputs
constexpr uint32_t log_parts1 = 8;//9;         //< 12      2^(log_parts1 + log_parts2 + p_d + 5) ~= 'hash table size"  ~= 2 * input size
constexpr uint32_t log_parts2 = 5;//6;//8;      //< 12

constexpr int32_t g_d        = log_parts1 + log_parts2;
constexpr int32_t p_d        = 3;

constexpr int32_t max_chain  = (32 - 1) * 1 - 1; //(32 - 1) * 2 - 1;

#define hj_d (5 + p_d + g_d)

constexpr uint32_t hj_mask = ((1 << hj_d) - 1);

constexpr int32_t partitions = 1 << p_d;
constexpr int32_t partitions_mask = partitions - 1;

constexpr int32_t grid_parts = 1 << g_d;
constexpr int32_t grid_parts_mask = grid_parts - 1;

constexpr uint32_t log2_bucket_size = 12;
constexpr uint32_t bucket_size      = 1 << log2_bucket_size;
constexpr uint32_t bucket_size_mask = bucket_size - 1;

#define OMP_PARALLELISM1 16

