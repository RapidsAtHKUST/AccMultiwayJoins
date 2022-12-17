/*Copyright (c) 2018 Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#pragma once

#include "common.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define LOCAL_BUCKETS_BITS 10
#define LOCAL_BUCKETS ((1 << LOCAL_BUCKETS_BITS))

/*the number of elements that can be stored in a warp-level buffer during the join materialization*/
#define SHUFFLE_SIZE 16

__global__ void init_metadata_double (
        uint64_t  * __restrict__ heads1,
        uint32_t  * __restrict__ buckets_used1,
        uint32_t  * __restrict__ chains1,
        uint32_t  * __restrict__ out_cnts1,
        uint32_t parts1,
        uint32_t buckets_num1,
        uint64_t  * __restrict__ heads2,
        uint32_t  * __restrict__ buckets_used2,
        uint32_t  * __restrict__ chains2,
        uint32_t  * __restrict__ out_cnts2,
        uint32_t parts2,
        uint32_t buckets_num2
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < buckets_num1; i += blockDim.x*gridDim.x)
        chains1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        out_cnts1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        heads1[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used1 = parts1; //reserve a bucket to each partition at the beginning
    }

    for (int i = tid; i < buckets_num2; i += blockDim.x*gridDim.x)
        chains2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        out_cnts2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        heads2[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used2 = parts2; //reserve a bucket to each partition at the beginning
    }
}

/*
S= keys of data to be partitioned
P= payloads of data to be partitioned
heads= keeps information on first bucket per partition and number of elements in it, packet in one 64-bit integer (only used here)
chains= the successor of a bucket in the bucket list
out_cnts= number of elements per partition
buckets_used= how many buckets are reserved by the partitioning already
offsets= describe the segments that occur due to partitioning
note: multithreaded partitioning creates partitions that consist of contiguous segments
=> iterate over these segments to avoid handling empty slots

output_S= bucketized partitions of data keys
output_P= bucketized partitions of data payloads
cnt= number of elements to partition on total
log_parts- log of number of partitions
first_bit= shift the keys before "hashing"
num_threads= number of threads used in CPU side, used together with offsets

preconditions:
heads: current bucket (1 << 18) [special value for no bucket] and -1 elements (first write allocates bucket)
out_cnts: 0
buckets_used= number of partitions (first num_parts buckets are reserved)
*/
__global__ void partition_pass_one (
        const int32_t   * __restrict__ S,
        const int32_t   * __restrict__ P,
        const size_t    * __restrict__ offsets,
        uint64_t  * __restrict__ heads,
        uint32_t  * __restrict__ buckets_used,
        uint32_t  * __restrict__ chains,
        uint32_t  * __restrict__ out_cnts,
        int32_t   * __restrict__ output_S,
        int32_t   * __restrict__ output_P,
        size_t                   cnt,
        uint32_t                 log_parts,
        uint32_t                 first_bit,
        uint32_t                 num_threads) {
    assert((((size_t) bucket_size) + ((size_t) blockDim.x) * gridDim.x) < (((size_t) 1) << 32));
    const uint32_t parts = 1 << log_parts;
    const int32_t parts_mask = parts - 1;
    uint32_t * router = (uint32_t *) int_shared;
    size_t* shared_offsets = (size_t*) (int_shared + 1024*4 + 4*parts);

    /*if no segmentation in input use one segment with all data, else copy the segment info*/
    if (offsets != NULL) {
        for (int i = threadIdx.x; i < 4*num_threads; i += blockDim.x) shared_offsets[i] = offsets[i];
    } else {
        for (int i = threadIdx.x; i < 4*num_threads; i += blockDim.x) {
            if (i == 1) shared_offsets[i] = cnt;
            else        shared_offsets[i] = 0;
        }
    }
    shared_offsets[4*num_threads] = cnt+4096;
    shared_offsets[4*num_threads+1] = cnt+4096;

    /*partition element counter starts at 0*/
    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x) router[1024*4 + parts + j] = 0;
    if (threadIdx.x == 0) router[0] = 0;
    __syncthreads();

    /*iterate over the segments*/
    for (int u = 0; u < 2*num_threads; u++) {
        size_t segment_start = shared_offsets[2*u];
        size_t segment_limit = shared_offsets[2*u + 1];
        size_t segment_end   = segment_start + ((segment_limit - segment_start + 4096 - 1)/4096)*4096;

        for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x) + segment_start;
             i < segment_end ;
             i += 4 * blockDim.x * gridDim.x) {
//            vec4 thread_vals = *(reinterpret_cast<const vec4 *>(S + i));
            vec4 thread_vals;  //zlai: change to avoid illeagle access
            for(auto k = 0; k < 4; k++)
                if (i+k < cnt) thread_vals.i[k] = S[i+k];

            uint32_t thread_keys[4];

            /*compute local histogram for a chunk of 4*blockDim.x elements*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (i + k < segment_limit){
                    uint32_t partition = (hasht(thread_vals.i[k]) >> first_bit) & parts_mask;
                    atomicAdd(router + (1024 * 4 + parts + partition), 1);
                    thread_keys[k] = partition;
                }
                else thread_keys[k] = 0;
            }
            __syncthreads();

            for (size_t j = threadIdx.x; j < parts ; j += blockDim.x ) {
                uint32_t cnt = router[1024 * 4 + parts + j];
                if (cnt > 0){
                    auto active_mask = __activemask(); //need to identify active threads
                    atomicAdd(out_cnts + j, cnt);
                    uint32_t pcnt, bucket, next_buck;
                    bool repeat = true;
                    while (__any_sync(active_mask, repeat)){
                        if (repeat){
                            /*check if any of the output bucket is filling up*/
                            uint64_t old_heads = atomicAdd(heads + j, ((uint64_t) cnt) << 32);
                            atomicMin(heads + j, ((uint64_t) (2*bucket_size)) << 32);
                            pcnt       = ((uint32_t) (old_heads >> 32));
                            bucket     =  (uint32_t)  old_heads        ;

                            /*now there are two cases:
                            // 2) old_heads.cnt >  bucket_size ( => locked => retry)
                            // if (pcnt       >= bucket_size) continue;*/

                            if (pcnt < bucket_size){
                                /* 1) old_heads.cnt <= bucket_size*/

                                /*check if the bucket was filled*/
                                if (pcnt + cnt >= bucket_size){
                                    if (bucket < (1 << 18)) {
                                        next_buck = atomicAdd(buckets_used, 1);
                                        chains[bucket]     = next_buck;
                                    }
                                    else next_buck = j;
                                    uint64_t tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);
                                    atomicExch(heads + j, tmp);
                                }
                                else next_buck = bucket;
                                repeat = false;
                            }
                        }
                    }
                    router[1024 * 4             + j] = atomicAdd(router, cnt);
                    router[1024 * 4 +     parts + j] = 0;//cnt;//pcnt     ;
                    router[1024 * 4 + 2 * parts + j] = (bucket    << log2_bucket_size) + pcnt;
                    router[1024 * 4 + 3 * parts + j] =  next_buck << log2_bucket_size        ;
                }
            }
            __syncthreads();
            uint32_t total_cnt = router[0];
            __syncthreads();

            /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k)
                if (i + k < segment_limit)
                    thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);

            /*write the keys in shared memory*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k)
                if (i + k < segment_limit)
                    router[thread_keys[k]] = thread_vals.i[k];
            __syncthreads();

            int32_t thread_parts[4];

            /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (threadIdx.x + 1024 * k < total_cnt) {
                    int32_t val = router[threadIdx.x + 1024 * k];
                    uint32_t partition = (hasht(val) >> first_bit) & parts_mask;
                    uint32_t cnt = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);
                    uint32_t bucket = router[1024 * 4 + 2 * parts + partition];
                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                        cnt = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                    bucket += cnt;
                    output_S[bucket] = val;
                    thread_parts[k] = partition;
                }
            }
            __syncthreads();

            /*read payloads of original data*/
//            thread_vals = *(reinterpret_cast<const vec4 *>(P + i));
            for(auto k = 0; k < 4; k++) //zlai: change to avoid illeagle access
                if (i+k < cnt) thread_vals.i[k] = P[i+k];

            /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k)
                if (i + k < segment_limit)
                    router[thread_keys[k]] = thread_vals.i[k];
            __syncthreads();

            /*write payloads to partition buckets in "somewhat coalesced manner"*/
#pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (threadIdx.x + 1024 * k < total_cnt) {
                    int32_t  val       = router[threadIdx.x + 1024 * k];
                    int32_t partition = thread_parts[k];
                    uint32_t cnt       = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);
                    uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];
                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                        cnt = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                    bucket += cnt;
                    output_P[bucket] = val;
                }
            }
            if (threadIdx.x == 0) router[0] = 0;
        }
    }
}

/*
compute information for the second partitioning pass

input:
chains=points to the successor in the bucket list for each bucket (hint: we append new buckets to the end)
out_cnts=count of elements per partition
output:
chains=packed value of element count in bucket and the partition the bucket belongs to
*/
__global__ void compute_bucket_info (uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t cnt = out_cnts[p];

        while (cnt > 0) {
            uint32_t local_cnt = (cnt >= 4096)? 4096 : cnt;
            uint32_t val = (p << 13) + local_cnt;

            uint32_t next = chains[cur];
            chains[cur] = val;

            cur = next;
            cnt -= 4096;
        }
    }
}

__global__ void compute_bucket_info1 (uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts, uint32_t buc_num) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t cnt = out_cnts[p];

        while (cnt > 0) {
            uint32_t local_cnt = (cnt >= 4096)? 4096 : cnt;
            uint32_t val = (p << 13) + local_cnt;

            uint32_t next = chains[cur];
            chains[cur] = val;

            cur = next;
            cnt -= 4096;
        }
    }
}

/*
S= keys of data to be re-partitioned
P= payloads of data to be re-partitioned
heads= keeps information on first bucket per partition and number of elements in it, packet in one 64-bit integer (only used here)
chains= the successor of a bucket in the bucket list
out_cnts= number of elements per partition
buckets_used= how many buckets are reserved by the partitioning already
offsets= describe the segments that occur due to partitioning
note: multithreaded partitioning creates partitions that consist of contiguous segments
=> iterate over these segments to avoid handling empty slots

output_S= bucketized partitions of data keys (results)
output_P= bucketized partitions of data payloads (results)

S_log_parts- log of number of partitions for previous pass
log_parts- log of number of partitions for this pass
first_bit= shift the keys before "hashing"
bucket_num_ptr: number of input buckets

preconditions:
heads: current bucket (1 << 18) [special value for no bucket] and -1 elements (first write allocates bucket)
out_cnts: 0
buckets_used= number of partitions (first num_parts buckets are reserved)
*/
__global__ void partition_pass_two (
        const int32_t   * __restrict__ S,
        const int32_t   * __restrict__ P,
        const uint32_t  * __restrict__ bucket_info,
        uint32_t  * __restrict__ buckets_used,
        uint64_t  *              heads,
        uint32_t  * __restrict__ chains,
        uint32_t  * __restrict__ out_cnts,
        int32_t   * __restrict__ output_S,
        int32_t   * __restrict__ output_P,
        uint32_t                 S_log_parts,
        uint32_t                 log_parts,
        uint32_t                 first_bit,
        uint32_t  *              bucket_num_ptr) {
    assert((((size_t) bucket_size) + ((size_t) blockDim.x) * gridDim.x) < (((size_t) 1) << 32));
    const uint32_t parts = 1 << log_parts;
    const int32_t parts_mask = parts - 1;
    uint32_t buckets_num = *bucket_num_ptr;
    uint32_t * router = (uint32_t *) int_shared; //[1024*4 + parts];

    for (size_t j = threadIdx.x; j < parts ; j += blockDim.x) router[1024*4 + parts + j] = 0;
    if (threadIdx.x == 0) router[0] = 0;
    __syncthreads();

    /*each CUDA block processes a bucket at a time*/
    for (size_t i = blockIdx.x; i < buckets_num; i += gridDim.x) {
        uint32_t info = bucket_info[i];
        uint32_t cnt = info & ((1 << 13) - 1); //number of elements per bucket
        uint32_t pid = info >> 13; //id of original partition

        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(S + bucket_size * i + 4*threadIdx.x));
        uint32_t thread_keys[4];

        /*compute local histogram for the bucket*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt){
                uint32_t partition = (hasht(thread_vals.i[k]) >> first_bit) & parts_mask;
                atomicAdd(router + (1024 * 4 + parts + partition), 1);
                thread_keys[k] = partition;
            }
            else thread_keys[k] = 0;
        }
        __syncthreads();

        for (size_t j = threadIdx.x; j < parts ; j += blockDim.x) {
            uint32_t cnt = router[1024 * 4 + parts + j];
            if (cnt > 0){
                auto active_mask = __activemask(); //need to identify active threads
                atomicAdd(out_cnts + (pid << log_parts) + j, cnt);
                uint32_t pcnt, bucket, next_buck;
                bool repeat = true;
                while (__any_sync(active_mask, repeat)){
                    if (repeat){
                        uint64_t old_heads = atomicAdd(heads + (pid << log_parts) + j, ((uint64_t) cnt) << 32);
                        atomicMin(heads + (pid << log_parts) + j, ((uint64_t) (2*bucket_size)) << 32);
                        pcnt = ((uint32_t)(old_heads >> 32));
                        bucket = (uint32_t)old_heads;

                        if (pcnt < bucket_size){
                            if (pcnt + cnt >= bucket_size){
                                if (bucket < (1 << 18)) {
                                    next_buck = atomicAdd(buckets_used, 1);
                                    chains[bucket]     = next_buck;
                                }
                                else next_buck = (pid << log_parts) + j;
                                uint64_t tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);
                                atomicExch(heads + (pid << log_parts) + j, tmp);
                            }
                            else next_buck = bucket;
                            repeat = false;
                        }
                    }
                }
                router[1024 * 4             + j] = atomicAdd(router, cnt);
                router[1024 * 4 +     parts + j] = 0;
                router[1024 * 4 + 2 * parts + j] = (bucket    << log2_bucket_size) + pcnt;
                router[1024 * 4 + 3 * parts + j] =  next_buck << log2_bucket_size        ;
            }
        }
        __syncthreads();

        uint32_t total_cnt = router[0];
        __syncthreads();

        /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt) thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);
        }

        /*write the keys in shared memory*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k)
            if (4*threadIdx.x + k < cnt)
                router[thread_keys[k]] = thread_vals.i[k];
        __syncthreads();

        int32_t thread_parts[4];

        /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t val = router[threadIdx.x + 1024 * k];
                uint32_t partition = (hasht(val) >> first_bit) & parts_mask;
                uint32_t cnt = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);
                uint32_t bucket = router[1024 * 4 + 2 * parts + partition];
                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                bucket += cnt;
                output_S[bucket] = val;
                thread_parts[k] = partition;
            }
        }
        __syncthreads();

        /*read payloads of original data*/
        thread_vals = *(reinterpret_cast<const vec4 *>(P + i*bucket_size + 4*threadIdx.x));

        /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k)
            if (4*threadIdx.x + k < cnt) router[thread_keys[k]] = thread_vals.i[k];
        __syncthreads();

        /*write payloads to partition buckets in "somewhat coalesced manner"*/
#pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t val = router[threadIdx.x + 1024 * k];
                int32_t partition = thread_parts[k];
                uint32_t cnt = router[1024 * 4 + partition] - (threadIdx.x + 1024 * k);
                uint32_t bucket = router[1024 * 4 + 2 * parts + partition];
                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                bucket += cnt;
                output_P[bucket] = val;
            }
        }
        if (threadIdx.x == 0) router[0] = 0;
    }
}

__global__ void decompose_chains (uint32_t* bucket_info, uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts, int threshold) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t  cnt = out_cnts[p];
        uint32_t first_cnt = (cnt >= threshold)? threshold : cnt;
        int32_t  cutoff = 0;

        while (cnt > 0) {
            cutoff += bucket_size;
            cnt -= bucket_size;

            uint32_t next = chains[cur];

            if (cutoff >= threshold && cnt > 0) {
                uint32_t local_cnt = (cnt >= threshold)? threshold : cnt;

                bucket_info[next] = (p << 15) + local_cnt;
                chains[cur] = 0;
                cutoff = 0;
            } else if (next != 0) {
                bucket_info[next] = 0;
            }
            cur = next;
        }

        bucket_info[p] = (p << 15) + first_cnt;
        assert((bucket_info[p]>> 15) == p);
    }
}

/*kernel for performing the join between the partitioned relations

R,Pr= bucketized keys and payloads for relation R (probe side)
S,Ps= buckerized keys and payloads for relation S (build side)
bucket_info=the info that tells us which partition each bucket belongs to, the number of elements (or whether it belongs to a chain)
S_cnts, S_chain= for build-side we don't pack the info since we operate under the assumption that it is usually one bucket per partition (we don't load balance)
buckets_num=number of buckets for R
results=the memory address where we aggregate
*/
__global__ void join_partitioned_aggregate (
        const int32_t*               R,
        const int32_t*               Pr,
        const uint32_t*              R_chain,
        const uint32_t*              bucket_info,
        const int32_t*               S,
        const int32_t*               Ps,
        const uint32_t*              S_cnts,
        const uint32_t*              S_chain,
        int32_t                      log_parts,
        uint32_t*                    buckets_num,
        int32_t*                     results) {

    /*in order to saze space, we discard the partitioning bits, then we can try fitting keys in int16_t [HACK]*/
    __shared__ int16_t elem[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int pwidth = gridDim.x;

    int count = 0;
    int buckets_cnt = *buckets_num;

    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0) {
            /*unpack information on the subchain*/
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);
            int len_S = S_cnts[p];

            /*S partition doesn't fit in shared memory*/
            int bucket_r_loop = bucket_r;

            /*now we will build a bucket of R side in the shared memory at a time and then probe it with S-side
            sensible because
            1) we have guarantees on size of R from the chain decomposition
            2) this is a skewed scenario so size of S can be arbitrary*/
            for (int offset_r = 0; offset_r < len_R; offset_r += bucket_size) {
                for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                    head[i] = -1;
                __syncthreads();

                /*build a hashtable from an R bucket*/
                for (int base_r = 0; base_r < bucket_size; base_r += 4 * blockDim.x) {
                    vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * bucket_r_loop + base_r + 4 * threadIdx.x));
                    int l_cnt_R = len_R - offset_r - base_r - 4 * threadIdx.x;

#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if (k < l_cnt_R) {
                            int val = data_R.i[k];
                            elem[base_r + k * blockDim.x + tid] = (int16_t) (val
                                    >> (LOCAL_BUCKETS_BITS + log_parts));
                            int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);
                            int32_t last = atomicExch(&head[hval], base_r + k * blockDim.x + tid);
                            next[base_r + k * blockDim.x + tid] = last;
                        }
                    }
                }

                bucket_r_loop = R_chain[bucket_r_loop];

                __syncthreads();

                int bucket_s_loop = p;
                int base_s = 0;

                /*probe hashtable from an S bucket*/
                for (int offset_s = 0; offset_s < len_S; offset_s += 4 * blockDim.x) {
                    vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * bucket_s_loop + base_s +
                                                                   4 * threadIdx.x));
                    int l_cnt_S = len_S - offset_s - 4 * threadIdx.x;

#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int32_t val = data_S.i[k];
                        int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        int32_t hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                        if (k < l_cnt_S) {
                            int32_t pos = head[hval];
                            while (pos >= 0) {
                                if (elem[pos] == tval) {
                                    count++;
                                }
                                pos = next[pos];
                            }
                        }
                    }

                    base_s += 4 * blockDim.x;
                    if (base_s >= bucket_size) {
                        bucket_s_loop = S_chain[bucket_s_loop];
                        base_s = 0;
                    }
                }
                __syncthreads();
            }
        }
    }

    atomicAdd(results, count);
}

/*practically the same as join_partitioned_aggregate
i add extra comments for the materialization technique*/
__global__ void join_partitioned_results (
        const int32_t*               R,
        const int32_t*               Pr,
        const uint32_t*              R_chain,
        const uint32_t*              bucket_info,
        const int32_t*               S,
        const int32_t*               Ps,
        const uint32_t*              S_cnts,
        const uint32_t*              S_chain,
        int32_t                      log_parts,
        uint32_t*                    buckets_num,
        int32_t*                     results,
        int*                     outputIdxR,
        int*                     outputIdxS) {
    __shared__ int16_t elem[4096 + 512];
    __shared__ int32_t payload[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];
    __shared__ uint32_t shuffleIdxR[SHUFFLE_SIZE*32];
    __shared__ uint32_t shuffleIdxS[SHUFFLE_SIZE*32];

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int pwidth = gridDim.x;
    int lid = tid % 32;
    int gid = tid / 32;

    int count = 0;
    int ptr;
    int threadmask = (lid < 31)? ~((1 << (lid+1)) - 1) : 0;
    int shuffle_ptr = 0;

    uint32_t* warp_shuffle_R = shuffleIdxR + gid * SHUFFLE_SIZE;
    uint32_t* warp_shuffle_S = shuffleIdxS + gid * SHUFFLE_SIZE;
    int buckets_cnt = *buckets_num;

    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0) {
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);
            int len_S = S_cnts[p];
            int bucket_r_loop = bucket_r;

            for (int offset_r = 0; offset_r < len_R; offset_r += bucket_size) {
                for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                    head[i] = -1;
                __syncthreads();

                for (int base_r = 0; base_r < bucket_size; base_r += 4*blockDim.x) {
                    vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                    vec4 data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                    int l_cnt_R = len_R - offset_r - base_r - 4 * threadIdx.x;

#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if (k < l_cnt_R) {
                            int val = data_R.i[k];
                            elem[base_r + k*blockDim.x + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                            payload[base_r + k*blockDim.x + tid] = data_Pr.i[k];
                            int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                            int32_t last = atomicExch(&head[hval], base_r + k*blockDim.x + tid);
                            next[base_r + k*blockDim.x + tid] = last;
                        }
                    }
                }

                bucket_r_loop = R_chain[bucket_r_loop];

                __syncthreads();

                int bucket_s_loop = p;
                int base_s = 0;

                for (int offset_s = 0; offset_s < len_S; offset_s += 4*blockDim.x) {
                    vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                    vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                    int l_cnt_S = len_S - offset_s - 4 * threadIdx.x;

#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        uint32_t val = data_S.i[k];
                        uint32_t pval = data_Ps.i[k];
                        int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);
                        int32_t pay;

                        int32_t pos = (k < l_cnt_S)? head[hval] : -1;

                        /*check at warp level whether someone is still following chain => this way we can shuffle without risk*/
                        int pred = (pos >= 0);

                        while (__any_sync(0xffffffff, pred)){
                            int wr_intention = 0;

                            /*we have a match, fetch the data to be written*/
                            if (pred) {
                                if (elem[pos] == tval) {
                                    pay = payload[pos];
                                    wr_intention = 1;
                                    count++;
                                }

                                pos = next[pos];
                                pred = (pos >= 0);
                            }

                            /*find out who had a match in this execution step*/
                            int mask = __ballot_sync(0xffffffff, wr_intention);

                            /*our software managed buffer will overflow, flush it*/
                            int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
                            shuffle_ptr = shuffle_ptr + __popc(mask);

                            /*while it overflows, flush
                            we flush 16 keys and then the 16 corresponding payloads consecutively, of course other formats might be friendlier*/
                            while (shuffle_ptr >= SHUFFLE_SIZE) {
                                if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                                    warp_shuffle_R[wr_offset] = pay;
                                    warp_shuffle_S[wr_offset] = pval;
                                    wr_intention = 0;
                                }

                                if (lid == 0) {
                                    ptr = atomicAdd(results, SHUFFLE_SIZE);
                                }

                                ptr = __shfl_sync(0xffffffff, ptr, 0);

                                /*
                             * zlai: the original instruction writes both keys and payloads
                             * since SHUFFLE_SIZE = 16 = WARP_SIZE/2
                             * To separate the writes of keys and payloads,
                             * we should add if (lid < SHUFFLE_SIZE)
                             * */
//                                output[ptr + lid] = warp_shuffle[lid];
                                if (lid < SHUFFLE_SIZE)
                                {
                                    outputIdxR[ptr + lid] = warp_shuffle_R[lid];
                                    outputIdxS[ptr + lid] = warp_shuffle_S[lid];
                                }

                                wr_offset -= SHUFFLE_SIZE;
                                shuffle_ptr -= SHUFFLE_SIZE;
                            }

                            /*now the fit, write them in buffer*/
                            if (wr_intention && (wr_offset >= 0)) {
                                warp_shuffle_R[wr_offset] = pay;
                                warp_shuffle_S[wr_offset] = pval;
                                wr_intention = 0;
                            }
                        }
                    }

                    base_s += 4*blockDim.x;
                    if (base_s >= bucket_size) {
                        bucket_s_loop = S_chain[bucket_s_loop];
                        base_s = 0;
                    }
                }

                __syncthreads();
            }
        }
    }

    if (lid == 0) {
        ptr = atomicAdd(results, shuffle_ptr);
    }

    ptr = __shfl_sync(0xffffffff, ptr, 0);

    if (lid < shuffle_ptr) {
        outputIdxR[ptr + lid] = warp_shuffle_R[lid];
        outputIdxS[ptr + lid] = warp_shuffle_S[lid];
    }
}

/*partition and compute metadata for relation with key+payload. We use different buffers at the end (it makes sense for UVA based techniques)*/
void prepare_Relation_payload_triple (int* R, int* R_temp, int* R_final, int* P, int* P_temp, int* P_final, size_t RelsNum, uint32_t buckets_num, uint64_t* heads[2], uint32_t* cnts[2], uint32_t* chains[2], uint32_t* buckets_used[2], uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit, cudaStream_t streams, size_t* offsets_GPU, uint32_t num_threads, CUDATimeStat *timing) {

    execKernelDynamicAllocation(init_metadata_double, 64, 1024, 0, timing, false,
                                heads[0], buckets_used[0], chains[0], cnts[0], 1 << log_parts1, buckets_num,
                                heads[1], buckets_used[1], chains[1], cnts[1], 1 << (log_parts1 + log_parts2),
                                buckets_num);

    execKernelDynamicAllocation(partition_pass_one, 64, 1024, (1024*4 + 4*(1 << log_parts1)) * sizeof(int32_t) + (4*num_threads+2)*sizeof(size_t), timing, false,
                                R,
                                P,
                                offsets_GPU,
                                heads[0],
                                buckets_used[0],
                                chains[0],
                                cnts[0],
                                R_temp,
                                P_temp,RelsNum,
                                log_parts1,
                                first_bit + log_parts2,
                                num_threads);

    execKernelDynamicAllocation(compute_bucket_info1, 64, 1024, 0, timing, false,
                                chains[0],
                                cnts[0],
                                log_parts1, buckets_num);

    execKernelDynamicAllocation(partition_pass_two,64, 1024, (1024*4 + 4*(1 << log_parts2)) * sizeof(int32_t) + ((2 * (1 << log_parts2) + 1)* sizeof(int32_t)), timing, false,
                                R_temp,
                                P_temp,chains[0],
                                buckets_used[1],
                                heads[1],
                                chains[1],
                                cnts[1],
                                R_final,
                                P_final,
                                log_parts1,
                                log_parts2,
                                first_bit,
                                buckets_used[0]);
}

