//
// Created by Bryan on 30/7/2019.
//
#pragma once

#include <moderngpu/kernel_load_balance.hxx>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>

#include "../types.h"
#include "../conf.h"
#include "cuda/primitives.cuh"
#include "cuda/sharedMem.cuh"
#include "helper.h"

std::map<uint32_t, vector<uint32_t>> default_bucket_settings = {
        {1, {1}},
        {2, {2}},
        {4, {4}},
        {8, {8}},
        {16, {16}},
        {32, {32}},
        {64, {64}},
        {128, {32,4}},
        {256, {32,8}},
        {512, {32,16}},
        {1024, {32,32}},
        {2048, {32,64}},
        {4096, {32,32,4}},
        {8192, {32,64,4}},
        {16384, {32,64,8}},
        {32768, {32,64,16}},
        {65536, {16,512,8}},
        {131072, {16,512,16}},
        {262144, {32,64,16,8}},
        {524288, {32,64,16,16}},
        {1048576, {32,64,16,32}},
        {2097152, {32,64,16,64}},
        {4194304, {32,64,16,128}},
        {8388608, {32,64,16,256}},
        {16777216, {32,64,16,512}},
        {33554432, {32,64,16,1024}},
        {67108864, {32,64,16,2048}},
        {134217728, {32,64,16,4096}},
        {268435456, {32,16,16,16,2048}},
        {536870912, {32,16,16,16,4096}}
};

template<typename DataType, typename CntType>
__global__
void buildBucPtrs(
        DataType *data, CntType len,
        uint32_t buckets, CntType *startPos,
        CntType offset = 0) {
    CntType gtid = (CntType)(threadIdx.x + blockDim.x * blockIdx.x);
    CntType gtnum = (CntType)(blockDim.x * gridDim.x);
    uint32_t mask = buckets-1;

    for(auto i = gtid; i < len-1; i += gtnum) {
        uint32_t b_f = data[i] & mask;
        uint32_t b_l = data[i+1] & mask;

        /*write up those empty buckets in the middle*/
        for(uint32_t j = b_f+1; j <= b_l; j++)
            startPos[j] = i+1 + offset;
    }

    /*write up those poses for empty buckets in the end*/
    auto buc_last = data[len-1] & mask;
    for(int i = buc_last+gtid+1; i <= buckets; i += gtnum)
        startPos[i] = len + offset;

    auto buc_first = data[0] & mask;
    for(int i = buc_first-gtid; i >= 0; i -= gtnum)
        startPos[i] = offset;
}

/*
 * Device-wise histogram primitive, all the thread blocks cooperate to compute the global prefix sum
 * */
template <typename DataType, typename CntType, typename HashFuncType>
__global__
void histogramDeviceWise(
        const DataType *keys,
        CntType length,
        CntType *hist,
        const uint32_t buckets,
        uint32_t offsetBits,
        HashFuncType f) {
    SharedMemory<CntType> smem;
    CntType *localBucPtrs = smem.getPointer();

    auto tid = threadIdx.x, tnum = blockDim.x;
    auto bid = blockIdx.x, bnum = gridDim.x;
    auto gtid = (CntType)(tid + bid * tnum);
    auto gtnum = (CntType)(tnum * bnum);
    uint32_t hashValue, mask = buckets - 1; /*numBuckets is 2^n*/

    /*local buckets initialization*/
    for(auto i = tid; i < buckets; i += tnum) localBucPtrs[i] = 0;
    __syncthreads();

    for(auto i = gtid; i < length; i += gtnum) {
        hashValue = f(keys[i],offsetBits,mask);
        atomicAdd(&localBucPtrs[hashValue], 1);
    }
    __syncthreads();

    for(auto i = tid; i < buckets; i += tnum)
        hist[i*bnum+bid] = localBucPtrs[i];
}

/*
 * Block-wise histogram primitive, each thread block computes the prefix sum of a bucket
 * */
template <typename DataType, typename CntType, typename HashFuncType>
__global__
void histogramBlockWise(
        const DataType *keys,
        CntType *hist,
        const uint32_t buckets_last_passes,  //#blocks generated in the first pass
        const uint32_t buckets_this_pass, //#subblocks split in this pass
        CntType *buc_ptrs_last_passes,               //global bucket ptrs
        uint32_t offsetBits,
        HashFuncType f) {
    SharedMemory<CntType> smem;
    CntType *localBucPtrs = smem.getPointer();

    auto tid = threadIdx.x, tnum = blockDim.x;
    auto bid = blockIdx.x;
    uint32_t mask = buckets_this_pass - 1; /*numBuckets is 2^n*/

    /*local buckets initialization*/
    for(auto i = tid; i < buckets_this_pass; i += tnum) localBucPtrs[i] = 0;
    __syncthreads();

    for(auto i = buc_ptrs_last_passes[bid] + tid; i < buc_ptrs_last_passes[bid+1]; i += tnum) {
        auto hash_value = f(keys[i],offsetBits,mask);
        atomicAdd(&localBucPtrs[hash_value], 1);
    }
    __syncthreads();

    for(auto i = tid; i < buckets_this_pass; i += tnum)
        hist[bid*buckets_this_pass+i] = localBucPtrs[i];
}

/*
 * Device-wise shuffle primitive, all the thread blocks cooperate to shuffle the data according to the global histogram
 * Will generate indexes
 * */
template <typename KType, typename VType, typename CntType, typename HashFuncType>
__global__
void shuffleDeviceWise(
        KType *keysIn, KType *keysOut,
        VType *valuesIn, VType *valuesOut,
        CntType *histScanned, const CntType length, const uint32_t buckets,
        uint32_t offsetBits, HashFuncType f) {
    SharedMemory<CntType> smem;
    CntType *localBucPtrs = smem.getPointer();

    auto tid = threadIdx.x, tnum = blockDim.x;
    auto bid = blockIdx.x, bnum = gridDim.x;
    auto gtid = (CntType)(tid + bid * tnum);
    auto gtnum = (CntType)(tnum * bnum);
    uint32_t hashValue, mask = buckets - 1;

    /*load the unscanned histogram*/
    for(auto i = tid; i < buckets; i += tnum)
        localBucPtrs[i] = histScanned[i * bnum + bid];
    __syncthreads();

    if ((!valuesIn) && (!valuesOut)) { //key-only
        for(CntType i = gtid; i < length; i += gtnum) {
            hashValue = f(keysIn[i],offsetBits,mask);
            auto cur = atomicAdd(&localBucPtrs[hashValue], 1);
            keysOut[cur] = keysIn[i];
        }
    }
    else if (!valuesIn) { //key-index
        for(CntType i = gtid; i < length; i += gtnum) {
            hashValue = f(keysIn[i],offsetBits,mask);
            auto cur = atomicAdd(&localBucPtrs[hashValue], 1);
            keysOut[cur] = keysIn[i];
            valuesOut[cur] = (VType)i;
        }
    }
    else {  //key-value
        for(CntType i = gtid; i < length; i += gtnum) {
            hashValue = f(keysIn[i],offsetBits,mask);
            auto cur = atomicAdd(&localBucPtrs[hashValue], 1);
            keysOut[cur] = keysIn[i];
            valuesOut[cur] = valuesIn[i];
        }
    }

}

/*
 * Block-wise shuffle primitive, each thread block shuffles the data according to the block-wise histogram
 * */
template <typename KType, typename CntType, typename VType, typename HashFuncType>
__global__
void shuffleBlockWise(
        KType *keys_in, KType *keys_out,
        VType *values_in, VType *values_out,
        CntType *hist_scanned,     /*block-wise scanned histogram */
        const uint32_t buckets_last_passes,  //#blocks generated in the first pass
        const uint32_t buckets_this_pass, //#subblocks split in this pass
        CntType *buc_ptrs_last_passes,
        CntType *buc_ptrs_this_pass,               //global bucket ptrs
        uint32_t offset_bits,
        HashFuncType f) {
    SharedMemory<CntType> smem;
    CntType *localBucPtrs = smem.getPointer();

    auto tid = threadIdx.x, tnum = blockDim.x;
    auto bid = blockIdx.x;
    uint32_t hashValue, mask = buckets_this_pass - 1;
    auto offset = buc_ptrs_last_passes[bid];

    /*load the scanned histogram*/
    for(auto i = tid; i < buckets_this_pass; i += tnum)
        localBucPtrs[i] = hist_scanned[bid*buckets_this_pass+i];
    __syncthreads();

    if ((!values_in) && (!values_out)) { //key-only
        for(auto i = offset + tid; i < buc_ptrs_last_passes[bid+1]; i += tnum) {
            hashValue = f(keys_in[i],offset_bits,mask);
            auto cur = atomicAdd(&localBucPtrs[hashValue], 1);
            keys_out[cur+offset] = keys_in[i];
        }
    }
    else { //key-index or key-value
        for(auto i = offset + tid; i < buc_ptrs_last_passes[bid+1]; i += tnum) {
            hashValue = f(keys_in[i],offset_bits,mask);
            auto cur = atomicAdd(&localBucPtrs[hashValue], 1);
            keys_out[cur+offset] = keys_in[i];
            values_out[cur+offset] = values_in[i];
        }
    }
    __syncthreads();

    /*update the buc_ptrs_this_pass*/
    for(auto i = bid*buckets_this_pass + tid; i < (bid + 1)*buckets_this_pass; i += tnum) {
        buc_ptrs_this_pass[i] = hist_scanned[i] + offset;
    }
}

template<typename KType, typename VType, typename CntType>
__global__ void lookup_KV_device_wise(
        KType *hash_keys, VType *hash_values, CntType capacity, CntType *buc_ptrs,
        KType *lookup_keys, VType *lookup_values, CntType num) {
    auto gtid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gtnum = blockDim.x * gridDim.x;
    auto gwnum = gtnum >> 5;
    auto gwid = gtid >> 5;
    auto lane = threadIdx.x & 31;

    auto start_id = num / gwnum * gwid;
    for(auto i = start_id; i < start_id + WARP_SIZE; i++) {
        auto key = lookup_keys[i];
        auto buc = key & (capacity - 1);
        for(auto j = buc_ptrs[buc] + lane; j < buc_ptrs[buc+1]; j += WARP_SIZE) {
            if (key == hash_keys[j]) {
                lookup_values[i] = hash_values[j];
            }
        }
    }

//    if(gtid < num) {
//        auto key = lookup_keys[gtid];
//        auto buc = key & (capacity - 1);
////        lookup_values[gtid] = hash_values[buc];
//
//        for(auto j = buc; j < buc+1; j ++) {
//            if (key == hash_keys[j]) {
//                lookup_values[gtid] = hash_values[j];
//                break;
//            }
//        }
//    }
}

/*----------------------- HOST FUNCTIONS -------------------------*/
template<typename KType, typename VType, typename CntType>
class RadixPartitioner {
    int _num_passes;
    CntType _data_len;
    vector<uint32_t> _bucket_setting;
    uint32_t _total_buckets;

    /*intermediate memory objets*/
    KType *_keys_inter;
    VType *_values_inter;
    CntType *_hists_inter;
    CntType *_buc_ptrs_inter;
    int *_seg_scan_flags;
    int *_partitioner;

    /*profiling structures*/
    uint32_t _max_bucket_per_pass;
    CUDAMemStat *_memstat;
    CUDATimeStat *_timing;

    void profile() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        _max_bucket_per_pass = (uint32_t)(prop.sharedMemPerBlock / sizeof(CntType));
        log_info("Maximum buckets per pass: %llu", _max_bucket_per_pass);
    }
    void mem_allocate() {
        if (_num_passes > 1) {
            CUDA_MALLOC(&_keys_inter, sizeof(KType)*_data_len, _memstat);
            CUDA_MALLOC(&_values_inter, sizeof(VType)*_data_len, _memstat);
            CUDA_MALLOC(&_hists_inter, sizeof(CntType)*_total_buckets, _memstat);
            CUDA_MALLOC(&_buc_ptrs_inter, sizeof(CntType)*(_total_buckets+1), _memstat);
            CUDA_MALLOC(&_seg_scan_flags, sizeof(int)*_total_buckets, _memstat);
            CUDA_MALLOC(&_partitioner, sizeof(int)*_total_buckets, _memstat);
        }
    }
    void pass_planner_equal() {
        auto total_bits = logFunc(_total_buckets);
        auto bits_per_pass = (total_bits + _num_passes - 1 ) / _num_passes;
        for(auto i = 0; i < _num_passes - 1; i++) {
            uint32_t buckets_this_pass = (uint32_t)(pow(2, bits_per_pass));
            if (buckets_this_pass > _max_bucket_per_pass) {
                log_error("Too many buckets in a pass (%llu > %llu)", buckets_this_pass, _max_bucket_per_pass);
                exit(1);
            }
            _bucket_setting.emplace_back(buckets_this_pass);
        }
        auto bits_final_pass = total_bits - bits_per_pass * (_num_passes-1);
        _bucket_setting.emplace_back((uint32_t)(pow(2, bits_final_pass))); //last pass
    }
    void pass_planner_default() {
        auto set_entry = default_bucket_settings.find(_total_buckets);
        if (set_entry == default_bucket_settings.end()) {
            log_error("No default setting for thie bucket number (%llu)", _total_buckets);
            exit(1);
        }
        _bucket_setting = set_entry->second;
        _num_passes = _bucket_setting.size();
    }
public:
    /*constructor with total number of buckets set*/
    RadixPartitioner(CntType data_len, int num_passes, uint32_t total_buckets,
                     CUDAMemStat *memstat, CUDATimeStat *timing) {
        profile();
        this->_num_passes = num_passes;
        this->_data_len = data_len;
        this->_memstat = memstat;
        this->_timing = timing;
        this->_total_buckets = total_buckets;
        if (!isPowerOf2(_total_buckets)) {
            log_error("The number of buckets (%d) is not a power of 2.", _total_buckets);
            exit(1);
        }
        pass_planner_equal(); //set the number of buckets in each pass
        mem_allocate();
    }
    /*constructor with total number of buckets and default setting*/
    RadixPartitioner(CntType data_len, uint32_t total_buckets,
                     CUDAMemStat *memstat, CUDATimeStat *timing) {
        profile();
        this->_data_len = data_len;
        this->_memstat = memstat;
        this->_timing = timing;
        this->_total_buckets = total_buckets;
        if (!isPowerOf2(_total_buckets)) {
            log_error("The number of buckets (%d) is not a power of 2.", _total_buckets);
            exit(1);
        }
        pass_planner_default();
        mem_allocate();
    }
    /*constructor with bucket settings set*/
    RadixPartitioner(CntType data_len, vector<uint32_t> bucket_setting,
                     CUDAMemStat *memstat, CUDATimeStat *timing) {
        profile();
        this->_num_passes = (int)bucket_setting.size();
        this->_bucket_setting = bucket_setting;
        this->_data_len = data_len;
        this->_memstat = memstat;
        this->_timing = timing;
        this->_total_buckets = 1;
        for(auto i = 0; i < _num_passes; i++) {
            if (!isPowerOf2(this->_bucket_setting[i])) {
                log_error("The number of buckets (%d) is not a power of 2.", this->_bucket_setting[i]);
                exit(1);
            }
            if (this->_bucket_setting[i] > _max_bucket_per_pass) {
                log_error("Too many buckets in a pass (%llu > %llu)", this->_bucket_setting[i], _max_bucket_per_pass);
                exit(1);
            }
            this->_total_buckets *= this->_bucket_setting[i];
        }
        mem_allocate();
    }

    ~RadixPartitioner() {
        if (this->_num_passes > 1) {
            CUDA_FREE(this->_keys_inter, this->_memstat);
            CUDA_FREE(this->_values_inter, this->_memstat);
            CUDA_FREE(this->_hists_inter, this->_memstat);
            CUDA_FREE(this->_buc_ptrs_inter, this->_memstat);
            CUDA_FREE(this->_seg_scan_flags, this->_memstat);
            CUDA_FREE(this->_partitioner, this->_memstat);
        }
    }

    /*
     * splitDeviceWise: split without buffers, used for very small inputs
     *  only output the split result and the bucket start positions, no buffers
     *
     * @param d_keyIn: input keys
     * @param d_keyOut: output keys that are split
     * @param keyLength: length of the keys
     * @param buckets: number of buckets split to
     * @param d_bucPtr: pointers array to the start position of each bucket
     * */
    template<typename HashFuncType>
    void RP_splitDeviceWise(KType *keys_in, KType *keys_out,
                            VType *values_in, VType *values_out,
                            const uint32_t buckets, CntType *buc_ptrs,
                            const uint32_t offset_bits, HashFuncType f) {
        int blockSize = 1024;
        auto gridSize = (_data_len + blockSize - 1)/blockSize;
        if (gridSize > 32768) gridSize = 32768;

        CntType *hists = nullptr;
        checkCudaErrors(cudaMalloc((void**)&hists, sizeof(CntType)*gridSize*buckets));

        /*1.histogram*/
        execKernelDynamicAllocation(histogramDeviceWise, gridSize, blockSize, sizeof(CntType)*buckets,
                                    _timing, false, keys_in, _data_len, hists, buckets, offset_bits, f);

        /*2.global scan*/
        CUBScanExclusive(hists, hists, gridSize*buckets, _memstat, _timing);

        /*3.shuffle*/
        execKernelDynamicAllocation(shuffleDeviceWise, gridSize, blockSize, sizeof(CntType)*buckets,
                                    _timing, false, keys_in, keys_out, values_in, values_out, hists,
                                    _data_len, buckets, offset_bits, f);

        /*4.retrieve the bucket pointers from hists*/
        auto asc_begin = thrust::make_counting_iterator((uint32_t)0);
        auto asc_end = thrust::make_counting_iterator(buckets);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, buc_ptrs, [=] __device__(uint32_t idx) {
                return hists[idx*gridSize];
        }), _timing);

#ifdef FREE_DATA
        checkCudaErrors(cudaFree(hists));
#endif
    }

    template<typename HashFuncType>
    void RP_splitBlockWise(
            KType *keys_in, KType *keys_out,
            VType *values_in, VType *values_out,
            const uint32_t buckets_last_passes,  //#bucket_0 generated in all prior passes
            const uint32_t buckets_this_pass, //#buckett_1 split in this pass for each of the bucket_0
            CntType *buc_ptrs_last_passes,
            CntType *buc_ptrs_this_pass,
            const uint32_t offset_bits,
            HashFuncType f) {
        int blockSize = 1024;
        int gridSize = buckets_last_passes; //each block processes a bucket
        CntType *hists = _hists_inter;

        /*1.histogram*/
        execKernelDynamicAllocation(histogramBlockWise, gridSize, blockSize, sizeof(CntType)*buckets_this_pass,
                                    _timing, false, keys_in, hists, buckets_last_passes, buckets_this_pass,
                                    buc_ptrs_last_passes, offset_bits, f);

        /*2.segmented scan with load-balance search*/
        for(int i = 0; i < buckets_last_passes; ++i) {
            _partitioner[i] = i * buckets_this_pass;
        }

        mgpu::standard_context_t context(false);
        timingKernel(
                mgpu::load_balance_search(buckets_last_passes*buckets_this_pass, _partitioner,
                                          buckets_last_passes, _seg_scan_flags, context), _timing);
        timingKernel(
                thrust::exclusive_scan_by_key(thrust::device, _seg_scan_flags,
                                              _seg_scan_flags+buckets_last_passes*buckets_this_pass,
                                              hists, hists), _timing);

        /*3.shuffle*/
        execKernelDynamicAllocation(shuffleBlockWise, gridSize, blockSize, sizeof(CntType)*buckets_this_pass,
                                    _timing, false, keys_in, keys_out, values_in, values_out, hists,
                                    buckets_last_passes, buckets_this_pass, buc_ptrs_last_passes,
                                    buc_ptrs_this_pass, offset_bits, f);
    }

    /*
     * Split keys and values according to the keys
     * E.g.,
     * Original k,v | Shuffled k,v
     *      3,2            5,1
     *      5,1            9,0
     *      2,9            2,9
     *      9,0            3,2
     * */
    void splitKV(KType *keys_in, KType *&keys_out,
                 VType *values_in, VType *&values_out,
                 CntType *&buc_ptrs) {
        assert(keys_in && keys_out && buc_ptrs);
        log_info("Total buckets: %llu", _total_buckets);
        log_no_newline("Passes: %d, buckets in each pass: ", _num_passes);
        for(auto i = 0; i < _num_passes; i++) {
            printf("%llu ", _bucket_setting[i]);
        }
        printf("\n");

        /*lambda functions, Murmur3 hash*/
        auto hash_func = [] __device__ (uint32_t x, uint32_t offsetBits, uint32_t mask) {
//        x ^= x >> 16;
//        x *= 0x85ebca6b;
//        x ^= x >> 13;
//        x *= 0xc2b2ae35;
//        x ^= x >> 16;
            return (uint32_t)(x>>offsetBits) & mask;
        };

        /*double buffering*/
        KType *buf_keys_out[2];
        VType *buf_vals_out[2];
        CntType *buf_buc_ptrs[2];
        buf_keys_out[0] = keys_out; buf_keys_out[1] = _keys_inter;
        buf_vals_out[0] = values_out; buf_vals_out[1] = _values_inter;
        buf_buc_ptrs[0] = buc_ptrs; buf_buc_ptrs[1] = _buc_ptrs_inter;

        /*init the the head and tail buc_ptrs*/
        buf_buc_ptrs[0][0] = 0;
        buf_buc_ptrs[0][_bucket_setting[0]] = _data_len;

        int dec_offset_bits = logFunc(_total_buckets/_bucket_setting[0]);
        CntType acc_buckets = _bucket_setting[0];
        int flag = 0;

        RP_splitDeviceWise(keys_in, buf_keys_out[0], values_in, buf_vals_out[0],
                           _bucket_setting[0], buf_buc_ptrs[0], dec_offset_bits, hash_func); //first pass

        for(auto p = 1; p < _num_passes; p++) {
            dec_offset_bits -= logFunc(_bucket_setting[p]);
            buf_buc_ptrs[1-flag][0] = 0; //init the head and tail of target buc_ptrs of this pass
            buf_buc_ptrs[1-flag][acc_buckets * _bucket_setting[p]] = _data_len;

            RP_splitBlockWise(buf_keys_out[flag], buf_keys_out[1-flag], buf_vals_out[flag], buf_vals_out[1-flag],
                              acc_buckets, _bucket_setting[p], buf_buc_ptrs[flag],
                              buf_buc_ptrs[1-flag], dec_offset_bits, hash_func);
            flag = 1 - flag; //switch buffer
            acc_buckets *= _bucket_setting[p];
        }

        if ((_num_passes % 2) == 0) { /*swap when with even number of passes*/
            std::swap(keys_out, _keys_inter);
            std::swap(values_out, _values_inter);
            std::swap(buc_ptrs, _buc_ptrs_inter);
        }
    }

    /*
     * Split keys and generate indexes with the shuffled keys which record the original position of the keys
     * E.g.,
     * Original keys | Shuffled keys | generated indexes
     *      3               5               1
     *      5               9               3
     *      2               2               2
     *      9               3               0
     * */
    void splitKI(KType *keys_in, KType *&keys_out, VType *&index_in_origin, CntType *&buc_ptrs) {
        assert(index_in_origin);
        splitKV(keys_in, keys_out, nullptr, index_in_origin, buc_ptrs);
    }

    /*
     * Split keys
     * E.g.,
     * Original keys | Shuffled keys
     *      3               5
     *      5               9
     *      2               2
     *      9               3
     * */
    void splitK(KType *keys_in, KType *&keys_out, CntType *&bucPtrs) {
        VType *dummy_values_out = nullptr;
        splitKV(keys_in, keys_out, nullptr, dummy_values_out, bucPtrs);
    }

    /*
     * Look up values accroding to the keys
     * */
    void lookup_vals(KType *lookup_keys, VType *lookup_values, CntType n,
                     KType *ht_keys, VType *ht_vals, CntType ht_capacity, CntType *buc_ptrs) {
        int mingridsize, block_size;
        cudaOccupancyMaxPotentialBlockSize(&mingridsize, &block_size,
                                           lookup_KV_device_wise<KType,VType,CntType>, 0, 0);
        auto grid_size = (n + block_size - 1) / block_size;
        cudaDeviceSynchronize();
        execKernel((lookup_KV_device_wise<KType,VType,CntType>), grid_size, block_size, _timing, false,
                   ht_keys, ht_vals, ht_capacity, buc_ptrs, lookup_keys, lookup_values, n);
    }
};

///*todo: has bugs, stuck*/
///*sort the keys and rearrange the values*/
//template <typename DataType, typename CntType, typename IndexType>
//void splitWithCUBSort(
//        DataType *keys_in, DataType *&keys_out,
//        IndexType *&index_in_origin, const CntType length,
//        uint32_t buckets, CntType *&buc_ptrs,
//        CUDAMemStat *memstat, CUDATimeStat *timing) {
//    log_info("Split function: %s", __FUNCTION__);
//    bool isPower2 = isPowerOf2(buckets);
//    if (!isPower2) {
//        log_error("The number of buckets (%d) is not a power 2.", buckets);
//        exit(1);
//    }
//
//    /*find the log of the bucket number for CUB sort*/
//    int buc_log = floorPowerOf2(buckets);
//    if (!keys_out)
//        CUDA_MALLOC(&keys_out, sizeof(DataType)*length, memstat);
//    if (!index_in_origin)
//        CUDA_MALLOC(&index_in_origin, sizeof(CntType)*length, memstat);
//    if (!buc_ptrs) {
//        CUDA_MALLOC(&buc_ptrs, sizeof(CntType)*(buckets+1), memstat);
//        checkCudaErrors(cudaMemset(buc_ptrs, 0, sizeof(CntType)*(buckets+1)));
//    }
//
//    /*CUB radix sort*/
//    void *d_temp_storage = nullptr;
//    size_t temp_storage_bytes = 0;
//
//    CntType *asc_indexes_temp;
//    CUDA_MALLOC(&asc_indexes_temp, sizeof(CntType)*length, memstat);
//
//    thrust::counting_iterator<CntType> iter(0);
//    timingKernel(
//            thrust::copy(iter, iter + length, asc_indexes_temp), timing);
//
//    timingKernel(
//            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_in, keys_out, asc_indexes_temp, index_in_origin, length, 0, buc_log), timing);
//    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
//    timingKernel(
//            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_in, keys_out, asc_indexes_temp, index_in_origin, length, 0, buc_log), timing);
//
//    CUDA_FREE(asc_indexes_temp, memstat);
//    CUDA_FREE(d_temp_storage, memstat);
//
//    /*get the bucket ptrs by reading the output array once*/
//    execKernel(buildBucPtrs, GRID_SIZE, BLOCK_SIZE, timing, false, keys_out, length, buckets, buc_ptrs);
//}