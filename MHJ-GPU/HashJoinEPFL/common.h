//
// Created by Bryan on 15/9/2019.
//

#pragma once

#ifdef __JETBRAINS_IDE__
#include "../../common-utils/cuda/cuda_fake/fake.h"
#endif

#include <chrono>
#include <sys/time.h>

#include "types_EPFL.h"
#include "../../common-utils/cuda/cuda_base.cuh"
#include "../../common-utils/cuda/CUDAStat.cuh"
#include "../../common-utils/timer.h"
#include "../../common-utils/pretty_print.h"
#include "../../common-utils/log.h"

#define data_type int
extern __shared__ data_type int_shared[];

union vec4{
    int4    vec ;
    int32_t i[4];
};

__host__ __device__ __forceinline__ uint32_t hasht(uint32_t x) {
    return x;
}

inline double cpuSeconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Handle missmatch of atomics for (u)int64/32_t with cuda's definitions
template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned int*) address, (unsigned int) val);
}


template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((int*) address, (int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
                int>::type = 0>
__device__ __forceinline__ T atomicOr_block(T *address, T val){
    return (T) atomicOr_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicOr_block(T *address, T val){
    return (T) atomicOr_block((unsigned int*) address, (unsigned int) val);
}


template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((int*) address, (int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((int*) address, (int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((int*) address, (int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((int*) address, (int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
        typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
                int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((int*) address, (int) val);
}
