//
// Created by Bryan on 24/4/2020.
//

#pragma once

#include "types.h"

#define BARRIER_ARRIVE(B,RV)                            \
    RV = pthread_barrier_wait(B);                       \
    if(RV !=0 && RV != PTHREAD_BARRIER_SERIAL_THREAD){  \
        printf("Couldn't wait on barrier\n");           \
        exit(EXIT_FAILURE);                             \
    }

/** checks malloc() result */
#ifndef MALLOC_CHECK
#define MALLOC_CHECK(M)                                                 \
    if(!M){                                                             \
        printf("[ERROR] MALLOC_CHECK: %s : %d\n", __FILE__, __LINE__);  \
        perror(": malloc() failed!\n");                                 \
        exit(EXIT_FAILURE);                                             \
    }
#endif

#define HASH_BIT_MODULO(K, MASK, NBITS) (((K) & MASK) >> NBITS)

static void *alloc_aligned(size_t size) {
    void * ret;
    int rv;
    rv = posix_memalign((void**)&ret, CACHE_LINE_SIZE, size);
    if (rv) {
        perror("alloc_aligned() failed: out of memory");
        return 0;
    }
    return ret;
}