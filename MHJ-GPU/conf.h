//
// Created by Bryan on 26/7/2019.
//

#pragma once

/*CUDA configuration*/
#define WARP_BITS   (5)                                 //log(WARP_SIZE)
#define WARP_SIZE   (1<<WARP_BITS)                      //warp size
#define WARP_MASK   (WARP_SIZE-1)                       //warp mask
#define BLOCK_SIZE   (256)                              //default block size
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)
#define GRID_SIZE    (1024)                             //default grid size
#define MAX_THREADS_PER_SM  (2048)

/*note that to execute q8, set MAX_MULTIWAYJOIN_BUILDTABLE_NUM to 7 and MAX_NUM_RES_ATTRS to 8*/
/*Program configuration*/
#define MAX_NUM_BUILDTABLES (5)             //maximum number of build tables, 7
#define MAX_NUM_ATTRS_IN_BUILD_TABLE    (2)             //maximum number of attrs in each build table
#define MAX_NUM_RES_ATTRS               (6)             //maximum number of attrs in result set, 8

#define SHARED_MATCHING_SET_LEN         (64)            //length of matching set in shared mem

#define MHJ_BUC_RATIO   (1)                             //Cardinality : buckets in MHJ
#define AMHJ_BUC_RATIO   (1)                            //Cardinality : buckets in AMHJ

#define MAX_PINNED_MEMORY_SIZE  (4ull*1024*1024*1024)  //4GB pinned-memory
#define OOC_RESERVE_SIZE        (100 * 1024 * 1024)     //reserve bytes for ooc
#define MAX_OOC_ITERATIONS      (1000)
#define MAX_PROBE_PREFETCH_THRES (1073741824)           //do not prefetch the probe table if its cardinality is greater than this number