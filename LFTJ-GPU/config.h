//
// Created by Bryan on 26/7/2019.
//

#pragma once

/*CUDA configuration*/
#define WARP_BITS               (5)                                 //log(WARP_SIZE)
#define WARP_SIZE               (1<<WARP_BITS)                      //warp size
#define WARP_MASK               (WARP_SIZE-1)                       //warp mask
#define BLOCK_SIZE              (256)                                  //default block size
#define GRID_SIZE               (1024)                                 //default grid size
#define WARPS_PER_BLOCK         (BLOCK_SIZE/WARP_SIZE)

/*Program configuration*/
#define MAX_NUM_TABLES          (3) //maximum number of tables being joined
#define MAX_NUM_ATTRS           (3) //maximum number fo attributes in the query, todo: affect the occupancy

#define MAX_PINNED_MEMORY_SIZE  (4ull*1024*1024*1024)  //4GB pinned-memory
#define OOC_RESERVE_SIZE        (100 * 1024 * 1024)     //reserve bytes for ooc
#define MAX_OOC_ITERATIONS      (1000)
#define MAX_PROBE_PREFETCH_THRES (1073741824)           //do not prefetch the probe table if its cardinality is