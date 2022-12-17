#include <cstdlib>
#include <cstring>
#include "HashFunction.h"
#include "LossyCountMinSketch.h"

/* Allocates a row of topkapi sketch data structure */
void allocate_sketch( LossySketch* sketch, const unsigned range )
{
    int i;
    (*sketch)._b = range;
    (*sketch).identity = (uint32_t*) malloc(range* sizeof(uint32_t));
    (*sketch).lossyCount = (int*) malloc(range*sizeof(int));
    if ( (*sketch).identity == NULL || (*sketch).lossyCount == NULL )
    {
        fprintf(stderr, "LossySketch allocation error!\n");
        exit(EXIT_FAILURE);
    }
    /* set counts to -1 to indicate empty counter */
    for (i = 0; i < range; ++i)
        (*sketch).lossyCount[i] = -1;
}

/* Frees a row of topkapi sketch data structure */
void deallocate_sketch( LossySketch* sketch )
{
    free((*sketch).identity);
    free((*sketch).lossyCount);
}

/* This function updates the local sketch 
 * based on input word
 */ 
void update_sketch( LossySketch* _sketch,
                    uint32_t val,
                    uint32_t hash_func_rand,
                    unsigned log2range)
{
    unsigned bucket = gethash( &val, hash_func_rand, log2range );

    int* count_ptr = &((*_sketch).lossyCount[bucket]);
    uint32_t* sketch_val_ptr = &((*_sketch).identity[bucket]);

    if (*count_ptr == -1)    /*empty*/
    {
        *sketch_val_ptr = val;
        *count_ptr = 1;
    }
    else /* if counter is not empty */
    {   /* if same value*/
        if ((*sketch_val_ptr) == (val)) (*count_ptr) ++;
        else
        { /* if words are different */
            if (--(*count_ptr) < 0)
            {
                /* replace previous word with new word and set counter */
                *sketch_val_ptr = val;
                *count_ptr = 1;
            }
        }
    }
}

/* This function merges thread local sketches to
 * create the final sketch for a node
 * Note: it is used only when multi-threaded
 * execution happens
 */
void local_merge_sketch( LossySketch* LCMS,
                         const unsigned num_local_copies,
                         const unsigned num_hash_func,
                         const unsigned hash_func_index )
{
    uint32_t *vals[num_local_copies];
    int count[num_local_copies];
    unsigned i, j, k, diff_words;
    int max_selected;
    uint32_t* current_val;
    int max_count;
    unsigned range = LCMS[0]._b;

    for (i = 0; i < range; ++i)
    {
        vals[0] = &(LCMS[hash_func_index].identity[i]);
        count[0] = LCMS[hash_func_index].lossyCount[i];
        diff_words = 1;
        for (j = 1; j < num_local_copies; ++j)
        {
            current_val = &(LCMS[j*num_hash_func+hash_func_index].identity[i]);
            for ( k = 0; k < diff_words; ++k)
            {
                if (((*current_val) == (*(vals[k]))) &&
                    LCMS[j*num_hash_func+hash_func_index].lossyCount[i] != (-1))
                {   /* if same word */
                    count[k] += LCMS[j*num_hash_func+hash_func_index].lossyCount[i];
                    break;
                }
            }
            if (k == diff_words)
            {
                vals[diff_words] = current_val;
                count[diff_words] = LCMS[j*num_hash_func+hash_func_index].lossyCount[i];
                diff_words++;
            }
        }
        max_count = -1;
        max_selected = 0;
        k = 0;
        for (j = 0; j < diff_words; ++j)
        {
            if (count[j] != (-1))
            {
                if (max_selected)
                {
                    if (count[j] > max_count)
                    {
                        max_count = (count[j] - max_count);
                        k = j;
                    } else {
                        max_count -= count[j];
                    }
                } else {
                    max_count = count[j];
                    k = j;
                    max_selected = 1;
                }
            }
        }
        if (k != 0)
        {
            (*vals[0]) = (*vals[k]);
        }
        LCMS[hash_func_index].lossyCount[i] = max_count;
    }
}
