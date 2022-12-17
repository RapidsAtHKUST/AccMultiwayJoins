#include "HashFunction.h"
#include "MurmurHash3.h"

/* Details about the API are given in HashFunction.h */
/* Note: this function uses VLA (Variable Length Array)*/
unsigned gethash( uint32_t *val, uint32_t hash_rand, unsigned log2range)
{
    unsigned hash;
    uint32_t seed = 0x9747b28c;
    uint32_t val_with_salt = (*val) + hash_rand;

    /* hash_rand is hash function specific random value,
    * we put it at the end of the string */
    MurmurHash3_x86_32( &val_with_salt , sizeof(uint32_t), seed, &hash );
    /* Now, we shift bits to limit the
    * hash value  within log2range */
    hash = hash << (32 - log2range);
    hash = hash >> (32 - log2range);
    return hash;
}
