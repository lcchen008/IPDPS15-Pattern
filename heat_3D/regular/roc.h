#ifndef ROC
#define ROC

#include "parameters.h"

namespace FFCPU
{
	typedef void (* reduce_fp)(void *value1, unsigned short value1_size, void *value2, unsigned short value2_size);
}

using namespace FFCPU;

struct intl
{
    int key_index;
    int value_index;
};

struct roc
{
    public:
        unsigned num_buckets;
        int locks[R_NUM_BUCKETS_G];
        struct intl buckets[R_NUM_BUCKETS_G];
        int memory_pool[GLOBAL_POOL_SIZE];
        unsigned int key_size_per_bucket[R_NUM_BUCKETS_G];
        unsigned int value_size_per_bucket[R_NUM_BUCKETS_G];
        unsigned int pairs_per_bucket[R_NUM_BUCKETS_G];
	unsigned int offsets[GPU_BLOCKS*GPU_THREADS/WARP_SIZE];
        unsigned memory_offset;

        unsigned omalloc(unsigned int size);
        void *get_address(unsigned index);
        void *get_key_address(unsigned index);
        unsigned get_key_size(unsigned index);
        void *get_value_address(unsigned index);
        unsigned get_value_size(unsigned index);

        bool insert(void *key, unsigned key_size, void *value, unsigned value_size, reduce_fp reduce_ptr);
        void merge(struct roc *object, reduce_fp reduce_ptr);
};

typedef struct roc CO;

#endif
