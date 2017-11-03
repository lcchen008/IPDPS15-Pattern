#define CPU_KERNEL
//#include "../regular.h"
#include "roc.h"
#include "hash.h"

bool get_lock(volatile int *lock_value)
{
    return __sync_val_compare_and_swap(lock_value, 0, 1) == 0;
}

bool release_lock(volatile int *lock_value)
{
    return __sync_val_compare_and_swap(lock_value, 1, 0); 
}

unsigned roc::omalloc(unsigned int size)
{
    size = FFCPU::align(size)/ALIGN_SIZE;
    unsigned offset = __sync_fetch_and_add(&memory_offset, size); 
    //printf("offset is: %d\n", offset);
    return offset;
}

void *roc::get_address(unsigned index)
{
    return memory_pool + index; 
}

void *roc::get_key_address(unsigned index)
{
    unsigned key_index = buckets[index].key_index;
    return get_address(key_index);
}

unsigned roc::get_key_size(unsigned index)
{
    return key_size_per_bucket[index];
}

void *roc::get_value_address(unsigned index)
{
    unsigned value_index = buckets[index].value_index;
    return get_address(value_index);
}

unsigned roc::get_value_size(unsigned index)
{
    return value_size_per_bucket[index];
}

bool roc::insert(void *key, unsigned key_size, void *value, unsigned value_size, reduce_fp reduce_ptr)
{
    int h = FFCPU::hash(key, key_size); 
    int index = h%R_NUM_BUCKETS_G;

    bool finish = false; 
    bool DoWork = true;
    int stride = 1;

    while(!finish)
    {
        DoWork = true; 
        while(DoWork)
        {
            //if(get_lock(&locks[index])) 
            {
                if(buckets[index].value_index==0) 
                {
                    //store key
                    int k = omalloc(key_size);
                    void * key_address = get_address(k);
                    memcpy(key_address, key, key_size);
                    key_size_per_bucket[index] = key_size;

                    //store value
                    int v = omalloc(value_size);
                    void * value_address = get_address(v);
                    memcpy(value_address, value, value_size);
                    value_size_per_bucket[index] = value_size;

                    pairs_per_bucket[index] = 1;
                    buckets[index].key_index = k;
                    buckets[index].value_index = v;

                    finish = true;
                    DoWork = false;
                    //release_lock(&locks[index]);
                }
                else
                {
                    void *key_data = get_key_address(index); 
                    unsigned key_data_size = get_key_size(index);

                    if(FFCPU::equal(key, key_size, key_data, key_data_size))
                    {
                        //FFCPU::reduce(get_value_address(index), get_value_size(index), value, value_size);          
                        reduce_ptr(get_value_address(index), get_value_size(index), value, value_size);          
                        DoWork = false;
                        finish = true;
                        //release_lock(&locks[index]);
                    }
                    else
                    {
                        DoWork = false;
                        //release_lock(&locks[index]);
                        index = (index+stride)%R_NUM_BUCKETS_G;
                    }
                }
            }
        }
    }
    return true;
}

void roc::merge(struct roc *object, reduce_fp reduce_ptr)
{
   for(int i = 0; i < R_NUM_BUCKETS_G; i++) 
   {
        if(object->pairs_per_bucket[i]!=0)
        {
            void *key = object->get_key_address(i); 
            unsigned key_size = object->get_key_size(i);
            void *value = object->get_value_address(i); 
            unsigned value_size = object->get_value_size(i);
	    //printf("key: %d\n", *(int *)key);
            insert(key, key_size, value, value_size, reduce_ptr);
        }
   }
}
