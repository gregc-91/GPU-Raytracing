#ifndef _RADIX_SORT_H_
#define _RADIX_SORT_H_

#include <stdint.h>

void RadixSort(uint32_t* gpu_keys, uint32_t* gpu_values, uint32_t* gpu_temp1,
               uint32_t* gpu_temp2, uint32_t count);

#endif  // _RADIX_SORT_H_