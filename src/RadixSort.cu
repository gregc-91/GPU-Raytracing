#include "RadixSort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WG_SIZE 512
#define NUM_SEGMENTS 128
#define WARP_SIZE_LOG2 5
#define WARP_SIZE (1 << WARP_SIZE_LOG2)

//===========================================================================//
// Introduction                                                              //
//===========================================================================//

// In this radix sort implementation, the input array is logically partitioned
// into a fixed number of segments It performs 4 passes from LSB to MSB
// operating on a byte at a time Each pass consists of three kernels
// 1. Histogram generation
// 2. Prefix sum
// 3. Distribution

//===========================================================================//
// Histogram                                                                 //
//===========================================================================//

// Function to create two histograms, one of the value of a particular byte
// (digit), and one further partitioned by segment
__global__ void CreateHistograms(const uint32_t* data,
                                 uint32_t* digit_segment_histo,
                                 uint32_t* digit_histo, uint32_t digit)
{
    uint32_t input_size = gridDim.x * blockDim.x;
    uint32_t threads_per_segment =
        (input_size + NUM_SEGMENTS - 1) / NUM_SEGMENTS;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t segment_id = gid / threads_per_segment;

    uint32_t key = (data[gid] >> (digit * 8)) & 0xFF;
    uint32_t key_segement = key * NUM_SEGMENTS + segment_id;
    atomicAdd(&digit_histo[key], 1);
    atomicAdd(&digit_segment_histo[key_segement], 1);
}

// An improvement on the above histogram function
// It launches one workgroup per segment and computes individual histograms in
// local memory At the end they will be copied to global memory
__shared__ uint32_t shared_digit_histo[256];
__global__ void CreateHistogramsLM(const uint32_t* data, uint32_t count,
                                   uint32_t* digit_segment_histo,
                                   uint32_t* digit_histo, uint32_t digit)
{
    if (threadIdx.x < 256) {
        shared_digit_histo[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t threads_per_segment = (count + NUM_SEGMENTS - 1) / NUM_SEGMENTS;
    uint32_t segment_id = blockIdx.x;
    uint32_t segment_start = segment_id * threads_per_segment;

    for (unsigned i = segment_start + threadIdx.x;
         i < count && i < segment_start + threads_per_segment;
         i += blockDim.x) {
        uint32_t key = (data[i] >> (digit * 8)) & 0xFF;
        atomicAdd(&shared_digit_histo[key], 1);
    }

    __syncthreads();
    if (threadIdx.x < 256) {
        atomicAdd(&digit_histo[threadIdx.x], shared_digit_histo[threadIdx.x]);
        // atomicAdd(&digit_segment_histo[threadIdx.x * NUM_SEGMENTS +
        // segment_id], shared_digit_histo[threadIdx.x]);
        digit_segment_histo[threadIdx.x * NUM_SEGMENTS + segment_id] =
            shared_digit_histo[threadIdx.x];
    }
}

//===========================================================================//
// Prefix Sum                                                                //
//===========================================================================//

__shared__ uint32_t x[1024];
__shared__ uint32_t y[1024];

__global__ void PrefixSumInclusive(const uint32_t* data, uint32_t* sum)
{
    uint32_t b = blockIdx.x * NUM_SEGMENTS;
    uint32_t i = threadIdx.x;
    uint32_t n = blockDim.x;

    x[i] = i > 0 ? data[b + i] + data[b + i - 1] : data[b + i];
    __syncthreads();
    for (unsigned k = 2; k < n; k *= 4) {
        y[i] = i >= k ? x[i] + x[i - k] : x[i];
        __syncthreads();
        x[i] = i >= k * 2 ? y[i] + y[i - k * 2] : y[i];
        __syncthreads();
    }
    sum[b + i] = x[i];
}

__global__ void PrefixSumExclusive(const uint32_t* data, uint32_t* sum)
{
    uint32_t b = blockIdx.x * NUM_SEGMENTS;
    uint32_t i = threadIdx.x;
    uint32_t n = blockDim.x;

    x[i] = i > 0 ? data[b + i] + data[b + i - 1] : data[b + i];
    __syncthreads();
    for (unsigned k = 2; k < n; k *= 4) {
        y[i] = i >= k ? x[i] + x[i - k] : x[i];
        __syncthreads();
        x[i] = i >= k * 2 ? y[i] + y[i - k * 2] : y[i];
        __syncthreads();
    }
    sum[b + i] = i > 0 ? x[i - 1] : 0;
}

//===========================================================================//
// Distribution                                                              //
//===========================================================================//

__shared__ uint32_t shared_digit_scan[32][256];
__global__ void Distribute(const uint32_t* keys, const uint32_t* values,
                           uint32_t* keys_out, uint32_t* values_out,
                           uint32_t count, uint32_t* digit_segment_scan,
                           uint32_t* digit_scan, uint32_t k)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t segment_id = gid >> WARP_SIZE_LOG2;  // One warp per segment
    uint32_t warp_id = threadIdx.x >> WARP_SIZE_LOG2;
    uint32_t offset_in_warp = gid & ((1 << WARP_SIZE_LOG2) - 1);
    uint32_t threads_per_segment = (count + NUM_SEGMENTS - 1) / NUM_SEGMENTS;

    // Copy the slice of digit_segment_scan to local memory for each warp in a
    // workgroup Each warp does 32 of the 256 values, so 8 iterations are needed
    for (unsigned i = 0; i < 8; i++) {
        shared_digit_scan[warp_id][i * 32 + offset_in_warp] =
            digit_segment_scan[(i * 32 + offset_in_warp) * NUM_SEGMENTS +
                               segment_id];
    }

    __syncthreads();

    uint32_t segment_start = segment_id * threads_per_segment;
    uint32_t segment_end = segment_start + threads_per_segment;

    for (unsigned i = segment_start + offset_in_warp;
         i < segment_end && i < count; i += WARP_SIZE) {
        uint32_t key = keys[i];
        uint32_t digit = (keys[i] >> (k * 8)) & 0xFF;
        uint32_t off = 0;

        // The scan provides a write position of the digit within the segment
        // We had to serialise this atomic to ensure a stable output
        for (unsigned j = 0; j < WARP_SIZE; j++) {
            if (offset_in_warp == j)
                off = atomicAdd(&shared_digit_scan[warp_id][digit], 1);
        }

        // Add the digit scan for the final index
        uint32_t write_index = digit_scan[digit] + off;
        keys_out[write_index] = key;
        values_out[write_index] = values[i];
    }
}

//===========================================================================//
// Host Code                                                                 //
//===========================================================================//

void RadixSort(uint32_t* gpu_keys, uint32_t* gpu_values,
               uint32_t* gpu_temp_keys, uint32_t* gpu_temp_values,
               uint32_t count)
{
    // Define the GPU buffers
    uint32_t* gpu_digit_segment_histo;
    uint32_t* gpu_digit_histo;
    uint32_t* gpu_digit_segment_scan;
    uint32_t* gpu_digit_scan;

    // Allocate GPU buffers
    // Note: this can be made ~1ms faster if these are preallocated and used for
    // multiple builds
    size_t digit_segment_size = 256 * NUM_SEGMENTS * sizeof(uint32_t);
    size_t digit_size = 256 * sizeof(uint32_t);

    cudaMalloc(&gpu_digit_segment_histo, digit_segment_size);
    cudaMalloc(&gpu_digit_segment_scan, digit_segment_size);
    cudaMalloc(&gpu_digit_histo, digit_size);
    cudaMalloc(&gpu_digit_scan, digit_size);

    for (unsigned digit = 0; digit < 4; digit += 2) {
        cudaMemset(gpu_digit_segment_histo, 0, digit_segment_size);
        cudaMemset(gpu_digit_histo, 0, digit_size);

        CreateHistogramsLM<<<NUM_SEGMENTS, WG_SIZE>>>(
            gpu_keys, count, gpu_digit_segment_histo, gpu_digit_histo, digit);
        PrefixSumExclusive<<<256, NUM_SEGMENTS>>>(gpu_digit_segment_histo,
                                                  gpu_digit_segment_scan);
        PrefixSumExclusive<<<1, 256>>>(gpu_digit_histo, gpu_digit_scan);
        Distribute<<<(NUM_SEGMENTS * WARP_SIZE) / WG_SIZE, WG_SIZE>>>(
            gpu_keys, gpu_values, gpu_temp_keys, gpu_temp_values, count,
            gpu_digit_segment_scan, gpu_digit_scan, digit);
        cudaDeviceSynchronize();

        cudaMemset(gpu_digit_segment_histo, 0, digit_segment_size);
        cudaMemset(gpu_digit_histo, 0, digit_size);

        CreateHistogramsLM<<<NUM_SEGMENTS, WG_SIZE>>>(
            gpu_temp_keys, count, gpu_digit_segment_histo, gpu_digit_histo,
            digit + 1);
        PrefixSumExclusive<<<256, NUM_SEGMENTS>>>(gpu_digit_segment_histo,
                                                  gpu_digit_segment_scan);
        PrefixSumExclusive<<<1, 256>>>(gpu_digit_histo, gpu_digit_scan);
        Distribute<<<(NUM_SEGMENTS * WARP_SIZE) / WG_SIZE, WG_SIZE>>>(
            gpu_temp_keys, gpu_temp_values, gpu_keys, gpu_values, count,
            gpu_digit_segment_scan, gpu_digit_scan, digit + 1);
        cudaDeviceSynchronize();
    }

    cudaFree(gpu_digit_segment_histo);
    cudaFree(gpu_digit_histo);
    cudaFree(gpu_digit_segment_scan);
    cudaFree(gpu_digit_scan);
}