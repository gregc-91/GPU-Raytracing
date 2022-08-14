
#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda_runtime_api.h>

#include <iomanip>
#include <iostream>

#include "device_launch_parameters.h"
#include "helper_math.h"

#define BLOCK_GRID_DIM 4
#define NUM_BLOCKS BLOCK_GRID_DIM* BLOCK_GRID_DIM* BLOCK_GRID_DIM

typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;

// Struct to store the rotations for each triangle to make a pair
struct Rotations {
    int rot_a;
    int rot_b;
};

typedef enum ChildType {
    ChildType_None,
    ChildType_Box,
    ChildType_Tri,
    ChildType_Inst,
    ChildType_Proc
} ChildType;

struct Camera {
    float3 position;
    float pitch;
    float3 w;
    float yaw;
    float3 u;
    float scale;
    float3 v;
    float max_depth;
};

struct Node {
    float3 min;
    uint32_t parent : 29;
    uint32_t count : 3;
    float3 max;
    uint32_t child : 29;
    uint32_t type : 3;
};

struct TrianglePair {
    float3 v0;
    float pad0;
    float3 v1;
    float pad1;
    float3 v2;
    float pad2;
    float3 v3;
    float pad3;

    __host__ __device__ TrianglePair() {}

    __host__ __device__ TrianglePair(const float3& v0_, const float3& v1_,
                                     const float3& v2_)
    {
        v0 = v0_;
        v1 = v1_;
        v2 = v2_;
    }

    __host__ __device__ TrianglePair(const float3& v0_, const float3& v1_,
                                     const float3& v2_, const float3& v3_)
    {
        v0 = v0_;
        v1 = v1_;
        v2 = v2_;
        v3 = v3_;
    }
};

struct AABB {
    float3 min;
    float3 max;

    __host__ __device__ AABB() {}

    __host__ __device__ AABB(const float3& amin, const float3& amax)
    {
        min = amin;
        max = amax;
    }

    __host__ __device__ AABB(const float3* minmax)
    {
        min = minmax[0];
        max = minmax[1];
    }

    __host__ __device__ AABB intersection(const AABB& other)
    {
        return AABB{fmaxf(min, other.min), fminf(max, other.max)};
    }

    __host__ __device__ bool valid()
    {
        return max.x >= min.x && max.y >= min.y && max.z >= min.z;
    };
};

__device__ inline float sa(const AABB& a)
{
    float3 len = a.max - a.min;
    return 2.0f * (len.x * len.y + len.x * len.z + len.y * len.z);
}

struct PrimitiveID {
    uint32_t id;
    uint32_t count;
};

struct Task {
    AABB c_aabb;
    AABB p_aabb;
    unsigned start;
    unsigned end;
    unsigned parent_idx;
    unsigned buffer_idx;
};

struct Bin {
    AABB c_aabb;
    AABB p_aabb;
    unsigned count;
};

typedef struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;

    Triangle()
    {
        v0.x = 0;
        v0.y = 0;
        v0.z = 0;
        v1.x = 0;
        v1.y = 0;
        v1.z = 0;
        v2.x = 0;
        v2.y = 0;
        v2.z = 0;
    }

    __host__ __device__ Triangle(float3 a, float3 b, float3 c)
        : v0(a), v1(b), v2(c)
    {
    }

    Triangle(float ax, float ay, float az, float bx, float by, float bz,
             float cx, float cy, float cz)
    {
        v0.x = ax;
        v0.y = ay;
        v0.z = az;
        v1.x = bx;
        v1.y = by;
        v1.z = bz;
        v2.x = cx;
        v2.y = cy;
        v2.z = cz;
    }
} Triangle;

#pragma warning(disable : 26812)
#define check(ans)                            \
    {                                         \
        GpuAssert((ans), __FILE__, __LINE__); \
    }
inline void GpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort) exit(code);
    }
}

// Macro to call a kernel and time it with events
#define run(_name_, _call_)                                   \
    {                                                         \
        cudaEvent_t event_start, event_stop;                  \
        cudaEventCreate(&event_start);                        \
        cudaEventCreate(&event_stop);                         \
        cudaEventRecord(event_start);                         \
        _call_;                                               \
        cudaEventRecord(event_stop);                          \
        cudaEventSynchronize(event_stop);                     \
                                                              \
        check(cudaPeekAtLastError());                         \
        check(cudaDeviceSynchronize());                       \
                                                              \
        float time = 0;                                       \
        cudaEventElapsedTime(&time, event_start, event_stop); \
                                                              \
        if (should_print) {                                   \
            printf("%s time elapsed: %fms\n", _name_, time);  \
        }                                                     \
    }

#endif