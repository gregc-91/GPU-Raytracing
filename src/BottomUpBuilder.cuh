#ifndef _BUILD_AS_H_
#define _BUILD_AS_H_

#include <stdint.h>

#include "Common.cuh"

__device__ __host__ inline float3 f3min(const float3& a, const float3& b)
{
    return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

__device__ __host__ inline float3 f3max(const float3& a, const float3& b)
{
    return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__device__ inline void f3min(volatile float3& o, volatile float3& a,
                             volatile float3& b)
{
    o.x = min(a.x, b.x);
    o.y = min(a.y, b.y);
    o.z = min(a.z, b.z);
}

__device__ inline void f3max(volatile float3& o, volatile float3& a,
                             volatile float3& b)
{
    o.x = max(a.x, b.x);
    o.y = max(a.y, b.y);
    o.z = max(a.z, b.z);
}

__device__ inline void assign(volatile float3& o, const float3& a)
{
    o.x = a.x;
    o.y = a.y;
    o.z = a.z;
}

__global__ void GenerateMortonCodes(unsigned* codes, unsigned* values,
                                    float3* vertices, AABB* scene_aabb,
                                    unsigned count);

__global__ void GenerateMortonCodesPairs(unsigned* codes, unsigned* values,
                                         float3* vertices, AABB* scene_aabb,
                                         unsigned* num_leaves, unsigned count);

__global__ void GenerateHierarchy(volatile Node* nodes, unsigned* leaf_indices,
                                  unsigned* sorted_morton_codes,
                                  unsigned* sorted_indices, int num_objects);

__global__ void GenerateAABBs(volatile Node* nodes, unsigned* leaf_indices,
                              unsigned* sorted_indices, unsigned* locks,
                              TrianglePair* vertices, unsigned count);

__global__ void GenerateTriangles(unsigned* sortedIndices, float3* vertices,
                                  TrianglePair* triangles, unsigned count);

__global__ void ExtractDepth(Node* nodes, PrimitiveID* output_indices,
                             AABB* output_aabbs, int* output_count,
                             AABB* c_aabb, AABB* p_aabb, unsigned root,
                             unsigned target_depth, unsigned count);

#endif  // _BUILD_AS_H_