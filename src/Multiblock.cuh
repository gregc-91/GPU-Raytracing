#include "Common.cuh"

__global__ void CalculateSceneAabb(Triangle* triangles, int count, AABB* aabb);

__global__ void Setup(Triangle* triangles, TrianglePair* triangles_out,
                      int num_triangles, AABB* s_c_aabb, AABB* s_p_aabb,
                      volatile AABB* aabbs, volatile PrimitiveID* ids,
                      int* num_leaves, bool pair_triangles);

__global__ void SetupSplits(Triangle* triangles, TrianglePair* triangles_out,
                            int num_triangles, int extra_leaf_thresh,
                            AABB* s_c_aabb, AABB* s_p_aabb,
                            volatile AABB* aabbs, volatile PrimitiveID* ids,
                            int* num_leaves, int* extra_leaves);

__global__ void SetupPairSplits(Triangle* triangles,
                                TrianglePair* triangles_out, int num_triangles,
                                int extra_leaf_thresh, AABB* s_c_aabb,
                                AABB* s_p_aabb, volatile AABB* aabbs,
                                volatile PrimitiveID* ids,
                                int* num_triangles_out, int* num_leaves,
                                int* extra_leaves);

__global__ void GridBlockCounts(AABB* c_aabb, AABB* p_aabb, AABB* block_c_aabbs,
                                AABB* block_p_aabbs, AABB* aabbs,
                                int* block_counts, int* num_leaves);

__global__ void GridBlockScan(int* block_counts, int* block_scan, AABB* c_aabb,
                              AABB* p_aabb);

__global__ void GridBlockDistribute(AABB* c_aabb, AABB* block_c_aabbs,
                                    AABB* block_p_aabbs, AABB* aabbs,
                                    int* num_leaves, int* block_hist,
                                    int* block_scan, volatile int* references,
                                    int* root_count);
