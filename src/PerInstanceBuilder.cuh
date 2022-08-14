#include "Common.cuh"

__global__ void PerInstanceBuild(Task* task_queue, AABB* c_aabb, AABB* p_aabb,
                                 Node* nodes, AABB* aabbs,
                                 PrimitiveID* primitive_ids, int* scratch,
                                 int* block_counts, int* block_scan,
                                 int* num_leaves, bool top_of_tree);