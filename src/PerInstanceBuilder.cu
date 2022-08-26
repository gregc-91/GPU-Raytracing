#include <stdio.h>

#include "PerInstanceBuilder.cuh"
#include "helper_math.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347E+38F
#endif

#define NUM_BINS 8
#define LEAF_THRESHOLD 4

__shared__ int shared_write_index;
__shared__ int shared_queue_size;

__shared__ bool error;

__shared__ Task* global_task_queue;
__shared__ Node* global_nodes;
__shared__ AABB* global_aabbs;
__shared__ volatile int* global_ids[2];
__shared__ int* global_block_scan;
__shared__ PrimitiveID* global_prim_ids;

__device__ static float Get(float3& f, int i)
{
    return i == 0 ? f.x : i == 1 ? f.y : f.z;
}

__device__ static void Reset(AABB& a)
{
    a.min = make_float3(FLT_MAX);
    a.max = make_float3(-FLT_MAX);
}

__device__ static Bin Combine(const Bin& a, const Bin& b)
{
    Bin r;
    r.c_aabb = Combine(a.c_aabb, b.c_aabb);
    r.p_aabb = Combine(a.p_aabb, b.p_aabb);
    r.count = a.count + b.count;
    return r;
}

__device__ static void Initialise(PrimitiveID* prim_ids, Node* nodes,
                                  AABB* aabbs, int* scratch, Task* task_queue,
                                  AABB* c_aabbs, AABB* p_aabbs,
                                  int* block_counts, int* block_scan,
                                  int num_leaves, bool top_of_tree)
{
    if (threadIdx.x == 0) {
        global_nodes = nodes;
        global_aabbs = aabbs;
        global_ids[0] = scratch;
        global_ids[1] = scratch + num_leaves;
        global_task_queue = task_queue + block_scan[blockIdx.x];
        shared_queue_size = 1;
        shared_write_index = top_of_tree ? 0 : 128 + block_scan[blockIdx.x] * 2;
        global_block_scan = block_scan;
        global_prim_ids = prim_ids;

        Task task;
        task.c_aabb = c_aabbs[blockIdx.x];
        task.p_aabb = p_aabbs[blockIdx.x];
        task.start = block_scan[blockIdx.x];
        task.end = block_scan[blockIdx.x] + block_counts[blockIdx.x];
        task.parent_idx = atomicAdd(&shared_write_index, 1);
        task.buffer_idx = 0;

        global_task_queue[0] = task;
        error = false;
    }
}

__device__ static Task SetupTask()
{
    unsigned id = atomicAdd(&shared_queue_size, -1);
    return global_task_queue[id - 1];
}

__device__ static int SelectAxis(AABB& aabb)
{
    float3 length = aabb.max - aabb.min;
    int result = 0;
    result += 2 * (length.z > length.x && length.z > length.y);
    result += 1 * (length.y > length.x && length.y >= length.z);
    return result;
}

__device__ static void BinCentroids(Task& task, Bin bins[NUM_BINS], int axis)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (Get(task.c_aabb.max, axis) - Get(task.c_aabb.min, axis));

    unsigned start = task.start;
    unsigned end = task.end;
    AABB& c_aabb = task.c_aabb;

    for (unsigned i = start; i < end; i++) {
        AABB& aabb = global_aabbs[global_ids[task.buffer_idx][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (Get(centre, axis) - Get(c_aabb.min, axis)));

        bins[bin_id].p_aabb.min.x =
            fminf(bins[bin_id].p_aabb.min.x, aabb.min.x);
        bins[bin_id].p_aabb.min.y =
            fminf(bins[bin_id].p_aabb.min.y, aabb.min.y);
        bins[bin_id].p_aabb.min.z =
            fminf(bins[bin_id].p_aabb.min.z, aabb.min.z);
        bins[bin_id].p_aabb.max.x =
            fmaxf(bins[bin_id].p_aabb.max.x, aabb.max.x);
        bins[bin_id].p_aabb.max.y =
            fmaxf(bins[bin_id].p_aabb.max.y, aabb.max.y);
        bins[bin_id].p_aabb.max.z =
            fmaxf(bins[bin_id].p_aabb.max.z, aabb.max.z);

        bins[bin_id].c_aabb.min.x = fminf(bins[bin_id].c_aabb.min.x, centre.x);
        bins[bin_id].c_aabb.min.y = fminf(bins[bin_id].c_aabb.min.y, centre.y);
        bins[bin_id].c_aabb.min.z = fminf(bins[bin_id].c_aabb.min.z, centre.z);
        bins[bin_id].c_aabb.max.x = fmaxf(bins[bin_id].c_aabb.max.x, centre.x);
        bins[bin_id].c_aabb.max.y = fmaxf(bins[bin_id].c_aabb.max.y, centre.y);
        bins[bin_id].c_aabb.max.z = fmaxf(bins[bin_id].c_aabb.max.z, centre.z);

        bins[bin_id].count++;
    }
}

__device__ static int SelectPlane(Task& task, Bin bin[NUM_BINS],
                                  AABB child_p_aabb[2], AABB child_c_aabb[2])
{
    int result = 0;
    Bin l2r[NUM_BINS - 1];
    float best_score = FLT_MAX;

    // Initialise end bin
    l2r[0] = bin[0];

    // Linear pass left to right summing surface area
    for (int i = 1; i < NUM_BINS - 1; i++) {
        l2r[i] = Combine(l2r[i - 1], bin[i]);
    }

    // Linear pass right to left summing surface area
    Bin r2l = bin[NUM_BINS - 1];
    for (int i = NUM_BINS - 2; i >= 0; i--) {
        float score =
            sa(l2r[i].p_aabb) * l2r[i].count + sa(r2l.p_aabb) * r2l.count;

        if (score < best_score && l2r[i].count && r2l.count) {
            best_score = score;
            result = i;
            child_p_aabb[0] = l2r[i].p_aabb;
            child_p_aabb[1] = r2l.p_aabb;
            child_c_aabb[0] = l2r[i].c_aabb;
            child_c_aabb[1] = r2l.c_aabb;
        }

        r2l = Combine(r2l, bin[i]);
    }

    return result;
}

__device__ static int PartitionIds(Task task, int axis, int plane)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (Get(task.c_aabb.max, axis) - Get(task.c_aabb.min, axis));
    int in_buf = task.buffer_idx;
    int out_buf = in_buf ^ 1;

    unsigned p1 = task.start;
    unsigned p2 = task.end;
    AABB& c_aabb = task.c_aabb;

    for (unsigned i = task.start; i < task.end; i++) {
        AABB& aabb = global_aabbs[global_ids[in_buf][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (Get(centre, axis) - Get(c_aabb.min, axis)));

        if (bin_id <= plane) {
            global_ids[out_buf][p1++] = global_ids[in_buf][i];
        } else {
            global_ids[out_buf][--p2] = global_ids[in_buf][i];
        }
    }

    return p1;
}

__device__ static void RunTask(Task task, bool top_of_tree)
{
    Bin bins[NUM_BINS];
    AABB child_c_aabb[2];
    AABB child_p_aabb[2];
    int count = task.end - task.start;
    bool bounds_too_small = sa(task.c_aabb) <= 0.0f;
    bool partition = count > LEAF_THRESHOLD;
    int axis = 0;
    int mid = 0;

    // Initialise the bins
    for (unsigned i = 0; i < NUM_BINS; i++) {
        bins[i].c_aabb = {make_float3(FLT_MAX), make_float3(-FLT_MAX)};
        bins[i].p_aabb = {make_float3(FLT_MAX), make_float3(-FLT_MAX)};
        bins[i].count = 0;
    }

    if (!partition) {
        // Don't partition and write a node that points to the leaf nodes
        int idx = task.parent_idx;
        int child = atomicAdd(&shared_write_index, count);

        for (unsigned i = 0; i < count; i++) {
            unsigned tri_idx = global_ids[task.buffer_idx][task.start + i];

            if (!top_of_tree) {
                unsigned tri_idx = global_ids[task.buffer_idx][task.start + i];
                AABB& aabb = global_aabbs[tri_idx];

                global_nodes[child + i].child = global_prim_ids[tri_idx].id;
                global_nodes[child + i].count = global_prim_ids[tri_idx].count;
                global_nodes[child + i].min = aabb.min;
                global_nodes[child + i].max = aabb.max;
                global_nodes[child + i].type = ChildType_Tri;
            } else {
                // Create N contiguous "leaf" nodes and point them to the
                // children of each sub-root
                unsigned block_idx =
                    global_ids[task.buffer_idx][task.start + i];
                AABB& aabb = global_aabbs[block_idx];
                Node& sub_root =
                    global_nodes[128 + global_block_scan[block_idx] * 2];

                global_nodes[child + i].child = sub_root.child;
                global_nodes[child + i].count = sub_root.count;
                global_nodes[child + i].min = aabb.min;
                global_nodes[child + i].max = aabb.max;
                global_nodes[child + i].type = ChildType_Box;
            }
        }

        global_nodes[idx].child = child;
        global_nodes[idx].count = count;
        global_nodes[idx].min = task.p_aabb.min;
        global_nodes[idx].max = task.p_aabb.max;
        global_nodes[idx].type = ChildType_Box;
    } else if (bounds_too_small) {
        // Object partition at midpoint
        unsigned start = task.start;
        unsigned end = task.end;
        mid = task.start + (count >> 1);

        Reset(child_c_aabb[0]);
        Reset(child_c_aabb[1]);
        Reset(child_p_aabb[0]);
        Reset(child_p_aabb[1]);

        for (int i = start; i < mid; i++) {
            unsigned tri_idx = global_ids[task.buffer_idx][i];
            float3 centre =
                (global_aabbs[tri_idx].max + global_aabbs[tri_idx].min) * 0.5f;

            child_p_aabb[0] = Combine(child_p_aabb[0], global_aabbs[tri_idx]);
            child_c_aabb[0] = Combine(child_c_aabb[0], centre);

            global_ids[task.buffer_idx ^ 1][i] = global_ids[task.buffer_idx][i];
        }
        for (int i = mid; i < end; i++) {
            unsigned tri_idx = global_ids[task.buffer_idx][i];
            float3 centre =
                (global_aabbs[tri_idx].max + global_aabbs[tri_idx].min) * 0.5f;

            child_p_aabb[1] = Combine(child_p_aabb[1], global_aabbs[tri_idx]);
            child_c_aabb[1] = Combine(child_c_aabb[1], centre);

            global_ids[task.buffer_idx ^ 1][i] = global_ids[task.buffer_idx][i];
        }
    } else {
        // Partition with bins
        axis = SelectAxis(task.c_aabb);

        BinCentroids(task, bins, axis);

        int plane = SelectPlane(task, bins, child_p_aabb, child_c_aabb);

        mid = PartitionIds(task, axis, plane);
    }

    if (partition) {
        // Reserve space for the two children
        int child_index = atomicAdd(&shared_write_index, 2);

        // Create the parent node and point it to the reserved nodes
        int idx = task.parent_idx;
        global_nodes[idx].child = child_index;
        global_nodes[idx].count = 2;
        global_nodes[idx].min = task.p_aabb.min;
        global_nodes[idx].max = task.p_aabb.max;
        global_nodes[idx].type = ChildType_Box;

        // Create child tasks
        Task left_child;
        left_child.c_aabb = child_c_aabb[0];
        left_child.p_aabb = child_p_aabb[0];
        left_child.parent_idx = child_index;
        left_child.start = task.start;
        left_child.end = mid;
        left_child.buffer_idx = task.buffer_idx ^ 1;

        Task right_child;
        right_child.c_aabb = child_c_aabb[1];
        right_child.p_aabb = child_p_aabb[1];
        right_child.parent_idx = child_index + 1;
        right_child.start = mid;
        right_child.end = task.end;
        right_child.buffer_idx = task.buffer_idx ^ 1;

        // Write the child tasks to the queue
        int task_idx = atomicAdd(&shared_queue_size, 2);
        global_task_queue[task_idx + 0] = left_child;
        global_task_queue[task_idx + 1] = right_child;
    }
}

__global__ void PerInstanceBuild(Task* task_queue, AABB* c_aabbs, AABB* p_aabbs,
                                 Node* nodes, AABB* aabbs,
                                 PrimitiveID* primitive_ids, int* scratch,
                                 int* block_counts, int* block_scan,
                                 int* num_leaves, bool top_of_tree)
{
    Initialise(primitive_ids, nodes, aabbs, scratch, task_queue, c_aabbs,
               p_aabbs, block_counts, block_scan, *num_leaves, top_of_tree);

    __syncthreads();

    int iteration = 0;
    while (shared_queue_size) {
        unsigned num_tasks = min(shared_queue_size, blockDim.x);

        if (threadIdx.x < num_tasks) {
            Task task = SetupTask();
            RunTask(task, top_of_tree);
        }

        iteration++;

        __syncthreads();
    }
}