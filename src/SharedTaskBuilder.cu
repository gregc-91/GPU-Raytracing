#include <stdio.h>

#include "SharedTaskBuilder.cuh"
#include "helper_math.h"
#include "DeviceUtils.cuh"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347E+38F
#endif

#define MAX_PARALLEL_TASKS 64
#define NUM_BINS 8
#define LEAF_THRESHOLD 2
#define DEFER_SMALL_TASKS 1
#define SMALL_TASK_THRESHOLD 32

__shared__ int shared_num_leaves;
__shared__ int shared_write_index;
__shared__ int shared_queue_size;
__shared__ int shared_queue_base;
__shared__ int shared_small_task_queue_size;
__shared__ int shared_plane[MAX_PARALLEL_TASKS];
__shared__ int shared_partition_id[MAX_PARALLEL_TASKS][2];
__shared__ Task shared_task[MAX_PARALLEL_TASKS];
__shared__ Bin shared_bin[MAX_PARALLEL_TASKS][NUM_BINS];
__shared__ AABB shared_child_p_aabb[MAX_PARALLEL_TASKS][2];
__shared__ AABB shared_child_c_aabb[MAX_PARALLEL_TASKS][2];

__shared__ bool error;

__shared__ volatile Task* global_task_queue;
__shared__ volatile Node* global_nodes;
__shared__ AABB* global_aabbs;
__shared__ volatile int* global_ids[2];
__shared__ int* global_block_scan;
__shared__ int* global_block_counts;
__shared__ PrimitiveID* global_prim_ids;

__device__ static float get(float3& f, int i)
{
    return i == 0 ? f.x : i == 1 ? f.y : f.z;
}

__device__ static Bin Combine(const Bin& a, const Bin& b)
{
    Bin r;
    r.c_aabb = Combine(a.c_aabb, b.c_aabb);
    r.p_aabb = Combine(a.p_aabb, b.p_aabb);
    r.count = a.count + b.count;
    return r;
}

__device__ static void CopyToVolatile(volatile Task& a, Task& b)
{
    a.buffer_idx = b.buffer_idx;
    a.start = b.start;
    a.end = b.end;
    a.parent_idx = b.parent_idx;
    a.c_aabb.min.x = b.c_aabb.min.x;
    a.c_aabb.min.y = b.c_aabb.min.y;
    a.c_aabb.min.z = b.c_aabb.min.z;
    a.c_aabb.max.x = b.c_aabb.max.x;
    a.c_aabb.max.y = b.c_aabb.max.y;
    a.c_aabb.max.z = b.c_aabb.max.z;
    a.p_aabb.min.x = b.p_aabb.min.x;
    a.p_aabb.min.y = b.p_aabb.min.y;
    a.p_aabb.min.z = b.p_aabb.min.z;
    a.p_aabb.max.x = b.p_aabb.max.x;
    a.p_aabb.max.y = b.p_aabb.max.y;
    a.p_aabb.max.z = b.p_aabb.max.z;
}

__device__ static void CopyFromVolatile(Task& a, volatile Task& b)
{
    a.buffer_idx = b.buffer_idx;
    a.start = b.start;
    a.end = b.end;
    a.parent_idx = b.parent_idx;
    a.c_aabb.min.x = b.c_aabb.min.x;
    a.c_aabb.min.y = b.c_aabb.min.y;
    a.c_aabb.min.z = b.c_aabb.min.z;
    a.c_aabb.max.x = b.c_aabb.max.x;
    a.c_aabb.max.y = b.c_aabb.max.y;
    a.c_aabb.max.z = b.c_aabb.max.z;
    a.p_aabb.min.x = b.p_aabb.min.x;
    a.p_aabb.min.y = b.p_aabb.min.y;
    a.p_aabb.min.z = b.p_aabb.min.z;
    a.p_aabb.max.x = b.p_aabb.max.x;
    a.p_aabb.max.y = b.p_aabb.max.y;
    a.p_aabb.max.z = b.p_aabb.max.z;
}

__device__ static void Initialise(PrimitiveID* primitive_ids, Node* nodes,
                                  AABB* aabbs, int* scratch, Task* task_queue,
                                  AABB* c_aabbs, AABB* p_aabbs,
                                  int* block_counts, int* block_scan,
                                  int num_leaves, bool top_of_tree,
                                  unsigned base_write_offset)
{
    if (threadIdx.x == 0) {
        if (block_counts[blockIdx.x] != 0) {
            global_nodes = nodes;
            global_aabbs = aabbs;
            global_ids[0] = scratch;
            global_ids[1] = scratch + num_leaves;
            global_task_queue = task_queue + block_scan[blockIdx.x];
            global_block_scan = block_scan;
            global_block_counts = block_counts;
            global_prim_ids = primitive_ids;

            shared_num_leaves = num_leaves;
            shared_queue_size = 1;
            shared_queue_base = 0;
            shared_small_task_queue_size = 0;
            shared_write_index =
                top_of_tree ? base_write_offset
                            : base_write_offset + block_scan[blockIdx.x] * 2;

            Task task;
            task.c_aabb = c_aabbs[blockIdx.x];
            task.p_aabb = p_aabbs[blockIdx.x];
            task.start = block_scan[blockIdx.x];
            task.end = block_scan[blockIdx.x] + block_counts[blockIdx.x];
            task.parent_idx = atomicAdd(&shared_write_index, 1);
            task.buffer_idx = 0;

            // global_task_queue[0] = task;
            CopyToVolatile(global_task_queue[0], task);
            error = false;
        } else {
            shared_queue_size = 0;
            shared_small_task_queue_size = 0;
        }
    }
}

__device__ static void setup_task(unsigned id_of_task, unsigned iteration,
                                  bool small_task)
{
    unsigned id = 0;

    if (small_task) {
        id = global_block_counts[blockIdx.x] -
             atomicAdd(&shared_small_task_queue_size, -1);
    } else {
        id = atomicAdd(&shared_queue_size, -1) - 1;
    }

    // atomicAdd(&shared_queue_size, -1);
    // int id = atomicAdd(&shared_queue_base, 1);

    // shared_task[id_of_task] = global_task_queue[id - 1];
    CopyFromVolatile(shared_task[id_of_task], global_task_queue[id]);

    shared_partition_id[id_of_task][0] = shared_task[id_of_task].start;
    shared_partition_id[id_of_task][1] = shared_task[id_of_task].end;

    for (unsigned i = 0; i < NUM_BINS; i++) {
        shared_bin[id_of_task][i].c_aabb = {
            make_float3(__int_as_float(FloatToOrderedInt(FLT_MAX))),
            make_float3(__int_as_float(FloatToOrderedInt(-FLT_MAX)))};
        shared_bin[id_of_task][i].p_aabb = {
            make_float3(__int_as_float(FloatToOrderedInt(FLT_MAX))),
            make_float3(__int_as_float(FloatToOrderedInt(-FLT_MAX)))};
        shared_bin[id_of_task][i].count = 0;
    }
}

[[maybe_unused]] __device__ static void CheckTask(Task& task, int iteration,
                                                  int mid)
{
    AABB test_aabb;
    Reset(test_aabb);

    for (unsigned i = task.start; i < task.end; i++) {
        int tri_idx = global_ids[task.buffer_idx][i];
        AABB& aabb = global_aabbs[tri_idx];
        float3 centre = (aabb.min + aabb.max) * 0.5f;

        test_aabb = Combine(test_aabb, centre);
    }

    if (!Equal(test_aabb, task.c_aabb)) {
        printf(
            "Task check failed! block %d start %d end %d iteration %d mid %d\n "
            " %f %f %f %f %f %f     %f %f %f %f %f %f\n",
            blockIdx.x, task.start, task.end, iteration, mid, task.c_aabb.min.x,
            task.c_aabb.min.y, task.c_aabb.min.z, task.c_aabb.max.x,
            task.c_aabb.max.y, task.c_aabb.max.z, test_aabb.min.x,
            test_aabb.min.y, test_aabb.min.z, test_aabb.max.x, test_aabb.max.y,
            test_aabb.max.z);

        error = true;
    }
}

__device__ static int SelectAxis(AABB& aabb)
{
    float3 length = aabb.max - aabb.min;
    int result = 0;
    result += 2 * (length.z > length.x && length.z > length.y);
    result += 1 * (length.y > length.x && length.y >= length.z);
    return result;
}

__device__ static void BinCentroids(unsigned id_of_task, unsigned id_in_task,
                                    unsigned threads_per_task, int axis)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (get(shared_task[id_of_task].c_aabb.max, axis) -
                get(shared_task[id_of_task].c_aabb.min, axis));

    unsigned start = shared_task[id_of_task].start + id_in_task;
    unsigned end = shared_task[id_of_task].end;
    AABB& c_aabb = shared_task[id_of_task].c_aabb;

    for (unsigned i = start; i < end; i += threads_per_task) {
        AABB& aabb =
            global_aabbs[global_ids[shared_task[id_of_task].buffer_idx][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (get(centre, axis) - get(c_aabb.min, axis)));

        if (bin_id < 0 || bin_id >= NUM_BINS) {
            printf(
                "*** Error: bin out of bounds\n           task_id: %d block %d "
                "thread %d\n           bin: %d\n           start %d end %d\n   "
                "        i %d buf %d indirect %d\n",
                id_of_task, blockIdx.x, threadIdx.x, bin_id, start, end, i,
                shared_task[id_of_task].buffer_idx,
                global_ids[shared_task[id_of_task].buffer_idx][i]);
            error = true;
            break;
        }

        atomicMin((int*)&shared_bin[id_of_task][bin_id].p_aabb.min.x,
                  FloatToOrderedInt(aabb.min.x));
        atomicMin((int*)&shared_bin[id_of_task][bin_id].p_aabb.min.y,
                  FloatToOrderedInt(aabb.min.y));
        atomicMin((int*)&shared_bin[id_of_task][bin_id].p_aabb.min.z,
                  FloatToOrderedInt(aabb.min.z));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].p_aabb.max.x,
                  FloatToOrderedInt(aabb.max.x));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].p_aabb.max.y,
                  FloatToOrderedInt(aabb.max.y));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].p_aabb.max.z,
                  FloatToOrderedInt(aabb.max.z));

        atomicMin((int*)&shared_bin[id_of_task][bin_id].c_aabb.min.x,
                  FloatToOrderedInt(centre.x));
        atomicMin((int*)&shared_bin[id_of_task][bin_id].c_aabb.min.y,
                  FloatToOrderedInt(centre.y));
        atomicMin((int*)&shared_bin[id_of_task][bin_id].c_aabb.min.z,
                  FloatToOrderedInt(centre.z));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].c_aabb.max.x,
                  FloatToOrderedInt(centre.x));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].c_aabb.max.y,
                  FloatToOrderedInt(centre.y));
        atomicMax((int*)&shared_bin[id_of_task][bin_id].c_aabb.max.z,
                  FloatToOrderedInt(centre.z));

        atomicAdd(&shared_bin[id_of_task][bin_id].count, 1);
    }
}

__device__ static void ConvertBins(unsigned id_of_task, unsigned id_in_task)
{
    if (id_in_task < NUM_BINS) {
        shared_bin[id_of_task][id_in_task].c_aabb.min.x = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.min.x));
        shared_bin[id_of_task][id_in_task].c_aabb.min.y = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.min.y));
        shared_bin[id_of_task][id_in_task].c_aabb.min.z = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.min.z));
        shared_bin[id_of_task][id_in_task].c_aabb.max.x = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.max.x));
        shared_bin[id_of_task][id_in_task].c_aabb.max.y = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.max.y));
        shared_bin[id_of_task][id_in_task].c_aabb.max.z = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].c_aabb.max.z));

        shared_bin[id_of_task][id_in_task].p_aabb.min.x = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.min.x));
        shared_bin[id_of_task][id_in_task].p_aabb.min.y = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.min.y));
        shared_bin[id_of_task][id_in_task].p_aabb.min.z = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.min.z));
        shared_bin[id_of_task][id_in_task].p_aabb.max.x = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.max.x));
        shared_bin[id_of_task][id_in_task].p_aabb.max.y = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.max.y));
        shared_bin[id_of_task][id_in_task].p_aabb.max.z = OrderedIntToFloat(
            __float_as_int(shared_bin[id_of_task][id_in_task].p_aabb.max.z));
    }
}

__device__ static int SelectPlane(unsigned id_of_task)
{
    int result = 0;
    Bin l2r[NUM_BINS - 1];
    float best_score = FLT_MAX;

    // Initialise end bin
    l2r[0] = shared_bin[id_of_task][0];

    // Linear pass left to right summing surface area
    for (int i = 1; i < NUM_BINS - 1; i++) {
        l2r[i] = Combine(l2r[i - 1], shared_bin[id_of_task][i]);
    }

    // Linear pass right to left summing surface area
    Bin r2l = shared_bin[id_of_task][NUM_BINS - 1];
    for (int i = NUM_BINS - 2; i >= 0; i--) {
        float score =
            sa(l2r[i].p_aabb) * l2r[i].count + sa(r2l.p_aabb) * r2l.count;

        if (score < best_score && l2r[i].count && r2l.count) {
            best_score = score;
            result = i;
            shared_child_p_aabb[id_of_task][0] = l2r[i].p_aabb;
            shared_child_p_aabb[id_of_task][1] = r2l.p_aabb;
            shared_child_c_aabb[id_of_task][0] = l2r[i].c_aabb;
            shared_child_c_aabb[id_of_task][1] = r2l.c_aabb;
        }

        r2l = Combine(r2l, shared_bin[id_of_task][i]);
    }

    if (best_score == FLT_MAX) {
        printf("*** Error: failed to find valid partition\n");
        printf("***        task_id: %d\n", id_of_task);
        printf("***        c_aabb: %f %f %f %f %f %f\n",
               shared_task[id_of_task].c_aabb.min.x,
               shared_task[id_of_task].c_aabb.min.y,
               shared_task[id_of_task].c_aabb.min.z,
               shared_task[id_of_task].c_aabb.max.x,
               shared_task[id_of_task].c_aabb.max.y,
               shared_task[id_of_task].c_aabb.max.z);
        printf("***        start %d end %d\n", shared_task[id_of_task].start,
               shared_task[id_of_task].end);
        error = true;
    }

    if (result < 0 || result >= NUM_BINS) {
        printf("*** Error: plane out of range\n");
        error = true;
    }

    return result;
}

__device__ static void PartitionIds(int id_of_task, int id_in_task,
                                    int threads_per_task, int axis)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (get(shared_task[id_of_task].c_aabb.max, axis) -
                get(shared_task[id_of_task].c_aabb.min, axis));
    int plane = shared_plane[id_of_task];
    int in_buf = shared_task[id_of_task].buffer_idx;
    int out_buf = in_buf ^ 1;

    unsigned start = shared_task[id_of_task].start + id_in_task;
    unsigned end = shared_task[id_of_task].end;
    AABB& c_aabb = shared_task[id_of_task].c_aabb;

    for (unsigned i = start; i < end; i += threads_per_task) {
        AABB& aabb = global_aabbs[global_ids[in_buf][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (get(centre, axis) - get(c_aabb.min, axis)));

        if (bin_id <= plane) {
            int idx = atomicAdd(&shared_partition_id[id_of_task][0], 1);
            global_ids[out_buf][idx] = global_ids[in_buf][i];
        } else {
            int idx = atomicAdd(&shared_partition_id[id_of_task][1], -1) - 1;
            global_ids[out_buf][idx] = global_ids[in_buf][i];
        }
    }
}

__device__ static void RunTask(unsigned id_of_task, unsigned id_in_task,
                               unsigned threads_per_task, unsigned num_tasks,
                               unsigned iteration, bool top_of_tree,
                               ChildType leaf_type)
{
    int count = id_of_task < num_tasks ? shared_task[id_of_task].end -
                                             shared_task[id_of_task].start
                                       : 0;
    bool bounds_too_small =
        id_of_task < num_tasks ? sa(shared_task[id_of_task].c_aabb) <= 0.0f : 0;
    bool partition = count > LEAF_THRESHOLD;
    int axis = 0;

    if (id_of_task < num_tasks) {
        if (!partition) {
            // Don't partition and write a node that points to the leaf nodes
            if (id_in_task == 0) {
                int idx = shared_task[id_of_task].parent_idx;

                // If we have a singleton then we write the leaf directly into
                // the parent
                int child =
                    count == 1 ? idx : atomicAdd(&shared_write_index, count);

                for (unsigned i = 0; i < count; i++) {
                    if (!top_of_tree) {
                        unsigned tri_idx =
                            global_ids[shared_task[id_of_task].buffer_idx]
                                      [shared_task[id_of_task].start + i];
                        AABB& aabb = global_aabbs[tri_idx];

                        global_nodes[child + i].child =
                            global_prim_ids[tri_idx].id;
                        global_nodes[child + i].count =
                            global_prim_ids[tri_idx].count;
                        global_nodes[child + i].min.x = aabb.min.x;
                        global_nodes[child + i].min.y = aabb.min.y;
                        global_nodes[child + i].min.z = aabb.min.z;
                        global_nodes[child + i].max.x = aabb.max.x;
                        global_nodes[child + i].max.y = aabb.max.y;
                        global_nodes[child + i].max.z = aabb.max.z;
                        global_nodes[child + i].type = leaf_type;
                    } else {
                        // Create N contiguous "leaf" nodes and point them to
                        // the children of each sub-root
                        unsigned block_idx =
                            global_ids[shared_task[id_of_task].buffer_idx]
                                      [shared_task[id_of_task].start + i];
                        AABB& aabb = global_aabbs[block_idx];
                        volatile Node& sub_root =
                            global_nodes[2 * NUM_BLOCKS +
                                         global_block_scan[block_idx] * 2];

                        global_nodes[child + i].child = sub_root.child;
                        global_nodes[child + i].count = sub_root.count;
                        global_nodes[child + i].min.x = aabb.min.x;
                        global_nodes[child + i].min.y = aabb.min.y;
                        global_nodes[child + i].min.z = aabb.min.z;
                        global_nodes[child + i].max.x = aabb.max.x;
                        global_nodes[child + i].max.y = aabb.max.y;
                        global_nodes[child + i].max.z = aabb.max.z;
                        global_nodes[child + i].type = ChildType_Box;
                    }
                }

                if (count > 1) {
                    global_nodes[idx].child = child;
                    global_nodes[idx].count = count;
                    global_nodes[idx].min.x =
                        shared_task[id_of_task].p_aabb.min.x;
                    global_nodes[idx].min.y =
                        shared_task[id_of_task].p_aabb.min.y;
                    global_nodes[idx].min.z =
                        shared_task[id_of_task].p_aabb.min.z;
                    global_nodes[idx].max.x =
                        shared_task[id_of_task].p_aabb.max.x;
                    global_nodes[idx].max.y =
                        shared_task[id_of_task].p_aabb.max.y;
                    global_nodes[idx].max.z =
                        shared_task[id_of_task].p_aabb.max.z;
                    global_nodes[idx].type = ChildType_Box;
                }
            }
        } else if (bounds_too_small) {
            if (id_in_task == 0) {
                // Object partition at midpoint
                unsigned start = shared_task[id_of_task].start;
                unsigned mid = shared_task[id_of_task].start + (count >> 1);
                unsigned end = shared_task[id_of_task].end;
                shared_partition_id[id_of_task][0] = mid;

                Reset(shared_child_c_aabb[id_of_task][0]);
                Reset(shared_child_c_aabb[id_of_task][1]);
                Reset(shared_child_p_aabb[id_of_task][0]);
                Reset(shared_child_p_aabb[id_of_task][1]);

                for (int i = start; i < mid; i++) {
                    unsigned tri_idx =
                        global_ids[shared_task[id_of_task].buffer_idx][i];
                    float3 centre = (global_aabbs[tri_idx].max +
                                     global_aabbs[tri_idx].min) *
                                    0.5f;

                    shared_child_p_aabb[id_of_task][0] =
                        Combine(shared_child_p_aabb[id_of_task][0],
                                global_aabbs[tri_idx]);
                    shared_child_c_aabb[id_of_task][0] =
                        Combine(shared_child_c_aabb[id_of_task][0], centre);

                    global_ids[shared_task[id_of_task].buffer_idx ^ 1][i] =
                        global_ids[shared_task[id_of_task].buffer_idx][i];
                }
                for (int i = mid; i < end; i++) {
                    unsigned tri_idx =
                        global_ids[shared_task[id_of_task].buffer_idx][i];
                    float3 centre = (global_aabbs[tri_idx].max +
                                     global_aabbs[tri_idx].min) *
                                    0.5f;

                    shared_child_p_aabb[id_of_task][1] =
                        Combine(shared_child_p_aabb[id_of_task][1],
                                global_aabbs[tri_idx]);
                    shared_child_c_aabb[id_of_task][1] =
                        Combine(shared_child_c_aabb[id_of_task][1], centre);

                    global_ids[shared_task[id_of_task].buffer_idx ^ 1][i] =
                        global_ids[shared_task[id_of_task].buffer_idx][i];
                }
            }
        } else {
            // Partition with bins
            axis = SelectAxis(shared_task[id_of_task].c_aabb);

            BinCentroids(id_of_task, id_in_task, threads_per_task, axis);
        }
    }

    __syncthreads();

    if (error) {
        return;
    }

    if (id_of_task < num_tasks && partition && !bounds_too_small) {
        ConvertBins(id_of_task, id_in_task);
    }

    __syncthreads();

    if (id_of_task < num_tasks && partition && !bounds_too_small &&
        id_in_task == 0) {
        shared_plane[id_of_task] = SelectPlane(id_of_task);
    }

    __syncthreads();

    if (id_of_task < num_tasks && partition && !bounds_too_small) {
        PartitionIds(id_of_task, id_in_task, threads_per_task, axis);
    }

    __syncthreads();

    if (id_of_task < num_tasks && partition && id_in_task == 0) {
        int mid = shared_partition_id[id_of_task][0];

        // Reserve space for the two children
        int child_index = atomicAdd(&shared_write_index, 2);

        // Create the parent node and point it to the reserved nodes
        int idx = shared_task[id_of_task].parent_idx;
        global_nodes[idx].child = child_index;
        global_nodes[idx].count = 2;
        global_nodes[idx].min.x = shared_task[id_of_task].p_aabb.min.x;
        global_nodes[idx].min.y = shared_task[id_of_task].p_aabb.min.y;
        global_nodes[idx].min.z = shared_task[id_of_task].p_aabb.min.z;
        global_nodes[idx].max.x = shared_task[id_of_task].p_aabb.max.x;
        global_nodes[idx].max.y = shared_task[id_of_task].p_aabb.max.y;
        global_nodes[idx].max.z = shared_task[id_of_task].p_aabb.max.z;
        global_nodes[idx].type = ChildType_Box;

        // Create child tasks
        Task left_child;
        left_child.c_aabb = shared_child_c_aabb[id_of_task][0];
        left_child.p_aabb = shared_child_p_aabb[id_of_task][0];
        left_child.parent_idx = child_index;
        left_child.start = shared_task[id_of_task].start;
        left_child.end = mid;
        left_child.buffer_idx = shared_task[id_of_task].buffer_idx ^ 1;

        Task right_child;
        right_child.c_aabb = shared_child_c_aabb[id_of_task][1];
        right_child.p_aabb = shared_child_p_aabb[id_of_task][1];
        right_child.parent_idx = child_index + 1;
        right_child.start = mid;
        right_child.end = shared_task[id_of_task].end;
        right_child.buffer_idx = shared_task[id_of_task].buffer_idx ^ 1;

#if DEFER_SMALL_TASKS
        if (left_child.end - left_child.start < SMALL_TASK_THRESHOLD) {
            int task_idx = global_block_counts[blockIdx.x] - 1 -
                           atomicAdd(&shared_small_task_queue_size, 1);
            CopyToVolatile(global_task_queue[task_idx], left_child);
        } else {
            int task_idx = atomicAdd(&shared_queue_size, 1);
            CopyToVolatile(global_task_queue[task_idx], left_child);
        }

        if (right_child.end - right_child.start < SMALL_TASK_THRESHOLD) {
            int task_idx = global_block_counts[blockIdx.x] - 1 -
                           atomicAdd(&shared_small_task_queue_size, 1);
            CopyToVolatile(global_task_queue[task_idx], right_child);
        } else {
            int task_idx = atomicAdd(&shared_queue_size, 1);
            CopyToVolatile(global_task_queue[task_idx], right_child);
        }
#else
        // Write the child tasks to the queue
        int task_idx = atomicAdd(&shared_queue_size, 2);

        // global_task_queue[task_idx+0] = left_child;
        // global_task_queue[task_idx+1] = right_child;
        CopyToVolatile(global_task_queue[task_idx + 0], left_child);
        CopyToVolatile(global_task_queue[task_idx + 1], right_child);
#endif
    }
}

__device__ static Task PerInstanceSetupTask()
{
    unsigned id = global_block_counts[blockIdx.x] -
                  atomicAdd(&shared_small_task_queue_size, -1);

    Task result;
    CopyFromVolatile(result, global_task_queue[id]);
    return result;
}

__device__ static int per_instance_SelectAxis(AABB& aabb)
{
    float3 length = aabb.max - aabb.min;
    int result = 0;
    result += 2 * (length.z > length.x && length.z > length.y);
    result += 1 * (length.y > length.x && length.y >= length.z);
    return result;
}

__device__ static void per_instance_BinCentroids(Task& task, Bin bins[NUM_BINS],
                                                 int axis)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (get(task.c_aabb.max, axis) - get(task.c_aabb.min, axis));

    unsigned start = task.start;
    unsigned end = task.end;
    AABB& c_aabb = task.c_aabb;

    for (unsigned i = start; i < end; i++) {
        AABB& aabb = global_aabbs[global_ids[task.buffer_idx][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (get(centre, axis) - get(c_aabb.min, axis)));

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

__device__ static int per_instance_SelectPlane(Task& task, Bin bin[NUM_BINS],
                                               AABB child_p_aabb[2],
                                               AABB child_c_aabb[2])
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

    if (best_score == FLT_MAX) {
        printf("*** Error: failed to find valid partition\n");
        error = true;
    }

    if (result < 0 || result >= NUM_BINS) {
        printf("*** Error: plane out of range\n");
        error = true;
    }

    return result;
}

__device__ static int per_instance_PartitionIds(Task task, int axis, int plane)
{
    float epsilon = 1.1920929e-7;  // 2^-23
    float k1 = NUM_BINS * (1 - epsilon) /
               (get(task.c_aabb.max, axis) - get(task.c_aabb.min, axis));
    int in_buf = task.buffer_idx;
    int out_buf = in_buf ^ 1;

    unsigned p1 = task.start;
    unsigned p2 = task.end;
    AABB& c_aabb = task.c_aabb;

    for (unsigned i = task.start; i < task.end; i++) {
        AABB& aabb = global_aabbs[global_ids[in_buf][i]];
        float3 centre = (aabb.min + aabb.max) * 0.5f;
        int bin_id = int(k1 * (get(centre, axis) - get(c_aabb.min, axis)));

        if (bin_id <= plane) {
            global_ids[out_buf][p1++] = global_ids[in_buf][i];
        } else {
            global_ids[out_buf][--p2] = global_ids[in_buf][i];
        }
    }

    return p1;
}

__device__ static void PerInstanceRunTask(Task task, bool top_of_tree,
                                          ChildType leaf_type)
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

        // If we have a singleton then we write the leaf directly into the
        // parent
        int child = count == 1 ? idx : atomicAdd(&shared_write_index, count);

        // Note: The order of these leaves is non-deterministic
        //       We tiebreak hits on an ID during traversal to make it
        //       deterministic
        for (unsigned i = 0; i < count; i++) {
            unsigned tri_idx = global_ids[task.buffer_idx][task.start + i];

            if (!top_of_tree) {
                unsigned tri_idx = global_ids[task.buffer_idx][task.start + i];
                AABB& aabb = global_aabbs[tri_idx];

                global_nodes[child + i].child = global_prim_ids[tri_idx].id;
                global_nodes[child + i].count = global_prim_ids[tri_idx].count;
                global_nodes[child + i].min.x = aabb.min.x;
                global_nodes[child + i].min.y = aabb.min.y;
                global_nodes[child + i].min.z = aabb.min.z;
                global_nodes[child + i].max.x = aabb.max.x;
                global_nodes[child + i].max.y = aabb.max.y;
                global_nodes[child + i].max.z = aabb.max.z;
                global_nodes[child + i].type = leaf_type;
            } else {
                // Create N contiguous "leaf" nodes and point them to the
                // children of each sub-root
                unsigned block_idx =
                    global_ids[task.buffer_idx][task.start + i];
                AABB& aabb = global_aabbs[block_idx];
                volatile Node& sub_root =
                    global_nodes[128 + global_block_scan[block_idx] * 2];

                global_nodes[child + i].child = sub_root.child;
                global_nodes[child + i].count = sub_root.count;
                global_nodes[child + i].min.x = aabb.min.x;
                global_nodes[child + i].min.y = aabb.min.y;
                global_nodes[child + i].min.z = aabb.min.z;
                global_nodes[child + i].max.x = aabb.max.x;
                global_nodes[child + i].max.y = aabb.max.y;
                global_nodes[child + i].max.z = aabb.max.z;
                global_nodes[child + i].type = ChildType_Box;
            }
        }

        if (count > 1) {
            global_nodes[idx].child = child;
            global_nodes[idx].count = count;
            global_nodes[idx].min.x = task.p_aabb.min.x;
            global_nodes[idx].min.y = task.p_aabb.min.y;
            global_nodes[idx].min.z = task.p_aabb.min.z;
            global_nodes[idx].max.x = task.p_aabb.max.x;
            global_nodes[idx].max.y = task.p_aabb.max.y;
            global_nodes[idx].max.z = task.p_aabb.max.z;
            global_nodes[idx].type = ChildType_Box;
        }
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
        axis = per_instance_SelectAxis(task.c_aabb);

        per_instance_BinCentroids(task, bins, axis);

        int plane =
            per_instance_SelectPlane(task, bins, child_p_aabb, child_c_aabb);

        mid = per_instance_PartitionIds(task, axis, plane);
    }

    if (partition) {
        // Reserve space for the two children
        int child_index = atomicAdd(&shared_write_index, 2);

        // Create the parent node and point it to the reserved nodes
        int idx = task.parent_idx;
        global_nodes[idx].child = child_index;
        global_nodes[idx].count = 2;
        global_nodes[idx].min.x = task.p_aabb.min.x;
        global_nodes[idx].min.y = task.p_aabb.min.y;
        global_nodes[idx].min.z = task.p_aabb.min.z;
        global_nodes[idx].max.x = task.p_aabb.max.x;
        global_nodes[idx].max.y = task.p_aabb.max.y;
        global_nodes[idx].max.z = task.p_aabb.max.z;
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
        // int task_idx = atomicAdd(&shared_queue_size, 2);
        int task_idx = global_block_counts[blockIdx.x] - 1 -
                       atomicAdd(&shared_small_task_queue_size, 2);

        // global_task_queue[task_idx + 0] = left_child;
        // global_task_queue[task_idx + 1] = right_child;

        CopyToVolatile(global_task_queue[task_idx - 1], left_child);
        CopyToVolatile(global_task_queue[task_idx - 0], right_child);
    }
}

__global__ void SharedTaskBuild(Task* task_queue, AABB* c_aabbs, AABB* p_aabbs,
                                Node* nodes, AABB* aabbs,
                                PrimitiveID* primitive_ids, int* scratch,
                                int* block_counts, int* block_scan,
                                int* num_leaves, bool top_of_tree,
                                ChildType leaf_type, unsigned base_write_offset)
{
    Initialise(primitive_ids, nodes, aabbs, scratch, task_queue, c_aabbs,
               p_aabbs, block_counts, block_scan, *num_leaves, top_of_tree,
               base_write_offset);

    __syncthreads();

    int iteration = 0;

    while (shared_queue_size) {
        unsigned num_tasks = min(shared_queue_size, MAX_PARALLEL_TASKS);
        unsigned threads_per_task = blockDim.x / num_tasks;
        unsigned id_of_task = threadIdx.x / threads_per_task;
        unsigned id_in_task = threadIdx.x % threads_per_task;

        __syncthreads();

        if (id_of_task < num_tasks && id_in_task == 0) {
            setup_task(id_of_task, iteration, false);
        }

        __syncthreads();

        RunTask(id_of_task, id_in_task, threads_per_task, num_tasks, iteration,
                top_of_tree, leaf_type);

        __syncthreads();

        if (error) {
            return;
        }

        iteration++;
    }

    __syncthreads();

    iteration = 0;

    while (shared_small_task_queue_size) {
        unsigned num_tasks = min(shared_small_task_queue_size, blockDim.x);

        __syncthreads();

        if (threadIdx.x < num_tasks) {
            Task task = PerInstanceSetupTask();
            PerInstanceRunTask(task, top_of_tree, leaf_type);
        }

        iteration++;

        __syncthreads();
    }

    /*
    while (shared_small_task_queue_size) {
        unsigned num_tasks = min(shared_small_task_queue_size,
    MAX_PARALLEL_TASKS); unsigned threads_per_task = blockDim.x / num_tasks;
        unsigned id_of_task = threadIdx.x / threads_per_task;
        unsigned id_in_task = threadIdx.x % threads_per_task;

        __syncthreads();

        if (id_of_task < num_tasks && id_in_task == 0) {
            setup_task(id_of_task, iteration, true);
        }

        __syncthreads();

        RunTask(id_of_task, id_in_task, threads_per_task, num_tasks, iteration,
    top_of_tree);

        __syncthreads();

        if (error) {
            return;
        }

        iteration++;
    }*/
}