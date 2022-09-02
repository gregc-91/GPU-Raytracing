#include "Multiblock.cuh"
#include "DeviceUtils.cuh"
#include "Pairing.cuh"

__shared__ int shared_block_counts[NUM_BLOCKS];
__shared__ AABB shared_block_c_aabbs[NUM_BLOCKS];
__shared__ AABB shared_block_p_aabbs[NUM_BLOCKS];

__device__ float& Get(float3& f, int x)
{
    return x == 0 ? f.x : x == 1 ? f.y : f.z;
}

__device__ static void AssignVolatile(volatile AABB& a, AABB& b)
{
    a.min.x = b.min.x;
    a.min.y = b.min.y;
    a.min.z = b.min.z;
    a.max.x = b.max.x;
    a.max.y = b.max.y;
    a.max.z = b.max.z;
}

__device__ static void AtomicCombine(AABB& a, AABB& b)
{
    atomicMin((int*)&a.min.x, __float_as_int(b.min.x));
    atomicMin((int*)&a.min.y, __float_as_int(b.min.y));
    atomicMin((int*)&a.min.z, __float_as_int(b.min.z));
    atomicMax((int*)&a.max.x, __float_as_int(b.max.x));
    atomicMax((int*)&a.max.y, __float_as_int(b.max.y));
    atomicMax((int*)&a.max.z, __float_as_int(b.max.z));
}

__device__ static void AtomicConvertCombine(AABB& a, AABB& b)
{
    atomicMin((int*)&a.min.x, FloatToOrderedInt(b.min.x));
    atomicMin((int*)&a.min.y, FloatToOrderedInt(b.min.y));
    atomicMin((int*)&a.min.z, FloatToOrderedInt(b.min.z));
    atomicMax((int*)&a.max.x, FloatToOrderedInt(b.max.x));
    atomicMax((int*)&a.max.y, FloatToOrderedInt(b.max.y));
    atomicMax((int*)&a.max.z, FloatToOrderedInt(b.max.z));
}

__device__ static void AtomicConvertCombine(AABB& a, float3& c)
{
    atomicMin((int*)&a.min.x, FloatToOrderedInt(c.x));
    atomicMin((int*)&a.min.y, FloatToOrderedInt(c.y));
    atomicMin((int*)&a.min.z, FloatToOrderedInt(c.z));
    atomicMax((int*)&a.max.x, FloatToOrderedInt(c.x));
    atomicMax((int*)&a.max.y, FloatToOrderedInt(c.y));
    atomicMax((int*)&a.max.z, FloatToOrderedInt(c.z));
}

__device__ static bool Equal(const int3& a, const int3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ bool IntersectEdge(float3& a, float3& b, float3& out, float plane,
                              int axis)
{
    // Check if the vertices cross the plane
    if ((Get(a, axis) < plane) == (Get(b, axis) < plane)) return false;

    int q = axis == 0 ? 1 : 0;
    int r = axis == 2 ? 1 : 2;

    // Compute gradients
    float dq_by_dp = (Get(b, q) - Get(a, q)) / (Get(b, axis) - Get(a, axis));
    float dr_by_dp = (Get(b, r) - Get(a, r)) / (Get(b, axis) - Get(a, axis));

    float pq = (plane - Get(a, axis)) * dq_by_dp;
    float pr = (plane - Get(a, axis)) * dr_by_dp;

    float3 result;
    Get(result, axis) = plane;
    Get(result, q) = pq;
    Get(result, r) = pr;
    out = result;

    return true;
}

__device__ int3 CalculateGridcell(float3 p, const AABB& grid_bounds)
{
    return clamp(make_int3(floorf((p - grid_bounds.min) * (BLOCK_GRID_DIM) /
                                  (grid_bounds.max - grid_bounds.min))),
                 0, BLOCK_GRID_DIM - 1);
}

__device__ AABB CellToBounds(int3 cell_id, const AABB& grid_bounds)
{
    float3 mi = grid_bounds.min +
                make_float3(cell_id) *
                    ((grid_bounds.max - grid_bounds.min) / BLOCK_GRID_DIM);
    float3 ma = grid_bounds.min +
                make_float3(cell_id + 1) *
                    ((grid_bounds.max - grid_bounds.min) / BLOCK_GRID_DIM);
    return AABB{mi, ma};
}

// kernel: CalculateSceneAabb
// Calculates the aabb of the scene
__global__ void CalculateSceneAabb(Triangle* triangles, int count, AABB* aabb)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        AABB x = {fminf(fminf(triangles[gid].v0, triangles[gid].v1),
                        triangles[gid].v2),
                  fmaxf(fmaxf(triangles[gid].v0, triangles[gid].v1),
                        triangles[gid].v2)};
        AtomicConvertCombine(*aabb, x);
    }
}

// Step to next cell
// Returns true if finished
__device__ bool GridNextCell(int3& cell, int3 min_cell, int3 max_cell)
{
    bool end_x = cell.x == max_cell.x;
    bool end_y = cell.y == max_cell.y;
    bool end_z = cell.z == max_cell.z;

    cell.x = end_x ? min_cell.x : cell.x + 1;
    if (end_x) cell.y = end_y ? min_cell.y : cell.y + 1;
    if (end_x && end_y) cell.z = end_z ? min_cell.z : cell.z + 1;
    if (end_x && end_y && end_z) return true;
    ;

    return false;
}

// kernel: Setup
// Pairs triangles, creates the bounding box for each pair, and for the whole
// scene Parallelisation: multi-block, one thread per pair of input primitives
__global__ void Setup(Triangle* triangles, TrianglePair* triangles_out,
                      int num_triangles, AABB* s_c_aabb, AABB* s_p_aabb,
                      volatile AABB* aabbs, volatile PrimitiveID* ids,
                      int* num_leaves, bool pair_triangles)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tri_id = gid * 2;

    if (tri_id < num_triangles) {
        bool second_valid = (tri_id + 1) < num_triangles;
        Triangle* a = &triangles[tri_id];
        Triangle* b =
            second_valid ? &triangles[tri_id + 1] : &triangles[tri_id];

        // Calculate per primitive bounding boxes and centres
        AABB a_aabb = {fminf(fminf(a->v0, a->v1), a->v2),
                       fmaxf(fmaxf(a->v0, a->v1), a->v2)};
        AABB b_aabb = {fminf(fminf(b->v0, b->v1), b->v2),
                       fmaxf(fmaxf(b->v0, b->v1), b->v2)};
        float3 a_centre = (a_aabb.max + a_aabb.min) * 0.5f;
        float3 b_centre = (b_aabb.max + b_aabb.min) * 0.5f;

        // Calculate the combined bounding boxes and centres
        AABB p_aabb = Combine(a_aabb, b_aabb);
        AABB c_aabb = {fminf(a_centre, b_centre), fmaxf(a_centre, b_centre)};

        // Update the global bounds
        AtomicConvertCombine(*s_p_aabb, p_aabb);
        AtomicConvertCombine(*s_c_aabb, c_aabb);

        // Check if they have a shared edge
        Rotations r = {0, 0};
        bool merge = pair_triangles && second_valid &&
                     CanFormTrianglePair(*a, *b, r) &&
                     ShouldFormTrianglePair(a_aabb, b_aabb, p_aabb);

        // Reserve references
        unsigned num_valid = 1 + (second_valid && !merge);
        unsigned idx = atomicAdd(num_leaves, num_valid);

        // Write out the primitive aabbs and update refs
        if (merge) {
            AssignVolatile(aabbs[idx], p_aabb);
            triangles_out[idx] =
                CreateTrianglePair(a, b, tri_id, tri_id + 1, r);
        } else {
            AssignVolatile(aabbs[idx], a_aabb);
            triangles_out[idx + 0] = CreateTrianglePair(a, NULL, tri_id, 0, {0,0});
            triangles_out[idx + 1] =
                CreateTrianglePair(b, NULL, tri_id + 1, 0, {0,0});
        }
        // aabbs[tri_id] = merge ? p_aabb : a_aabb;
        ids[idx].id = idx;
        ids[idx].count = merge ? 2 : 1;

        if (second_valid && !merge) {
            // aabbs[idx + 1] = b_aabb;
            AssignVolatile(aabbs[idx + 1], b_aabb);
            ids[idx + 1].id = idx + 1;
            ids[idx + 1].count = 1;
        }
    }
}

__global__ void SetupSplits(Triangle* triangles, TrianglePair* triangles_out,
                            int num_triangles, int extra_leaf_thresh,
                            AABB* s_c_aabb, AABB* s_p_aabb,
                            volatile AABB* aabbs, volatile PrimitiveID* ids,
                            int* num_leaves, int* extra_leaves)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tri_id = gid;

    if (tri_id < num_triangles) {
        Triangle* a = &triangles[tri_id];
        AABB p_aabb_float = OrderedIntToFloat(*s_p_aabb);

        // Calculate per primitive bounding boxes and centres
        AABB aabb = {fminf(fminf(a->v0, a->v1), a->v2),
                     fmaxf(fmaxf(a->v0, a->v1), a->v2)};
        float3 centre = (aabb.max + aabb.min) * 0.5f;

        // Decide if it wants to be split
        int3 min_cell = CalculateGridcell(aabb.min, p_aabb_float);
        int3 max_cell = CalculateGridcell(aabb.max, p_aabb_float);
        bool split_a = !Equal(min_cell, max_cell);
        int3 range = max_cell - min_cell;
        uint num_extra_cells =
            (range.x + 1) * (range.y + 1) * (range.z + 1) - 1;
        if (split_a)
            split_a = (atomicAdd(extra_leaves, num_extra_cells) +
                       num_extra_cells) < extra_leaf_thresh;

        Rotations r = {0, 0};
        triangles_out[tri_id] = CreateTrianglePair(a, NULL, tri_id, 0, r);

        if (split_a) {
            int3 cell = min_cell;
            uint actual_cells = 0;

            while (1) {
                // Compute the intersection of the prim aabb and the cell (to be
                // replaced by clipper later)
                AABB cell_aabb = CellToBounds(cell, p_aabb_float);

                cell_aabb = aabb.Intersection(cell_aabb);

                // Update the global bounds
                AtomicConvertCombine(*s_p_aabb, cell_aabb);
                AtomicConvertCombine(*s_c_aabb,
                                     (cell_aabb.max + cell_aabb.min) * 0.5f);

                // Increment the counter
                unsigned idx = atomicAdd(num_leaves, 1);
                actual_cells++;

                // Write out the prim aabb
                aabbs[idx].min.x = cell_aabb.min.x;
                aabbs[idx].min.y = cell_aabb.min.y;
                aabbs[idx].min.z = cell_aabb.min.z;
                aabbs[idx].max.x = cell_aabb.max.x;
                aabbs[idx].max.y = cell_aabb.max.y;
                aabbs[idx].max.z = cell_aabb.max.z;

                // Write out the reference
                ids[idx].id = tri_id;
                ids[idx].count = 1;

                // Step to next cell
                if (GridNextCell(cell, min_cell, max_cell)) break;
            }
        } else {
            // Update the global bounds
            AtomicConvertCombine(*s_p_aabb, aabb);
            AtomicConvertCombine(*s_c_aabb, centre);

            // Increment the counter
            unsigned idx = atomicAdd(num_leaves, 1);

            // Write out the prim aabb
            aabbs[idx].min.x = aabb.min.x;
            aabbs[idx].min.y = aabb.min.y;
            aabbs[idx].min.z = aabb.min.z;
            aabbs[idx].max.x = aabb.max.x;
            aabbs[idx].max.y = aabb.max.y;
            aabbs[idx].max.z = aabb.max.z;

            // Write out the reference
            ids[idx].id = tri_id;
            ids[idx].count = 1;
        }
    }
}

__global__ void SetupPairSplits(Triangle* triangles,
                                TrianglePair* triangles_out, int num_triangles,
                                int extra_leaf_thresh, AABB* s_c_aabb,
                                AABB* s_p_aabb, volatile AABB* aabbs,
                                volatile PrimitiveID* ids,
                                int* num_triangles_out, int* num_leaves,
                                int* extra_leaves)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tri_id = gid * 2;

    if (tri_id < num_triangles) {
        bool second_valid = (tri_id + 1) < num_triangles;
        Triangle* a = &triangles[tri_id];
        Triangle* b =
            second_valid ? &triangles[tri_id + 1] : &triangles[tri_id];

        AABB p_aabb_float = OrderedIntToFloat(*s_p_aabb);

        // Calculate per primitive bounding boxes and centres
        AABB a_aabb = {fminf(fminf(a->v0, a->v1), a->v2),
                       fmaxf(fmaxf(a->v0, a->v1), a->v2)};
        AABB b_aabb = {fminf(fminf(b->v0, b->v1), b->v2),
                       fmaxf(fmaxf(b->v0, b->v1), b->v2)};
        AABB c_aabb = Combine(a_aabb, b_aabb);
        float3 a_centre = (a_aabb.max + a_aabb.min) * 0.5f;
        float3 b_centre = (b_aabb.max + b_aabb.min) * 0.5f;

        // Check if they have a shared edge
        Rotations r = {0, 0};
        bool merge = second_valid && CanFormTrianglePair(*a, *b, r) &&
                     ShouldFormTrianglePair(a_aabb, b_aabb, c_aabb);
        unsigned count = 1 + (second_valid && !merge);

        unsigned tri_block_index = atomicAdd(num_triangles_out, count);
        if (merge) {
            triangles_out[tri_block_index] =
                CreateTrianglePair(a, b, tri_id, tri_id + 1, r);
        } else {
            triangles_out[tri_block_index + 0] =
                CreateTrianglePair(a, NULL, tri_id, 0, {0,0});
            triangles_out[tri_block_index + 1] =
                CreateTrianglePair(b, NULL, tri_id + 1, 0, {0,0});
        }

        // Loop for one tri, two tris, or a pair
        for (unsigned i = 0; i < count; i++, tri_block_index++) {
            AABB& aabb = merge ? c_aabb : i ? b_aabb : a_aabb;

            // Decide if it wants to be split
            int3 min_cell = CalculateGridcell(aabb.min, p_aabb_float);
            int3 max_cell = CalculateGridcell(aabb.max, p_aabb_float);
            bool split = !Equal(min_cell, max_cell);
            int3 range = max_cell - min_cell;
            uint num_extra_cells =
                (range.x + 1) * (range.y + 1) * (range.z + 1) - 1;
            if (split)
                split = (atomicAdd(extra_leaves, num_extra_cells) +
                         num_extra_cells) < extra_leaf_thresh;

            if (split) {
                int3 cell = min_cell;
                uint actual_cells = 0;
                uint skipped_cells = 0;

                while (1) {
                    // Compute the intersection of the prim aabb and the cell
                    // (to be replaced by clipper later)
                    AABB cell_aabb = CellToBounds(cell, p_aabb_float);

                    // When merging, throw away the voxel if both prim AABBs
                    // don't overlap the cell
                    if (merge) {
                        if (!a_aabb.Intersection(cell_aabb).Valid() &&
                            !b_aabb.Intersection(cell_aabb).Valid()) {
                            skipped_cells++;
                            // atomicAdd(extra_leaves, -1); // Free up one
                            // reserved entry

                            // Step to next cell
                            if (GridNextCell(cell, min_cell, max_cell)) break;
                            continue;
                        }
                        cell_aabb = Combine(a_aabb.Intersection(cell_aabb),
                                            b_aabb.Intersection(cell_aabb));
                    } else {
                        cell_aabb = aabb.Intersection(cell_aabb);
                    }

                    // Update the global bounds
                    AtomicConvertCombine(*s_p_aabb, cell_aabb);
                    AtomicConvertCombine(
                        *s_c_aabb, (cell_aabb.max + cell_aabb.min) * 0.5f);

                    // Increment the counter
                    unsigned idx = atomicAdd(num_leaves, 1);
                    actual_cells++;

                    // Write out the prim aabb
                    aabbs[idx].min.x = cell_aabb.min.x;
                    aabbs[idx].min.y = cell_aabb.min.y;
                    aabbs[idx].min.z = cell_aabb.min.z;
                    aabbs[idx].max.x = cell_aabb.max.x;
                    aabbs[idx].max.y = cell_aabb.max.y;
                    aabbs[idx].max.z = cell_aabb.max.z;

                    // Write out the reference
                    ids[idx].id = tri_block_index;
                    ids[idx].count = merge ? 2 : 1;

                    // Step to next cell
                    if (GridNextCell(cell, min_cell, max_cell)) break;
                }
            } else {
                // Update the global bounds
                AtomicConvertCombine(*s_p_aabb, aabb);
                AtomicConvertCombine(*s_c_aabb, (aabb.max + aabb.min) * 0.5f);

                // Increment the counter
                unsigned idx = atomicAdd(num_leaves, 1);

                // Write out the prim aabb
                aabbs[idx].min.x = aabb.min.x;
                aabbs[idx].min.y = aabb.min.y;
                aabbs[idx].min.z = aabb.min.z;
                aabbs[idx].max.x = aabb.max.x;
                aabbs[idx].max.y = aabb.max.y;
                aabbs[idx].max.z = aabb.max.z;

                // Write out the reference
                ids[idx].id = tri_block_index;
                ids[idx].count = merge ? 2 : 1;
            }
        }
    }
}

// kernel: GridBlockCounts
//   Creates a histogram of primitive centroids in a coarse uniform grid across
//   the scene and generates per-block AABBs Parallelisation: multi-block, one
//   thread per input primitive
__global__ void GridBlockCounts(AABB* c_aabb, AABB* p_aabb, AABB* block_c_aabbs,
                                AABB* block_p_aabbs, AABB* aabbs,
                                int* block_counts, int* num_leaves)
{
    AABB empty_aabb = {make_float3(FLT_MAX), make_float3(-FLT_MAX)};
    if (threadIdx.x < NUM_BLOCKS) {
        shared_block_counts[threadIdx.x] = 0;
        shared_block_p_aabbs[threadIdx.x] = FloatToOrderedInt(empty_aabb);
        shared_block_c_aabbs[threadIdx.x] = FloatToOrderedInt(empty_aabb);
    }
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float epsilon = 1.1920929e-7;  // 2^-23

    if (gid < *num_leaves) {
        float3 centre = (aabbs[gid].min + aabbs[gid].max) * 0.5f;
        AABB c_aabb_float = OrderedIntToFloat(*c_aabb);

        int3 block_id = make_int3((centre - c_aabb_float.min) *
                                  (BLOCK_GRID_DIM * (1 - epsilon)) /
                                  (c_aabb_float.max - c_aabb_float.min));
        int block_id_1d = block_id.x + block_id.y * BLOCK_GRID_DIM +
                          block_id.z * BLOCK_GRID_DIM * BLOCK_GRID_DIM;

        atomicAdd(&shared_block_counts[block_id_1d], 1);
        AtomicConvertCombine(shared_block_p_aabbs[block_id_1d], aabbs[gid]);
        AtomicConvertCombine(shared_block_c_aabbs[block_id_1d], centre);
    }

    __syncthreads();

    if (threadIdx.x < NUM_BLOCKS) {
        atomicAdd(&block_counts[threadIdx.x], shared_block_counts[threadIdx.x]);
        AtomicCombine(block_p_aabbs[threadIdx.x],
                      shared_block_p_aabbs[threadIdx.x]);
        AtomicCombine(block_c_aabbs[threadIdx.x],
                      shared_block_c_aabbs[threadIdx.x]);
    }
}

// kernel: GridBlockScan
//   Creates an inclusive prefix sum of the block counts
//   Parallelisation: single block, one thread per element, logN iterations
__global__ void GridBlockScan(int* block_counts, int* block_scan, AABB* c_aabb,
                              AABB* p_aabb)
{
    __shared__ int x[NUM_BLOCKS];
    __shared__ int y[NUM_BLOCKS];

    x[threadIdx.x] = block_counts[threadIdx.x];
    const int iterations = __ffs(NUM_BLOCKS);
    __syncthreads();

    int k = threadIdx.x;
    for (int d = 1; d <= iterations; d += 2) {
        if (k >= (1 << (d - 1))) {
            y[k] = x[k - (1 << (d - 1))] + x[k];
        } else {
            y[k] = x[k];
        }
        __syncthreads();

        if (k >= (1 << d)) {
            x[k] = y[k - (1 << d)] + y[k];
        } else {
            x[k] = y[k];
        }
        __syncthreads();
    }

    block_scan[threadIdx.x] = x[threadIdx.x];

    if (threadIdx.x == 0) {
        *c_aabb = OrderedIntToFloat(*c_aabb);
        *p_aabb = OrderedIntToFloat(*p_aabb);
    }
}

// kernel: GridBlockDistribute
//   Distributes primitive references sorted by thier grid block, and convert
//   the per-block aabbs from int to float Parallelisation: multi-block, one
//   thread per primitive
__global__ void GridBlockDistribute(AABB* c_aabb, AABB* block_c_aabbs,
                                    AABB* block_p_aabbs, AABB* aabbs,
                                    int* num_leaves, int* block_hist,
                                    int* block_scan, volatile int* references,
                                    int* root_count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float epsilon = 1.1920929e-7;  // 2^-23

    if (gid < *num_leaves) {
        float3 centre = (aabbs[gid].min + aabbs[gid].max) * 0.5f;

        int3 block_id = make_int3((centre - c_aabb->min) *
                                  (BLOCK_GRID_DIM * (1 - epsilon)) /
                                  (c_aabb->max - c_aabb->min));
        int block_id_1d = block_id.x + block_id.y * BLOCK_GRID_DIM +
                          block_id.z * BLOCK_GRID_DIM * BLOCK_GRID_DIM;

        int idx = atomicAdd(&block_scan[block_id_1d], -1) - 1;
        references[2 * NUM_BLOCKS + idx] = gid;
    }

    if (blockIdx.x == 0 && threadIdx.x < NUM_BLOCKS) {
        block_c_aabbs[threadIdx.x] =
            OrderedIntToFloat(block_c_aabbs[threadIdx.x]);
        block_p_aabbs[threadIdx.x] =
            OrderedIntToFloat(block_p_aabbs[threadIdx.x]);

        if (block_hist[threadIdx.x] != 0) {
            int root_idx = atomicAdd(root_count, 1);
            references[root_idx] = threadIdx.x;
        }
    }
}