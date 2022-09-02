#include <stdio.h>

#include "BottomUpBuilder.cuh"
#include "DeviceUtils.cuh"
#include "Multiblock.cuh"
#include "Pairing.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int ExpandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int Morton3D(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = ExpandBits((unsigned int)x);
    unsigned int yy = ExpandBits((unsigned int)y);
    unsigned int zz = ExpandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__device__ int cpl(unsigned* codes, unsigned i, unsigned j)
{
    return codes[i] == codes[j] ? 32 + __clz(i ^ j)
                                : __clz(codes[i] ^ codes[j]);
}

__device__ int Sign(int a) { return a >= 0 ? 1 : -1; }

__device__ int2 DetermineRange(unsigned* codes, unsigned count, int i)
{
    if (i == 0) {
        return make_int2(0, count - 1);
    }

    // Determine direction of the range (+1 or -1)
    int d = Sign(cpl(codes, i, i + 1) - cpl(codes, i, i - 1));

    // Compute upper bound for the length of the range
    int cpl_min = cpl(codes, i, i - d);
    int lmax = 2;
    while ((i + lmax * d) >= 0 && i + lmax * d < count &&
           cpl(codes, i, i + lmax * d) > cpl_min)
        lmax *= 2;

    // Find the other end using binary search
    int l = 0;
    for (int t = lmax >> 1; t; t >>= 1) {
        if ((i + (l + t) * d) >= 0 && i + (l + t) * d < count &&
            cpl(codes, i, i + (l + t) * d) > cpl_min)
            l += t;
    }
    int j = i + l * d;

    return d > 0 ? make_int2(i, j) : make_int2(j, i);
}

__device__ int FindSplit(unsigned* sorted_morton_codes, int first, int last)
{
    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int common_prefix = cpl(sorted_morton_codes, first, last);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than common_prefix bits with the first one.

    int split = first;  // initial guess
    int step = last - first;

    do {
        step = (step + 1) >> 1;        // exponential decrease
        int new_split = split + step;  // proposed new position

        if (new_split < last) {
            int splitPrefix = cpl(sorted_morton_codes, first, new_split);
            if (splitPrefix > common_prefix)
                split = new_split;  // accept proposal
        }
    } while (step > 1);

    return split;
}

__global__ void GenerateMortonCodes(unsigned* codes, unsigned* values,
                                    float3* vertices, AABB* p_aabb,
                                    unsigned count)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        float3 centre = (vertices[gid * 3] + vertices[gid * 3 + 1] +
                         vertices[gid * 3 + 2]) /
                        3.0f;

        AABB scene_aabb = OrderedIntToFloat(*p_aabb);
        centre = (centre - scene_aabb.min) / (scene_aabb.max - scene_aabb.min);
        centre = clamp(centre, 0.0f, 1.0f);

        codes[gid] = Morton3D(centre.x, centre.y, centre.z);
        values[gid] = gid;
    }
}

__global__ void GenerateMortonCodesPairs(unsigned* codes, unsigned* values,
                                         float3* vertices, AABB* p_aabb,
                                         unsigned* num_leaves, unsigned count)
{
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid = gid * 2;

    if (tid >= count) return;

    bool second_valid = (tid + 1) < count;
    Triangle a = Triangle(&vertices[tid * 3]);
    Triangle b = second_valid ? Triangle(&vertices[(tid + 1) * 3])
                              : Triangle(&vertices[tid * 3]);
    AABB a_aabb = AABB(a);
    AABB b_aabb = AABB(b);
    AABB c_aabb = Combine(a_aabb, b_aabb);

    // Check if they have a shared edge
    Rotations r = {0, 0};
    bool merge = second_valid && CanFormTrianglePair(a, b, r) &&
                 ShouldFormTrianglePair(a_aabb, b_aabb, c_aabb);

    // Reserve references
    unsigned num_valid = 1 + (second_valid && !merge);
    unsigned idx = atomicAdd(num_leaves, num_valid);

    // Compute the centres
    float3 centre = a.Centre();
    float3 centre2 = b.Centre();
    if (merge) centre = (centre + centre2) * 0.5f;

    AABB scene_aabb = OrderedIntToFloat(*p_aabb);
    centre = (centre - scene_aabb.min) / (scene_aabb.max - scene_aabb.min);
    centre = clamp(centre, 0.0f, 1.0f);

    // Use the MSB to indicate it's a pair
    values[idx] = merge ? tid | 0x80000000 : tid;
    codes[idx] = Morton3D(centre.x, centre.y, centre.z);

    if (second_valid && !merge) {
        centre2 =
            (centre2 - scene_aabb.min) / (scene_aabb.max - scene_aabb.min);
        centre2 = clamp(centre2, 0.0f, 1.0f);

        values[idx + 1] = tid + 1;
        codes[idx + 1] = Morton3D(centre2.x, centre2.y, centre2.z);
    }
}

// Node 0 will be the root
__global__ void GenerateHierarchy(volatile Node* nodes, unsigned* leaf_indices,
                                  unsigned* sorted_morton_codes,
                                  unsigned* sorted_indices, int num_objects)
{
    // Construct internal nodes.

    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_objects - 1) {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)
        int2 range = DetermineRange(sorted_morton_codes, num_objects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range.

        int split = FindSplit(sorted_morton_codes, first, last);

        // Select child_a.

        uint32_t child_a = (split == first) ? split : split * 2;
        ChildType type_a = (split == first) ? ChildType_Tri : ChildType_Box;

        // Select child_b.

        uint32_t child_b = (split + 1 == last) ? split + 1 : (split + 1) * 2;
        ChildType type_b = (split + 1 == last) ? ChildType_Tri : ChildType_Box;

        // Record parent-child relationships.

        nodes[idx * 2 + 0].child = child_a;
        nodes[idx * 2 + 1].child = child_b;
        nodes[idx * 2 + 0].type = type_a;
        nodes[idx * 2 + 1].type = type_b;
        // nodes[idx * 2 + 0].count = type_a == ChildType_Box ? 2 : 1;
        // nodes[idx * 2 + 1].count = type_b == ChildType_Box ? 2 : 1;
        if (type_a == ChildType_Box) {
            nodes[child_a + 0].parent = idx << 1;
            nodes[child_a + 1].parent = idx << 1;
        } else
            leaf_indices[split] = idx << 1;
        if (type_b == ChildType_Box) {
            nodes[child_b + 0].parent = (idx << 1) + 1;
            nodes[child_b + 1].parent = (idx << 1) + 1;
        } else
            leaf_indices[split + 1] = (idx << 1) + 1;
    }
}

__device__ void UpdateAABB(volatile Node* nodes, unsigned index, unsigned count,
                           float3& cur_min, float3& cur_max, int right)
{
    volatile Node& node = nodes[index];
    volatile float3& box_min = node.min;
    volatile float3& box_max = node.max;
    unsigned child = node.child;

    if (nodes[index].type == ChildType_Box) {
        f3min(cur_min, cur_min, nodes[child + 1 - right].min);
        f3max(cur_max, cur_max, nodes[child + 1 - right].max);

        assign(box_min, cur_min);
        assign(box_max, cur_max);
    }
}

__device__ void UpdateAABB(volatile Node* nodes, unsigned index, unsigned count)
{
    volatile Node& node = nodes[index];
    volatile float3& box_min = node.min;
    volatile float3& box_max = node.max;
    unsigned child = node.child;

    if (nodes[index].type == ChildType_Box) {
        f3min(box_min, nodes[child].min, nodes[child + 1].min);
        f3max(box_max, nodes[child].max, nodes[child + 1].max);
    }
}

__global__ void GenerateAABBs(volatile Node* nodes, unsigned* leaf_indices,
                              unsigned* sorted_indices, unsigned* locks,
                              TrianglePair* triangles, unsigned count)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;

    unsigned leaf_index = leaf_indices[gid];

    // Get the AABB for the leaf and write into the node

    bool is_pair = sorted_indices[gid] >> 31;
    float3 v0 = triangles[gid].v0;
    float3 v1 = triangles[gid].v1;
    float3 v2 = triangles[gid].v2;
    float3 aabb_min = f3min(f3min(v0, v1), v2);
    float3 aabb_max = f3max(f3max(v0, v1), v2);
    if (is_pair) {
        aabb_min = f3min(aabb_min, triangles[gid].v3);
        aabb_max = f3max(aabb_max, triangles[gid].v3);
    }

    assign(nodes[leaf_index].min, aabb_min);
    assign(nodes[leaf_index].max, aabb_max);
    nodes[leaf_index].count = 1;

    unsigned index = leaf_index;

    while (index > 1) {
        // Decide who will continue up
        if (!atomicAdd(&locks[index >> 1], 1)) return;

        unsigned p = nodes[index].parent;

        UpdateAABB(nodes, p, count, aabb_min, aabb_max, index & 1);
        nodes[p].count = 2;
        index = p;
    }
}

__global__ void GenerateTriangles(unsigned* sorted_indices, float3* vertices,
                                  TrianglePair* triangles, unsigned count)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;

    bool is_pair = sorted_indices[gid] >> 31;
    unsigned index = sorted_indices[gid] & 0x7FFFFFFF;

    Triangle a(&vertices[index * 3]);
    Triangle b(&vertices[(index + 1) * 3]);

    Rotations r;
    TrianglePair result;
    if (is_pair) {
        CanFormTrianglePair(a, b, r);
        result = CreateTrianglePair(&a, &b, index, index + 1, r);
    } else {
        result.v0 = a.v0;
        result.v1 = a.v1;
        result.v2 = a.v2;
        result.v3 = result.v2;
    }

    triangles[gid] = result;
}

__global__ void ExtractDepth(Node* nodes, PrimitiveID* output_indices,
                             AABB* output_aabbs, int* output_count,
                             AABB* c_aabb, AABB* p_aabb, unsigned root,
                             unsigned target_depth, unsigned count)
{
    unsigned current_depth = 0;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned current_index = root;

    // Also convert the scene aabb from int to float
    if (tid == 0) {
        *p_aabb = OrderedIntToFloat(*p_aabb);
    }

    while (current_depth < target_depth) {
        // Leaf node in one of the boxes
        if (nodes[current_index].type == ChildType_Tri ||
            nodes[current_index + 1].type == ChildType_Tri) {
            if ((tid >> current_depth) == 0) {
                unsigned write_index = atomicAdd(output_count, 1);
                output_indices[write_index] = {current_index, 2};
                output_aabbs[write_index] = {
                    f3min(nodes[current_index].min,
                          nodes[current_index + 1].min),
                    f3max(nodes[current_index].max,
                          nodes[current_index + 1].max),
                };
                atomicMin(&c_aabb->min.x, output_aabbs[write_index].min.x);
                atomicMin(&c_aabb->min.y, output_aabbs[write_index].min.y);
                atomicMin(&c_aabb->min.z, output_aabbs[write_index].min.z);
                atomicMax(&c_aabb->max.x, output_aabbs[write_index].max.x);
                atomicMax(&c_aabb->max.y, output_aabbs[write_index].max.y);
                atomicMax(&c_aabb->max.z, output_aabbs[write_index].max.z);
            }
            return;
        }

        unsigned direction = (tid >> current_depth) & 1;
        current_index = direction ? nodes[current_index].child
                                  : nodes[current_index + 1].child;

        current_depth++;
    }

    unsigned write_index = atomicAdd(output_count, 1);
    output_indices[write_index] = {current_index, 2};
    output_aabbs[write_index] = {
        f3min(nodes[current_index].min, nodes[current_index + 1].min),
        f3max(nodes[current_index].max, nodes[current_index + 1].max),
    };

    atomicMin(&c_aabb->min.x, output_aabbs[write_index].min.x);
    atomicMin(&c_aabb->min.y, output_aabbs[write_index].min.y);
    atomicMin(&c_aabb->min.z, output_aabbs[write_index].min.z);
    atomicMax(&c_aabb->max.x, output_aabbs[write_index].max.x);
    atomicMax(&c_aabb->max.y, output_aabbs[write_index].max.y);
    atomicMax(&c_aabb->max.z, output_aabbs[write_index].max.z);
}
