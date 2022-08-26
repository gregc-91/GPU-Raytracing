#include <stdio.h>

#include "BottomUpBuilder.cuh"
#include "Multiblock.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ static float OrderedIntToFloat(int i)
{
    return __int_as_float((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

__device__ static float3 OrderedIntToFloat(float3 a)
{
    return make_float3(OrderedIntToFloat(__float_as_int(a.x)),
                       OrderedIntToFloat(__float_as_int(a.y)),
                       OrderedIntToFloat(__float_as_int(a.z)));
}

__device__ static AABB OrderedIntToFloat(AABB a)
{
    return {OrderedIntToFloat(a.min), OrderedIntToFloat(a.max)};
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old =
            ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old =
            ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

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

__device__ static bool Equal(const float3& a, const float3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Check if triangle t shares an edge with edge (a->b)
// Returns how many steps to
//  t such that v0==t.v0 && v1==t.v1, if there is a shared edge
// Returns -1 if there is no shared edge
__device__ int FindSharedEdge(const float3& a, const float3& b, const float3* t)
{
    if (Equal(a, t[0]) && Equal(b, t[1])) return 0;
    if (Equal(a, t[1]) && Equal(b, t[2])) return 2;
    if (Equal(a, t[2]) && Equal(b, t[0])) return 1;
    return -1;
}

// Checks if two triangles are able to form a triangle pair
__device__ bool CanFormTrianglePair(const float3* a, const float3* b,
                                    Rotations& r)
{
    int t0_rotate = 3;
    int t1_rotate = -1;
    for (uint32_t u = 2, v = 0; v < 3; u = v, v++) {
        t1_rotate = FindSharedEdge(a[v], a[u], b);
        t0_rotate--;
        if (t1_rotate != -1) break;
    }
    if (t1_rotate == -1) return false;

    r.rot_a = t0_rotate;
    r.rot_b = t1_rotate;

    return true;
}

// Checks if the combined surface area is too large
// Creating large boxes is unlikely to give a quality result
__device__ static bool ShouldFormTrianglePair(const AABB& a, const AABB& b,
                                              const AABB& p)
{
    return sa(p) * 0.7f < sa(a) + sa(b);
}

__device__ Triangle RotateTriangle(const float3* a, int rot)
{
    switch (rot) {
        case 0:
            return Triangle(a[0], a[1], a[2]);
        case 1:
            return Triangle(a[2], a[0], a[1]);
        case 2:
            return Triangle(a[1], a[2], a[0]);
        default:
            return Triangle(a[0], a[1], a[2]);
    }
}

__device__ static TrianglePair CreateTrianglePair(const float3* a,
                                                  const float3* b,
                                                  uint32_t a_id, uint32_t b_id,
                                                  Rotations r)
{
    if (b == NULL) {
        return TrianglePair(a[0], a[1], a[2], a[2], a_id, 0);
    }
    Triangle a_rotated = RotateTriangle(a, r.rot_a);

    TrianglePair result = TrianglePair(a_rotated.v0, a_rotated.v1, a_rotated.v2,
                                       r.rot_b == 2   ? b[0]
                                       : r.rot_b == 1 ? b[1]
                                                      : b[2],
                                       a_id, b_id);

    return result;
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
    float3* a = &vertices[tid * 3];
    float3* b = second_valid ? &vertices[(tid + 1) * 3] : &vertices[tid * 3];
    AABB a_aabb = {fminf(fminf(a[0], a[1]), a[1]),
                   fmaxf(fmaxf(a[0], a[1]), a[2])};
    AABB b_aabb = {fminf(fminf(b[0], b[1]), b[2]),
                   fmaxf(fmaxf(b[0], b[1]), b[2])};
    AABB c_aabb = Combine(a_aabb, b_aabb);

    // Check if they have a shared edge
    Rotations r = {0, 0};
    bool merge = second_valid && CanFormTrianglePair(a, b, r) &&
                 ShouldFormTrianglePair(a_aabb, b_aabb, c_aabb);

    // Reserve references
    unsigned num_valid = 1 + (second_valid && !merge);
    unsigned idx = atomicAdd(num_leaves, num_valid);

    // Compute the centres
    float3 centre = (a[0] + a[1] + a[2]) / 3.0f;
    float3 centre2 = (b[0] + b[1] + b[2]) / 3.0f;
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

    float3* a = &vertices[index * 3];
    float3* b = &vertices[(index + 1) * 3];

    Rotations r;
    TrianglePair result;
    if (is_pair) {
        CanFormTrianglePair(a, b, r);
        result = CreateTrianglePair(a, b, index, index + 1, r);
    } else {
        result.v0 = a[0];
        result.v1 = a[1];
        result.v2 = a[2];
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
