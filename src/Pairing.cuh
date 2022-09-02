
#include "Common.cuh"

__device__ inline float3 Get(const Triangle& t, int i)
{
    return i == 0 ? t.v0 : i == 1 ? t.v1 : t.v2;
}

__device__ inline Triangle RotateTriangle(const Triangle& a, int rot)
{
    switch (rot) {
        case 0:
            return a;
        case 1:
            return Triangle(a.v2, a.v0, a.v1);
        case 2:
            return Triangle(a.v1, a.v2, a.v0);
        default:
            return a;
    }
}

// Check if triangle t shares an edge with edge (a->b)
// Returns how many steps to rotate t such that v0==t.v0 && v1==t.v1, if there
// is a shared edge Returns -1 if there is no shared edge
__device__ inline int FindSharedEdge(const float3& a, const float3& b,
                                     const Triangle& t)
{
    if (Equal(a, t.v0) && Equal(b, t.v1)) return 0;
    if (Equal(a, t.v1) && Equal(b, t.v2)) return 2;
    if (Equal(a, t.v2) && Equal(b, t.v0)) return 1;
    return -1;
}

__device__ inline bool ShouldFormTrianglePair(const AABB& a, const AABB& b,
                                              const AABB& p)
{
    return sa(p) * 0.5f < sa(a) + sa(b);
}

// Checks if two triangles are able to form a triangle pair
__device__ inline bool CanFormTrianglePair(const Triangle& a, const Triangle& b,
                                           Rotations& r)
{
    int t0_rotate = 3;
    int t1_rotate = -1;
    for (uint32_t u = 2, v = 0; v < 3; u = v, v++) {
        t1_rotate = FindSharedEdge(Get(a, v), Get(a, u), b);
        t0_rotate--;
        if (t1_rotate != -1) break;
    }
    if (t1_rotate == -1) return false;

    r.rot_a = t0_rotate;
    r.rot_b = t1_rotate;

    return true;
}

__device__ inline TrianglePair CreateTrianglePair(const Triangle* a,
                                                  const Triangle* b,
                                                  uint32_t a_id, uint32_t b_id,
                                                  Rotations r)
{
    if (b == NULL) {
        return TrianglePair(a->v0, a->v1, a->v2, a->v2, a_id, 0, r.rot_a,
                            r.rot_b);
    }
    Triangle a_rotated = RotateTriangle(*a, r.rot_a);

    TrianglePair result = TrianglePair(a_rotated.v0, a_rotated.v1, a_rotated.v2,
                                       r.rot_b == 2   ? b->v0
                                       : r.rot_b == 1 ? b->v1
                                                      : b->v2,
                                       a_id, b_id, r.rot_a, r.rot_b);

    return result;
}