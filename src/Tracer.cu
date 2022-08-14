
#include "Tracer.cuh"
#include "helper_math.h"

__device__ bool IntersectRayAabb(const Node& node, const Ray& ray,
                                 float& distance)
{
    float3 inv_dir = 1.0f / ray.direction;
    float3 t1 = (node.min - ray.origin) * inv_dir;
    float3 t2 = (node.max - ray.origin) * inv_dir;
    float3 tmin = fminf(t1, t2);
    float3 tmax = fmaxf(t1, t2);
    float front = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float back = fminf(fminf(tmax.x, tmax.y), tmax.z);

    distance = front;
    return back >= front && front <= ray.tmax && back >= ray.tmin;
}

__device__ bool IntersectRayTriangle(Triangle& tri, Ray& ray)
{
    const float epsilon = 0.000000001f;

    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;
    float3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = cross(ray.direction, edge2);
    a = dot(edge1, h);

    if (a > -epsilon && a < epsilon) return false;

    f = 1.0f / a;
    s = ray.origin - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    q = cross(s, edge1);
    v = f * dot(ray.direction, q);
    if (v < 0.0f || (u + v) > 1.0f) return false;

    float t = f * dot(edge2, q);
    if (t < ray.tmin || t > ray.tmax) return false;

    ray.tmax = t;
    return true;
}

__device__ bool IntersectRayTrianglePair(TrianglePair& tri, Ray& ray, bool pair)
{
    Triangle tri_a(tri.v0, tri.v1, tri.v2);
    Triangle tri_b(tri.v2, tri.v1, tri.v3);
    bool hitA = IntersectRayTriangle(tri_a, ray);
    bool hitB = pair ? IntersectRayTriangle(tri_b, ray) : false;
    return hitA || hitB;
}

__device__ bool TraceRay(TrianglePair* triangles, Node* nodes, Ray& ray,
                         unsigned root, unsigned count, TraceStats& stats,
                         bool debug)
{
    bool tri_hit = false;
    unsigned stack_size = 1;
    StackEntry stack[64];
    stack[0] = {root, count};

    while (stack_size) {
        StackEntry entry = stack[--stack_size];
        unsigned num_hits = 0;
        StackEntry child_buffer;
        float child_dist;

        for (unsigned i = 0; i < entry.count; i++) {
            Node& node = nodes[entry.index + i];
            if (node.type == ChildType_None) continue;

            float dist;
            bool hit = IntersectRayAabb(node, ray, dist);

            bool is_leaf = node.type == ChildType_Tri;
            stats.box_tests++;

            if (hit && is_leaf) {
                // for (unsigned j = 0; j < node.count; j++) {
                stats.tri_tests++;
                bool hit_tri = IntersectRayTrianglePair(triangles[node.child],
                                                        ray, node.count > 0);
                tri_hit |= hit_tri;
                //}
            } else if (hit && num_hits == 0) {
                child_buffer = {node.child, node.count};
                child_dist = dist;
                num_hits++;
            } else if (hit) {
                if (dist < child_dist ||
                    (dist == child_dist && node.child > child_buffer.index)) {
                    StackEntry tmp = child_buffer;
                    child_buffer = {node.child, node.count};
                    child_dist = dist;

                    stack[stack_size++] = tmp;
                    if (stack_size >= 64) {
                        printf("Error: stack overflow");
                    }
                } else {
                    stack[stack_size++] = {node.child, node.count};
                    if (stack_size >= 64) {
                        printf("Error: stack overflow");
                    }
                }
            }
        }

        if (num_hits > 0) {
            stack[stack_size++] = child_buffer;
            if (stack_size >= 64) {
                printf("Error: stack overflow");
            }
        }
    }
    return tri_hit;
}

__global__ void TraceRays(TrianglePair* triangles, Node* nodes, Camera* camera,
                          uint32_t* num_tests, RenderType render_type,
                          cudaSurfaceObject_t image, unsigned root,
                          unsigned count)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned w = blockDim.x * gridDim.x;
    unsigned h = blockDim.y * gridDim.y;

    float2 coord = make_float2((float)x, (float)y);
    float2 ndc = 2 * ((coord + 0.5f) / make_float2(w, h)) - 1;
    float3 ndc3 = make_float3(ndc, 1);
    float3 p = (ndc.x * camera->u) + (ndc.y * camera->v) + (ndc3.z * camera->w);

    float max_depth = camera->max_depth;

    Ray ray;
    ray.direction = normalize(p);
    ray.origin = camera->position;
    ray.tmin = 0.00001f;
    ray.tmax = max_depth;

    TraceStats stats = {};
    stats.box_tests = 0;
    bool hit = TraceRay(triangles, nodes, ray, root, count, stats, false);
    float depth = hit ? ray.tmax : 0.0f;
    atomicAdd(num_tests, stats.box_tests);

    uchar4 colour;
    if (render_type == RenderType::kDepth) {
        colour.x = min(1.0f, depth / max_depth) * 255;
        colour.y = min(1.0f, depth / max_depth) * 255;
        colour.z = min(1.0f, depth / max_depth) * 255;
        colour.w = 255;
    } else if (render_type == RenderType::kBoxtests) {
        colour.x = 0;
        colour.y = min(stats.box_tests / 180.0f, 1.0f) * 255;
        colour.z = min(stats.box_tests / 180.0f, 1.0f) * 255;
        colour.w = 255;
    } else if (render_type == RenderType::kTriangleTests) {
        colour.x = min(stats.tri_tests / 32.0f, 1.0f) * 100;
        colour.y = min(stats.tri_tests / 32.0f, 1.0f) * 255;
        colour.z = min(stats.tri_tests / 32.0f, 1.0f) * 100;
        colour.w = 255;
    }

    surf2Dwrite(colour, image, x * 4, y);
}