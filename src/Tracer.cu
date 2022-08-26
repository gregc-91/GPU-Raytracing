
#include "Tracer.cuh"
#include "helper_math.h"

struct RayResult {
    uint32_t primitive_id;
    float2 barycentrics;
};

__device__ float2 InterpolateUVs(Attributes& attributes, float2 barycentrics)
{
    return attributes.uv[0] * (1 - barycentrics.x - barycentrics.y) +
           attributes.uv[1] * barycentrics.x +
           attributes.uv[2] * barycentrics.y;
}

__device__ float3 InterpolateNormals(Attributes& attributes,
                                     float2 barycentrics)
{
    return attributes.normal[0] * (1 - barycentrics.x - barycentrics.y) +
           attributes.normal[1] * barycentrics.x +
           attributes.normal[2] * barycentrics.y;
}

__device__ float4 Sample(Material& material, int2 xy, int lod)
{
    if (material.textures[lod] == NULL) {
        return make_float4(255, 0, 255, 255);
    }

    xy = clamp(xy, make_int2(0), material.texture_sizes[lod] - 1);
    return make_float4(
        material.textures[lod][xy.y * material.texture_sizes[lod].x + xy.x]);
}

__device__ uchar4 Sample(Material& material, float2 uv, int lod)
{
    float2 coord = fracf(uv) * make_float2(material.texture_sizes[lod] - 1);
    int2 icoord =
        make_int2(coord.x, material.texture_sizes[lod].y - coord.y - 1);

    if (material.textures[lod] == NULL) {
        return make_uchar4(255, 0, 255, 255);
    }

    return make_uchar4(Sample(material, icoord, lod));
}

__device__ uchar4 BilinearSample(Material& m, float2 uv, int lod)
{
    float2 coord = fracf(uv) * make_float2(m.texture_sizes[lod]) - 0.5f;
    coord = make_float2(coord.x, m.texture_sizes[lod].y - coord.y);
    int2 i0 = make_int2(coord);
    int2 i1 = i0 + make_int2(1, 0);
    int2 i2 = i0 + make_int2(0, -1);
    int2 i3 = i0 + make_int2(1, -1);

    float2 d = coord - make_float2(make_int2(coord));

    float w0 = (1.0f - d.x) * d.y;
    float w1 = d.x * d.y;
    float w2 = (1.0f - d.x) * (1.0f - d.y);
    float w3 = d.x * (1.0f - d.y);

    return make_uchar4(Sample(m, i0, lod) * w0 + Sample(m, i1, lod) * w1 +
                       Sample(m, i2, lod) * w2 + Sample(m, i3, lod) * w3);
}

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

__device__ float4 RayTriangleGradients(Triangle& tri, Ray& ray, float spread)
{
    float3 v0 = tri.v0;
    float3 v1 = tri.v1;
    float3 v2 = tri.v2;
    float3 edge1, edge2, h0, h1, s, q;
    float a0, f0, bu0, bv0, a1, f1, bu1, bv1;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    s = ray.origin - v0;
    q = cross(s, edge1);

    float3 x = normalize(cross(ray.direction, make_float3(0, 1, 0))) *
               ray.tmax * spread;
    float3 y = normalize(cross(ray.direction, x)) * ray.tmax * spread;
    float3 hit_point = (ray.origin + ray.direction * ray.tmax);

    float3 dirx = normalize((hit_point + x) - ray.origin);
    float3 diry = normalize((hit_point + y) - ray.origin);

    h0 = cross(dirx, edge2);
    a0 = dot(edge1, h0);
    f0 = 1.0f / a0;
    bu0 = f0 * dot(s, h0);
    bv0 = f0 * dot(dirx, q);

    h1 = cross(diry, edge2);
    a1 = dot(edge1, h1);
    f1 = 1.0f / a1;
    bu1 = f1 * dot(s, h1);
    bv1 = f1 * dot(diry, q);

    return make_float4(bu0, bv0, bu1, bv1);
}

__device__ float ComputeLOD(Ray& ray, RayResult& ray_result, float spread,
                            TrianglePair& tri, Attributes& attribs,
                            Material& mat)
{
    Triangle t(tri.v0, tri.v1, tri.v2);
    float4 spread_barys = RayTriangleGradients(t, ray, spread);

    float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
    float2 uvs_x =
        InterpolateUVs(attribs, make_float2(spread_barys.x, spread_barys.y));
    float2 uvs_y =
        InterpolateUVs(attribs, make_float2(spread_barys.z, spread_barys.w));

    float2 dtdx = fabs(uvs_x - uvs) * make_float2(mat.texture_sizes[0]);
    float2 dtdy = fabs(uvs_y - uvs) * make_float2(mat.texture_sizes[0]);
    float max_change = fmaxf(fmaxf(dtdx.x, dtdx.y), fmaxf(dtdy.x, dtdy.y));
    float lod = clamp(log2f(max_change), 0.0f, float(mat.max_lod));
    return lod;
}

__device__ bool IntersectRayTriangle(Triangle& tri, Ray& ray,
                                     RayResult& ray_result, uint32_t prim_id)
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
    ray_result.primitive_id = prim_id;
    ray_result.barycentrics = make_float2(u, v);
    return true;
}

__device__ bool IntersectRayTrianglePair(TrianglePair& tri, Ray& ray,
                                         RayResult& ray_result, bool pair)
{
    Triangle tri_a(tri.v0, tri.v1, tri.v2);
    Triangle tri_b(tri.v2, tri.v1, tri.v3);
    bool hitA =
        IntersectRayTriangle(tri_a, ray, ray_result, tri.primitive_id_0);
    bool hitB =
        pair ? IntersectRayTriangle(tri_b, ray, ray_result, tri.primitive_id_1)
             : false;
    return hitA || hitB;
}

__device__ bool TraceRay(TrianglePair* triangles, Node* nodes,
                         Attributes* attributes, Ray& ray,
                         RayResult& ray_result, unsigned root, unsigned count,
                         TraceStats& stats, bool debug)
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
                bool hit_tri = IntersectRayTrianglePair(
                    triangles[node.child], ray, ray_result, node.count > 0);
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

__device__ uchar4 AmbientShader(TrianglePair* triangles, Node* nodes,
                                uint32_t root, uint32_t count,
                                Attributes* attributes, Ray& ray,
                                RayResult& ray_result, Material& mat,
                                Attributes& attribs, float3 light, float spread,
                                bool textures, bool shadows)
{
    float3 light_colour = {1.0f, 0.9f, 0.8f};
    float3 light_pos = light;

    float3 hit_pos = ray.origin + ray.direction * ray.tmax;
    float3 normal = InterpolateNormals(attribs, ray_result.barycentrics);

    float3 light_dir = normalize(light_pos - hit_pos);

    float ambient_strength = 0.2f;
    float diffuse_strength = 0.8f;

    float3 ambient = light_colour * ambient_strength;
    float3 diffuse =
        max(dot(normal, light_dir), 0.0f) * light_colour * diffuse_strength;

    float3 object_colour = mat.diffuse;
    if (textures && mat.textures[0]) {
        float lod =
            ComputeLOD(ray, ray_result, spread,
                       triangles[ray_result.primitive_id], attribs, mat);
        uchar4 tex = BilinearSample(
            mat, InterpolateUVs(attribs, ray_result.barycentrics), lod);
        object_colour.x *= float(tex.x) / 255;
        object_colour.y *= float(tex.y) / 255;
        object_colour.z *= float(tex.z) / 255;
    }

    if (shadows) {
        Ray shadow_ray;
        RayResult shadow_result;
        TraceStats shadow_stats = {0};
        shadow_ray.origin = hit_pos;
        shadow_ray.direction = light_dir;
        shadow_ray.tmin = 0.001f;
        shadow_ray.tmax = length(light_pos - hit_pos);

        bool hit = TraceRay(triangles, nodes, attributes, shadow_ray,
                            shadow_result, root, count, shadow_stats, false);

        if (hit) diffuse = make_float3(0.0f);
    }

    float3 colour = diffuse * object_colour + ambient * object_colour;

    return make_uchar4(colour.x * 255, colour.y * 255, colour.z * 255, 255);
}

__global__ void TraceRays(TrianglePair* triangles, Node* nodes,
                          Attributes* attributes, Material* materials,
                          Camera* camera, uint32_t* num_tests,
                          RenderType render_type, cudaSurfaceObject_t image,
                          unsigned root, unsigned count, float3 light)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned w = blockDim.x * gridDim.x;
    unsigned h = blockDim.y * gridDim.y;

    float2 coord = make_float2((float)x, (float)y);
    float2 ndc = 2 * ((coord + 0.5f) / make_float2(w, h)) - 1;
    float3 ndc3 = make_float3(ndc, 1);
    float3 p = (ndc.x * camera->u) + (ndc.y * camera->v) + (ndc3.z * camera->w);
    float spread = 2.0f / w;

    float max_depth = camera->max_depth;

    Ray ray;
    ray.direction = normalize(p);
    ray.origin = camera->position;
    ray.tmin = 0.00001f;
    ray.tmax = max_depth;

    RayResult ray_result;
    ray_result.primitive_id = 0;
    TraceStats stats = {};
    stats.box_tests = 0;
    bool hit = TraceRay(triangles, nodes, attributes, ray, ray_result, root,
                        count, stats, false);
    float depth = hit ? ray.tmax : 0.0f;
    atomicAdd(num_tests, stats.box_tests);

    uchar4 colour;
    TrianglePair& pair = triangles[ray_result.primitive_id];
    Attributes& attribs = attributes[ray_result.primitive_id];
    Material& mat = materials[attribs.material_id];

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
    } else if (render_type == RenderType::kDiffuse) {
        if (hit) {
            colour =
                AmbientShader(triangles, nodes, root, count, attributes, ray,
                              ray_result, materials[attribs.material_id],
                              attribs, light, spread, false, false);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kLODs) {
        if (mat.textures[0]) {
            float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
            float lod =
                ComputeLOD(ray, ray_result, 2.0f / w, pair, attribs, mat);
            colour = make_uchar4((unsigned char)(int(lod) * 20));
        } else {
            colour = make_uchar4(255, 0, 255, 255);
        }
    } else if (render_type == RenderType::kTexture) {
        if (hit) {
            if (mat.textures[0]) {
                float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
                float lod =
                    ComputeLOD(ray, ray_result, 2.0f / w, pair, attribs, mat);

                colour = BilinearSample(mat, uvs, lod);
            } else {
                colour.x = mat.diffuse.x * 255;
                colour.y = mat.diffuse.y * 255;
                colour.z = mat.diffuse.z * 255;
                colour.w = 255;
            }
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kTextureLit) {
        if (hit) {
            colour =
                AmbientShader(triangles, nodes, root, count, attributes, ray,
                              ray_result, materials[attribs.material_id],
                              attribs, light, spread, true, false);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kTextureLitShadows) {
        if (hit) {
            colour =
                AmbientShader(triangles, nodes, root, count, attributes, ray,
                              ray_result, materials[attribs.material_id],
                              attribs, light, spread, true, true);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    }

    surf2Dwrite(colour, image, x * 4, y);
}