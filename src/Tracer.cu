
#include "Tracer.cuh"
#include "helper_math.h"

struct RayResult {
    uint32_t primitive_id;
    uint32_t pair_id;
    float2 barycentrics;
};

struct float3x3 {
    float3 row[3];
};

__device__ uchar3 HsvToRgb(float h, float s, float v)
{
    h = clamp(h, 0.f, 1.f) * 360.0f;
    s = clamp(s, 0.f, 1.f);
    v = clamp(v, 0.f, 1.f);

    float c = s * v;
    float x = c * (1 - fabsf((int(h) % 120) / 60.0f - 1));
    float m = v - c;
    float3 rgb;
    if (h >= 0 && h < 60) {
        rgb = {c, x, 0};
    } else if (h >= 60 && h < 120) {
        rgb = {x, c, 0};
    } else if (h >= 120 && h < 180) {
        rgb = {0, c, x};
    } else if (h >= 180 && h < 240) {
        rgb = {0, x, c};
    } else if (h >= 240 && h < 300) {
        rgb = {x, 0, c};
    } else {
        rgb = {c, 0, x};
    }

    return make_uchar3((rgb + m) * 255);
}

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

__device__ float3x3 TangentMatrix(Triangle& triangle, Attributes& attributes)
{
    float3 e1 = triangle.v1 - triangle.v0;
    float3 e2 = triangle.v2 - triangle.v0;
    float2 duv1 = attributes.uv[1] - attributes.uv[0];
    float2 duv2 = attributes.uv[2] - attributes.uv[0];

    float f = 1.0f / (duv1.x * duv2.y - duv1.y * duv2.x);

    float3 normal = normalize(cross(e1, e2));
    float3 tangent = normalize(f * (e1 * duv2.y - e2 * duv1.y));
    float3 bitangent = normalize(f * (e2 * duv1.x - e1 * duv2.x));

    // Stored in row major order to make mult easier
    return {make_float3(tangent.x, bitangent.x, normal.x),
            make_float3(tangent.y, bitangent.y, normal.y),
            make_float3(tangent.z, bitangent.z, normal.z)};
}

__device__ float4 Sample(Texture& texture, int2 xy, int lod)
{
    xy = clamp(xy, make_int2(0), texture.sizes[lod] - 1);
    return make_float4(
        texture.gpu_mips[lod][xy.y * texture.sizes[lod].x + xy.x]);
}

__device__ uchar4 Sample(Texture& texture, float2 uv, int lod)
{
    float2 coord = fracf(uv) * make_float2(texture.sizes[lod] - 1);
    int2 icoord = make_int2(coord.x, texture.sizes[lod].y - coord.y - 1);

    if (texture.gpu_mips[lod] == NULL) {
        return make_uchar4(255, 0, 255, 255);
    }

    return make_uchar4(Sample(texture, icoord, lod));
}

__device__ uchar4 BilinearSample(Texture& t, float2 uv, int lod)
{
    float2 coord = fracf(uv) * make_float2(t.sizes[lod]) - 0.5f;
    coord = make_float2(coord.x, t.sizes[lod].y - coord.y);
    int2 i0 = make_int2(coord);
    int2 i1 = i0 + make_int2(1, 0);
    int2 i2 = i0 + make_int2(0, -1);
    int2 i3 = i0 + make_int2(1, -1);

    float2 d = coord - make_float2(make_int2(coord));

    float w0 = (1.0f - d.x) * d.y;
    float w1 = d.x * d.y;
    float w2 = (1.0f - d.x) * (1.0f - d.y);
    float w3 = d.x * (1.0f - d.y);

    return make_uchar4(Sample(t, i0, lod) * w0 + Sample(t, i1, lod) * w1 +
                       Sample(t, i2, lod) * w2 + Sample(t, i3, lod) * w3);
}

__device__ uchar4 TrilinearSample(Texture& t, float2 uv, float lod)
{
    uint min_lod = floorf(lod);
    uint max_lod = min_lod + 1;
    min_lod = clamp(min_lod, 0u, t.max_lod);
    max_lod = clamp(max_lod, 0u, t.max_lod);

    float4 sample1 = make_float4(BilinearSample(t, uv, min_lod));
    float4 sample2 = make_float4(BilinearSample(t, uv, max_lod));

    float frac = fracf(lod);

    return make_uchar4(sample1 * (1.0f - frac) + sample2 * frac);
}

__device__ float3 Bump2Normal(Texture& texture, float3x3 tbn, float2 uv,
                              float lod)
{
    // Work out how much to offset the samples
    float texel_step = pow(2.0f, lod);
    float2 step = make_float2(texel_step) / make_float2(texture.sizes[0]);

    // Take 3 samples of the bump map
    uchar a = TrilinearSample(texture, uv - step * 0.5f, lod).x;
    uchar b =
        TrilinearSample(texture, uv + make_float2(step.x * 0.5f, 0), lod).x;
    uchar c =
        TrilinearSample(texture, uv + make_float2(0, step.y * 0.5f), lod).x;

    // Compute the gradients
    float2 grad = make_float2(float(b) - a, float(c) - a);

    // Cross product the gradients to get the bump normal
    float d = 4.0f;  // Represents depth of the bump map
    float3 normal =
        normalize(cross(make_float3(1, 0, d * grad.x / (texel_step * 256.0f)),
                        make_float3(0, 1, d * grad.y / (texel_step * 256.0f))));

    // Transform the bump normal into the triangle normal space
    normal = make_float3(dot(tbn.row[0], normal), dot(tbn.row[1], normal),
                         dot(tbn.row[2], normal));

    return normalize(normal);
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
                            Texture& tex)
{
    Triangle t(tri.v0, tri.v1, tri.v2);
    float4 spread_barys = RayTriangleGradients(t, ray, spread);

    float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
    float2 uvs_x =
        InterpolateUVs(attribs, make_float2(spread_barys.x, spread_barys.y));
    float2 uvs_y =
        InterpolateUVs(attribs, make_float2(spread_barys.z, spread_barys.w));

    float2 dtdx = fabs(uvs_x - uvs) * make_float2(tex.sizes[0]);
    float2 dtdy = fabs(uvs_y - uvs) * make_float2(tex.sizes[0]);
    float max_change = fmaxf(length(dtdx), length(dtdy));

    float lod = clamp(log2f(max_change), 0.0f, float(tex.max_lod));
    return lod;
}

__device__ bool IntersectRayTriangle(Triangle& tri, Ray& ray,
                                     RayResult& ray_result, uint32_t pair_id,
                                     uint32_t prim_id)
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
    ray_result.pair_id = pair_id;
    ray_result.barycentrics = make_float2(u, v);
    return true;
}

__device__ bool IntersectRayTrianglePair(TrianglePair& tri, Ray& ray,
                                         RayResult& ray_result,
                                         uint32_t pair_id, bool pair)
{
    Triangle tri_a(tri.v0, tri.v1, tri.v2);
    Triangle tri_b(tri.v2, tri.v1, tri.v3);
    bool hitA = IntersectRayTriangle(tri_a, ray, ray_result, pair_id,
                                     tri.primitive_id_0);
    bool hitB = pair ? IntersectRayTriangle(tri_b, ray, ray_result, pair_id,
                                            tri.primitive_id_1)
                     : false;
    return hitA || hitB;
}

__device__ bool TraceRay(DeviceAccelerationStructure as, Attributes* attributes,
                         Ray& ray, RayResult& ray_result, TraceStats& stats,
                         bool debug)
{
    bool tri_hit = false;
    unsigned stack_size = 1;
    StackEntry stack[64];
    stack[0] = {as.root, as.count};

    while (stack_size) {
        StackEntry entry = stack[--stack_size];
        unsigned num_hits = 0;
        StackEntry child_buffer;
        float child_dist;

        for (unsigned i = 0; i < entry.count; i++) {
            Node& node = as.nodes[entry.index + i];
            if (node.type == ChildType_None) continue;

            float dist;
            bool hit = IntersectRayAabb(node, ray, dist);

            bool is_leaf = node.type == ChildType_Tri;
            stats.box_tests++;

            if (hit && is_leaf) {
                // for (unsigned j = 0; j < node.count; j++) {
                stats.tri_tests++;
                bool hit_tri = IntersectRayTrianglePair(
                    as.triangles[node.child], ray, ray_result, node.child,
                    node.count > 0);
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

__device__ uchar4 AmbientShader(DeviceAccelerationStructure as,
                                DeviceScene scene, Ray& ray,
                                RayResult& ray_result, Material& mat,
                                Attributes& attribs, float spread,
                                bool use_textures, bool use_shadows,
                                bool use_bump)
{
    float3 light_colour = {1.0f, 0.9f, 0.8f};
    float3 light_pos = scene.light;

    float3 hit_pos = ray.origin + ray.direction * ray.tmax;
    float3 normal = InterpolateNormals(attribs, ray_result.barycentrics);
    if (use_bump && mat.disp != -1) {
        TrianglePair& pair = as.triangles[ray_result.pair_id];
        Triangle tri(pair.v0, pair.v1, pair.v2);
        Texture& disp = scene.textures[mat.disp];
        float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
        float lod = ComputeLOD(ray, ray_result, spread, pair, attribs, disp);
        float3x3 tbn = TangentMatrix(tri, attribs);
        uchar4 disp_smp = TrilinearSample(disp, uvs, lod);
        normal = make_float3(disp_smp.x / 255.0f, disp_smp.y / 255.0f,
                             disp_smp.z / 255.0f);
        normal = normalize(normal * 2.0f - 1.0f);

        normal = normalize(make_float3(dot(tbn.row[0], normal),
                                       dot(tbn.row[1], normal),
                                       dot(tbn.row[2], normal)));

    } else if (use_bump && mat.bump != -1) {
        TrianglePair& pair = as.triangles[ray_result.primitive_id];
        Triangle tri(pair.v0, pair.v1, pair.v2);
        Texture& bump = scene.textures[mat.bump];
        float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
        float lod = ComputeLOD(ray, ray_result, spread, pair, attribs, bump);
        float3x3 tbn = TangentMatrix(tri, attribs);
        normal = Bump2Normal(bump, tbn, uvs, lod);
    }

    float3 light_dir = normalize(light_pos - hit_pos);

    float3 ambient = 0.2f * light_colour;
    float3 diffuse = 1.0f * max(dot(normal, light_dir), 0.0f) * light_colour;
    float3 specular =
        1.0f *
        pow(max(dot(-ray.direction, reflect(-light_dir, normal)), 0.0),
            mat.specular_exp) *
        light_colour;

    float3 object_ambient = mat.ambient;
    float3 object_diffuse = mat.diffuse;
    float3 object_specular = mat.specular;

    if (use_textures && mat.texture != -1) {
        Texture& tex = scene.textures[mat.texture];
        float lod =
            ComputeLOD(ray, ray_result, spread,
                       as.triangles[ray_result.pair_id], attribs, tex);
        uchar4 smp = BilinearSample(
            tex, InterpolateUVs(attribs, ray_result.barycentrics), lod);
        object_diffuse.x = float(smp.x) / 255;
        object_diffuse.y = float(smp.y) / 255;
        object_diffuse.z = float(smp.z) / 255;
    }

    if (use_shadows) {
        Ray shadow_ray;
        RayResult shadow_result;
        TraceStats shadow_stats = {0};
        shadow_ray.origin = hit_pos;
        shadow_ray.direction = light_dir;
        shadow_ray.tmin = 0.001f;
        shadow_ray.tmax = length(light_pos - hit_pos);

        bool hit = TraceRay(as, scene.attributes, shadow_ray, shadow_result,
                            shadow_stats, false);

        if (hit) {
            diffuse = make_float3(0.0f);
            specular = make_float3(0.0f);
        }
    }

    float3 colour = diffuse * object_diffuse + ambient * object_ambient +
                    specular * object_specular;
    colour = clamp(colour, 0.0f, 1.0f);

    return make_uchar4(colour.x * 255, colour.y * 255, colour.z * 255, 255);
}

__global__ void TraceRays(DeviceAccelerationStructure as, DeviceScene scene,
                          uint32_t* num_tests, RenderType render_type,
                          cudaSurfaceObject_t image)
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned w = blockDim.x * gridDim.x;
    unsigned h = blockDim.y * gridDim.y;

    Camera& camera = *scene.camera;

    float2 coord = make_float2((float)x, (float)y);
    float2 ndc = 2 * ((coord + 0.5f) / make_float2(w, h)) - 1;
    float3 ndc3 = make_float3(ndc, 1);
    float3 p = (ndc.x * camera.u) + (ndc.y * camera.v) + (ndc3.z * camera.w);
    float spread = 2.0f / w;

    float max_depth = camera.max_depth;

    Ray ray;
    ray.direction = normalize(p);
    ray.origin = camera.position;
    ray.tmin = 0.00001f;
    ray.tmax = max_depth;

    RayResult ray_result;
    ray_result.primitive_id = 0;
    ray_result.pair_id = 0;
    TraceStats stats = {};
    stats.box_tests = 0;
    bool hit = TraceRay(as, scene.attributes, ray, ray_result, stats, false);
    float depth = hit ? ray.tmax : 0.0f;
    atomicAdd(num_tests, stats.box_tests);

    uchar4 colour;
    TrianglePair& pair = as.triangles[ray_result.pair_id];
    Attributes& attribs = scene.attributes[ray_result.primitive_id];
    Material& mat = scene.materials[attribs.material_id];

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
    } else if (render_type == RenderType::kMaterialId) {
        if (hit) {
            int32_t material_id = attribs.material_id;
            uchar3 rgb =
                HsvToRgb(float(material_id) / scene.num_materials, 1.0f, 1.0f);
            colour = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kDiffuse) {
        if (hit) {
            colour = AmbientShader(as, scene, ray, ray_result,
                                   scene.materials[attribs.material_id],
                                   attribs, spread, false, false, false);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kLODs) {
        if (mat.texture != -1 && hit) {
            Texture& texture = scene.textures[mat.texture];
            float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
            float lod =
                ComputeLOD(ray, ray_result, 2.0f / w, pair, attribs, texture);
            colour = make_uchar4((unsigned char)(int(lod) * 20));
        } else {
            colour = make_uchar4(255, 0, 255, 255);
        }
    } else if (render_type == RenderType::kTexture) {
        if (hit) {
            if (mat.texture != -1) {
                Texture& texture = scene.textures[mat.texture];
                float2 uvs = InterpolateUVs(attribs, ray_result.barycentrics);
                float lod = ComputeLOD(ray, ray_result, 2.0f / w, pair, attribs,
                                       texture);

                colour = TrilinearSample(texture, uvs, lod);
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
            colour = AmbientShader(as, scene, ray, ray_result,
                                   scene.materials[attribs.material_id],
                                   attribs, spread, true, false, true);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    } else if (render_type == RenderType::kTextureLitShadows) {
        if (hit) {
            colour = AmbientShader(as, scene, ray, ray_result,
                                   scene.materials[attribs.material_id],
                                   attribs, spread, true, true, true);
        } else {
            colour = make_uchar4(0, 0, 0, 255);
        }
    }

    surf2Dwrite(colour, image, x * 4, y);
}