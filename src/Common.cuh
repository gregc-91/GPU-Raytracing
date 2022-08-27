
#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda_runtime_api.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "device_launch_parameters.h"
#include "helper_math.h"

#define BLOCK_GRID_DIM 4
#define NUM_BLOCKS BLOCK_GRID_DIM* BLOCK_GRID_DIM* BLOCK_GRID_DIM
#define MAX_TEXTURE_SIZE (1024 * 8)
#define NUM_LODS 13

typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;

// Struct to store the rotations for each triangle to make a pair
struct Rotations {
    int rot_a;
    int rot_b;
};

typedef enum ChildType {
    ChildType_None,
    ChildType_Box,
    ChildType_Tri,
    ChildType_Inst,
    ChildType_Proc
} ChildType;

struct Camera {
    float3 position;
    float pitch;
    float3 w;
    float yaw;
    float3 u;
    float scale;
    float3 v;
    float max_depth;
};

struct Attributes {
    float3 normal[3];
    float2 uv[3];
    int32_t material_id;
};

struct Texture {
    std::string name;
    uchar4* mips[NUM_LODS] = {NULL};
    uchar4* gpu_mips[NUM_LODS] = {NULL};
    int2 sizes[NUM_LODS];
    uint max_lod;

    Texture(std::string name) : name(name)
    {
        for (unsigned i = 0; i < NUM_LODS; i++) mips[i] = NULL;
    }

    Texture(std::string name, uchar4* mip0, int2 size0) : name(name)
    {
        mips[0] = mip0;
        sizes[0] = size0;
        max_lod = 0;
    }

    Texture(const Texture& other) : name(other.name), max_lod(other.max_lod)
    {
        for (unsigned i = 0; i < NUM_LODS; i++) {
            mips[i] = other.mips[i];
            sizes[i] = other.sizes[i];
        }
    }

    uchar4 ReadTexel(int2 coord, int lod);
    void WriteTexel(int2 coord, int lod, uchar4 val);
    void GenerateLODs();
};

struct Material {
    std::string name;
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float specular_exp;
    int32_t texture;

    Material() : name("") {}
    Material(std::string s) : name(s)
    {
        ambient = make_float3(0.0f);
        diffuse = make_float3(0.0f);
        specular = make_float3(0.0f);
        specular_exp = 0.0f;
        texture = -1;
    }
    Material(const Material& other)
        : name(other.name),
          ambient(other.ambient),
          diffuse(other.diffuse),
          specular(other.specular),
          specular_exp(other.specular_exp),
          texture(other.texture)
    {
    }
    ~Material();
    uchar4 ReadTexel(Texture* textures, int2 coord, int lod);
    void WriteTexel(Texture* textures, int2 coord, int lod, uchar4 val);
    bool HasTexture();
};

struct Library {
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::map<std::string, uint32_t> name_to_mat;
    std::map<std::string, uint32_t> name_to_tex;

    Material* gpu_materials;
    Texture* gpu_textures;

    void AddMaterial(std::string name);
    int32_t GetMaterialId(std::string name);
    Material& GetMaterial(std::string name);
    Material& GetMaterial(uint32_t i);
    int32_t GetTextureId(std::string name);
    Texture& GetTexture(uint32_t i);
    Texture& GetTextureFromMat(uint32_t i);

    void CopyToDevice();
};

struct Node {
    float3 min;
    uint32_t parent : 29;
    uint32_t count : 3;
    float3 max;
    uint32_t child : 29;
    uint32_t type : 3;
};

struct TrianglePair {
    float3 v0;
    uint32_t primitive_id_0;
    float3 v1;
    uint32_t primitive_id_1;
    float3 v2;
    float pad2;
    float3 v3;
    float pad3;

    __host__ __device__ TrianglePair() {}

    __host__ __device__ TrianglePair(const float3& v0_, const float3& v1_,
                                     const float3& v2_, uint32_t id)
    {
        v0 = v0_;
        v1 = v1_;
        v2 = v2_;
        primitive_id_0 = id;
    }

    __host__ __device__ TrianglePair(const float3& v0_, const float3& v1_,
                                     const float3& v2_, const float3& v3_,
                                     uint32_t id_0, uint32_t id_1)
    {
        v0 = v0_;
        v1 = v1_;
        v2 = v2_;
        v3 = v3_;
        primitive_id_0 = id_0;
        primitive_id_1 = id_1;
    }
};

struct AABB {
    float3 min;
    float3 max;

    __host__ __device__ AABB() {}

    __host__ __device__ AABB(const float3& amin, const float3& amax)
    {
        min = amin;
        max = amax;
    }

    __host__ __device__ AABB(const float3* minmax)
    {
        min = minmax[0];
        max = minmax[1];
    }

    __host__ __device__ AABB Intersection(const AABB& other)
    {
        return AABB{fmaxf(min, other.min), fminf(max, other.max)};
    }

    __host__ __device__ bool Valid()
    {
        return max.x >= min.x && max.y >= min.y && max.z >= min.z;
    };

    __host__ __device__ float3 Centre() { return (min + max) * 0.5f; }
};

__device__ inline float sa(const AABB& a)
{
    float3 len = a.max - a.min;
    return 2.0f * (len.x * len.y + len.x * len.z + len.y * len.z);
}

__host__ __device__ inline AABB Combine(const AABB& a, const AABB& b)
{
    AABB r;
    r.min = fminf(a.min, b.min);
    r.max = fmaxf(a.max, b.max);
    return r;
}

__host__ __device__ inline AABB Combine(const AABB& a, const float3& b)
{
    AABB r;
    r.min = fminf(a.min, b);
    r.max = fmaxf(a.max, b);
    return r;
}

struct PrimitiveID {
    uint32_t id;
    uint32_t count;
};

struct Task {
    AABB c_aabb;
    AABB p_aabb;
    unsigned start;
    unsigned end;
    unsigned parent_idx;
    unsigned buffer_idx;
};

struct Bin {
    AABB c_aabb;
    AABB p_aabb;
    unsigned count;
};

typedef struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;

    Triangle()
    {
        v0.x = 0;
        v0.y = 0;
        v0.z = 0;
        v1.x = 0;
        v1.y = 0;
        v1.z = 0;
        v2.x = 0;
        v2.y = 0;
        v2.z = 0;
    }

    __host__ __device__ Triangle(float3 a, float3 b, float3 c)
        : v0(a), v1(b), v2(c)
    {
    }

    Triangle(float ax, float ay, float az, float bx, float by, float bz,
             float cx, float cy, float cz)
    {
        v0.x = ax;
        v0.y = ay;
        v0.z = az;
        v1.x = bx;
        v1.y = by;
        v1.z = bz;
        v2.x = cx;
        v2.y = cy;
        v2.z = cz;
    }
} Triangle;

#pragma warning(disable : 26812)
#define check(ans)                            \
    {                                         \
        GpuAssert((ans), __FILE__, __LINE__); \
    }
inline void GpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort) exit(code);
    }
}

// Macro to call a kernel and time it with events
#define run(_name_, _call_)                                   \
    {                                                         \
        cudaEvent_t event_start, event_stop;                  \
        cudaEventCreate(&event_start);                        \
        cudaEventCreate(&event_stop);                         \
        cudaEventRecord(event_start);                         \
        _call_;                                               \
        cudaEventRecord(event_stop);                          \
        cudaEventSynchronize(event_stop);                     \
                                                              \
        check(cudaPeekAtLastError());                         \
        check(cudaDeviceSynchronize());                       \
                                                              \
        float time = 0;                                       \
        cudaEventElapsedTime(&time, event_start, event_stop); \
                                                              \
        if (should_print) {                                   \
            printf("%s time elapsed: %fms\n", _name_, time);  \
        }                                                     \
    }

#endif