#include "Arguments.h"
#include "Common.cuh"

typedef struct TraceStats {
    unsigned box_tests;
    unsigned tri_tests;
} TraceStats;

typedef struct Ray {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
} Ray;

typedef struct StackEntry {
    unsigned index;
    unsigned count;
} StackEntry;

__global__ void TraceRays(TrianglePair* triangles, Node* nodes,
                          Attributes* attributes, Material* materials,
                          Texture* textures, Camera* camera,
                          uint32_t* num_tests, RenderType render_type,
                          cudaSurfaceObject_t image, unsigned root,
                          unsigned count, float3 light);