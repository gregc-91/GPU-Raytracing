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

__global__ void TraceRays(DeviceAccelerationStructure as, DeviceScene scene,
                          uint32_t* num_tests, RenderType render_type,
                          cudaSurfaceObject_t image);