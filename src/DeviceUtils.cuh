#include "Common.cuh"

__device__ inline int FloatToOrderedInt(float f)
{
    int i = __float_as_int(f);
    return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__device__ inline float OrderedIntToFloat(int i)
{
    return __int_as_float((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

__device__ inline float3 FloatToOrderedInt(float3 a)
{
    return make_float3(__int_as_float(FloatToOrderedInt(a.x)),
                       __int_as_float(FloatToOrderedInt(a.y)),
                       __int_as_float(FloatToOrderedInt(a.z)));
}

__device__ inline float3 OrderedIntToFloat(float3 a)
{
    return make_float3(OrderedIntToFloat(__float_as_int(a.x)),
                       OrderedIntToFloat(__float_as_int(a.y)),
                       OrderedIntToFloat(__float_as_int(a.z)));
}

__device__ inline AABB FloatToOrderedInt(AABB a)
{
    return {FloatToOrderedInt(a.min), FloatToOrderedInt(a.max)};
}

__device__ inline AABB OrderedIntToFloat(AABB a)
{
    return {OrderedIntToFloat(a.min), OrderedIntToFloat(a.max)};
}

__device__ inline float atomicMin(float* address, float val)
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

__device__ inline float atomicMax(float* address, float val)
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