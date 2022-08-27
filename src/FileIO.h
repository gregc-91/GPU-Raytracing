#pragma once

#include <string>
#include <vector>
#include <map>

#include "Common.cuh"

struct Triangle;

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Attributes> attributes;

    Attributes* gpu_attributes;

    Library library;
    AABB aabb;
    float3 light;

    void CopyToDevice();
};

Scene LoadOBJFromFile(const std::string filename);
