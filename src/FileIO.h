#pragma once

#include <string>
#include <vector>
#include <map>

#include "Common.cuh"

struct Triangle;

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Attributes> attributes;

    Library library;
    AABB aabb;
};

Scene LoadOBJFromFile(const std::string filename);
