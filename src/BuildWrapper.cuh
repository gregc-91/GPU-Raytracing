#pragma once

#include "Arguments.h"
#include "Common.cuh"

struct BuildInput {
    Triangle* triangles_in;
    TrianglePair* triangles_out;
    unsigned num_triangles;
    Node* nodes_out;
    void* scratch;
};

size_t SahMemoryRequirements(uint32_t num_triangles);

size_t BuMemoryRequirements(uint32_t num_triangles);

void RunSahBuild(BuildInput input, Arguments args);

void RunBottomUpBuild(BuildInput input, Arguments args, bool hybrid);