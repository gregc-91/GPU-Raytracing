
#include "Utilities.h"

#include <assert.h>

#include "helper_math.h"

void count_nodes_recurse(Node* nodes, uint32_t index, HierarchyStats& stats)
{
    stats.numNodes++;

    // printf("index %d child %d count %d aabb %f %f %f %f %f %f\n", index,
    // nodes[index].child, nodes[index].count, 	nodes[index].min.x,
    //	nodes[index].min.y,
    //	nodes[index].min.z,
    //	nodes[index].max.x,
    //	nodes[index].max.y,
    //	nodes[index].max.z
    //);

    if (nodes[index].type == ChildType_Tri)
        stats.numLeafNodes++;
    else if (nodes[index].type == ChildType_Box) {
        stats.numTreeNodes++;

        for (uint32_t i = 0; i < nodes[index].count; i++) {
            count_nodes_recurse(nodes, nodes[index].child + i, stats);
        }
    }
}

HierarchyStats CountNodes(Node* nodes, unsigned root, unsigned count)
{
    HierarchyStats stats;
    memset(&stats, 0, sizeof(stats));

    for (uint32_t i = 0; i < count; i++) {
        if (nodes[root + i].type == ChildType_Box) {
            count_nodes_recurse(nodes, root + i, stats);
        }
    }

    return stats;
}

void verify_hierarchy_recurse(Node* nodes, uint32_t index)
{
    if (nodes[index].type != ChildType_Box) return;

    float3 children_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 children_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (unsigned i = 0; i < nodes[index].count; i++) {
        Node& child = nodes[nodes[index].child + i];
        children_min = fminf(children_min, child.min);
        children_max = fmaxf(children_max, child.max);
    }

    if (nodes[index].min.x != children_min.x) goto error;
    if (nodes[index].min.y != children_min.y) goto error;
    if (nodes[index].min.z != children_min.z) goto error;
    if (nodes[index].max.x != children_max.x) goto error;
    if (nodes[index].max.y != children_max.y) goto error;
    if (nodes[index].max.z != children_max.z) goto error;

    for (unsigned i = 0; i < nodes[index].count; i++) {
        verify_hierarchy_recurse(nodes, nodes[index].child + i);
    }
    return;

error:
    fprintf(
        stderr,
        "Error: Invalid hierarchy; aabb inclusion check failed on index %d\n",
        index);
}

void VerifyHierarchy(Node* nodes, unsigned root, unsigned count)
{
    for (uint32_t i = 0; i < count; i++) {
        if (nodes[root + i].type == ChildType_Box) {
            verify_hierarchy_recurse(nodes, root + i);
        }
    }
}
