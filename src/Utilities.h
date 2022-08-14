#include "Common.cuh"

struct HierarchyStats {
    int numNodes;
    int numLeafNodes;
    int numTreeNodes;
};

HierarchyStats CountNodes(Node* nodes, unsigned root, unsigned count);

void VerifyHierarchy(Node* nodes, unsigned root, unsigned count);