#ifndef _ARGUMENTS_H_
#define _ARGUMENTS_H_

#include <string>

extern std::string g_filename;

enum BuildType {
    kSAH,
    kBottomUp,
    kHybrid,
    kNone
};

enum RenderType {
    kDepth = 0,
    kBoxtests = 1,
    kTriangleTests = 2,
    kLODs = 3,
    kDiffuse = 4,
    kTexture = 5,
    kTextureLit = 6,
    kTextureLitShadows = 7,
    kCount = 8
};

struct Arguments {
    BuildType build_type = kSAH;
    bool enable_splits = false;
    bool enable_pairs = false;
    RenderType render_type = RenderType::kDepth;
};

Arguments ParseCmd(int arc, char** argv);

#endif _ARGUMENTS_H_
