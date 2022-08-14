#include "Arguments.h"

#include <assert.h>

#include <string>

std::string g_filename = "";

std::string BuildTypeToString(BuildType b)
{
    switch (b) {
        case kHybrid:
            return "hybrid";
        case kSAH:
            return "sah";
        case kBottomUp:
            return "bottom-up";
        default:
            assert(0);
    }
    return "none";
}

BuildType ParseType(std::string s)
{
    if (s == "hybrid") return BuildType::kHybrid;
    if (s == "sah") return BuildType::kSAH;
    if (s == "bottom-up") return BuildType::kBottomUp;
    assert(0);
    return BuildType::kNone;
}

void PrintArgs(Arguments args)
{
    printf("Arguments:\n");
    printf("  BuildType: %s\n", BuildTypeToString(args.build_type).c_str());
    printf("  Pairs: %s\n", args.enable_pairs ? "true" : "false");
    printf("  Splits: %s\n", args.enable_splits ? "true" : "false");
    printf("\n");
}

Arguments ParseCmd(int argc, char** argv)
{
    Arguments args;

    assert(argc >= 2 && "No input file");

    g_filename = std::string(argv[1]);

    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == std::string("--pairs")) {
            args.enable_pairs = true;
        } else if (std::string(argv[i]) == std::string("--splits")) {
            args.enable_splits = true;
        } else if (std::string(argv[i]) == std::string("--type")) {
            args.build_type = ParseType(argv[i + 1]);
            i++;
        }
    }

    PrintArgs(args);
    return args;
}
