#include <vector>

#include "BottomUpBuilder.cuh"
#include "BuildWrapper.cuh"
#include "Multiblock.cuh"
#include "RadixSort.cuh"
#include "SharedTaskBuilder.cuh"

struct SAHScratchSizes {
    size_t scratch_num_pairs_size = sizeof(int);
    size_t scratch_num_leaves_size = sizeof(int);
    size_t scratch_root_count_size = sizeof(int);
    size_t scratch_block_counts_size = (sizeof(int) * NUM_BLOCKS);
    size_t scratch_block_scan_size = (sizeof(int) * NUM_BLOCKS);
    size_t scratch_num_extra_leaves_size = sizeof(int);
    size_t scratch_c_aabb_size = sizeof(AABB);
    size_t scratch_p_aabb_size = sizeof(AABB);
    size_t scratch_block_c_aabbs_size = (sizeof(AABB) * NUM_BLOCKS);
    size_t scratch_block_p_aabbs_size = (sizeof(AABB) * NUM_BLOCKS);
    size_t scratch_aabbs_size;
    size_t scratch_tasks_size;
    size_t scratch_temp_ids_size;
    size_t scratch_prim_ids_size;

    size_t scratch_num_pairs_offset = 0;
    size_t scratch_num_leaves_offset =
        scratch_num_pairs_offset + scratch_num_pairs_size;
    size_t scratch_root_count_offset =
        scratch_num_leaves_offset + scratch_num_leaves_size;
    size_t scratch_block_counts_offset =
        scratch_root_count_offset + scratch_root_count_size;
    size_t scratch_block_scan_offset =
        scratch_block_counts_offset + scratch_block_counts_size;
    size_t scratch_num_extra_leaves_offset =
        scratch_block_scan_offset + scratch_block_scan_size;
    size_t scratch_c_aabb_offset =
        scratch_num_extra_leaves_offset + scratch_num_extra_leaves_size;
    size_t scratch_p_aabb_offset = scratch_c_aabb_offset + scratch_c_aabb_size;
    size_t scratch_block_c_aabbs_offset =
        scratch_p_aabb_offset + scratch_p_aabb_size;
    size_t scratch_block_p_aabbs_offset =
        scratch_block_c_aabbs_offset + scratch_block_c_aabbs_size;
    size_t scratch_aabbs_offset;
    size_t scratch_tasks_offset;
    size_t scratch_temp_ids_offset;
    size_t scratch_prim_ids_offset;

    size_t total_size;

    SAHScratchSizes(unsigned num_triangles)
    {
        scratch_aabbs_size = sizeof(AABB) * num_triangles;
        scratch_tasks_size = sizeof(Task) * num_triangles;
        scratch_temp_ids_size = sizeof(int) * (num_triangles + NUM_BLOCKS) * 2;
        scratch_prim_ids_size = sizeof(PrimitiveID) * num_triangles;

        scratch_aabbs_offset =
            scratch_block_p_aabbs_offset + scratch_block_p_aabbs_size;
        scratch_tasks_offset = scratch_aabbs_offset + scratch_aabbs_size;
        scratch_temp_ids_offset = scratch_tasks_offset + scratch_tasks_size;
        scratch_prim_ids_offset =
            scratch_temp_ids_offset + scratch_temp_ids_size;

        total_size = scratch_prim_ids_offset + scratch_prim_ids_size;
    }
};

struct BUScratchSizes {
    size_t scratch_zero_size = sizeof(uint32_t);
    size_t scratch_num_leaves_size = sizeof(uint32_t);
    size_t scratch_subroot_count_size = sizeof(uint32_t);
    size_t scratch_subroots_size = sizeof(PrimitiveID) * 512;
    size_t scratch_subroot_aabbs_size = sizeof(AABB) * 512;
    size_t scratch_p_aabb_size = sizeof(AABB);
    size_t scratch_c_aabb_size = sizeof(AABB);
    size_t scratch_morton_size;
    size_t scratch_sorted_indices_size;
    size_t scratch_locks_size;
    size_t scratch_leaf_indices_size;
    size_t scratch_tasks_size;
    size_t scratch_tmp_ids_size;

    size_t scratch_zero_offset = 0;
    size_t scratch_num_leaves_offset = scratch_zero_offset + scratch_zero_size;
    size_t scratch_subroot_count_offset =
        scratch_num_leaves_offset + scratch_num_leaves_size;
    size_t scratch_subroots_offset =
        scratch_subroot_count_offset + scratch_subroot_count_size;
    size_t scratch_subroot_aabbs_offset =
        scratch_subroots_offset + scratch_subroots_size;
    size_t scratch_p_aabb_offset =
        scratch_subroot_aabbs_offset + scratch_subroot_aabbs_size;
    size_t scratch_c_aabb_offset = scratch_p_aabb_offset + scratch_p_aabb_size;
    size_t scratch_morton_offset;
    size_t scratch_sorted_indices_offset;
    size_t scratch_locks_offset;
    size_t scratch_leaf_indices_offset;
    size_t scratch_tasks_offset;
    size_t scratch_tmp_ids_offset;

    size_t total_size;

    BUScratchSizes(unsigned num_triangles)
    {
        scratch_morton_size = sizeof(uint32_t) * num_triangles;
        scratch_sorted_indices_size = sizeof(uint32_t) * num_triangles;
        scratch_locks_size = sizeof(uint32_t) * num_triangles;
        scratch_leaf_indices_size = sizeof(uint32_t) * num_triangles;
        scratch_tasks_size = sizeof(uint32_t) * 512;
        scratch_tmp_ids_size = sizeof(uint32_t) * 512 * 2;

        scratch_morton_offset = scratch_c_aabb_offset + scratch_c_aabb_size;
        scratch_sorted_indices_offset =
            scratch_morton_offset + scratch_morton_size;
        scratch_locks_offset =
            scratch_sorted_indices_offset + scratch_sorted_indices_size;
        scratch_leaf_indices_offset = scratch_locks_offset + scratch_locks_size;
        scratch_tasks_offset =
            scratch_leaf_indices_offset + scratch_leaf_indices_size;
        scratch_tmp_ids_offset = scratch_tasks_offset + scratch_tasks_size;

        total_size = scratch_tmp_ids_offset + scratch_tmp_ids_size;
    }
};

size_t SahMemoryRequirements(uint32_t num_triangles)
{
    SAHScratchSizes sizes(num_triangles * 2);
    return sizes.total_size;
}

size_t BuMemoryRequirements(uint32_t num_triangles)
{
    BUScratchSizes sizes(num_triangles);
    return sizes.total_size;
}

void* Offset(void* p, size_t s) { return (void*)(((char*)p) + s); }

void RunSahBuild(BuildInput input, Arguments args)
{
    unsigned num_triangles = input.num_triangles;
    unsigned extra_leaves_threshold = num_triangles / 5;
    unsigned num_leaves = 0;
    bool should_print = true;

    SAHScratchSizes s(num_triangles + extra_leaves_threshold);

    // Setup scratch pointers

    int* p_num_pairs = (int*)Offset(input.scratch, s.scratch_num_pairs_offset);
    int* p_num_leaves =
        (int*)Offset(input.scratch, s.scratch_num_leaves_offset);
    int* p_num_extra_leaves =
        (int*)Offset(input.scratch, s.scratch_num_extra_leaves_offset);
    int* p_root_count =
        (int*)Offset(input.scratch, s.scratch_root_count_offset);
    int* p_block_counts =
        (int*)Offset(input.scratch, s.scratch_block_counts_offset);
    int* p_block_scan =
        (int*)Offset(input.scratch, s.scratch_block_scan_offset);
    AABB* p_c_aabb = (AABB*)Offset(input.scratch, s.scratch_c_aabb_offset);
    AABB* p_p_aabb = (AABB*)Offset(input.scratch, s.scratch_p_aabb_offset);
    AABB* p_block_c_aabbs =
        (AABB*)Offset(input.scratch, s.scratch_block_c_aabbs_offset);
    AABB* p_block_p_aabbs =
        (AABB*)Offset(input.scratch, s.scratch_block_p_aabbs_offset);
    AABB* p_aabbs = (AABB*)Offset(input.scratch, s.scratch_aabbs_offset);
    PrimitiveID* p_prim_ids =
        (PrimitiveID*)Offset(input.scratch, s.scratch_prim_ids_offset);
    int* p_temp_ids = (int*)Offset(input.scratch, s.scratch_temp_ids_offset);
    Task* p_tasks = (Task*)Offset(input.scratch, s.scratch_tasks_offset);

    // Initialise values on host

    uint32_t empty_int_aabb[6] = {0x7f7fffff, 0x7f7fffff, 0x7f7fffff,
                                  0x80800000, 0x80800000, 0x80800000};
    uint32_t empty_int_aabbs[NUM_BLOCKS][6];
    for (unsigned i = 0; i < NUM_BLOCKS; i++) {
        memcpy(&empty_int_aabbs[i][0], empty_int_aabb, sizeof(AABB));
    }

    // Initialse buffers on device

    cudaMemset(input.scratch, 0, s.scratch_aabbs_offset);
    cudaMemcpy(p_c_aabb, empty_int_aabb, sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(p_p_aabb, empty_int_aabb, sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(p_block_c_aabbs, &empty_int_aabbs[0][0],
               sizeof(AABB) * NUM_BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(p_block_p_aabbs, &empty_int_aabbs[0][0],
               sizeof(AABB) * NUM_BLOCKS, cudaMemcpyHostToDevice);

    // Run the kernels

    if (args.enable_splits) {
        run("scene aabb              ",
            (CalculateSceneAabb<<<(num_triangles + 1023) / 1024, 1024>>>(
                input.triangles_in, num_triangles, p_p_aabb)));

        if (args.enable_pairs) {
            run("triangle pair splitting      ",
                (SetupPairSplits<<<(((num_triangles + 1) / 2) + 1023) / 1024,
                                   1024>>>(
                    input.triangles_in, input.triangles_out, num_triangles,
                    extra_leaves_threshold, p_c_aabb, p_p_aabb, p_aabbs,
                    p_prim_ids, p_num_pairs, p_num_leaves,
                    p_num_extra_leaves)));
        } else {
            run("triangle splitting      ",
                (SetupSplits<<<(num_triangles + 1023) / 1024, 1024>>>(
                    input.triangles_in, input.triangles_out, num_triangles,
                    extra_leaves_threshold, p_c_aabb, p_p_aabb, p_aabbs,
                    p_prim_ids, p_num_leaves, p_num_extra_leaves)));
        }
    } else {
        run("triangle pairing        ",
            (Setup<<<(((num_triangles + 1) / 2) + 1023) / 1024, 1024>>>(
                input.triangles_in, input.triangles_out, num_triangles,
                p_c_aabb, p_p_aabb, p_aabbs, p_prim_ids, p_num_leaves,
                args.enable_pairs)));
    }

    // Read back the number of leaves so we know how many blocks to launch
    cudaMemcpy(&num_leaves, p_num_leaves, 4, cudaMemcpyDeviceToHost);

    run("GridBlockCounts       ",
        (GridBlockCounts<<<(num_leaves + 1023) / 1024, 1024>>>(
            p_c_aabb, p_p_aabb, p_block_c_aabbs, p_block_p_aabbs, p_aabbs,
            p_block_counts, p_num_leaves)));

    run("GridBlockScan         ",
        (GridBlockScan<<<1, NUM_BLOCKS>>>(p_block_counts, p_block_scan,
                                          p_c_aabb, p_p_aabb)));

    run("GridBlockDistribute   ",
        (GridBlockDistribute<<<(num_leaves + 1023) / 1024, 1024>>>(
            p_c_aabb, p_block_c_aabbs, p_block_p_aabbs, p_aabbs, p_num_leaves,
            p_block_counts, p_block_scan, p_temp_ids, p_root_count)));

    run("SharedTaskBuild       ",
        (SharedTaskBuild<<<NUM_BLOCKS, 512>>>(
            p_tasks, p_block_c_aabbs, p_block_p_aabbs, input.nodes_out, p_aabbs,
            p_prim_ids, p_temp_ids + 2 * NUM_BLOCKS, p_block_counts,
            p_block_scan, p_num_leaves, false, ChildType_Tri, 2 * NUM_BLOCKS)));

    run("SharedTaskBuild (top) ",
        (SharedTaskBuild<<<1, 1024>>>(
            p_tasks, p_c_aabb, p_p_aabb, input.nodes_out, p_block_p_aabbs,
            p_prim_ids, p_temp_ids, p_root_count, p_block_scan, p_root_count,
            true, ChildType_Box, 0)));
}

void RunBottomUpBuild(BuildInput input, Arguments args, bool hybrid)
{
    BUScratchSizes s(input.num_triangles);
    unsigned num_triangles = input.num_triangles;
    unsigned num_leaves = num_triangles;
    unsigned blocks = (num_triangles + 1023) / 1024;
    unsigned blocks_pairs = (((num_triangles + 1) / 2) + 1023) / 1024;
    unsigned blocks_256 = (num_triangles + 255) / 256;
    bool should_print = true;

    // Setup scratch
    int32_t* p_zero = (int32_t*)Offset(input.scratch, s.scratch_zero_offset);
    int32_t* p_subroot_count =
        (int32_t*)Offset(input.scratch, s.scratch_subroot_count_offset);
    PrimitiveID* p_subroots =
        (PrimitiveID*)Offset(input.scratch, s.scratch_subroots_offset);
    AABB* p_subroot_aabbs =
        (AABB*)Offset(input.scratch, s.scratch_subroot_aabbs_offset);
    AABB* p_p_aabb = (AABB*)Offset(input.scratch, s.scratch_p_aabb_offset);
    AABB* p_c_aabb = (AABB*)Offset(input.scratch, s.scratch_c_aabb_offset);
    uint32_t* p_morton =
        (uint32_t*)Offset(input.scratch, s.scratch_morton_offset);
    uint32_t* p_sorted_indices =
        (uint32_t*)Offset(input.scratch, s.scratch_sorted_indices_offset);
    uint32_t* p_locks =
        (uint32_t*)Offset(input.scratch, s.scratch_locks_offset);
    uint32_t* p_leaf_indices =
        (uint32_t*)Offset(input.scratch, s.scratch_leaf_indices_offset);
    uint32_t* p_num_leaves =
        (uint32_t*)Offset(input.scratch, s.scratch_num_leaves_offset);
    Task* p_tasks = (Task*)Offset(input.scratch, s.scratch_tasks_offset);
    int32_t* p_tmp_ids =
        (int32_t*)Offset(input.scratch, s.scratch_tmp_ids_offset);

    // Initialise values on host
    uint32_t empty_int_aabb[6] = {0x7f7fffff, 0x7f7fffff, 0x7f7fffff,
                                  0x80800000, 0x80800000, 0x80800000};
    float empty_float_aabb[6] = {FLT_MAX,  FLT_MAX,  FLT_MAX,
                                 -FLT_MAX, -FLT_MAX, -FLT_MAX};
    std::vector<int> ascending_ints(512);
    for (unsigned i = 0; i < 512; i++) ascending_ints[i] = i;

    // Initialse buffers on device
    cudaMemset(p_zero, 0, sizeof(uint32_t));
    cudaMemset(p_subroot_count, 0, s.scratch_subroot_count_size);
    cudaMemset(p_locks, 0, s.scratch_locks_size);
    cudaMemcpy(p_p_aabb, empty_int_aabb, sizeof(AABB), cudaMemcpyHostToDevice);
    cudaMemcpy(p_c_aabb, empty_float_aabb, sizeof(AABB),
               cudaMemcpyHostToDevice);
    cudaMemcpy(p_tmp_ids, ascending_ints.data(), sizeof(int) * 512,
               cudaMemcpyHostToDevice);

    // Run the kernels

    run("SceneAabb          ",
        (CalculateSceneAabb<<<blocks, 1024>>>(input.triangles_in, num_triangles,
                                              p_p_aabb)));

	if (args.enable_pairs) {
		run("GenerateMortonCodesPairs",
			(GenerateMortonCodesPairs<<<blocks_pairs, 1024>>>(
				p_morton, p_sorted_indices, (float3*)input.triangles_in, p_p_aabb,
				p_num_leaves, num_triangles)));

		// Read the number of leaves back
		cudaMemcpy(&num_leaves, p_num_leaves, sizeof(uint32_t),
				cudaMemcpyDeviceToHost);
		blocks = (num_leaves + 1023) / 1024;
		blocks_256 = (num_leaves + 255) / 256;
	}
	else {
		run("GenerateMortonCodes",
			(GenerateMortonCodes<<<blocks, 1024>>>(p_morton, p_sorted_indices,
												(float3*)input.triangles_in,
												p_p_aabb, num_triangles)));
	}

    run("RadixSort          ",
        (RadixSort(p_morton, p_sorted_indices, p_locks, p_leaf_indices,
                   num_leaves)));  // locks and leaf indices used a temp scratch

    run("GenerateHierarchy  ", (GenerateHierarchy<<<blocks, 1024>>>(
                                   input.nodes_out, p_leaf_indices, p_morton,
                                   p_sorted_indices, num_leaves)));

    cudaMemset(p_locks, 0, sizeof(uint32_t) * num_leaves);

    run("GenerateTriangles  ",
        (GenerateTriangles<<<blocks, 1024>>>(p_sorted_indices,
                                             (float3*)input.triangles_in,
                                             input.triangles_out, num_leaves)));

    run("GenerateAABBs      ",
        (GenerateAABBs<<<blocks_256, 256>>>(input.nodes_out, p_leaf_indices,
                                            p_sorted_indices, p_locks,
                                            input.triangles_out, num_leaves)));

    if (hybrid) {
        run("Extract            ",
            (ExtractDepth<<<1, 256>>>(input.nodes_out, p_subroots,
                                      p_subroot_aabbs, p_subroot_count,
                                      p_c_aabb, p_p_aabb, 0, 8, num_leaves)));

        run("SharedTaskBuild  ",
            (SharedTaskBuild<<<1, 512>>>(
                p_tasks, p_c_aabb, p_p_aabb, input.nodes_out, p_subroot_aabbs,
                p_subroots, p_tmp_ids, p_subroot_count, p_zero, p_subroot_count,
                false, ChildType_Box, num_leaves * 2)));
    }
}