#pragma once

#include <cstdlib>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/gemm_a2a_transpose.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/gemm_rs_compute.hpp"

namespace deep_gemm {

// Host-side mirror of the kernel's GemmA2ATransposeScatterMaps (layout-compatible: CUtensorMap[8]).
// maps[d] describes dst_rank d's symmetric OUTPUT buffer as a 2D [bs*seq x local_n] tensor (BSHD)
// for the epilogue's transpose-scatter 2D-TMA push (remote P2P for d != my_rank, local for self).
// NOTE: distinct name from GemmRSScatterMaps to avoid an ODR clash when python_api.cpp includes
// both headers in the same translation unit.
struct GemmA2ATransposeScatterMaps {
    CUtensorMap maps[8];
};

// Raw-pointer 2D CD-style TMA descriptor builder over an arbitrary device base pointer (a peer
// rank's symmetric OUTPUT buffer). Identical semantics to GEMM-RS's make_tma_2d_desc_raw, kept
// local here so this header is self-contained (only one of the two RS/A2A headers defines the
// helper; pick a unique name to avoid ODR clashes).
static CUtensorMap make_a2a_transpose_tma_2d_desc_raw(
        void* gmem_ptr, const at::ScalarType& dtype, const int& elem_size,
        int gmem_inner_dim, int gmem_outer_dim,
        int smem_inner_dim, int smem_outer_dim,
        const int& gmem_outer_stride,
        const int& swizzle_mode, const int& swizzle_base = 0) {
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;
    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size), };
    const cuuint32_t elem_strides[2] = {1, 1};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(dtype, false, true),
        2, gmem_ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

// ════════════════════════════════════════════════════════════════
//  Ulysses-SP pre-attn fused GEMM + All2All-transpose (SINGLE kernel)
//
//    This is the DUAL of post-attn `a2a_transpose_gemm`. We do the QKV/Q projection GEMM first,
//    then scatter the result by head (All2All-transpose) so each rank ends up owning a head group
//    with the FULL seq (BSHD [bs, seq, local_nheads, head_dim]) — ready for FlashAttention.
//
//    It is literally GEMM-RS with three changes:
//      1. dst_rank is sliced along N (head group) instead of M (token chunk).
//      2. The GEMM M is bs*local_seq (this rank only projects its own seq shard).
//      3. NO reduce kernel: A2A is a pure permutation (every output position written exactly once),
//         so each rank pushes straight into the dst's single OUTPUT buffer at the seq offset
//         (rank*local_seq), and the symmetric buffer IS the result. Single kernel, single stream.
// ════════════════════════════════════════════════════════════════

class SM100BF16GemmA2ATransposeRuntime final : public LaunchRuntime<SM100BF16GemmA2ATransposeRuntime> {
public:
    struct Args {
        int m, n, k;
        int bs, seq, local_seq, local_n;
        int num_ranks;
        at::ScalarType cd_dtype;
        at::ScalarType comm_dtype;
        GemmRSComputeConfig config;

        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        GemmA2ATransposeScatterMaps scatter_maps;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Kernel template parameters:
        //   BLOCK_M, BLOCK_N, BLOCK_K, kNumStages,
        //   kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
        //   kNumMulticast, kIsMulticastOnA,
        //   kSwapAB, kWithAccumulation,
        //   kNumNonEpilogueThreads, kNumEpilogueThreads,
        //   kNumSMs, kNumRanks,
        //   cd_dtype_t, comm_dtype_t
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_a2a_transpose.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_a2a_transpose_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.swizzle_a_mode, args.config.swizzle_b_mode, args.config.swizzle_cd_mode,
    args.config.num_multicast, args.config.is_multicast_on_a ? "true" : "false",
    args.config.swap_ab ? "true" : "false", args.config.with_accumulation ? "true" : "false",
    args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.cd_dtype),
    to_string(args.comm_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t shape_m = static_cast<uint32_t>(args.m);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        uint32_t bs = static_cast<uint32_t>(args.bs);
        uint32_t seq = static_cast<uint32_t>(args.seq);
        uint32_t local_seq = static_cast<uint32_t>(args.local_seq);
        uint32_t local_n = static_cast<uint32_t>(args.local_n);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            shape_m, shape_n, shape_k,
            bs, seq, local_seq, local_n,
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_b,
            args.scatter_maps));
    }
};

// ════════════════════════════════════════════════════════════════
//  统一入口: pre-attn GEMM + All2All-transpose (single-kernel, single-stream)
// ════════════════════════════════════════════════════════════════
//   a   : [bs*local_seq, k]  this rank's local activations (NOT symmetric, read locally)
//   b   : [n, k]             QKV/Q projection weights (NT layout), n = nheads*head_dim
//   sym : symmetric OUTPUT buffer (peer-mapped via sym_buffer_ptrs), holds [bs*seq, local_n]
static void sm100_bf16_gemm_a2a_transpose_nt(const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             const torch::Tensor& sym_buffer,
                                             const std::vector<int64_t>& sym_buffer_ptrs,
                                             const int& rank_idx,
                                             const int& bs,
                                             const int& local_seq,
                                             const int& n,
                                             const int& k,
                                             const std::string& compiled_dims,
                                             const at::ScalarType& comm_dtype = torch::kBFloat16) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();

    const int seq = local_seq * num_ranks;          // full sequence length after gather
    const int m = bs * local_seq;                   // this rank's GEMM rows
    DG_HOST_ASSERT(n % num_ranks == 0);
    const int local_n = n / num_ranks;              // head group width = local_nheads * head_dim

    // Reuse the GEMM-RS heuristic with num_ranks=1 so m_per_rank == m (this rank already only
    // holds its own seq shard; there is no further M-splitting in the A2A path).
    auto config = get_gemm_rs_compute_config(m, n, k, num_sms, static_cast<int>(a.element_size()), 1);

    DG_HOST_ASSERT(config.block_k == 64);
    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // Tile-alignment guarantees for the transpose-scatter epilogue:
    //   local_seq % BLOCK_M == 0  → no M-tile crosses a batch boundary (b = global_m / local_seq
    //                               is constant within a tile).
    //   local_n   % BLOCK_N == 0  → no N-tile crosses a head-group (dst_rank) boundary.
    DG_HOST_ASSERT(local_seq % config.block_m == 0);
    DG_HOST_ASSERT(local_n % config.block_n == 0);

    // ── Create TMA descriptors ──
    // A: local activations [m, k] → TMA 2D load (multicast to 2 CTAs when mc=2)
    const auto tensor_map_a = make_tma_2d_desc(a,
                                               k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(a.stride(-2)),
                                               config.swizzle_a_mode);
    // B: weights [n, k] → TMA 2D load
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);

    // Transpose-scatter TMA descriptors (one per dst_rank). maps[d] targets dst_rank d's symmetric
    // OUTPUT buffer (offset kNumBarrierSignalBytes) viewed as a 2D [bs*seq x local_n] tensor (row
    // stride local_n) with the CD swizzle. There are NO per-source slots: every source rank writes
    // a disjoint seq region (b*seq + rank*local_seq + s_local), so the single output region is
    // shared across sources and each position is written exactly once.
    const int comm_elem_size = (comm_dtype == torch::kFloat) ? 4 : 2;
    const int64_t out_offset = static_cast<int64_t>(layout::GemmA2ATransposeWorkspace::kNumBarrierSignalBytes);
    GemmA2ATransposeScatterMaps scatter_maps{};
    for (int d = 0; d < num_ranks; ++ d) {
        auto* out_ptr = reinterpret_cast<char*>(static_cast<uintptr_t>(sym_buffer_ptrs[d])) + out_offset;
        scatter_maps.maps[d] = make_a2a_transpose_tma_2d_desc_raw(
            reinterpret_cast<void*>(out_ptr), comm_dtype, comm_elem_size,
            local_n, bs * seq,
            config.swizzle_cd_mode / comm_elem_size, config.block_m,
            local_n,
            config.swizzle_cd_mode);
    }

    // ── Launch the fused kernel on the compute stream (single kernel, no reduce) ──
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads;

    const SM100BF16GemmA2ATransposeRuntime::Args args = {
        .m = m, .n = n, .k = k,
        .bs = bs, .seq = seq, .local_seq = local_seq, .local_n = local_n,
        .num_ranks = num_ranks,
        .cd_dtype = comm_dtype,
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .scatter_maps = scatter_maps,
        .launch_args = LaunchArgs(num_sms,
                                  total_threads,
                                  config.smem_size,
                                  config.num_multicast)
    };

    const auto code = SM100BF16GemmA2ATransposeRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_a2a_transpose_nt", code);
    SM100BF16GemmA2ATransposeRuntime::launch(runtime, args);
}

} // namespace deep_gemm
