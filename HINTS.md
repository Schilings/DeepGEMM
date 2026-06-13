# AKO4ALL Optimization Hints — AG GEMM (mc2, Phase 3)

## Workspace Adaptation
- Do NOT copy to solution/. Edit original files in place on main branch.
- Do NOT create opt/ branch. Work directly on main.
- Push after every iteration to prevent data loss from server crashes.
- Keep iteration notes in `docs/AG_GEMM_ITERATION.md` (append to existing file)
- Keep progress notes in `docs/PROGRESS.md`

## Constraints
- Max iterations: 30
- Preferred language: CUDA
- Web search: disabled
- No pip/apt installs
- ncu profiling: AVAILABLE (use for direction when stuck)

## Files to Optimize
- **Kernel**: `deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh`
- **Related**: `csrc/jit_kernels/impls/sm100_bf16_ag_gemm.hpp`, `csrc/jit_kernels/heuristics/ag_gemm.hpp`
- After editing .cuh/.hpp: `python3 setup.py build_ext --inplace`
- After editing heuristics: `rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_ag_gemm*`
- After any kernel change: clear JIT cache AND rebuild

## Benchmark Commands
- Correctness (quick): `CUDA_VISIBLE_DEVICES=0,1 python tests/test_ag_gemm.py 2`
- Correctness (8 GPU): `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tests/test_ag_gemm.py 8`
- Performance (4 GPU, faster iteration): `CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/bench_ag_gemm.py 4 10`
- Performance (8 GPU, final verdict): `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python benchmarks/bench_ag_gemm.py 8 10`
- All benchmark commands need: `PYTHONUNBUFFERED=1`

## Current Baseline (mc2, 8 GPU)
- Geo Mean: 1.135x
- Fused wins: 17/17 shapes
- Fused TFLOPS range: 1025-1314T (vs baseline 858-1128T)
- Best: 1.52x (6144×4096×4096)
- Worst: 1.02x (8192×7168×7168, 10240×7168×7168)

## Architecture
- 8× NVIDIA B300 SXM6 (SM100, 148 SMs per GPU)
- NVLink Gen5 (900 GB/s bidirectional)
- BF16 peak: ~1400 TFLOPS per GPU
- Current kernel: 256T (0 AG + 128 non-epi + 128 epilogue), mc2
- Split-warp: Load A (warp0) + Load B (warp1) + MMA (warp2, leader only) + Epilogue (4 warps)

## Known Bottlenecks
1. N=7168,K=7168 shapes only ~1.02x — GEMM compute-heavy, fused overhead not amortized
2. Standard GEMM achieves ~1100-1250T, fused AG GEMM achieves ~1025-1314T
3. Barrier polling overhead (ptx::ld_acq_sys) adds latency for remote chunks
4. Epilogue store is single-CTA only (is_leader_cta guard) — may bottleneck in mc2

## Optimization Targets
- Get N=7168,K=7168 shapes above 1.10x
- Increase Fused TFLOPS toward 1300+ consistently
- Reduce barrier polling overhead
- Better comm-compute overlap for medium shapes
