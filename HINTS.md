# AKO4ALL Optimization Hints

## Workspace Adaptation
- Do NOT copy to solution/. Edit original files in place.
- Do NOT create opt/ branch. Work directly on main.
- Push after every iteration to prevent data loss from server crashes.
- Related files: csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp, csrc/jit_kernels/heuristics/gemm_rs.hpp

## Constraints
- Max iterations: 30
- Preferred language: CUDA
- Web search: disabled
- No pip/apt installs (ncu installed but ERR_NVGPUCTRPERM — cannot use in this container)
- ncu profiling: DISABLED (use analytical reasoning + runtime stats instead)

## Benchmark Commands
- Correctness: `python tests/test_gemm_rs.py 8`
- Performance: `python benchmarks/bench_gemm_rs.py 8 20`
- Quick perf (2 GPU, faster iteration): `python benchmarks/bench_gemm_rs.py 2 10`

## Architecture
- 8× NVIDIA B300 SXM6 (SM100, compute 10.3, 148 SMs per GPU)
- NVLink Gen5 (900 GB/s bidirectional)
- BF16 peak: ~1400 TFLOPS per GPU
- Current fusion kernel: 170-600 TFLOPS (vs ~1100 TFLOPS standard GEMM)

## Key Bottleneck (from analysis)
- 384 threads total: 128 Comm + 128 non-epi + 128 Epilogue
- Only 1 warpgroup (128T) does MMA computation
- Comm warps (128T) do P2P pull + reduce — mostly memory-bound
- Epilogue (128T) does TMEM→smem→TMA store — serial per row
- GEMM compute efficiency is the primary bottleneck (not comm bandwidth)
