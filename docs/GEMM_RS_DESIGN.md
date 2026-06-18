# GEMM-RS 设计说明（唯一主线）

> 本文档只描述当前有效实现：`bf16_gemm_rs_nt`。
> 历史探索内容已归档，不作为当前决策依据。

---

## 1. 目标

在 SM100 路径上提供稳定可复现的 **GEMM + Reduce-Scatter 融合**，优先保证：

1. 多卡正确性稳定
2. benchmark 可复现
3. 在可复现基线之上持续迭代性能

---

## 2. 主线实现

- **算子入口**：`deep_gemm.bf16_gemm_rs_nt`
- **核心实现**：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
- **JIT 运行时**：`csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`
- **Python 接口**：`deep_gemm/gemm_rs/__init__.py`
- **测试**：`tests/test_gemm_rs.py`、`tests/test_gemm_rs_quick.py`
- **性能脚本**：`benchmarks/bench_gemm_rs.py`

---

## 3. 评测口径（统一）

只保留以下对比：

- `Separate`：`bf16_gemm_nt + torch.distributed.reduce_scatter_tensor`
- `Main Fused`：`bf16_gemm_rs_nt`

不再在主文档中维护多版本横向对比术语，避免误导。

---

## 4. 运行与验证

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

# 正确性
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

# 性能
DG_BENCH_MAX_SHAPES=3 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3
```

可选：
- `DG_BENCH_SINGLE_SHAPE=M,N,K`
- `DG_BENCH_SYNC_EACH_ITER=1`

---

## 5. 当前设计原则

1. **先稳后快**：先保证正确性和可复现，再做激进优化。
2. **最小化分叉**：只维护唯一主线路径，减少并行分支维护成本。
3. **结果驱动**：所有结论以本机可复现测试和 benchmark 输出为准。
