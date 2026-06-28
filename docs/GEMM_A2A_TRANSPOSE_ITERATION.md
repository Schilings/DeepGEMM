# GEMM + A2A-transpose 迭代记录（Ulysses SP pre-attn）

> 入口符号：`bf16_gemm_a2a_transpose_nt`
> 设计文档：`docs/GEMM_A2A_TRANSPOSE_DESIGN.md`
> test / bench：`tests/test_gemm_a2a_transpose.py` / `benchmarks/bench_gemm_a2a_transpose.py`

---

## 当前状态（接班看这里）

- **正确性**：`{2,4,8}` 卡全 PASS，max_diff/rel_err/consistency **恒为 0.0**（A2A 是纯排列，
  与 `bf16_gemm_nt` 的精确参考逐元素一致），同时与 torch 原生 `matmul + all_to_all_single`
  baseline 也 0.0 吻合。覆盖 bs=1（THD）与 bs=2/4（BSHD）、nheads∈{32,64}、K∈{4096,7168,8192}。
- **性能**（B300，单 kernel，无 reduce）：
  - 8 卡：geo vs torch-native **1.538x**，geo vs deepgemm-separate **1.420x**，fused 平均 **1187 TFLOPS**；
    最好/最差 vs torch 3.25x / 1.19x。
  - 4 卡：geo vs torch-native **1.552x**，geo vs deepgemm-separate **1.437x**，fused 平均 **1203 TFLOPS**。
  - 所有 shape 均比两条 baseline 快（无短板）。
- **分支**：`main`。

---

## 设计要点（与 GEMM-RS 的关系）

本算子是 post-attn `a2a_transpose_gemm` 的**对偶**，也是 GEMM-RS 的**直接改写**——
「人家本来就是 gemm+a2a 的形式」，只改三处：

1. **dst 切分轴 M→N**：GEMM-RS 按 token chunk（M）散射到各 rank；本算子按 head 组（N）散射，
   `dst_rank = (n_block*BLOCK_N) / local_n`，`local_n = N/num_ranks = local_nheads*head_dim`。
2. **GEMM 的 M = bs*local_seq**：每 rank 只投影自己的 seq 分片（heuristic 用 `num_ranks=1` 调用，
   使 `m_per_rank == M`）。
3. **删掉 reduce kernel**：A2A 是纯排列，每个输出位置恰被写一次（不累加、不清零）。symm buffer
   的输出区本身即结果，单 kernel / 单 stream。

### 转置-散射索引契约（epilogue）

对 GEMM 输出 `D[global_m, n]`：
```
b          = global_m / local_seq          # batch（local_seq % BLOCK_M == 0 → tile 不跨 batch）
s_local    = global_m % local_seq
dst_rank   = n / local_n                    # head 组属主（local_n % BLOCK_N == 0 → tile 不跨 dst）
base_n_idx = n % local_n
base_m_idx = b*seq + rank_idx*local_seq + s_local   # 写入 dst 的 seq 偏移（本 rank 的 seq 分片）
```
再用 `scatter_maps.maps[dst_rank]`（dst 输出区的 2D `[bs*seq, local_n]` TMA descriptor）
经 `sm100_store_cd` → `SM90_TMA_STORE_2D` 直推 peer HBM（NVLink，dst==self 时本地写）。

### 跨调用屏障

起始 `nvlink_barrier`（tag 51，保证 peers 上一轮读完我的 buffer 再覆盖）+ 结束（tag 52，
保证 peers push 全局可见再读），自复位 +1/-1 协议，**禁止 per-call memset**（与 GEMM-RS 同理）。

---

## 约束

- `local_seq % BLOCK_M(128) == 0`（tile 不跨 batch，base_m 线性可表）。
- `local_n % BLOCK_N(128) == 0`（head_dim=128 时 local_nheads≥1 即满足）。
- `N % num_ranks == 0`、`N % 128 == 0`、`K % 64 == 0`。
- `nheads % num_ranks == 0`。

---

## 涉及文件

- kernel：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_a2a_transpose.cuh`
- layout：`deep_gemm/include/deep_gemm/layout/gemm_a2a_transpose.cuh`
- host：`csrc/jit_kernels/impls/sm100_bf16_gemm_a2a_transpose.hpp`
- apis：`csrc/apis/gemm_a2a_transpose.hpp`（+ `csrc/python_api.cpp` 注册）
- python：`deep_gemm/gemm_a2a_transpose/__init__.py`（+ `deep_gemm/__init__.py` 暴露）
- 复用：`heuristics/gemm_rs_compute.hpp`（`get_gemm_rs_compute_config(..., num_ranks=1)`）、
  `epilogue/sm100_store_cd.cuh`、`comm/barrier.cuh`、`layout/sym_buffer.cuh`。

---

## 接班命令

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

# 正确性（2/4/8）
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python tests/test_gemm_a2a_transpose.py 8 --all

# benchmark（focus 子集 / 全集）
DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python benchmarks/bench_gemm_a2a_transpose.py 8 20
```

---

## 迭代历史

- **2026-06-28 v1（落地）**：抄 GEMM-RS 256T 模板，改 N 切分 + 删 reduce + 转置散射 epilogue，
  单 kernel 单 stream。{2,4,8} 正确性 0.0，8 卡 geo vs torch 1.54x / vs sep 1.42x。建立基线。
