# GEMM + A2A-transpose 迭代记录（Ulysses SP pre-attn）

> 入口符号：`bf16_gemm_a2a_transpose_nt`
> 设计文档：`docs/GEMM_A2A_TRANSPOSE_DESIGN.md`
> 单算子 test / bench：`tests/test_gemm_a2a_transpose.py` / `benchmarks/bench_gemm_a2a_transpose.py`
> Ulysses 端到端 test：`tests/test_ulysses_pre_attn_flow.py`（pre-attn 融合 QKV proj+A2A）、
> `tests/test_ulysses_full_attn_flow.py`（pre+post 两算子端到端）；post-attn 对照见
> `tests/test_ulysses_post_attn_flow.py` / `tests/test_ulysses_post_attn_varlen_thd.py`

---

## 当前状态（接班看这里）

- **正确性**：`{2,4,8}` 卡全 PASS，max_diff/rel_err/consistency **恒为 0.0**（A2A 是纯排列，
  与 `bf16_gemm_nt` 的精确参考逐元素一致），同时与 torch 原生 `matmul + all_to_all_single`
  baseline 也 0.0 吻合。覆盖 bs=1（THD）与 bs=2/4（BSHD）、nheads∈{32,64}、K∈{4096,7168,8192}。
- **性能**（B300 SXM6 AC，单 kernel，无 reduce，iters=30）：
  - 8 卡：geo vs torch-native **1.555x**，geo vs deepgemm-separate **1.420x**，fused 平均 **1184 TFLOPS**；
    最好/最差 vs torch 2.86x / 1.18x。
  - 4 卡：geo vs torch-native **1.521x**，geo vs deepgemm-separate **1.426x**，fused 平均 **1213 TFLOPS**；
    最好/最差 vs torch 2.28x / 1.19x。
  - 所有 shape 均比两条 baseline 快（无短板）。
- **分支**：`main`。

### 逐 shape 性能（8 卡，时延 μs，bf16）

shape 列：`(bs, local_seq, nheads, head_dim, K)`；N = nheads·head_dim，seq = local_seq·num_ranks。

| Shape (bs,lseq,h,hd,K) | Torch | Separate | Fused | Fused TFLOPS | vs Torch | vs Sep |
|---|---:|---:|---:|---:|---:|---:|
| 1,1024,32,128,4096 | 116.7 | 80.3 | **50.7** | 677.7 | **2.30x** | 1.58x |
| 1,2048,64,128,8192 | 291.8 | 283.1 | **227.8** | 1206.5 | 1.28x | 1.24x |
| 1,4096,32,128,7168 | 274.1 | 270.7 | **179.5** | 1340.1 | 1.53x | 1.51x |
| 1,4096,64,128,8192 | 524.7 | 526.0 | **436.6** | 1259.3 | 1.20x | 1.20x |
| 1,8192,64,128,8192 | 1044.3 | 1072.0 | **875.3** | 1256.2 | 1.19x | 1.22x |
| 2,1024,64,128,8192 | 310.3 | 296.2 | **232.3** | 1183.2 | 1.34x | 1.28x |
| 2,2048,32,128,4096 | 326.6 | 210.4 | **114.3** | 1202.4 | **2.86x** | **1.84x** |
| 2,2048,64,128,8192 | 524.1 | 568.8 | **443.4** | 1239.9 | 1.18x | 1.28x |
| 4,1024,64,128,4096 | 389.1 | 381.0 | **213.4** | 1288.2 | 1.82x | 1.79x |

### 逐 shape 性能（4 卡，时延 μs，bf16）

| Shape (bs,lseq,h,hd,K) | Torch | Separate | Fused | Fused TFLOPS | vs Torch | vs Sep |
|---|---:|---:|---:|---:|---:|---:|
| 1,1024,32,128,4096 | 107.5 | 80.9 | **47.1** | 729.4 | **2.28x** | 1.72x |
| 1,2048,64,128,8192 | 278.9 | 274.8 | **223.4** | 1230.5 | 1.25x | 1.23x |
| 1,4096,32,128,7168 | 272.5 | 266.5 | **177.1** | 1358.1 | 1.54x | 1.50x |
| 1,4096,64,128,8192 | 519.4 | 516.2 | **426.4** | 1289.2 | 1.22x | 1.21x |
| 1,8192,64,128,8192 | 1020.9 | 1046.9 | **846.8** | 1298.5 | 1.21x | 1.24x |
| 2,1024,64,128,8192 | 369.9 | 300.0 | **232.4** | 1183.0 | 1.59x | 1.29x |
| 2,2048,32,128,4096 | 211.3 | 203.5 | **110.4** | 1245.4 | **1.91x** | **1.84x** |
| 2,2048,64,128,8192 | 514.0 | 516.2 | **431.1** | 1275.4 | 1.19x | 1.20x |
| 4,1024,64,128,4096 | 390.9 | 378.6 | **210.4** | 1306.3 | 1.86x | 1.80x |

> 规律：小 K / 大 batch（计算占比低、通信占比高）的 shape 融合收益最大（如 `2,2048,32,128,4096` 达 2.86x）；
> 大 K 计算-bound 的 shape 收益收敛到 ~1.2x（通信被 GEMM 掩盖的空间有限），但仍稳定快于两条 baseline。

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
  单 kernel 单 stream。{2,4,8} 正确性 0.0，8 卡 geo vs torch 1.555x / vs sep 1.420x（4 卡 1.521x / 1.426x）。
  建立基线，补全逐 shape 明细表（见上）。
