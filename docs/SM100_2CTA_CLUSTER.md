# SM100 2-CTA Cluster BF16 GEMM 计算流程详解

> **最后更新**: 2026-06-11
> **验证脚本**: `debug_cluster_tiles.py`
> **参考 CUTLASS Tutorial**: `third-party/cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`

---

## 🧠 从宏观到微观：为什么之前 AI 总在自我怀疑

之前 AI 反复纠结的核心原因：**没有建立从「整个计算在做什么」到「每个硬件单元负责什么」的完整图景**。下面从三层彻底讲清楚。

### 第 1 层：整个计算在做什么？（矩阵乘法视角）

GEMM 就是 `C[M, N] = A[M, K] @ B[K, N]`。把大矩阵切成 tile 后：
```
每个 output tile 是 128×128 的 C 子矩阵
= 128 行 A @ 128 列 B 的转置
```

**2SM UMMA 做的事**：把两个相邻的 128×128 output tile 合并成一条 256×128 的指令一起算。
```
传统方式（2 条独立指令）：
  CTA0: A[m0:m0+128, :] @ B[:, n0:n0+128] → 128×128 tile_0
  CTA1: A[m1:m1+128, :] @ B[:, n0:n0+128] → 128×128 tile_1

2SM UMMA（1 条协作指令）：
  合并: A[m0:m0+256, :] @ B[:, n0:n0+128] → 256×128 大 tile
  = tile_0 (上 128 行) + tile_1 (下 128 行)
```

**关键洞察**：2SM UMMA 的 256 行 = 两个 CTA 各自的 128 行拼在一起。它们共享同一组 B 列，但各自有不同的 A 行。

### 第 2 层：Scheduler 怎么给每个 Block 分配 tile？（调度逻辑）

持久化调度模型：所有 SM 反复取 tile，直到所有 tile 算完。

```c++
// 每个 CTA 每次迭代获取的线性 block_idx
const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;
//                                                 ↑ 总 SM 数    ↑ 每个 CTA 唯一
```

**blockIdx.x 是什么？** 在 2-CTA cluster 模式下：
- 同一个 cluster 的 2 个 CTA 有**不同的** `blockIdx.x`（硬件保证）
- 比如 cluster 0 的 CTA0=blockIdx.x 0，CTA1=blockIdx.x 1

**线性 block_idx → (m_block_idx, n_block_idx) 的映射**（`get_swizzled_block_idx`）：

当 `kIsMulticastOnA=false`（我们的默认配置）：
```
M 是主维度 (primary)，N 是副维度 (secondary)
m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group   ← 相邻 in_group_idx → 相邻 m
n_block_idx = in_group_idx / num_blocks_in_group                     ← 同组内 n 相同
```

**实例**（假设 num_m_blocks=8, num_n_blocks=4, kNumSMs=4, cluster_m=2）：

| iter | blockIdx.x | block_idx | m_block | n_block | cluster |
|------|-----------|-----------|---------|---------|---------|
| 0 | 0 | 0 | 0 | 0 | CTA0 |
| 0 | 1 | 1 | 1 | 0 | CTA1 ← 和 CTA0 共享 n_block=0，m_block 差 1 |
| 0 | 2 | 2 | 2 | 0 | CTA0 |
| 0 | 3 | 3 | 3 | 0 | CTA1 |
| 1 | 0 | 4 | 4 | 0 | ... |
| ... | | | | | |

**核心**：`blockIdx.x` 差 1 的两个 CTA，`m_block_idx` 差 1，`n_block_idx` 相同 → 正好组成 2-CTA cluster 所需的「相邻 M-tile，共享 N-tile」。

### 第 3 层：2 个 CTA 怎么协作？（硬件执行视角）

```
┌─────────────────────────────────────────────────────────┐
│                    2-CTA Cluster                         │
│                                                         │
│   CTA0 (SM0)              CTA1 (SM1)                    │
│   m_block=X               m_block=X+1                   │
│   n_block=Y               n_block=Y                     │
│                                                         │
│   ① TMA Load:            ① TMA Load:                    │
│     A[X*128:(X+1)*128]     A[(X+1)*128:(X+2)*128]       │
│     → SMEM_A@SM0           → SMEM_A@SM1                 │
│     B[:,Y*128:Y*128+64]    B[:,Y*128+64:(Y+1)*128]      │
│     → SMEM_B前半@SM0       → SMEM_B后半@SM1             │
│                                                         │
│   ② UMMA (仅 CTA0/warp1 发射):                          │
│     2SM UMMA 硬件自动读取两个 SM 的 SMEM:                │
│     SMEM_A@SM0 + SMEM_A@SM1 → 合并 256 行 A             │
│     SMEM_B@SM0 + SMEM_B@SM1 → 拼接完整 128 列 B         │
│     → 计算 256×128 输出                                  │
│                                                         │
│   ③ Epilogue (各自独立):                                 │
│     TMEM[0:128] → D[X*128:(X+1)*128]                    │
│                     TMEM[128:256] → D[(X+1)*128:(X+2)*128]│
└─────────────────────────────────────────────────────────┘
```

### 之前 AI 纠结的三个问题 → 确切答案

| 纠结点 | 答案 | 依据 |
|--------|------|------|
| 两个 CTA 的 m_block_idx 相同还是不同？ | **不同**。blockIdx.x 不同 → block_idx 不同 → m_block_idx 不同 | scheduler 用 blockIdx.x，不用 cluster_idx |
| A 行加载：要不要偏移？ | **不需要额外偏移**。scheduler 已给不同 m_block_idx，各自 `m_idx = m_block_idx * BLOCK_M` 就不同 | 标准 GEMM 第 220-223 行确认 |
| B 列加载：怎么 split？ | **block_rank_in_cluster() 偏移 LOAD_BLOCK_N**。CTA0 加载前 64 列，CTA1 加载后 64 列 | 标准 GEMM `n_idx += block_rank * LOAD_BLOCK_N` |

---

## 📌 核心结论

**一个 2-CTA Cluster 中，每个 CTA 仍然各自计算一个独立的 128×128 output tile。**

但通过 2-CTA 协作：
1. **B 矩阵带宽减半**：B 列数据只从 HBM 加载一次，通过 TMA multicast 共享给两个 CTA
2. **UMMA 指令效率更高**：一条 256×128 的 2SM UMMA 比两条独立的 128×128 吞吐更高（减少指令开销）

---

## ⚙️ 配置参数（以 kIsMulticastOnA=False，即 cluster_m=2 为例）

| 参数 | 值 | 说明 |
|------|-----|------|
| BLOCK_M | 128 | 每个 CTA 的输出行数 |
| BLOCK_N | 128 | 每个 CTA 的输出列数 |
| LOAD_BLOCK_M | 128 | 每个 CTA 加载完整 M 行 |
| LOAD_BLOCK_N | 64 | 每个 CTA 只加载一半 N 列 |
| UMMA_M | 256 = 128 × 2 | 2SM UMMA 的 M 维度 |
| cluster_shape | (2, 1, 1) | M 方向 2 个 CTA 组成 cluster |

---

## 📐 Tile 分配逻辑

一个 Cluster 处理 **2 个相邻 M-tile，共享同一个 N-tile**：

```
Cluster 分配:
  CTA0 (block_rank=0): tile (m_blk=X,   n_blk=Y)
  CTA1 (block_rank=1): tile (m_blk=X+1, n_blk=Y)
```

Scheduler 中两个 CTA 各自独立调用 `get_next_block()`，但由于 cluster 约束，它们获得相邻的 M-tile 且共享同一个 N-tile。

---

## 🔄 三步数据流

### Step 1: TMA Load（两个 CTA 都执行）

```
CTA0:
  - 加载 A[X*128 : (X+1)*128, :]        → SMEM_A (CTA0)
  - 加载 B[:, Y*128 : Y*128+64]          → SMEM_B 前半 (CTA0)

CTA1:
  - 加载 A[(X+1)*128 : (X+2)*128, :]    → SMEM_A (CTA1)
  - 加载 B[:, Y*128+64 : (Y+1)*128]     → SMEM_B 后半 (CTA1)
```

**TMA `cta_group::2`**：使用 `SM100_TMA_2SM_LOAD_2D` 指令（`shared::cluster` 地址空间）。
每个 CTA 的 TMA load 数据只写入本地 SMEM，但 2SM UMMA 可通过 cluster shared memory 跨 SM 访问。

**注意**：这里不是 TMA multicast！`SM100_TMA_2SM_LOAD_2D` 没有 `multicast::cluster` 修饰。
数据不会被硬件复制到另一个 SM。而是 2SM UMMA 硬件自动从两个 SM 的 SMEM 读取。

**关键代码**（`sm100_bf16_gemm.cuh` 第 220-223 行）：
```c++
// B split: each CTA loads half
n_idx += kIsMulticastOnA ? 0 : (block_rank_in_cluster() * LOAD_BLOCK_N);
```

每个 CTA 只加载 B 的一半列（LOAD_BLOCK_N=64），2SM UMMA 跨两个 SM 的 SMEM 拼成完整的 128 列。

---

### Step 2: 2SM UMMA（仅 Leader CTA 发射指令）

```
tcgen05.mma.cta_group::2  (2SM MMA 指令)
```

- **仅 leader CTA（CTA0）的 warp_idx==1 发射 MMA 指令**
- 硬件自动读取两个 SM 的 SMEM：
  - 从 CTA0 的 SMEM 读取 A[X*128:(X+1)*128]
  - 从 CTA1 的 SMEM 读取 A[(X+1)*128:(X+2)*128]
  - 从两个 CTA 的 SMEM 读取完整 B[Y*128:(Y+1)*128]
- 计算一个 **256×128** 的输出存入 TMEM：

```
TMEM Layout (256×128):
┌────────────────────────────────┐
│ TMEM[0:128, 0:128]            │ = A[X*128:(X+1)*128] @ B[Y*128:(Y+1)*128]^T
│ (CTA0 的结果, 128×128)        │
├────────────────────────────────┤
│ TMEM[128:256, 0:128]          │ = A[(X+1)*128:(X+2)*128] @ B[Y*128:(Y+1)*128]^T
│ (CTA1 的结果, 128×128)        │
└────────────────────────────────┘
```

**关键**: TMEM 是 256KB/SM 的专用张量内存，2SM UMMA 可以跨两个 SM 的 TMEM 写入。

---

### Step 3: Epilogue（两个 CTA 都执行）

```
CTA0: 从 TMEM[0:128, 0:128] 读出   → 写到 D[X*128:(X+1)*128, Y*128:(Y+1)*128]
CTA1: 从 TMEM[128:256, 0:128] 读出 → 写到 D[(X+1)*128:(X+2)*128, Y*128:(Y+1)*128]
```

每个 CTA 独立负责自己那 128 行的输出写回。

---

## 📊 CUTLASS Tutorial 验证

来自 `04_mma_tma_2sm_sm100.cu` 的关键信息：

### TiledMMA Layout（第 456-464 行）

```
ThrLayoutVMNK: (_2,_1,_1,_1):(_1,_0,_0,_0)
LayoutA_TV:    (_2,(_128,_16)):(_128,(_1,_256))   // 2 CTAs, each sees 128 rows of A
LayoutC_TV:    (_2,(_128,_256)):(_128,(_1,_256))   // 2 CTAs, each sees 128 rows of C
```

- `ThrID = _2:_1` 表示 2 个 peer CTA
- Layout A 的第一维 `_2` 对应 2 个 CTA，各自看到 128 行
- Layout C 的第一维 `_2` 对应 2 个 CTA，各自输出 128 行

### Tutorial 注释（第 96-99 行）

> SM100 2SM tcgen05.mma instructions operate as follows:
> - MMA is launched by only one SM
> - With 2SM MMA instructions, only 1 of the 2 CTAs collaborating on MMA executes the instruction.
>   We call the collaborating CTAs, peer CTAs. And the CTA executing the MMA instruction is called leader CTA.

---

## 🔑 与 GEMM-RS 融合 Kernel 的关系

在 GEMM-RS 融合 kernel (`sm100_bf16_gemm_rs.cuh`) 中：

1. **当 multicast=2 时**：启用 2-CTA cluster 模式
   - 两个 CTA 协作计算相邻 M-tile
   - B 矩阵利用 multicast 减半带宽
   - 每个 CTA 的 Epilogue 独立将结果写入对应 dst_rank 的 partial buffer slot

2. **当 multicast=1 时**（当前临时状态）：
   - 每个 CTA 完全独立工作
   - 没有利用 2SM UMMA 的硬件优势
   - 导致 tensor core 利用率仅约一半

3. **性能影响**：
   - multicast=2 是最关键的性能优化路径
   - 预期可将 GEMM 计算效率提升 ~2x
   - 当前 benchmark 显示融合 kernel 仅 150-620 TFLOPS vs 标准 GEMM 的 1000-1250 TFLOPS

---

## 🧪 Debug 验证方法

运行 `debug_cluster_tiles.py` 可以模拟 scheduler 的 tile 分配：

```bash
cd /workspace/codebuddy/DeepGEMM
python debug_cluster_tiles.py
```

脚本会输出：
1. 每个 cluster 中 CTA0/CTA1 分到的 (m_block, n_block)
2. 各 CTA 的 TMA load 数据范围
3. UMMA 计算的等价矩阵乘
4. Epilogue 写回的 D 矩阵地址

---

## 📋 kIsMulticastOnA=True vs kIsMulticastOnA=False 对比

| | kIsMulticastOnA=False (cluster_m=2) | kIsMulticastOnA=True (cluster_n=2) |
|--|------|------|
| Cluster 方向 | M 方向 2 CTA | N 方向 2 CTA |
| A 加载 | 每个 CTA 加载不同 M 行 | 每个 CTA 加载同一 M 行（multicast） |
| B 加载 | 每个 CTA 加载一半 N 列（multicast） | 每个 CTA 加载不同 N 列 |
| UMMA 输出 | 256×128 (M 方向拼接) | 128×256 (N 方向拼接) |
| Epilogue | CTA0 写上半，CTA1 写下半 | CTA0 写左半，CTA1 写右半 |
| 共享矩阵 | B (multicast) | A (multicast) |

DeepGEMM 默认使用 **kIsMulticastOnA=False**（cluster_m=2），因为 M 方向相邻 tile 对应的 A 行不同，B 列相同，multicast B 更自然。

---

## 🏗️ 硬件约束和注意事项

1. **Cluster 大小固定为 2**: SM100 的 2SM UMMA 要求恰好 2 个 CTA 组成 cluster
2. **Leader CTA 唯一性**: 只有 `block_rank_in_cluster() == 0` 的 CTA 是 leader
3. **TMEM 跨 SM 写入**: 2SM UMMA 会跨两个 SM 的 TMEM 写入结果
4. **TMA Multicast 需对齐**: multicast 的数据必须地址对齐，且两个 CTA 的 SMEM 布局一致
5. **mbarrier 同步**: TMA load 完成后需要 barrier 同步，确保两个 CTA 的 SMEM 都就绪后才能发射 UMMA
6. **Epilogue 独立性**: 尽管 MMA 是协作的，Epilogue 阶段两个 CTA 完全独立

---

## 🐛 multicast=2 Bug 根因分析（已确认）

### 问题

GEMM-RS 中 `multicast=2` 时结果错误/hang。

### 根因

GEMM-RS 的 scheduler 使用 `cluster_idx`（同 cluster 内两个 CTA 相同）获取 tile，导致两个 CTA 获得**相同的 m_block_idx**，加载**相同的 A 行**。2SM UMMA 的 256 行输出中上下两半完全相同 → 结果错误。

而标准 GEMM 中，两个 CTA 通过不同的 `blockIdx.x` 获取 tile，得到**不同的 m_block_idx**：
```c++
const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;
```

### 修复方向

让两个 CTA 各自用独立的 `blockIdx.x`（或 `cta_rank`）调度，获得不同的 m_block_idx，就像标准 GEMM 一样。M-Swizzle 调度需确保两个 CTA 处理相邻 M-tile。

### 标准参考验证

**Scheduler M-major 排列**（`kIsMulticastOnA=false`）：
```c++
const auto primary_num_blocks = num_m_blocks;  // M 是 primary
m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
n_block_idx = in_group_idx / num_blocks_in_group;
```
相邻 block_idx 差 1 → m_block_idx 差 1，n_block_idx 相同 → 正是 cluster_m=2 需要的。

**Load 偏移**（标准 GEMM 第 220-223 行）：
```c++
if constexpr (kNumMulticast > 1) {
    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
}
```
`kIsMulticastOnA=false` → m_idx 不偏移（scheduler 已给不同 m_block），n_idx 按 block_rank 偏移。

### 已确认的正确行为

| 组件 | 行为 | 验证状态 |
|------|------|---------|
| Scheduler | 两个 CTA 用不同 blockIdx.x → 不同 m_block_idx | ✓ 与标准 GEMM 一致 |
| TMA Load | SM100_TMA_2SM_LOAD_2D（无 multicast），各 CTA 数据在本地 SMEM，shared::cluster 允许跨 SM 访问 | ✓ |
| A 加载 | 各 CTA 加载不同 m_block 的 128 行，无需额外 m_idx 偏移 | ✓ |
| B 加载 | 各 CTA 加载一半 N 列（n_idx += block_rank * 64） | ✓ |
| Barrier | leader expect_tx = (SMEM_A + SMEM_B) * 2 | ✓ 无需修改 |
| Epilogue TMEM | TMEM per-SM 本地，各 CTA 读自己的 128 行 | ✓ |
| Epilogue 写出 | local_m 不同 → 写不同行；ready_flag 按各自 local_m_block_idx 设置 | ✓ |