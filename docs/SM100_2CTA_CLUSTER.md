# SM100 2-CTA Cluster BF16 GEMM 计算流程详解

> **最后更新**: 2026-06-10
> **验证脚本**: `debug_cluster_tiles.py`
> **参考 CUTLASS Tutorial**: `third-party/cutlass/examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu`

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
答案是：在这种配置下（kIsMulticastOnA=false），标准 GEMM 的 scheduler 给两个 CTA 不同的 m_block_idx！回看标准 GEMM：

c++
复制
插入
新建文件
保存
应用
const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;
两个 CTA 的 blockIdx.x 不同，所以 next_block_idx 不同，得到不同的 m_block_idx。CTA0 加载 A 的 m_block_0 行，CTA1 加载 A 的 m_block_1 行。

但 load 时有 load_m_idx += 0（不偏移），这意味着两个 CTA 各自加载了不同 m_block 对应的 128 行 A：

CTA0: A[m_block_0 * 128 : m_block_0 * 128 + 128]
CTA1: A[m_block_1 * 128 : m_block_1 * 128 + 128]
而 multicast 使得两个 SM 的 SMEM 都有完整的数据。2SM UMMA 从两个 SM 的 SMEM 读取不同的 A 行，计算出 256 行输出：

TMEM[0:128] = CTA0 的 A 行 @ B
TMEM[128:256] = CTA1 的 A 行 @ B
这就是关键！标准 GEMM 中两个 CTA 获得不同的 m_block_idx，各自加载不同的 A 行，然后 2SM UMMA 跨两个 SM 的 SMEM 计算 256 行。

而在 GEMM-RS 中，我使用了 cluster_idx（两个 CTA 相同），所以两个 CTA 获得相同的 m_block_idx，加载相同的 A 行！这导致 UMMA 的 256 行输出中上下两半是完全相同的——这就是 multicast=2 时结果错误的根因！

修复方案：在 GEMM-RS 中，两个 CTA 应该获得不同的 m_block_idx，就像标准 GEMM 一样。具体来说：

Load warp 中：每个 CTA 应该用自己的 blockIdx.x 来调度，而不是 cluster_idx
但 M-Swizzle 调度需要确保两个 CTA 处理相邻的 M-tile
Epilogue 也需要对应修改
让我看看在标准 GEMM 中 scheduler 如何确保相邻 CTA 得到相邻 M-tile，以及 kIsMulticastOnA 的 swizzle 逻辑如何工作：

在 get_swizzled_block_idx 中，当 kIsMulticastOnA=false 时：

c++
复制
插入
新建文件
保存
应用
const auto primary_num_blocks = num_m_blocks;  // M 是 primary
m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
n_block_idx = in_group_idx / num_blocks_in_group;
相邻的 block_idx 差 1 → in_group_idx 差 1 → m_block_idx 差 1。所以相邻 blockIdx 的两个 CTA 确实得到相邻的 m_block_idx 但相同的 n_block_idx——这正是 cluster_m=2 需要的！

现在我明白了修复方向。在 GEMM-RS 中需要：

让两个 CTA 各自用独立的 block_idx 调度
scheduler 给出相邻 M-tile
Epilogue 中每个 CTA 写自己 m_block_idx 对应的数据
让我修改 GEMM-RS kernel：

读取
sm100_bf16_gemm_rs.cuh
L174-203
现在我理解了问题的根源。让我修复这个 bug。关键修改：

Scheduler 应该按 blockIdx.x 给每个 CTA 独立的 m_block_idx（就像标准 GEMM 一样），不能用 cluster_idx
Load warp 中 load_m_idx 不应再偏移 block_rank * LOAD_BLOCK_M（因为每个 CTA 已经加载不同 m_block 的 A 行）
N 方向的 B split 仍保持（load_n_idx += block_rank * LOAD_BLOCK_N）
但等等——这里有个复杂度：GEMM-RS 的 scheduler 有 M-Swizzle 逻辑（先算其他 rank 的 chunk），我需要确保两个 CTA 得到相邻 M-tile 且共享 N-tile。

让我仔细设计修复方案。看标准 GEMM scheduler 怎么做的（第 95-131 行 get_swizzled_block_idx）：当 kIsMulticastOnA=false 时，in_group_idx % num_blocks_in_group 给 m_block_idx，连续的 in_group_idx 给相邻的 m_block_idx。

在 GEMM-RS 中，我的自定义 get_next_block 计算方式：

c++
复制
插入
新建文件
保存
应用
const uint32_t local_m_block_idx = rem / num_n_blocks;
n_block_idx = rem - local_m_block_idx * num_n_blocks;
m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;
block_idx += kNumClusters;
这里是 N-major 排列：先遍历 N-tile，再递增 M-tile。所以连续的 block_idx 给相邻的 n_block_idx，不是 m_block_idx！

为了让两个 CTA 得到相邻的 M-tile 且相同的 N-tile，我需要改为 M-major 排列，或者用不同的方式处理 multicast=2。

实际上更简单的方案是：保持 cluster_idx 调度，但让两个 CTA 用 block_rank_in_cluster() 偏移自己的 m_block_idx：

c++
复制
插入
新建文件
保存
应用
// 每个 cluster 调度到 (m_block_pair_start, n_block)
// CTA0: m_block = m_block_pair_start * 2
// CTA1: m_block = m_block_pair_start * 2 + 1
但这需要 scheduler 调整。让我用更直接的方法——在 scheduler 中让每个 CTA 独立获取不同的 m_block。

在我明白了：

SM100_TMA_2SM_LOAD_2D（DeepGEMM 当前使用的）：cta_group::2 + shared::cluster = 两个 CTA 都执行 TMA，数据写入 cluster 共享 SMEM 空间
SM100_TMA_2SM_LOAD_MULTICAST_2D：cta_group::2 + shared::cluster + multicast::cluster = 额外添加 multicast，数据自动复制到 cluster 中所有 CTA 的 SMEM
关键区别：

无 multicast（SM100_TMA_2SM_LOAD）：每个 CTA 执行 TMA，数据只写入自己的 SMEM（但在 shared::cluster 地址空间，所以另一个 CTA 也能访问）
有 multicast（SM100_TMA_2SM_LOAD_MULTICAST）：数据被硬件复制到 cluster 中所有 CTA 的 SMEM
回到 DeepGEMM 的 tma::copy 函数。当 num_tma_multicast > 1 时，它使用的是 SM100_TMA_2SM_LOAD_2D（不带 multicast）。这意味着：

每个 CTA 执行 TMA 后，数据只在自己的 SMEM 中！ 但因为使用 shared::cluster 地址空间，2SM UMMA 可以跨两个 SM 的 SMEM 读取。

这就完全说通了！在标准 GEMM 中（kIsMulticastOnA=false）：

CTA0 TMA load A[m0] → CTA0 的 SMEM
CTA1 TMA load A[m1] → CTA1 的 SMEM
CTA0 TMA load B[n:n+64] → CTA0 的 SMEM（但因为 shared::cluster，CTA1 也能访问）
CTA1 TMA load B[n+64:n+128] → CTA1 的 SMEM（但因为 shared::cluster，CTA0 也能访问）
2SM UMMA 跨两个 SM 的 SMEM 读取 A[m0] + A[m1] + B[n:n+128]，计算 256×128 输出
所以在 GEMM-RS 中也应该这样：每个 CTA 加载自己的 A 行，B 按列 split，2SM UMMA 跨两个 SM 计算。

我刚才的 scheduler 修改是正确的！每个 CTA 通过 cta_rank 获得不同的 m_block_idx，加载不同 A 行。B 仍然 split（load_n_idx += block_rank * LOAD_BLOCK_N）。2SM UMMA 自动跨两个 SM 的 SMEM 计算。

但等等——我修改 Load warp 时移除了 A 的 m_idx 偏移，但保留了 B 的 n_idx 偏移。让我验证这是否和标准 GEMM 一致：

标准 GEMM（第 220-223 行）：

c++
复制
插入
新建文件
保存
应用
if constexpr (kNumMulticast > 1) {
    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
}
当 kIsMulticastOnA=false 时：m_idx += 0（不偏移 M），n_idx += block_rank * LOAD_BLOCK_N（偏移 N）。

但标准 GEMM 中两个 CTA 通过不同的 blockIdx.x 获得不同的 m_block_idx（scheduler 中用 blockIdx.x 而非 cluster_idx），所以 m_idx = m_block_idx * BLOCK_M 本身就不同。

我的修改也是这样：scheduler 中通过 cta_rank 给不同的 local_m_block_idx，所以 global_m = m_block_idx * BLOCK_M 对两个 CTA 不同。不需要额外的 m_idx 偏移。✓

好，但现在有一个问题：arrive_and_expect_tx 的字节数。在标准 GEMM 中：

c++
复制
插入
新建文件
保存
应用
constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
if (is_leader_cta) {
    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
} else {
    full_barriers[stage_idx]->arrive(0u);
}
kNumMulticast = 2，所以 expect 的字节 = (SMEM_A + SMEM_B) * 2。这是因为 cluster 中两个 CTA 都向同一个 barrier (leader CTA 的 barrier) 报告各自的 TMA 完成。每个 CTA 贡献 SMEM_A + SMEM_B 字节的 transaction。

在我的 GEMM-RS 中也是这样写的（第 517-523 行），所以不需要改。✓

现在还有 Epilogue 部分需要检查。Epilogue 中 TMEM 读取——根据上面的分析，每个 CTA 从自己 SM 的 TMEM 读自己的 128 行。但在 GEMM-RS epilogue 中（第 674 行），读取 TMEM 时用的地址是：

c++
复制
插入
新建文件
保存
应用
uint32_t tmem_col = accum_stage_idx * UMMA_N + s * STORE_BLOCK_N + st * kElemsPerStore;
这个 tmem_col 对两个 CTA 相同。但由于 TMEM 是每个 SM 本地的，每个 CTA 读到的是自己 SM 的 128 行数据。CTA0 读到的是 A[m0] @ B 的结果，CTA1 读到的是 A[m1] @ B 的结果。✓

Epilogue 写出时（第 716-728 行）：

c++
复制
插入
新建文件
保存
应用
uint32_t base_row = local_m + w * STORE_BLOCK_M;
uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;
comm_dtype_t* dst_ptr = workspace.get_partial_ptr<comm_dtype_t>(dst_rank, base_row + row, base_col);
这里 local_m = local_m_block_idx * BLOCK_M。因为两个 CTA 的 m_block_idx 不同，它们的 local_m_block_idx 也不同，所以 local_m 不同。CTA0 写 row 0..127，CTA1 写 row 128..255。✓

Ready flag（第 742 行）：

c++
复制
插入
新建文件
保存
应用
auto* ready_ptr = workspace.get_ready_ptr(dst_rank, local_m_block_idx, n_block_idx);
两个 CTA 的 local_m_block_idx 不同，所以设置不同 tile 的 ready flag。✓