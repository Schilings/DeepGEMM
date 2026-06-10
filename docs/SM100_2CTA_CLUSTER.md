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

**TMA Multicast**：B 的两半通过 TMA multicast 分发到两个 CTA 的 SMEM，使得两个 SM 都拥有完整的 B[Y*128:(Y+1)*128]。

**关键代码**（`sm100_bf16_gemm.cuh` 第 220-223 行）：
```c++
// TMA load offset by block_rank_in_cluster()
auto const n_offset = block_rank_in_cluster() * LOAD_BLOCK_N;
```

每个 CTA 只加载 B 的一半列（LOAD_BLOCK_N=64），但 multicast 确保两个 CTA 的 SMEM 中都有完整的 128 列。

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
