# DeepGEMM Scheduler 深度源码解析

> 逐行剖析 `gemm.cuh`，从 work-stealing 到 swizzle，覆盖 6 种 GEMM 类型的所有分支差异

---

## 目录

- [1. 核心问题与 Work-Stealing](#1-核心问题与-work-stealing)
- [2. Swizzle 分组：L2 Cache 优化](#2-swizzle-分组l2-cache-优化)
- [3. 六大 GEMM 类型速览](#3-六大-gemm-类型速览)
- [4. 构造函数](#4-构造函数)
- [5. get_swizzled_block_idx()](#5-get_swizzled_block_idx)
- [6. get_global_idx()](#6-get_global_idx)
- [7. get_next_block()](#7-get_next_block)
- [8. 辅助函数](#8-辅助函数)

---

## 1. 核心问题与 Work-Stealing

### 问题

矩阵乘法 `C[M,N] = A[M,K] × B[K,N]`，每个 CTA 处理 `BLOCK_M × BLOCK_N` 的 tile。

**关键矛盾**：tile 数量通常远大于 SM 数 → 如何分配？且 CTA 速度不均（有的 K block 快，有的慢）。

### 传统方案缺陷

| 方案 | 问题 |
|------|------|
| 静态分配（grid-stride loop） | 快的 CTA 空等慢的，无法动态适应 |
| 原子操作抢 tile | 所有 CTA 争一个原子 → 串行瓶颈 |
| 多次 kernel launch | 启动开销 μs 级，对大矩阵不可用 |

### DeepGEMM 方案

**Persistently-scheduled work-stealing**，单次 kernel launch：

```
next_tile = current_iter × kNumSMs + blockIdx.x
```

每个 CTA 独立递增 `current_iter`（私有寄存器变量），零原子操作，零同步。

**推演（132 SM，400 tile）**：

```
iter=0: CTA 0→t0, CTA 1→t1, ..., CTA 131→t131     (全员有活)
iter=1: CTA 0→t132, ..., CTA 131→t263               (全员有活)
iter=2: CTA 0→t264, ..., CTA 131→t395               (全员有活)
iter=3: CTA 0→t396 ✓, CTA 1→t397 ✓, CTA 2→t398 ✓, CTA 3→t399 ✓
        CTA 4→t400 ✗ (≥400, stop), ..., CTA 131→stop
iter=4: 全员 stop

分配: 前 4 CTA 各 4 tile，后 128 CTA 各 3 tile
      = 16 + 384 = 400 ✓
```

快的 CTA 多抢（多迭代），慢的少抢 → **天然负载均衡**。

---

## 2. Swizzle 分组：L2 Cache 优化

### 为什么需要

假设 100 M-blocks × 4 N-blocks，逐行遍历：

```
CTA 0: (M0,N0)→(M0,N1)→(M0,N2)→(M0,N3)
CTA 1: (M1,N0)→(M1,N1)→(M1,N2)→(M1,N3)
```

CTA 0 跳到 `(M132,N0)` 时，HBM 中两批 tile 间距 ≈ `132 × M_blk_size` ≈ MB 级 → **L2 一定 miss**。

### Swizzle 方案

把 tile 分"组"（8 或 16 个一批），组内 M 方向连续：

```
组 0: (M0,N0), (M1,N0), ..., (M7,N0)   ← M 连续 8 个
      (M0,N1), (M1,N1), ..., (M7,N1)
      ...(所有 N 列 × 8 M 行)
组 1: (M8,N0), ..., (M15,N0), ...
组 2: (M16,N0), ...
```

组间跳跃时，相邻组的 M 行在 HBM 中连续 → L2 prefetch 命中。

### 分组大小选择

源码：

```cpp
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M :
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N;
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}
```

**`usage` 含义**：一组 tile 覆盖的元素总数。`candidate * [分组维度block大小]` 是组宽度，`ceil(SMs/candidate)` 是需要同时存活的组数。选 `usage` 最小 → L2 footprint 最小。

**实例（B200, BLOCK_M=N=128, SMs=132, kIsMulticastOnA=false）**：

```
candidate=8:  usage = 8*128 + ceil(132/8)*128 = 1024 + 17*128 = 3200
candidate=16: usage = 16*128 + ceil(132/16)*128 = 2048 + 9*128 = 3200
→ 相同，选小的：kNum1DBlocksPerGroup = 8
```

**`kIsMulticastOnA` 的影响**：

- `true`（multicast 在 A 侧，N 维被共享）→ 分组在 **N 方向**（M 不重要）
- `false`（multicast 在 B 侧，M 维被共享）→ 分组在 **M 方向**（N 不重要）

---

## 3. 六大 GEMM 类型速览

| 类型 | 场景 | **num_blocks** | **group 含义** | **get_global_idx 特点** |
|------|------|---------------|---------------|----------------------|
| **Normal** | 单矩阵乘 | `M×N` 网格 | 无 | `block_idx * block_size` |
| **MGroupedContiguous** | MoE 推理（tokens 拼接） | 同 Normal | expert 按 M 切分 | `expert_id * shape_dim + ...` |
| **MGroupedMasked** | MoE 推理（独立存储） | 动态（跨 expert) | expert 循环 | `group_idx * shape_dim + ...` |
| **KGroupedContiguous** | SwiGLU gate+up | `M×N` 固定 | K 组循环 | MN 用 `group_idx`，K 用 `cumsum`，SF_K 用 `sf_cumsum` |
| **Batched** | Batch MatMul | `batch × M×N` | batch 索引 | MN 无偏移，SF_K 用 `group_idx` |
| **MGroupedContiguousWithPsumLayout** | MoE 训练 psum | 动态（跨 psum 组） | psum 组循环 | 同 MGroupedMasked |

---

## 4. 构造函数

**完整源码**：

```cpp
CUTLASS_DEVICE explicit Scheduler(
    const uint32_t& shape_m, const uint32_t& shape_n,
    const uint32_t& shape_k, int* grouped_layout = nullptr)
{
    num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
    num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
    current_shape_k = shape_k;

    if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
        // (1) Normal / Batched
        num_blocks = num_m_blocks * num_n_blocks;

    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        // (2) MGroupedContiguous
        num_blocks = num_m_blocks * num_n_blocks;
        this->grouped_layout = grouped_layout;

    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
        // (3) MGroupedMasked
        this->grouped_layout = grouped_layout;
        // num_blocks 不设！每个 expert 的 M tiles 不同，get_next_block 动态算

    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        // (4) Psum Layout
        this->grouped_layout = grouped_layout;
        current_psum_m = grouped_layout[0];
        num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        // num_blocks 不显式设，get_next_block 动态维护

    } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
        // (5) KGroupedContiguous
        num_blocks = num_m_blocks * num_n_blocks;
        this->grouped_layout = grouped_layout;
        get_next_k_group(current_group_idx, current_shape_k);
        next_group_idx = current_group_idx + 1;
        get_next_k_group(next_group_idx, next_shape_k);
    }
}
```

### 逐类型分析

#### (1) Normal / Batched

```
num_m_blocks = ceil(M / BLOCK_M)   // e.g. 4096/128=32
num_n_blocks = ceil(N / BLOCK_N)   // e.g. 2048/128=16
num_blocks   = 32 × 16 = 512
```

最简单的场景。Batched 构造时和 Normal 完全一样，调度时才有 batch 分支。

#### (2) MGroupedContiguous

```
num_blocks = num_m_blocks * num_n_blocks   // 同 Normal
grouped_layout 被保存，但 num_blocks 完全同 Normal
```

**关键**：虽然所有 expert 的 tokens 拼成了一个大矩阵，但调度角度看和 Normal 一模一样。grouped_layout 只在 `get_global_idx` 和 SM90 multicast 检查时用到。

#### (3) MGroupedMasked

```
不设 num_blocks！
num_m_blocks = ceil(shape_m / BLOCK_M)   // shape_m 是第一个 expert 的 M
num_n_blocks = ceil(shape_n / BLOCK_N)
```

**为什么 shape_m 是第一个 expert 的**？调用方传的 shape_m 就是首个 expert 的 token 数。后续 expert 的 num_m_blocks 在 `get_next_block` 中动态更新。`num_blocks` 不存，因为每个 expert 的 tile 数不同。

#### (4) MGroupedContiguousWithPsumLayout

```
current_psum_m = grouped_layout[0]           // 第一个 psum 的 M
num_m_blocks = ceil(current_psum_m, BLOCK_M)
```

第一组的 psum 决定初始 tile 数，后续组的 M 可能不同。

#### (5) KGroupedContiguous

```
num_blocks = num_m_blocks × num_n_blocks   // 固定
get_next_k_group(current_group_idx, current_shape_k)  // 首个非零 K 组
next_group_idx = current_group_idx + 1
get_next_k_group(next_group_idx, next_shape_k)        // 预取下一组
```

**K 组初始化**：`get_next_k_group` 从 grouped_layout[0] 开始扫描，跳过 `shape_k=0` 的空组。找到第一个非零组作为 `current_shape_k`，预取 `next_shape_k`。这样在 `get_next_block` 中切换时不需要查表。

---

## 5. get_swizzled_block_idx()

将扁平 tile 编号 → `(m_block_idx, n_block_idx)` 的 swizzle 映射。

**完整源码（剔注释）**：

```cpp
CUTLASS_DEVICE void get_swizzled_block_idx(
    const uint32_t& block_idx,
    uint32_t& m_block_idx, uint32_t& n_block_idx)
{
    DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

    // Step 1: 计算 primary/secondary 维度
    const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
    const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;

    // Step 2: 计算组号和组内偏移
    const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
    const auto group_idx = block_idx / num_blocks_per_group;
    auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
    auto in_group_idx = block_idx % num_blocks_per_group;
    num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

    // Step 3: SM90 multicast 对齐修正
#if __CUDA_ARCH__ < 1000
    if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
        if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
            num_blocks_in_group = num_blocks_in_group ^ 1;
        } else {
            in_group_idx -= (num_blocks_in_group ^ 1) * secondary_num_blocks;
            first_block_idx += num_blocks_in_group ^ 1;
            num_blocks_in_group = 1;
        }
    }
#endif

    // Step 4: 映射到 (m, n)
    if constexpr (kIsMulticastOnA) {
        m_block_idx = in_group_idx / num_blocks_in_group;
        n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    } else {
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
    }
}
```

### 逐步推演（Normal, kIsMulticastOnA=false）

假设 `kNum1DBlocksPerGroup=8`, `num_m_blocks=100`, `num_n_blocks=4`, `block_idx=137`：

**Step 1**：
```
primary_num_blocks = 100 (分组在 M)
secondary_num_blocks = 4
```

**Step 2**：
```
num_blocks_per_group = 4 × 8 = 32
group_idx = 137 / 32 = 4
first_block_idx = 4 × 8 = 32
in_group_idx = 137 % 32 = 9
num_blocks_in_group = min(8, 100-32) = 8
```

**Step 3**: SM100 → 跳过

**Step 4**（分组在 M）：
```
m_block_idx = 32 + 9 % 8 = 32 + 1 = 33
n_block_idx = 9 / 8 = 1
```

→ tile 137 对应 `(M33, N1)`

### 验证：组 4 的 tile 范围

```
组 4: (M32,N0)..(M39,N0) → block_idx: 4*32 + 0..7     = 128..135
      (M32,N1)..(M39,N1) → block_idx: 4*32 + 8..15    = 136..143  ← 137 在这里
      (M32,N2)..(M39,N2) → block_idx: 4*32 + 16..23   = 144..151
      (M32,N3)..(M39,N3) → block_idx: 4*32 + 24..31   = 152..159

in_group_idx=9 → 第二行 (N=1) 的第一个 tile (M=32+0?... 不对)
  9 % 8 = 1, 9 / 8 = 1
  → m = 32 + 1 = 33, n = 1 ✓
```

### kIsMulticastOnA=true 时的差异

```
// 分组在 N 方向
primary_num_blocks = num_n_blocks   // 4
secondary_num_blocks = num_m_blocks // 100
num_blocks_per_group = 100 × 8 = 800

// 映射为:
m_block_idx = in_group_idx / num_blocks_in_group;      // M 递增
n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group; // N 用 first_block 偏移
```

分组在 N 意味着组内 tile 共享 N 方向的前 8 个 block，M 方向按列展开。

### SM90 multicast 修正

```
if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
    // 组大小奇数时，末尾 1 个 tile 只能单播
    // 前 (num_blocks_in_group ^ 1) 个 tile 仍可双播
    // 最后一个 tile: first_block_idx += 偶数, in_group_idx 调整
}
```

`num_blocks_in_group ^ 1`：把组大小从奇数变成相邻偶数（如 9→8, 15→14）。SM90 multicast 要求偶数，SM100 的 2-CTA 不可动态禁用，所以不在 SM100 上执行此修正。

---

## 6. get_global_idx()

**最频繁调用的函数** — kernel 中每次 TMA load 都靠它计算 HBM 地址。

**完整源码**：

```cpp
template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
CUTLASS_DEVICE uint32_t get_global_idx(
    const uint32_t shape_dim, const uint32_t block_size,
    const uint32_t& block_idx, const uint32_t& m_block_idx = 0)
{
    if constexpr (kGemmType == GemmType::Normal) {
        // ─── (1) Normal ───
        return block_idx * block_size;

    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        // ─── (2) MGroupedContiguous ───
        const auto offset = kWithGroupOffset
            ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M])  // 查 expert 编号
            : 0;
        return offset * shape_dim + block_idx * block_size;

    } else if constexpr (kGemmType == GemmType::MGroupedMasked
                      or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        // ─── (3) MGroupedMasked ───
        const auto offset = kWithGroupOffset ? current_group_idx : 0;
        return offset * shape_dim + block_idx * block_size;

    } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
        // ─── (4) KGroupedContiguous ───
        auto offset = 0;
        if constexpr (kWithGroupOffset) {
            if constexpr (kIndexType == IndexType::MN)
                offset = current_group_idx * shape_dim;
            else if constexpr (kIndexType == IndexType::K)
                offset = current_k_cumsum;
            else if constexpr (kIndexType == IndexType::SF_K)
                offset = current_sf_k_cumsum;
        }
        return offset + block_idx * block_size;

    } else if constexpr (kGemmType == GemmType::Batched) {
        // ─── (5) Batched ───
        const auto offset = kIndexType == IndexType::SF_K
            ? current_group_idx : 0;
        return offset * shape_dim + block_idx * block_size;
    }
}
```

### 调用点速查表（以 fp8 1d1d kernel 为例）

| 矩阵 | 维度 | `kWithGroupOffset` | `kIndexType` | 返回值含义 |
|------|------|---------------------|-------------|-----------|
| A | M | `kGemmType==MGroupedMasked` | `MN` | A 的 M 偏移 |
| B | N | `kMajorB==Major::K` | `MN` | B 的 N 偏移 |
| A | K | `kMajorA==Major::MN` | `K` | A 的 K 偏移 |
| B | K | `kMajorB==Major::MN` | `K` | B 的 K 偏移 |
| SF A | K | `not is_m_grouped_contiguous` | `SF_K` | SFA 的 K 偏移 |
| SF B | K | `true` | `SF_K` | SFB 的 K 偏移 |
| D | M | `not is_m_grouped_contiguous` | `MN` | 输出 D 的 M 偏移 |

### 逐类型详细分析

#### (1) Normal

```cpp
return block_idx * block_size;
```

完全线性映射。M 偏移 = `m_block_idx * BLOCK_M`，N 偏移 = `n_block_idx * BLOCK_N`。

**为什么 `kWithGroupOffset` 和 `kIndexType` 模板参数还在？** 编译期消去。调用时传了这些参数，但 Normal 分支完全无视它们，编译器直接优化掉。

#### (2) MGroupedContiguous

```cpp
const auto offset = kWithGroupOffset
    ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M])
    : 0;
return offset * shape_dim + block_idx * block_size;
```

**核心思路**：A 是所有 expert 的 tokens 拼接，B 按 expert 分组存储。

- **A 矩阵的 M 偏移**：`kWithGroupOffset = false`，直接 `m_block_idx * BLOCK_M` — A 是连续矩阵，不需要分组
- **B/D 矩阵的 N 偏移**：`kWithGroupOffset = true`，`grouped_layout[m_block_idx * BLOCK_M]` 查该行对应哪个 expert，然后 `expert_id * shape_n` 跳到该 expert 的 B/D 存储区

**示例**（3 个 expert，shape_m=600，shape_n=256 per expert，BLOCK_M=128）：

```
grouped_layout: [expert0, expert0, expert0, ..., expert1, expert1, ..., expert2, ...]

m_block_idx=0 → 行 0     → expert_id=0 → B 偏移 = 0*256 = 0
m_block_idx=1 → 行 128   → expert_id=0 → B 偏移 = 0
m_block_idx=2 → 行 256   → expert_id=1 → B 偏移 = 1*256 = 256  ← 跳到 expert 1 的 B
m_block_idx=4 → 行 512   → expert_id=2 → B 偏移 = 2*256 = 512
```

#### (3) MGroupedMasked

```cpp
const auto offset = kWithGroupOffset ? current_group_idx : 0;
return offset * shape_dim + block_idx * block_size;
```

每个 expert 独立存储 A 和 B。`current_group_idx` 跟踪当前 Expert。

- **A 矩阵 M**：`kWithGroupOffset = true`，`current_group_idx * shape_m + m_block_idx * BLOCK_M`
  - `shape_m` 是**每个 expert 的 M 维度**（不是总 M）
- **B 矩阵 N**：`kWithGroupOffset = false`（B 不是按 expert 分组的）

#### (4) KGroupedContiguous

```cpp
auto offset = 0;
if constexpr (kWithGroupOffset) {
    if constexpr (kIndexType == IndexType::MN)
        offset = current_group_idx * shape_dim;
    else if constexpr (kIndexType == IndexType::K)
        offset = current_k_cumsum;
    else if constexpr (kIndexType == IndexType::SF_K)
        offset = current_sf_k_cumsum;
}
return offset + block_idx * block_size;
```

**三种 IndexType 对应三种偏移来源**：

- **MN**：`current_group_idx * shape_dim` — K 组的 M/N 维偏移（不常用）
- **K**：`current_k_cumsum` — 前面 K 组的累积和 → 跳转到当前 K 组的起始位置
- **SF_K**：`current_sf_k_cumsum` — 前面 K 组的 SF 累积和 → 跳转到 SF 的正确位置

**示例（K0=4096, K1=2048, SF_K_ALIGNMENT=512）**：

```
处理 K0 组:
  get_global_idx<true, K>(shape_k=?, block_size=BLOCK_K, block_idx=3)
  → offset = current_k_cumsum = 0, result = 3 * BLOCK_K
  → 加载 K0 组的第 4 个 K block

  get_global_idx<true, SF_K>(shape_sfa_k=?, block_size=1, block_idx=2)
  → offset = current_sf_k_cumsum = 0, result = 2
  → 加载 K0 组的第 3 个 SF

处理 K1 组:
  current_k_cumsum = 4096, current_sf_k_cumsum = ceil(4096/512) = 8
  get_global_idx<true, K>(..., block_idx=1)
  → offset = 4096, result = 4096 + 1*BLOCK_K
  → 加载 K1 组的第 2 个 K block

  get_global_idx<true, SF_K>(..., block_idx=0)
  → offset = 8, result = 8
  → 加载 K1 组的第 1 个 SF
```

#### (5) Batched

```cpp
const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
return offset * shape_dim + block_idx * block_size;
```

- 普通维度（MN）：`offset = 0`，batch_idx 通过 TMA 的 3D descriptor 处理
- SF_K：`offset = current_group_idx * shape_dim`，不同 batch 的 SF 独立存储

---

## 7. get_next_block()

**主调度循环的入口**。Kernel 中的典型用法：

```cpp
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // ① 等待 pipeline barrier
    // ② get_global_idx → HBM 偏移
    // ③ TMA load A/B/SF
    // ④ UMMA fma
    // ⑤ epilogue 写回
}
```

**完整源码**：

```cpp
CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // ══════════════════════════════════════════
    // WORK-STEALING: 每个 CTA 独立计算 tile 编号
    // ══════════════════════════════════════════
    const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

    // ══════════════════════════════════════════
    // 分支 (3): MGroupedMasked
    // ══════════════════════════════════════════
    if constexpr (kGemmType == GemmType::MGroupedMasked) {
        while (true) {
            if (current_group_idx == kNumGroups)
                return false;

            num_m_blocks = math::ceil_div(
                static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
            const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
            if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                break;

            current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
        }
        get_swizzled_block_idx(
            next_block_idx - current_m_cumsum * num_n_blocks,
            m_block_idx, n_block_idx);

    // ══════════════════════════════════════════
    // 分支 (6): MGroupedContiguousWithPsumLayout
    // ══════════════════════════════════════════
    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        while (true) {
            if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                break;
            if (++ current_group_idx == kNumGroups)
                return false;

            last_psum_m = math::align(current_psum_m, BLOCK_M);
            current_psum_m = grouped_layout[current_group_idx];
            current_m_block_cumsum += num_m_blocks;
            num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
        }
        get_swizzled_block_idx(
            next_block_idx - current_m_block_cumsum * num_n_blocks,
            m_block_idx, n_block_idx);
        m_block_idx += last_psum_m / BLOCK_M;

    // ══════════════════════════════════════════
    // 分支 (5): KGroupedContiguous
    // ══════════════════════════════════════════
    } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
        while (true) {
            if (current_group_idx == kNumGroups)
                return false;
            if (next_block_idx < (current_num_valid_groups + 1) * num_blocks)
                break;

            current_k_cumsum += current_shape_k;
            current_sf_k_cumsum += math::ceil_div(current_shape_k, SF_K_ALIGNMENT);
            current_num_valid_groups ++;

            current_group_idx = next_group_idx ++;
            current_shape_k = next_shape_k;
            get_next_k_group(next_group_idx, next_shape_k);
        }
        get_swizzled_block_idx(
            next_block_idx - current_num_valid_groups * num_blocks,
            m_block_idx, n_block_idx);

    // ══════════════════════════════════════════
    // 分支 (4): Batched
    // ══════════════════════════════════════════
    } else if constexpr (kGemmType == GemmType::Batched) {
        if (next_block_idx >= num_blocks * kNumGroups)
            return false;
        current_group_idx = next_block_idx / num_blocks;
        const auto block_idx = next_block_idx - current_group_idx * num_blocks;
        if constexpr (kIsMulticastOnA) {
            m_block_idx = block_idx / num_n_blocks;
            n_block_idx = block_idx % num_n_blocks;
        } else {
            m_block_idx = block_idx % num_m_blocks;
            n_block_idx = block_idx / num_m_blocks;
        }

    // ══════════════════════════════════════════
    // 分支 (1,2): Normal / MGroupedContiguous
    // ══════════════════════════════════════════
    } else {
        if (next_block_idx >= num_blocks)
            return false;
        is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or
                            num_m_blocks % kNumMulticast == 0 or
                            (next_block_idx ^ 1) < num_blocks;
        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    }
    return true;
}
```

### 逐分支详细走读

#### 分支 (1,2)：Normal / MGroupedContiguous — 标准 swizzle

```cpp
if (next_block_idx >= num_blocks) return false;
is_peer_cta_alive = ...;
get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
```

**最简洁的分支**。`num_blocks = M × N` 总 tile 数。直接边界检查 + swizzle。

`is_peer_cta_alive`（SM90 专用）：判断 multicast 的另一个 CTA 是否还有 tile 可做。3 种情况下永远存活：
- N 块数被 multicast 整除 → 所有组都是满的
- M 块数被 multicast 整除 → 同上
- `next_block_idx ^ 1 < num_blocks` → xor 翻转 peer 的 tile 还在范围内

**推演（MGroupedContiguous=Normal 调度上等价）**：上面 4.2 已详述。

#### 分支 (3)：MGroupedMasked — 跨 expert 循环

```cpp
while (true) {
    if (current_group_idx == kNumGroups) return false;
    num_m_blocks = ceil_div(grouped_layout[current_group_idx], BLOCK_M);
    const auto cumsum = current_m_cumsum + num_m_blocks;
    if (next_block_idx < cumsum * num_n_blocks) break;
    current_group_idx++, current_m_cumsum = cumsum;
}
get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, ...);
```

**核心机制**：`next_block_idx` 是全局 tile 编号。while 循环判断它落在哪个 expert。找到后，减去前面所有 expert 的 tile 数，在当前 expert 内做 swizzle。

**推演（3 个 expert：M0=300→3blocks, M1=500→4blocks, M2=200→2blocks）**：

```
num_n_blocks = 2
Expert 0: 3 * 2 = 6 tiles  → cumsum 范围 [0, 6)
Expert 1: 4 * 2 = 8 tiles  → cumsum 范围 [6, 14)
Expert 2: 2 * 2 = 4 tiles  → cumsum 范围 [14, 18)

CTA 0, iter=-1: next = 0 × 132 + 0 = 0
  while:
    expert 0: cumsum = 0+3 = 3, 0 < 3*2=6 ✓ → break
  get_swizzled(0 - 0*2, m, n) = get_swizzled(0, m, n)

CTA 0, iter=0: next = 132
  while:
    expert 0: 132 < 6? ✗
    expert 1: cumsum = 3+4 = 7, 132 < 7*2=14? ✗
    expert 2: cumsum = 7+2 = 9, 132 < 9*2=18? ✗
    expert 3 == kNumGroups(3) → return false
  仅 1 tile

CTA 10, iter=-1: next = 10
  while:
    expert 0: 10 < 6? ✗
    expert 1: cumsum = 7, 10 < 14? ✓ → break
  get_swizzled(10 - 6, m, n) = get_swizzled(4, m, n)
  → expert 1 的组内第 4 个 tile
```

**为何不存 num_blocks**：每个 expert 的 M 块数不同（3, 4, 2），无法统一。每次循环动态计算。

#### 分支 (5)：KGroupedContiguous — 跨 K 组循环

```cpp
while (true) {
    if (current_group_idx == kNumGroups) return false;
    // num_blocks 固定! 每轮处理完整的 M×N 网格
    if (next_block_idx < (current_num_valid_groups + 1) * num_blocks) break;

    current_k_cumsum += current_shape_k;
    current_sf_k_cumsum += ceil_div(current_shape_k, SF_K_ALIGNMENT);
    current_num_valid_groups++;

    current_group_idx = next_group_idx++;
    current_shape_k = next_shape_k;
    get_next_k_group(next_group_idx, next_shape_k);
}
get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_blocks, ...);
```

**与 MGroupedMasked 的本质区别**：所有 CTA **先处理完 K0 组的所有 M×N tile，再推进到 K1 组**。这意味着：

- `num_blocks` 是 M×N 网格，始终固定
- 每处理完 `num_blocks` 个 tile，推进一个 K 组
- 切换时更新 K 累积偏移（用于 `get_global_idx` 计算正确的 HBM 地址）

**推演（K0=4096, K1=2048, M-blocks=4, N-blocks=2, SMs=132）**：

```
num_blocks = 8

CTA 0, iter=-1: next = 0
  valid_groups=0: 0 < 1*8=8 ✓ → break
  get_swizzled(0-0*8, m, n) → K0 组的 (m0, n0) tile

CTA 0, iter=0: next = 132
  valid_groups=0: 132 < 8? ✗
  → 推进 K 组! current_k_cumsum = 0+4096 = 4096
          current_sf_k_cumsum = 0+8 = 8
          valid_groups = 1
  valid_groups=1: 132 < 2*8=16 ✓ → break
  get_swizzled(132-8, m, n) = get_swizzled(124, m, n)
  → K1 组的某个 tile

CTA 0, iter=1: next = 264
  valid_groups=1: 264 < 16? ✗
  valid_groups=2: current_group_idx=2 == kNumGroups=2 → return false
  总共 2 tile (K0 一个, K1 一个)
```

**K 组切换时维护的累积偏移**：

```
current_k_cumsum += current_shape_k           // K 累积 → 用于 IndexType::K 偏移
current_sf_k_cumsum += ceil(K, SF_K_ALIGNMENT) // SF 累积 → 用于 IndexType::SF_K 偏移
```

#### 分支 (4)：Batched — 线性跨 batch

```cpp
if (next_block_idx >= num_blocks * kNumGroups)
    return false;
current_group_idx = next_block_idx / num_blocks;
const auto block_idx = next_block_idx - current_group_idx * num_blocks;
if constexpr (kIsMulticastOnA) {
    m_block_idx = block_idx / num_n_blocks;
    n_block_idx = block_idx % num_n_blocks;
} else {
    m_block_idx = block_idx % num_m_blocks;
    n_block_idx = block_idx / num_m_blocks;
}
```

**不做 swizzle**。每个 batch 独立，跨 batch 跳跃没有 cache 局部性（数据在不同位置）。

`current_group_idx = next_block_idx / num_blocks`：直接除总 M×N tile 数得到 batch 号。

#### 分支 (6)：MGroupedContiguousWithPsumLayout

```cpp
while (true) {
    if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
        break;
    if (++current_group_idx == kNumGroups) return false;

    last_psum_m = math::align(current_psum_m, BLOCK_M);
    current_psum_m = grouped_layout[current_group_idx];
    current_m_block_cumsum += num_m_blocks;
    num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
}
get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks,
                       m_block_idx, n_block_idx);
m_block_idx += last_psum_m / BLOCK_M;
```

**与 MGroupedMasked 类似的 while 循环**，但：
- `num_m_blocks` 动态变化（因为 psum 不规整）
- 切换时更新 `last_psum_m`（前一组对齐后的 M）、`current_psum_m`（当前组的 M）
- 最后 `m_block_idx += last_psum_m / BLOCK_M`：加上前面所有组的全局 M 偏移

---

## 8. 辅助函数

### is_computation_valid() — SM90 尾部有效性检查

```cpp
CUTLASS_DEVICE bool is_computation_valid(
    const uint32_t& m_block_idx, const uint32_t& m_offset) const
{
    if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
        return true;  // Normal 没有尾部问题
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
        return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        // 检查行号对应的 grouped_layout 是否标记为有效 (>=0)
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
        return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        // 检查 M 偏移是否在当前 expert 的 token 范围内
    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
        return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        // 检查 M 偏移是否在当前 psum 范围内
    } else {
        DG_TRAP_ONLY_DEVICE_ASSERT(false);  // KGroupedContiguous 不需要
    }
}
```

**使用场景（SM90 UMMA）**：MGroupedContiguous/Masked 时，末尾 expert 的 tokens 可能不足一个完整的 BLOCK_M。在 WGMMA 的 M wave 循环中，每个 wave 移位后检查是否还在有效 M 范围内，无效的 wave 跳过 MMA 计算。

### is_tma_multicast_valid() — SM90 multicast 安全检查

```cpp
CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
    if (num_blocks_in_group == 1) return false;  // 单 tile 组 → 不能 multicast

    // Normal, Masked, KGrouped, Batched, PsumLayout → 始终可以
    if constexpr (kGemmType == GemmType::Normal or ...)
        return true;
    else {
        // MGroupedContiguous 时需检查: 两个相邻 M tile 在同一 expert?
        const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
        const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
        return group_idx == peer_group_idx;
    }
}
```

**仅在 MGroupedContiguous 时需要额外检查**：SM90 TMA multicast 要求两个 CTA 的目标 SMEM 在同一 expert 的数据空间内。若 `m_block_idx` 和 `m_block_idx^1` 跨 expert（一个在 expert 0，另一个在 expert 1），则不能 multicast。

### get_aligned_effective_m_in_block() — Psum 尾部对齐

```cpp
CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
    const uint32_t& m_block_idx) const
{
    constexpr uint32_t UMMA_STEP_N = 16;
    if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout)
        return math::align(
            m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1
                ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M,
            UMMA_STEP_N);
    return BLOCK_M;
}
```

Psum layout 场景下，最后一块的 M 可能不足 BLOCK_M，需要对有效行数做 16 对齐（UMMA 硬件要求）。
