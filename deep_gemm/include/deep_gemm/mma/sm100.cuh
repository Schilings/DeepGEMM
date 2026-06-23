#pragma once

#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>

/*
═══════════════════════════════════════════════════════════════════════
SM100 (Blackwell) UMMA 描述符与指令构造工具

本文件实现 SM100 block-scaled UMMA 所需的全部描述符构造和操作函数:
  1. make_smem_desc  — 构造 SmemDescriptor (告知硬件 SMEM 布局)
  2. make_sf_desc    — 构造 SF descriptor (UTCCP 专用)
  3. make_umma_desc  — 构造 TMA↔UMMA swizzle 匹配的完整 SmemDescriptor
  4. to_umma_layout_type — swizzle 字节大小 → UMMA layout 枚举
  5. advance_umma_desc_lo — swizzle 感知的 SMEM 地址推进
  6. make_runtime_instr_desc_with_sf_id — 将编译期指令描述符 + 运行时 SF ID 合成 64 位立即数

核心设计理念:
  - TMA(生产者)和 UMMA(消费者)通过同一个 kSwizzleMode 联动
  - SmemDescriptor 将 swizzle 物理布局编码为硬件可自动解析的格式
  - 所有字段以 16B 为单位 (>>4), 因为 SMEM 地址是 16B 对齐的
═══════════════════════════════════════════════════════════════════════
*/

namespace deep_gemm::mma::sm100 {

/*
╔══════════════════════════════════════════════════════════════════════╗
║  1. make_smem_desc — SmemDescriptor 底层构造                         ║
╚══════════════════════════════════════════════════════════════════════╝

SmemDescriptor 是 SM100 UMMA 硬件读取 SMEM 的唯一依据, 描述数据在
shared memory 中的物理布局。硬件在 fma 时根据此描述符自动完成:
  (逻辑行列) → (物理地址) 的映射, 包括反解 swizzle。

字段说明:
  - version_     = 1 (SM100 固定)
  - lbo_mode_    = 0 (legacy mode, 不使用 NV MMA 的新 LBO 编码方式)
  - layout_type_ = SWIZZLE_NONE / SWIZZLE_32B / ... / SWIZZLE_128B
    告诉硬件 SMEM 是否经过 swizzle、以多少字节为 stripe
  - start_address_ = SMEM 起始物理地址 >> 4 (必须以 16B 对齐)
  - base_offset_ = 0 (子块内偏移, 通常为 0, 原子边界已对齐)
  - stride_byte_offset_ (SBO): 相邻 atom 在 SMEM 中的字节跨步 >> 4
  - leading_byte_offset_ (LBO): 行/列间的字节跨步 >> 4

SBO/LBO 含义取决于 layout 方向:
  - K-major (swizzle 在 K 维): {SBO, LBO} = {MN方向atom间距, K方向atom间距}
  - MN-major (swizzle 在 MN 维): {SBO, LBO} = {K方向atom间距, MN方向atom间距}
  - kSwizzleMode==16 例外: 非 swizzle 而是 interleave, 交换 SBO/LBO
*/
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    cute::UMMA::SmemDescriptor desc;

    // SM100 固定版本号
    desc.version_ = 1;

    // Legacy LBO 编码模式 (非 NV MMA 新标准)
    desc.lbo_mode_ = 0;

    // swizzle 类型: SM100 硬件由此知道按 32B/64B/128B 哪个粒度反解 swizzle
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // SMEM 起始物理地址 (16B 为单位, 低 4 位始终为 0)
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // 子块内无偏移 — 调用方保证地址已对齐到 atom 边界
    desc.base_offset_ = 0;

    // SBO 和 LBO 同样以 16B 为单位编码
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  2. make_sf_desc — SF(scale factor) 描述符 (UTCCP 专用)             ║
╚══════════════════════════════════════════════════════════════════════╝

UTCCP (Unified Tensor Core Copy Pipeline) 用于将 UE8M0 scale factor
从 SMEM 拷贝到 TMEM 的 SF 列区, 是 block-scaled FP8 MMA 的必要前置步骤。

SF 无需 swizzle (SWIZZLE_NONE): UE8M0 数据是 uint4 打包格式, 4 个
scale factor 组成一个 uint32, 在 SMEM 中行优先连续存储。

Atom 规格: 8 行 × 128 bits = 8×16 bytes = 128 bytes
  - SBO = 8 × 16 = 128 bytes (MN 方向 atom 间距)
  - LBO = 0 (K 方向只有 1 个 atom, 没有第二个 atom 需要跨步)
*/
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // UTCCP 布局默认为 K-major
    // Atom: 8 × 128 bits (每 atom 8 行, 宽度 128 bits = 16 bytes)
    // {SBO, LBO} = {MN 方向 atom 跨步, K 方向 atom 跨步}
    // UTCCP 宽度 128b → K 方向只有 1 个 atom → LBO = 0
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  3. replace_smem_desc_addr — 原地替换描述符中的 SMEM 起始地址        ║
╚══════════════════════════════════════════════════════════════════════╝

用于 UTCCP 循环中复用同一个 sf_desc, 只需改 start_address 指向不同的
SMEM 行, 其余字段 (layout_type, SBO, LBO) 保持不变。
*/
CUTLASS_DEVICE
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  4. get_atom_base — 获取 layout 的 base (16 或 32)                  ║
╚══════════════════════════════════════════════════════════════════════╝

SWIZZLE_128B_BASE32B: 以 32B 为基本单元做 swizzle 排列, 其余以 16B 为单元.
用于计算 num_non_contiguous = 128 / base → 也就是 swizzle 块内有多少个
"不连续段" (对于 BASE32B 是 4 段, BASE16B 是 8 段).
*/
CUTLASS_DEVICE
static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type) {
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  5. to_umma_layout_type — swizzle 字节大小 → UMMA layout 枚举       ║
╚══════════════════════════════════════════════════════════════════════╝

将 TMA 的 swizzle 字节大小 (0/16/32/64/128) 映射为 SM100 UMMA 硬件
可识别的 LayoutType 枚举值。这是 TMA↔UMMA 协议匹配的关键环节。

特殊规则:
  - float + MN-major: 必须用 SWIZZLE_128B_BASE32B (FP32 4 字节对齐特殊处理)
  - kUseBase32: 强制使用 BASE32B (BF16 某些 layout 场景)
  - kSwizzleMode==0 或 ==16: 映射为 SWIZZLE_NONE (无 swizzle 或仅 interleave)
  - kSwizzleMode==16 的 interleave: 不是地址打乱, 而是行列交替存储,
    make_umma_desc 中通过交换 SBO/LBO 处理
*/
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");

    // FP32 MN-major 或显式 kUseBase32 → SWIZZLE_128B_BASE32B
    // 原因: FP32 4B 元素在 32B 基下整对齐, swizzle 反解更高效
    if constexpr ((cute::is_same_v<dtype_t, float> and kMajorMode == cute::UMMA::Major::MN) or kUseBase32) {
        DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
        return cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
    }

    // 标准映射表
    // ┌──────────────┬───────────────────────┐
    // │ kSwizzleMode │   LayoutType          │
    // ├──────────────┼───────────────────────┤
    // │  0, 16       │   SWIZZLE_NONE        │
    // │  32          │   SWIZZLE_32B          │
    // │  64          │   SWIZZLE_64B          │
    // │  128         │   SWIZZLE_128B         │
    // └──────────────┴───────────────────────┘
    if constexpr (kSwizzleMode == 0)   return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return cute::UMMA::LayoutType::SWIZZLE_128B;
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  6. get_umma_desc_stride_k — K 方向推进步长                         ║
╚══════════════════════════════════════════════════════════════════════╝

K-major:
  stride_k = 1 → K 维逐元素连续, advace 时直接 k_idx * 1

MN-major:
  stride_k = BLOCK_INNER_ATOM → K 维需要跳跃整个 swizzle atom
  例: swizzle=128B, dtype=BF16(2B) → BLOCK_INNER_ATOM=64
      推进 k_idx*64 个元素才能跳到下一个 atom 内对应的 K 位置

原因: MN-major 时, 物理 SMEM 先放所有行的 atom[0], 再放所有行的 atom[1],
      所以沿 K 方向前进不是逐元素, 而是逐 atom 跳跃.
*/
template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_umma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K
        ? 1  // K-major: K 方向逐元素连续
        : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();  // MN-major: K 方向跨一个 atom
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  7. advance_umma_desc_lo — swizzle 感知的 SMEM 地址推进             ║
╚══════════════════════════════════════════════════════════════════════╝

推进 desc.lo (SmemDescriptor 低 32 位) 中的 start_address 字段,
公式:
    new_lo = base_lo + ((offset + k_idx * stride_k) * sizeof(dtype)) >> 4

    desc.lo 位布局 (32 bits):
    ┌──────────┬─────────┬──────────────────┐
    │  [31:23] │ [22:16] │     [15:0]       │
    │base_offset│layout  │   start_address   │  ← 增加此字段来推进地址
    └──────────┴─────────┴──────────────────┘

参数:
  - base:   desc.lo 基线值 (某个 stage 的 SMEM 起始地址)
  - offset:  额外偏移量 (如 A 矩阵 MN-major 跨 M atom 的大跳)
  - k_idx:   K 维度元素索引, 乘以 stride_k 得到 K 方向偏移

  >>4: 因为 start_address 以 16B 为单位

典型调用:
  - B 矩阵 (K-major):  offset=0,  k_idx=k*UMMA_K
  - A 矩阵 (MN-major): offset=w*WAVE_BLOCK_M*BLOCK_K, k_idx=k*UMMA_K
    offset 负责跨 M 段跳跃 (整个 SMEM 的 M 维度换段),
    k_idx 负责在 atom 内沿 K 方向微调
*/
template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {
    return base + (((offset + k_idx * get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>())
                    * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  8. make_umma_desc — 构造完整的 TMA↔UMMA swizzle 匹配描述符          ║
╚══════════════════════════════════════════════════════════════════════╝

这是整个文件最核心的函数。它构造一个 SmemDescriptor, 告诉 SM100 UMMA
硬件如何从 swizzled SMEM 中正确读取数据。

核心职责:
  1. 将 kSwizzleMode 映射为 UMMA LayoutType (调用 to_umma_layout_type)
  2. 计算 num_non_contiguous = 128 / base → swizzle 块内不连续段数
  3. 根据 K-major 还是 MN-major, 设置正确的 {SBO, LBO} 和起始地址

模板参数:
  - kMajorMode:  Major::K (K 连续) 或 Major::MN (M/N 连续)
  - BLOCK_MN:    M 或 N 维度 block 大小 (LOAD_BLOCK_M / LOAD_BLOCK_N)
  - BLOCK_K:     K 维度 block 大小 (固定 128 for FP8)
  - kSwizzleMode: swizzle 字节大小 (0/16/32/64/128)
  - kUseBase32:  是否使用 32B 基 (仅特殊 layout 场景)
  - dtype_t:     数据类型 (FP8/BF16/FP32)

函数参数:
  - base_smem_ptr:  stage 0 的 SMEM 起始地址
  - mn_idx:         M/N 维度起始索引 (通常为 0)
  - k_idx:          K 维度起始索引 (通常为 0)
*/
template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    const uint32_t stride_k = get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    // num_non_contiguous = swizzle-128B 区域内不连续段数
    // BASE16B → 128/16=8 段, BASE32B → 128/32=4 段

    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        /*
        ───────────────────────────────────────────────────────────
        K-major 分支: K 方向连续, swizzle 在 MN 方向
        ───────────────────────────────────────────────────────────

        SMEM 物理布局 (K-major, LOAD_BLOCK_M=128, BLOCK_K=128, swizzle=128B):
        ┌────────────────────────┐ ← smem 起始
        │ K=0..127 的全部 M 行    │ 这是 1 个 swizzle atom
        │ (128行 × 128列)        │ (BLOCK_K × sizeof = 128×1B for FP8, swizzle=128B刚好1 atom)
        └────────────────────────┘

        由于 swizzle 在 MN 方向:
          - SBO = num_non_contiguous × BLOCK_K × sizeof(dtype): MN 方向 atom 间距
          - LBO = 0: K 方向只有 1 个 atom, 不需要跨步

        约束: kSwizzleMode 必须等于 BLOCK_K × sizeof(dtype_t)
              这样 swizzle 块恰好与 K 维宽度匹配, K 方向恰好 1 个 atom
        */
        DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // K 方向: stride_k=1 (K 逐元素连续)
        // MN 方向: 每个 MN 行由 BLOCK_K 个元素组成
        // 起始地址: base + mn_idx 行偏移 (BLOCK_K 元素/行) + k_idx 元素偏移
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t);
        const uint32_t leading_byte_offset = 0;
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    } else {
        /*
        ───────────────────────────────────────────────────────────
        MN-major 分支: M/N 方向连续, swizzle 在 K 方向
        ───────────────────────────────────────────────────────────

        SMEM 物理布局 (MN-major, LOAD_BLOCK_M=128, BLOCK_K=128, swizzle=128B):
        ┌──────────────────────┐ ← smem 起始
        │ atom 0: 128行 × 64列  │ (所有行的前 64 列, swizzled)
        └──────────────────────┘
        ┌──────────────────────┐ ← smem + 128×64
        │ atom 1: 128行 × 64列  │ (所有行的后 64 列, swizzled)
        └──────────────────────┘

        BLOCK_MN_ATOM = swizzle_mode / sizeof(dtype): 一个 atom 的列数
        例: swizzle=128B, FP8(1B) → atom=128 列, 刚好 = BLOCK_K
              但 swizzle=128B, BF16(2B) → atom=64 列, BLOCK_K=128 需 2 个 atom

        swizzle 在 K 方向:
          - SBO = num_non_contiguous × BLOCK_MN_ATOM × sizeof(dtype): K 方向 atom 间距
          - LBO = BLOCK_K × BLOCK_MN_ATOM × sizeof(dtype): MN 方向 atom 间距
            (跳一行/列需要跨过当前 K 段的所有 MN atom)

        kSwizzleMode==16 例外: 非 swizzle 而是 interleave
          → 交换 SBO 和 LBO (行列交替而非地址打乱)
        */
        constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // mn_idx 必须对齐到 atom 边界 (运行时 assert, 编译器保证传入常量)
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);  // interleave 模式交换方向
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    }
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  9. make_runtime_instr_desc_with_sf_id — 运行时合成 UMMA 指令描述符 ║
╚══════════════════════════════════════════════════════════════════════╝

将编译期构造的 InstrDescriptorBlockScaled (包含 UMMA_M/N/K, layout, dtype)
与运行时确定的 sfa_id/sfb_id 合并, 生成实际的 64 位立即数。

UMMA 指令描述符是 64 位立即数, 其中:
  - 高 32 位 → 指令操作码 + shape (由 InstrDescriptorBlockScaled 的 uint32 转换)
  - 低 32 位 → TMEM 的 SF 列 ID (运行时 sfa_id/sfb_id)

返回值为 64 位, 其中高 32 位 = 编译期描述符, 低 32 位留空供后续填充。
实际上 sfa_id/sfb_id 先写入 desc, 然后取低 32 位作为 TMEM 地址的一部分。
*/
CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    // 将运行时 SF ID 填入编译期描述符
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    // 取 32 位描述符作为高 32 位 (<<32), 低 32 位在指令发射时处理
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

/*
╔══════════════════════════════════════════════════════════════════════╗
║  10. update_instr_desc_with_umma_n — 动态修改 UMMA N 维度            ║
╚══════════════════════════════════════════════════════════════════════╝

用于 psum/runtime shape 等场景, 需要在运行时修改 UMMA 指令的 N 维度.
n_dim_ 以 8 列为单位编码 (移位右 3 位), 因为 UMMA 硬件最低 8 列对齐.

重载了两个版本:
  - BlockScaled: FP8 带 scale factor 的 UMMA 指令
  - 普通:       BF16/TF32 等标准 UMMA 指令
*/
CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptorBlockScaled& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;  // N 维以 8 列编码
}

CUTLASS_DEVICE void update_instr_desc_with_umma_n(
    cute::UMMA::InstrDescriptor& desc, const uint32_t& umma_n) {
    desc.n_dim_ = umma_n >> 3;
}

} // namespace deep_gemm::mma::sm100
