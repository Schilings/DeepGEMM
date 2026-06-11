# GEMM+RS 调研与设计决策备忘

> **最后更新**: 2026-06-10
> **进度日志**: 见 [PROGRESS.md](./PROGRESS.md)
> **方案设计**: 见 [GEMM_RS_DESIGN.md](./GEMM_RS_DESIGN.md)

---

## 调研总结

### Flux (ByteDance) 核心发现

- **4-way Warp Specialization**: Mainloop/Epilogue/RS Fetch/RS Reduce 四个角色
- **TMA 硬件异步 fetch**: 不占 CUDA core，用 TMA 从远端拉取数据到 SMEM
- **Per-tile Flag 128B 对齐**: 避免 false sharing，system-scope 原子操作跨 GPU 可见
- **Persistent Kernel**: 持久化运行，避免 kernel launch overhead
- **两种调度模式**: Cooperative（大 tile）vs Pingpong（更好隐藏 epilogue）
- **核心差异**: Flux 是 SM90 (Hopper)，我们是 SM100 (Blackwell)
  - Hopper: WGMMA（128T warp group 驱动）
  - Blackwell: UMMA（32T 单 warp 驱动，2-CTA 协作）

### MegaMoe (DeepSeek-V4) 核心发现

- **5 类 Warp Specialization**: Dispatch/Load A/Load B/MMA/Epilogue
- **非均匀寄存器分配**: 48/40/208 regs for 不同角色
- **Expert Wave 流水线**: 细粒度通信-计算重叠
- **Dispatch 6 阶段流水**: 统计→广播→写索引→Barrier→Pull→清理
- **Min-Peeling 负载均衡**: Round-Robin 从不同 rank pull token
- **AB Swap**: Weight 作为 A 操作数（对齐 M=128），Activation 作为 B
- **L1/L2 Arrival 机制**: 计数器 + 位图，精确通知数据就绪

### Blackwell (SM100) 架构关键特性

| 特性 | 说明 | 我们的使用 |
|------|------|-----------|
| TMEM (256KB/SM) | 专用张量内存，累加器存储 | 双缓冲 UMMA 输出 |
| UMMA | 2-CTA 协作的统一 MMA | 1 warp 发射，2 CTA 计算 |
| TMA Multicast | 一次 HBM 读写入多个 CTA | A 矩阵广播到 2 SM |
| 2-CTA Cluster | 硬件级 CTA 协作 | 共享 TMEM + multicast |
| NVLink Symmetric Memory | 跨 GPU 对称地址空间 | SymBuffer::map() P2P 访问 |
| warpgroup_reg_reconfig | 运行时调整寄存器配额 | 不同角色不同寄存器数 |
| PTX ld_acq_sys/st_rel_sys | System-scope 内存一致性 | 跨 GPU per-tile flag |

### 已确认的 2-CTA Cluster 行为（不再质疑）

1. **两个 CTA 的 m_block_idx 一定不同**：标准 GEMM 通过不同 blockIdx.x 调度，相邻 CTA 获得相邻 m_block_idx、相同 n_block_idx
2. **SM100_TMA_2SM_LOAD_2D 无 multicast**：各 CTA 数据在本地 SMEM，通过 shared::cluster 允许跨 SM 访问，2SM UMMA 自动跨两个 SM 的 SMEM 读取
3. **Load 偏移规则**：`kIsMulticastOnA=false` 时 m_idx 不偏移（scheduler 已给不同 m_block），n_idx 按 block_rank 偏移一半 N 列
4. **TMEM 是 per-SM 本地的**：Epilogue 各 CTA 读自己 SM 的 128 行，写各自的输出行和 ready_flag
5. **Barrier**：leader expect_tx = (SMEM_A + SMEM_B) * kNumMulticast，因为两个 CTA 都向 leader 的 barrier 报告

### 参考文章

- [DeepSeek-V4 MegaMoE 详细分析 - 渣B zartbot](https://mp.weixin.qq.com/s/S-ej9ybT3sbFA8dqHLZafg)

---

## 设计决策备忘

### 为什么选 Pull（而非 Push）？

1. 天然适配 Tile 级 Overlap：接收端看到一个 tile 就绪就拉过来 reduce
2. SM100 TMA Load 从远端读是硬件异步的
3. Reduce 融入 kernel：Pull 回来在 SMEM 中直接 FP32 累加
4. Bandwidth-optimal：等同 NCCL ring RS 通信量

### 为什么 12 warps (384 threads)？

- 4 个 Comm warps (128T) 提供更高 P2P read 并行度
- 寄存器预算充裕 (37888/64512 = 59%)

### 为什么 Per-Rank Pipelined Reduce？

1. 延迟隐藏：早到的 rank 数据立即开始 reduce
2. 匹配 M-Swizzle：ring order 确保最可能先 ready 的 rank 被最先处理
3. 简化同步：每次只等一个 flag

### 性能瓶颈分析

| 场景 | 瓶颈 | 解决思路 |
|------|------|---------|
| 大 M + 小 K | 通信量大，GEMM 快 | 更多 comm warps / TMA pull |
| 小 M + 大 K | GEMM 慢，通信少 | 完全 overlap，comm 几乎免费 |
| 大 N (7168) | 通信和计算都大 | tile 级流水最有效 |
| 多 rank (8+) | 每个 rank 要 pull 7 次 | ring 多步流水 |

---

## 已知风险

1. Per-tile flag 跨 NVLink 延迟 — 如果 ld_acq_sys 自旋成本高，考虑批量 flag
2. 384 线程 = 12 warps — SM 占用率约 1 block/SM，需 profiling 确认
3. M-Swizzle 调度 — 所有 CTA 同时写同一个远端 rank 的 flag 可能造成热点
4. Comm Warps 用 global load 做 P2P read — 未来应改为 TMA（性能关键优化）
