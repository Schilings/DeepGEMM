# 算子开发进度总览（索引）

> 最后更新：2026-07-02
> 用途：**新会话据此挑选/确认目标算子**，再读该算子的 `*_DESIGN` / `*_ITERATION` 文档接班继续开发。
> 规则见 `docs/RULE.md`；接班流程见 `docs/SESSION_MEMORY.md`。

---

## 各算子状态一览

| 算子 | 入口符号 | 当前最佳（focus 中大 / geo vs sep） | 正确性 | 分支 / tag | 详细文档 |
|------|---------|------|------|------|------|
| **GEMM-RS** | `bf16_gemm_rs_nt` | focus vs sep：8卡 **1.20x** / 4卡 **1.20x**；vs torch-native：8卡 1.16x / 4卡 1.18x；全13-shape geo **1.14x** | 6/6 PASS @ {2,4,8}（test 已加 torch 原生 gemm+RS 交叉对照）| `main`（2D TMA push）；1D 版存档 tag `gemm-rs-1d-stable` | `GEMM_RS_DESIGN.md` / `GEMM_RS_ITERATION.md` / `FLUX_GEMM_RS_STUDY.md` |
| **A2A-transpose-GEMM**（Ulysses post-attn，正确版）| `bf16_a2a_transpose_gemm_nt`=**默认 M0 串行** / `_fused`=M1 overlap(opt-in)；`seq_major=True` 接 FA 原生 BSHD/THD | **M0 单节点最优**：比 torch(NCCL+2转置)基线快 **1.6~2.4×**，comm 快 3~4×（只转 1 次、融进 comm）；M1 overlap 单节点 **~parity（净负）**；`seq_major` 是 layout 匹配(非加速)，**直接覆盖 THD/varlen（uniform 切，comm 不需要 cu_seqlens）** | **{2,4,8} 4/4 PASS**（M0/fused/seq_major 三路）；`test_ulysses_attn_flow` {8}3/3；`test_ulysses_varlen_thd`(FA varlen){8} PASS | **已并入 `main`**（merge `5516d8d`） | `A2A_TRANSPOSE_GEMM_DESIGN.md`（flux 双stream/基线口径/资源/seq_major/THD 全记录）|
| **GEMM-A2A-transpose**（Ulysses **pre-attn**，GEMM+A2A）| `bf16_gemm_a2a_transpose_nt` | 8卡 geo vs sep **1.42x** / vs torch-native **1.54x**；4卡 vs sep **1.44x** / vs torch **1.55x**；fused 平均 ~1190 TFLOPS | **{2,4,8} 全 PASS**，max_diff/rel/consist **恒 0.0**（纯排列，逐元素等于精确参考；torch `matmul+all_to_all` 亦 0.0）| `main`（单 kernel，抄 GEMM-RS 改 N 切分+删 reduce+转置散射）| `GEMM_A2A_TRANSPOSE_DESIGN.md` / `GEMM_A2A_TRANSPOSE_ITERATION.md` |
| **A2A-GEMM**（旧 token-A2A，**语义错误**）| `bf16_a2a_gemm_nt` | — | ⚠️ **3/6 FAIL** + 语义错位（token(M)-A2A，非 Ulysses）→ 已被上面的 transpose 版取代 | `main`（保留未删）| `A2A_GEMM_ITERATION.md` / `A2A_GEMM_DESIGN.md`（旧）|
| **AG-GEMM** | `bf16_ag_gemm_nt` | geo **~1.13x**（8 卡）| PASS | `main`（仅 PDL 默认开启被保留）| `AG_GEMM_ITERATION.md` / `AG_GEMM_FLUX_REFERENCE.md` |
| **Fused QKV+Norm+A2A** | `bf16_fused_qkv_norm_a2a_transpose_nt` | **单 kernel CUDA**：geo vs serial **1.61x**（8卡）；norm 可选 + GQA-aware | 8卡 **20/20 PASS**（MHA/GQA × norm on/off）| `feat/fused-qkv-norm-a2a` | `FUSED_QKV_NORM_A2A_DESIGN.md` / `FUSED_QKV_NORM_A2A_ITERATION.md` |

---

## Unified Symm Buffer（2026-07-02 新增）

**一个 sym buffer 服务所有通信融合算子**——消除每层每步 `symm_mem.rendezvous` 的毫秒级开销。

| 算子 | 需要 attention 参数？ | 说明 |
|------|---------------------|------|
| GEMM-RS | ❌ 不需要 | 通用 linear+RS，用于 MLP/FFN/classifier 等任意 linear |
| AG-GEMM | ❌ 不需要 | 通用 AG+linear，GEMM-RS 的 backward 对偶 |
| GEMM-A2A-transpose | ✅ 需要 q_nheads/head_dim | pre-attn QKV scatter |
| A2A-transpose-GEMM | ✅ 需要 nheads/head_dim | post-attn Wo gather |
| Fused-QKV-Norm-A2A | ✅ 需要 q/kv_nheads/head_dim | pre-attn with RMSNorm |

```python
# Attention 场景
sym = get_unified_symm_buffer(group, bs, seq, hidden,
                               q_nheads=32, kv_nheads=32, head_dim=128)

# 非 attention 场景（MLP+RS / AG+GEMM）
sym = get_unified_symm_buffer(group, bs, seq, hidden)
```

- **正确性**：2卡 7/7 PASS，8卡 7/7 PASS（含无 attention 参数的 Test 6）
- **修复**：AG-GEMM local_x 大小从 `1*M*H` 改为 `num_ranks*M*H`（与 C++ BF16AGGemmWorkspace 对齐）
- **修复**：Test 3/6 GEMM-RS 维度（total_m=bs*seq, tokens_per_rank=bs*local_seq）
- 分支 `feat/fused-qkv-norm-a2a`

### 2026-07-17：wan21 例子迁移到 unified buffer

- `examples/wan21` 的 `GemmRSFunction` + `variant.py` 已从旧两独立 buffer 迁到单个 `get_unified_symm_buffer`：fwd GEMM+RS、bwd AG+GEMM 共用同一块物理 buffer，跨层经 `share_buffers_from` 共享。
- 动机：用户要求「全局所有层、一个固定大小 buffer、fwd/bwd 都复用，不增加多余开销和显存」——这正是 `UnifiedSymmBuffer` 的设计。
- 实现细节见 `docs/SYM_BUF_SHARING_ANALYSIS.md` 2026-07-17 节。

### 2026-07-20：fused 策略实现 + UnifiedSymmBuffer 统一 + 文件重命名

- **文件重命名**：`fused_standard.py` → `fused.py`（`FusedUlysses`），`fused_variant.py` → `variant.py`（`FusedVariantUlysses`）。
- **`FusedUlysses`（POST 融合）**：POST 使用 `bf16_a2a_transpose_gemm_nt`（A2A+GEMM），PRE 暂时继承 serial baseline（QK RMSNorm 必须在 A2A 前做，`bf16_fused_qkv_norm_a2a_nt` 融合 PRE 是 WIP）。正确性验证：`fwd rel=0.002706, grad_X rel=0.000166, grad_Wo rel=0.000000`。
- **`UnifiedSymmBuffer` 统一**：fused 和 variant 共用同一种 buffer 类型。新增 `x`/`gathered` 属性（A2A-transpose-GEMM 兼容）、`reset()`/`get_out_view()`/`get_rms_view()` 别名（Fused-QKV-Norm-A2A 兼容）。`x`/`gathered` 在 `_has_attn=False` 时 raise `AttributeError`，避免 AG-GEMM 误用。
- **`ag_gemm` 修复**：`bf16_ag_gemm_nt_with_input` 优先用 `ag_x`/`ag_slots_x`（而非 `x`/`slots_x`），避免与 `UnifiedSymmBuffer` 的 attention `x` 属性冲突。
- **`FusedPreQKVFunction`（WIP）**：forward 用 `bf16_fused_qkv_norm_a2a_nt`，已验证 forward 正确（Q/K/V rel=0.014 vs serial，seq 排列经 `view(sp,local_seq).transpose(1,2)` permute 后与 serial 一致）。backward（inverse A2A + RMSNorm backward + GEMM）有 bug 导致 grad 误差大（grad_X rel=1.38），需进一步调试 inverse A2A 的排列和 norm backward 的重算逻辑。PRE 暂回退 serial baseline。
- **FA4 精度限制**：发现 BF16 FA4 对 V 的系统性 BF16 差异（rel=0.014）放大到 attention 输出 rel=0.26，但对随机 V 噪声（rel=0.014）只产生 rel=0.005。这是 BF16 数值精度限制，不是 fused kernel bug。
- **所有测试通过**：`test_unified_buffer` 7/7、`test_gemm_rs` 6/6、`test_a2a_transpose_gemm` 4/4。

---

## Standard Ulysses baseline vs fused（2026-07-20）

> 权威结果见 `examples/ulysses_fused/ULYSSES_FULL_ATTN_BENCH.md`。

`examples/ulysses_fused/` 已清理为标准 Ulysses forward 的严格两臂：torch matmul + 同步 NCCL A2A baseline，对比 DeepGEMM fused GEMM+A2A PRE 和 A2A+GEMM POST。B300×8、10 iters、rank-max：BSHD chain/PRE+POST 几何平均 1.032x/1.111x，THD 为 1.026x/1.098x；正确性 3/3 PASS。

---

## Wan2.1 14B Ulysses POST 变体（2026-07-20 更新）

> 当前权威结果见 `examples/ulysses_variant/WAN21_ULYSSES_BENCH.md`，代码在 `examples/ulysses_variant/`。通用 profiling 方法见 `docs/GPU_PROFILING_GUIDE.md`。

当前采用严格两臂 POST-only 消融：PRE/RoPE/FA4 完全共用，baseline 为同步 A2A+完整 Wo，variant 为 Wo 输入列分片 GEMM-RS/AG-GEMM。旧三策略 benchmark 和沿 SP group 做 FSDP2 的结果不再作为当前显存或真实训练吞吐结论。

B300×8 当前结论：

- 40 层 attention stack + FP32 Adam：8K 峰值显存省 9,913.9MB（20.6%），32K 省 9,629.9MB（14.0%）；
- 官方 14B 权重、40 个完整 Transformer block、8K、DDP overlap：serial 29,249.9 tokens/s，variant 28,193.0 tokens/s（-3.61%）；
- SP=8 所有 rank `grad_X rel=0`；
- Nsight 已确认 POST BWD 慢点是 AG 8×远端 payload和 kernel 内等待，不是 torch GEMM。

---

> 每个算子的「最新当前状态 / 接班信息」见对应 `*_ITERATION.md` 顶部「当前状态」节，
> 本表只做一行总览，避免与各算子文档重复维护细节。

---

## benchmark 统一口径（所有通信类融合算子通用）

- `separate` 基线 = 标准 GEMM + 对应通信原语（如 RS：`bf16_gemm_nt + reduce_scatter_tensor`）。
- `fused` = 该算子的融合入口（如 `bf16_gemm_rs_nt`）。
- 迭代时必须先保证 correctness 不退化，再比较 `fused` vs `separate` 的增益。
- Megatron SP 导向：优先看中大 shape 分组（`N=7168` / `K=7168` / `M/rank>=2048`）的几何均值与短板 shape。
- benchmark 脚本通用开关：
  - `DG_BENCH_FOCUS_ONLY=1`（只跑重点中大 shape 子集）
  - `DG_BENCH_SHAPES="M,N,K;..."` / `DG_BENCH_SINGLE_SHAPE=M,N,K`（显式 shape）
  - `DG_JIT_USE_NVRTC=1`（无 nvcc 时）

---

## 接班即用命令（以 GEMM-RS 为例，其它算子替换脚本名）

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/comm/test_gemm_rs.py 8

DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 8 2
```
