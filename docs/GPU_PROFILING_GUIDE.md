# DeepGEMM 通用 GPU Profiling 指南

本文是仓库级通用性能分析 SOP，适用于：

- 单卡或多卡 GEMM；
- 通信融合算子；
- NCCL collectives；
- forward、backward、optimizer step；
- PyTorch autograd、DDP/FSDP；
- 完整模型训练和独立算子；
- CUDA 多 stream、Copy Engine、NVLink/P2P；
- DeepGEMM JIT kernel。

Ulysses POST backward 仅作为文末案例，不是本文唯一适用对象。

## 1. Profiling 的目标

开始前先把问题写成可证伪的假设，例如：

- kernel 本身算得慢；
- 通信量过大；
- 通信与计算没有重叠；
- CTA 在等待远端 signal；
- DDP 通信暴露在关键路径；
- layout conversion 或临时 tensor 很贵；
- CPU launch、barrier 或 device synchronize 造成气泡；
- 某个优化只改善 microbench，没有改善完整训练。

最终报告应至少回答：

1. 正确性是否保持；
2. 无 profiler 的真实延迟/吞吐是多少；
3. 时间由哪些 kernel、copy、collective 和 barrier 构成；
4. 哪部分在关键路径，哪部分被 overlap；
5. microbench 差异是否转化成 end-to-end 收益；
6. 结果是否可重复。

## 2. 推荐的五层测量法

不要直接从完整模型跳到单个 kernel。按下面顺序逐层收敛。

| 层级 | 测什么 | 典型工具 | 回答的问题 |
|---|---|---|---|
| L0 正确性 | 输出、梯度、跨 rank 一致性 | reference、relative error | 优化是否数学正确 |
| L1 组件 | GEMM、copy、collective、transpose、barrier | CUDA Event | 哪个组件最慢 |
| L2 生产 Function | 真实 forward/autograd backward | CUDA Event + rank MAX | 测试替身是否偏离生产路径 |
| L3 子图/模块 | 一层、一个 block、一个 pipeline stage | CUDA Event + wall-clock | 组件差异如何进入依赖链 |
| L4 完整训练 | 多层真实权重、DDP/FSDP、optimizer | rank-max wall-clock | 最终吞吐和显存 |

Nsight Systems 用于给 L1～L4 做时间线归因，不应替代无 profiler 的正式延迟。

## 3. 最重要的实验纪律

### 3.1 禁止并发争用同一批 GPU

两个 benchmark 同时使用相同 GPU 会竞争：

- SM/Tensor Core；
- HBM/L2；
- NVLink/PCIe；
- Copy Engine；
- NCCL stream；
- 功耗和频率预算。

结果可能产生数倍波动。不同 shape、策略和同步模式必须顺序运行。

运行前检查：

```bash
nvidia-smi
```

如果脚本使用全部 8 张卡，不要同时发起另一个 8 卡测试。自动化工具调用也不能把两个 GPU benchmark 并行提交。

### 3.2 先无 profiler 测量，再打开 profiler

推荐顺序：

1. warmup；
2. 无 profiler 重复多轮；
3. 记录均值、范围和异常值；
4. 用少量 iteration 采集 Nsight trace；
5. 修改实现；
6. 再次无 profiler 复测；
7. 回到完整模型验证。

Nsight、Kineto 和 NCU 都会引入开销。profile 下的绝对时间不能直接作为最终吞吐。

### 3.3 多卡取最慢 rank

分布式迭代由最慢 rank 决定。不要只报告 rank 0。

通用做法：

```python
elapsed = torch.tensor(local_elapsed_ms, device="cuda")
torch.distributed.all_reduce(
    elapsed,
    op=torch.distributed.ReduceOp.MAX,
    group=group,
)
rank_max_ms = elapsed.item()
```

每个 measured iteration 都取 rank MAX，再计算平均值。

### 3.4 至少重复三轮

NCCL、JIT cache、GPU 频率、allocator 和后台任务都会造成波动。建议：

- microbench：warmup 5～10，measured 30～100；
- 完整模型：warmup 2～3，measured 5～10；
- 整个命令独立重复至少三轮；
- 如果对比两臂，反转执行顺序再测一次。

不要只选择最快一次。

## 4. 环境准备

从仓库根目录：

```bash
cd /root/.local/codebuddy/DeepGEMM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
nsys --version
```

DeepGEMM 常用环境：

```bash
export DG_JIT_USE_NVRTC=1
export PYTHONPATH=$PWD/examples:$PWD
```

如果 `deep_gemm._C` 缺失：

```bash
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force
```

如果目标包含 FA4，按 `docs/INSTALL_FA4.md` 安装固定版本。

Nsight Compute 检查：

```bash
ncu --version
```

缺失时参考 `docs/install_ncu.sh`。通常先用 Nsight Systems 定位，再对单个 kernel 使用 NCU。

## 5. 正确的计时方式

### 5.1 单 stream 或已建立依赖的组件：CUDA Event

```python
for _ in range(warmup):
    fn()
torch.cuda.synchronize()

total_ms = 0.0
for _ in range(iters):
    torch.distributed.barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end)
    # 对 local_ms 做 rank MAX
    total_ms += rank_max_ms
avg_ms = total_ms / iters
```

注意：Event 只天然覆盖其所在 stream；如果 side stream 没有通过 event/wait 回连 current stream，Event 可能漏掉尾部工作。

### 5.2 多 stream/end-to-end：同步后的 wall-clock

```python
import time

torch.cuda.synchronize()
torch.distributed.barrier(group)
start = time.perf_counter()

fn()

torch.cuda.synchronize()
local_ms = (time.perf_counter() - start) * 1000.0
# 对 local_ms 做 rank MAX
```

完整训练吞吐优先使用这个口径。它覆盖：

- current stream；
- comm side stream；
- DDP reducer stream；
- Copy Engine 尾部；
- host barrier 和 launch stall。

建议同时打印 CUDA Event 和 wall-clock。如果差异明显，优先排查 side-stream 尾部或 CPU stall。

### 5.3 只测 backward

先构造真实 forward graph，再同步，然后计时 backward：

```python
output, inputs = build_real_graph()
torch.cuda.synchronize()
torch.distributed.barrier(group)

start.record()
torch.autograd.grad(output, inputs, grad_output)
end.record()
torch.cuda.synchronize()
```

不要用一个手写矩阵乘替代生产 autograd Function 后就宣称 backward 已验证。理想做法是同时报告：

- 手拆组件和；
- 真实 production Function 的 autograd backward。

### 5.4 吞吐计算

明确分子是什么：

- global tokens/s；
- samples/s；
- local tokens/s；
- TFLOP/s；
- bytes/s。

例如全局 sequence tokens/s：

\[
\text{tokens/s}=\frac{\text{global sequence tokens}}{\text{rank-max wall seconds}}
\]

不要把每 rank token 数和 global time 混用。

## 6. Benchmark 应如何拆组件

对于任意融合算子，至少准备三条路径：

1. baseline 真实路径；
2. fused 真实路径；
3. 与 fused 相同 shape 的纯计算对照。

例如通信融合 GEMM 可以拆成：

- baseline GEMM；
- baseline collective；
- fused kernel；
- 纯 GEMM same-shape；
- input/output transpose；
- staging copy；
- barrier；
- weight-gradient GEMM。

这样可以区分：

- 是 GEMM shape 不友好；
- 还是通信慢；
- 还是融合调度慢；
- 还是额外 layout conversion 慢。

### 6.1 通信量必须先算清楚

对每个 collective 记录：

- 每 rank logical input/output bytes；
- 单向远端 payload；
- 总 send/receive bytes；
- 是否包含 self chunk；
- 通信 dtype；
- world size；
- 算法（ring/tree/all-to-all/pull/push）。

不要用不同口径比较两个算子，例如一边算 total tensor、一边只算 remote payload。

## 7. 添加 NVTX ranges

通用写法：

```python
torch.cuda.nvtx.range_push("my_component")
fn()
torch.cuda.nvtx.range_pop()
```

命名建议：

```text
phase/component/variant/shape
```

例如：

```text
bwd/dx_gemm/baseline/8k
bwd/all_gather_gemm/fused/8k
sync/ddp_bucket_3
optimizer/adam_step
```

每个 range 应覆盖逻辑阶段，而不是整个脚本。对 side stream 的工作，NVTX GPU projection 可能无法完整归入 launch stream 的 range，因此还要结合 CUDA trace。

## 8. Nsight Systems 通用采集

### 8.1 通用命令模板

```bash
rm -f my_profile.nsys-rep my_profile.sqlite

DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --force-overwrite=true \
  --output=my_profile \
  python3 path/to/benchmark.py <args>
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--trace=cuda,nvtx,osrt,cublas` | CUDA、NVTX、OS runtime、cuBLAS |
| `--sample=none` | 关闭 CPU sampling，减小开销 |
| `--cpuctxsw=none` | 关闭 CPU context-switch tracing |
| `--wait=all` | 等待多进程子进程结束 |
| `--force-overwrite=true` | 覆盖同名报告 |

`torch.multiprocessing.spawn` 场景下，Nsight Systems 默认按 process tree 跟踪子进程。

如果目标代码内部使用 Kineto/`torch.profiler`，不要与 Nsight 嵌套。仓库中调用 `deep_gemm.testing.bench_kineto` 的测试可设置：

```bash
export DG_USE_NVIDIA_TOOLS=1
```

使其跳过内部 Kineto profiling。

### 8.2 trace iteration 要少

Nsight trace 通常只需要：

- warmup 1～2；
- measured 2～5。

大量 iteration 会生成巨大的 `.nsys-rep` 和 SQLite 文件，并显著增加运行时间。

## 9. Nsight Stats 常用报告

先查看当前版本支持的报告：

```bash
nsys stats --help-reports
```

### 9.1 NVTX GPU 投影

```bash
nsys stats \
  --report nvtx_gpu_proj_sum \
  --format csv \
  --force-export=true \
  my_profile.nsys-rep
```

默认时间单位通常是纳秒。`Avg / 1000` 才是微秒。

多进程时，`Instances` 是所有被 trace 进程的合计。例如 8 ranks × 5 calls 通常得到 40 instances。

### 9.2 NVTX range 内的 kernel

```bash
nsys stats \
  --report nvtx_kern_sum \
  --format csv \
  my_profile.nsys-rep
```

用于回答某个逻辑 range 内执行了哪些 kernel、每个 kernel 多久。

### 9.3 给 operation 加 NVTX 前缀

```bash
nsys stats \
  --report 'cuda_gpu_sum:nvtx-name' \
  --format csv \
  my_profile.nsys-rep
```

`nvtx-name` 是“在 operation 名称前显示所属 NVTX range”，不是过滤值。下面这种写法是错误的：

```text
cuda_gpu_sum:my_range_name
```

正确做法是生成带前缀的报告后使用 `grep` 或 CSV 工具过滤。

### 9.4 CUDA kernel 汇总

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  my_profile.nsys-rep
```

可搜索：

- 自定义 kernel 名称；
- `ncclDevKernel_SendRecv`；
- `ncclDevKernel_AllReduce`；
- cuBLAS/CUTLASS/DeepGEMM kernel。

### 9.5 D2D/P2P copy

按大小汇总：

```bash
nsys stats \
  --report cuda_gpu_mem_size_sum \
  --format csv \
  my_profile.nsys-rep \
| grep 'Device-to-Device'
```

查看时间线样本：

```bash
nsys stats \
  --report cuda_gpu_trace \
  --format csv \
  my_profile.nsys-rep \
| grep 'Device-to-Device' \
| head -n 30
```

peer-mapped copy 在独立 comm stream 上时，未必完整归入 launch stream 的 NVTX projection。必须联合看：

- NVTX range；
- kernel summary；
- D2D trace；
- 外层 CUDA Event；
- 同步后的 wall-clock。

### 9.6 CUDA API 和 CPU launch

如果怀疑 CPU launch、同步或 API 阻塞：

```bash
nsys stats --report cuda_api_sum --format csv my_profile.nsys-rep
nsys stats --report osrt_sum --format csv my_profile.nsys-rep
```

重点关注：

- `cudaDeviceSynchronize`；
- `cudaStreamSynchronize`；
- `cudaEventSynchronize`；
- 大量短 kernel launch；
- host-side barrier；
- tensor map/JIT 首次构建。

## 10. 如何读时间线

### 10.1 判断是否真的 overlap

需要看到：

- compute kernel 位于 compute stream；
- communication/copy 位于独立 stream；
- 两者 GPU 时间区间重叠；
- 最终 current stream 只等待必要 completion event；
- 没有每层 device-wide synchronize。

仅仅“使用了两个 stream”不代表发生 overlap。如果 compute kernel 占满全部 SM，另一个 SM kernel 可能无法并发；Copy Engine 活动则可能仍可并行。

### 10.2 判断是否在等待远端数据

症状包括：

- fused kernel 生命周期远大于同 shape 纯 GEMM；
- kernel 内有 flag polling/spin；
- copy 完成时间决定后续 tile；
- CTA 先启动但长期没有有效计算；
- 增大 world size 后 kernel 时间按通信量增长。

此时 kernel duration 包含等待，不等于 Tensor Core 实际计算时间。

### 10.3 判断 DDP 是否被隐藏

准备两个口径：

1. backward 后手动同步：`BWD + SYNC`；
2. DDP bucket overlap：`BWD(with DDP)`。

粗略 overlap：

\[
T_{hidden}\approx T_{bwd,manual}+T_{sync,manual}-T_{bwd,DDP}
\]

这是近似值，因为 DDP hook、bucket、通信竞争会改变计算时间，但足以判断是否存在显著 overlap。

### 10.4 判断优化是否进入关键路径

一个 kernel 即使快很多，也可能不改善 wall-clock，原因包括：

- 原 kernel 已被其他工作完全隐藏；
- 优化后暴露了新的同步尾部；
- 与 NCCL/HBM 竞争导致其他阶段变慢；
- end-to-end 主要耗时在别处；
- 优化增加了 transpose/staging/barrier。

必须回到 L3/L4 验证。

## 11. Nsight Compute 的使用边界

Nsight Systems 用于“什么时候、谁与谁重叠”。Nsight Compute 用于“单个 kernel 为什么慢”。

先在 Systems 中定位 kernel，再用 NCU 检查：

- Tensor Core utilization；
- occupancy；
- memory throughput；
- warp stall reasons；
- instruction mix；
- L2/HBM traffic；
- cluster/CTA 调度。

不要直接对完整多卡训练运行 `ncu --set full`：开销巨大，collective 还可能超时。推荐：

1. 提取单 kernel microbench；
2. 降低 iteration；
3. 必要时只 profile 一个 rank；
4. 使用 kernel-name regex；
5. 先采轻量 metric set，再扩大范围。

## 12. Torch Profiler / Kineto

仓库提供：

```text
deep_gemm/testing/bench.py
```

其中：

- `bench`：CUDA Event microbench；
- `bench_kineto`：提取 CUDA kernel 平均时间和 Chrome trace。

Torch Profiler 适合快速查看 PyTorch op 和 kernel，但多进程、NCCL、P2P copy、跨 stream 依赖通常还是 Nsight Systems 更直观。

不要同时启用 Kineto 和 Nsight Systems/Compute。

## 13. 结果报告模板

每次性能结论建议记录：

```text
Hardware:
Software:
Git commit:
Command:
GPU set:
Shape/dtype/world size:
Warmup/iters/repeats:
Correctness threshold/result:

No-profiler latency:
- component A:
- component B:
- production Function:
- end-to-end wall:

Profiler attribution:
- dominant kernel:
- collective:
- memcpy:
- barrier/sync:
- overlap observed:

Conclusion:
Next experiment:
```

必须记录完整命令和 Git commit，否则新会话难以复现。

## 14. 临时产物清理

Nsight 会生成：

```text
*.nsys-rep
*.sqlite
*.qdstrm
```

这些通常很大，不应提交。提取文本结果后删除：

```bash
rm -f my_profile.nsys-rep my_profile.sqlite
```

提交前执行：

```bash
git status --short
git diff --check
```

## 15. 通用新会话 Checklist

1. 阅读本文和 `.codebuddy/memory/MEMORY.md`。
2. 明确要验证的性能假设。
3. 检查 GPU 是否空闲，禁止 benchmark 并发争卡。
4. 记录硬件、软件、Git commit、shape、dtype 和 world size。
5. 先跑正确性。
6. 拆 L1 组件并增加 same-shape 纯计算对照。
7. 测 L2 production Function/autograd，不只测手写替身。
8. 无 profiler 重复至少三轮。
9. 用少量 iteration 采 NVTX+Nsight Systems。
10. 联合分析 kernel、collective、copy、barrier 和 side stream。
11. 回到完整模块/训练，用 rank-max wall-clock 验证。
12. 清理 profile 二进制，只保存文档和文本结论。

---

# 附录 A：Ulysses POST Backward 案例

本案例展示如何把上述通用方法应用到通信融合 backward。

## A.1 专用入口

```text
examples/ulysses_variant/bench_wan21_post_bwd.py
```

它提供：

- baseline/fused 组件 CUDA Event；
- production `NCCLAllToAll + linear` 与 `FusedPostLinearFunction` 的真实 autograd backward；
- NVTX ranges；
- 8K/32K 通信量打印。

## A.2 正确性

```bash
DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/debug/debug_var_bwd.py 8
```

目标：所有 rank `grad_X rel=0`。

## A.3 无 profiler 独立测试

8K：

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 8192 --warmup 10 --iters 100 --publish-sync symm
```

32K：

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 32768 --warmup 10 --iters 50 --publish-sync symm
```

当前结果：

| Shape | Baseline local-op | Variant local-op | AG+GEMM | Actual autograd variant/baseline |
|---|---:|---:|---:|---:|
| 8K/SP8 | 0.218–0.261 ms | 0.669–0.707 ms | 0.540–0.577 ms | 约 1.34× |
| 32K/SP8 | 0.417–0.457 ms | 1.242–1.291 ms | 0.982–1.030 ms | 约 1.97× |

## A.4 Nsight 命令

```bash
rm -f post_bwd_8k_profile.nsys-rep post_bwd_8k_profile.sqlite

DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --wait=all \
  --force-overwrite=true \
  --output=post_bwd_8k_profile \
  python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
    8 --seq 8192 --warmup 2 --iters 5 --publish-sync symm
```

常用 range：

```text
baseline_dX_GEMM
baseline_dW_GEMM
baseline_dX_A2A
variant_AG+GEMM_(symm)
diagnostic_variant_dX_GEMM_only
actual_baseline_POST_BWD
actual_variant_POST_BWD
```

关键 Nsight 结果：

| GPU operation | 平均时间 |
|---|---:|
| `sm100_bf16_ag_gemm` | 475.7 us |
| 两个 symmetric barrier | 合计约 51.0 us |
| 同 shape 纯 dX GEMM | 45.1 us |
| baseline dX GEMM | 42.2 us |
| baseline dW GEMM | 32.9 us |
| NCCL A2A kernel | 46.9 us |

结论：慢点是 8×远端 payload、peer/chunk 调度和 AG kernel 内等待，不是 torch GEMM 本体。

## A.5 完整模型验证

```bash
DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_14b_train.py \
  8 --layers 40 --seq 8192 --warmup 3 --iters 10 \
  --strategies serial,fused_var --sync-mode ddp
```

当前 rank-max wall-clock：

| Strategy | Wall | Tokens/s |
|---|---:|---:|
| serial | 280.07 ms | 29,249.9 |
| fused_var | 290.57 ms | 28,193.0 |

局部 POST backward 差距更大，但完整模型中有公共计算和 DDP overlap，因此最终吞吐约下降 3.6%。
