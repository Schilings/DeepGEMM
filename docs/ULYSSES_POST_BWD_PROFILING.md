# Wan2.1 Ulysses POST Backward Profiling 指南

本文记录如何复现、拆分和分析 Wan2.1 14B Ulysses POST backward 的性能。目标是区分：

- baseline：完整 `Wo` 的 dX/dW GEMM，以及反向 NCCL All-to-All；
- variant：Wo shard 转置、AG+GEMM、sharded dW GEMM 和 dW 转置；
- 参数梯度 DDP 通信与 activation-gradient 通信；
- GEMM 本体、通信、barrier、布局转换和 autograd 开销。

当前专用入口：

```text
examples/ulysses_variant/bench_wan21_post_bwd.py
```

该脚本同时提供：

1. CUDA Event 组件计时；
2. 生产 `NCCLAllToAll + linear` 与 `FusedPostLinearFunction` 的真实 autograd backward；
3. NVTX ranges，供 Nsight Systems 做 CUDA timeline 归因。

## 1. 基本原则

### 1.1 不要并行运行占满同一组 GPU 的 benchmark

本机测试通常使用全部 8 张 GPU。两个 8-GPU benchmark 同时运行会竞争：

- SM；
- HBM；
- NVLink；
- Copy Engine；
- NCCL stream。

这会造成数倍波动，结果不可用。必须顺序执行不同 shape、不同同步模式和不同重复轮次。

### 1.2 先测无 profiler 延迟，再用 profiler 做归因

Nsight Systems 会增加 tracing 开销。正确流程是：

1. 用 CUDA Event / wall-clock 得到性能数字；
2. 重复多轮确认稳定性；
3. 用 Nsight 判断时间花在哪些 kernel、copy 和 barrier；
4. 不把 Nsight 下的绝对延迟直接当最终吞吐。

### 1.3 多卡时间取最慢 rank

分布式迭代由最慢 rank 决定。专用 benchmark 每次测量都会：

1. 测本 rank CUDA Event；
2. 对 elapsed time 做 `MAX all-reduce`；
3. 累加 rank-max 时间。

不要只打印 rank 0 本地时间。

### 1.4 区分三种口径

| 口径 | 用途 | 是否包含参数梯度同步 |
|---|---|---|
| 组件计时 | 定位 GEMM/A2A/AG/transpose | 否 |
| 实际 autograd POST BWD | 比较生产 POST backward | 否 |
| 40-block 训练 wall-clock | 最终训练吞吐 | 是，manual 或 DDP overlap |

组件和真实 POST BWD 不能替代完整训练结果；完整训练结果也无法直接告诉我们具体哪个 kernel 慢。

## 2. 环境检查

从仓库根目录执行：

```bash
cd /root/.local/codebuddy/DeepGEMM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
nsys --version
```

运行 DeepGEMM benchmark 时统一设置：

```bash
export DG_JIT_USE_NVRTC=1
export PYTHONPATH=$PWD/examples:$PWD
```

如果 `deep_gemm._C` 缺失：

```bash
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force
```

POST benchmark 会 import Wan2.1 模块，因此环境还需要仓库指定的 FlashAttention-4。安装方式见 `docs/INSTALL_FA4.md`。

当前环境有 Nsight Systems。若需要 Nsight Compute，可先检查：

```bash
ncu --version
```

缺失时参考 `docs/install_ncu.sh`。本文验证结论只依赖 Nsight Systems，不要求 `ncu`。

## 3. 先验证数值正确性

在 profiling 前先验证 production variant 的 activation gradient：

```bash
DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/debug/debug_var_bwd.py 8
```

目标结果：

```text
grad_X rel (serial vs var): 0.000000
```

所有 rank 都应通过。forward 因 BF16 reduce 顺序不同，relative difference 通常约为 `0.0029`。

`DG_AG_PUBLISH_SYNC` 的含义：

| 值 | 含义 |
|---|---|
| `symm` | 正式路径；stream-ordered symmetric-memory 发布/消费 barrier |
| `host` | 旧对照；两次 device sync + host process-group barrier |
| `none` | 不保证正确性，只允许受控诊断，不能用于正式结论 |

## 4. 独立组件和真实 autograd POST BWD

### 4.1 8K

```bash
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 8192 --warmup 10 --iters 100 --publish-sync symm
```

脚本会输出：

- baseline forward A2A、GEMM；
- variant forward GEMM-RS；
- baseline dX GEMM、dW GEMM、dX A2A；
- variant Wo transpose、AG+GEMM、dW GEMM、dW transpose；
- 与 variant 相同形状的纯 dX GEMM；
- production Function 的实际 autograd POST BWD。

### 4.2 32K

```bash
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 32768 --warmup 10 --iters 50 --publish-sync symm
```

### 4.3 顺序重复三轮

不要并行启动 8K 和 32K。正确示例：

```bash
for i in 1 2 3; do
  DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
  python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
    8 --seq 8192 --warmup 10 --iters 100 --publish-sync symm
done
```

记录每轮结果，不要只挑最快一次。重点观察：

- AG+GEMM 是否稳定；
- NCCL A2A 是否波动；
- actual autograd variant 是否稳定；
- 结果是否随 shape 符合通信量变化。

## 5. 当前独立测试基线

B300×8、SP=8、BF16：

### 5.1 8K 组件测试

三轮各 100 次：

| 指标 | 结果 |
|---|---:|
| baseline POST BWD local-op | 0.218–0.261 ms |
| variant POST BWD local-op | 0.669–0.707 ms |
| variant AG+GEMM | 0.540–0.577 ms |
| 同形状纯 dX GEMM | 约 0.07 ms |
| baseline A2A 远端 payload/rank | 8.75 MiB |
| variant AG 远端 payload/rank | 70 MiB |

### 5.2 32K 组件测试

三轮各 50 次：

| 指标 | 结果 |
|---|---:|
| baseline POST BWD local-op | 0.417–0.457 ms |
| variant POST BWD local-op | 1.242–1.291 ms |
| variant AG+GEMM | 0.982–1.030 ms |
| baseline A2A 远端 payload/rank | 35 MiB |
| variant AG 远端 payload/rank | 280 MiB |

远端 payload 比例始终为：

\[
\frac{\text{AG remote payload}}{\text{A2A remote payload}}=SP
\]

SP=8 时为 8 倍。

### 5.3 真实 autograd POST backward

直接调用 production `NCCLAllToAll + linear` 与 `FusedPostLinearFunction`：

| Shape | Baseline 均值 | Variant 均值 | Variant/Baseline |
|---|---:|---:|---:|
| 8K/SP8 | 约 0.772 ms | 约 1.035 ms | 约 1.34× |
| 32K/SP8 | 约 0.831 ms | 约 1.641 ms | 约 1.97× |

Baseline NCCL 延迟的轮间波动通常大于 variant AG 路径，因此必须重复多轮。

## 6. Nsight Systems 采集

### 6.1 为什么脚本中要加 NVTX

`bench_wan21_post_bwd.py` 会给每个组件添加 NVTX range，例如：

```text
baseline_dX_GEMM
baseline_dW_GEMM
baseline_dX_A2A
variant_AG+GEMM_(symm)
diagnostic_variant_dX_GEMM_only
actual_baseline_POST_BWD
actual_variant_POST_BWD
```

NVTX 让 Nsight 能把 kernel 与逻辑阶段关联起来。`mp.spawn` 会启动 8 个进程，Nsight Systems 默认按 process tree 跟踪子进程。

### 6.2 采集命令

先删除同名旧报告：

```bash
rm -f post_bwd_8k_profile.nsys-rep post_bwd_8k_profile.sqlite
```

采集少量迭代即可，避免 trace 过大：

```bash
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

参数说明：

| 参数 | 作用 |
|---|---|
| `--trace=cuda,nvtx,osrt,cublas` | 捕获 CUDA、NVTX、OS runtime 和 cuBLAS |
| `--sample=none` | 关闭 CPU sampling，降低额外开销 |
| `--cpuctxsw=none` | 关闭 CPU context-switch tracing |
| `--wait=all` | 等待 `mp.spawn` 子进程完成 |
| `--force-overwrite=true` | 覆盖同名报告 |

即使没有单独启用 NCCL report，NCCL kernels 仍会出现在 CUDA kernel timeline，例如 `ncclDevKernel_SendRecv`。

## 7. Nsight 统计提取

### 7.1 NVTX range 的 GPU 投影

```bash
nsys stats \
  --report nvtx_gpu_proj_sum \
  --format csv \
  --force-export=true \
  post_bwd_8k_profile.nsys-rep \
| grep -E 'baseline_dX_GEMM|baseline_dW_GEMM|baseline_dX_A2A|variant_AG\+GEMM|diagnostic_variant_dX_GEMM_only|actual_.*POST_BWD'
```

注意：该报告时间单位默认是纳秒。`Avg` 除以 1000 才是微秒。

8 ranks × 5 measured calls 对应 40 个 range instance。看到 `Instances=40` 属于预期。

### 7.2 NVTX 与内部 kernel 的对应关系

```bash
nsys stats \
  --report nvtx_kern_sum \
  --format csv \
  post_bwd_8k_profile.nsys-rep \
| grep -E 'variant_AG\+GEMM|baseline_dX_A2A|baseline_dX_GEMM|baseline_dW_GEMM'
```

它可以回答：

- AG range 内哪个 kernel 最慢；
- barrier kernel 占多少；
- baseline A2A 对应哪个 NCCL kernel；
- cuBLAS/torch GEMM 的真实 GPU kernel 时间。

### 7.3 给所有 kernel 加 NVTX 前缀

```bash
nsys stats \
  --report 'cuda_gpu_sum:nvtx-name' \
  --format csv \
  post_bwd_8k_profile.nsys-rep \
| grep 'variant_AG+GEMM_(symm)'
```

`cuda_gpu_sum:nvtx-name` 的含义是给 operation 名称添加 NVTX 前缀，不是把冒号后的文本当过滤条件。下面这种写法是错误的：

```text
cuda_gpu_sum:variant_AG+GEMM_(symm)
```

### 7.4 全局 kernel 汇总

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  post_bwd_8k_profile.nsys-rep \
| grep -E 'sm100_bf16_ag_gemm|ncclDevKernel_SendRecv|gemm'
```

### 7.5 D2D/P2P copy

按大小汇总：

```bash
nsys stats \
  --report cuda_gpu_mem_size_sum \
  --format csv \
  post_bwd_8k_profile.nsys-rep \
| grep 'Device-to-Device'
```

查看时间线样本：

```bash
nsys stats \
  --report cuda_gpu_trace \
  --format csv \
  post_bwd_8k_profile.nsys-rep \
| grep 'Device-to-Device' \
| head -n 30
```

当前 AG 将每个 local grad 分为 4 个 chunk。8K 时常见单 chunk 大小约为 `2.621 MiB`，完整 local grad 约为 `10.486 MiB`。

某些 peer-mapped copy 不一定全部归入 NVTX range 的 GPU projection，因为 copy 在独立 comm stream 上运行。此时应结合：

- `nvtx_gpu_proj_sum`；
- `nvtx_kern_sum`；
- `cuda_gpu_trace`；
- CUDA Event 外层延迟；

而不是只看一张汇总表。

## 8. 当前 Nsight 关键结果

8K/SP8、8 ranks × 5 calls：

| GPU operation | 平均时间 |
|---|---:|
| `sm100_bf16_ag_gemm` kernel | 475.7 us |
| 两个 symmetric-memory barrier 合计 | 约 51.0 us |
| 同形状纯 variant dX GEMM | 45.1 us |
| baseline dX GEMM | 42.2 us |
| baseline dW GEMM | 32.9 us |
| baseline NCCL A2A kernel | 46.9 us |
| AG NVTX range | 773.7 us，含 profiling overhead |

关键比值：

\[
\frac{475.7}{45.1}\approx 10.5
\]

这表示 AG fused kernel 的 GPU 生命周期约为同形状纯 GEMM 的 10.5 倍。AG kernel 内会等待远端 chunk，因此这个时间包含通信等待，不代表 Tensor Core GEMM 本身慢了 10.5 倍。

该结果与无 profiler 组件计时一致：慢点位于 AG 通信、ready-state 等待和调度，而不是 torch GEMM 或 dW GEMM。

## 9. 完整 14B 训练复核

组件结论最终必须回到 40-block 真实权重训练：

```bash
DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_14b_train.py \
  8 --layers 40 --seq 8192 --warmup 3 --iters 10 \
  --strategies serial,fused_var --sync-mode ddp
```

权威吞吐使用同步后的 rank-max wall-clock，而不是只使用 current-stream CUDA Event。脚本同时打印 `cuda=` 和 `wall=`，两者应非常接近。

当前结果：

| Strategy | FWD | BWD with DDP overlap | Wall | Tokens/s |
|---|---:|---:|---:|---:|
| serial | 94.33 ms | 185.77 ms | 280.07 ms | 29,249.9 |
| fused_var | 92.13 ms | 198.46 ms | 290.57 ms | 28,193.0 |

Variant 最终吞吐约下降 3.6%。POST backward 局部差距更大，但完整模型中还有大量公共计算，且 variant 不需要同步 replicated full Wo。

## 10. 常见错误

### 10.1 同时运行两个 8-GPU benchmark

症状：同一 shape 的结果在 `1.1×～3×` 间剧烈变化。

处理：确认没有其他训练或 benchmark 占用 GPU；所有 shape 顺序执行。

### 10.2 只看 rank 0

处理：使用 rank-max，而不是 rank-0 local time。

### 10.3 用 profiler 数字代替正式 benchmark

Nsight 下 AG range 从无 profiler 的约 `0.54–0.58 ms` 上升到约 `0.77 ms`。Nsight 用于归因，不用于最终吞吐。

### 10.4 组件和包含 DDP all-reduce

`BWD local-op sum` 明确不含 replicated-parameter all-reduce。要分析参数同步，使用完整训练的 `--sync-mode manual` 和 `--sync-mode ddp` 两个口径。

### 10.5 忽略 side stream

只在 default stream 上记录 CUDA Event 可能漏掉没有建立 event dependency 的 side-stream 尾部。完整训练使用：

1. 开始前 `torch.cuda.synchronize()`；
2. `time.perf_counter()`；
3. 结束后 `torch.cuda.synchronize()`；
4. rank-max wall-clock。

### 10.6 提交 profiler 二进制

`.nsys-rep` 和导出的 `.sqlite` 文件通常很大，不应提交。提取结果后清理：

```bash
rm -f post_bwd_8k_profile.nsys-rep post_bwd_8k_profile.sqlite
```

## 11. 新会话复现清单

1. 阅读本文和 `.codebuddy/memory/MEMORY.md`。
2. 确认 8 张 GPU 空闲，禁止并发跑多个 8-GPU任务。
3. 检查 `deep_gemm._C`、FA4、`nsys`。
4. 先跑 `debug_var_bwd.py 8` 验证 `grad_X rel=0`。
5. 无 profiler 顺序跑 8K/32K，至少重复三轮。
6. 对照组件计时与 actual autograd POST BWD。
7. 只用 2～5 measured iterations 采集 Nsight trace。
8. 用 NVTX、kernel summary 和 D2D trace 联合归因。
9. 回到 40-block 真实权重 wall-clock 验证优化是否真正转化为吞吐。
10. 删除 `.nsys-rep/.sqlite`，只提交文档和文本结果。
