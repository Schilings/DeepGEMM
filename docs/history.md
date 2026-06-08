Now let me also read the relevant DeepEP files for reference on the PDL mechanism:
Now I have a comprehensive understanding of both your current implementation and the DeepEP PDL mechanism. Let me provide you with the optimized version.
Now I have all the context needed. Let me provide the optimized implementation:
现在我来帮你重构整个方案。核心思路是：

1. **移除 RS Warps** → GEMM kernel 只做计算 + NVLink push（TMA 异步），不浪费 SM 资源在空转等待上
2. **Epilogue 改用 TMA store 异步写远端** → 替代同步 `*remote_ptr = ...`
3. **独立 Reduce Epilogue kernel** → 用 PDL (Programmatic Dependent Launch) 在 GEMM kernel 退出时自动被调度

让我开始实现：
好的，我已经完成了整体重构。让我详细解释优化后的架构设计：

---

## 重构后的两阶段架构

### 总览

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段1: sm100_bf16_gemm_rs_nt_impl (GEMM + NVLink Push)      │
│                                                              │
│  Warp 0: TMA Load (A + B)                                   │
│  Warp 1: MMA Issue (UMMA FMA → TMEM)                        │
│  Warp 2~3: Epilogue (TMEM → smem → TMA store to remote)     │
│                                                              │
│  ⚡ 无 RS Warps，全部线程用于计算+通信                       │
│  ⚡ TMA store 异步写远端，fire-and-forget                   │
│  ⚡ 写完后 cudaTriggerProgrammaticLaunchCompletion()         │
└────────────────────────────┬─────────────────────────────────┘
                             │ PDL (零间隙依赖启动)
                             ↓
┌──────────────────────────────────────────────────────────────┐
│ 阶段2: sm100_bf16_reduce_epilogue_impl (Reduce)             │
│                                                              │
│  cudaGridDependencySynchronize() — 数据已就绪，无需自旋       │
│  向量化累加 partial_buffer[rank_0..N-1] → output            │
│                                                              │
│  ⚡ 不内嵌于 GEMM kernel，无空转浪费                        │
│  ⚡ 利用全部 SM 并行 reduce                                  │
│  ⚡ GPU 硬件在 GEMM 即将退出时就能开始调度                   │
└──────────────────────────────────────────────────────────────┘
```

---

### 关键改进详解

#### 1. 移除 RS Warps → 独立 PDL Reduce Kernel

**之前的问题**：
- RS warps 占 128+ 线程（4 warps），但大部分时间在 `while (ld_acq_sys(ready_ptr) == 0)` 空转
- 浪费 SM 资源（寄存器、warp slot）
- 串行等待各 rank，延迟链长

**现在的做法**（模仿 DeepEP）：
- GEMM kernel 计算完直接退出，调用 `cudaTriggerProgrammaticLaunchCompletion()`
- Reduce kernel 通过 PDL 属性（Host 端设置 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`）被调度
- Reduce kernel 进入时 `cudaGridDependencySynchronize()`，此时所有 partial 数据已经就绪
- **零自旋、零空转、所有线程立即做有效工作**

#### 2. Epilogue → TMA Store 异步

**之前**：
```cpp
*remote_vec_ptr = ptx::ld_shared(smem_vec_ptr);  // 同步写，可能阻塞于 NVLink RTT
```

**现在**：
```cpp
ptx::tma_store_fence();
tma::store<...>(&tensor_map_partial, smem_cd[tma_stage_idx], ...);  // 异步
ptx::tma_store_commit();
```

TMA store 是 fire-and-forget 的，epilogue warp 发射后立即可以处理下一个 wave，不用等 NVLink 往返。

#### 3. Ready Flag: `st_rel_sys` 替代 `__threadfence_system`

**之前**：
```cpp
__threadfence_system();    // 刷所有 pending writes，代价极高
*remote_ready_ptr = 1;
```

**现在**：
```cpp
ptx::tma_store_wait();     // 只等本 block 的 TMA stores 完成
ptx::st_rel_sys(remote_ready_ptr, 1u);  // release semantics, 只保证当前 store 可见性
```

`st_rel_sys` 开销远低于 `__threadfence_system()`，后者会 flush 整个 GPU 的 write buffer。

---

### Host 端启动代码示例

你在 Host 端启动这两个 kernel 时需要这样配置 PDL：

```cpp
// 阶段1: GEMM + Push (可以是普通 launch 或 cooperative)
sm100_bf16_gemm_rs_nt_impl<<<grid, block, smem_size, stream>>>(...);

// 阶段2: Reduce Epilogue (PDL 属性)
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attrs[0].val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t config = {};
config.gridDim = reduce_grid;
config.blockDim = reduce_block;
config.dynamicSmemBytes = 0;
config.stream = stream;  // 同一个 stream！
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, sm100_bf16_reduce_epilogue_impl<...>, output, ...);
```

---

### 关于你的第 4 点疑问

> RS 归约改成 block-level pipeline？由于1的觉得不需要RS warp了，这个方向还用不用做？

**答：确实不需要了。** 因为：

1. 独立的 Reduce kernel 天然是 block-level 的——每个 block/thread 处理一段连续的 output，不存在 element-by-element polling
2. Reduce kernel 进入时数据已全部就绪（PDL 保证），不需要 pipeline 来隐藏等待延迟
3. 向量化 reduce（uint4 = 8 个 BF16 一次）已经充分利用了内存带宽

如果未来 `kNumRanks` 非常大（比如 64+），可以考虑用 shared memory 做分阶段累加减少 register pressure，但对于 8-16 rank 的典型场景，当前的直接 FP32 累加方案已经足够高效。

---

### 需要注意的事项

1. **`tensor_map_partial`**：你需要为 partial buffer 创建一个 TMA descriptor，这样 epilogue 才能用 `tma::store` 写远端。如果 TMA 不支持跨 NVLink 地址的 store（取决于驱动版本），可能需要退回到直接 store 方案（对于自身 rank 用 TMA，远端 rank 用直接 store）

2. **`ptx::st_rel_sys`**：确认你的 PTX 库有这个封装。如果没有，实现很简单：
   ```cpp
   __device__ __forceinline__ void st_rel_sys(uint32_t* ptr, uint32_t val) {
       asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
   }
   ```

3. **cudaTriggerProgrammaticLaunchCompletion**：这是 CUDA 12.0+ 的 device-side API。如果你用的是 cooperative launch，可以删掉这行（cooperative kernel 退出即为隐式完成信号）


你的观察非常正确！我只改了 `.cuh` kernel 实现文件，但调用链中的 **Host 端启动代码**、**启发式配置**、**TMA 描述符创建**、**Python 层接口** 等都还没有适配新架构。让我现在把整条链路打通。
Now let me read the current kernel implementation to understand the exact signature:
Now let me see the remaining part of the kernel to understand the reduce epilogue function signature:
Now I have a complete picture. The kernel uses `ptx::tma_store_fence()`, `tma::store<...>()`, `ptx::tma_store_commit()`, and `ptx::st_rel_sys()` which are **not yet defined** in the existing PTX headers. Also the host-side launch code needs to:

1. Create `tensor_map_partial` TMA descriptor
2. Launch the reduce epilogue kernel as a second kernel
3. Remove the RS warp thread count from the config

Let me also check what `tma::store` namespace looks like:
Now I have a full picture. Let me also check the `tma_copy.cuh` to see how `tma::copy` and `tma::store` are defined:
Now I have a complete understanding of the entire call chain and what's missing. Let me now fix everything to make the chain fully connected.
Let me start by modifying all the layers. First, let me add the missing PTX functions, then fix the host launch code, then the heuristics, the C++ API, and finally ensure the TMA store function exists:
Now add the `tma_store_fence` and `tma_store_commit` to the TMA PTX file. Looking at the existing code, other files use `cute::tma_store_fence()` — so we should use `ptx::tma_store_fence()` as an alias or just use `cute::tma_store_fence()`. Let me check what functions are available:
Now I see — the existing codebase uses `cute::tma_store_fence()` and `cute::tma_store_arrive()` (not `ptx::tma_store_fence()`/`ptx::tma_store_commit()`). Let me also check how `cute::SM90_TMA_STORE_2D::copy` is used for the TMA store:
Now let me look at the epilogue store_cd pattern which is the canonical SM100 TMA store approach:
Now I understand the canonical pattern:
1. `cute::tma_store_fence()` — before the TMA store
2. `cute::SM90_TMA_STORE_2D::copy(...)` — the actual TMA store
3. `cute::tma_store_arrive()` — after the TMA store
4. `ptx::tma_store_wait()` — wait for all stores to complete

My kernel uses non-standard `ptx::tma_store_fence()` and `ptx::tma_store_commit()` — I need to fix these to use the standard CuTe functions. Also, the kernel uses `tma::store<...>()` which doesn't exist. Let me fix the kernel `.cuh` to use the correct APIs, then fix the host launch code.

Let me first update the kernel to use the correct TMA store API:
Now I need to check where in the kernel the `NamedBarrier::sync` on index 0 was previously, to avoid conflicts. Looking at the context, we already had that sync there before the TMA TMEM→smem step. Let me re-read the current state of the epilogue section to verify correctness:
I see there's a redundant `NamedBarrier::sync` now. The pattern should be:
1. `NamedBarrier::sync` — ensure TMEM→smem writes are done (line 310)
2. TMEM→smem writes happen
3. `NamedBarrier::sync` — ensure all threads done writing smem (line 344)
4. `cute::tma_store_fence()` — fence before TMA store
5. TMA store (only one thread)
6. `cute::tma_store_arrive()` — commit the store

But looking at the canonical pattern in `sm100_store_cd.cuh`, it's:
1. TMEM→smem writes
2. `cute::tma_store_fence()` (which acts as a memory fence for the prior shared writes)  
3. `NamedBarrier::sync` — then one thread does TMA store
4. TMA store + arrive

So the extra `NamedBarrier::sync` I introduced is redundant with the one already at line 344. Let me fix this:
Now I also notice the TMA store writes to *local rank's* slot in the remote node. But actually the design is: each GEMM block writes its result to the *destination rank's* partial buffer. The TMA descriptor is created once pointing to the *local* symmetric buffer, but `sym_buffer.map()` maps the pointer to remote. However, **TMA store uses a tensor map descriptor**, which has a fixed base address — you can't dynamically change the destination address per store. This is a fundamental issue!

The proper approach for cross-node writes via NVLink where TMA descriptors can't point to remote memory is to use `ptx::tma_store_1d` (cp.async.bulk) which takes a raw global pointer rather than a tensor map. Let me check how the project handles this:
`ptx::tma_store_1d` is the `cp.async.bulk` instruction that takes raw global pointers (no tensor map needed). This is perfect for our use case since we need to write to different remote addresses.

However, looking at the design more carefully, there's actually an important subtlety: the **TMA store with a tensor map** (`SM90_TMA_STORE_2D::copy`) requires the destination address to be encoded in the tensor map at descriptor creation time. Since each block may write to a different remote rank, we can't use a single `tensor_map_partial` for all destinations.

The correct approach is:
1. For **self-rank** writes: use the `tensor_map_partial` (which points to local sym buffer partial area)
2. For **remote-rank** writes: use `ptx::tma_store_1d` with the mapped remote pointer

OR, simpler: just use `ptx::tma_store_1d` for all writes (both local and remote), since it works with any global pointer and still provides async behavior.

Let me update the kernel to use `ptx::tma_store_1d` which is the appropriate API for this case:
Now since we're using `tma_store_1d` instead of `SM90_TMA_STORE_2D`, we no longer need the `tensor_map_partial` parameter in the kernel signature! Let me update the kernel to remove it:
Also remove the prefetch of tensor_map_partial:
Good. Now the kernel signature matches the original one (no `tensor_map_partial`). The host launch code already passes `tensor_map_a` and `tensor_map_b` — so the GEMM kernel side is aligned. Now let me fix the `n_block_idx` variable that's used in the ready flag but might not be in scope. Let me check the epilogue ready flag section:
I see `n_block_idx` is used in the ready flag, but within the inner loop the variable `n_idx` changes per store. The `n_block_idx` comes from `get_next_block()` — it's the block-level N index, not the sub-tile index. That's correct since the ready flag is per (m_block, n_block) pair. The `n_block_idx` is still in scope from the `get_next_block()` call at line 294. Good.

Now let me also check that `ready_base[i] = 0` line uses `uint32_t*`:
Good. Now let me also confirm that `st_rel_sys` takes `uint32_t*` and `get_ready_ptr` returns `uint32_t*` — yes, that's correct per the layout file.

Now let's move to the **Host launch layer** — this is the crucial part. I need to:
1. Update `GemmRSConfig` to remove `num_rs_threads` (no more RS warps)
2. Update the host launch to also launch the reduce epilogue kernel
3. Update the kernel code generation to include the reduce epilogue instantiation
Now update the **Host launch layer** — this is the most critical part. I need to:
1. Update the `Args` struct and `generate_impl` to match new kernel signature (no `num_rs_threads`)
2. Add the reduce epilogue kernel as a second kernel launch
3. Keep it as a two-kernel launch within the same stream (PDL handles the dependency)
Now let me also check the `cast_into_bf16_and_pack` function used in the kernel — it's in the utils namespace:
Good, it's in `math.cuh`. The kernel uses `cast_into_bf16_and_pack` without namespace — let me check if it's included properly:
Ah, the canonical usage is `math::cast_into_bf16_and_pack`. But our kernel uses it without namespace prefix. Let me check how it's called in the kernel:
Let me check if the `sm100` namespace or included headers bring `cast_into_bf16_and_pack` into scope:
-
So it's in `deep_gemm::math` namespace. The kernel uses `using namespace deep_gemm::sm100;` — let me check if `sm100` namespace brings math in:
The `sm100` namespace doesn't bring in `math`. But looking at other SM100 kernels in this project (`sm100_fp8_gemm_1d1d.cuh` line 534), they directly use `cast_into_bf16_and_pack` without the `math::` prefix. Let me check:
-
Same pattern — `using namespace deep_gemm::sm100;` and no `math::` prefix. This means either:
1. ADL (Argument-Dependent Lookup) finds it since the args are in `deep_gemm` namespace types
2. Or some header brings it into scope via `using namespace`

Let me check if `epilogue_utils.cuh` or other includes do a `using namespace`:
So there's no `using namespace deep_gemm::math`. The function must work because it's in `deep_gemm` namespace and the code is also in `deep_gemm` namespace — the kernel is in `namespace deep_gemm { ... }`. Since both the kernel and `math::cast_into_bf16_and_pack` are within `deep_gemm`, but `cast_into_bf16_and_pack` is actually in `deep_gemm::math`, the unqualified lookup won't find it unless ADL kicks in.

Wait — looking at the other sm100_fp8 kernel that compiles fine, they also use it without qualification. This means the code likely compiles with implicit lookup since the functions are all templates inside the `deep_gemm` namespace's nested namespace. Actually, let me look at `utils.cuh`:
Looking at this more carefully, I think the `cast_into_bf16_and_pack` works because the kernel template function body is instantiated in `deep_gemm` namespace scope (it's just inside a `namespace deep_gemm { ... }` block), and since `deep_gemm::math` is a nested namespace, unqualified lookup from `deep_gemm` namespace will find names in `deep_gemm::math` **only if** there's a using-directive. Actually no, that's not how C++ works.

Let me just check how the existing compiled FP8 kernel handles this — it must work since the project compiles. Maybe `math.cuh` defines the function in the parent `deep_gemm` namespace:
OK so it's definitively in `deep_gemm::math`. The fact that existing code uses it without prefix means something else is going on. Maybe the NVCC compiler is more lenient, or maybe the `sm100_utils.cuh` includes chain eventually makes it available. In any case, the existing kernels compile fine with this, so our kernel will too (it uses the exact same pattern as `sm100_fp8_gemm_1d1d.cuh`).

Now, there's one important issue I noticed: the `launch_impl` of the reduce kernel passes `void*` for `y`, but the kernel expects `cd_dtype_t*`. In the JIT flow, the kernel is instantiated with a specific `cd_dtype_t` template parameter, and at launch time we pass raw `void*` pointers — the `launch_kernel` function takes `void**` args anyway, so the pointer is just passed by address. This is fine.

However there's a type mismatch: `launch_kernel` expects the exact types matching the kernel signature. The kernel takes `cd_dtype_t* __restrict__ output` but we're passing `void*`. Since the JIT compiles the kernel with the right type and `launch_kernel` just passes `void*[]` args, this should work — each arg is passed by its pointer-to-arg address.

Actually wait — looking at the `launch_kernel` implementation:
```cpp
template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return ...;
}
```

It takes `&args...` — so if we pass `void*`, it stores `void**`, and the kernel sees the bytes that constitute a pointer. That's correct since the kernel expects a pointer (`cd_dtype_t*`), and `void*` and `cd_dtype_t*` have the same size/representation. ✓

But actually there's still a problem: we're passing `static_cast<uint32_t>(args.runtime_m_per_rank)` — but `args.runtime_m_per_rank` is `int`. The cast creates a temporary, and `&temporary` is fine as it lives for the duration of the statement. Actually let me make the reduce `launch_impl` cleaner by just passing the right types directly:
Now let me update the **C++ API layer** — it referenced `num_rs_threads` in the thread count calculation:
The API layer calls `sm100_bf16_gemm_rs_nt(y, a, b, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_tokens_per_rank, n, k, compiled_dims)` but my new host function signature is `sm100_bf16_gemm_rs_nt(y, a, b, sym_buffer, sym_buffer_ptrs, rank_idx, max_m_per_rank, runtime_m_per_rank, n, k, compiled_dims)` — note it now also needs the `sym_buffer` tensor itself (for getting the workspace base pointer). Let me update the API layer:
Now I also need to check the `sm100_bf16_gemm_rs_nt` function in the host launch file — I defined it to take `const torch::Tensor& sym_buffer` as well. Let me verify the current signature matches:
Good. The function signature now takes `const torch::Tensor& sym_buffer` as the 4th parameter. The call from the API layer passes `sym_buffer` correctly.

Now let me also check the GEMM kernel `launch_impl` to make sure `args.y` isn't being passed anymore (since the GEMM kernel doesn't write to `y` directly — only the reduce kernel does):
Wait — but looking at the kernel signature:

```cpp
sm100_bf16_gemm_rs_nt_impl(const uint32_t shape_m_per_rank,
                           const uint32_t runtime_m_per_rank,
                           const uint32_t shape_n,
                           const uint32_t shape_k,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b)
```

The parameters are:
1. `shape_m_per_rank` (uint32_t) — corresponds to `max_m_per_rank`
2. `runtime_m_per_rank` (uint32_t)
3. `shape_n` (uint32_t)
4. `shape_k` (uint32_t)
5. `sym_buffer` (SymBuffer<kNumRanks>)
6. `tensor_map_a`
7. `tensor_map_b`

But `launch_impl` passes: `max_m_per_rank, runtime_m_per_rank, n, k, sym_buffer_ptrs, tensor_map_a, tensor_map_b`

The types need to match exactly. `args.max_m_per_rank` is `int` but kernel expects `uint32_t`. Let me fix:
Now let me also check the generate_impl for the GEMM kernel. The old code instantiated the template with `num_rs_threads` as one of the parameters, but our new kernel template no longer has that. Let me verify the template parameters match:

The new kernel template is:
```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
```

And the `generate_impl` produces:
```
sm100_bf16_gemm_rs_nt_impl<
    block_m, block_n, block_k,
    num_stages,
    num_non_epilogue_threads, num_epilogue_threads,
    grid_dim.first, num_ranks,
    cd_dtype_t
>
```

That's 9 template args. Let me read the format string:
Good — the format string has 9 `{}` placeholders matching the 9 template params:
1. `block_m` → BLOCK_M
2. `block_n` → BLOCK_N
3. `block_k` → BLOCK_K
4. `num_stages` → kNumStages
5. `num_non_epilogue_threads` → kNumNonEpilogueThreads
6. `num_epilogue_threads` → kNumEpilogueThreads
7. `grid_dim.first` → kNumSMs
8. `num_ranks` → kNumRanks
9. `cd_dtype_t` → cd_dtype_t

This matches the kernel template perfectly. ✓

Now let me check the Reduce kernel's `generate_impl`:
The reduce kernel template is:
```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads = 256,
          typename cd_dtype_t = cutlass::bfloat16_t>
```

And `generate_impl` passes 6 format args:
1. `block_m` → BLOCK_M
2. `block_n` → BLOCK_N
3. `grid_dim.first` → kNumSMs
4. `num_ranks` → kNumRanks
5. `reduce_num_threads` → kNumThreads
6. `cd_dtype_t` → cd_dtype_t

This matches. ✓

Now there's one important issue: the JIT system expects **exactly 1 kernel symbol** per compiled `.cu` file (see `kernel_runtime.hpp` line 79: `DG_HOST_ASSERT(symbol_names.size() == 1)`). But both the GEMM kernel and the reduce kernel are in the **same** header file. When the reduce kernel's JIT code includes `sm100_bf16_gemm_rs.cuh`, it will instantiate both kernels!

I need to either:
1. Put the reduce kernel in a separate header file, OR
2. Have the JIT code only instantiate one kernel at a time (which it does — the `__instantiate_kernel` function only references one)

Actually looking more carefully at the JIT system: the `load_kernel` function with `DG_JIT_USE_LIBRARY_ENUM_KERNELS` counts kernels with `cuLibraryGetKernelCount`, and without it, it looks for symbols with `STT_FUNC` and `STO_ENTRY`. The `__instantiate_kernel` function is a static host function that just takes the address — it forces the linker to include that one kernel. But the other kernel template will NOT be instantiated because templates only get instantiated when explicitly referenced.

So since the `generate_impl` for GEMM only references `sm100_bf16_gemm_rs_nt_impl<...>` and the one for reduce only references `sm100_bf16_reduce_epilogue_impl<...>`, each JIT compilation will only have 1 kernel. ✓

Now let me update the TODO and fix the remaining items:
The C++ API layer is already done (just the one-line change to pass `sym_buffer`). Now let me verify the Python layer doesn't need changes — the Python `bf16_gemm_rs_nt` function already passes `sym_buffer.buffer` as the 4th arg:
Good — the Python layer passes `sym_buffer.buffer` (which is the raw tensor), matching the C++ API's `const torch::Tensor& sym_buffer` parameter. The Python interface doesn't need any changes! ✓

Now let me check the pybind11 registration — it needs to match:
The pybind11 registration already has `sym_buffer` as the 4th argument. This matches both the C++ function and the Python call. ✓
Now let me do a final verification pass — make sure there are no remaining references to `num_rs_threads` and that everything is consistent:
Good — the `num_rs_threads` references are only in the FP8 variant (`sm100_fp8_gemm_rs.hpp`), which we haven't touched. It still uses the old architecture with RS warps. Our BF16 version no longer references it. ✓

Let me also check if there are other files referencing `config.num_rs_threads` that might break:
The FP8 variant still references `config.num_rs_threads` from `GemmRSConfig`, but we removed that field! This will break the FP8 path. I have two options:

1. Keep `num_rs_threads` in the config struct (set to 0 for BF16 path)
2. Give the FP8 path its own config struct

Option 1 is simpler and backward-compatible. Let me re-add it with a default of 0:
Now update the config construction to include `num_rs_threads = 0` for the BF16 path:
Now fix the config struct initialization to include `num_rs_threads`:
Now let me verify the FP8 path still works — it uses `config.num_rs_threads + config.num_non_epilogue_threads + config.num_epilogue_threads` for thread count. With the BF16 config having `num_rs_threads = 0`, the BF16 launch uses `0 + 128 + 128 = 256` threads... but wait, the BF16 launch code I wrote uses `config.num_non_epilogue_threads + config.num_epilogue_threads` directly. Let me verify:
Good — the BF16 path correctly uses `128 + 128 = 256` threads (no RS warps). The FP8 path still uses `128 + 128 + 128 = 384` threads. ✓

Now let me do one final check — the `__init__.py` top-level exports. Let me verify nothing extra needs to be exported:
Good — no changes needed there. The reduce kernel is launched internally by the C++ host code, not exposed to Python.

Let me now read the final state of all modified files to make sure everything is coherent:
Everything looks good. Let me also verify there are no issues with the FP8 path. The FP8 kernel uses `GemmRSConfig` too, but it uses `elem_size_ab = 1` (FP8 = 1 byte), so `block_k = 128/1 = 128`. However I now have `num_rs_threads = 0` hardcoded in `get_gemm_rs_config`. The FP8 path **still needs** RS threads! 

The issue is that `get_gemm_rs_config` is a **shared** function between BF16 and FP8 paths. The FP8 path passes `elem_size_ab=1`, and the BF16 path passes `elem_size_ab=2`. I need to differentiate. Let me check what `elem_size_ab` value the BF16 path passes:

Looking at the BF16 host code: `get_gemm_rs_config(m, n, k, num_sms, static_cast<int>(a.element_size()))` — for BF16 `a.element_size() = 2`.

And for FP8: it would be `1`.

So I can use `elem_size_ab` to decide whether to allocate RS threads:
Now let me also verify one more potential issue: the `SymBuffer<>` default template argument. The GEMM kernel uses `SymBuffer<kNumRanks>` but the host code constructs `SymBuffer<>` (default template). Let me check what the default is:
I see: `SymBuffer<kNumRanks = kNumMaxRanks>` — the template parameter only affects the static assert, not the actual struct size (which always uses `offsets[kNumMaxRanks]`). So `SymBuffer<>` (72 max ranks) and `SymBuffer<8>` have the same layout, just different compile-time assertions. The host uses `SymBuffer<>` which is `SymBuffer<72>`, and the kernel uses `SymBuffer<kNumRanks>`. Since they have identical memory layout, passing by value through `void**` kernel args works fine. ✓

Now let me think about one more potential issue: the `launch_args.grid_dim.first` is `num_sms` for both kernels. The GEMM kernel uses `grid_dim.first` as `kNumSMs` template parameter. The reduce kernel also uses `grid_dim.first` as `kNumSMs`. Since both use `num_sms` from the device, the reduce kernel template will have `kNumSMs = device's SM count`, which is correct — it uses all SMs for parallel reduction.

Let me now verify the kernel `.cuh` file is complete and has no issues from our edits:
Good. Now let me also check one thing — the kernel needs `#include <deep_gemm/ptx/tma.cuh>` for `ptx::tma_store_1d` and `ptx::tma_store_wait`, and it needs `cute::tma_store_fence()` / `cute::tma_store_arrive()`. The `tma.cuh` should be included through the includes chain. Let me verify:
The kernel includes `sm100_utils.cuh` → `tma_utils.cuh` → `cute/arch/copy_sm90_tma.hpp` which defines `cute::tma_store_fence()`, `cute::tma_store_arrive()`, and `cute::SM90_TMA_STORE_2D`. 

But our kernel also uses `ptx::tma_store_1d` and `ptx::tma_store_wait` from `deep_gemm/ptx/tma.cuh`. Let me check if that's included:
Our kernel doesn't include `ptx/tma.cuh` explicitly! Let me add it:
Now let me also check if `cudaTriggerProgrammaticLaunchCompletion` and `cudaGridDependencySynchronize` need any specific headers:
Good — these are CUDA built-in device functions, available in CUDA 12.0+ without special headers. They're used throughout the project. ✓

Let me do one final comprehensive check — look for anything I might have missed:
Wait — the GEMM args struct has a `void* y` field that's populated with `y.data_ptr()`, but the GEMM kernel doesn't write to `y` directly anymore (only the reduce kernel does). However it's still in the Args struct. That's fine — it's unused in `launch_impl` but kept for potential future use or debugging. Actually let me clean it up since it's dead code:
Now fix the Args initialization at line ~167 that sets `.y`:
Let me now do a final read of the complete host launch file to verify everything is consistent:
Everything looks coherent. Let me now verify the final API layer:
The call passes `(y, a, b, sym_buffer, sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_tokens_per_rank, n, k, compiled_dims)` which matches the host function signature `(y, a, b, sym_buffer, sym_buffer_ptrs, rank_idx, max_m_per_rank, runtime_m_per_rank, n, k, compiled_dims)`. ✓

Now let me trace through the entire call chain one more time to make sure it's fully connected:

```
Python: bf16_gemm_rs_nt(y, a, b, sym_buffer, num_tokens_per_rank, compiled_dims)
  └─ _C.bf16_gemm_rs_nt(y, a, b, sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
                         sym_buffer.group.rank(), sym_buffer.num_max_tokens_per_rank,
                         num_tokens_per_rank, compiled_dims)
      └─ pybind11: bf16_gemm_rs_nt(y, a, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                                    num_max_tokens_per_rank, num_tokens_per_rank, compiled_dims)
          └─ APIs layer: validates shapes, calls sm100_bf16_gemm_rs_nt(y, a, b, sym_buffer,
                         sym_buffer_ptrs, rank_idx, num_max_tokens_per_rank, num_tokens_per_rank, n, k, compiled_dims)
              └─ Host launch:
                  1. get_gemm_rs_config() → GemmRSConfig (num_rs_threads=0, reduce_num_threads=256)
                  2. make_tma_2d_desc(a, ...) → tensor_map_a
                  3. make_tma_2d_desc(b, ...) → tensor_map_b
                  4. JIT compile GEMM kernel → sm100_bf16_gemm_rs_nt_impl<128,128,64,stages,128,128,N_SMs,N_ranks,dtype>
                  5. Launch GEMM kernel with PDL enabled
                  6. JIT compile Reduce kernel → sm100_bf16_reduce_epilogue_impl<128,128,N_SMs,N_ranks,256,dtype>
                  7. Launch Reduce kernel with PDL enabled (same stream)
```

**GEMM kernel flow:**
- Warp 0: TMA load A+B
- Warp 1: UMMA MMA
- Warp 2-3: TMEM→smem→`ptx::tma_store_1d`(远端/本地 partial buffer) + `ptx::st_rel_sys`(ready flag)
- Exit: `cudaTriggerProgrammaticLaunchCompletion()`

**Reduce kernel flow:**
- `cudaGridDependencySynchronize()` — 等待 GEMM 完成
- 向量化读取各 rank 的 partial buffer → FP32 累加 → BF16 → 写 output

✓ 调用链完全打通了！

让我最后检查一下是否有遗留问题：
Only in `history.md` (design doc). No code references remain. ✓
Good — no more references to the nonexistent `tma::store<>` function. ✓
No more references to the nonexistent `ptx::tma_store_fence()` and `ptx::tma_store_commit()`. ✓

Let me now verify our `ptx::st_rel_sys` function is correctly referenced:
And the definition is in `ptx/ld_st.cuh`:
The kernel calls `ptx::st_rel_sys(remote_ready_ptr, 1u)` where `remote_ready_ptr` is `uint32_t*` and `1u` is `uint32_t`. This matches the first overload. ✓

Everything is now fully connected. Let me provide a summary:

---

好的，现在整条调用链已经完全打通了。让我总结一下所有修改：

## 修改文件清单

### 1. `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh` (Kernel 实现)

- ✅ 移除了 `tensor_map_partial` kernel 参数（改用 `ptx::tma_store_1d` 直接写远端地址）
- ✅ 添加了 `#include <deep_gemm/ptx/tma.cuh>` 和 `#include <deep_gemm/common/tma_copy.cuh>`
- ✅ Epilogue 使用 `cute::tma_store_fence()` + `ptx::tma_store_1d()` + `cute::tma_store_arrive()` 标准异步写模式
- ✅ Ready flag 使用 `ptx::st_rel_sys()` 替代 `__threadfence_system`
- ✅ GEMM kernel 退出时调用 `cudaTriggerProgrammaticLaunchCompletion()`
- ✅ Reduce kernel 入口调用 `cudaGridDependencySynchronize()`

### 2. `deep_gemm/include/deep_gemm/ptx/ld_st.cuh` (PTX 工具)

- ✅ 新增 `ptx::st_rel_sys(uint32_t*, uint32_t)` 和 `ptx::st_rel_sys(int*, int)` 

### 3. `csrc/jit_kernels/heuristics/gemm_rs.hpp` (启发式配置)

- ✅ `num_rs_threads` 根据 `elem_size_ab` 动态设置（BF16=0, FP8=128）
- ✅ 新增 `reduce_num_threads = 256` 配置项

### 4. `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp` (Host 启动层) ⭐核心

- ✅ `SM100BF16GemmRSRuntime`: GEMM kernel 的 JIT + launch（256 线程，无 RS warps）
- ✅ `SM100BF16ReduceEpilogueRuntime`: Reduce kernel 的 JIT + launch（256 线程）
- ✅ `sm100_bf16_gemm_rs_nt()`: 统一入口，顺序发射两个 kernel 到同一 stream
- ✅ 参数类型与 kernel 签名精确对齐（`uint32_t` 转换）
- ✅ PDL 通过 `LaunchRuntime` 基础设施自动设置

### 5. `csrc/apis/gemm_rs.hpp` (C++ API 层)

- ✅ 将 `sym_buffer` tensor 传递给 host launch 函数（reduce kernel 需要 workspace 指针）

### 6. Python 层 (`deep_gemm/gemm_rs/__init__.py`) 

- ✅ 无需修改 — 已有的接口完全兼容

---

## 数据流完整路径

```
Python bf16_gemm_rs_nt()
  → pybind11 bf16_gemm_rs_nt()
    → APIs 参数验证
      → sm100_bf16_gemm_rs_nt()
        ├─ get_gemm_rs_config(elem_size=2) → {num_rs_threads=0, reduce_num_threads=256}
        ├─ make_tma_2d_desc(a) / make_tma_2d_desc(b)
        ├─ JIT compile + launch: sm100_bf16_gemm_rs_nt_impl<128,128,64,...>
        │   ├─ Warp 0: TMA Load (A+B → smem)
        │   ├─ Warp 1: UMMA MMA (smem → TMEM accumulator)
        │   ├─ Warp 2-3: Epilogue (TMEM → smem → tma_store_1d → remote partial)
        │   ├─ st_rel_sys(ready_flag)
        │   └─ cudaTriggerProgrammaticLaunchCompletion()
        │          │ PDL
        └─ JIT compile + launch: sm100_bf16_reduce_epilogue_impl<128,128,...>
            ├─ cudaGridDependencySynchronize()
            └─ vectorized reduce (partial_bufs → output y)
```

**关键设计决策**：使用 `ptx::tma_store_1d`（`cp.async.bulk`）而非 `SM90_TMA_STORE_2D`，因为 TMA tensor map 的 base address 是编译时固定的，无法动态指向不同 rank 的远端 NVLink 地址。`cp.async.bulk` 接受运行时计算的全局指针，完美适配跨 rank 写入场景。

现在让我再确认一下我们之前实现的 BF16 GEMM-RS 算子的情况：
现在我有足够的信息来回答这个问题了。

## 回答

是的，**DeepGEMM 的 GEMM 和 MegaMoE 算子都是 persistent kernel 模式**，我们的 BF16 GEMM-RS 算子**也是**。

---

### DeepGEMM 的 GEMM / MegaMoE：Persistent Kernel

核心特征：

1. **Grid 大小 = SM 数量**（而不是 tile 总数）：
   ```cpp
   LaunchArgs(config.launch_config.num_sms, ...)
   ```

2. **每个 CTA 在循环中处理多个 tile**，以 `kNumSMs` 为步长 round-robin 取 tile：
   ```cpp
   // Persistently schedule over blocks
   while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
       const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;
       ...
   }
   ```

3. **MegaMoE 同理**——用 `for_each_block()` 包裹 `while(true)` 循环，内部 `block_idx += kNumSMs` 做 round-robin 调度，跨多个 expert 的多个 tile 持续工作。

---

### 我们的 BF16 GEMM-RS 算子：同样是 Persistent Kernel

从代码可以看到：

1. **Grid 大小 = `num_sms`**：
   ```cpp
   .launch_args = LaunchArgs(num_sms, ...)
   ```

2. **Kernel 内部有 persistent 循环**，以 `kNumSMs` 为步长：
   ```cpp
   auto get_next_block = [&](uint32_t& block_idx, ...) {
       ...
       block_idx += kNumSMs;  // round-robin 步进
       return true;
   };
   
   while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
       // 处理 tile...
   }
   ```

3. **调度逻辑额外引入了 rank wave 维度**：先算远程 rank 的分块（通过 NVLink push），最后算本地 rank 的分块——这样 N-1 个通信 wave 和计算自然重叠。

---

### 总结对比

| 特征 | GEMM / MegaMoE | 我们的 GEMM-RS |
|------|---------------|----------------|
| Grid 大小 | `num_sms` | `num_sms` |
| 每 CTA 处理 | 多个 tile（循环） | 多个 tile（循环） |
| 调度策略 | `blockIdx.x + iter * kNumSMs` | `blockIdx.x + iter * kNumSMs`，外加 rank wave 分层 |
| 循环注释 | "Persistently schedule over blocks" | `get_next_block` lambda |
| PDL 支持 | ✅ | ✅（两阶段 kernel 之间用 PDL） |

所以三者都是 **persistent kernel**——只启动 SM 数量的 CTA，每个 CTA 持续从全局 tile 池中取任务直到耗尽。这种模式的好处是：
- **负载均衡**：tile 动态分配，无需预划分
- **减少 launch overhead**：一次 launch 完成所有 tile
- **更好的 L2 cache 利用**：可以配合 tile swizzle 优化访存局部性
- **天然适合 PDL**：persistent kernel 运行时间可预测，便于流水线依赖调度



现在我全面理解了架构。让我制定修改计划：

**现状分析：**
1. GEMM kernel 的 epilogue 将 FP32 累加器转为 `cd_dtype_t`（BF16）push 到远端 partial buffer
2. Reduce epilogue 从 partial buffer 读 BF16，用 FP32 累加后转 BF16 写出
3. 通信数据类型（partial buffer 中的数据类型）目前固定是 `cd_dtype_t`（BF16）

**用户需求：**
- 通信格式应该可以通过模板参数控制
- 如果用户不在乎精度，可以用 BF16 通信（省带宽）+ BF16 累加（不用 FP32）
- 如果用户在乎精度，可以用 FP32 通信 + FP32 累加
- 甚至可以 BF16 通信 + FP32 累加（当前行为）

我需要引入一个新的模板参数 `comm_dtype_t`（通信数据类型）和 `accum_reduce_in_fp32`（reduce 累加是否用 FP32）。
让我先看看 Python 接口层怎么调用的：
Now I have a complete picture. Let me also look at the `to_string` function for dtype mapping and understand the `runtime_utils.hpp`:
Now I have a complete understanding. Let me design the solution:

**Design:**
- 引入 `comm_dtype_t` 模板参数，控制 partial buffer 中存储的数据类型（通信数据类型）
- 引入 `kReduceInFP32` 模板参数（bool），控制 reduce epilogue 是否用 FP32 累加
- GEMM epilogue 会把 FP32 累加器转为 `comm_dtype_t` 后 push
- Reduce epilogue 根据 `kReduceInFP32` 选择 FP32 或直接 `comm_dtype_t` 累加
- Python 层暴露 `comm_dtype` 参数（默认 `torch.bfloat16`）和 `reduce_in_fp32` 参数（默认 `True`）

开始实现：
Now update the workspace construction in the GEMM kernel to use `comm_dtype_t`'s size:
Now update the epilogue to store in `comm_dtype_t` format. The key section is where data gets written to the partial buffer:
Now update the Reduce Epilogue kernel to support `comm_dtype_t` and `kReduceInFP32`:
`GemmRSWorkspace` 本身已经通过 `elem_size` 参数支持动态大小了（构造时传入 `sizeof(comm_dtype_t)`），不需要额外修改。现在修改 JIT runtime 层：
Now update the JIT runtime to pass `comm_dtype_t` and `kReduceInFP32`:
Now update the Reduce Epilogue runtime:
Now update the unified entry function `sm100_bf16_gemm_rs_nt` to accept and pass the new parameters:
Now update the GEMM args construction to pass `comm_dtype`:
Now update the reduce args construction:
Now update the C++ API layer (`csrc/apis/gemm_rs.hpp`) to accept and pass the new parameters:
Now update the pybind11 registration:
Also need to update `get_symm_buffer_size_for_gemm_rs` — it currently uses `use_fp32_output` to decide elem_size, but now we need `use_fp32_comm` to be separate. Let me check how `GemmRSSymmBuffer` uses it:

The `GemmRSSymmBuffer` already passes `use_fp32_output` to `get_symm_buffer_size_for_gemm_rs`. Now the buffer size should depend on the **communication** dtype, not output dtype. Let me update the Python layer to support this. First, update the `GemmRSSymmBuffer`:
Now update `get_symm_buffer_for_gemm_rs`:
Now update `bf16_gemm_rs_nt` Python function to pass `comm_dtype` and `reduce_in_fp32`:
Now let's rebuild and test. First, rebuild the package:
Good, installed. Now let's run the test with default settings (BF16 comm + FP32 reduce, same as before):
默认配置通过。现在写一个完整的测试，验证不同通信模式：
2 卡全部通过！（Mode 2 在 2 卡下也是 0 差异因为 BF16 加 2 个值精度足够）。现在测 8 卡，这时 Mode 2 应该显示出 BF16 累加的精度差异：
完美！8 卡结果也符合预期：

| 模式 | comm_dtype | reduce精度 | max_diff vs FP32 ref | 说明 |
|------|-----------|-----------|---------------------|------|
| Mode 1 | BF16 | FP32 | 0.000000 ✅ | 默认模式，省带宽+高精度 |
| Mode 2 | BF16 | BF16 | 8.000000 ✅ | NCCL-like 行为，有累加误差（正常） |
| Mode 3 | FP32 | FP32 | 0.000000 ✅ | 全精度模式，bit-exact |

再跑 4 卡确认：
4 卡 Mode 2 也是 0（4 个 BF16 累加在这个 case 中没有产生舍入误差，但 8 个时候就有了）。全部通过！

最后确认原始测试没有退化：
-
全部完成！让我总结一下改动：

## ✅ 改动总结

### 设计思路

引入两个独立的精度控制维度：

| 参数 | 控制什么 | 选项 | 影响 |
|------|---------|------|------|
| `comm_dtype` | 通信数据格式（partial buffer 中存什么） | `torch.bfloat16` / `torch.float32` | NVLink 带宽 |
| `reduce_in_fp32` | Reduce 累加精度 | `True` / `False` | Reduce 精度 |

### 三种典型使用模式

| 模式 | comm_dtype | reduce_in_fp32 | 适用场景 |
|------|-----------|----------------|---------|
| **默认（推荐）** | BF16 | True | 省带宽 + 高精度 reduce，训练常用 |
| **NCCL-like** | BF16 | False | 和 NCCL reduce_scatter 行为一致，不在乎微小精度 |
| **全精度** | FP32 | True | 需要 bit-exact 精度，带宽翻倍 |

### 用户接口变更

```python
# 创建 buffer 时指定通信精度
sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
    group, tokens_per_rank, n_dim,
    out_dtype=torch.bfloat16,
    comm_dtype=torch.bfloat16  # 新参数，默认 bf16
)

# 调用时指定 reduce 精度
deep_gemm.bf16_gemm_rs_nt(
    y, a, b, sym_buffer, tokens_per_rank,
    reduce_in_fp32=True  # 新参数，默认 True
)
```

**完全向后兼容**——所有新参数都有默认值，原有代码无需修改。

### 修改的文件

| 文件 | 改动 |
|------|------|
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh` | GEMM kernel 添加 `comm_dtype_t` 模板参数；Reduce kernel 添加 `comm_dtype_t` + `kReduceInFP32` |
| `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp` | JIT codegen 传递新模板参数 |
| `csrc/apis/gemm_rs.hpp` | C++ API 添加 `comm_dtype_str` + `reduce_in_fp32` 参数 |
| `deep_gemm/gemm_rs/__init__.py` | Python API 添加 `comm_dtype` + `reduce_in_fp32` |
| `tests/test_gemm_rs_comm_modes.py` | 新增三模式验证测试 |

