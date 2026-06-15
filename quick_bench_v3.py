"""Quick benchmark for V3 dual-kernel GEMM+RS - with barrier reset fix"""
import torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist

def bench(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f"cuda:{local_rank}"
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    shapes = [
        (2048, 7168, 4096),
        (4096, 7168, 2048),
        (4096, 7168, 4096),
        (8192, 7168, 2048),
        (8192, 7168, 4096),
        (16384, 7168, 2048),
    ]

    if rank_idx == 0:
        print(f"\n  {'Shape':<22} | {'Separate':>10} {'V3 Dual':>10} | {'Sep TFLOPS':>10} {'V3 TFLOPS':>10} | {'Speedup':>8}")
        print(f"  {'-'*22}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*8}")

    for m_per_rank, n, k in shapes:
        total_m = m_per_rank * num_ranks

        a = torch.randn((total_m, k), dtype=torch.bfloat16, device=device)
        b = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        y_v3 = torch.zeros((m_per_rank, n), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_v3)
        d_full = torch.zeros((total_m, n), dtype=torch.bfloat16, device=device)

        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(group, m_per_rank, n, out_dtype=torch.bfloat16)
        dist.barrier()

        def run_v3():
            # Reset NVLink barrier state (first 32 bytes) before each call
            sym_buffer.buffer[:32].zero_()
            deep_gemm.bf16_gemm_rs_nt_v3(y_v3, a, b, sym_buffer, m_per_rank, compiled_dims="nk")

        # Warmup V3
        for _ in range(3):
            run_v3()
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark V3
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run_v3()
        end.record()
        torch.cuda.synchronize()
        v3_time = start.elapsed_time(end) / 20  # ms

        # Warmup Separate
        for _ in range(3):
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark Separate
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        start2.record()
        for _ in range(20):
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)
        end2.record()
        torch.cuda.synchronize()
        sep_time = start2.elapsed_time(end2) / 20  # ms

        if rank_idx == 0:
            tflops_sep = 2.0 * total_m * n * k / (sep_time * 1e-3) / 1e12
            tflops_v3 = 2.0 * total_m * n * k / (v3_time * 1e-3) / 1e12
            speedup = sep_time / v3_time
            print(f"  {m_per_rank}x{n}x{k:<5} | {sep_time*1000:>8.1f}u {v3_time*1000:>8.1f}u | "
                  f"{tflops_sep:>8.1f}T {tflops_v3:>8.1f}T | {speedup:>7.3f}x")

        sym_buffer.destroy()
        dist.barrier()

    if rank_idx == 0:
        print()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    num_gpus = 8
    mp.spawn(bench, args=(num_gpus,), nprocs=num_gpus, join=True)
