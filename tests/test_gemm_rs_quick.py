"""
Minimal 2-GPU test to diagnose multi-GPU environment issues.
"""
import os
import sys
import time
import signal

# Set timeout to avoid infinite hang
def timeout_handler(signum, frame):
    print(f"[TIMEOUT] Test timed out after 120 seconds!", flush=True)
    os._exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(120)  # 120 second timeout

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(local_rank: int, num_gpus: int):
    print(f"[Rank {local_rank}] Worker started, PID={os.getpid()}", flush=True)

    try:
        # Step 1: init dist
        port = int(os.getenv('MASTER_PORT', '48123'))
        print(f"[Rank {local_rank}] Initializing dist on port {port}...", flush=True)

        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',
            world_size=num_gpus,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        print(f"[Rank {local_rank}] dist.init_process_group OK", flush=True)

        # Step 2: simple barrier
        dist.barrier()
        print(f"[Rank {local_rank}] barrier OK", flush=True)

        # Step 3: simple allreduce
        t = torch.tensor([local_rank + 1.0], device=f'cuda:{local_rank}')
        dist.all_reduce(t)
        expected = sum(range(1, num_gpus + 1))
        assert abs(t.item() - expected) < 0.01, f"allreduce failed: got {t.item()}, expected {expected}"
        print(f"[Rank {local_rank}] allreduce OK: {t.item()}", flush=True)

        # Step 4: try import deep_gemm and basic GEMM
        import deep_gemm
        print(f"[Rank {local_rank}] deep_gemm imported OK", flush=True)

        # Step 5: try a small standard GEMM
        M, N, K = 256, 512, 1024
        a = torch.randn((M, K), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        b = torch.randn((N, K), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        c = torch.zeros((M, N), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        deep_gemm.bf16_gemm_nt(a, b, c)
        torch.cuda.synchronize(local_rank)
        print(f"[Rank {local_rank}] bf16_gemm_nt OK, c.abs().mean()={c.abs().mean().item():.4f}", flush=True)

        # Step 6: try GEMM-RS
        tokens_per_rank = 256
        total_m = tokens_per_rank * num_gpus
        n_dim, k_dim = 512, 1024

        a_full = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        dist.broadcast(a_full, src=0)
        b_mat = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        dist.broadcast(b_mat, src=0)

        group = dist.new_group(list(range(num_gpus)))
        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )
        print(f"[Rank {local_rank}] sym_buffer created OK", flush=True)

        y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        deep_gemm.bf16_gemm_rs_nt(y, a_full, b_mat, sym_buffer, tokens_per_rank, compiled_dims='nk')
        torch.cuda.synchronize(local_rank)
        dist.barrier()
        print(f"[Rank {local_rank}] bf16_gemm_rs_nt OK, y.abs().mean()={y.abs().mean().item():.4f}", flush=True)

        sym_buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        print(f"[Rank {local_rank}] ALL TESTS PASSED! ✅", flush=True)

        # Force exit to avoid NCCL background threads holding ports
        time.sleep(1)
        os._exit(0)

    except Exception as e:
        print(f"[Rank {local_rank}] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        os._exit(1)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"=== Quick GEMM-RS diagnostic test with {num_gpus} GPUs ===", flush=True)
    print(f"    MASTER_PORT={os.getenv('MASTER_PORT', '48123')}", flush=True)
    print(f"    PID={os.getpid()}", flush=True)

    mp.spawn(worker, args=(num_gpus,), nprocs=num_gpus, join=True)
    print("=== Main process: spawn completed ===", flush=True)
