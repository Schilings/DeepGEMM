"""Debug: narrow down the failing boundary."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist

def run(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'

    # The last PASS was 256×512×1024. First FAIL is 512×2048×4096.
    # Let's explore in between.
    shapes = [
        # Increase one dimension at a time from 256×512×1024
        (256, 1024, 1024),  # N doubled
        (256, 2048, 1024),  # N x4
        (256, 2048, 2048),  # N x4, K doubled
        (256, 2048, 4096),  # N x4, K x4
        (512, 512, 1024),   # M doubled
        (512, 1024, 1024),  # M doubled, N doubled
        (512, 2048, 1024),  # M doubled, N x4
        (512, 1024, 4096),  # M doubled, K x4
        (512, 512, 4096),   # M doubled, K x4
        (512, 2048, 2048),  # M doubled, N x4, K doubled
        # Also try multicast=2 trigger (m_per_rank >= 128 and compute_waves >= 0.5)
        # waves = total_blocks / (sms/multicast) = (m_blocks * n_blocks * ranks) / (148/mc)
        (512, 2048, 128),   # many m×n blocks, small K
    ]

    if rank_idx == 0:
        print(f"{'Shape':<20} | {'Max Diff':>10} | {'Status'}")
        print(f"{'-'*20}-+-{'-'*10}-+-{'-'*8}")

    for tpr, n, k in shapes:
        total_m = tpr * num_ranks
        torch.manual_seed(42)
        a = torch.randn((total_m, k), dtype=torch.bfloat16, device=device)
        b = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        dist.broadcast(a, src=0)
        dist.broadcast(b, src=0)
        torch.cuda.synchronize(); dist.barrier()

        d_full = torch.zeros((total_m, n), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_nt(a, b, d_full)
        y_ref = torch.zeros((tpr, n), dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(y_ref, d_full, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize(); dist.barrier()

        try:
            sym_buf = deep_gemm.get_symm_buffer_for_gemm_rs(group, tpr, n, out_dtype=torch.bfloat16)
            dist.barrier()
            y_fused = torch.zeros((tpr, n), dtype=torch.bfloat16, device=device)
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buf, tpr, compiled_dims='nk')
            torch.cuda.synchronize(); dist.barrier()

            diff = (y_fused.float() - y_ref.float()).abs().max().item()
            status = '✅ PASS' if diff < 1.0 else '❌ FAIL'
            sym_buf.destroy(); dist.barrier()
        except Exception as e:
            diff = float('nan')
            status = f'❌ ERR: {e}'
            dist.barrier()

        if rank_idx == 0:
            print(f"{tpr}×{n}×{k:<5d}       | {diff:10.4f} | {status}")

    dist.destroy_process_group()

if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
