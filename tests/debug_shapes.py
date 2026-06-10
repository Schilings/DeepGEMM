"""Debug: test various shapes to find where correctness breaks."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist

def run(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'

    shapes = [
        (128, 128, 128),   # 1 m, 1 n, 2 k_blocks
        (128, 128, 256),   # 1 m, 1 n, 4 k_blocks
        (128, 128, 512),   # 1 m, 1 n, 8 k_blocks
        (128, 128, 1024),  # 1 m, 1 n, 16 k_blocks
        (128, 128, 2048),  # 1 m, 1 n, 32 k_blocks
        (128, 128, 4096),  # 1 m, 1 n, 64 k_blocks
        (128, 256, 128),   # 1 m, 2 n, 2 k_blocks
        (128, 512, 128),   # 1 m, 4 n, 2 k_blocks
        (256, 128, 128),   # 2 m per rank, 1 n, 2 k_blocks
        (256, 256, 128),   # 2 m per rank, 2 n, 2 k_blocks
        (256, 256, 256),   # 2 m per rank, 2 n, 4 k_blocks
        (512, 128, 128),   # 4 m per rank, 1 n
        (512, 256, 128),   # 4 m per rank, 2 n
        (256, 512, 1024),  # the PASSING one from test_gemm_rs
        (512, 2048, 4096), # the FAILING one from test_gemm_rs
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
