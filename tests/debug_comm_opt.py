"""Debug script to identify where the comm reduce optimization fails."""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist


def run(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokens_per_rank = 256
    n_dim = 2048
    k_dim = 2048
    total_m = tokens_per_rank * num_ranks

    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(a, src=0)
    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(b, src=0)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Reference: since all ranks compute same GEMM, reduce-scatter = N * gemm[my_chunk]
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    torch.cuda.synchronize(local_rank)
    ref = (d_full[rank_idx*tokens_per_rank:(rank_idx+1)*tokens_per_rank, :].float() * num_ranks).bfloat16()

    # Fused
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16)
    y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    diff = (y.float() - ref.float()).abs()
    if rank_idx == 0:
        print(f'\nrank={rank_idx}: max_diff={diff.max().item():.2f}, mean_diff={diff.mean().item():.4f}')
        # Find where max diff is
        max_idx = diff.argmax().item()
        row = max_idx // n_dim
        col = max_idx % n_dim
        print(f'  max_diff at row={row}, col={col}')
        print(f'  y[{row},{col}]={y[row,col].item():.4f}, ref[{row},{col}]={ref[row,col].item():.4f}')

        # Per-row analysis: check which m_block has issues
        block_m = 128
        num_m_blocks = tokens_per_rank // block_m
        for mb in range(num_m_blocks):
            start = mb * block_m
            end = start + block_m
            block_diff = (y[start:end,:].float() - ref[start:end,:].float()).abs()
            print(f'  m_block {mb} (rows {start}-{end}): max_diff={block_diff.max().item():.2f}, mean_diff={block_diff.mean().item():.4f}')

        # Check if y is all zeros or has valid data
        print(f'\n  y stats: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.float().mean().item():.4f}')
        print(f'  ref stats: min={ref.min().item():.4f}, max={ref.max().item():.4f}, mean={ref.float().mean().item():.4f}')

        # Check ratio y/ref to see if some blocks got wrong rank contributions
        # If y == ref/2, it means only 1 rank contributed (instead of 2)
        ratio = y.float() / (ref.float() + 1e-10)
        for mb in range(num_m_blocks):
            start = mb * block_m
            end = start + block_m
            block_ratio = ratio[start:end, :]
            print(f'  m_block {mb} ratio (y/ref): mean={block_ratio.mean().item():.4f}, std={block_ratio.std().item():.4f}')

    sym_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
