"""Quick correctness test for dual-kernel GEMM+RS v3 (stream-level overlap)"""
import os, sys, time, torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist

SHAPES = [
    (512, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 7168, 2048),
    (2048, 4096, 7168),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
]

def compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank):
    total_m = tokens_per_rank * num_ranks
    n_dim = b.shape[0]
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    torch.cuda.synchronize(local_rank)
    all_d = [torch.empty_like(d_full) for _ in range(num_ranks)]
    dist.all_gather(all_d, d_full)
    torch.cuda.synchronize(local_rank)
    start_row = rank_idx * tokens_per_rank
    end_row = start_row + tokens_per_rank
    ref_fp32 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    for r in range(num_ranks):
        ref_fp32 += all_d[r][start_row:end_row, :].float()
    ref = ref_fp32.bfloat16()
    del all_d, d_full, ref_fp32
    return ref

def run_test(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    if rank_idx == 0:
        print(f"\n{'='*80}")
        print(f"  BF16 GEMM-RS v3 (Dual-Kernel Overlap) Correctness Test: {num_ranks} GPUs")
        print(f"  Testing {len(SHAPES)} shapes")
        print(f"{'='*80}\n")
        print(f"{'Shape':<22} | {'Max Diff':>9} {'Mean Diff':>10} {'Rel Err':>9} | Status")
        print(f"{'-'*22} | {'-'*9} {'-'*10} {'-'*9} | {'-'*8}")

    all_passed = True
    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_ranks
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        dist.broadcast(a, src=0)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        dist.broadcast(b, src=0)
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        ref = compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank)
        dist.barrier()

        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16)

        # Test v3 dual-kernel overlap
        y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
        deep_gemm.bf16_gemm_rs_nt_v3(y, a, b, sym_buffer, tokens_per_rank)
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        max_diff = (y.float() - ref.float()).abs().max().item()
        mean_diff = (y.float() - ref.float()).abs().mean().item()
        ref_abs_mean = ref.float().abs().mean().item()
        rel_error = mean_diff / max(ref_abs_mean, 1e-8)
        max_rel = 0.01 * num_ranks
        passed = rel_error < max_rel

        if rank_idx == 0:
            status = "PASS" if passed else "FAIL"
            print(f"{tokens_per_rank}x{n_dim}x{k_dim:<10} | {max_diff:>9.4f} {mean_diff:>10.6f} {rel_error:>9.6f} | {status}")

        if not passed:
            all_passed = False

        sym_buffer.destroy()
        dist.barrier()

    if rank_idx == 0:
        print(f"\n{'='*80}")
        print(f"  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print(f"{'='*80}")

if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    os.environ.setdefault('MASTER_PORT', str(port))
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
