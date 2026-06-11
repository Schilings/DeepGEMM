"""Single-rank ncu profiling helper for GEMM-RS kernel.
Usage: ncu [options] python scripts/ncu_profile.py [num_gpus] [m_per_rank] [n] [k]
"""
import os, sys, socket, torch, torch.distributed as dist, torch.multiprocessing as mp

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def worker(rank, world_size, port, m_per_rank, n, k):
    os.environ.update({'MASTER_ADDR':'localhost','MASTER_PORT':str(port),
                       'RANK':str(rank),'WORLD_SIZE':str(world_size),'LOCAL_RANK':str(rank)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    group = dist.group.WORLD
    import deep_gemm

    total_m = m_per_rank * world_size
    a = torch.randn((total_m, k), dtype=torch.bfloat16, device=f'cuda:{rank}')
    dist.broadcast(a, src=0)
    b = torch.randn((n, k), dtype=torch.bfloat16, device=f'cuda:{rank}')
    dist.broadcast(b, src=0)
    torch.cuda.synchronize(rank); dist.barrier()

    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(group, m_per_rank, n, out_dtype=torch.bfloat16)
    y = torch.zeros((m_per_rank, n), dtype=torch.bfloat16, device=f'cuda:{rank}')

    # Warmup
    deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, m_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(rank); dist.barrier()

    # Profiled run
    if rank == 0: print(f"Profiling: {m_per_rank}x{n}x{k}, {world_size} GPUs", flush=True)
    deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, m_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(rank); dist.barrier()

    sym_buffer.destroy()
    dist.destroy_process_group()
    os._exit(0)

if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    m_per_rank = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 7168
    port = find_free_port()
    mp.spawn(worker, args=(num_gpus, port, m_per_rank, n, k), nprocs=num_gpus, join=True)
