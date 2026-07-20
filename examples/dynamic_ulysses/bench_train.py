"""Real training benchmark: dynamic SP vs static SP with DP parallelism.

Key fix: microbatches with the same SP size run their DP copies in PARALLEL
(each rank processes one sequence). Different SP sizes run sequentially.

This models the realistic scenario where DP copies overlap, so wall-clock =
max(DP copy time), not sum(all microbatch times).

Run: python examples/dynamic_ulysses/bench_train.py 8
"""
import os, sys, time, torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader


class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward_sp(self, x_local, sp_group, sp_size, local_seq):
        bs = 1
        local_nh = self.num_heads // sp_size
        seq = sp_size * local_seq

        q = self.q(x_local).view(bs, local_seq, sp_size, local_nh, self.head_dim)
        k = self.k(x_local).view(bs, local_seq, sp_size, local_nh, self.head_dim)
        v = self.v(x_local).view(bs, local_seq, sp_size, local_nh, self.head_dim)

        def scatter(t):
            send = t.permute(2, 0, 1, 3, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=sp_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(bs, seq, local_nh, self.head_dim)

        q, k, v = scatter(q), scatter(k), scatter(v)

        try:
            from flash_attn import flash_attn_func
            o = flash_attn_func(q, k, v, causal=True)
        except ImportError:
            attn = torch.einsum('bshd,bthd->bsht', q.float(), k.float()) * self.scale
            mask = torch.triu(torch.full((seq, seq), float('-inf')), 1)
            attn = attn + mask
            attn = torch.softmax(attn, dim=-1)
            o = torch.einsum('bsht,bthd->bshd', attn, v.float()).to(q.dtype)

        o = o.view(bs, sp_size, local_seq, local_nh, self.head_dim)
        send = o.permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=sp_group)
        o = recv.permute(1, 2, 0, 3, 4).reshape(bs * local_seq, -1)
        return self.o(o)

    def forward_dp(self, x_local, local_seq):
        """SP=1: no A2A, standard attention."""
        bs = 1
        q = self.q(x_local).view(bs, local_seq, self.num_heads, self.head_dim)
        k = self.k(x_local).view(bs, local_seq, self.num_heads, self.head_dim)
        v = self.v(x_local).view(bs, local_seq, self.num_heads, self.head_dim)
        try:
            from flash_attn import flash_attn_func
            o = flash_attn_func(q, k, v, causal=True)
        except ImportError:
            attn = torch.einsum('bshd,bthd->bsht', q.float(), k.float()) * self.scale
            mask = torch.triu(torch.full((local_seq, local_seq), float('-inf')), 1)
            attn = attn + mask
            attn = torch.softmax(attn, dim=-1)
            o = torch.einsum('bsht,bthd->bshd', attn, v.float()).to(q.dtype)
        o = o.reshape(local_seq, -1)
        return self.o(o)


def run_benchmark(rank, ng, port, dim, num_heads, head_dim, num_layers, model, gm, loader,
                  scenario_name, seq_lengths):
    dev = torch.device(f'cuda:{rank}')
    total_tokens = sum(seq_lengths)

    # --- Static SP=8 ---
    # All sequences at SP=ng, 1 DP copy (sequential)
    info8 = gm.get_groups(ng)
    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
    t0 = time.time()
    for s in seq_lengths:
        aligned = ((s + 127) // 128) * 128
        local_seq = aligned // ng
        x = torch.randn(local_seq, dim, dtype=torch.bfloat16, device=dev, requires_grad=True)
        for layer in model:
            x = layer.forward_sp(x, info8.sp_group, ng, local_seq)
        x.sum().backward()
    torch.cuda.synchronize()
    t_static = (time.time() - t0) * 1000

    # --- Dynamic SP ---
    # Schedule: group by SP size, DP copies run in parallel
    mbs = loader.schedule(seq_lengths)

    # Group MBs by SP size
    by_sp = {}
    for mb in mbs:
        by_sp.setdefault(mb.sp_size, []).append(mb)

    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
    t0 = time.time()

    # Process SP sizes sequentially, DP copies in parallel
    for sp_size in sorted(by_sp.keys(), reverse=True):
        info = gm.get_groups(sp_size)
        dp_size = ng // sp_size
        mbs_at_sp = by_sp[sp_size]

        # Each rank is in one DP copy. Process the sequence assigned to this rank's DP copy.
        # If there are more sequences than DP copies, process multiple rounds.
        for round_idx in range(0, len(mbs_at_sp), dp_size):
            # This round: DP copies [round_idx : round_idx + dp_size]
            dp_idx = rank % dp_size  # which DP copy this rank belongs to
            mb_idx = round_idx + dp_idx
            if mb_idx < len(mbs_at_sp):
                mb = mbs_at_sp[mb_idx]
                x = torch.randn(mb.local_seq, dim, dtype=torch.bfloat16, device=dev, requires_grad=True)
                for layer in model:
                    if sp_size == 1:
                        x = layer.forward_dp(x, mb.local_seq)
                    else:
                        x = layer.forward_sp(x, info.sp_group, sp_size, mb.local_seq)
                x.sum().backward()
            # Barrier to sync this round across all ranks
            dist.barrier()

    torch.cuda.synchronize()
    t_dynamic = (time.time() - t0) * 1000

    speedup = t_static / t_dynamic if t_dynamic > 0 else 0
    sp_dist = {}
    for mb in mbs:
        sp_dist[mb.sp_size] = sp_dist.get(mb.sp_size, 0) + 1

    if rank == 0:
        tps_s = total_tokens / (t_static / 1000) if t_static > 0 else 0
        tps_d = total_tokens / (t_dynamic / 1000) if t_dynamic > 0 else 0
        print(f'{scenario_name:<20} {"Static SP=8":<12} {t_static:<10.1f} {total_tokens:<10} {tps_s:<12.0f} {t_static/t_dynamic:<8.3f}x')
        print(f'{"":<20} {"Dynamic":<12} {t_dynamic:<10.1f} {total_tokens:<10} {tps_d:<12.0f} {"1.000x":<8}')
        print(f'  SP schedule: {sp_dist}')
        print()

    return speedup


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')

    import time

    gm = DynamicSPGroupManager(ng)
    loader = BalancedDataLoader(ng)

    dim = 5120
    num_heads = 40
    head_dim = 128
    num_layers = 4

    torch.manual_seed(42)
    model = nn.ModuleList([SimpleAttention(dim, num_heads, head_dim) for _ in range(num_layers)])
    model.to(device=dev, dtype=torch.bfloat16)

    scenarios = {
        "uniform_8K": [8192] * ng,
        "mixed": [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048],
        "all_short": [2048] * ng,
        "bimodal": [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048],
        "one_long_tail": [32768] + [2048] * (ng - 1),
    }

    if rank == 0:
        print(f'\n{"="*100}')
        print(f'Real Training Benchmark (B300 x{ng}, dim={dim}, heads={num_heads}, layers={num_layers})')
        print(f'DP copies run in PARALLEL within each SP size group')
        print(f'{"="*100}')
        print(f'{"Scenario":<20} {"Strategy":<12} {"Wall(ms)":<10} {"Tokens":<10} {"tok/s":<12} {"Speedup":<8}')
        print('-' * 100)

    speedups = []
    for name, seqs in scenarios.items():
        sp = run_benchmark(rank, ng, port, dim, num_heads, head_dim, num_layers,
                          model, gm, loader, name, seqs)
        speedups.append(sp)

    if rank == 0:
        import math
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f'{"="*100}')
        print(f'Geometric mean speedup (Dynamic vs Static SP=8): {geo:.3f}x')
        print(f'{"="*100}\n')

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
