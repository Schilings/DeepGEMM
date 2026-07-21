"""Controlled benchmark: Dynamic SP vs Static SP baselines.

DESIGN PRINCIPLE — control variables:
  The ONLY independent variable is the SP scheduling strategy.
  Everything else is held identical between arms:
    1. Same attention implementation (UlyssesScatterAttn, single code path)
    2. Same model weights & shapes
    3. Same input sequences
    4. Same DP parallelism opportunity (DP copies run in parallel rounds)

  Baselines (all use the SAME UlyssesScatterAttn code path):
    - Static-SP8:   every sequence uses SP=8, processed sequentially (no DP)
    - Static-SP4×2: every sequence uses SP=4, 2 DP copies run in parallel each round
    - Static-SP2×4: every sequence uses SP=2, 4 DP copies run in parallel each round
    - Static-SP1×8: every sequence uses SP=1, 8 DP copies run in parallel each round (pure DP)

  Dynamic SP:
    - Each sequence assigned an SP size by BalancedDataLoader
    - Sequences grouped by SP size; DP copies within a group run in parallel
    - Different SP groups run sequentially

  This isolates the effect of *dynamic SP selection* from confounders like
  different attention code paths or unequal DP parallelism.

Run: python examples/dynamic_ulysses/bench_train.py [ngpus]
"""
import os, sys, time, math, socket
import torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader, Microbatch


# ----------------------------------------------------------------------------
# Single attention implementation shared by ALL arms (control variable)
# ----------------------------------------------------------------------------
class UlyssesScatterAttn(nn.Module):
    """Ulysses scatter-gather attention.

    One unified code path: when sp_size==1 the A2A is a no-op, so this works
    for both pure-DP (SP=1) and Ulysses (SP>1) without branching into a
    different implementation. This guarantees baseline and dynamic arms run
    literally the same kernel code.
    """

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

    def forward(self, x_local, sp_group, sp_size, local_seq):
        """x_local: [local_seq, dim]. Returns [local_seq, dim]."""
        bs = 1
        local_nh = self.num_heads // sp_size
        full_seq = sp_size * local_seq
        hd = self.head_dim

        q = self.q(x_local).view(bs, local_seq, sp_size, local_nh, hd)
        k = self.k(x_local).view(bs, local_seq, sp_size, local_nh, hd)
        v = self.v(x_local).view(bs, local_seq, sp_size, local_nh, hd)

        # A2A: scatter heads, gather sequence. For sp_size==1 this is identity.
        def scatter_heads(t):
            if sp_size == 1:
                return t.reshape(bs, local_seq, local_nh, hd)
            send = t.permute(2, 0, 1, 3, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=sp_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(bs, full_seq, local_nh, hd)

        q, k, v = scatter_heads(q), scatter_heads(k), scatter_heads(v)

        # Attention (fallback to torch SDPA if flash_attn unavailable)
        try:
            from flash_attn import flash_attn_func
            o = flash_attn_func(q, k, v, causal=True)
        except ImportError:
            attn = torch.einsum('bshd,bthd->bsht', q.float(), k.float()) * self.scale
            mask = torch.triu(torch.full((full_seq, full_seq), float('-inf'),
                                         device=x_local.device), 1)
            attn = attn + mask
            attn = torch.softmax(attn, dim=-1)
            o = torch.einsum('bsht,bthd->bshd', attn, v.float()).to(q.dtype)

        # A2A inverse: scatter sequence, gather heads
        def gather_heads(t):
            if sp_size == 1:
                return t.reshape(local_seq, -1)
            send = t.view(bs, sp_size, local_seq, local_nh, hd).permute(1, 0, 2, 3, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=sp_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(local_seq, -1)

        o = gather_heads(o)
        return self.o(o)


# ----------------------------------------------------------------------------
# Runner: process a list of microbatches with a fixed static SP size.
# DP copies run in parallel rounds — same parallelism model as Dynamic SP.
# ----------------------------------------------------------------------------
def run_static(model, gm, mbs, sp_size, dim, dev):
    """All microbatches use the same SP size; DP copies run in parallel."""
    info = gm.get_groups(sp_size)
    dp_size = gm.world_size // sp_size

    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
    t0 = time.time()

    # Process in rounds of dp_size parallel copies
    for round_idx in range(0, len(mbs), dp_size):
        dp_idx = rank_global % dp_size  # which DP copy this rank belongs to
        mb_idx = round_idx + dp_idx
        if mb_idx < len(mbs):
            mb = mbs[mb_idx]
            # NOTE: local_seq must be consistent with this sp_size.
            # We re-derive local_seq = full_seq / sp_size for static arms.
            full_seq = mb.seq_len
            local_seq = full_seq // sp_size
            x = torch.randn(local_seq, dim, dtype=torch.bfloat16,
                            device=dev, requires_grad=True)
            for layer in model:
                x = layer.forward(x, info.sp_group, sp_size, local_seq)
            x.sum().backward()
        dist.barrier()

    torch.cuda.synchronize()
    return (time.time() - t0) * 1000


# ----------------------------------------------------------------------------
# Runner: Dynamic SP — each microbatch uses its assigned SP size.
# ----------------------------------------------------------------------------
def run_dynamic(model, gm, mbs, dim, dev):
    # Group microbatches by SP size; DP copies within a group run in parallel.
    by_sp = {}
    for mb in mbs:
        by_sp.setdefault(mb.sp_size, []).append(mb)

    for p in model.parameters():
        p.grad = None
    torch.cuda.synchronize()
    t0 = time.time()

    # Process SP groups sequentially (largest SP first — longest jobs first)
    for sp_size in sorted(by_sp.keys(), reverse=True):
        info = gm.get_groups(sp_size)
        dp_size = gm.world_size // sp_size
        group_mbs = by_sp[sp_size]

        for round_idx in range(0, len(group_mbs), dp_size):
            dp_idx = rank_global % dp_size
            mb_idx = round_idx + dp_idx
            if mb_idx < len(group_mbs):
                mb = group_mbs[mb_idx]
                x = torch.randn(mb.local_seq, dim, dtype=torch.bfloat16,
                                device=dev, requires_grad=True)
                for layer in model:
                    x = layer.forward(x, info.sp_group, sp_size, mb.local_seq)
                x.sum().backward()
            dist.barrier()

    torch.cuda.synchronize()
    return (time.time() - t0) * 1000


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
rank_global = 0


def run(rank, ng, port):
    global rank_global
    rank_global = rank

    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')

    gm = DynamicSPGroupManager(ng)
    loader = BalancedDataLoader(ng)

    dim = 5120
    num_heads = 40
    head_dim = 128
    num_layers = 4

    torch.manual_seed(42)
    model = nn.ModuleList([UlyssesScatterAttn(dim, num_heads, head_dim)
                           for _ in range(num_layers)])
    model.to(device=dev, dtype=torch.bfloat16)

    scenarios = {
        "uniform_8K":       [8192] * ng,
        "uniform_32K":      [32768] * (ng // 4) if ng >= 4 else [32768],
        "mixed":            [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048][:ng],
        "all_short_2K":     [2048] * ng,
        "bimodal":          [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048][:ng],
        "one_long_tail":    [32768] + [2048] * (ng - 1),
    }

    if rank == 0:
        print(f'\n{"="*110}')
        print(f'Controlled Benchmark: Dynamic SP vs Static SP baselines')
        print(f'  B300 x{ng}, dim={dim}, heads={num_heads}, head_dim={head_dim}, layers={num_layers}')
        print(f'  SAME attention code path for all arms (UlyssesScatterAttn)')
        print(f'  DP copies run in parallel rounds in ALL arms (controlled)')
        print(f'{"="*110}')
        hdr = (f'{"Scenario":<18} {"SP8":>9} {"SP4x2":>9} {"SP2x4":>9} {"SP1x8":>9} '
               f'{"Dynamic":>9} {"Best Static":>12} {"Dyn/Best":>9}')
        print(hdr)
        print('-' * 110)

    speedups = []
    for name, seqs in scenarios.items():
        # Build microbatches for the dynamic schedule
        mbs_dyn = loader.schedule(seqs)
        # For static arms, we use the same sequences but force a single SP size.
        # We build "static microbatches" that just carry seq_len (sp assigned at runtime).
        from dataclasses import replace
        mbs_static = [Microbatch(sp_size=1, seq_len=s,
                                 local_seq=s, dp_copy=0, tokens=s)
                      for s in seqs]

        total_tokens = sum(seqs)

        # Static baselines (each uses a single SP size for all sequences)
        t_sp8  = run_static(model, gm, mbs_static, 8,  dim, dev) if ng >= 8 else float('inf')
        t_sp4  = run_static(model, gm, mbs_static, 4,  dim, dev) if ng >= 4 else float('inf')
        t_sp2  = run_static(model, gm, mbs_static, 2,  dim, dev)
        t_sp1  = run_static(model, gm, mbs_static, 1,  dim, dev)

        # Dynamic SP
        t_dyn = run_dynamic(model, gm, mbs_dyn, dim, dev)

        # Best static = fastest among the static baselines
        static_times = {'SP8': t_sp8, 'SP4x2': t_sp4, 'SP2x4': t_sp2, 'SP1x8': t_sp1}
        best_name = min(static_times, key=static_times.get)
        best_time = static_times[best_name]
        speedup = best_time / t_dyn if t_dyn > 0 else 0
        speedups.append(speedup)

        if rank == 0:
            print(f'{name:<18} {t_sp8:>8.1f}m {t_sp4:>8.1f}m {t_sp2:>8.1f}m {t_sp1:>8.1f}m '
                  f'{t_dyn:>8.1f}m {best_name:>12} {speedup:>8.3f}x')

    if rank == 0:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f'{"="*110}')
        print(f'Geometric mean speedup (Dynamic vs Best Static): {geo:.3f}x')
        print(f'{"="*110}\n')

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
