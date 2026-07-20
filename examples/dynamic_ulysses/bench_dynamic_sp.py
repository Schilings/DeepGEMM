"""Benchmark: dynamic SP vs static SP — realistic compute+comm model.

Model includes:
- GEMM FLOPs: 12 * S * H^2 * L / SP (QKV + O + FFN, per-GPU)
- Attention FLOPs: 4 * S^2 * H * L / SP (QK + AV, per-GPU)
- A2A communication: 2 * (alpha + beta * S * H / SP) per layer (PRE + POST)
  - SP=1: zero (no communication)
  - SP>1: latency + bandwidth cost

B300 constants (estimated):
- GEMM throughput: ~1500 TFLOPS (bf16)
- A2A latency: ~20 µs (NVLink)
- A2A bandwidth: ~300 GB/s (per-GPU, bidirectional)

Run: python examples/dynamic_ulysses/bench_dynamic_sp.py 8
"""
import os, sys, math, torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader


# B300 constants
GEMM_TFLOPS = 1500.0       # bf16 GEMM throughput (TFLOPS)
ATTN_TFLOPS = 800.0        # FA4 throughput (TFLOPS)
A2A_LATENCY_US = 20.0      # NVLink A2A latency (µs)
A2A_BW_GB_S = 300.0        # NVLink bandwidth (GB/s per GPU)
NUM_LAYERS = 40
HIDDEN = 5120


def estimate_step_time(seq_lengths, sp_size, world_size):
    """Estimate wall-clock time (seconds) for processing all sequences at given SP size.

    Each microbatch uses all `world_size` GPUs organized into SP groups.
    DP copies run in parallel, so wall-clock = max DP copy time.
    """
    dp_size = world_size // sp_size
    # Distribute sequences across DP copies (round-robin)
    dp_times = [0.0] * dp_size
    for i, s in enumerate(seq_lengths):
        aligned = ((s + 127) // 128) * 128
        local_seq = aligned // sp_size

        # GEMM time: 12 * S * H^2 * L / SP (bf16)
        gemm_flops = 12 * aligned * HIDDEN * HIDDEN * NUM_LAYERS / sp_size
        gemm_time = gemm_flops / (GEMM_TFLOPS * 1e12)

        # Attention time: 4 * S^2 * H * L / SP
        attn_flops = 4 * aligned * aligned * HIDDEN * NUM_LAYERS / sp_size
        attn_time = attn_flops / (ATTN_TFLOPS * 1e12)

        # A2A communication time (PRE + POST, per layer)
        if sp_size > 1:
            a2a_bytes = 2 * aligned * HIDDEN * 2 * NUM_LAYERS / sp_size  # bf16=2 bytes, PRE+POST
            a2a_time = (A2A_LATENCY_US * 1e-6 * 2 * NUM_LAYERS  # latency per call
                        + a2a_bytes / (A2A_BW_GB_S * 1e9))
        else:
            a2a_time = 0.0

        mb_time = gemm_time + attn_time + a2a_time
        dp_copy = i % dp_size
        dp_times[dp_copy] += mb_time

    return max(dp_times)


def estimate_dynamic_time(microbatches, world_size):
    """Estimate wall-clock time for dynamic SP schedule."""
    # Group by SP size — microbatches with same SP size can share DP copies
    # For simplicity: process sequentially (conservative, no pipeline overlap)
    # In practice, DP copies run in parallel within each SP size

    # Group MBs by sp_size
    by_sp = {}
    for mb in microbatches:
        by_sp.setdefault(mb.sp_size, []).append(mb)

    total_time = 0.0
    for sp_size, mbs in by_sp.items():
        dp_size = world_size // sp_size
        dp_times = [0.0] * dp_size
        for i, mb in enumerate(mbs):
            local_seq = mb.local_seq
            # Same model as above
            gemm_flops = 12 * mb.seq_len * HIDDEN * HIDDEN * NUM_LAYERS / sp_size
            gemm_time = gemm_flops / (GEMM_TFLOPS * 1e12)

            attn_flops = 4 * mb.seq_len * mb.seq_len * HIDDEN * NUM_LAYERS / sp_size
            attn_time = attn_flops / (ATTN_TFLOPS * 1e12)

            if sp_size > 1:
                a2a_bytes = 2 * mb.seq_len * HIDDEN * 2 * NUM_LAYERS / sp_size
                a2a_time = (A2A_LATENCY_US * 1e-6 * 2 * NUM_LAYERS
                            + a2a_bytes / (A2A_BW_GB_S * 1e9))
            else:
                a2a_time = 0.0

            mb_time = gemm_time + attn_time + a2a_time
            dp_copy = i % dp_size
            dp_times[dp_copy] += mb_time

        total_time += max(dp_times)  # different SP sizes run sequentially

    return total_time


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)

    loader = BalancedDataLoader(ng)

    scenarios = {
        "uniform_8K": [8192] * ng,
        "uniform_32K": [32768] * (ng // 4),
        "mixed_realistic": [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048],
        "one_long_tail": [32768] + [2048] * (ng - 1),
        "bimodal": [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048],
        "all_short": [2048] * ng,
    }

    if rank == 0:
        print(f'\n{"="*110}')
        print(f'Dynamic SP vs Static SP Benchmark (B300 x{ng}, H={HIDDEN}, L={NUM_LAYERS})')
        print(f'Model: GEMM={GEMM_TFLOPS}T, ATTN={ATTN_TFLOPS}T, A2A_lat={A2A_LATENCY_US}us, A2A_bw={A2A_BW_GB_S}GB/s')
        print(f'{"="*110}')
        print(f'{"Scenario":<20} {"Strategy":<12} {"Wall(s)":<10} {"Tokens":<10} {"tok/s":<12} {"Speedup":<10}')
        print('-' * 110)

    all_speedups = []

    for name, seqs in scenarios.items():
        total_tokens = sum(s for s in seqs)
        mbs = loader.schedule(seqs)
        sp_dist = {}
        for mb in mbs:
            sp_dist[mb.sp_size] = sp_dist.get(mb.sp_size, 0) + 1

        # Static SP=8
        t_s8 = estimate_step_time(seqs, ng, ng)
        # Static SP=4
        t_s4 = estimate_step_time(seqs, ng // 2, ng)
        # Dynamic
        t_dyn = estimate_dynamic_time(mbs, ng)

        speedup_8 = t_s8 / t_dyn if t_dyn > 0 else 0
        speedup_4 = t_s4 / t_dyn if t_dyn > 0 else 0
        all_speedups.append((name, speedup_8, speedup_4))

        if rank == 0:
            for label, t, sp in [("Static SP=8", t_s8, ng), ("Static SP=4", t_s4, ng//2)]:
                tps = total_tokens / t if t > 0 else 0
                print(f'{name:<20} {label:<12} {t:<10.4f} {total_tokens:<10} {tps:<12.0f} {t/t_dyn:<10.3f}x')
            tps = total_tokens / t_dyn if t_dyn > 0 else 0
            print(f'{"":<20} {"Dynamic":<12} {t_dyn:<10.4f} {total_tokens:<10} {tps:<12.0f} {"1.000x":<10}')
            print(f'  SP schedule: {sp_dist}')
            print()

    if rank == 0:
        geo_8 = math.exp(sum(math.log(s[1]) for s in all_speedups) / len(all_speedups))
        geo_4 = math.exp(sum(math.log(s[2]) for s in all_speedups) / len(all_speedups))
        print(f'{"="*110}')
        print(f'Geometric mean speedup of Dynamic vs Static SP=8: {geo_8:.3f}x')
        print(f'Geometric mean speedup of Dynamic vs Static SP=4: {geo_4:.3f}x')
        print(f'{"="*110}\n')

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
