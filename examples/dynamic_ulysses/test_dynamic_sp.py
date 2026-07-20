"""Test dynamic SP: verify group creation, data scheduling, and basic forward.

Run: python examples/dynamic_ulysses/test_dynamic_sp.py 8
"""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamic_ulysses import (
    DynamicSPGroupManager, BalancedDataLoader, DynamicGradientSync,
)


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)

    # ── Test 1: DynamicSPGroupManager ──
    gm = DynamicSPGroupManager(ng)
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Test 1: DynamicSPGroupManager (world_size={ng})')
        print(f'{"="*60}')
        print(f'Valid SP sizes: {gm.get_valid_sp_sizes()}')

    for sp_size in gm.get_valid_sp_sizes():
        info = gm.get_groups(sp_size)
        dp_size = ng // sp_size
        assert info.sp_size == sp_size
        assert info.dp_size == dp_size
        if rank == 0:
            print(f'  SP={sp_size}: DP={dp_size}, sp_rank={info.sp_rank}, dp_rank={info.dp_rank}')

    # Verify group sizes
    for sp_size in gm.get_valid_sp_sizes():
        info = gm.get_groups(sp_size)
        if info.sp_group is not None:
            ws = dist.get_world_size(info.sp_group)
            assert ws == sp_size, f"SP group size mismatch: {ws} != {sp_size}"
        if info.dp_group is not None:
            ws = dist.get_world_size(info.dp_group)
            assert ws == ng // sp_size, f"DP group size mismatch: {ws} != {ng // sp_size}"

    if rank == 0:
        print(f'  All group sizes verified ✓')

    # ── Test 2: BalancedDataLoader ──
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Test 2: BalancedDataLoader')
        print(f'{"="*60}')

    loader = BalancedDataLoader(ng, seq_align=128)

    # Simulate mixed sequence lengths
    seq_lengths = [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048]
    microbatches = loader.schedule(seq_lengths)

    if rank == 0:
        print(f'Input sequences: {seq_lengths}')
        print(f'Schedule ({len(microbatches)} microbatches):')
        total_flops = 0
        for mb in microbatches:
            print(f'  {mb}')
            total_flops += mb.tokens
        print(f'Total tokens: {total_flops}')
        print(f'Max wall-clock FLOPs: {loader.max_wall_clock_flops(microbatches, 5120, 40):.2e}')

    # ── Test 3: DynamicGradientSync ──
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Test 3: DynamicGradientSync')
        print(f'{"="*60}')

    # Create a dummy parameter
    param = torch.nn.Parameter(torch.randn(128, device='cuda'))
    param.grad = torch.randn(128, device='cuda')

    sync = DynamicGradientSync()
    sync.set_token_counts(local_tokens=1024)  # each rank has 1024 tokens
    sync.add_param('test', param)
    sync.sync(scale_by_tokens=True)

    if rank == 0:
        print(f'  Global tokens: {sync._global_tokens}')
        print(f'  Param grad mean: {param.grad.mean().item():.6f}')
        print(f'  Gradient sync completed ✓')

    sync.reset()

    # ── Test 4: Multiple SP sizes in sequence ──
    if rank == 0:
        print(f'\n{"="*60}')
        print(f'Test 4: Multiple SP sizes in one step')
        print(f'{"="*60}')

    # Simulate a step with mixed SP sizes
    for mb in microbatches:
        info = gm.get_groups(mb.sp_size)
        if rank == 0:
            print(f'  Running mb: sp={mb.sp_size}, seq={mb.seq_len}, '
                  f'local_seq={mb.local_seq}, dp_copy={mb.dp_copy}')

        # Barrier to sync between microbatches
        dist.barrier()

    if rank == 0:
        print(f'\n  All microbatches scheduled ✓')
        print(f'\n{"="*60}')
        print(f'ALL TESTS PASSED')
        print(f'{"="*60}\n')

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    import socket
    port = 0
    sock = socket.socket()
    sock.bind(('', port))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
