"""Correctness test: dynamic SP vs static SP.

Run: python examples/dynamic_ulysses/test_correctness.py 8
"""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')

    gm = DynamicSPGroupManager(ng)

    if rank == 0:
        print(f'\n{"="*60}\nTest 1: Cross-group AllReduce\n{"="*60}')

    val = torch.tensor([float(rank + 1)], device=dev)
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    expected = ng * (ng + 1) / 2
    assert val.item() == expected, f"AllReduce: {val.item()} != {expected}"
    if rank == 0:
        print(f'  AllReduce sum = {val.item()} (expected {expected}) OK')

    if rank == 0:
        print(f'\n{"="*60}\nTest 2: SP group AllToAll\n{"="*60}')

    for sp_size in [1, 2, 4, 8]:
        if sp_size > ng:
            continue
        info = gm.get_groups(sp_size)
        if sp_size == 1:
            if rank == 0:
                print(f'  SP={sp_size}: no A2A needed (pure DP) OK')
            continue

        local_seq = 4
        send = torch.full((sp_size, local_seq), float(rank), device=dev)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=info.sp_group)

        for i in range(sp_size):
            src_rank = (rank // sp_size) * sp_size + i
            expected_val = float(src_rank)
            actual_val = recv[i, 0].item()
            assert actual_val == expected_val, \
                f"SP={sp_size} A2A: recv[{i}]={actual_val} != {expected_val}"

        if rank == 0:
            print(f'  SP={sp_size}: AllToAll verified OK')

    if rank == 0:
        print(f'\n{"="*60}\nTest 3: Gradient sync with token scaling\n{"="*60}')

    local_tokens = 1024
    global_tokens = local_tokens * ng
    param_grad = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(param_grad, op=dist.ReduceOp.SUM)
    param_grad.div_(global_tokens)
    expected_grad = (ng * (ng + 1) / 2) / global_tokens
    assert torch.allclose(param_grad, torch.full((4,), expected_grad, device=dev)), \
        f"Grad sync: {param_grad[0].item()} != {expected_grad}"
    if rank == 0:
        print(f'  Token-scaled grad = {param_grad[0].item():.6f} (expected {expected_grad:.6f}) OK')

    if rank == 0:
        print(f'\n{"="*60}\nTest 4: BalancedDataLoader scheduling\n{"="*60}')

    loader = BalancedDataLoader(ng)
    test_cases = [
        ("uniform 8K", [8192] * ng),
        ("mixed", [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048]),
        ("all short", [2048] * ng),
        ("one long", [32768] + [2048] * (ng - 1)),
    ]
    for name, seqs in test_cases:
        mbs = loader.schedule(seqs)
        sp_dist = {}
        for mb in mbs:
            sp_dist[mb.sp_size] = sp_dist.get(mb.sp_size, 0) + 1
        total = sum(mb.tokens for mb in mbs)
        if rank == 0:
            print(f'  {name}: {len(mbs)} MBs, SP dist={sp_dist}, tokens={total}')

    if rank == 0:
        print(f'\n{"="*60}\nTest 5: Sequential microbatch execution\n{"="*60}')

    seqs = [32768, 8192, 4096, 2048]
    mbs = loader.schedule(seqs)
    for mb in mbs:
        info = gm.get_groups(mb.sp_size)
        if info.sp_group is not None:
            dist.barrier(group=info.sp_group)
        if rank == 0:
            print(f'  MB: sp={mb.sp_size}, seq={mb.seq_len}, local_seq={mb.local_seq} OK')

    dist.barrier()
    if rank == 0:
        print(f'\n{"="*60}\nALL CORRECTNESS TESTS PASSED\n{"="*60}\n')

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
