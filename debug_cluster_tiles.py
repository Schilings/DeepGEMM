"""
Debug script to simulate the SM100 BF16 GEMM scheduler behavior.
Verifies what data each CTA in a 2-CTA cluster computes.

Key observations from code:
- grid_dim = kNumSMs (e.g., 128 blocks total)
- cluster_size = 2 (every 2 adjacent blocks form a cluster)
- scheduler.get_next_block() uses: next_block_idx = (++current_iter) * kNumSMs + blockIdx.x
- MMA only runs on leader CTA (block_rank_in_cluster() == 0)
- TMA load adds offset based on block_rank_in_cluster()
- Epilogue runs on BOTH CTAs

This means the two CTAs in a cluster have DIFFERENT blockIdx.x and thus get
DIFFERENT tile assignments from the scheduler!

But wait - MMA only runs on one CTA. How does this work?
The answer: The two CTAs load DIFFERENT parts of the data (via TMA multicast split),
but the UMMA instruction is a 2SM instruction that operates on BOTH CTAs' shared memory.
"""

import math


def ceil_div(a, b):
    return (a + b - 1) // b


def get_num_1d_blocks_per_group(num_sms, block_m, block_n, is_multicast_on_a):
    """Select best group size from candidates {8, 16}"""
    num_best_blocks = 0
    min_usage = float('inf')
    for candidate in [8, 16]:
        if is_multicast_on_a:
            usage = candidate * block_n + ceil_div(num_sms, candidate) * block_m
        else:
            usage = candidate * block_m + ceil_div(num_sms, candidate) * block_n
        if usage < min_usage:
            min_usage = usage
            num_best_blocks = candidate
    return num_best_blocks


def get_swizzled_block_idx(block_idx, num_m_blocks, num_n_blocks, 
                           num_1d_blocks_per_group, is_multicast_on_a, num_multicast):
    """Swizzle block index into (m_block_idx, n_block_idx)"""
    primary_num_blocks = num_n_blocks if is_multicast_on_a else num_m_blocks
    secondary_num_blocks = num_m_blocks if is_multicast_on_a else num_n_blocks
    num_blocks_per_group = secondary_num_blocks * num_1d_blocks_per_group
    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * num_1d_blocks_per_group
    in_group_idx = block_idx % num_blocks_per_group
    num_blocks_in_group = min(num_1d_blocks_per_group, primary_num_blocks - first_block_idx)

    # SM100 does NOT have the unaligned multicast fix (that's SM90 only)
    # Convert to final M/N block indices
    if is_multicast_on_a:
        m_block_idx = in_group_idx // num_blocks_in_group
        n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
    else:
        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
        n_block_idx = in_group_idx // num_blocks_in_group
    
    return m_block_idx, n_block_idx


def simulate_cluster(shape_m, shape_n, shape_k, block_m, block_n, block_k,
                     num_sms, num_multicast=2, is_multicast_on_a=False):
    """
    Simulate what tiles are assigned to the two CTAs in a cluster.
    
    For SM100 with 2-CTA clusters:
    - grid_dim = num_sms (e.g., 128)
    - cluster_size = 2
    - Adjacent blockIdx.x values form a cluster: (0,1), (2,3), (4,5)...
    """
    num_m_blocks = ceil_div(shape_m, block_m)
    num_n_blocks = ceil_div(shape_n, block_n)
    num_blocks = num_m_blocks * num_n_blocks
    num_1d_blocks_per_group = get_num_1d_blocks_per_group(
        num_sms, block_m, block_n, is_multicast_on_a)
    
    load_block_m = block_m // (num_multicast if is_multicast_on_a else 1)
    load_block_n = block_n // (1 if is_multicast_on_a else num_multicast)

    print(f"=" * 80)
    print(f"GEMM Configuration:")
    print(f"  Shape: M={shape_m}, N={shape_n}, K={shape_k}")
    print(f"  Block: M={block_m}, N={block_n}, K={block_k}")
    print(f"  Grid:  num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}, total_blocks={num_blocks}")
    print(f"  Cluster: num_multicast={num_multicast}, is_multicast_on_a={is_multicast_on_a}")
    print(f"  kNumSMs={num_sms}, kNum1DBlocksPerGroup={num_1d_blocks_per_group}")
    print(f"  LOAD_BLOCK_M={load_block_m}, LOAD_BLOCK_N={load_block_n}")
    print(f"=" * 80)
    
    # Simulate first cluster (blockIdx.x = 0 and 1)
    print(f"\n--- Cluster 0 (blockIdx.x = 0, 1) ---")
    print(f"{'Iter':<6}{'CTA0 block_idx':<16}{'CTA0 (m,n) tile':<20}{'CTA1 block_idx':<16}{'CTA1 (m,n) tile':<20}")
    print("-" * 80)
    
    max_iters = min(10, ceil_div(num_blocks, num_sms))  # Show first 10 iterations
    
    cta0_tiles = []
    cta1_tiles = []
    
    for iter_idx in range(max_iters):
        # CTA0: blockIdx.x = 0
        block_idx_0 = iter_idx * num_sms + 0
        # CTA1: blockIdx.x = 1
        block_idx_1 = iter_idx * num_sms + 1
        
        if block_idx_0 >= num_blocks and block_idx_1 >= num_blocks:
            break
            
        m0, n0 = None, None
        m1, n1 = None, None
        
        if block_idx_0 < num_blocks:
            m0, n0 = get_swizzled_block_idx(block_idx_0, num_m_blocks, num_n_blocks,
                                             num_1d_blocks_per_group, is_multicast_on_a, num_multicast)
            cta0_tiles.append((m0, n0))
        
        if block_idx_1 < num_blocks:
            m1, n1 = get_swizzled_block_idx(block_idx_1, num_m_blocks, num_n_blocks,
                                             num_1d_blocks_per_group, is_multicast_on_a, num_multicast)
            cta1_tiles.append((m1, n1))
        
        tile0_str = f"m_blk={m0}, n_blk={n0}" if m0 is not None else "done"
        tile1_str = f"m_blk={m1}, n_blk={n1}" if m1 is not None else "done"
        
        print(f"{iter_idx:<6}{block_idx_0:<16}{tile0_str:<20}{block_idx_1:<16}{tile1_str:<20}")
    
    # Now show what ACTUAL data each CTA loads (with multicast offset)
    print(f"\n--- TMA Load Addresses (with multicast offset) ---")
    print(f"{'Iter':<6}{'CTA0 load region':<40}{'CTA1 load region':<40}")
    print("-" * 86)
    
    for iter_idx in range(min(max_iters, len(cta0_tiles))):
        if iter_idx >= len(cta0_tiles) or iter_idx >= len(cta1_tiles):
            break
        m0, n0 = cta0_tiles[iter_idx]
        m1, n1 = cta1_tiles[iter_idx]
        
        # CTA0 (block_rank = 0)
        m_idx_0 = m0 * block_m + (0 * load_block_m if is_multicast_on_a else 0)
        n_idx_0 = n0 * block_n + (0 if is_multicast_on_a else 0 * load_block_n)
        
        # CTA1 (block_rank = 1)
        m_idx_1 = m1 * block_m + (1 * load_block_m if is_multicast_on_a else 0)
        n_idx_1 = n1 * block_n + (0 if is_multicast_on_a else 1 * load_block_n)
        
        cta0_str = f"A[{m_idx_0}:{m_idx_0+load_block_m},:] x B[:,{n_idx_0}:{n_idx_0+load_block_n}]"
        cta1_str = f"A[{m_idx_1}:{m_idx_1+load_block_m},:] x B[:,{n_idx_1}:{n_idx_1+load_block_n}]"
        
        print(f"{iter_idx:<6}{cta0_str:<40}{cta1_str:<40}")
    
    # Explain the 2SM MMA execution model
    print(f"\n--- MMA Execution Model ---")
    print(f"  - MMA warp ONLY runs on leader CTA (block_rank_in_cluster() == 0)")
    print(f"  - BUT it's a 2SM UMMA instruction (SM100_MMA_F16BF16_2x1SM_SS)")
    print(f"  - The 2SM UMMA reads from BOTH CTAs' shared memory simultaneously")
    print(f"  - UMMA shape: M={128 * num_multicast}, N={block_n}")
    print(f"  - So UMMA computes a {128 * num_multicast}x{block_n} output tile")
    
    # Explain epilogue
    print(f"\n--- Epilogue (Store back to global memory) ---")
    print(f"  - Epilogue runs on BOTH CTAs")
    print(f"  - Each CTA has its OWN scheduler, gets DIFFERENT tiles from get_next_block()")
    print(f"  - But the UMMA result is in tensor memory (TMEM) which is shared across the cluster")
    print(f"")
    
    # The KEY insight:
    print(f"=== KEY INSIGHT ===")
    print(f"  The scheduler gives each CTA a DIFFERENT tile.")
    print(f"  - CTA0 (iter 0): processes tile (m_blk={cta0_tiles[0][0]}, n_blk={cta0_tiles[0][1]})")
    print(f"  - CTA1 (iter 0): processes tile (m_blk={cta1_tiles[0][0]}, n_blk={cta1_tiles[0][1]})")
    print(f"")
    print(f"  For TMA load (warp 0): each CTA loads its own portion of data:")
    if not is_multicast_on_a:
        print(f"    - CTA0 loads A[m0*{block_m} : m0*{block_m}+{load_block_m}] and B[n0*{block_n} : n0*{block_n}+{load_block_n}]")
        print(f"    - CTA1 loads A[m1*{block_m} : m1*{block_m}+{load_block_m}] and B[n1*{block_n}+{load_block_n} : n1*{block_n}+{block_n}]")
        print(f"    - i.e., they split the N dimension: each loads half of B's columns")
    else:
        print(f"    - CTA0 loads A[m0*{block_m} : m0*{block_m}+{load_block_m}] (first half of M)")
        print(f"    - CTA1 loads A[m1*{block_m}+{load_block_m} : m1*{block_m}+{block_m}] (second half of M)")
        print(f"    - i.e., they split the M dimension: each loads half of A's rows")
    print(f"")
    print(f"  For MMA (warp 1, leader only): executes 2SM UMMA which reads from BOTH CTAs' smem")
    print(f"    - Result shape: {128*num_multicast} x {block_n} (covers data from BOTH CTAs)")
    print(f"")
    print(f"  For Epilogue (warp 2+, both CTAs): each CTA stores its portion of the result")
    print(f"    - CTA0: stores its tile's rows of the output")
    print(f"    - CTA1: stores its tile's rows of the output")
    print(f"")
    
    # Wait, let me re-examine. Let's check if both CTAs get SAME or DIFFERENT tiles
    # from their respective schedulers in a single iteration
    print(f"=== VERIFICATION: Do CTAs in same cluster process same or different tiles? ===")
    print(f"")
    
    # Let's trace what happens:
    # 1. TMA warp (warp 0): both CTAs execute get_next_block() independently
    # 2. MMA warp (warp 1): only leader CTA executes get_next_block()
    # 3. Epilogue warp: both CTAs execute get_next_block()
    #
    # Each warp has its OWN scheduler instance! So they're independent.
    # But the MMA warp's barrier sync (full_barriers[stage_idx]->wait(phase))
    # depends on BOTH CTAs' TMA warps arriving.
    #
    # This means: at iteration i, both CTAs' TMA warps must have finished loading
    # before leader CTA's MMA warp can proceed.
    #
    # The TMA warp on CTA0 loads tile (m0, n0), and CTA1's TMA loads tile (m1, n1).
    # These are DIFFERENT tiles!
    #
    # But the UMMA instruction reads from local smem (which has CTA0's data)
    # AND remote smem (which has CTA1's data) via the 2SM mechanism.
    #
    # So what does the UMMA actually compute?
    
    print(f"  Each CTA has 3 independent scheduler instances (one per warp role).")
    print(f"  Each scheduler starts with current_iter = -1 and increments independently.")
    print(f"")
    print(f"  === Critical Question: What does the 2SM UMMA actually compute? ===")
    print(f"")
    print(f"  UMMA_M = 128 * kNumMulticast = {128 * num_multicast}")
    print(f"  This means the UMMA operates on {128*num_multicast} rows in TMEM.")
    print(f"")
    
    if not is_multicast_on_a:
        # kIsMulticastOnA = false: grouping on M, split on N
        print(f"  Since kIsMulticastOnA=false:")
        print(f"    - LOAD_BLOCK_M = BLOCK_M / 1 = {load_block_m} (each CTA loads full M rows of A)")
        print(f"    - LOAD_BLOCK_N = BLOCK_N / 2 = {load_block_n} (each CTA loads half N cols of B)")
        print(f"")
        print(f"  TMA warp behavior (iteration 0):")
        print(f"    CTA0 (rank=0): loads A[{cta0_tiles[0][0]*block_m}:{cta0_tiles[0][0]*block_m+load_block_m}, 0:{block_k}]")
        print(f"                    loads B[0:{block_k}, {cta0_tiles[0][1]*block_n+0*load_block_n}:{cta0_tiles[0][1]*block_n+0*load_block_n+load_block_n}]")
        print(f"    CTA1 (rank=1): loads A[{cta1_tiles[0][0]*block_m}:{cta1_tiles[0][0]*block_m+load_block_m}, 0:{block_k}]")
        print(f"                    loads B[0:{block_k}, {cta1_tiles[0][1]*block_n+1*load_block_n}:{cta1_tiles[0][1]*block_n+1*load_block_n+load_block_n}]")
        print(f"")
        print(f"  WAIT! CTA0 and CTA1 get DIFFERENT m_block_idx from scheduler!")
        print(f"  CTA0 m_block_idx={cta0_tiles[0][0]}, CTA1 m_block_idx={cta1_tiles[0][0]}")
        print(f"  CTA0 n_block_idx={cta0_tiles[0][1]}, CTA1 n_block_idx={cta1_tiles[0][1]}")
        print(f"")
        print(f"  But the 2SM UMMA in the leader CTA reads from:")
        print(f"    - smem_a on CTA0 (leader's local smem)")
        print(f"    - smem_b on CTA0 (leader's local smem)")
        print(f"    PLUS via 2SM mechanism:")
        print(f"    - smem_a on CTA1 (peer's remote smem)")
        print(f"    - smem_b on CTA1 (peer's remote smem)")
        print(f"")
        print(f"  The UMMA descriptor uses LOAD_BLOCK_M rows from smem_a,")
        print(f"  but with UMMA_M=256 it expects 256 rows in TMEM output.")
        print(f"  The 2SM instruction automatically concatenates:")
        print(f"    rows [0:128) from CTA0's smem_a")  
        print(f"    rows [128:256) from CTA1's smem_a")
        print(f"  And for B (LOAD_BLOCK_N={load_block_n}):")
        print(f"    cols [0:{load_block_n}) from CTA0's smem_b")
        print(f"    cols [{load_block_n}:{block_n}) from CTA1's smem_b")
        print(f"")
        print(f"  So the 2SM UMMA computes:")
        print(f"    D[0:256, 0:{block_n}] = concat(A_cta0, A_cta1) @ concat(B_cta0, B_cta1)^T")
        print(f"")
        print(f"  WHERE:")
        print(f"    A_cta0 = A[{cta0_tiles[0][0]*block_m}:{cta0_tiles[0][0]*block_m+load_block_m}]  ({load_block_m} rows)")
        print(f"    A_cta1 = A[{cta1_tiles[0][0]*block_m}:{cta1_tiles[0][0]*block_m+load_block_m}]  ({load_block_m} rows)")
        print(f"    B_cta0 = B[{cta0_tiles[0][1]*block_n}:{cta0_tiles[0][1]*block_n+load_block_n}]  ({load_block_n} cols)")
        print(f"    B_cta1 = B[{cta1_tiles[0][1]*block_n+load_block_n}:{cta1_tiles[0][1]*block_n+block_n}]  ({load_block_n} cols)")
    
    return cta0_tiles, cta1_tiles


def main():
    print("=" * 80)
    print("DEBUG: SM100 2-CTA Cluster Tile Assignment")
    print("=" * 80)
    
    # Typical configuration for BF16 GEMM NT on SM100
    # Based on heuristics: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    # cluster_m=2, cluster_n=1 => kNumMulticast=2, kIsMulticastOnA=false
    
    print("\n\n### Case 1: Small GEMM (M=512, N=512, K=256) ###")
    simulate_cluster(
        shape_m=512, shape_n=512, shape_k=256,
        block_m=128, block_n=128, block_k=64,
        num_sms=128, num_multicast=2, is_multicast_on_a=False
    )
    
    print("\n\n### Case 2: Typical GEMM (M=4096, N=7168, K=2048) ###")
    simulate_cluster(
        shape_m=4096, shape_n=7168, shape_k=2048,
        block_m=128, block_n=128, block_k=64,
        num_sms=128, num_multicast=2, is_multicast_on_a=False
    )
    
    # Also test with multicast on A (cluster_n > 1)
    print("\n\n### Case 3: Multicast on A (M=4096, N=7168, K=2048) ###")
    simulate_cluster(
        shape_m=4096, shape_n=7168, shape_k=2048,
        block_m=128, block_n=128, block_k=64,
        num_sms=128, num_multicast=2, is_multicast_on_a=True
    )


if __name__ == "__main__":
    main()
