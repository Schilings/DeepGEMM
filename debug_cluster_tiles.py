"""
Debug script to verify what data each CTA in a 2-CTA cluster computes in SM100 BF16 GEMM.

=== CONCLUSION from CUTLASS Tutorial 04 ===
In SM100 2SM mode:
- A cluster of 2 CTAs (peer CTAs) collaborates on the SAME output tile.
- The 2SM UMMA instruction (cta_group::2) operates on data from BOTH CTAs' shared memory.
- Only the leader CTA issues the MMA instruction, but it reads from both SMs' smem.
- Each CTA loads a PORTION of the input data (split along M or N).
- The result in TMEM is shared across the 2 SMs.
- Both CTAs participate in storing the result back to global memory.

Key from TiledMMA layout:
  LayoutA_TV: (_2,(_128,_16)):(_128,(_1,_256))  => 2 CTAs, each provides 128 rows of A
  LayoutC_TV: (_2,(_128,_256)):(_128,(_1,_256))  => 2 CTAs, each stores 128 rows of C/D

=== DeepGEMM Implementation Details ===
In DeepGEMM's SM100 BF16 GEMM:
- grid_dim = num_sms (e.g., 128)
- cluster_size = 2
- CUDA runtime groups every 2 adjacent blocks into a cluster: (0,1), (2,3), ...
- Each CTA has its OWN scheduler instance per warp role
- The scheduler assigns tiles based on: next_block_idx = (++iter) * kNumSMs + blockIdx.x
- TWO CTAs in the same cluster get DIFFERENT block_idx from the scheduler!

BUT: The hardware enforces that 2 CTAs in a cluster process together.
So the REAL question is: what does the code actually do with these different block indices?
"""

import math


def ceil_div(a, b):
    return (a + b - 1) // b


def get_num_1d_blocks_per_group(num_sms, block_m, block_n, is_multicast_on_a):
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
                           num_1d_blocks_per_group, is_multicast_on_a):
    primary_num_blocks = num_n_blocks if is_multicast_on_a else num_m_blocks
    secondary_num_blocks = num_m_blocks if is_multicast_on_a else num_n_blocks
    num_blocks_per_group = secondary_num_blocks * num_1d_blocks_per_group
    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * num_1d_blocks_per_group
    in_group_idx = block_idx % num_blocks_per_group
    num_blocks_in_group = min(num_1d_blocks_per_group, primary_num_blocks - first_block_idx)

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
    Simulate the full data flow for a 2-CTA cluster in DeepGEMM SM100 BF16 GEMM.
    """
    num_m_blocks = ceil_div(shape_m, block_m)
    num_n_blocks = ceil_div(shape_n, block_n)
    num_blocks = num_m_blocks * num_n_blocks
    num_1d_blocks_per_group = get_num_1d_blocks_per_group(
        num_sms, block_m, block_n, is_multicast_on_a)
    
    load_block_m = block_m // (num_multicast if is_multicast_on_a else 1)
    load_block_n = block_n // (1 if is_multicast_on_a else num_multicast)
    
    umma_m = 128 * num_multicast  # LAYOUT_AD_M * kNumMulticast
    umma_n = block_n if not False else block_m  # kSwapAB=false

    print(f"=" * 90)
    print(f"GEMM Configuration:")
    print(f"  Shape: M={shape_m}, N={shape_n}, K={shape_k}")
    print(f"  Block: M={block_m}, N={block_n}, K={block_k}")
    print(f"  Grid:  num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}, total_blocks={num_blocks}")
    print(f"  Cluster: num_multicast={num_multicast}, is_multicast_on_a={is_multicast_on_a}")
    print(f"  kNumSMs={num_sms}, kNum1DBlocksPerGroup={num_1d_blocks_per_group}")
    print(f"  LOAD_BLOCK_M={load_block_m}, LOAD_BLOCK_N={load_block_n}")
    print(f"  UMMA shape: M={umma_m}, N={umma_n}, K=16")
    print(f"=" * 90)
    
    # ================================================================
    # TRACE: First few iterations of the first cluster (blockIdx.x = 0 and 1)
    # ================================================================
    print(f"\n{'='*90}")
    print(f"TRACE: Cluster 0 (CTA0: blockIdx.x=0, CTA1: blockIdx.x=1)")
    print(f"{'='*90}")
    
    max_iters = min(5, ceil_div(num_blocks, num_sms))
    
    for iter_idx in range(max_iters):
        block_idx_0 = iter_idx * num_sms + 0
        block_idx_1 = iter_idx * num_sms + 1
        
        if block_idx_0 >= num_blocks and block_idx_1 >= num_blocks:
            break
        
        print(f"\n--- Iteration {iter_idx} ---")
        
        # CTA0's scheduler
        if block_idx_0 < num_blocks:
            m0, n0 = get_swizzled_block_idx(block_idx_0, num_m_blocks, num_n_blocks,
                                            num_1d_blocks_per_group, is_multicast_on_a)
        else:
            m0, n0 = None, None
            
        # CTA1's scheduler
        if block_idx_1 < num_blocks:
            m1, n1 = get_swizzled_block_idx(block_idx_1, num_m_blocks, num_n_blocks,
                                            num_1d_blocks_per_group, is_multicast_on_a)
        else:
            m1, n1 = None, None
        
        print(f"  Scheduler results:")
        print(f"    CTA0: block_idx={block_idx_0} => m_block_idx={m0}, n_block_idx={n0}")
        print(f"    CTA1: block_idx={block_idx_1} => m_block_idx={m1}, n_block_idx={n1}")
        
        if m0 is None or m1 is None:
            continue
        
        # ==== TMA WARP (warp 0) - Both CTAs execute ====
        # Code: lines 220-223
        #   m_idx += kIsMulticastOnA ? (block_rank * load_block_m) : 0
        #   n_idx += kIsMulticastOnA ? 0 : (block_rank * LOAD_BLOCK_N)
        
        # CTA0 (block_rank=0):
        m_idx_0 = m0 * block_m + (0 * load_block_m if is_multicast_on_a else 0)
        n_idx_0 = n0 * block_n + (0 if is_multicast_on_a else 0 * load_block_n)
        # CTA1 (block_rank=1):
        m_idx_1 = m1 * block_m + (1 * load_block_m if is_multicast_on_a else 0)
        n_idx_1 = n1 * block_n + (0 if is_multicast_on_a else 1 * load_block_n)
        
        print(f"\n  TMA Load (warp 0):")
        print(f"    CTA0 (rank=0): A rows [{m_idx_0}:{m_idx_0+load_block_m}], B cols [{n_idx_0}:{n_idx_0+load_block_n}]")
        print(f"    CTA1 (rank=1): A rows [{m_idx_1}:{m_idx_1+load_block_m}], B cols [{n_idx_1}:{n_idx_1+load_block_n}]")
        
        # ==== MMA WARP (warp 1) - ONLY leader CTA executes ====
        # It also calls get_next_block() independently.
        # Since only leader CTA runs this, it gets block_idx = iter * kNumSMs + 0 = same as CTA0's TMA warp
        print(f"\n  MMA (warp 1, leader CTA only):")
        print(f"    Leader's scheduler: block_idx={block_idx_0} => m_block_idx={m0}, n_block_idx={n0}")
        print(f"    Issues tcgen05.mma.cta_group::2 (2SM UMMA)")
        print(f"    Reads from BOTH CTAs' smem:")
        print(f"      - CTA0's smem_a: {load_block_m} rows of A starting at row {m_idx_0}")
        print(f"      - CTA1's smem_a: {load_block_m} rows of A starting at row {m_idx_1}")
        print(f"      - CTA0's smem_b: {load_block_n} cols of B starting at col {n_idx_0}")
        print(f"      - CTA1's smem_b: {load_block_n} cols of B starting at col {n_idx_1}")
        print(f"    Output in TMEM: {umma_m} rows x {umma_n} cols (shared across 2 SMs)")
        
        # ==== EPILOGUE WARP - Both CTAs execute ====
        # Each CTA calls get_next_block() independently again.
        # CTA0's epilogue warp gets the same tile as its TMA warp (same scheduler pattern).
        # CTA1's epilogue warp gets the same tile as ITS TMA warp.
        
        # base_m_idx for epilogue:
        # scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx) = m_block_idx * BLOCK_M (for Normal GEMM)
        epi_base_m_0 = m0 * block_m  # CTA0's epilogue
        epi_base_n_0 = n0 * block_n
        epi_base_m_1 = m1 * block_m  # CTA1's epilogue
        epi_base_n_1 = n1 * block_n
        
        print(f"\n  Epilogue (both CTAs):")
        print(f"    CTA0: stores D[{epi_base_m_0}:{epi_base_m_0+block_m}, {epi_base_n_0}:{epi_base_n_0+block_n}]")
        print(f"    CTA1: stores D[{epi_base_m_1}:{epi_base_m_1+block_m}, {epi_base_n_1}:{epi_base_n_1+block_n}]")
        
        # ==== What does the UMMA output ACTUALLY contain? ====
        print(f"\n  === WHAT THE 2SM UMMA ACTUALLY COMPUTES ===")
        if not is_multicast_on_a:
            # kIsMulticastOnA=false: each CTA loads full BLOCK_M=128 rows of A, half of B
            # CTA0 loads A[m0*128 : m0*128+128] and B[:,n0*128 : n0*128+64]
            # CTA1 loads A[m1*128 : m1*128+128] and B[:,n1*128+64 : n1*128+128]
            #
            # The 2SM UMMA with UMMA_M=256 operates on 256 rows of A from TMEM perspective.
            # The hardware concatenates:
            #   First 128 rows from CTA0's smem_a (logical A rows [m0*128 : m0*128+128])
            #   Next 128 rows from CTA1's smem_a (logical A rows [m1*128 : m1*128+128])
            # And BLOCK_N=128 cols of B:
            #   First 64 cols from CTA0's smem_b (logical B cols [n0*128 : n0*128+64])
            #   Next 64 cols from CTA1's smem_b (logical B cols [n1*128+64 : n1*128+128])
            
            if m0 != m1 or n0 != n1:
                print(f"    WARNING: CTA0 and CTA1 have DIFFERENT tile assignments!")
                print(f"    CTA0: tile (m={m0}, n={n0}), CTA1: tile (m={m1}, n={n1})")
                print(f"")
                print(f"    The 2SM UMMA hardware concatenates data from both SMs:")
                print(f"      A_combined[0:128]   = A[{m_idx_0}:{m_idx_0+load_block_m}]  (from CTA0's smem)")
                print(f"      A_combined[128:256] = A[{m_idx_1}:{m_idx_1+load_block_m}]  (from CTA1's smem)")
                print(f"      B_combined[0:64]    = B[{n_idx_0}:{n_idx_0+load_block_n}]  (from CTA0's smem)")
                print(f"      B_combined[64:128]  = B[{n_idx_1}:{n_idx_1+load_block_n}]  (from CTA1's smem)")
                print(f"")
                print(f"    TMEM output (256 x 128 in FP32):")
                print(f"      D[row, col] = sum_k( A_combined[row, k] * B_combined[col, k] )  for NT layout")
                print(f"")
                
                # Is this mathematically correct?
                # D[0:128, 0:64]   = A[m0*128:(m0+1)*128] @ B[n0*128:n0*128+64]^T   ✓ (CTA0's portion)
                # D[0:128, 64:128] = A[m0*128:(m0+1)*128] @ B[n1*128+64:n1*128+128]^T
                # D[128:256, 0:64] = A[m1*128:(m1+1)*128] @ B[n0*128:n0*128+64]^T
                # D[128:256, 64:128] = A[m1*128:(m1+1)*128] @ B[n1*128+64:n1*128+128]^T  ✓ (CTA1's portion)
                
                # For this to be correct, we need n0 == n1 (same N block)!
                # AND m0+1 == m1 (adjacent M blocks)!
                
                if n0 == n1 and m1 == m0 + 1:
                    print(f"    ✓ CORRECT: CTA0 and CTA1 process ADJACENT M blocks with SAME N block")
                    print(f"    The 2SM UMMA computes a combined {umma_m}x{block_n} tile:")
                    print(f"      TMEM[0:128, 0:128] = A[{m0*block_m}:{(m0+1)*block_m}] @ B[{n0*block_n}:{(n0+1)*block_n}]^T")
                    print(f"      TMEM[128:256, 0:128] = A[{m1*block_m}:{(m1+1)*block_m}] @ B[{n1*block_n}:{(n1+1)*block_n}]^T")
                    print(f"")
                    print(f"    After epilogue:")
                    print(f"      CTA0 stores TMEM[0:128, :] → D[{epi_base_m_0}:{epi_base_m_0+block_m}, {epi_base_n_0}:{epi_base_n_0+block_n}]")
                    print(f"      CTA1 stores TMEM[128:256, :] → D[{epi_base_m_1}:{epi_base_m_1+block_m}, {epi_base_n_1}:{epi_base_n_1+block_n}]")
                    print(f"")
                    print(f"    ★ EACH CTA processes its own independent 128x128 output tile!")
                    print(f"    ★ The 2SM UMMA simply batches 2 adjacent tiles into one instruction!")
                    print(f"    ★ B data is shared (multicast) - both tiles use the SAME B columns!")
                else:
                    print(f"    ✗ Tiles are not adjacent or don't share N - unexpected pattern!")
                    print(f"    This would mean the UMMA computes cross products between different tiles.")
            else:
                print(f"    CTA0 and CTA1 have the SAME tile assignment - unexpected!")
        else:
            # kIsMulticastOnA=true: each CTA loads half of A (split M), full B
            if n0 != n1:
                print(f"    CTA0 n_block={n0}, CTA1 n_block={n1}")
                if m0 == m1:
                    print(f"    ✓ Same M block, different N blocks")
                    print(f"    A is shared (multicast) - both tiles use the SAME A rows!")
            
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*90}")
    print(f"SUMMARY: How a 2-CTA cluster works in DeepGEMM SM100 BF16 GEMM")
    print(f"{'='*90}")
    print(f"""
For kIsMulticastOnA=False (the common case, cluster_m=2, cluster_n=1):

1. SCHEDULER: Each CTA gets a DIFFERENT tile from the scheduler.
   - The swizzle pattern ensures adjacent block_idx values map to
     adjacent M blocks with the SAME N block.
   - CTA0 (block_idx=2k):   m_block = X,   n_block = Y  
   - CTA1 (block_idx=2k+1): m_block = X+1, n_block = Y

2. TMA LOAD (both CTAs):
   - CTA0 loads: A[X*{block_m} : (X+1)*{block_m}, :] and B[:, Y*{block_n} : Y*{block_n}+{load_block_n}]
   - CTA1 loads: A[(X+1)*{block_m} : (X+2)*{block_m}, :] and B[:, Y*{block_n}+{load_block_n} : (Y+1)*{block_n}]
   - Each CTA loads full {load_block_m} rows of A (for its own tile)
   - B is SPLIT: CTA0 loads first {load_block_n} cols, CTA1 loads last {load_block_n} cols
   - Via TMA multicast, BOTH CTAs get the FULL B (each CTA's B half is multicast to the other)

3. 2SM UMMA (leader CTA only):
   - The tcgen05.mma.cta_group::2 instruction operates on BOTH SMs' shared memory
   - It computes a {umma_m}x{umma_n} output:
     * Rows [0:128) use CTA0's A data: A[X*{block_m} : (X+1)*{block_m}]
     * Rows [128:256) use CTA1's A data: A[(X+1)*{block_m} : (X+2)*{block_m}]
     * All 128 columns of B (combined from both CTAs' halves)
   - Result is stored in TMEM (shared across both SMs)
   - Effectively computes TWO independent {block_m}x{block_n} tiles in one shot!

4. EPILOGUE (both CTAs):
   - CTA0 reads TMEM rows [0:128) and stores to D[X*{block_m}:(X+1)*{block_m}, Y*{block_n}:(Y+1)*{block_n}]
   - CTA1 reads TMEM rows [128:256) and stores to D[(X+1)*{block_m}:(X+2)*{block_m}, Y*{block_n}:(Y+1)*{block_n}]

KEY INSIGHT:
  The 2-CTA cluster processes TWO adjacent M-tiles that share the same N-tile.
  The B matrix is loaded once (split across 2 CTAs, then multicast) and reused for both tiles.
  This HALVES the B memory bandwidth requirement compared to processing tiles independently!
  
  From each individual CTA's perspective:
    It still computes ONE {block_m}x{block_n} output tile.
    The benefit is bandwidth saving from B-matrix sharing via multicast.
""")


def main():
    print("=" * 90)
    print("DEBUG: SM100 2-CTA Cluster Data Flow in DeepGEMM BF16 GEMM")
    print("=" * 90)
    
    # Case 1: Small example
    print("\n\n### Case 1: Small GEMM (M=512, N=512, K=256) ###")
    simulate_cluster(
        shape_m=512, shape_n=512, shape_k=256,
        block_m=128, block_n=128, block_k=64,
        num_sms=128, num_multicast=2, is_multicast_on_a=False
    )
    
    # Case 2: Typical size
    print("\n\n### Case 2: Typical GEMM (M=4096, N=7168, K=2048) ###")
    simulate_cluster(
        shape_m=4096, shape_n=7168, shape_k=2048,
        block_m=128, block_n=128, block_k=64,
        num_sms=128, num_multicast=2, is_multicast_on_a=False
    )


if __name__ == "__main__":
    main()
